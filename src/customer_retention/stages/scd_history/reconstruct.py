"""SCD history reconstruction.

For each ``(parent_record, grid_date)`` pair this module reconstructs the value
of every tracked field as of that grid date using the SCD history change log.
The output is a wide ``(parent_record_key, as_of_date, *tracked_fields)`` view
that the framework's existing temporal merger handles via the equi-join branch.

Backend dispatch:
- pyspark.pandas / native Spark inputs are routed through
  ``_reconstruct_distributed`` which uses native Spark ``unionByName`` + a
  **single** ``Window`` pass with one conditional ``F.last(when(field=F, value),
  ignoreNulls=True)`` aggregate per tracked field. All K fields are computed
  in one stage, then a single broadcast-eligible parent join applies fallbacks
  in bulk — no per-field Window passes, no chained outer joins.
- Native pandas inputs are routed through ``_reconstruct_pandas`` which uses
  ``merge_asof`` per field — the natural pandas equivalent of the Spark
  Window pattern.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from customer_retention.core.compat import (
    _is_native_spark_df,
    _is_spark_pandas,
    as_pandas_api,
    as_spark_df,
    native_pd,
    safe_drop_duplicates,
    safe_to_datetime,
)

from .config import SCDHistoryReconstructionConfig

logger = logging.getLogger(__name__)

_AS_OF_COLUMN = "as_of_date"
_ANCHOR_SENTINEL = "__anchor__"


def reconstruct_scd_history_at_grid(
    history_df: Any,
    grid_dates: Sequence[Any],
    config: SCDHistoryReconstructionConfig,
    parent_df: Optional[Any] = None,
) -> Any:
    """Reconstruct per-(parent_record, grid_date) tracked-field state.

    Output schema: ``(parent_record_key, as_of_date, *tracked_fields)``.
    """
    grid_list = _normalize_grid_dates(grid_dates)
    effective_fields = _resolve_effective_fields(history_df, config)

    if _is_spark_pandas(history_df):
        spark_history = as_spark_df(history_df)
        spark_parent = as_spark_df(parent_df) if parent_df is not None else None
        spark_result = _reconstruct_distributed(
            spark_history, grid_list, config, effective_fields, spark_parent
        )
        return as_pandas_api(spark_result)
    if _is_native_spark_df(history_df):
        return _reconstruct_distributed(
            history_df, grid_list, config, effective_fields, parent_df
        )
    return _reconstruct_pandas(
        history_df, grid_list, config, effective_fields, parent_df
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalize_grid_dates(grid_dates: Sequence[Any]) -> List[native_pd.Timestamp]:
    if grid_dates is None:
        raise ValueError("grid_dates must not be None")
    normalized = [native_pd.Timestamp(d) for d in grid_dates]
    if not normalized:
        raise ValueError("grid_dates must not be empty")
    return sorted(normalized)


def _resolve_effective_fields(
    history_df: Any, config: SCDHistoryReconstructionConfig
) -> List[str]:
    if _is_spark_pandas(history_df) or _is_native_spark_df(history_df):
        present_fields = _spark_distinct_field_values(
            as_spark_df(history_df) if _is_spark_pandas(history_df) else history_df,
            config.field_column,
        )
    else:
        present_fields = set(history_df[config.field_column].dropna().unique())

    missing = [f for f in config.tracked_fields if f not in present_fields]
    if not missing:
        return list(config.tracked_fields)

    if config.on_unknown_field == "raise":
        raise ValueError(
            f"tracked_fields not present in history data: {missing}. "
            f"Set on_unknown_field='skip' or 'warn' to triage."
        )
    logger.warning(
        "Dropping tracked_fields not present in history: %s", missing
    )
    return [f for f in config.tracked_fields if f not in missing]


def _spark_distinct_field_values(spark_df: Any, field_column: str) -> set:
    rows = spark_df.select(field_column).distinct().collect()
    return {row[field_column] for row in rows if row[field_column] is not None}


def _parent_value_lookup(
    config: SCDHistoryReconstructionConfig,
) -> dict[str, str]:
    return {f: c for f, c in config.parent_value_columns}


# ---------------------------------------------------------------------------
# Native pandas path
# ---------------------------------------------------------------------------


def _reconstruct_pandas(
    history_df: Any,
    grid_dates: List[native_pd.Timestamp],
    config: SCDHistoryReconstructionConfig,
    effective_fields: List[str],
    parent_df: Optional[Any],
) -> native_pd.DataFrame:
    history = history_df.copy()
    history[config.change_timestamp_column] = safe_to_datetime(
        history[config.change_timestamp_column]
    )
    history = history[history[config.field_column].isin(effective_fields)]

    parent_record_ids = _all_parent_record_ids_pandas(history, parent_df, config)
    spine = _build_spine_pandas(parent_record_ids, grid_dates, config)
    spine = _apply_parent_creation_filter_pandas(spine, parent_df, config)

    parent_lookup = _parent_value_lookup(config)
    field_results: list[native_pd.DataFrame] = []
    for field_name in effective_fields:
        per_field = _reconstruct_single_field_pandas(
            history, spine, config, field_name
        )
        if parent_df is not None and field_name in parent_lookup:
            per_field = _apply_parent_fallback_pandas(
                per_field,
                parent_df,
                config,
                field_name,
                parent_lookup[field_name],
            )
        field_results.append(per_field)

    result = field_results[0]
    for extra in field_results[1:]:
        result = result.merge(
            extra,
            on=[config.parent_record_key, _AS_OF_COLUMN],
            how="outer",
        )
    return result.reset_index(drop=True)


def _all_parent_record_ids_pandas(
    history: native_pd.DataFrame,
    parent_df: Optional[native_pd.DataFrame],
    config: SCDHistoryReconstructionConfig,
) -> List[Any]:
    ids: set[Any] = set(history[config.parent_record_key].dropna().unique())
    if parent_df is not None:
        ids |= set(parent_df[config.parent_record_key].dropna().unique())
    return sorted(ids, key=lambda v: (v is None, str(v)))


def _build_spine_pandas(
    parent_record_ids: List[Any],
    grid_dates: List[native_pd.Timestamp],
    config: SCDHistoryReconstructionConfig,
) -> native_pd.DataFrame:
    rows = [
        {config.parent_record_key: rid, _AS_OF_COLUMN: anchor}
        for rid in parent_record_ids
        for anchor in grid_dates
    ]
    return native_pd.DataFrame(rows)


def _apply_parent_creation_filter_pandas(
    spine: native_pd.DataFrame,
    parent_df: Optional[native_pd.DataFrame],
    config: SCDHistoryReconstructionConfig,
) -> native_pd.DataFrame:
    if parent_df is None or not config.parent_creation_timestamp_column:
        return spine
    creation_col = _resolve_parent_column(
        list(parent_df.columns), config.parent_creation_timestamp_column
    )
    if creation_col is None:
        return spine
    creation_lookup = safe_drop_duplicates(
        parent_df[[config.parent_record_key, creation_col]],
        subset=[config.parent_record_key],
        keep="last",
    ).rename(columns={creation_col: "__parent_created_at__"})
    creation_lookup["__parent_created_at__"] = safe_to_datetime(
        creation_lookup["__parent_created_at__"]
    )
    merged = spine.merge(creation_lookup, on=config.parent_record_key, how="left")
    keep = merged["__parent_created_at__"].isna() | (
        merged[_AS_OF_COLUMN] >= merged["__parent_created_at__"]
    )
    return merged.loc[keep, [config.parent_record_key, _AS_OF_COLUMN]].reset_index(drop=True)


def _reconstruct_single_field_pandas(
    history: native_pd.DataFrame,
    spine: native_pd.DataFrame,
    config: SCDHistoryReconstructionConfig,
    field_name: str,
) -> native_pd.DataFrame:
    field_history = history[history[config.field_column] == field_name][
        [
            config.parent_record_key,
            config.change_timestamp_column,
            config.new_value_column,
        ]
        + (
            [config.unique_row_id_column]
            if config.unique_row_id_column
            and config.unique_row_id_column in history.columns
            else []
        )
    ].copy()

    # merge_asof requires both sides to be sorted globally by the time key.
    # When unique_row_id_column ties exist at the same timestamp, the larger
    # row id wins — sort ascending so the LAST row of a tied group is the
    # backward-asof match.
    history_sort = [config.change_timestamp_column]
    if (
        config.unique_row_id_column
        and config.unique_row_id_column in field_history.columns
    ):
        history_sort.append(config.unique_row_id_column)
    field_history = field_history.sort_values(history_sort).reset_index(drop=True)

    field_spine = spine.sort_values(_AS_OF_COLUMN).reset_index(drop=True)
    merged = native_pd.merge_asof(
        field_spine,
        field_history,
        left_on=_AS_OF_COLUMN,
        right_on=config.change_timestamp_column,
        by=config.parent_record_key,
        direction="backward",
    )
    merged = merged.rename(columns={config.new_value_column: field_name})
    return merged[[config.parent_record_key, _AS_OF_COLUMN, field_name]]


def _apply_parent_fallback_pandas(
    per_field: native_pd.DataFrame,
    parent_df: native_pd.DataFrame,
    config: SCDHistoryReconstructionConfig,
    field_name: str,
    parent_column: str,
) -> native_pd.DataFrame:
    resolved = _resolve_parent_column_or_raise(
        parent_df, parent_column, field_name, config
    )
    fallback_alias = f"__parent__{field_name}__"
    parent_subset = safe_drop_duplicates(
        parent_df[[config.parent_record_key, resolved]],
        subset=[config.parent_record_key],
        keep="last",
    ).rename(columns={resolved: fallback_alias})
    merged = per_field.merge(
        parent_subset, on=config.parent_record_key, how="left"
    )
    merged[field_name] = merged[field_name].where(
        merged[field_name].notna(), merged[fallback_alias]
    )
    return merged.drop(columns=[fallback_alias])


def _resolve_parent_column(parent_columns: Sequence[str], requested: str) -> Optional[str]:
    """Return the parent column whose name matches ``requested`` case-insensitively.

    Snowflake-sourced parent tables routinely mix unquoted UPPERCASE columns
    with quoted-identifier camelCase columns. Users should not have to know
    the warehouse's quoting history when configuring ``parent_value_columns``.
    """
    if requested in parent_columns:
        return requested
    lowered = requested.lower()
    for actual in parent_columns:
        if str(actual).lower() == lowered:
            return actual
    return None


def _resolve_parent_column_or_raise(
    parent_df: Any,
    parent_column: str,
    field_name: str,
    config: SCDHistoryReconstructionConfig,
) -> str:
    resolved = _resolve_parent_column(list(parent_df.columns), parent_column)
    if resolved is None:
        raise ValueError(
            f"parent_value_columns references column {parent_column!r} for "
            f"tracked field {field_name!r} but it is not present in the parent "
            f"dataset {config.parent_table_dataset_name!r} (resolution is "
            f"case-insensitive)"
        )
    return resolved


# ---------------------------------------------------------------------------
# Distributed (native Spark) path
# ---------------------------------------------------------------------------


def _reconstruct_distributed(
    spark_history: Any,
    grid_dates: List[native_pd.Timestamp],
    config: SCDHistoryReconstructionConfig,
    effective_fields: List[str],
    spark_parent: Optional[Any],
) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    history = spark_history.withColumn(
        config.change_timestamp_column,
        F.expr(f"try_to_timestamp(`{config.change_timestamp_column}`)"),
    ).filter(F.col(config.field_column).isin(list(effective_fields)))

    spine = _build_spine_distributed(history, grid_dates, config, spark_parent)
    has_row_id = (
        bool(config.unique_row_id_column)
        and config.unique_row_id_column in spark_history.columns
    )

    unioned = _build_history_anchor_union(history, spine, config, has_row_id)
    valued = _attach_field_values_in_one_window_pass(
        unioned, config, effective_fields, has_row_id
    )

    result = (
        valued.filter(F.col("__is_anchor__"))
        .select(
            F.col("__rec__").alias(config.parent_record_key),
            F.col("__ts__").alias(_AS_OF_COLUMN),
            *[F.col(name) for name in effective_fields],
        )
    )

    parent_lookup = _parent_value_lookup(config)
    applicable_fallbacks = [
        (f, parent_lookup[f]) for f in effective_fields if f in parent_lookup
    ]
    if spark_parent is not None and applicable_fallbacks:
        result = _apply_parent_fallbacks_distributed(
            result, spark_parent, config, applicable_fallbacks
        )
    return result


def _build_spine_distributed(
    spark_history: Any,
    grid_dates: List[native_pd.Timestamp],
    config: SCDHistoryReconstructionConfig,
    spark_parent: Optional[Any],
) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    spark = spark_history.sparkSession
    grid_rows = [(native_pd.Timestamp(d).to_pydatetime(),) for d in grid_dates]
    grid_df = spark.createDataFrame(grid_rows, schema=[_AS_OF_COLUMN]).withColumn(
        _AS_OF_COLUMN, F.col(_AS_OF_COLUMN).cast("timestamp")
    )

    record_ids_with_creation = _record_ids_with_creation_distributed(
        spark_history, spark_parent, config
    )
    crossed = record_ids_with_creation.crossJoin(F.broadcast(grid_df))
    if "__parent_created_at__" not in crossed.columns:
        return crossed
    return crossed.filter(
        F.col("__parent_created_at__").isNull()
        | (F.col(_AS_OF_COLUMN) >= F.col("__parent_created_at__"))
    ).drop("__parent_created_at__")


def _record_ids_with_creation_distributed(
    spark_history: Any,
    spark_parent: Optional[Any],
    config: SCDHistoryReconstructionConfig,
) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    history_ids = spark_history.select(config.parent_record_key).distinct()
    if spark_parent is None or config.parent_record_key not in spark_parent.columns:
        return history_ids

    parent_ids = spark_parent.select(config.parent_record_key).distinct()
    record_ids = history_ids.unionByName(parent_ids).distinct()
    creation_col = (
        _resolve_parent_column(list(spark_parent.columns), config.parent_creation_timestamp_column)
        if config.parent_creation_timestamp_column
        else None
    )
    if creation_col is None:
        return record_ids

    creation_lookup = spark_parent.select(
        F.col(config.parent_record_key),
        F.expr(f"try_to_timestamp(`{creation_col}`)").alias("__parent_created_at__"),
    ).dropDuplicates([config.parent_record_key])
    return record_ids.join(
        creation_lookup, on=config.parent_record_key, how="left"
    )


def _build_history_anchor_union(
    history: Any,
    spine: Any,
    config: SCDHistoryReconstructionConfig,
    has_row_id: bool,
) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    history_select = [
        F.col(config.parent_record_key).alias("__rec__"),
        F.col(config.change_timestamp_column).alias("__ts__"),
        F.col(config.field_column).alias("__field__"),
        F.col(config.new_value_column).cast("string").alias("__value__"),
        F.lit(False).alias("__is_anchor__"),
    ]
    spine_select = [
        F.col(config.parent_record_key).alias("__rec__"),
        F.col(_AS_OF_COLUMN).alias("__ts__"),
        F.lit(None).cast("string").alias("__field__"),
        F.lit(None).cast("string").alias("__value__"),
        F.lit(True).alias("__is_anchor__"),
    ]
    if has_row_id:
        history_select.append(
            F.col(config.unique_row_id_column).cast("string").alias("__row_id__")
        )
        spine_select.append(F.lit(_ANCHOR_SENTINEL).alias("__row_id__"))
    return history.select(*history_select).unionByName(spine.select(*spine_select))


def _attach_field_values_in_one_window_pass(
    unioned: Any,
    config: SCDHistoryReconstructionConfig,
    effective_fields: List[str],
    has_row_id: bool,
) -> Any:
    from pyspark.sql import Window
    from pyspark.sql import functions as F  # noqa: N812

    order_keys = [F.col("__ts__").asc(), F.col("__is_anchor__").asc()]
    if has_row_id:
        order_keys.append(F.col("__row_id__").asc())
    window = (
        Window.partitionBy("__rec__")
        .orderBy(*order_keys)
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    field_value_exprs = [
        F.last(
            F.when(F.col("__field__") == name, F.col("__value__")),
            ignorenulls=True,
        ).over(window).alias(name)
        for name in effective_fields
    ]
    return unioned.select(
        F.col("__rec__"),
        F.col("__ts__"),
        F.col("__is_anchor__"),
        *field_value_exprs,
    )


def _apply_parent_fallbacks_distributed(
    per_field: Any,
    spark_parent: Any,
    config: SCDHistoryReconstructionConfig,
    applicable: List[tuple],
) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    parent_columns = list(spark_parent.columns)
    resolved_pairs: List[tuple] = []
    missing: List[str] = []
    for field_name, parent_column in applicable:
        resolved = _resolve_parent_column(parent_columns, parent_column)
        if resolved is None:
            missing.append(parent_column)
        else:
            resolved_pairs.append((field_name, resolved))
    if missing:
        raise ValueError(
            f"parent_value_columns reference column(s) {missing} that are not "
            f"present in the parent dataset {config.parent_table_dataset_name!r} "
            f"(resolution is case-insensitive)"
        )

    select_cols = [F.col(config.parent_record_key)]
    aliases: dict[str, str] = {}
    for field_name, resolved in resolved_pairs:
        alias = f"__parent__{field_name}__"
        aliases[field_name] = alias
        select_cols.append(F.col(resolved).cast("string").alias(alias))

    parent_subset = spark_parent.select(*select_cols).dropDuplicates(
        [config.parent_record_key]
    )
    joined = per_field.join(
        parent_subset, on=config.parent_record_key, how="left"
    )
    for field_name, alias in aliases.items():
        joined = joined.withColumn(
            field_name, F.coalesce(F.col(field_name), F.col(alias))
        )
    return joined.drop(*aliases.values())
