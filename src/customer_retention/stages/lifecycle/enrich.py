"""Lifecycle enrichment: convert an entity-with-lifespan table into a doubled
event stream.

For every row in the source table this function emits:

1. A START event, with ``event_timestamp = valid_from_column``.
2. (Optionally) a TERMINATE event, with ``event_timestamp = effective_valid_to``,
   where ``effective_valid_to`` is a status-gated COALESCE over
   ``valid_to_columns``.

The doubled stream is roughly 1.6x the source size for typical SaaS contract
data and lets the framework's existing event-level pipeline (bronze_event +
temporal merger) capture the lifecycle gradient that a snapshot view loses.

Backend dispatch:
- pyspark.pandas / native Spark inputs are routed through ``_enrich_distributed``
  which uses native Spark ``withColumn`` / ``filter`` / ``unionByName``. This
  avoids the ``_psseries`` corruption antipattern that arises when chaining
  in-place column mutations on pyspark.pandas DataFrames (see
  docs/Coding_Practices.md).
- Native pandas inputs are routed through ``_enrich_pandas`` which uses
  in-place column mutations safely (native pandas has no ``_psseries`` cache).
"""
from __future__ import annotations

import logging
from typing import Any, List

from customer_retention.core.compat import (
    NaT,
    _is_native_spark_df,
    _is_spark_pandas,
    as_pandas_api,
    as_spark_df,
    concat,
    safe_to_datetime,
)

from .config import LifecycleEnrichmentConfig

logger = logging.getLogger(__name__)


def enrich_lifecycle_dataset(
    raw_df: Any, config: LifecycleEnrichmentConfig
) -> Any:
    """Convert a lifespan table into a doubled START/TERMINATE event stream."""
    if _is_spark_pandas(raw_df):
        spark_df = as_spark_df(raw_df)
        enriched_spark = _enrich_distributed(spark_df, config)
        return as_pandas_api(enriched_spark)
    if _is_native_spark_df(raw_df):
        return _enrich_distributed(raw_df, config)
    return _enrich_pandas(raw_df, config)


# ---------------------------------------------------------------------------
# Native pandas path
# ---------------------------------------------------------------------------


def _enrich_pandas(raw_df: Any, config: LifecycleEnrichmentConfig) -> Any:
    df = raw_df.copy()
    df[config.valid_from_column] = safe_to_datetime(df[config.valid_from_column])
    for col in config.valid_to_columns:
        if col in df.columns:
            df[col] = safe_to_datetime(df[col])

    effective = _coalesce_pandas(df, list(config.valid_to_columns))
    if config.status_column and config.terminal_status_values:
        is_terminal = df[config.status_column].isin(list(config.terminal_status_values))
        effective = effective.where(is_terminal, NaT)
    df["__effective_valid_to__"] = effective

    df = _apply_corrupt_row_policy_pandas(df, config)

    drop_now = [c for c in config.drop_columns if c in df.columns]
    if drop_now:
        df = df.drop(columns=drop_now)

    starts = df.copy()
    starts[config.event_type_column] = "start"
    starts[config.event_timestamp_column] = starts[config.valid_from_column]

    has_terminate = df["__effective_valid_to__"].notna()
    terminates = df[has_terminate].copy()
    terminates[config.event_type_column] = "terminate"
    terminates[config.event_timestamp_column] = terminates["__effective_valid_to__"]

    geometry_cols = _geometry_columns(config)
    starts = _drop_if_present(starts, geometry_cols)
    terminates = _drop_if_present(terminates, geometry_cols)
    return concat([starts, terminates], ignore_index=True)


def _coalesce_pandas(df: Any, valid_to_columns: List[str]) -> Any:
    present = [c for c in valid_to_columns if c in df.columns]
    if not present:
        return df[valid_to_columns[0]]
    result = df[present[0]]
    for col in present[1:]:
        result = result.fillna(df[col])
    return result


def _apply_corrupt_row_policy_pandas(df: Any, config: LifecycleEnrichmentConfig) -> Any:
    valid_from = df[config.valid_from_column]
    eff = df["__effective_valid_to__"]
    inverted_mask = eff.notna() & (valid_from > eff)
    n_inverted = int(inverted_mask.sum())
    if n_inverted == 0:
        return df

    if config.on_corrupt_row == "raise":
        raise ValueError(
            f"{n_inverted} rows have inverted lifecycle "
            f"(valid_from > effective_valid_to). Set "
            f"on_corrupt_row='skip' or 'warn' to triage."
        )
    if config.on_corrupt_row == "skip":
        logger.warning(
            "Dropping %d rows with inverted lifecycle (valid_from > effective_valid_to)",
            n_inverted,
        )
        return df[~inverted_mask]
    logger.warning(
        "Clamping %d rows with inverted lifecycle to valid_from",
        n_inverted,
    )
    df["__effective_valid_to__"] = df["__effective_valid_to__"].where(
        ~inverted_mask, df[config.valid_from_column]
    )
    return df


# ---------------------------------------------------------------------------
# Distributed (native Spark) path
# ---------------------------------------------------------------------------


def _enrich_distributed(spark_df: Any, config: LifecycleEnrichmentConfig) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    df = _cast_lifecycle_columns_to_timestamp(spark_df, config)
    df = df.withColumn(
        "__effective_valid_to__", _build_effective_valid_to_expr(df, config),
    )
    df = _apply_corrupt_row_policy_spark(df, config)

    drop_now = [c for c in config.drop_columns if c in df.columns]
    if drop_now:
        df = df.drop(*drop_now)

    starts = (
        df.withColumn(config.event_type_column, F.lit("start"))
          .withColumn(config.event_timestamp_column, F.col(config.valid_from_column))
    )
    terminates = (
        df.filter(F.col("__effective_valid_to__").isNotNull())
          .withColumn(config.event_type_column, F.lit("terminate"))
          .withColumn(config.event_timestamp_column, F.col("__effective_valid_to__"))
    )

    geometry_cols = _geometry_columns(config)
    starts_drop = [c for c in geometry_cols if c in starts.columns]
    terms_drop = [c for c in geometry_cols if c in terminates.columns]
    if starts_drop:
        starts = starts.drop(*starts_drop)
    if terms_drop:
        terminates = terminates.drop(*terms_drop)

    return starts.unionByName(terminates)


def _cast_lifecycle_columns_to_timestamp(
    spark_df: Any, config: LifecycleEnrichmentConfig
) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    df = spark_df.withColumn(
        config.valid_from_column, F.to_timestamp(F.col(config.valid_from_column)),
    )
    for col in config.valid_to_columns:
        if col in df.columns:
            df = df.withColumn(col, F.to_timestamp(F.col(col)))
    return df


def _build_effective_valid_to_expr(spark_df: Any, config: LifecycleEnrichmentConfig) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    present = [c for c in config.valid_to_columns if c in spark_df.columns]
    if not present:
        coalesced = F.lit(None).cast("timestamp")
    elif len(present) == 1:
        coalesced = F.col(present[0])
    else:
        coalesced = F.coalesce(*[F.col(c) for c in present])

    if config.status_column and config.terminal_status_values:
        is_terminal = F.col(config.status_column).isin(list(config.terminal_status_values))
        return F.when(is_terminal, coalesced).otherwise(F.lit(None).cast("timestamp"))
    return coalesced


def _apply_corrupt_row_policy_spark(
    spark_df: Any, config: LifecycleEnrichmentConfig
) -> Any:
    from pyspark.sql import functions as F  # noqa: N812

    inverted_expr = F.col("__effective_valid_to__").isNotNull() & (
        F.col(config.valid_from_column) > F.col("__effective_valid_to__")
    )

    if config.on_corrupt_row == "skip":
        return spark_df.filter(~inverted_expr)

    n_inverted = spark_df.filter(inverted_expr).count()
    if n_inverted == 0:
        return spark_df

    if config.on_corrupt_row == "raise":
        raise ValueError(
            f"{n_inverted} rows have inverted lifecycle "
            f"(valid_from > effective_valid_to). Set "
            f"on_corrupt_row='skip' or 'warn' to triage."
        )
    logger.warning(
        "Clamping %d rows with inverted lifecycle to valid_from",
        n_inverted,
    )
    return spark_df.withColumn(
        "__effective_valid_to__",
        F.when(inverted_expr, F.col(config.valid_from_column))
         .otherwise(F.col("__effective_valid_to__")),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _geometry_columns(config: LifecycleEnrichmentConfig) -> List[str]:
    cols = [config.valid_from_column, "__effective_valid_to__"]
    cols.extend(config.valid_to_columns)
    return cols


def _drop_if_present(df: Any, columns: List[str]) -> Any:
    present = [c for c in columns if c in df.columns]
    return df.drop(columns=present) if present else df
