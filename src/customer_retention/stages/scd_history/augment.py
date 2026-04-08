"""Augment a parent dataset with reconstructed SCD state.

The augmentation joins a static parent dataset (one row per parent record)
with a reconstructed state view (one row per ``(parent_record, anchor_date)``
from :func:`reconstruct_scd_history_at_grid`). Parent columns whose name
case-insensitively collides with a column on the state view are dropped from
the parent before the join — Spark's default ``spark.sql.caseSensitive=false``
resolver would otherwise raise ``[AMBIGUOUS_REFERENCE]`` when downstream code
reads the joined schema. The state view is the source of truth for those
fields (it has already applied any parent fallback configured on the
reconstruction config).

Stays distributed end-to-end on Databricks: a single Spark join + bulk
column drop, no per-row Python collection. Backend dispatch matches
:func:`reconstruct_scd_history_at_grid` so callers can pass native Spark,
pyspark.pandas, or native pandas DataFrames interchangeably.

Fail-fast guards surface upstream column-name dupes (e.g. a Snowflake source
that has both ``Origin`` and ``ORIGIN`` from quoted-identifier history) at
this cell, instead of letting them propagate to NB01's data-load step as an
``[AMBIGUOUS_REFERENCE]`` error several cells removed from the root cause.
"""
from __future__ import annotations

from typing import Any

from customer_retention.core.compat import (
    _is_native_spark_df,
    _is_spark_pandas,
    as_pandas_api,
    as_spark_df,
)


def augment_parent_with_scd_state(
    parent_df: Any,
    state_view: Any,
    parent_record_key: str,
    *,
    join_type: str = "inner",
) -> Any:
    """Join ``parent_df`` to ``state_view`` on ``parent_record_key``.

    Drops parent columns whose name case-insensitively matches any column on
    ``state_view`` (excluding the join key) before the join. The state view
    is the source of truth for those fields. The drop handles upstream
    parent dupes (multiple case variants of the same name) by rebuilding
    the parent through positional aliases. Returns a DataFrame in the same
    backend as ``parent_df``.
    """
    _require_join_key(parent_df, parent_record_key, "parent_df")
    _require_join_key(state_view, parent_record_key, "state_view")
    _assert_no_case_insensitive_duplicates(state_view, "state_view")

    state_columns_ci = _state_collision_set(state_view, parent_record_key)

    if _is_spark_pandas(parent_df) or _is_native_spark_df(parent_df):
        return _augment_distributed(
            parent_df, state_view, parent_record_key, state_columns_ci, join_type
        )
    return _augment_pandas(
        parent_df, state_view, parent_record_key, state_columns_ci, join_type
    )


def _require_join_key(df: Any, key: str, label: str) -> None:
    if key not in df.columns:
        raise ValueError(
            f"parent_record_key {key!r} not found in {label} columns: "
            f"{list(df.columns)}"
        )


def _state_collision_set(state_view: Any, parent_record_key: str) -> set[str]:
    state_columns_ci = {str(c).lower() for c in state_view.columns}
    state_columns_ci.discard(parent_record_key.lower())
    return state_columns_ci


def _case_insensitive_duplicates(columns: list[str]) -> list[str]:
    seen: dict[str, list[str]] = {}
    for c in columns:
        seen.setdefault(str(c).lower(), []).append(str(c))
    return sorted(name for variants in seen.values() if len(variants) > 1 for name in variants)


def _assert_no_case_insensitive_duplicates(df: Any, label: str) -> None:
    duplicates = _case_insensitive_duplicates(list(df.columns))
    if duplicates:
        raise ValueError(
            f"{label} has case-insensitive duplicate columns: {duplicates}. "
            f"Spark cannot disambiguate them downstream and "
            f"`[AMBIGUOUS_REFERENCE]` will surface at the next analysis step. "
            f"Fix at the source — drop or rename the offending columns "
            f"before passing the DataFrame to augment_parent_with_scd_state()."
        )


def _augment_distributed(
    parent_df: Any,
    state_view: Any,
    parent_record_key: str,
    state_columns_ci: set[str],
    join_type: str,
) -> Any:
    spark_parent = as_spark_df(parent_df) if _is_spark_pandas(parent_df) else parent_df
    spark_state = as_spark_df(state_view) if _is_spark_pandas(state_view) else state_view
    spark_static = _drop_collisions_distributed(
        spark_parent, state_columns_ci, parent_record_key
    )
    _assert_no_case_insensitive_duplicates(spark_static, "parent_df after collision drop")
    joined = spark_static.join(spark_state, on=parent_record_key, how=join_type)
    _assert_no_case_insensitive_duplicates(joined, "augmented output")
    if _is_spark_pandas(parent_df):
        return as_pandas_api(joined)
    return joined


def _drop_collisions_distributed(
    spark_parent: Any, state_columns_ci: set[str], parent_record_key: str
) -> Any:
    """Rebuild ``spark_parent`` keeping only columns that do not collide.

    Uses positional aliases (``__cr_pos_{i}__``) so that name resolution
    cannot be ambiguous when the parent has upstream duplicates (e.g. both
    ``Origin`` and ``ORIGIN``). Both case variants of any colliding name
    are dropped because the state view is the source of truth for that
    field. The join key is preserved.
    """
    original_columns = list(spark_parent.columns)
    positional_names = [f"__cr_pos_{i}__" for i in range(len(original_columns))]
    parent_positional = spark_parent.toDF(*positional_names)

    join_key_lower = parent_record_key.lower()
    keep_indices = [
        i
        for i, original in enumerate(original_columns)
        if str(original).lower() == join_key_lower
        or str(original).lower() not in state_columns_ci
    ]
    keep_positional = [positional_names[i] for i in keep_indices]
    keep_originals = [original_columns[i] for i in keep_indices]
    return parent_positional.select(*keep_positional).toDF(*keep_originals)


def _augment_pandas(
    parent_df: Any,
    state_view: Any,
    parent_record_key: str,
    state_columns_ci: set[str],
    join_type: str,
) -> Any:
    parent_static = _drop_collisions_pandas(
        parent_df, state_columns_ci, parent_record_key
    )
    _assert_no_case_insensitive_duplicates(parent_static, "parent_df after collision drop")
    joined = parent_static.merge(state_view, on=parent_record_key, how=join_type)
    _assert_no_case_insensitive_duplicates(joined, "augmented output")
    return joined


def _drop_collisions_pandas(
    parent_df: Any, state_columns_ci: set[str], parent_record_key: str
) -> Any:
    join_key_lower = parent_record_key.lower()
    keep_mask = [
        str(c).lower() == join_key_lower or str(c).lower() not in state_columns_ci
        for c in parent_df.columns
    ]
    return parent_df.loc[:, keep_mask]
