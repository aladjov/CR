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
    is the source of truth for those fields. Returns a DataFrame in the
    same backend as ``parent_df``.
    """
    _require_join_key(parent_df, parent_record_key, "parent_df")
    _require_join_key(state_view, parent_record_key, "state_view")

    overlap_cols = _parent_overlap_columns(parent_df, state_view, parent_record_key)

    if _is_spark_pandas(parent_df) or _is_native_spark_df(parent_df):
        return _augment_distributed(
            parent_df, state_view, parent_record_key, overlap_cols, join_type
        )
    return _augment_pandas(
        parent_df, state_view, parent_record_key, overlap_cols, join_type
    )


def _require_join_key(df: Any, key: str, label: str) -> None:
    if key not in df.columns:
        raise ValueError(
            f"parent_record_key {key!r} not found in {label} columns: "
            f"{list(df.columns)}"
        )


def _parent_overlap_columns(
    parent_df: Any, state_view: Any, parent_record_key: str
) -> list[str]:
    state_columns_ci = {str(c).lower() for c in state_view.columns}
    state_columns_ci.discard(parent_record_key.lower())
    return [c for c in parent_df.columns if str(c).lower() in state_columns_ci]


def _augment_distributed(
    parent_df: Any,
    state_view: Any,
    parent_record_key: str,
    overlap_cols: list[str],
    join_type: str,
) -> Any:
    spark_parent = as_spark_df(parent_df) if _is_spark_pandas(parent_df) else parent_df
    spark_state = as_spark_df(state_view) if _is_spark_pandas(state_view) else state_view
    spark_static = (
        spark_parent.drop(*overlap_cols) if overlap_cols else spark_parent
    )
    joined = spark_static.join(spark_state, on=parent_record_key, how=join_type)
    if _is_spark_pandas(parent_df):
        return as_pandas_api(joined)
    return joined


def _augment_pandas(
    parent_df: Any,
    state_view: Any,
    parent_record_key: str,
    overlap_cols: list[str],
    join_type: str,
) -> Any:
    parent_static = (
        parent_df.drop(columns=overlap_cols) if overlap_cols else parent_df
    )
    return parent_static.merge(state_view, on=parent_record_key, how=join_type)
