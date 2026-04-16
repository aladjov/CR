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

:func:`augment_and_persist_parent_dataset` is the single supported NB00
entry point. It composes :func:`augment_parent_with_scd_state` with
``save_active_dataset`` so the augmented schema lands on the canonical
landing Delta path that NB01 reads from. Manual ``join + register_temp_view``
patterns produce session-scoped views that are invisible to downstream
notebooks — use the facade instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from customer_retention.core.compat import (
    _is_native_spark_df,
    _is_spark_pandas,
    as_pandas_api,
    as_spark_df,
)
from customer_retention.core.compat.detection import get_default_parallelism

from .config import SCDHistoryReconstructionConfig

_AUGMENT_PARTITION_FLOOR_MULTIPLIER = 8
_AUGMENT_MAX_PARTITIONS = 4096
_DEFAULT_RETENTION_FLOOR = 0.70


@dataclass(frozen=True)
class AugmentationDiagnostics:
    parent_record_count: int
    augmented_record_count: int
    retention: float
    augmented_rows: int
    augmented_cols: int
    grid_anchor_count: int
    fanout: float

    def assert_retention(self, floor: float = _DEFAULT_RETENTION_FLOOR, parent_name: str = "") -> None:
        if self.retention < floor:
            label = f" for {parent_name!r}" if parent_name else ""
            raise AssertionError(
                f"Augmentation retention {self.retention:.1%}{label} below "
                f"{floor:.0%} floor — investigate parent-table join coverage."
            )

    def summary_lines(self, parent_name: str = "") -> list[str]:
        label = f"[{parent_name}] " if parent_name else ""
        return [
            f"SCD augmentation {label}retained {self.augmented_record_count:,} of "
            f"{self.parent_record_count:,} parent records ({self.retention:.1%})",
            f"SCD augmentation {label}expanded to "
            f"{self.augmented_rows:,} rows x {self.augmented_cols} cols "
            f"across {self.grid_anchor_count} grid anchors "
            f"(fan-out {self.fanout:.1f}x per retained parent)",
        ]


@dataclass
class AugmentationResult:
    augmented_df: Any
    schema: dict[str, str]
    diagnostics: AugmentationDiagnostics
    landing_path: str

    def __iter__(self) -> Iterator:
        return iter((self.augmented_df, self.schema))


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


def augment_and_persist_parent_dataset(
    *,
    namespace: Any,
    parent_dataset_name: str,
    parent_df: Any,
    state_view: Any,
    config: SCDHistoryReconstructionConfig,
    join_type: str = "inner",
    grid_anchor_count: Optional[int] = None,
    target_partitions: Optional[int] = None,
) -> AugmentationResult:
    """Augment a parent with SCD state and overwrite its landing Delta.

    The single supported entry point for SCD augmentation in NB00.

    Returns :class:`AugmentationResult` which supports 2-tuple unpacking
    ``(augmented_df, schema_dict)`` for backward compatibility, and also
    exposes ``.diagnostics`` (:class:`AugmentationDiagnostics`) and
    ``.landing_path``.

    Distributed-safe end-to-end: reuses :func:`augment_parent_with_scd_state`
    (native Spark ``.toDF()`` aliasing + ``.select()`` + ``.join(on=key)``)
    and ``save_active_dataset`` (``as_spark_df`` + ``delta.write``). The
    schema-drift verification reads only ``columns`` from the persisted
    Delta — no row scan, no ``.collect()``.

    ``target_partitions`` lets the caller override the write-path partition
    floor — SCD fan-out (parents × anchors) can produce billions of rows; a
    64-partition default lands a multi-GB spill per task which exhausts
    executor disk. When unset, the facade derives a hint from
    ``grid_anchor_count * default_parallelism * 8`` (capped at 4096) which
    scales partition count with fan-out magnitude.
    """
    _require_dataset_name(parent_dataset_name)
    _require_config_matches(parent_dataset_name, config)

    augmented = augment_parent_with_scd_state(
        parent_df, state_view, config.parent_record_key, join_type=join_type,
    )
    _assert_no_case_insensitive_duplicates(
        augmented, "augment_and_persist_parent_dataset output",
    )

    effective_partitions = _resolve_target_partitions(
        target_partitions, grid_anchor_count
    )
    augmented_for_write = _repartition_by_record_key(
        augmented, config.parent_record_key, effective_partitions,
    )
    landing_path = _persist_to_landing(
        namespace, parent_dataset_name, augmented_for_write, effective_partitions,
    )
    _assert_landing_delta_matches_schema(
        landing_path, augmented, parent_dataset_name,
    )

    diagnostics = _compute_diagnostics(
        parent_df, augmented, config.parent_record_key, grid_anchor_count or 0,
    )
    return AugmentationResult(
        augmented_df=augmented,
        schema=_augmented_schema(augmented),
        diagnostics=diagnostics,
        landing_path=str(landing_path),
    )


def _resolve_target_partitions(
    explicit: Optional[int], grid_anchor_count: Optional[int]
) -> Optional[int]:
    if explicit is not None:
        return max(int(explicit), 1)
    if not grid_anchor_count or grid_anchor_count <= 0:
        return None
    cores = get_default_parallelism()
    if cores <= 0:
        return None
    derived = int(grid_anchor_count) * int(cores) * _AUGMENT_PARTITION_FLOOR_MULTIPLIER
    return max(min(derived, _AUGMENT_MAX_PARTITIONS), cores)


def _repartition_by_record_key(
    augmented: Any, parent_record_key: str, target_partitions: Optional[int]
) -> Any:
    if target_partitions is None or target_partitions <= 0:
        return augmented
    if _is_native_spark_df(augmented):
        return augmented.repartition(int(target_partitions), parent_record_key)
    if _is_spark_pandas(augmented):
        spark_df = as_spark_df(augmented).repartition(
            int(target_partitions), parent_record_key
        )
        return as_pandas_api(spark_df)
    return augmented


def _require_dataset_name(name: str) -> None:
    if not name or not str(name).strip():
        raise ValueError(
            "augment_and_persist_parent_dataset: parent_dataset_name must "
            "not be empty or whitespace-only",
        )


def _require_config_matches(
    parent_dataset_name: str, config: SCDHistoryReconstructionConfig
) -> None:
    configured = config.parent_table_dataset_name
    if configured is not None and configured != parent_dataset_name:
        raise ValueError(
            f"augment_and_persist_parent_dataset: parent_table_dataset_name "
            f"mismatch — config says {configured!r}, caller passed "
            f"{parent_dataset_name!r}. Refusing to persist under the wrong "
            f"dataset name (catches copy-paste of the wrong config object).",
        )


def _persist_to_landing(
    namespace: Any,
    dataset_name: str,
    augmented: Any,
    target_partitions: Optional[int] = None,
) -> Any:
    from customer_retention.analysis.auto_explorer.active_dataset_store import (
        save_active_dataset,
    )

    return save_active_dataset(
        namespace, dataset_name, augmented, target_partitions=target_partitions,
    )


def _read_landing_columns(landing_path: Any) -> list[str]:
    from customer_retention.integrations.adapters.factory import get_delta

    return list(get_delta().read(str(landing_path)).columns)


def _assert_landing_delta_matches_schema(
    landing_path: Any, augmented: Any, dataset_name: str
) -> None:
    actual = {str(c).lower() for c in _read_landing_columns(landing_path)}
    expected = {str(c).lower() for c in augmented.columns}
    if actual == expected:
        return
    only_actual = sorted(actual - expected)
    only_expected = sorted(expected - actual)
    raise ValueError(
        f"augment_and_persist_parent_dataset: landing Delta schema drift "
        f"for dataset {dataset_name!r} — Delta has columns not in augmented "
        f"frame={only_actual}, augmented frame has columns not in Delta="
        f"{only_expected}",
    )


def _compute_diagnostics(
    parent_df: Any, augmented: Any, parent_record_key: str, grid_anchor_count: int,
) -> AugmentationDiagnostics:
    parent_count = _distinct_key_count(parent_df, parent_record_key)
    augmented_count = _distinct_key_count(augmented, parent_record_key)
    augmented_rows = _row_count(augmented)
    return AugmentationDiagnostics(
        parent_record_count=parent_count,
        augmented_record_count=augmented_count,
        retention=augmented_count / parent_count if parent_count else 0.0,
        augmented_rows=augmented_rows,
        augmented_cols=len(augmented.columns),
        grid_anchor_count=grid_anchor_count,
        fanout=augmented_rows / augmented_count if augmented_count else 0.0,
    )


def _distinct_key_count(df: Any, key: str) -> int:
    if _is_native_spark_df(df):
        return df.select(key).distinct().count()
    if _is_spark_pandas(df):
        return as_spark_df(df).select(key).distinct().count()
    return int(df[key].nunique())


def _row_count(df: Any) -> int:
    if _is_native_spark_df(df):
        return df.count()
    if _is_spark_pandas(df):
        return as_spark_df(df).count()
    return len(df)


def _augmented_schema(augmented: Any) -> dict[str, str]:
    if hasattr(augmented, "dtypes"):
        dtypes = augmented.dtypes
        if hasattr(dtypes, "items"):
            return {str(name): str(dtype) for name, dtype in dtypes.items()}
        return {str(name): str(dtype) for name, dtype in dtypes}
    return {str(c): "unknown" for c in augmented.columns}
