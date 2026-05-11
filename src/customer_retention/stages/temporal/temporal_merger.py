from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from customer_retention.core.compat import Timedelta, as_tz_naive, native_pd, safe_drop_duplicates, to_datetime
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.parity import ApplyOpKind, apply_op
from customer_retention.stages.temporal.point_in_time_join import PointInTimeJoiner

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    entity_key: str = "entity_id"
    as_of_column: str = "as_of_date"
    tolerance_days: int | None = None
    conflict_separator: str = "__"
    validate_temporal: bool = True
    checkpoint_every: int = 4
    scratch_namespace: str | None = None


@dataclass
class DatasetMergeInput:
    name: str
    df: Any
    granularity: DatasetGranularity
    feature_timestamp_column: str | None = None


@dataclass
class MergeReport:
    spine_rows: int = 0
    spine_entities: int = 0
    spine_dates: int = 0
    datasets_merged: list[str] = field(default_factory=list)
    columns_per_dataset: dict[str, int] = field(default_factory=dict)
    seconds_per_dataset: dict[str, float] = field(default_factory=dict)
    total_columns: int = 0
    renamed_columns: dict[str, str] = field(default_factory=dict)
    temporal_integrity: dict[str, Any] = field(default_factory=dict)
    spine_stats_seconds: float = 0.0
    checkpoint_count: int = 0
    checkpoint_seconds: float = 0.0
    validation_seconds: float = 0.0
    merge_total_seconds: float = 0.0


class TemporalMerger:
    def __init__(self, config: MergeConfig | None = None):
        self.config = config or MergeConfig()

    def build_spine(
        self, entity_ids: Any, grid_dates: list[str]
    ) -> Any:
        unique_entities = native_pd.Series(entity_ids).drop_duplicates().reset_index(drop=True)
        parsed_dates = as_tz_naive(to_datetime(grid_dates))

        if len(unique_entities) == 0 or len(parsed_dates) == 0:
            return native_pd.DataFrame(
                columns=[self.config.entity_key, self.config.as_of_column]
            ).astype({self.config.as_of_column: "datetime64[ns]"})

        idx = native_pd.MultiIndex.from_product(
            [unique_entities, parsed_dates],
            names=[self.config.entity_key, self.config.as_of_column],
        )
        return idx.to_frame(index=False)

    @apply_op(kind=ApplyOpKind.SILVER_TEMPORAL_MERGE)
    def merge_all(
        self,
        spine: Any,
        datasets: list[DatasetMergeInput],
    ) -> tuple[Any, MergeReport]:
        report = MergeReport(
            spine_rows=len(spine),
            spine_entities=spine[self.config.entity_key].nunique() if len(spine) else 0,
            spine_dates=spine[self.config.as_of_column].nunique() if len(spine) else 0,
        )

        merged = spine.copy()

        for ds in datasets:
            existing_cols = set(merged.columns)
            before_cols = set(merged.columns)

            if ds.granularity == DatasetGranularity.EVENT_LEVEL:
                merged = self._merge_event_snapshot(merged, ds, existing_cols)
            elif ds.feature_timestamp_column and ds.feature_timestamp_column in set(ds.df.columns):
                merged = self._merge_entity_asof(merged, ds, existing_cols)
            else:
                if ds.feature_timestamp_column:
                    logger.warning(
                        "Dataset '%s': feature_timestamp_column '%s' not in columns, "
                        "using broadcast join",
                        ds.name, ds.feature_timestamp_column,
                    )
                merged = self._merge_entity_broadcast(merged, ds, existing_cols)

            new_cols = set(merged.columns) - before_cols
            report.datasets_merged.append(ds.name)
            report.columns_per_dataset[ds.name] = len(new_cols)

        report.total_columns = len(merged.columns)
        report.renamed_columns = {
            col: col
            for col in merged.columns
            if self.config.conflict_separator in col
        }

        if self.config.validate_temporal:
            report.temporal_integrity = PointInTimeJoiner.validate_temporal_integrity(
                merged
            )

        return merged, report

    def _merge_event_snapshot(
        self,
        spine: Any,
        dataset: DatasetMergeInput,
        existing_cols: set[str],
    ) -> Any:
        right_df = dataset.df.copy()

        if self.config.as_of_column not in right_df.columns:
            right_df = safe_drop_duplicates(right_df, subset=[self.config.entity_key], keep="last")
            return self._merge_entity_broadcast(
                spine, DatasetMergeInput(
                    name=dataset.name, df=right_df,
                    granularity=dataset.granularity,
                    feature_timestamp_column=dataset.feature_timestamp_column,
                ), existing_cols,
            )

        join_cols = [self.config.entity_key, self.config.as_of_column]
        right_df = safe_drop_duplicates(right_df, subset=join_cols, keep="last")

        right_feature_cols = set(right_df.columns) - set(join_cols)
        rename_map = self._resolve_conflicts(
            existing_cols, right_feature_cols, set(join_cols),
            dataset.name, self.config.conflict_separator,
        )
        if rename_map:
            right_df = right_df.rename(columns=rename_map)

        return spine.merge(right_df, on=join_cols, how="left")

    def _merge_entity_broadcast(
        self,
        spine: Any,
        dataset: DatasetMergeInput,
        existing_cols: set[str],
    ) -> Any:
        join_keys = {self.config.entity_key}
        right_df = safe_drop_duplicates(dataset.df, subset=[self.config.entity_key], keep="last")

        right_feature_cols = set(right_df.columns) - join_keys
        rename_map = self._resolve_conflicts(
            existing_cols, right_feature_cols, join_keys,
            dataset.name, self.config.conflict_separator,
        )
        if rename_map:
            right_df = right_df.rename(columns=rename_map)

        return spine.merge(right_df, on=self.config.entity_key, how="left")

    def _merge_entity_asof(
        self,
        spine: Any,
        dataset: DatasetMergeInput,
        existing_cols: set[str],
    ) -> Any:
        join_keys = {self.config.entity_key, self.config.as_of_column}
        ft_col = dataset.feature_timestamp_column
        right_df = dataset.df.copy()

        right_feature_cols = set(right_df.columns) - {self.config.entity_key, ft_col}
        rename_map = self._resolve_conflicts(
            existing_cols, right_feature_cols, join_keys,
            dataset.name, self.config.conflict_separator,
        )
        if rename_map:
            right_df = right_df.rename(columns=rename_map)
            ft_col = rename_map.get(ft_col, ft_col)

        result = PointInTimeJoiner.asof_join(
            spine,
            right_df,
            entity_key=self.config.entity_key,
            left_time_col=self.config.as_of_column,
            right_time_col=ft_col,
            direction="backward",
        )

        if ft_col in result.columns and ft_col not in existing_cols:
            result = result.drop(columns=[ft_col])

        if len(result) != len(spine):
            to_merge = result
            if self.config.as_of_column in result.columns:
                to_merge = result.drop(columns=[self.config.as_of_column])
            result = spine.merge(to_merge, on=self.config.entity_key, how="left")

        return result

    @staticmethod
    def _resolve_conflicts(
        left_cols: set[str],
        right_cols: set[str],
        join_keys: set[str],
        name: str,
        sep: str,
    ) -> dict[str, str]:
        # Case-insensitive collision detection. Python set intersection is
        # case-sensitive; Spark / Delta column resolution at write time is
        # case-insensitive. Without this, a right column `count_of_open_opps`
        # against a left column `COUNT_OF_OPEN_OPPS` slips through dedup,
        # both end up in the joined frame, and downstream gold encoding /
        # saveAsTable rejects them as duplicates with COLUMN_ALREADY_EXISTS.
        # Detection matches case-insensitively; rename target preserves the
        # right column's original case so existing reads against the
        # produced name continue to work.
        left_lower = {c.lower() for c in left_cols}
        join_lower = {c.lower() for c in join_keys}
        rename: dict[str, str] = {}
        for col in sorted(right_cols):
            col_lower = col.lower()
            if col_lower in join_lower:
                continue
            if col_lower in left_lower:
                rename[col] = f"{name}{sep}{col}"
        return rename

    @staticmethod
    def _apply_tolerance(
        df: Any,
        ft_col: str,
        as_of_col: str,
        days: int | None,
        feature_cols: list[str] | None = None,
    ) -> tuple[Any, int]:
        if days is None:
            return df, 0

        result = df.copy()
        stale_mask = (
            result[as_of_col] - result[ft_col]
        ) > Timedelta(days=days)

        null_count = int(stale_mask.sum())

        if null_count > 0 and feature_cols:
            for col in feature_cols:
                if col in result.columns:
                    result.loc[stale_mask, col] = native_pd.NA

        return result, null_count


_DEFAULT_SPINE_COLUMNS = ("entity_id", "as_of_date")


def classify_silver_columns(
    silver_columns,
    bronze_columns_by_dataset,
    base_source,
    conflict_separator="__",
    spine_columns=_DEFAULT_SPINE_COLUMNS,
):
    if base_source not in bronze_columns_by_dataset:
        raise KeyError(
            f"base_source={base_source!r} missing from bronze_columns_by_dataset "
            f"(got {sorted(bronze_columns_by_dataset)})"
        )

    origin: dict[str, str] = {c: "spine" for c in spine_columns}
    merged = set(spine_columns) | set(bronze_columns_by_dataset[base_source])
    for c in bronze_columns_by_dataset[base_source]:
        origin[c] = base_source

    ordered = [base_source] + [
        n for n in bronze_columns_by_dataset if n != base_source
    ]
    renamed: dict[str, str] = {}
    for name in ordered[1:]:
        cols = set(bronze_columns_by_dataset[name])
        conflicts = cols & merged
        non_conflicts = cols - merged
        for c in conflicts:
            alias = f"{name}{conflict_separator}{c}"
            renamed[alias] = name
            merged.add(alias)
        for c in non_conflicts:
            origin.setdefault(c, name)
            merged.add(c)

    result: dict[str, str] = {}
    for c in silver_columns:
        if c in origin:
            result[c] = origin[c]
        elif c in renamed:
            result[c] = renamed[c]
        else:
            raise ValueError(
                f"silver column {c!r} has no origin — not in spine, base source "
                f"{base_source!r}, any merged bronze, or any renamed alias"
            )
    return result
