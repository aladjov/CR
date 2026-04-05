from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from customer_retention.core.compat import Timedelta, as_tz_naive, native_pd, safe_drop_duplicates, to_datetime
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.temporal.point_in_time_join import PointInTimeJoiner

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    entity_key: str = "entity_id"
    as_of_column: str = "as_of_date"
    tolerance_days: int | None = None
    conflict_separator: str = "__"
    validate_temporal: bool = True


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
    total_columns: int = 0
    renamed_columns: dict[str, str] = field(default_factory=dict)
    temporal_integrity: dict[str, Any] = field(default_factory=dict)


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
        conflicts = (left_cols & right_cols) - join_keys
        return {col: f"{name}{sep}{col}" for col in sorted(conflicts)}

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
