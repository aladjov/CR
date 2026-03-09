from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from customer_retention.core.compat import (
    _is_spark_pandas,
    groupby_multi_agg,
    pd,
    timedelta_to_days,
    to_pandas,
)

from .temporal_feature_engineer import (
    FeatureGroup,
    FeatureGroupResult,
    TemporalFeatureEngineer,
    TemporalFeatureResult,
)


class SparkTemporalFeatureEngineer(TemporalFeatureEngineer):

    def compute(
        self,
        events_df: Any,
        entity_col: str,
        time_col: str,
        value_cols: List[str],
        reference_dates: Optional[Any] = None,
        reference_col: Optional[str] = None,
    ) -> TemporalFeatureResult:
        events_df = to_pandas(events_df)
        if reference_dates is not None:
            reference_dates = to_pandas(reference_dates)
        return super().compute(
            events_df, entity_col, time_col, value_cols, reference_dates, reference_col)

    def _compute_lifecycle(
        self,
        events_df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        value_cols: List[str],
        ref_dates: pd.DataFrame,
    ) -> tuple:
        result = ref_dates[[entity_col]].copy()
        feature_names = []
        min_days = self.config.min_history_days
        splits = self.config.lifecycle_splits

        history_stats = groupby_multi_agg(events_df, entity_col, time_col, ["min", "max"])
        history_stats.columns = [entity_col, "first_event", "last_event"]
        history_stats["history_days"] = timedelta_to_days(
            history_stats["last_event"] - history_stats["first_event"]
        )

        eligible = history_stats[
            history_stats["history_days"].notna() & (history_stats["history_days"] >= min_days)
        ]

        df = events_df.merge(eligible, on=entity_col)
        df["_split1"] = df["first_event"] + pd.to_timedelta(df["history_days"] * splits[0], unit="D")
        df["_split2"] = df["first_event"] + pd.to_timedelta(
            df["history_days"] * (splits[0] + splits[1]), unit="D")

        df["_phase"] = np.where(
            df[time_col] < df["_split1"], "beginning",
            np.where(df[time_col] < df["_split2"], "middle", "end"))

        for col in value_cols:
            for phase in ["beginning", "middle", "end"]:
                feat_name = f"{col}_{phase}"
                feature_names.append(feat_name)
                phase_sums = (
                    df[df["_phase"] == phase]
                    .groupby(entity_col)[col]
                    .sum()
                    .reset_index(name=feat_name)
                )
                result = result.merge(phase_sums, on=entity_col, how="left")

            trend_name = f"{col}_trend_ratio"
            feature_names.append(trend_name)
            beg_col = f"{col}_beginning"
            end_col = f"{col}_end"
            result[trend_name] = np.where(
                result[beg_col].notna() & (result[beg_col] > 0),
                result[end_col] / result[beg_col],
                np.nan,
            )

        return result, FeatureGroupResult(
            group=FeatureGroup.LIFECYCLE,
            features=feature_names,
            rationale=self.RATIONALES[FeatureGroup.LIFECYCLE],
        )

    def _compute_regularity(
        self,
        events_df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        ref_dates: pd.DataFrame,
    ) -> tuple:
        result = ref_dates[[entity_col]].copy()

        sorted_events = events_df.sort_values([entity_col, time_col])
        if _is_spark_pandas(sorted_events):
            import pyspark.sql.functions as F  # noqa: N812
            sorted_events["_ts_epoch"] = sorted_events[time_col].spark.transform(
                lambda c: F.unix_timestamp(c.cast("timestamp")).cast("double"))
            prev_epoch = sorted_events.groupby(entity_col)["_ts_epoch"].shift(1)
            sorted_events["_gap_seconds"] = (sorted_events["_ts_epoch"] - prev_epoch) / 86400
            sorted_events = sorted_events.drop(columns=["_ts_epoch"])
        else:
            sorted_events["_prev_time"] = sorted_events.groupby(entity_col)[time_col].shift(1)
            sorted_events["_gap_seconds"] = timedelta_to_days(
                sorted_events[time_col] - sorted_events["_prev_time"])
        gaps = sorted_events.dropna(subset=["_gap_seconds"])

        if len(gaps) > 0:
            grp = gaps.groupby(entity_col)["_gap_seconds"]
            gap_mean = grp.mean().reset_index(name="inter_event_gap_mean")
            gap_std = grp.std().reset_index(name="inter_event_gap_std")
            gap_max = grp.max().reset_index(name="inter_event_gap_max")
            gap_stats = gap_mean.merge(gap_std, on=entity_col).merge(gap_max, on=entity_col)
            result = result.merge(gap_stats, on=entity_col, how="left")
        else:
            result["inter_event_gap_mean"] = np.nan
            result["inter_event_gap_std"] = np.nan
            result["inter_event_gap_max"] = np.nan

        event_stats = groupby_multi_agg(events_df, entity_col, time_col, ["min", "max", "count"])
        event_stats.columns = [entity_col, "_first", "_last", "_count"]
        event_stats["_total_days"] = timedelta_to_days(event_stats["_last"] - event_stats["_first"])

        event_stats["event_frequency"] = np.where(
            event_stats["_total_days"] > 0,
            event_stats["_count"] / event_stats["_total_days"] * 30,
            event_stats["_count"],
        )
        result = result.merge(
            event_stats[[entity_col, "event_frequency"]], on=entity_col, how="left")

        if "inter_event_gap_mean" in result.columns:
            gap_mean = result["inter_event_gap_mean"]
            gap_std = result["inter_event_gap_std"].fillna(0)
            result["regularity_score"] = np.where(
                gap_mean.notna() & (gap_mean > 0),
                np.maximum(0, 1 - gap_std / gap_mean),
                np.where(gap_mean.notna() & (gap_mean == 0), 1.0, np.nan),
            )
        else:
            result["regularity_score"] = np.nan

        for col in ["event_frequency", "inter_event_gap_mean", "inter_event_gap_std",
                     "inter_event_gap_max", "regularity_score"]:
            if col not in result.columns:
                result[col] = np.nan

        feature_names = [
            "event_frequency", "inter_event_gap_mean", "inter_event_gap_std",
            "inter_event_gap_max", "regularity_score"
        ]

        return result, FeatureGroupResult(
            group=FeatureGroup.REGULARITY,
            features=feature_names,
            rationale=self.RATIONALES[FeatureGroup.REGULARITY],
        )
