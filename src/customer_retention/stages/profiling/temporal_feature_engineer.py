"""TemporalFeatureEngineer - temporal feature engineering with lagged windows.

Generates features across 7 groups:
1. Lagged Windows - Sequential non-overlapping time windows
2. Velocity - Rate of change between windows
3. Acceleration - Change in velocity (momentum)
4. Lifecycle - Beginning/Middle/End of customer history
5. Recency - Days since last/first event, tenure
6. Regularity - Frequency and consistency patterns
7. Cohort Comparison - Customer vs cohort averages

Key Concepts:
    Per-Customer Alignment: Each customer's features are computed relative to
    their own reference point (e.g., churn date, last activity), making
    historical churners comparable to current active customers.

    Lagged Windows: Sequential non-overlapping windows (Lag0=most recent,
    Lag1=previous period, etc.) enable velocity/acceleration computation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import (
    groupby_multi_agg,
    groupby_multi_col_agg,
    native_pd,
    pd,
    timedelta_to_days,
    to_datetime,
    to_numeric,
)


class ReferenceMode(Enum):
    """How to determine reference point for temporal alignment."""
    PER_CUSTOMER = "per_customer"  # Each customer has own reference date
    GLOBAL_DATE = "global_date"    # Single date for all customers


class FeatureGroup(Enum):
    """Categories of temporal features."""
    LAGGED_WINDOWS = "lagged_windows"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    LIFECYCLE = "lifecycle"
    RECENCY = "recency"
    REGULARITY = "regularity"
    COHORT_COMPARISON = "cohort_comparison"


@dataclass
class TemporalAggregationConfig:
    """Configuration for temporal feature engineering."""

    # Reference point alignment
    reference_mode: ReferenceMode = ReferenceMode.PER_CUSTOMER
    global_reference_date: Optional[datetime] = None

    # Lagged windows (Group 1)
    lag_window_days: int = 30
    num_lags: int = 4
    lag_aggregations: List[str] = field(default_factory=lambda: ["sum", "mean", "count", "max"])

    # Velocity/Acceleration (Groups 2-3)
    compute_velocity: bool = True
    compute_acceleration: bool = True

    # Lifecycle windows (Group 4)
    compute_lifecycle: bool = True
    lifecycle_splits: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    min_history_days: int = 60

    # Recency/Tenure (Group 5)
    compute_recency: bool = True

    # Frequency/Regularity (Group 6)
    compute_regularity: bool = True

    # Cohort Comparison (Group 7)
    compute_cohort: bool = True


@dataclass
class FeatureGroupResult:
    """Result for a single feature group."""
    group: FeatureGroup
    features: List[str]
    rationale: str
    enabled: bool = True


@dataclass
class TemporalFeatureResult:
    """Result from temporal feature computation."""
    features_df: pd.DataFrame
    feature_groups: List[FeatureGroupResult]
    config: TemporalAggregationConfig
    entity_col: str
    value_cols: List[str]

    def get_catalog(self) -> str:
        """Generate formatted feature catalog with rationale."""
        lines = []
        lines.append("=" * 80)
        lines.append("TEMPORAL FEATURE CATALOG")
        lines.append("=" * 80)

        for group_result in self.feature_groups:
            if not group_result.enabled:
                continue

            lines.append("")
            lines.append(f"GROUP: {group_result.group.value.upper()} ({len(group_result.features)} features)")
            lines.append(f"Rationale: {group_result.rationale}")
            lines.append("-" * 60)

            for feat in group_result.features[:10]:
                lines.append(f"  - {feat}")
            if len(group_result.features) > 10:
                lines.append(f"  ... and {len(group_result.features) - 10} more")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_features": len(self.features_df.columns) - 1,  # Exclude entity col
            "n_entities": len(self.features_df),
            "feature_groups": [
                {"group": g.group.value, "n_features": len(g.features), "enabled": g.enabled}
                for g in self.feature_groups
            ],
        }


class TemporalFeatureEngineer:
    """Engineers temporal features from event data with per-customer alignment.

    Supports 7 feature groups:
    1. Lagged Windows - lag{N}_{metric}_{agg}
    2. Velocity - {metric}_velocity, {metric}_velocity_pct
    3. Acceleration - {metric}_acceleration, {metric}_momentum
    4. Lifecycle - {metric}_beginning, {metric}_middle, {metric}_end, {metric}_trend_ratio
    5. Recency - days_since_last_event, days_since_first_event, active_span_days
    6. Regularity - event_frequency, inter_event_gap_mean, regularity_score
    7. Cohort - {metric}_vs_cohort_mean, {metric}_vs_cohort_pct
    """

    RATIONALES = {
        FeatureGroup.LAGGED_WINDOWS: "Capture behavior at sequential time horizons to enable trend detection",
        FeatureGroup.VELOCITY: "Rate of change is the #1 churn predictor - declining engagement signals risk",
        FeatureGroup.ACCELERATION: "Is the decline accelerating or stabilizing? Indicates intervention urgency",
        FeatureGroup.LIFECYCLE: "Customer lifecycle patterns reveal engagement trajectory over full history",
        FeatureGroup.RECENCY: "How recently active and tenure are fundamental churn signals",
        FeatureGroup.REGULARITY: "Consistent patterns indicate habit formation; irregular patterns suggest weak retention",
        FeatureGroup.COHORT_COMPARISON: "Compare customer to peers - is their behavior normal or anomalous?",
    }

    def __init__(self, config: Optional[TemporalAggregationConfig] = None):
        self.config = config or TemporalAggregationConfig()

    def compute(
        self,
        events_df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        value_cols: List[str],
        reference_dates: Optional[pd.DataFrame] = None,
        reference_col: Optional[str] = None,
    ) -> TemporalFeatureResult:
        """Compute temporal features for all entities.

        Args:
            events_df: Event-level data with timestamps
            entity_col: Column identifying entities (e.g., customer_id)
            time_col: Column with event timestamps
            value_cols: Columns to aggregate (e.g., amount, quantity)
            reference_dates: DataFrame with entity and reference date columns
            reference_col: Column name for reference date in reference_dates

        Returns:
            TemporalFeatureResult with features DataFrame and metadata
        """
        prep = events_df[[entity_col, time_col, *[c for c in value_cols if c in events_df.columns]]].copy()
        prep[time_col] = to_datetime(prep[time_col])
        for vc in value_cols:
            if vc in prep.columns:
                prep[vc] = to_numeric(prep[vc], errors="coerce").astype("float64")

        ref_dates = self._get_reference_dates(
            prep, entity_col, time_col, reference_dates, reference_col
        )

        all_features = []
        feature_groups = []

        lag_features, lag_group = self._compute_lagged_windows(
            prep, entity_col, time_col, value_cols, ref_dates
        )
        all_features.append(lag_features)
        feature_groups.append(lag_group)

        if self.config.compute_velocity:
            velocity_features, velocity_group = self._compute_velocity(lag_features, value_cols)
            all_features.append(velocity_features)
            feature_groups.append(velocity_group)
        else:
            feature_groups.append(FeatureGroupResult(
                group=FeatureGroup.VELOCITY, features=[],
                rationale=self.RATIONALES[FeatureGroup.VELOCITY], enabled=False
            ))

        if self.config.compute_acceleration and self.config.compute_velocity:
            accel_features, accel_group = self._compute_acceleration(
                all_features[1] if len(all_features) > 1 else lag_features,
                lag_features, value_cols, entity_col
            )
            all_features.append(accel_features)
            feature_groups.append(accel_group)
        else:
            feature_groups.append(FeatureGroupResult(
                group=FeatureGroup.ACCELERATION, features=[],
                rationale=self.RATIONALES[FeatureGroup.ACCELERATION], enabled=False
            ))

        if self.config.compute_lifecycle:
            lifecycle_features, lifecycle_group = self._compute_lifecycle(
                prep, entity_col, time_col, value_cols, ref_dates
            )
            all_features.append(lifecycle_features)
            feature_groups.append(lifecycle_group)
        else:
            feature_groups.append(FeatureGroupResult(
                group=FeatureGroup.LIFECYCLE, features=[],
                rationale=self.RATIONALES[FeatureGroup.LIFECYCLE], enabled=False
            ))

        if self.config.compute_recency:
            recency_features, recency_group = self._compute_recency(
                prep, entity_col, time_col, ref_dates
            )
            all_features.append(recency_features)
            feature_groups.append(recency_group)
        else:
            feature_groups.append(FeatureGroupResult(
                group=FeatureGroup.RECENCY, features=[],
                rationale=self.RATIONALES[FeatureGroup.RECENCY], enabled=False
            ))

        if self.config.compute_regularity:
            regularity_features, regularity_group = self._compute_regularity(
                prep, entity_col, time_col, ref_dates
            )
            all_features.append(regularity_features)
            feature_groups.append(regularity_group)
        else:
            feature_groups.append(FeatureGroupResult(
                group=FeatureGroup.REGULARITY, features=[],
                rationale=self.RATIONALES[FeatureGroup.REGULARITY], enabled=False
            ))

        if self.config.compute_cohort:
            cohort_features, cohort_group = self._compute_cohort_comparison(
                lag_features, value_cols, entity_col
            )
            all_features.append(cohort_features)
            feature_groups.append(cohort_group)
        else:
            feature_groups.append(FeatureGroupResult(
                group=FeatureGroup.COHORT_COMPARISON, features=[],
                rationale=self.RATIONALES[FeatureGroup.COHORT_COMPARISON], enabled=False
            ))

        result_df = all_features[0]
        for df in all_features[1:]:
            result_df = result_df.merge(df, on=entity_col, how="left")

        return TemporalFeatureResult(
            features_df=result_df,
            feature_groups=feature_groups,
            config=self.config,
            entity_col=entity_col,
            value_cols=value_cols,
        )

    def _get_reference_dates(
        self,
        events_df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        reference_dates: Optional[pd.DataFrame],
        reference_col: Optional[str],
    ) -> pd.DataFrame:
        """Determine reference date for each entity."""
        entities = events_df[entity_col].unique()

        if self.config.reference_mode == ReferenceMode.GLOBAL_DATE:
            ref_date = self.config.global_reference_date or datetime.now()
            return native_pd.DataFrame({
                entity_col: entities,
                "reference_date": ref_date,
            })

        if reference_dates is not None and reference_col is not None:
            ref_df = reference_dates[[entity_col, reference_col]].copy()
            ref_df.columns = [entity_col, "reference_date"]
            ref_df["reference_date"] = to_datetime(ref_df["reference_date"])
            return ref_df

        # Default: Use last event date per entity
        ref_df = events_df.groupby(entity_col)[time_col].max().reset_index()
        ref_df.columns = [entity_col, "reference_date"]
        return ref_df

    def _compute_lagged_windows(
        self,
        events_df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        value_cols: List[str],
        ref_dates: pd.DataFrame,
    ) -> tuple:
        """Compute lagged window aggregations (Group 1).

        Uses one batched groupby per lag window across all value_cols and
        aggregation functions, instead of per-column-per-agg iteration.
        """
        if not value_cols:
            return ref_dates[[entity_col]].copy(), FeatureGroupResult(
                group=FeatureGroup.LAGGED_WINDOWS,
                features=[],
                rationale=self.RATIONALES[FeatureGroup.LAGGED_WINDOWS],
            )

        window_days = self.config.lag_window_days
        num_lags = self.config.num_lags

        df = events_df.merge(ref_dates, on=entity_col)
        df["days_before_ref"] = timedelta_to_days(df["reference_date"] - df[time_col])

        result = ref_dates[[entity_col]].copy()
        feature_names = []

        for lag in range(num_lags):
            start_days = lag * window_days
            end_days = (lag + 1) * window_days
            lag_mask = (df["days_before_ref"] >= start_days) & (df["days_before_ref"] < end_days)
            lag_df = df[lag_mask]

            prefix = f"lag{lag}_"
            lag_agg = groupby_multi_col_agg(
                lag_df, entity_col, value_cols,
                self.config.lag_aggregations, col_prefix=prefix,
            )
            for col in value_cols:
                for agg in self.config.lag_aggregations:
                    feature_names.append(f"{prefix}{col}_{agg}")

            result = result.merge(lag_agg, on=entity_col, how="left")
            for col in value_cols:
                count_col = f"{prefix}{col}_count"
                if count_col in result.columns:
                    result[count_col] = result[count_col].fillna(0).astype(int)

        return result, FeatureGroupResult(
            group=FeatureGroup.LAGGED_WINDOWS,
            features=feature_names,
            rationale=self.RATIONALES[FeatureGroup.LAGGED_WINDOWS],
        )

    def _compute_velocity(
        self,
        lag_features: pd.DataFrame,
        value_cols: List[str],
    ) -> tuple:
        """Compute velocity features (Group 2)."""
        entity_col = lag_features.columns[0]
        result = lag_features[[entity_col]].copy()
        feature_names = []
        window_days = self.config.lag_window_days

        for col in value_cols:
            lag0_col = f"lag0_{col}_sum"
            lag1_col = f"lag1_{col}_sum"

            if lag0_col in lag_features.columns and lag1_col in lag_features.columns:
                # Velocity = (Lag0 - Lag1) / window_days
                velocity_name = f"{col}_velocity"
                result[velocity_name] = (
                    lag_features[lag0_col] - lag_features[lag1_col]
                ) / window_days
                feature_names.append(velocity_name)

                velocity_pct_name = f"{col}_velocity_pct"
                lag1_safe = lag_features[lag1_col].replace(0, np.nan)
                result[velocity_pct_name] = (lag_features[lag0_col] - lag1_safe) / lag1_safe
                feature_names.append(velocity_pct_name)

        group_result = FeatureGroupResult(
            group=FeatureGroup.VELOCITY,
            features=feature_names,
            rationale=self.RATIONALES[FeatureGroup.VELOCITY],
        )

        return result, group_result

    def _compute_acceleration(
        self,
        velocity_features: pd.DataFrame,
        lag_features: pd.DataFrame,
        value_cols: List[str],
        entity_col: str,
    ) -> tuple:
        """Compute acceleration and momentum features (Group 3)."""
        result = lag_features[[entity_col]].copy()
        feature_names = []
        window_days = self.config.lag_window_days

        for col in value_cols:
            velocity_col = f"{col}_velocity"
            lag0_col = f"lag0_{col}_sum"
            lag1_col = f"lag1_{col}_sum"
            lag2_col = f"lag2_{col}_sum"

            # Acceleration = change in velocity
            if lag1_col in lag_features.columns and lag2_col in lag_features.columns:
                velocity_01 = (lag_features[lag0_col] - lag_features[lag1_col]) / window_days
                velocity_12 = (lag_features[lag1_col] - lag_features[lag2_col]) / window_days
                accel_name = f"{col}_acceleration"
                result[accel_name] = velocity_01 - velocity_12
                feature_names.append(accel_name)

            # Momentum = Lag0 × Velocity
            if velocity_col in velocity_features.columns and lag0_col in lag_features.columns:
                momentum_name = f"{col}_momentum"
                result[momentum_name] = lag_features[lag0_col] * velocity_features[velocity_col]
                feature_names.append(momentum_name)

        group_result = FeatureGroupResult(
            group=FeatureGroup.ACCELERATION,
            features=feature_names,
            rationale=self.RATIONALES[FeatureGroup.ACCELERATION],
        )

        return result, group_result

    def _compute_lifecycle(
        self,
        events_df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        value_cols: List[str],
        ref_dates: pd.DataFrame,
    ) -> tuple:
        """Compute lifecycle features (Group 4): Beginning/Middle/End.

        Vectorized: classifies each event into a lifecycle phase via numeric
        day-offsets, then aggregates per (entity, phase) in bulk.
        """
        result = ref_dates[[entity_col]].copy()
        feature_names = []
        min_days = self.config.min_history_days
        splits = self.config.lifecycle_splits

        history_stats = groupby_multi_agg(events_df, entity_col, time_col, ["min", "max"])
        history_stats.columns = [entity_col, "first_event", "last_event"]
        history_stats["history_days"] = timedelta_to_days(
            history_stats["last_event"] - history_stats["first_event"]
        )

        df = events_df.merge(history_stats, on=entity_col)
        df = df[df["history_days"] >= min_days]

        df["_days_since_first"] = timedelta_to_days(df[time_col] - df["first_event"])
        split1_days = df["history_days"] * splits[0]
        split2_days = df["history_days"] * (splits[0] + splits[1])

        beginning_df = df[df["_days_since_first"] < split1_days]
        middle_df = df[(df["_days_since_first"] >= split1_days) & (df["_days_since_first"] < split2_days)]
        end_df = df[df["_days_since_first"] >= split2_days]

        phase_frames = [
            (beginning_df, "beginning"),
            (middle_df, "middle"),
            (end_df, "end"),
        ]
        for phase_df, phase_name in phase_frames:
            if len(value_cols) == 0:
                continue
            phase_sums = phase_df.groupby(entity_col)[value_cols].sum().reset_index()
            rename = {c: f"{c}_{phase_name}" for c in value_cols}
            phase_sums = phase_sums.rename(columns=rename)
            result = result.merge(phase_sums, on=entity_col, how="left")

        for col in value_cols:
            for phase_name in ("beginning", "middle", "end"):
                feat = f"{col}_{phase_name}"
                if feat not in result.columns:
                    result[feat] = np.nan
            trend_col = f"{col}_trend_ratio"
            beg_safe = result[f"{col}_beginning"].replace(0, np.nan)
            result[trend_col] = result[f"{col}_end"] / beg_safe
            feature_names.extend([
                f"{col}_beginning", f"{col}_middle", f"{col}_end", trend_col
            ])

        return result, FeatureGroupResult(
            group=FeatureGroup.LIFECYCLE,
            features=feature_names,
            rationale=self.RATIONALES[FeatureGroup.LIFECYCLE],
        )

    def _compute_recency(
        self,
        events_df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        ref_dates: pd.DataFrame,
    ) -> tuple:
        """Compute recency and tenure features (Group 5)."""
        result = ref_dates[[entity_col]].copy()

        # Get first and last event per entity
        event_stats = groupby_multi_agg(events_df, entity_col, time_col, ["min", "max", "count"])
        event_stats.columns = [entity_col, "first_event", "last_event", "event_count"]

        result = result.merge(event_stats, on=entity_col, how="left")
        result = result.merge(ref_dates, on=entity_col)

        # Days since last event (from reference date)
        result["days_since_last_event"] = timedelta_to_days(
            result["reference_date"] - result["last_event"]
        )

        # Days since first event (tenure)
        result["days_since_first_event"] = timedelta_to_days(
            result["reference_date"] - result["first_event"]
        )

        # Active span (first to last event)
        result["active_span_days"] = timedelta_to_days(
            result["last_event"] - result["first_event"]
        )

        # Recency ratio: days_since_last / active_span (0 = just active, 1 = dormant).
        # Undefined for entities whose active_span_days == 0 (single event, or
        # multi-event coincident on a single day) — emit NaN instead of 0 so
        # downstream imputers distinguish degenerate history from genuinely-zero
        # recency. `active_span_days` already carries the flag signal for models.
        active_positive = result["active_span_days"] > 0
        ratio = result["days_since_last_event"] / (result["active_span_days"] + result["days_since_last_event"])
        result["recency_ratio"] = ratio.where(active_positive, float("nan")).clip(0, 1)

        # Clean up
        result = result.drop(columns=["first_event", "last_event", "event_count", "reference_date"])

        feature_names = [
            "days_since_last_event", "days_since_first_event",
            "active_span_days", "recency_ratio"
        ]

        group_result = FeatureGroupResult(
            group=FeatureGroup.RECENCY,
            features=feature_names,
            rationale=self.RATIONALES[FeatureGroup.RECENCY],
        )

        return result, group_result

    def _compute_regularity(
        self,
        events_df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        ref_dates: pd.DataFrame,
    ) -> tuple:
        """Compute frequency and regularity features (Group 6).

        Vectorized: computes inter-event gaps via groupby().diff() on epoch-day
        column, then aggregates gap stats with three groupby calls (mean, std, max).
        """
        result = ref_dates[[entity_col]].copy()

        sorted_df = events_df[[entity_col, time_col]].sort_values([entity_col, time_col])
        epoch_ref = sorted_df[time_col].min()
        sorted_df["_epoch_days"] = timedelta_to_days(sorted_df[time_col] - epoch_ref)
        sorted_df["_gap_days"] = sorted_df.groupby(entity_col)["_epoch_days"].diff()

        gaps_df = sorted_df.dropna(subset=["_gap_days"])

        gap_mean = gaps_df.groupby(entity_col)["_gap_days"].mean().reset_index()
        gap_mean.columns = [entity_col, "inter_event_gap_mean"]

        gap_std = gaps_df.groupby(entity_col)["_gap_days"].std().reset_index()
        gap_std.columns = [entity_col, "inter_event_gap_std"]

        gap_max = gaps_df.groupby(entity_col)["_gap_days"].max().reset_index()
        gap_max.columns = [entity_col, "inter_event_gap_max"]

        event_stats = groupby_multi_agg(events_df, entity_col, time_col, ["min", "max", "count"])
        event_stats.columns = [entity_col, "first_event", "last_event", "event_count"]
        total_days = timedelta_to_days(event_stats["last_event"] - event_stats["first_event"])
        safe_total = total_days.replace(0, np.nan)
        event_stats["event_frequency"] = event_stats["event_count"] / safe_total * 30
        zero_mask = total_days == 0
        event_stats["event_frequency"] = event_stats["event_frequency"].where(
            ~zero_mask, event_stats["event_count"].astype(float),
        )

        result = result.merge(gap_mean, on=entity_col, how="left")
        result = result.merge(gap_std, on=entity_col, how="left")
        result = result.merge(gap_max, on=entity_col, how="left")
        result = result.merge(event_stats[[entity_col, "event_frequency"]], on=entity_col, how="left")

        mean_safe = result["inter_event_gap_mean"].replace(0, np.nan)
        result["regularity_score"] = (1 - result["inter_event_gap_std"] / mean_safe).clip(0, 1)
        zero_gap = result["inter_event_gap_mean"] == 0
        result["regularity_score"] = result["regularity_score"].where(~zero_gap, 1.0)

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

    def _compute_cohort_comparison(
        self,
        lag_features: pd.DataFrame,
        value_cols: List[str],
        entity_col: str,
    ) -> tuple:
        """Compute cohort comparison features (Group 7)."""
        result = lag_features[[entity_col]].copy()
        feature_names = []

        for col in value_cols:
            lag0_col = f"lag0_{col}_sum"

            if lag0_col in lag_features.columns:
                cohort_mean = lag_features[lag0_col].mean()
                cohort_std = lag_features[lag0_col].std()

                # Difference from cohort mean
                vs_mean_name = f"{col}_vs_cohort_mean"
                result[vs_mean_name] = lag_features[lag0_col] - cohort_mean
                feature_names.append(vs_mean_name)

                # Percentage of cohort mean
                vs_pct_name = f"{col}_vs_cohort_pct"
                if cohort_mean != 0:
                    result[vs_pct_name] = lag_features[lag0_col] / cohort_mean
                else:
                    result[vs_pct_name] = np.nan
                feature_names.append(vs_pct_name)

                # Z-score (standard deviations from mean)
                if cohort_std > 0:
                    zscore_name = f"{col}_cohort_zscore"
                    result[zscore_name] = (lag_features[lag0_col] - cohort_mean) / cohort_std
                    feature_names.append(zscore_name)

        group_result = FeatureGroupResult(
            group=FeatureGroup.COHORT_COMPARISON,
            features=feature_names,
            rationale=self.RATIONALES[FeatureGroup.COHORT_COMPARISON],
        )

        return result, group_result
