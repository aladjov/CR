"""Temporal feature analyzer for discovering feature engineering opportunities."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from customer_retention.core.compat import DataFrame, pd


class FeatureType(str, Enum):
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    MOMENTUM = "momentum"
    LAG = "lag"
    ROLLING = "rolling"
    RATIO = "ratio"


@dataclass
class VelocityResult:
    column: str
    window_days: int
    mean_velocity: float
    std_velocity: float
    trend_direction: str


@dataclass
class MomentumResult:
    column: str
    short_window: int
    long_window: int
    mean_momentum: float
    std_momentum: float
    interpretation: str


@dataclass
class LagCorrelationResult:
    column: str
    correlations: List[float]
    best_lag: int
    best_correlation: float
    has_weekly_pattern: bool


@dataclass
class PredictivePowerResult:
    column: str
    information_value: float
    iv_interpretation: str
    ks_statistic: float
    ks_pvalue: float
    ks_interpretation: str


@dataclass
class CohortComparison:
    velocity: float
    momentum: float
    mean_value: float


@dataclass
class FeatureRecommendation:
    feature_name: str
    feature_type: FeatureType
    formula: str
    rationale: str
    priority: int
    source_column: str


class TemporalFeatureAnalyzer:
    """Analyzes temporal patterns to inform feature engineering."""

    IV_THRESHOLDS = {"weak": 0.02, "medium": 0.1, "strong": 0.3, "suspicious": 0.5}
    KS_THRESHOLDS = {"weak": 0.2, "medium": 0.4}

    def __init__(self, time_column: str, entity_column: str):
        self.time_column = time_column
        self.entity_column = entity_column

    def calculate_velocity(
        self, df: DataFrame, value_columns: List[str], window_days: int = 7
    ) -> Dict[str, VelocityResult]:
        df = self._prepare_dataframe(df)
        return {col: self._velocity_for_column(df, col, window_days)
                for col in value_columns if col in df.columns}

    def _velocity_for_column(self, df: DataFrame, col: str, window_days: int) -> VelocityResult:
        daily = df.groupby(df[self.time_column].dt.date)[col].mean()
        velocity = daily.diff(window_days) / window_days
        mean_vel = velocity.mean()
        return VelocityResult(
            column=col,
            window_days=window_days,
            mean_velocity=float(mean_vel) if not np.isnan(mean_vel) else 0.0,
            std_velocity=float(velocity.std()) if not np.isnan(velocity.std()) else 0.0,
            trend_direction=self._classify_trend(mean_vel),
        )

    def _classify_trend(self, mean_velocity: float) -> str:
        if mean_velocity > 0.01:
            return "increasing"
        return "decreasing" if mean_velocity < -0.01 else "stable"

    def calculate_acceleration(self, df: DataFrame, value_columns: List[str], window_days: int = 7) -> Dict[str, float]:
        df = self._prepare_dataframe(df)
        return {col: self._acceleration_for_column(df, col, window_days)
                for col in value_columns if col in df.columns}

    def _acceleration_for_column(self, df: DataFrame, col: str, window_days: int) -> float:
        daily = df.groupby(df[self.time_column].dt.date)[col].mean()
        acceleration = daily.diff(window_days).diff(window_days)
        return float(acceleration.mean()) if not np.isnan(acceleration.mean()) else 0.0

    def calculate_momentum(
        self, df: DataFrame, value_columns: List[str], short_window: int = 7, long_window: int = 30
    ) -> Dict[str, MomentumResult]:
        df = self._prepare_dataframe(df)
        reference_date = df[self.time_column].max()
        return {col: self._momentum_for_column(df, col, short_window, long_window, reference_date)
                for col in value_columns if col in df.columns}

    def _momentum_for_column(
        self, df: DataFrame, col: str, short_window: int, long_window: int, reference_date
    ) -> MomentumResult:
        entity_momentum = []
        for entity_id in df[self.entity_column].unique():
            entity_data = df[df[self.entity_column] == entity_id].copy()
            entity_data["days_ago"] = (reference_date - entity_data[self.time_column]).dt.days
            short_mean = entity_data[entity_data["days_ago"] <= short_window][col].mean()
            long_mean = entity_data[entity_data["days_ago"] <= long_window][col].mean()
            if long_mean > 0 and not np.isnan(short_mean):
                entity_momentum.append(short_mean / long_mean)

        mean_mom = np.mean(entity_momentum) if entity_momentum else 1.0
        std_mom = np.std(entity_momentum) if entity_momentum else 0.0
        return MomentumResult(
            column=col, short_window=short_window, long_window=long_window,
            mean_momentum=float(mean_mom), std_momentum=float(std_mom),
            interpretation=self._classify_momentum(mean_mom),
        )

    def _classify_momentum(self, mean_momentum: float) -> str:
        if mean_momentum > 1.1:
            return "accelerating"
        return "decelerating" if mean_momentum < 0.9 else "stable"

    def calculate_lag_correlations(
        self, df: DataFrame, value_columns: List[str], max_lag: int = 14
    ) -> Dict[str, LagCorrelationResult]:
        df = self._prepare_dataframe(df)
        return {col: self._lag_correlation_for_column(df, col, max_lag)
                for col in value_columns if col in df.columns}

    def _lag_correlation_for_column(self, df: DataFrame, col: str, max_lag: int) -> LagCorrelationResult:
        daily = df.groupby(df[self.time_column].dt.date)[col].mean()
        correlations = [
            float(daily.autocorr(lag=lag)) if len(daily) > lag and not np.isnan(daily.autocorr(lag=lag)) else 0.0
            for lag in range(1, max_lag + 1)
        ]
        best_idx = int(np.argmax(np.abs(correlations)))
        return LagCorrelationResult(
            column=col, correlations=correlations, best_lag=best_idx + 1,
            best_correlation=correlations[best_idx] if correlations else 0.0,
            has_weekly_pattern=abs(correlations[6] if len(correlations) >= 7 else 0) > 0.2,
        )

    def _validate_target_constant_per_entity(self, df: DataFrame, target_column: str) -> None:
        import warnings
        varying_entities = (df.groupby(self.entity_column)[target_column].nunique() > 1).sum()
        if varying_entities > 0:
            warnings.warn(
                f"Target '{target_column}' varies within {varying_entities} entities. "
                f"Using first value per entity. Target should be constant for retention modeling.",
                UserWarning, stacklevel=3,
            )

    def calculate_predictive_power(
        self, df: DataFrame, value_columns: List[str], target_column: str
    ) -> Dict[str, PredictivePowerResult]:
        df = self._prepare_dataframe(df)
        self._validate_target_constant_per_entity(df, target_column)
        entity_data = self._aggregate_to_entity_level(df, value_columns, target_column)
        return {col: self._predictive_power_for_column(entity_data, col, target_column)
                for col in value_columns if col in entity_data.columns}

    def _aggregate_to_entity_level(self, df: DataFrame, value_columns: List[str], target_column: str) -> DataFrame:
        entity_features = df.groupby(self.entity_column)[value_columns].mean()
        entity_target = df.groupby(self.entity_column)[target_column].first()
        return entity_features.join(entity_target)

    def _predictive_power_for_column(self, entity_data: DataFrame, col: str, target_column: str) -> PredictivePowerResult:
        feature, target = entity_data[col], entity_data[target_column]
        iv = self._calculate_iv(feature, target)
        ks_stat, ks_pval = self._calculate_ks(feature, target)
        return PredictivePowerResult(
            column=col, information_value=iv, iv_interpretation=self._interpret_iv(iv),
            ks_statistic=ks_stat, ks_pvalue=ks_pval, ks_interpretation=self._interpret_ks(ks_stat),
        )

    def compare_cohorts(
        self, df: DataFrame, value_columns: List[str], target_column: str
    ) -> Dict[str, Dict[str, CohortComparison]]:
        """Compare metrics between retained and churned cohorts."""
        df = self._prepare_dataframe(df)
        self._validate_event_level_target_usage(df, target_column)
        self._validate_target_constant_per_entity(df, target_column)

        value_columns = [c for c in value_columns if c != target_column]
        df = self._add_entity_target_column(df, target_column)

        return {col: self._compare_cohorts_for_column(df, col)
                for col in value_columns if col in df.columns}

    def _add_entity_target_column(self, df: DataFrame, target_column: str) -> DataFrame:
        entity_target = df.groupby(self.entity_column)[target_column].first()
        return df.merge(entity_target.reset_index().rename(columns={target_column: "_target"}), on=self.entity_column)

    def _compare_cohorts_for_column(self, df: DataFrame, col: str) -> Dict[str, CohortComparison]:
        retained_df, churned_df = df[df["_target"] == 1], df[df["_target"] == 0]
        return {
            "retained": self._cohort_comparison(retained_df, col),
            "churned": self._cohort_comparison(churned_df, col),
        }

    def _cohort_comparison(self, cohort_df: DataFrame, col: str) -> CohortComparison:
        vel = self.calculate_velocity(cohort_df, [col])
        mom = self.calculate_momentum(cohort_df, [col])
        return CohortComparison(
            velocity=vel[col].mean_velocity if col in vel else 0,
            momentum=mom[col].mean_momentum if col in mom else 1,
            mean_value=float(cohort_df[col].mean()),
        )

    def get_feature_recommendations(
        self, df: DataFrame, value_columns: List[str], target_column: Optional[str] = None
    ) -> List[FeatureRecommendation]:
        recommendations: List[FeatureRecommendation] = []
        priority = [1]

        if target_column:
            self._add_predictive_power_recommendations(df, value_columns, target_column, recommendations, priority)
        self._add_velocity_recommendations(df, value_columns, recommendations, priority)
        self._add_momentum_recommendations(df, value_columns, recommendations, priority)
        self._add_lag_recommendations(df, value_columns, recommendations, priority)
        return recommendations

    def _add_predictive_power_recommendations(
        self, df: DataFrame, value_columns: List[str], target_column: str,
        recommendations: List[FeatureRecommendation], priority: List[int]
    ) -> None:
        power_results = self.calculate_predictive_power(df, value_columns, target_column)
        for col, result in sorted(power_results.items(), key=lambda x: x[1].information_value, reverse=True):
            if result.information_value > self.IV_THRESHOLDS["weak"]:
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_mean", feature_type=FeatureType.ROLLING,
                    formula=f"df.groupby(entity)['{col}'].transform('mean')",
                    rationale=f"IV={result.information_value:.3f} ({result.iv_interpretation})",
                    priority=priority[0], source_column=col,
                ))
                priority[0] += 1

    def _add_velocity_recommendations(
        self, df: DataFrame, value_columns: List[str],
        recommendations: List[FeatureRecommendation], priority: List[int]
    ) -> None:
        for col, result in self.calculate_velocity(df, value_columns).items():
            if result.trend_direction != "stable":
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_velocity_7d", feature_type=FeatureType.VELOCITY,
                    formula="(current - lag_7d) / lag_7d",
                    rationale=f"Detected {result.trend_direction} trend",
                    priority=priority[0], source_column=col,
                ))
                priority[0] += 1

    def _add_momentum_recommendations(
        self, df: DataFrame, value_columns: List[str],
        recommendations: List[FeatureRecommendation], priority: List[int]
    ) -> None:
        for col, result in self.calculate_momentum(df, value_columns).items():
            if result.interpretation != "stable":
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_momentum_7_30", feature_type=FeatureType.MOMENTUM,
                    formula="mean_7d / mean_30d",
                    rationale=f"Momentum indicates {result.interpretation} behavior",
                    priority=priority[0], source_column=col,
                ))
                priority[0] += 1

    def _add_lag_recommendations(
        self, df: DataFrame, value_columns: List[str],
        recommendations: List[FeatureRecommendation], priority: List[int]
    ) -> None:
        for col, result in self.calculate_lag_correlations(df, value_columns).items():
            if result.best_correlation > 0.3:
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_lag_{result.best_lag}d", feature_type=FeatureType.LAG,
                    formula=f"df['{col}'].shift({result.best_lag})",
                    rationale=f"Strong autocorrelation (r={result.best_correlation:.2f}) at lag {result.best_lag}",
                    priority=priority[0], source_column=col,
                ))
                priority[0] += 1
            if result.has_weekly_pattern:
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_weekly_pattern", feature_type=FeatureType.LAG,
                    formula=f"df['{col}'].shift(7)", rationale="Weekly seasonality detected",
                    priority=priority[0], source_column=col,
                ))
                priority[0] += 1

    def _prepare_dataframe(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        return df

    def _validate_event_level_target_usage(self, df: DataFrame, target_column: Optional[str]) -> None:
        if target_column is None:
            return
        n_entities, n_rows = df[self.entity_column].nunique(), len(df)
        if n_entities < n_rows:
            raise ValueError(
                f"Target comparisons not allowed on event-level data. "
                f"Found {n_rows:,} rows but only {n_entities:,} entities. "
                f"Aggregate to entity level first using TimeWindowAggregator."
            )

    def _calculate_iv(self, feature: pd.Series, target: pd.Series, bins: int = 10) -> float:
        df_iv = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if len(df_iv) < bins * 2:
            return 0.0
        try:
            df_iv["bin"] = pd.qcut(df_iv["feature"], q=bins, duplicates="drop")
        except ValueError:
            return 0.0

        grouped = df_iv.groupby("bin", observed=False)["target"].agg(["sum", "count"])
        grouped["non_events"] = grouped["count"] - grouped["sum"]
        grouped["events"] = grouped["sum"]
        total_events, total_non_events = grouped["events"].sum(), grouped["non_events"].sum()
        if total_events == 0 or total_non_events == 0:
            return 0.0

        grouped["pct_events"] = grouped["events"] / total_events
        grouped["pct_non_events"] = grouped["non_events"] / total_non_events
        grouped["pct_events"] = grouped["pct_events"].replace(0, 0.0001)
        grouped["pct_non_events"] = grouped["pct_non_events"].replace(0, 0.0001)
        grouped["woe"] = np.log(grouped["pct_events"] / grouped["pct_non_events"])
        grouped["iv"] = (grouped["pct_events"] - grouped["pct_non_events"]) * grouped["woe"]
        return float(grouped["iv"].sum())

    def _calculate_ks(self, feature: pd.Series, target: pd.Series) -> Tuple[float, float]:
        df_ks = pd.DataFrame({"feature": feature, "target": target}).dropna()
        group0, group1 = df_ks[df_ks["target"] == 0]["feature"], df_ks[df_ks["target"] == 1]["feature"]
        if len(group0) == 0 or len(group1) == 0:
            return 0.0, 1.0
        ks_stat, p_val = stats.ks_2samp(group0, group1)
        return float(ks_stat), float(p_val)

    def _interpret_iv(self, iv: float) -> str:
        if iv > self.IV_THRESHOLDS["suspicious"]:
            return "suspicious"
        if iv > self.IV_THRESHOLDS["strong"]:
            return "strong"
        if iv > self.IV_THRESHOLDS["medium"]:
            return "medium"
        if iv > self.IV_THRESHOLDS["weak"]:
            return "weak"
        return "very_weak"

    def _interpret_ks(self, ks: float) -> str:
        if ks > self.KS_THRESHOLDS["medium"]:
            return "strong"
        if ks > self.KS_THRESHOLDS["weak"]:
            return "medium"
        return "weak"
