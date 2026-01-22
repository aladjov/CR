"""
Temporal feature analyzer for discovering feature engineering opportunities.

Analyzes time series data to identify:
- Velocity (rate of change) patterns
- Acceleration (change in velocity) patterns
- Momentum (window ratios) patterns
- Lag correlations for feature selection
- Predictive power metrics (IV, KS)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from scipy import stats

from customer_retention.core.compat import pd, DataFrame


class FeatureType(str, Enum):
    """Types of temporal features."""
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    MOMENTUM = "momentum"
    LAG = "lag"
    ROLLING = "rolling"
    RATIO = "ratio"


@dataclass
class VelocityResult:
    """Result of velocity (rate of change) analysis."""
    column: str
    window_days: int
    mean_velocity: float
    std_velocity: float
    trend_direction: str  # "increasing", "decreasing", "stable"


@dataclass
class MomentumResult:
    """Result of momentum (window ratio) analysis."""
    column: str
    short_window: int
    long_window: int
    mean_momentum: float
    std_momentum: float
    interpretation: str  # "accelerating", "decelerating", "stable"


@dataclass
class LagCorrelationResult:
    """Result of lag correlation analysis."""
    column: str
    correlations: List[float]  # Correlation at each lag
    best_lag: int
    best_correlation: float
    has_weekly_pattern: bool


@dataclass
class PredictivePowerResult:
    """Result of predictive power analysis (IV and KS)."""
    column: str
    information_value: float
    iv_interpretation: str  # "weak", "medium", "strong", "suspicious"
    ks_statistic: float
    ks_pvalue: float
    ks_interpretation: str


@dataclass
class CohortComparison:
    """Comparison of metrics between cohorts."""
    velocity: float
    momentum: float
    mean_value: float


@dataclass
class FeatureRecommendation:
    """Recommended feature for engineering."""
    feature_name: str
    feature_type: FeatureType
    formula: str
    rationale: str
    priority: int  # 1=highest
    source_column: str


class TemporalFeatureAnalyzer:
    """Analyzes temporal patterns to inform feature engineering."""

    IV_THRESHOLDS = {"weak": 0.02, "medium": 0.1, "strong": 0.3, "suspicious": 0.5}
    KS_THRESHOLDS = {"weak": 0.2, "medium": 0.4}

    def __init__(self, time_column: str, entity_column: str):
        self.time_column = time_column
        self.entity_column = entity_column

    def calculate_velocity(
        self,
        df: DataFrame,
        value_columns: List[str],
        window_days: int = 7
    ) -> Dict[str, VelocityResult]:
        """Calculate velocity (rate of change) for specified columns."""
        df = self._prepare_dataframe(df)
        results = {}

        for col in value_columns:
            if col not in df.columns:
                continue

            daily = df.groupby(df[self.time_column].dt.date)[col].mean()
            velocity = daily.diff(window_days) / window_days

            mean_vel = velocity.mean()
            direction = (
                "increasing" if mean_vel > 0.01 else
                "decreasing" if mean_vel < -0.01 else
                "stable"
            )

            results[col] = VelocityResult(
                column=col,
                window_days=window_days,
                mean_velocity=float(mean_vel) if not np.isnan(mean_vel) else 0.0,
                std_velocity=float(velocity.std()) if not np.isnan(velocity.std()) else 0.0,
                trend_direction=direction
            )

        return results

    def calculate_acceleration(
        self,
        df: DataFrame,
        value_columns: List[str],
        window_days: int = 7
    ) -> Dict[str, float]:
        """Calculate acceleration (change in velocity)."""
        df = self._prepare_dataframe(df)
        results = {}

        for col in value_columns:
            if col not in df.columns:
                continue

            daily = df.groupby(df[self.time_column].dt.date)[col].mean()
            velocity = daily.diff(window_days)
            acceleration = velocity.diff(window_days)

            results[col] = float(acceleration.mean()) if not np.isnan(acceleration.mean()) else 0.0

        return results

    def calculate_momentum(
        self,
        df: DataFrame,
        value_columns: List[str],
        short_window: int = 7,
        long_window: int = 30
    ) -> Dict[str, MomentumResult]:
        """Calculate momentum (ratio of short to long window averages)."""
        df = self._prepare_dataframe(df)
        reference_date = df[self.time_column].max()
        results = {}

        for col in value_columns:
            if col not in df.columns:
                continue

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

            interpretation = (
                "accelerating" if mean_mom > 1.1 else
                "decelerating" if mean_mom < 0.9 else
                "stable"
            )

            results[col] = MomentumResult(
                column=col,
                short_window=short_window,
                long_window=long_window,
                mean_momentum=float(mean_mom),
                std_momentum=float(std_mom),
                interpretation=interpretation
            )

        return results

    def calculate_lag_correlations(
        self,
        df: DataFrame,
        value_columns: List[str],
        max_lag: int = 14
    ) -> Dict[str, LagCorrelationResult]:
        """Calculate autocorrelation at different lags."""
        df = self._prepare_dataframe(df)
        results = {}

        for col in value_columns:
            if col not in df.columns:
                continue

            daily = df.groupby(df[self.time_column].dt.date)[col].mean()
            correlations = []

            for lag in range(1, max_lag + 1):
                if len(daily) > lag:
                    corr = daily.autocorr(lag=lag)
                    correlations.append(float(corr) if not np.isnan(corr) else 0.0)
                else:
                    correlations.append(0.0)

            best_idx = int(np.argmax(np.abs(correlations)))
            lag_7_corr = correlations[6] if len(correlations) >= 7 else 0

            results[col] = LagCorrelationResult(
                column=col,
                correlations=correlations,
                best_lag=best_idx + 1,
                best_correlation=correlations[best_idx] if correlations else 0.0,
                has_weekly_pattern=abs(lag_7_corr) > 0.2
            )

        return results

    def calculate_predictive_power(
        self,
        df: DataFrame,
        value_columns: List[str],
        target_column: str
    ) -> Dict[str, PredictivePowerResult]:
        """Calculate Information Value and KS statistic."""
        df = self._prepare_dataframe(df)

        # Aggregate to entity level
        entity_features = df.groupby(self.entity_column)[value_columns].mean()
        entity_target = df.groupby(self.entity_column)[target_column].first()
        entity_data = entity_features.join(entity_target)

        results = {}

        for col in value_columns:
            if col not in entity_data.columns:
                continue

            feature = entity_data[col]
            target = entity_data[target_column]

            iv = self._calculate_iv(feature, target)
            ks_stat, ks_pval = self._calculate_ks(feature, target)

            iv_interp = self._interpret_iv(iv)
            ks_interp = self._interpret_ks(ks_stat)

            results[col] = PredictivePowerResult(
                column=col,
                information_value=iv,
                iv_interpretation=iv_interp,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                ks_interpretation=ks_interp
            )

        return results

    def compare_cohorts(
        self,
        df: DataFrame,
        value_columns: List[str],
        target_column: str
    ) -> Dict[str, Dict[str, CohortComparison]]:
        """Compare metrics between retained and churned cohorts."""
        df = self._prepare_dataframe(df)
        results = {}

        entity_target = df.groupby(self.entity_column)[target_column].first()
        df = df.merge(
            entity_target.reset_index().rename(columns={target_column: "_target"}),
            on=self.entity_column
        )

        velocity_results = self.calculate_velocity(df, value_columns)
        momentum_results = self.calculate_momentum(df, value_columns)

        for col in value_columns:
            if col not in df.columns:
                continue

            retained_df = df[df["_target"] == 1]
            churned_df = df[df["_target"] == 0]

            retained_vel = self.calculate_velocity(retained_df, [col])
            churned_vel = self.calculate_velocity(churned_df, [col])

            retained_mom = self.calculate_momentum(retained_df, [col])
            churned_mom = self.calculate_momentum(churned_df, [col])

            results[col] = {
                "retained": CohortComparison(
                    velocity=retained_vel[col].mean_velocity if col in retained_vel else 0,
                    momentum=retained_mom[col].mean_momentum if col in retained_mom else 1,
                    mean_value=float(retained_df[col].mean())
                ),
                "churned": CohortComparison(
                    velocity=churned_vel[col].mean_velocity if col in churned_vel else 0,
                    momentum=churned_mom[col].mean_momentum if col in churned_mom else 1,
                    mean_value=float(churned_df[col].mean())
                )
            }

        return results

    def get_feature_recommendations(
        self,
        df: DataFrame,
        value_columns: List[str],
        target_column: Optional[str] = None
    ) -> List[FeatureRecommendation]:
        """Generate feature engineering recommendations based on analysis."""
        recommendations = []
        priority = 1

        # Analyze predictive power if target available
        if target_column:
            power_results = self.calculate_predictive_power(df, value_columns, target_column)
            for col, result in sorted(
                power_results.items(),
                key=lambda x: x[1].information_value,
                reverse=True
            ):
                if result.information_value > self.IV_THRESHOLDS["weak"]:
                    recommendations.append(FeatureRecommendation(
                        feature_name=f"{col}_mean",
                        feature_type=FeatureType.ROLLING,
                        formula=f"df.groupby(entity)['{col}'].transform('mean')",
                        rationale=f"IV={result.information_value:.3f} ({result.iv_interpretation})",
                        priority=priority,
                        source_column=col
                    ))
                    priority += 1

        # Analyze velocity
        velocity_results = self.calculate_velocity(df, value_columns)
        for col, result in velocity_results.items():
            if result.trend_direction != "stable":
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_velocity_7d",
                    feature_type=FeatureType.VELOCITY,
                    formula=f"(current - lag_7d) / lag_7d",
                    rationale=f"Detected {result.trend_direction} trend",
                    priority=priority,
                    source_column=col
                ))
                priority += 1

        # Analyze momentum
        momentum_results = self.calculate_momentum(df, value_columns)
        for col, result in momentum_results.items():
            if result.interpretation != "stable":
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_momentum_7_30",
                    feature_type=FeatureType.MOMENTUM,
                    formula=f"mean_7d / mean_30d",
                    rationale=f"Momentum indicates {result.interpretation} behavior",
                    priority=priority,
                    source_column=col
                ))
                priority += 1

        # Analyze lag correlations
        lag_results = self.calculate_lag_correlations(df, value_columns)
        for col, result in lag_results.items():
            if result.best_correlation > 0.3:
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_lag_{result.best_lag}d",
                    feature_type=FeatureType.LAG,
                    formula=f"df['{col}'].shift({result.best_lag})",
                    rationale=f"Strong autocorrelation (r={result.best_correlation:.2f}) at lag {result.best_lag}",
                    priority=priority,
                    source_column=col
                ))
                priority += 1

            if result.has_weekly_pattern:
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_weekly_pattern",
                    feature_type=FeatureType.LAG,
                    formula=f"df['{col}'].shift(7)",
                    rationale="Weekly seasonality detected",
                    priority=priority,
                    source_column=col
                ))
                priority += 1

        return recommendations

    def _prepare_dataframe(self, df: DataFrame) -> DataFrame:
        """Ensure time column is datetime."""
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        return df

    def _calculate_iv(self, feature: pd.Series, target: pd.Series, bins: int = 10) -> float:
        """Calculate Information Value."""
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

        total_events = grouped["events"].sum()
        total_non_events = grouped["non_events"].sum()

        if total_events == 0 or total_non_events == 0:
            return 0.0

        grouped["pct_events"] = grouped["events"] / total_events
        grouped["pct_non_events"] = grouped["non_events"] / total_non_events

        # Avoid log(0)
        grouped["pct_events"] = grouped["pct_events"].replace(0, 0.0001)
        grouped["pct_non_events"] = grouped["pct_non_events"].replace(0, 0.0001)

        grouped["woe"] = np.log(grouped["pct_events"] / grouped["pct_non_events"])
        grouped["iv"] = (grouped["pct_events"] - grouped["pct_non_events"]) * grouped["woe"]

        return float(grouped["iv"].sum())

    def _calculate_ks(self, feature: pd.Series, target: pd.Series) -> Tuple[float, float]:
        """Calculate KS statistic between target classes."""
        df_ks = pd.DataFrame({"feature": feature, "target": target}).dropna()

        group0 = df_ks[df_ks["target"] == 0]["feature"]
        group1 = df_ks[df_ks["target"] == 1]["feature"]

        if len(group0) == 0 or len(group1) == 0:
            return 0.0, 1.0

        ks_stat, p_val = stats.ks_2samp(group0, group1)
        return float(ks_stat), float(p_val)

    def _interpret_iv(self, iv: float) -> str:
        """Interpret Information Value."""
        if iv > self.IV_THRESHOLDS["suspicious"]:
            return "suspicious"
        elif iv > self.IV_THRESHOLDS["strong"]:
            return "strong"
        elif iv > self.IV_THRESHOLDS["medium"]:
            return "medium"
        elif iv > self.IV_THRESHOLDS["weak"]:
            return "weak"
        return "very_weak"

    def _interpret_ks(self, ks: float) -> str:
        """Interpret KS statistic."""
        if ks > self.KS_THRESHOLDS["medium"]:
            return "strong"
        elif ks > self.KS_THRESHOLDS["weak"]:
            return "medium"
        return "weak"
