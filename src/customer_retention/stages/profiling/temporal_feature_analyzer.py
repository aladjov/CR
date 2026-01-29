from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from customer_retention.core.compat import DataFrame, pd
from customer_retention.core.utils import compute_effect_size


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


@dataclass
class CohortVelocityResult:
    column: str
    window_days: int
    retained_velocity: List[float]
    churned_velocity: List[float]
    overall_velocity: List[float]
    retained_accel: List[float]
    churned_accel: List[float]
    overall_accel: List[float]
    velocity_effect_size: float
    velocity_effect_interp: str
    accel_effect_size: float
    accel_effect_interp: str
    period_label: str


@dataclass
class VelocityRecommendation:
    source_column: str
    action: str
    description: str
    params: Dict[str, Any]
    effect_size: float
    priority: int


@dataclass
class CohortMomentumResult:
    column: str
    short_window: int
    long_window: int
    retained_momentum: float
    churned_momentum: float
    overall_momentum: float
    effect_size: float
    effect_interp: str
    window_label: str


class TemporalFeatureAnalyzer:
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

    def _daily_diff_series(self, df: DataFrame, col: str, window_days: int):
        daily = df.groupby(df[self.time_column].dt.date)[col].mean()
        return daily.diff(window_days)

    def _velocity_for_column(self, df: DataFrame, col: str, window_days: int) -> VelocityResult:
        velocity = self._daily_diff_series(df, col, window_days) / window_days
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
        acceleration = self._daily_diff_series(df, col, window_days).diff(window_days)
        return float(acceleration.mean()) if not np.isnan(acceleration.mean()) else 0.0

    def compute_cohort_velocity_signals(
        self, df: DataFrame, value_columns: List[str], target_column: str,
        windows: Optional[List[int]] = None
    ) -> Dict[str, List[CohortVelocityResult]]:
        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}' not found in DataFrame")
        windows = windows or [7, 14, 30, 90, 180, 365]
        df = self._prepare_dataframe(df)
        retained_df = df[df[target_column] == 1]
        churned_df = df[df[target_column] == 0]
        results = {}
        for col in value_columns:
            if col not in df.columns:
                continue
            col_results = []
            for window in windows:
                result = self._cohort_velocity_for_window(
                    retained_df, churned_df, col, window, df
                )
                col_results.append(result)
            results[col] = col_results
        return results

    def _cohort_velocity_for_window(
        self, retained_df: DataFrame, churned_df: DataFrame, col: str, window: int,
        overall_df: DataFrame
    ) -> CohortVelocityResult:
        ret_vel, ret_accel = self._velocity_accel_series(retained_df, col, window)
        churn_vel, churn_accel = self._velocity_accel_series(churned_df, col, window)
        overall_vel, overall_accel = self._velocity_accel_series(overall_df, col, window)
        vel_d, vel_interp = compute_effect_size(ret_vel, churn_vel)
        accel_d, accel_interp = compute_effect_size(ret_accel, churn_accel)
        period_label = self._window_to_period_label(window)
        return CohortVelocityResult(
            column=col, window_days=window,
            retained_velocity=ret_vel, churned_velocity=churn_vel, overall_velocity=overall_vel,
            retained_accel=ret_accel, churned_accel=churn_accel, overall_accel=overall_accel,
            velocity_effect_size=vel_d, velocity_effect_interp=vel_interp,
            accel_effect_size=accel_d, accel_effect_interp=accel_interp,
            period_label=period_label
        )

    _WINDOW_MAPPING = [
        (7, "W", "Weekly"),
        (14, "2W", "Bi-weekly"),
        (30, "M", "Monthly"),
        (90, "Q", "Quarterly"),
        (180, "2Q", "Semi-annual"),
    ]

    def _get_window_info(self, window_days: int) -> tuple:
        for threshold, period_code, label in self._WINDOW_MAPPING:
            if window_days <= threshold:
                return period_code, label
        return "Y", "Yearly"

    def _window_to_period_label(self, window_days: int) -> str:
        return self._get_window_info(window_days)[1]

    def _window_to_period(self, window_days: int) -> str:
        return self._get_window_info(window_days)[0]

    def _velocity_accel_series(self, df: DataFrame, col: str, window: int) -> Tuple[List[float], List[float]]:
        if df.empty or col not in df.columns:
            return [], []
        period_code = self._window_to_period(window)
        period_col = df[self.time_column].dt.to_period(period_code).dt.start_time
        period_means = df.groupby(period_col)[col].mean()
        velocity = period_means.diff().dropna()
        accel = velocity.diff().dropna()
        return velocity.tolist(), accel.tolist()

    def generate_velocity_recommendations(
        self, results: Dict[str, List[CohortVelocityResult]]
    ) -> List[VelocityRecommendation]:
        recommendations = []
        for col, col_results in results.items():
            best = self._find_best_velocity_window(col_results)
            if best and abs(best.velocity_effect_size) >= 0.5:
                recommendations.append(VelocityRecommendation(
                    source_column=col, action="add_velocity_feature",
                    description=f"Add {best.period_label} velocity for {col} (d={best.velocity_effect_size:.2f})",
                    params={"window_days": best.window_days, "period": best.period_label},
                    effect_size=best.velocity_effect_size, priority=1 if abs(best.velocity_effect_size) >= 0.8 else 2
                ))
            if best and abs(best.accel_effect_size) >= 0.5:
                recommendations.append(VelocityRecommendation(
                    source_column=col, action="add_acceleration_feature",
                    description=f"Add {best.period_label} acceleration for {col} (d={best.accel_effect_size:.2f})",
                    params={"window_days": best.window_days, "period": best.period_label},
                    effect_size=best.accel_effect_size, priority=2
                ))
        return sorted(recommendations, key=lambda r: (-abs(r.effect_size), r.priority))

    def _find_best_velocity_window(
        self, results: List[CohortVelocityResult]
    ) -> Optional[CohortVelocityResult]:
        if not results:
            return None
        return max(results, key=lambda r: abs(r.velocity_effect_size))

    def generate_velocity_interpretation(
        self, results: Dict[str, List[CohortVelocityResult]]
    ) -> List[str]:
        notes = []
        for col, col_results in results.items():
            best = self._find_best_velocity_window(col_results)
            if not best:
                continue
            d = best.velocity_effect_size
            if abs(d) >= 0.8:
                direction = "increasing" if d > 0 else "decreasing"
                notes.append(f"• {col}: Strong signal at {best.period_label} - retained customers show "
                           f"{direction} velocity vs churned (d={d:.2f})")
            elif abs(d) >= 0.5:
                notes.append(f"• {col}: Moderate signal at {best.period_label} (d={d:.2f}) - "
                           f"consider as secondary predictor")
            elif abs(d) >= 0.2:
                notes.append(f"• {col}: Weak signal at {best.period_label} (d={d:.2f}) - "
                           f"may contribute in feature combinations")
            else:
                notes.append(f"• {col}: No significant velocity difference between cohorts")
        return notes

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

    def compute_cohort_momentum_signals(
        self, df: DataFrame, value_columns: List[str], target_column: str,
        window_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, List[CohortMomentumResult]]:
        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}' not found in DataFrame")
        window_pairs = window_pairs or [(7, 30), (30, 90), (7, 90)]
        df = self._prepare_dataframe(df)
        retained_df = df[df[target_column] == 1]
        churned_df = df[df[target_column] == 0]
        results = {}
        for col in value_columns:
            if col not in df.columns:
                continue
            col_results = []
            for short_w, long_w in window_pairs:
                result = self._cohort_momentum_for_pair(
                    retained_df, churned_df, df, col, short_w, long_w
                )
                col_results.append(result)
            results[col] = col_results
        return results

    def _cohort_momentum_for_pair(
        self, retained_df: DataFrame, churned_df: DataFrame, overall_df: DataFrame,
        col: str, short_w: int, long_w: int
    ) -> CohortMomentumResult:
        ret_values = self._vectorized_entity_momentum(retained_df, col, short_w, long_w)
        churn_values = self._vectorized_entity_momentum(churned_df, col, short_w, long_w)
        overall_values = self._vectorized_entity_momentum(overall_df, col, short_w, long_w)
        ret_mom = float(np.mean(ret_values)) if ret_values else 1.0
        churn_mom = float(np.mean(churn_values)) if churn_values else 1.0
        overall_mom = float(np.mean(overall_values)) if overall_values else 1.0
        d, interp = compute_effect_size(ret_values, churn_values)
        return CohortMomentumResult(
            column=col, short_window=short_w, long_window=long_w,
            retained_momentum=ret_mom, churned_momentum=churn_mom, overall_momentum=overall_mom,
            effect_size=d, effect_interp=interp, window_label=f"{short_w}d/{long_w}d"
        )

    def _vectorized_entity_momentum(
        self, df: DataFrame, col: str, short_w: int, long_w: int
    ) -> List[float]:
        if df.empty or col not in df.columns:
            return []
        reference_date = df[self.time_column].max()
        df_calc = df[[self.entity_column, self.time_column, col]].copy()
        df_calc["_days_ago"] = (reference_date - df_calc[self.time_column]).dt.days
        short_means = df_calc[df_calc["_days_ago"] <= short_w].groupby(self.entity_column)[col].mean()
        long_means = df_calc[df_calc["_days_ago"] <= long_w].groupby(self.entity_column)[col].mean()
        valid = (long_means > 0) & short_means.notna() & long_means.notna()
        momentum = (short_means[valid] / long_means[valid]).dropna()
        return momentum.tolist()

    def generate_momentum_interpretation(
        self, results: Dict[str, List[CohortMomentumResult]]
    ) -> List[str]:
        notes = []
        for col, col_results in results.items():
            best = max(col_results, key=lambda r: abs(r.effect_size)) if col_results else None
            if not best:
                continue
            d = best.effect_size
            ret_trend = "accelerating" if best.retained_momentum > 1.05 else "decelerating" if best.retained_momentum < 0.95 else "stable"
            churn_trend = "accelerating" if best.churned_momentum > 1.05 else "decelerating" if best.churned_momentum < 0.95 else "stable"
            if abs(d) >= 0.5:
                notes.append(f"• {col}: Strong signal at {best.window_label} - "
                           f"retained {ret_trend} ({best.retained_momentum:.2f}), "
                           f"churned {churn_trend} ({best.churned_momentum:.2f}), d={d:.2f}")
            elif abs(d) >= 0.2:
                notes.append(f"• {col}: Moderate signal at {best.window_label} (d={d:.2f}) - "
                           f"retained={best.retained_momentum:.2f}, churned={best.churned_momentum:.2f}")
            else:
                notes.append(f"• {col}: No significant momentum difference between cohorts")
        return notes

    def generate_momentum_recommendations(
        self, results: Dict[str, List[CohortMomentumResult]]
    ) -> List[VelocityRecommendation]:
        recommendations = []
        for col, col_results in results.items():
            best = max(col_results, key=lambda r: abs(r.effect_size)) if col_results else None
            if best and abs(best.effect_size) >= 0.5:
                recommendations.append(VelocityRecommendation(
                    source_column=col, action="add_momentum_feature",
                    description=f"Add {best.window_label} momentum for {col} (d={best.effect_size:.2f})",
                    params={"short_window": best.short_window, "long_window": best.long_window},
                    effect_size=best.effect_size, priority=1 if abs(best.effect_size) >= 0.8 else 2
                ))
        return sorted(recommendations, key=lambda r: (-abs(r.effect_size), r.priority))

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

    def generate_lag_recommendations(self, results: Dict[str, LagCorrelationResult]) -> List[VelocityRecommendation]:
        recommendations = []
        for col, result in results.items():
            if result.best_correlation >= 0.3:
                recommendations.append(VelocityRecommendation(
                    source_column=col, action="add_lag_feature",
                    description=f"Add lag-{result.best_lag}d feature for {col} (r={result.best_correlation:.2f})",
                    params={"lag_days": result.best_lag, "correlation": result.best_correlation},
                    effect_size=result.best_correlation, priority=1 if result.best_correlation >= 0.5 else 2
                ))
            if result.has_weekly_pattern and result.best_lag != 7:
                recommendations.append(VelocityRecommendation(
                    source_column=col, action="add_weekly_lag",
                    description=f"Add lag-7d feature for {col} (weekly pattern detected)",
                    params={"lag_days": 7, "weekly_pattern": True},
                    effect_size=abs(result.correlations[6]) if len(result.correlations) >= 7 else 0.2,
                    priority=2
                ))
        return sorted(recommendations, key=lambda r: (-r.effect_size, r.priority))

    def generate_lag_interpretation(self, results: Dict[str, LagCorrelationResult]) -> List[str]:
        notes = []
        strong_lags = [(col, r) for col, r in results.items() if r.best_correlation >= 0.5]
        moderate_lags = [(col, r) for col, r in results.items() if 0.3 <= r.best_correlation < 0.5]
        weekly_patterns = [(col, r) for col, r in results.items() if r.has_weekly_pattern]
        weak_lags = [(col, r) for col, r in results.items() if r.best_correlation < 0.3]

        if strong_lags:
            cols = ", ".join(col for col, _ in strong_lags)
            notes.append(f"Strong autocorrelation (r >= 0.5): {cols}")
            notes.append("  → These variables have high predictability from past values")
            notes.append("  → Lag features will be highly informative")

        if moderate_lags:
            cols = ", ".join(col for col, _ in moderate_lags)
            notes.append(f"Moderate autocorrelation (0.3 <= r < 0.5): {cols}")
            notes.append("  → Past values provide useful but not dominant signal")

        if weekly_patterns:
            cols = ", ".join(col for col, _ in weekly_patterns)
            notes.append(f"Weekly patterns detected: {cols}")
            notes.append("  → Consider day_of_week features and lag-7d features")

        if weak_lags and len(weak_lags) == len(results):
            notes.append("All variables show weak autocorrelation (r < 0.3)")
            notes.append("  → Lag features may not be highly predictive")
            notes.append("  → Consider aggregated/rolling features instead")

        return notes

    def _validate_target_constant_per_entity(self, df: DataFrame, target_column: str) -> None:
        import warnings
        varying_entities = (df.groupby(self.entity_column)[target_column].nunique() > 1).sum()
        if varying_entities > 0:
            warnings.warn(
                f"Target '{target_column}' varies within {varying_entities} entities. "
                f"Using first value per entity. Target should be constant for retention modeling.",
                UserWarning, stacklevel=3,
            )

    def calculate_predictive_power(self, df: DataFrame, value_columns: List[str], target_column: str) -> Dict[str, PredictivePowerResult]:
        if self.time_column in df.columns:
            df = self._prepare_dataframe(df)
            self._validate_target_constant_per_entity(df, target_column)
            entity_data = self._aggregate_to_entity_level(df, value_columns, target_column)
        else:
            entity_data = df  # Already entity-level
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
        next_priority = 1

        if target_column:
            next_priority = self._add_predictive_power_recommendations(df, value_columns, target_column, recommendations, next_priority)
        next_priority = self._add_velocity_recommendations(df, value_columns, recommendations, next_priority)
        next_priority = self._add_momentum_recommendations(df, value_columns, recommendations, next_priority)
        self._add_lag_recommendations(df, value_columns, recommendations, next_priority)
        return recommendations

    def _add_predictive_power_recommendations(
        self, df: DataFrame, value_columns: List[str], target_column: str,
        recommendations: List[FeatureRecommendation], next_priority: int
    ) -> int:
        power_results = self.calculate_predictive_power(df, value_columns, target_column)
        for col, result in sorted(power_results.items(), key=lambda x: x[1].information_value, reverse=True):
            if result.information_value > self.IV_THRESHOLDS["weak"]:
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_mean", feature_type=FeatureType.ROLLING,
                    formula=f"df.groupby(entity)['{col}'].transform('mean')",
                    rationale=f"IV={result.information_value:.3f} ({result.iv_interpretation})",
                    priority=next_priority, source_column=col,
                ))
                next_priority += 1
        return next_priority

    def _add_velocity_recommendations(
        self, df: DataFrame, value_columns: List[str],
        recommendations: List[FeatureRecommendation], next_priority: int
    ) -> int:
        for col, result in self.calculate_velocity(df, value_columns).items():
            if result.trend_direction != "stable":
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_velocity_7d", feature_type=FeatureType.VELOCITY,
                    formula="(current - lag_7d) / lag_7d",
                    rationale=f"Detected {result.trend_direction} trend",
                    priority=next_priority, source_column=col,
                ))
                next_priority += 1
        return next_priority

    def _add_momentum_recommendations(
        self, df: DataFrame, value_columns: List[str],
        recommendations: List[FeatureRecommendation], next_priority: int
    ) -> int:
        for col, result in self.calculate_momentum(df, value_columns).items():
            if result.interpretation != "stable":
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_momentum_{result.short_window}_{result.long_window}",
                    feature_type=FeatureType.MOMENTUM,
                    formula=f"mean_{result.short_window}d / mean_{result.long_window}d",
                    rationale=f"Momentum indicates {result.interpretation} behavior",
                    priority=next_priority, source_column=col,
                ))
                next_priority += 1
        return next_priority

    def _add_lag_recommendations(
        self, df: DataFrame, value_columns: List[str],
        recommendations: List[FeatureRecommendation], next_priority: int
    ) -> int:
        for col, result in self.calculate_lag_correlations(df, value_columns).items():
            if result.best_correlation > 0.3:
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_lag_{result.best_lag}d", feature_type=FeatureType.LAG,
                    formula=f"df['{col}'].shift({result.best_lag})",
                    rationale=f"Strong autocorrelation (r={result.best_correlation:.2f}) at lag {result.best_lag}",
                    priority=next_priority, source_column=col,
                ))
                next_priority += 1
            if result.has_weekly_pattern:
                recommendations.append(FeatureRecommendation(
                    feature_name=f"{col}_weekly_pattern", feature_type=FeatureType.LAG,
                    formula=f"df['{col}'].shift(7)", rationale="Weekly seasonality detected",
                    priority=next_priority, source_column=col,
                ))
                next_priority += 1
        return next_priority

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
