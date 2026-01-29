from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from customer_retention.core.compat import DataFrame, pd
from customer_retention.core.utils import compute_effect_size


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class TrendResult:
    direction: TrendDirection
    strength: float
    slope: Optional[float] = None
    p_value: Optional[float] = None
    confidence: str = "low"

    @property
    def is_significant(self) -> bool:
        return self.p_value is not None and self.p_value < 0.05

    @property
    def has_direction(self) -> bool:
        return self.direction in [TrendDirection.INCREASING, TrendDirection.DECREASING]


@dataclass
class TrendRecommendation:
    action: str
    priority: str
    reason: str
    features: List[str] = field(default_factory=list)


@dataclass
class SeasonalityPeriod:
    period: int
    strength: float
    period_name: Optional[str] = None


@dataclass
class RecencyResult:
    avg_recency_days: float
    median_recency_days: float
    min_recency_days: float
    max_recency_days: float
    target_correlation: Optional[float] = None
    recency_distribution: Optional[dict] = None


@dataclass
class GroupStats:
    mean: float
    median: float
    std: float
    q25: float
    q75: float
    count: int


@dataclass
class RecencyBucketStats:
    bucket_label: str
    bucket_range: Tuple[int, int]
    entity_count: int
    target_rate: float


@dataclass
class RecencyInsight:
    finding: str
    metric_value: float
    metric_name: str


@dataclass
class AnomalyDiagnostics:
    target_1_is_minority: bool
    target_1_pct: float
    retained_median_tenure: Optional[float] = None
    churned_median_tenure: Optional[float] = None
    tenure_explains_pattern: bool = False


@dataclass
class RecencyComparisonResult:
    retained_stats: GroupStats
    churned_stats: GroupStats
    cohens_d: float
    effect_interpretation: str
    churned_higher: bool
    recommendations: List[Dict]
    bucket_stats: List[RecencyBucketStats] = field(default_factory=list)
    key_findings: List[RecencyInsight] = field(default_factory=list)
    inflection_bucket: Optional[str] = None
    distribution_pattern: str = "unknown"
    anomaly_diagnostics: Optional[AnomalyDiagnostics] = None


@dataclass
class CohortDistribution:
    year_counts: Dict[int, int]
    total_entities: int
    dominant_year: int
    dominant_pct: float
    num_years: int


@dataclass
class CohortRecommendation:
    action: str
    priority: str
    reason: str
    features: List[str] = field(default_factory=list)
    insight: Optional[str] = None


@dataclass
class TemporalPatternAnalysis:
    trend: Optional[TrendResult] = None
    seasonality: List[SeasonalityPeriod] = field(default_factory=list)
    cohort_analysis: Optional[DataFrame] = None
    recency_analysis: Optional[RecencyResult] = None


def compute_group_stats(values: np.ndarray) -> GroupStats:
    return GroupStats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        q25=float(np.percentile(values, 25)),
        q75=float(np.percentile(values, 75)),
        count=len(values)
    )


def generate_trend_recommendations(trend: TrendResult, mean_value: float = 1.0) -> List[TrendRecommendation]:
    recommendations = []
    daily_pct = (trend.slope / mean_value * 100) if trend.slope and mean_value else 0

    if trend.has_direction and trend.strength > 0.3 and trend.is_significant:
        recommendations.append(TrendRecommendation(
            action="add_trend_features", priority="high",
            features=["recent_vs_overall_ratio", "entity_trend_slope"],
            reason=f"Strong {trend.direction.value} trend (R²={trend.strength:.2f}, {daily_pct:+.2f}%/day)"
        ))
        recommendations.append(TrendRecommendation(
            action="consider_detrending", priority="medium", features=[],
            reason="Strong trend may dominate signal - consider detrending aggregated features"
        ))
        recommendations.append(TrendRecommendation(
            action="time_based_split", priority="high", features=[],
            reason="Strong trend detected - use time-based train/test split to avoid leakage"
        ))
    elif trend.has_direction and trend.strength > 0.1 and trend.is_significant:
        recommendations.append(TrendRecommendation(
            action="add_trend_features", priority="medium",
            features=["recent_vs_overall_ratio"],
            reason=f"Moderate {trend.direction.value} trend (R²={trend.strength:.2f})"
        ))
    elif trend.direction == TrendDirection.STABLE:
        recommendations.append(TrendRecommendation(
            action="skip_trend_features", priority="low", features=[],
            reason=f"No significant trend (R²={trend.strength:.2f}) - trend features unlikely to help"
        ))
    return recommendations


def analyze_cohort_distribution(first_events: DataFrame, time_column: str) -> CohortDistribution:
    years = first_events[time_column].dt.year
    year_counts = years.value_counts().sort_index().to_dict()
    total = len(first_events)
    dominant_year = years.mode().iloc[0] if len(years) > 0 else 0
    dominant_pct = (year_counts.get(dominant_year, 0) / total * 100) if total > 0 else 0
    return CohortDistribution(
        year_counts=year_counts, total_entities=total,
        dominant_year=int(dominant_year), dominant_pct=dominant_pct, num_years=len(year_counts)
    )


def generate_cohort_recommendations(
    dist: CohortDistribution, retention_variation: Optional[float] = None
) -> List[CohortRecommendation]:
    recommendations = []
    skew_threshold = 80

    if dist.dominant_pct > skew_threshold:
        recommendations.append(CohortRecommendation(
            action="skip_cohort_features", priority="low",
            reason=f"{dist.dominant_pct:.0f}% onboarded in {dist.dominant_year} - insufficient variation",
            insight="Established customer base, not a growing acquisition funnel"
        ))
    elif dist.num_years >= 3 and dist.dominant_pct < 60:
        recommendations.append(CohortRecommendation(
            action="add_cohort_features", priority="medium",
            features=["cohort_year", "cohort_quarter"],
            reason=f"Good variation across {dist.num_years} years - cohort features may be valuable"
        ))
    else:
        recommendations.append(CohortRecommendation(
            action="consider_cohort_features", priority="low",
            features=["cohort_year"],
            reason="Moderate variation - test if cohort features improve model"
        ))

    if retention_variation is not None and retention_variation > 0.1:
        recommendations.append(CohortRecommendation(
            action="investigate_cohort_retention", priority="medium",
            reason=f"Retention varies {retention_variation*100:.0f}pp across cohorts - investigate drivers"
        ))
    return recommendations


DEFAULT_BUCKET_EDGES = [0, 7, 30, 90, 180, float("inf")]
BUCKET_LABELS = ["0-7d", "8-30d", "31-90d", "91-180d", ">180d"]
INFLECTION_MIN_DROP = 0.10
MONOTONIC_TOLERANCE = 0.05


def compute_recency_buckets(
    df: DataFrame, entity_column: str, time_column: str, target_column: str,
    reference_date: pd.Timestamp, bucket_edges: Optional[List[float]] = None
) -> List[RecencyBucketStats]:
    edges = bucket_edges or DEFAULT_BUCKET_EDGES
    labels = _generate_bucket_labels(edges)
    entity_last = df.groupby(entity_column)[time_column].max().reset_index()
    entity_last["recency_days"] = (reference_date - entity_last[time_column]).dt.days
    entity_target = df.groupby(entity_column)[target_column].first().reset_index()
    entity_data = entity_last.merge(entity_target, on=entity_column)
    entity_data["bucket"] = pd.cut(entity_data["recency_days"], bins=edges, labels=labels, include_lowest=True)
    bucket_stats = []
    for i, label in enumerate(labels):
        bucket_data = entity_data[entity_data["bucket"] == label]
        if len(bucket_data) == 0:
            continue
        bucket_stats.append(RecencyBucketStats(
            bucket_label=label,
            bucket_range=(int(edges[i]), int(edges[i + 1]) if edges[i + 1] != float("inf") else 9999),
            entity_count=len(bucket_data),
            target_rate=float(bucket_data[target_column].mean())
        ))
    return bucket_stats


def _generate_bucket_labels(edges: List[float]) -> List[str]:
    labels = []
    for i in range(len(edges) - 1):
        start, end = int(edges[i]), edges[i + 1]
        if end == float("inf"):
            labels.append(f">{start}d")
        elif start == 0:
            labels.append(f"0-{int(end)}d")
        else:
            labels.append(f"{start + 1}-{int(end)}d")
    return labels


def detect_inflection_bucket(buckets: List[RecencyBucketStats]) -> Optional[str]:
    if len(buckets) < 2:
        return None
    max_drop, inflection_label = 0.0, None
    for i in range(len(buckets) - 1):
        drop = buckets[i].target_rate - buckets[i + 1].target_rate
        if drop > max_drop:
            max_drop, inflection_label = drop, buckets[i + 1].bucket_label
    return inflection_label if max_drop >= INFLECTION_MIN_DROP else None


def classify_distribution_pattern(buckets: List[RecencyBucketStats]) -> str:
    if len(buckets) < 2:
        return "insufficient_data"
    rates = [b.target_rate for b in buckets]
    total_drop = rates[0] - rates[-1]
    if abs(total_drop) < MONOTONIC_TOLERANCE:
        return "flat_no_pattern"
    drops = [rates[i] - rates[i + 1] for i in range(len(rates) - 1)]
    max_drop = max(drops) if drops else 0
    avg_drop = total_drop / (len(rates) - 1) if len(rates) > 1 else 0
    if max_drop > avg_drop * 2 and max_drop >= INFLECTION_MIN_DROP:
        return "threshold_step"
    if all(d >= -MONOTONIC_TOLERANCE for d in drops):
        return "monotonic_decline"
    return "variable"


def _diagnose_anomaly_pattern(
    df: DataFrame, entity_column: str, time_column: str, target_column: str
) -> AnomalyDiagnostics:
    entity_target = df.groupby(entity_column)[target_column].first()
    target_1_pct = float(entity_target.mean() * 100)
    target_1_is_minority = target_1_pct < 50
    entity_first = df.groupby(entity_column)[time_column].min()
    entity_last = df.groupby(entity_column)[time_column].max()
    tenure = (entity_last - entity_first).dt.days
    tenure_by_target = pd.DataFrame({"target": entity_target, "tenure": tenure})
    retained_tenure = tenure_by_target[tenure_by_target["target"] == 1]["tenure"]
    churned_tenure = tenure_by_target[tenure_by_target["target"] == 0]["tenure"]
    retained_median_tenure = float(retained_tenure.median()) if len(retained_tenure) > 0 else None
    churned_median_tenure = float(churned_tenure.median()) if len(churned_tenure) > 0 else None
    tenure_explains = False
    if retained_median_tenure and churned_median_tenure:
        tenure_explains = retained_median_tenure > churned_median_tenure * 1.5
    return AnomalyDiagnostics(
        target_1_is_minority=target_1_is_minority,
        target_1_pct=target_1_pct,
        retained_median_tenure=retained_median_tenure,
        churned_median_tenure=churned_median_tenure,
        tenure_explains_pattern=tenure_explains
    )


def generate_recency_insights(result: "RecencyComparisonResult") -> List[RecencyInsight]:
    insights = []
    median_gap = result.churned_stats.median - result.retained_stats.median
    gap_direction = "longer" if median_gap > 0 else "shorter"
    insights.append(RecencyInsight(
        finding=f"Churned entities last active {abs(median_gap):.0f} days {gap_direction} than retained (median: {result.churned_stats.median:.0f}d vs {result.retained_stats.median:.0f}d)",
        metric_value=median_gap,
        metric_name="median_gap_days"
    ))
    if not result.churned_higher and result.anomaly_diagnostics:
        diag = result.anomaly_diagnostics
        anomaly_parts = ["⚠️ Unusual pattern: churned have MORE recent activity."]
        if diag.target_1_is_minority:
            anomaly_parts.append(f"Target=1 is minority ({diag.target_1_pct:.0f}%) - likely means CHURN not retention.")
        else:
            anomaly_parts.append(f"Target=1 is majority ({diag.target_1_pct:.0f}%) - confirms retention label.")
        if diag.tenure_explains_pattern:
            anomaly_parts.append(f"Tenure gap explains pattern: retained={diag.retained_median_tenure:.0f}d vs churned={diag.churned_median_tenure:.0f}d median tenure.")
        insights.append(RecencyInsight(finding=" ".join(anomaly_parts), metric_value=0.0, metric_name="pattern_anomaly"))
    effect_desc = _effect_size_description(result.cohens_d, result.effect_interpretation)
    insights.append(RecencyInsight(finding=effect_desc, metric_value=abs(result.cohens_d), metric_name="effect_size"))
    if result.inflection_bucket and result.churned_higher:
        insights.append(RecencyInsight(
            finding=f"Sharpest target rate drop occurs at {result.inflection_bucket} boundary",
            metric_value=0.0, metric_name="inflection_point"
        ))
    return insights


def _effect_size_description(cohens_d: float, interpretation: str) -> str:
    abs_d = abs(cohens_d)
    if abs_d >= 0.8:
        return f"Recency strongly discriminates target ({interpretation}, d={cohens_d:+.2f}) - high predictive value"
    if abs_d >= 0.5:
        return f"Recency moderately discriminates target ({interpretation}, d={cohens_d:+.2f}) - useful predictor"
    if abs_d >= 0.2:
        return f"Recency weakly discriminates target ({interpretation}, d={cohens_d:+.2f}) - may help in combination"
    return f"Recency has minimal discriminative power ({interpretation}, d={cohens_d:+.2f})"


def _generate_enhanced_recommendations(
    churned_higher: bool, cohens_d: float, inflection_bucket: Optional[str],
    distribution_pattern: str, bucket_stats: List[RecencyBucketStats],
    anomaly_diagnostics: Optional[AnomalyDiagnostics] = None
) -> List[Dict]:
    recommendations = []
    if not churned_higher:
        diag = anomaly_diagnostics
        if diag and diag.target_1_is_minority:
            recommendations.append({
                "action": "invert_target_interpretation", "priority": "high",
                "reason": f"Target=1 is minority ({diag.target_1_pct:.0f}%) - interpret as CHURN; recency pattern is classic churn behavior",
                "features": ["days_since_last_event", "log_recency"]
            })
        elif diag and diag.tenure_explains_pattern:
            recommendations.append({
                "action": "use_tenure_adjusted_recency", "priority": "high",
                "reason": f"Retained have {diag.retained_median_tenure:.0f}d vs churned {diag.churned_median_tenure:.0f}d median tenure - use recency relative to tenure",
                "features": ["recency_vs_tenure_ratio", "normalized_recency"]
            })
        else:
            recommendations.append({
                "action": "investigate_further", "priority": "high",
                "reason": "Pattern unexpected and not explained by target definition or tenure - review data collection",
                "features": []
            })
        if diag and not diag.target_1_is_minority and not diag.tenure_explains_pattern:
            recommendations.append({
                "action": "check_pre_churn_activity", "priority": "medium",
                "reason": "Churned may show activity spike before leaving (support tickets, complaints)",
                "features": ["activity_trend_last_30d", "support_interaction_count"]
            })
        return recommendations[:3]
    abs_d = abs(cohens_d)
    if abs_d >= 0.5:
        recommendations.append({
            "action": "add_recency_features", "priority": "high",
            "reason": f"Strong effect size (d={cohens_d:+.2f}) - recency is a key predictor",
            "features": ["days_since_last_event", "log_recency"]
        })
    if inflection_bucket and distribution_pattern == "threshold_step":
        threshold_days = _extract_threshold_from_bucket(inflection_bucket)
        recommendations.append({
            "action": "create_activity_threshold_flag", "priority": "high",
            "reason": f"Clear threshold at {inflection_bucket}: create binary is_active_{threshold_days}d flag",
            "features": [f"is_active_{threshold_days}d"]
        })
    elif distribution_pattern == "monotonic_decline":
        recommendations.append({
            "action": "use_continuous_recency", "priority": "medium",
            "reason": "Monotonic decline pattern: continuous recency features outperform binary flags",
            "features": ["days_since_last_event", "log_recency", "recency_percentile"]
        })
    if len(recommendations) < 2 and bucket_stats:
        recommendations.append({
            "action": "add_recency_buckets", "priority": "medium",
            "reason": "Create recency bucket features for interpretable segments",
            "features": ["recency_bucket"]
        })
    return recommendations[:3]


def _extract_threshold_from_bucket(bucket_label: str) -> int:
    import re
    match = re.search(r"(\d+)", bucket_label)
    return int(match.group(1)) if match else 30


def compare_recency_by_target(
    df: DataFrame, entity_column: str, time_column: str, target_column: str,
    reference_date: Optional[pd.Timestamp] = None, cap_percentile: float = 0.99
) -> Optional[RecencyComparisonResult]:
    if target_column not in df.columns:
        return None
    ref_date = reference_date or df[time_column].max()
    entity_last = df.groupby(entity_column)[time_column].max().reset_index()
    entity_last["recency_days"] = (ref_date - entity_last[time_column]).dt.days
    entity_target = df.groupby(entity_column)[target_column].first().reset_index()
    entity_recency = entity_last.merge(entity_target, on=entity_column)
    cap = entity_recency["recency_days"].quantile(cap_percentile)
    entity_capped = entity_recency[entity_recency["recency_days"] <= cap]
    retained = entity_capped[entity_capped[target_column] == 1]["recency_days"].values
    churned = entity_capped[entity_capped[target_column] == 0]["recency_days"].values
    if len(retained) < 2 or len(churned) < 2:
        return None
    cohens_d, effect_interp = compute_effect_size(retained, churned)
    churned_higher = bool(np.median(churned) > np.median(retained))
    bucket_stats = compute_recency_buckets(df, entity_column, time_column, target_column, ref_date)
    inflection_bucket = detect_inflection_bucket(bucket_stats)
    distribution_pattern = classify_distribution_pattern(bucket_stats)
    anomaly_diag = _diagnose_anomaly_pattern(df, entity_column, time_column, target_column) if not churned_higher else None
    recommendations = _generate_enhanced_recommendations(
        churned_higher, cohens_d, inflection_bucket, distribution_pattern, bucket_stats, anomaly_diag
    )
    result = RecencyComparisonResult(
        retained_stats=compute_group_stats(retained),
        churned_stats=compute_group_stats(churned),
        cohens_d=cohens_d, effect_interpretation=effect_interp,
        churned_higher=churned_higher, recommendations=recommendations,
        bucket_stats=bucket_stats, inflection_bucket=inflection_bucket,
        distribution_pattern=distribution_pattern, anomaly_diagnostics=anomaly_diag
    )
    result.key_findings = generate_recency_insights(result)
    return result


class TemporalPatternAnalyzer:
    TREND_THRESHOLD = 0.001
    CONFIDENCE_HIGH_P = 0.01
    CONFIDENCE_HIGH_R2 = 0.5
    CONFIDENCE_MED_P = 0.05
    CONFIDENCE_MED_R2 = 0.3

    def __init__(self, time_column: str):
        self.time_column = time_column

    def analyze(self, df: DataFrame, value_column: str, entity_column: Optional[str] = None, target_column: Optional[str] = None) -> TemporalPatternAnalysis:
        if len(df) < 2:
            return TemporalPatternAnalysis()

        trend = self.detect_trend(df, value_column)
        seasonality = self.detect_seasonality(df, value_column)

        return TemporalPatternAnalysis(
            trend=trend,
            seasonality=seasonality,
        )

    @staticmethod
    def _unknown_trend() -> TrendResult:
        return TrendResult(direction=TrendDirection.UNKNOWN, strength=0.0, confidence="low")

    def detect_trend(self, df: DataFrame, value_column: str) -> TrendResult:
        if len(df) < 3:
            return self._unknown_trend()

        df_clean = df[[self.time_column, value_column]].dropna()
        if len(df_clean) < 3:
            return self._unknown_trend()

        time_col = pd.to_datetime(df_clean[self.time_column])
        x = (time_col - time_col.min()).dt.total_seconds() / 86400
        y = df_clean[value_column].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        mean_y = np.mean(y)
        normalized_slope = slope / mean_y if mean_y != 0 else 0

        if abs(normalized_slope) < self.TREND_THRESHOLD:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        if p_value < self.CONFIDENCE_HIGH_P and r_squared > self.CONFIDENCE_HIGH_R2:
            confidence = "high"
        elif p_value < self.CONFIDENCE_MED_P and r_squared > self.CONFIDENCE_MED_R2:
            confidence = "medium"
        else:
            confidence = "low"

        return TrendResult(
            direction=direction,
            strength=r_squared,
            slope=slope,
            p_value=p_value,
            confidence=confidence
        )

    def detect_seasonality(self, df: DataFrame, value_column: str, max_periods: int = 3, additional_lags: Optional[List[int]] = None) -> List[SeasonalityPeriod]:
        if len(df) < 14:
            return []

        df_clean = df[[self.time_column, value_column]].dropna()
        if len(df_clean) < 14:
            return []

        df_sorted = df_clean.sort_values(self.time_column)
        values = df_sorted[value_column].values

        results = []
        period_names = {7: "weekly", 14: "bi-weekly", 21: "tri-weekly", 30: "monthly", 90: "quarterly", 180: "semi-annual", 365: "yearly"}

        base_lags = [7, 14, 21, 30]
        all_lags = list(set(base_lags + (additional_lags or [])))

        for lag in all_lags:
            if lag >= len(values) // 2:
                continue

            acf = self._autocorrelation(values, lag)

            if acf > 0.3:
                period_name = period_names.get(lag, f"{lag}-day")
                results.append(SeasonalityPeriod(
                    period=lag,
                    strength=acf,
                    period_name=period_name
                ))

        results.sort(key=lambda x: x.strength, reverse=True)
        return results[:max_periods]

    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        n = len(series)
        if lag >= n:
            return 0.0

        mean = np.mean(series)
        var = np.var(series)

        if var == 0:
            return 0.0

        cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
        return cov / var

    def analyze_cohorts(self, df: DataFrame, entity_column: str, cohort_column: str, target_column: Optional[str] = None, period: str = "M") -> DataFrame:
        if len(df) == 0:
            return pd.DataFrame()

        df_copy = df.copy()
        entity_first_event = df_copy.groupby(entity_column)[cohort_column].min()
        df_copy["_cohort"] = df_copy[entity_column].map(entity_first_event)
        df_copy["_cohort"] = pd.to_datetime(df_copy["_cohort"]).dt.to_period(period)

        entity_cohorts = df_copy.groupby(entity_column)["_cohort"].first().reset_index()
        entity_cohorts.columns = [entity_column, "_cohort"]

        cohort_stats = entity_cohorts.groupby("_cohort").agg({entity_column: "count"}).reset_index()
        cohort_stats.columns = ["cohort", "entity_count"]

        cohort_dates = df_copy.groupby("_cohort")[self.time_column].agg(["min", "max"]).reset_index()
        cohort_dates.columns = ["cohort", "first_event", "last_event"]
        cohort_stats = cohort_stats.merge(cohort_dates, on="cohort", how="left")

        if target_column and target_column in df.columns:
            entity_target = df_copy.groupby(entity_column)[target_column].max()
            entity_cohorts["_target"] = entity_cohorts[entity_column].map(entity_target)
            target_stats = entity_cohorts.groupby("_cohort")["_target"].mean().reset_index()
            target_stats.columns = ["cohort", "retention_rate"]
            cohort_stats = cohort_stats.merge(target_stats, on="cohort", how="left")

        return cohort_stats.sort_values("cohort")

    def analyze_recency(self, df: DataFrame, entity_column: str, target_column: Optional[str] = None, reference_date: Optional[pd.Timestamp] = None) -> RecencyResult:
        if len(df) == 0:
            return RecencyResult(avg_recency_days=0, median_recency_days=0, min_recency_days=0, max_recency_days=0)

        ref_date = reference_date or pd.Timestamp.now()
        pd.to_datetime(df[self.time_column])

        entity_last = df.groupby(entity_column)[self.time_column].max()
        entity_last = pd.to_datetime(entity_last)
        recency_days = (ref_date - entity_last).dt.days

        target_correlation = None
        if target_column and target_column in df.columns:
            entity_target = df.groupby(entity_column)[target_column].first()
            combined = pd.DataFrame({"recency": recency_days, "target": entity_target}).dropna()

            if len(combined) > 2:
                corr, _ = stats.pearsonr(combined["recency"], combined["target"])
                target_correlation = corr

        return RecencyResult(
            avg_recency_days=float(recency_days.mean()),
            median_recency_days=float(recency_days.median()),
            min_recency_days=float(recency_days.min()),
            max_recency_days=float(recency_days.max()),
            target_correlation=target_correlation,
        )
