"""
Temporal pattern analyzer for time series datasets.

Analyzes patterns including:
- Trend detection (increasing, decreasing, stable)
- Seasonality detection (daily, weekly, monthly patterns)
- Cohort analysis (behavior by signup period)
- Recency analysis (time since last event vs target)
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy import stats

from customer_retention.core.compat import DataFrame, pd


class TrendDirection(str, Enum):
    """Direction of trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class TrendResult:
    """Result of trend detection."""
    direction: TrendDirection
    strength: float  # R-squared or similar (0-1)
    slope: Optional[float] = None
    p_value: Optional[float] = None
    confidence: str = "low"  # low, medium, high


@dataclass
class SeasonalityPeriod:
    """Result of seasonality detection."""
    period: int  # Period in the same units as data frequency
    strength: float  # Strength of seasonality (0-1)
    period_name: Optional[str] = None  # "daily", "weekly", etc.


@dataclass
class RecencyResult:
    """Result of recency analysis."""
    avg_recency_days: float
    median_recency_days: float
    min_recency_days: float
    max_recency_days: float
    target_correlation: Optional[float] = None
    recency_distribution: Optional[dict] = None


@dataclass
class TemporalPatternAnalysis:
    """Complete temporal pattern analysis result."""
    trend: Optional[TrendResult] = None
    seasonality: List[SeasonalityPeriod] = field(default_factory=list)
    cohort_analysis: Optional[DataFrame] = None
    recency_analysis: Optional[RecencyResult] = None


class TemporalPatternAnalyzer:
    """Analyzes temporal patterns in time series data."""

    TREND_THRESHOLD = 0.001  # Minimum normalized slope (daily change as % of mean)

    def __init__(self, time_column: str):
        self.time_column = time_column

    def analyze(self, df: DataFrame, value_column: str,
                entity_column: Optional[str] = None,
                target_column: Optional[str] = None) -> TemporalPatternAnalysis:
        """Run complete temporal pattern analysis."""
        if len(df) < 2:
            return TemporalPatternAnalysis()

        trend = self.detect_trend(df, value_column)
        seasonality = self.detect_seasonality(df, value_column)

        return TemporalPatternAnalysis(
            trend=trend,
            seasonality=seasonality,
        )

    def detect_trend(self, df: DataFrame, value_column: str) -> TrendResult:
        """Detect trend in time series data."""
        if len(df) < 3:
            return TrendResult(
                direction=TrendDirection.UNKNOWN,
                strength=0.0,
                confidence="low"
            )

        df_clean = df[[self.time_column, value_column]].dropna()
        if len(df_clean) < 3:
            return TrendResult(
                direction=TrendDirection.UNKNOWN,
                strength=0.0,
                confidence="low"
            )

        # Convert time to numeric (days from start)
        time_col = pd.to_datetime(df_clean[self.time_column])
        x = (time_col - time_col.min()).dt.total_seconds() / 86400
        y = df_clean[value_column].values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        # Normalize slope relative to mean value
        mean_y = np.mean(y)
        normalized_slope = slope / mean_y if mean_y != 0 else 0

        # Determine direction
        if abs(normalized_slope) < self.TREND_THRESHOLD:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Determine confidence
        if p_value < 0.01 and r_squared > 0.5:
            confidence = "high"
        elif p_value < 0.05 and r_squared > 0.3:
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

    def detect_seasonality(self, df: DataFrame, value_column: str,
                          max_periods: int = 3) -> List[SeasonalityPeriod]:
        """Detect seasonal patterns using autocorrelation."""
        if len(df) < 14:  # Need at least 2 weeks of data
            return []

        df_clean = df[[self.time_column, value_column]].dropna()
        if len(df_clean) < 14:
            return []

        # Sort by time
        df_sorted = df_clean.sort_values(self.time_column)
        values = df_sorted[value_column].values

        # Compute autocorrelation for different lags
        results = []
        period_names = {7: "weekly", 30: "monthly", 365: "yearly", 1: "daily"}

        for lag in [7, 30, 14, 21]:  # Common seasonal periods
            if lag >= len(values) // 2:
                continue

            # Calculate autocorrelation at this lag
            acf = self._autocorrelation(values, lag)

            if acf > 0.3:  # Significant correlation
                period_name = period_names.get(lag)
                results.append(SeasonalityPeriod(
                    period=lag,
                    strength=acf,
                    period_name=period_name
                ))

        # Sort by strength and return top results
        results.sort(key=lambda x: x.strength, reverse=True)
        return results[:max_periods]

    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        n = len(series)
        if lag >= n:
            return 0.0

        mean = np.mean(series)
        var = np.var(series)

        if var == 0:
            return 0.0

        cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
        return cov / var

    def analyze_cohorts(self, df: DataFrame,
                       entity_column: str,
                       cohort_column: str,
                       target_column: Optional[str] = None,
                       period: str = "M") -> DataFrame:
        """Analyze behavior by cohort (signup period)."""
        if len(df) == 0:
            return pd.DataFrame()

        df_copy = df.copy()
        df_copy["_cohort"] = pd.to_datetime(df_copy[cohort_column]).dt.to_period(period)

        # Aggregate by cohort
        cohort_stats = df_copy.groupby("_cohort").agg({
            entity_column: "nunique",
            self.time_column: ["min", "max"],
        }).reset_index()

        cohort_stats.columns = ["cohort", "entity_count", "first_event", "last_event"]

        # Add target stats if provided
        if target_column and target_column in df.columns:
            target_stats = df_copy.groupby("_cohort")[target_column].mean().reset_index()
            target_stats.columns = ["cohort", "retention_rate"]
            cohort_stats = cohort_stats.merge(
                target_stats,
                on="cohort",
                how="left"
            )

        return cohort_stats

    def analyze_recency(self, df: DataFrame,
                       entity_column: str,
                       target_column: Optional[str] = None,
                       reference_date: Optional[pd.Timestamp] = None) -> RecencyResult:
        """Analyze recency patterns (time since last event)."""
        if len(df) == 0:
            return RecencyResult(
                avg_recency_days=0,
                median_recency_days=0,
                min_recency_days=0,
                max_recency_days=0
            )

        ref_date = reference_date or pd.Timestamp.now()
        pd.to_datetime(df[self.time_column])

        # Calculate recency per entity (using last event date)
        entity_last = df.groupby(entity_column)[self.time_column].max()
        entity_last = pd.to_datetime(entity_last)
        recency_days = (ref_date - entity_last).dt.days

        # Calculate correlation with target if provided
        target_correlation = None
        if target_column and target_column in df.columns:
            entity_target = df.groupby(entity_column)[target_column].first()
            # Align the series
            combined = pd.DataFrame({
                "recency": recency_days,
                "target": entity_target
            }).dropna()

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
