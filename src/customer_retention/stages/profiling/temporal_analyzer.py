from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from customer_retention.core.compat import Series, ensure_pandas_series


class TemporalGranularity(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class SeasonalityResult:
    has_seasonality: bool
    dominant_period: Optional[str] = None  # 'weekly', 'monthly', 'yearly'
    peak_periods: List[str] = field(default_factory=list)
    trough_periods: List[str] = field(default_factory=list)
    monthly_pattern: Optional[pd.DataFrame] = None  # year x month heatmap data
    weekly_pattern: Optional[pd.Series] = None  # day of week counts
    confidence: float = 0.0
    seasonal_strength: float = 0.0


class TemporalRecommendationType(Enum):
    FEATURE_ENGINEERING = "feature_engineering"  # Create new feature from date
    MODELING_STRATEGY = "modeling_strategy"      # How to handle in train/test
    DATA_QUALITY = "data_quality"                # Quality issue to address


@dataclass
class TemporalRecommendation:
    feature_name: str
    recommendation_type: TemporalRecommendationType
    category: str  # 'recency', 'duration', 'cyclical', 'extraction', 'tenure', 'split', 'filter'
    reason: str
    priority: str  # 'high', 'medium', 'low'
    code_hint: Optional[str] = None

    @property
    def action_description(self) -> str:
        if self.recommendation_type == TemporalRecommendationType.FEATURE_ENGINEERING:
            return f"Create feature: {self.feature_name}"
        elif self.recommendation_type == TemporalRecommendationType.MODELING_STRATEGY:
            return f"Modeling: {self.feature_name}"
        else:
            return f"Data quality: {self.feature_name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "recommendation_type": self.recommendation_type.value,
            "category": self.category,
            "reason": self.reason,
            "priority": self.priority,
            "code_hint": self.code_hint,
        }


@dataclass
class TemporalAnalysis:
    granularity: TemporalGranularity
    min_date: pd.Timestamp
    max_date: pd.Timestamp
    span_days: int
    total_count: int
    null_count: int
    period_counts: pd.DataFrame

    @property
    def null_percentage(self) -> float:
        return (self.null_count / self.total_count * 100) if self.total_count > 0 else 0.0


class TemporalAnalyzer:
    GRANULARITY_THRESHOLDS = {
        90: TemporalGranularity.DAY,      # <= 90 days: daily
        365: TemporalGranularity.WEEK,    # <= 1 year: weekly
        730: TemporalGranularity.MONTH,   # <= 2 years: monthly
        1825: TemporalGranularity.QUARTER,  # <= 5 years: quarterly
    }

    def detect_granularity(self, dates: Series) -> TemporalGranularity:
        dates = ensure_pandas_series(dates)
        clean_dates = pd.to_datetime(dates, errors="coerce").dropna()
        if len(clean_dates) == 0:
            return TemporalGranularity.MONTH

        span_days = (clean_dates.max() - clean_dates.min()).days
        for threshold, granularity in self.GRANULARITY_THRESHOLDS.items():
            if span_days <= threshold:
                return granularity
        return TemporalGranularity.YEAR

    def aggregate_by_granularity(
        self, dates: Series, granularity: TemporalGranularity
    ) -> pd.DataFrame:
        dates = ensure_pandas_series(dates)
        clean_dates = pd.to_datetime(dates, errors="coerce").dropna()
        if len(clean_dates) == 0:
            return pd.DataFrame({"period": [], "count": []})

        period_series = self._extract_period(clean_dates, granularity)
        counts = period_series.value_counts().sort_index().reset_index()
        counts.columns = ["period", "count"]
        return counts

    def _extract_period(
        self, dates: pd.Series, granularity: TemporalGranularity
    ) -> pd.Series:
        if granularity == TemporalGranularity.DAY:
            return dates.dt.strftime("%Y-%m-%d")
        elif granularity == TemporalGranularity.WEEK:
            return dates.dt.to_period("W").astype(str)
        elif granularity == TemporalGranularity.MONTH:
            return dates.dt.to_period("M").astype(str)
        elif granularity == TemporalGranularity.QUARTER:
            return dates.dt.to_period("Q").astype(str)
        else:  # YEAR
            return dates.dt.year

    def analyze(
        self,
        dates: Series,
        granularity: Optional[TemporalGranularity] = None,
    ) -> TemporalAnalysis:
        dates = ensure_pandas_series(dates)
        total_count = len(dates)
        parsed_dates = pd.to_datetime(dates, errors="coerce")
        null_count = parsed_dates.isna().sum()
        clean_dates = parsed_dates.dropna()

        if len(clean_dates) == 0:
            return TemporalAnalysis(
                granularity=granularity or TemporalGranularity.MONTH,
                min_date=pd.NaT,
                max_date=pd.NaT,
                span_days=0,
                total_count=total_count,
                null_count=null_count,
                period_counts=pd.DataFrame({"period": [], "count": []}),
            )

        detected_granularity = granularity or self.detect_granularity(clean_dates)
        period_counts = self.aggregate_by_granularity(clean_dates, detected_granularity)

        return TemporalAnalysis(
            granularity=detected_granularity,
            min_date=clean_dates.min(),
            max_date=clean_dates.max(),
            span_days=(clean_dates.max() - clean_dates.min()).days,
            total_count=total_count,
            null_count=null_count,
            period_counts=period_counts,
        )

    def analyze_seasonality(self, dates: Series) -> SeasonalityResult:
        """Analyze seasonality patterns in datetime data."""
        dates = ensure_pandas_series(dates)
        parsed = pd.to_datetime(dates, errors="coerce").dropna()

        if len(parsed) < 30:
            return SeasonalityResult(has_seasonality=False, confidence=0.0)

        # Weekly pattern (day of week)
        dow_counts = parsed.dt.dayofweek.value_counts().sort_index()
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekly_pattern = pd.Series(
            [dow_counts.get(i, 0) for i in range(7)], index=dow_names
        )

        # Monthly pattern (year x month heatmap)
        monthly_df = parsed.to_frame(name="date")
        monthly_df["year"] = parsed.dt.year
        monthly_df["month"] = parsed.dt.month
        monthly_pivot = monthly_df.groupby(["year", "month"]).size().unstack(fill_value=0)
        monthly_pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ][:len(monthly_pivot.columns)]

        # Detect seasonality strength
        weekly_cv = weekly_pattern.std() / weekly_pattern.mean() if weekly_pattern.mean() > 0 else 0

        # Find peaks and troughs
        monthly_totals = parsed.dt.month.value_counts().sort_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        if len(monthly_totals) >= 3:
            peak_months = monthly_totals.nlargest(3).index.tolist()
            trough_months = monthly_totals.nsmallest(3).index.tolist()
            peak_periods = [month_names[m - 1] for m in peak_months if m <= 12]
            trough_periods = [month_names[m - 1] for m in trough_months if m <= 12]
        else:
            peak_periods = []
            trough_periods = []

        # Determine dominant pattern
        has_seasonality = weekly_cv > 0.15
        dominant_period = "weekly" if weekly_cv > 0.2 else None

        return SeasonalityResult(
            has_seasonality=has_seasonality,
            dominant_period=dominant_period,
            peak_periods=peak_periods,
            trough_periods=trough_periods,
            monthly_pattern=monthly_pivot,
            weekly_pattern=weekly_pattern,
            confidence=min(1.0, weekly_cv * 2) if has_seasonality else 0.0,
            seasonal_strength=float(weekly_cv),
        )

    def year_over_year_comparison(self, dates: Series) -> pd.DataFrame:
        """Compare record counts year-over-year by month."""
        dates = ensure_pandas_series(dates)
        parsed = pd.to_datetime(dates, errors="coerce").dropna()

        if len(parsed) == 0:
            return pd.DataFrame()

        df = parsed.to_frame(name="date")
        df["year"] = parsed.dt.year
        df["month"] = parsed.dt.month

        pivot = df.groupby(["year", "month"]).size().unstack(fill_value=0)
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ][:len(pivot.columns)]

        return pivot

    def calculate_growth_rate(self, dates: Series) -> Dict[str, Any]:
        """Calculate growth metrics over time."""
        dates = ensure_pandas_series(dates)
        parsed = pd.to_datetime(dates, errors="coerce").dropna()

        if len(parsed) < 2:
            return {"has_data": False}

        # Monthly counts
        monthly = parsed.dt.to_period("M").value_counts().sort_index()

        if len(monthly) < 2:
            return {"has_data": False}

        # Calculate month-over-month growth
        mom_growth = monthly.pct_change().dropna()

        # Calculate cumulative
        cumulative = monthly.cumsum()

        # Linear trend
        x = np.arange(len(monthly))
        y = monthly.values
        slope, intercept = np.polyfit(x, y, 1)
        trend_direction = "growing" if slope > 0 else "declining"

        # Overall growth rate
        overall_growth = ((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100) if monthly.iloc[0] > 0 else 0

        return {
            "has_data": True,
            "monthly_counts": monthly,
            "cumulative": cumulative,
            "avg_monthly_growth": float(mom_growth.mean() * 100),
            "overall_growth_pct": float(overall_growth),
            "trend_direction": trend_direction,
            "trend_slope": float(slope),
        }

    def recommend_features(
        self, dates: Series, column_name: str, other_date_columns: Optional[List[str]] = None
    ) -> List[TemporalRecommendation]:
        dates = ensure_pandas_series(dates)
        parsed = pd.to_datetime(dates, errors="coerce")
        valid_dates = parsed.dropna()
        recommendations = []

        if len(valid_dates) == 0:
            return recommendations

        analysis = self.analyze(dates)
        seasonality = self.analyze_seasonality(dates)
        growth = self.calculate_growth_rate(dates)

        # FEATURE ENGINEERING: Recency - always useful for dates
        recommendations.append(TemporalRecommendation(
            feature_name=f"days_since_{column_name}",
            recommendation_type=TemporalRecommendationType.FEATURE_ENGINEERING,
            category="recency",
            reason="Recency captures how recent an event occurred - useful for predicting behavior",
            priority="medium",
            code_hint=f"(reference_date - df['{column_name}']).dt.days",
        ))

        # FEATURE ENGINEERING: Duration between dates
        if other_date_columns:
            for other_col in other_date_columns:
                recommendations.append(TemporalRecommendation(
                    feature_name=f"days_between_{column_name}_and_{other_col}",
                    recommendation_type=TemporalRecommendationType.FEATURE_ENGINEERING,
                    category="duration",
                    reason="Duration between events captures behavioral patterns (e.g., time to convert)",
                    priority="medium",
                    code_hint=f"(df['{other_col}'] - df['{column_name}']).dt.days",
                ))

        # FEATURE ENGINEERING: Cyclical encoding for seasonality
        if seasonality.has_seasonality and seasonality.seasonal_strength > 0.15:
            priority = "high" if seasonality.seasonal_strength > 0.3 else "medium"
            recommendations.append(TemporalRecommendation(
                feature_name=f"{column_name}_month_sin_cos",
                recommendation_type=TemporalRecommendationType.FEATURE_ENGINEERING,
                category="cyclical",
                reason=f"Seasonality detected (strength: {seasonality.seasonal_strength:.2f}) - cyclical encoding preserves month proximity (Dec near Jan)",
                priority=priority,
                code_hint=f"np.sin(2 * np.pi * df['{column_name}'].dt.month / 12)",
            ))

        # MODELING STRATEGY: Time-based split for trends
        if growth.get("has_data") and abs(growth.get("overall_growth_pct", 0)) > 50:
            direction = growth["trend_direction"]
            pct = growth["overall_growth_pct"]
            recommendations.append(TemporalRecommendation(
                feature_name="time_based_train_test_split",
                recommendation_type=TemporalRecommendationType.MODELING_STRATEGY,
                category="split",
                reason=f"Significant {direction} trend ({pct:+.0f}%) detected - random splits would leak future patterns into training",
                priority="high",
            ))
        elif growth.get("has_data") and abs(growth.get("overall_growth_pct", 0)) > 20:
            recommendations.append(TemporalRecommendation(
                feature_name="time_aware_validation",
                recommendation_type=TemporalRecommendationType.MODELING_STRATEGY,
                category="split",
                reason="Moderate trend detected - time-aware validation ensures model generalizes to future data",
                priority="medium",
            ))

        # FEATURE ENGINEERING: Tenure for long histories
        if analysis.span_days > 365 * 2:
            years = analysis.span_days / 365
            recommendations.append(TemporalRecommendation(
                feature_name=f"tenure_from_{column_name}",
                recommendation_type=TemporalRecommendationType.FEATURE_ENGINEERING,
                category="tenure",
                reason=f"Long history ({years:.1f} years) enables tenure feature - captures customer maturity/loyalty",
                priority="medium",
                code_hint=f"(reference_date - df['{column_name}']).dt.days / 365",
            ))

        # DATA QUALITY: Placeholder dates
        placeholder_count = (valid_dates < "2000-01-01").sum()
        if placeholder_count > 0:
            pct = placeholder_count / len(valid_dates) * 100
            recommendations.append(TemporalRecommendation(
                feature_name=f"{column_name}_placeholder_flag",
                recommendation_type=TemporalRecommendationType.DATA_QUALITY,
                category="filter",
                reason=f"Found {placeholder_count:,} dates before 2000 ({pct:.1f}%) - likely system defaults, not real dates",
                priority="high",
                code_hint=f"df['{column_name}'] < '2000-01-01'",
            ))

        # FEATURE ENGINEERING: Weekend indicator
        dow_counts = valid_dates.dt.dayofweek.value_counts()
        if len(dow_counts) == 7:
            dow_imbalance = dow_counts.max() / dow_counts.min() if dow_counts.min() > 0 else 1
            if dow_imbalance > 1.5:
                weekday_pct = dow_counts[dow_counts.index < 5].sum() / len(valid_dates) * 100
                recommendations.append(TemporalRecommendation(
                    feature_name=f"{column_name}_is_weekend",
                    recommendation_type=TemporalRecommendationType.FEATURE_ENGINEERING,
                    category="extraction",
                    reason=f"Weekday/weekend imbalance ({weekday_pct:.0f}% weekday) suggests behavior differs by day type",
                    priority="low",
                    code_hint=f"df['{column_name}'].dt.dayofweek >= 5",
                ))

        return recommendations
