from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import Series, Timestamp, head_as_list, native_pd, to_datetime


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
    monthly_pattern: Optional[native_pd.DataFrame] = None
    weekly_pattern: Optional[native_pd.Series] = None
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
    min_date: Timestamp
    max_date: Timestamp
    span_days: int
    total_count: int
    null_count: int
    period_counts: native_pd.DataFrame

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

    def _granularity_from_span(self, span_days: int) -> TemporalGranularity:
        for threshold, granularity in self.GRANULARITY_THRESHOLDS.items():
            if span_days <= threshold:
                return granularity
        return TemporalGranularity.YEAR

    def detect_granularity(self, dates: Series) -> TemporalGranularity:
        clean_dates = to_datetime(dates, errors="coerce").dropna()
        if len(clean_dates) == 0:
            return TemporalGranularity.MONTH
        return self._granularity_from_span((clean_dates.max() - clean_dates.min()).days)

    def aggregate_by_granularity(
        self, dates: Series, granularity: TemporalGranularity
    ) -> native_pd.DataFrame:

        clean_dates = to_datetime(dates, errors="coerce").dropna()
        if len(clean_dates) == 0:
            return native_pd.DataFrame({"period": [], "count": []})

        period_series = self._extract_period(clean_dates, granularity)
        counts = period_series.value_counts().sort_index().reset_index()
        counts.columns = ["period", "count"]
        return counts

    def _extract_period(
        self, dates: Series, granularity: TemporalGranularity
    ) -> Series:
        if granularity == TemporalGranularity.DAY:
            return dates.dt.strftime("%Y-%m-%d")
        elif granularity == TemporalGranularity.WEEK:
            week_num = ((dates.dt.dayofyear - 1) // 7 + 1).astype(str).str.zfill(2)
            return dates.dt.year.astype(str) + "-W" + week_num
        elif granularity == TemporalGranularity.MONTH:
            return dates.dt.strftime("%Y-%m")
        elif granularity == TemporalGranularity.QUARTER:
            return dates.dt.year.astype(str) + "-Q" + dates.dt.quarter.astype(str)
        else:  # YEAR
            return dates.dt.year

    def analyze(
        self,
        dates: Series,
        granularity: Optional[TemporalGranularity] = None,
    ) -> TemporalAnalysis:

        total_count = len(dates)
        parsed_dates = to_datetime(dates, errors="coerce")
        null_count = parsed_dates.isna().sum()
        clean_dates = parsed_dates.dropna()

        if len(clean_dates) == 0:
            return TemporalAnalysis(
                granularity=granularity or TemporalGranularity.MONTH,
                min_date=native_pd.NaT,
                max_date=native_pd.NaT,
                span_days=0,
                total_count=total_count,
                null_count=null_count,
                period_counts=native_pd.DataFrame({"period": [], "count": []}),
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

        parsed = to_datetime(dates, errors="coerce").dropna()

        if len(parsed) < 30:
            return SeasonalityResult(has_seasonality=False, confidence=0.0)

        # Weekly pattern (day of week)
        dow_counts = parsed.dt.dayofweek.value_counts().sort_index()
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekly_pattern = native_pd.Series(
            [dow_counts.get(i, 0) for i in range(7)], index=dow_names
        )

        # Monthly pattern (year x month heatmap)
        monthly_df = parsed.to_frame(name="date")
        monthly_df["year"] = parsed.dt.year
        monthly_df["month"] = parsed.dt.month
        monthly_pivot = monthly_df.groupby(["year", "month"]).size().unstack().fillna(0)
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
            peak_months = head_as_list(monthly_totals.nlargest(3).index, 3)
            trough_months = head_as_list(monthly_totals.nsmallest(3).index, 3)
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

    def year_over_year_comparison(self, dates: Series) -> native_pd.DataFrame:

        parsed = to_datetime(dates, errors="coerce").dropna()

        if len(parsed) == 0:
            return native_pd.DataFrame()

        df = parsed.to_frame(name="date")
        df["year"] = parsed.dt.year
        df["month"] = parsed.dt.month

        pivot = df.groupby(["year", "month"]).size().unstack().fillna(0)
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ][:len(pivot.columns)]

        return pivot

    def calculate_growth_rate(self, dates: Series) -> Dict[str, Any]:
        """Calculate growth metrics over time."""

        parsed = to_datetime(dates, errors="coerce").dropna()

        if len(parsed) < 2:
            return {"has_data": False}

        monthly = parsed.dt.strftime("%Y-%m").value_counts().sort_index()

        if len(monthly) < 2:
            return {"has_data": False}

        mom_growth = monthly.pct_change().dropna()
        cumulative = monthly.cumsum()

        x = np.arange(len(monthly))
        y = monthly.to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        trend_direction = "growing" if slope > 0 else "declining"

        overall_growth = ((y[-1] - y[0]) / y[0] * 100) if y[0] > 0 else 0

        return {
            "has_data": True,
            "monthly_counts": monthly,
            "cumulative": cumulative,
            "avg_monthly_growth": float(mom_growth.mean() * 100),
            "overall_growth_pct": float(overall_growth),
            "trend_direction": trend_direction,
            "trend_slope": float(slope),
        }

    def _build_recommendations(
        self, column_name: str, analysis: TemporalAnalysis,
        seasonality: SeasonalityResult, growth: Dict[str, Any],
        valid_count: int, placeholder_count: int,
        dow_counts: List[int], other_date_columns: Optional[List[str]] = None,
    ) -> List[TemporalRecommendation]:
        if valid_count == 0:
            return []
        recommendations: List[TemporalRecommendation] = []
        recommendations.append(TemporalRecommendation(
            feature_name=f"days_since_{column_name}",
            recommendation_type=TemporalRecommendationType.FEATURE_ENGINEERING,
            category="recency",
            reason="Recency captures how recent an event occurred - useful for predicting behavior",
            priority="medium",
            code_hint=f"(reference_date - df['{column_name}']).dt.days",
        ))
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
        if placeholder_count > 0:
            pct = placeholder_count / valid_count * 100
            recommendations.append(TemporalRecommendation(
                feature_name=f"{column_name}_placeholder_flag",
                recommendation_type=TemporalRecommendationType.DATA_QUALITY,
                category="filter",
                reason=f"Found {placeholder_count:,} dates before 2000 ({pct:.1f}%) - likely system defaults, not real dates",
                priority="high",
                code_hint=f"df['{column_name}'] < '2000-01-01'",
            ))
        if len(dow_counts) == 7 and min(dow_counts) > 0:
            dow_imbalance = max(dow_counts) / min(dow_counts)
            if dow_imbalance > 1.5:
                weekday_pct = sum(dow_counts[:5]) / valid_count * 100
                recommendations.append(TemporalRecommendation(
                    feature_name=f"{column_name}_is_weekend",
                    recommendation_type=TemporalRecommendationType.FEATURE_ENGINEERING,
                    category="extraction",
                    reason=f"Weekday/weekend imbalance ({weekday_pct:.0f}% weekday) suggests behavior differs by day type",
                    priority="low",
                    code_hint=f"df['{column_name}'].dt.dayofweek >= 5",
                ))
        return recommendations

    def recommend_features(
        self, dates: Series, column_name: str, other_date_columns: Optional[List[str]] = None
    ) -> List[TemporalRecommendation]:
        parsed = to_datetime(dates, errors="coerce")
        valid_dates = parsed.dropna()
        if len(valid_dates) == 0:
            return []
        analysis = self.analyze(dates)
        seasonality = self.analyze_seasonality(dates)
        growth = self.calculate_growth_rate(dates)
        placeholder_count = int((valid_dates < "2000-01-01").sum())
        dow_vc = valid_dates.dt.dayofweek.value_counts()
        dow_list = [int(dow_vc.get(i, 0)) for i in range(7)]
        return self._build_recommendations(
            column_name, analysis, seasonality, growth,
            valid_count=len(valid_dates), placeholder_count=placeholder_count,
            dow_counts=dow_list, other_date_columns=other_date_columns,
        )

    # ------------------------------------------------------------------
    # Bulk-aware methods: build results from pre-aggregated stats
    # (no raw Series access — fully distributed on Databricks)
    # ------------------------------------------------------------------

    def _period_counts_from_monthly(
        self, monthly: List, granularity: TemporalGranularity,
    ) -> native_pd.DataFrame:
        if not monthly:
            return native_pd.DataFrame({"period": [], "count": []})
        if granularity in (TemporalGranularity.DAY, TemporalGranularity.WEEK, TemporalGranularity.MONTH):
            return native_pd.DataFrame({"period": [ym for ym, _ in monthly], "count": [c for _, c in monthly]})
        if granularity == TemporalGranularity.QUARTER:
            agg = {}
            for ym, cnt in monthly:
                y, m = ym.split("-")
                key = f"{y}-Q{(int(m) - 1) // 3 + 1}"
                agg[key] = agg.get(key, 0) + cnt
            return native_pd.DataFrame({"period": list(agg.keys()), "count": list(agg.values())})
        agg = {}
        for ym, cnt in monthly:
            agg[ym.split("-")[0]] = agg.get(ym.split("-")[0], 0) + cnt
        return native_pd.DataFrame({"period": list(agg.keys()), "count": list(agg.values())})

    def analyze_from_bulk(self, bulk: Any) -> TemporalAnalysis:
        if bulk.min_date is None:
            return TemporalAnalysis(
                granularity=TemporalGranularity.MONTH, min_date=native_pd.NaT,
                max_date=native_pd.NaT, span_days=0, total_count=bulk.total_count,
                null_count=bulk.null_count,
                period_counts=native_pd.DataFrame({"period": [], "count": []}),
            )
        raw = self._granularity_from_span(bulk.span_days)
        granularity = raw if raw in (
            TemporalGranularity.MONTH, TemporalGranularity.QUARTER, TemporalGranularity.YEAR,
        ) else TemporalGranularity.MONTH
        return TemporalAnalysis(
            granularity=granularity, min_date=bulk.min_date, max_date=bulk.max_date,
            span_days=bulk.span_days, total_count=bulk.total_count,
            null_count=bulk.null_count,
            period_counts=self._period_counts_from_monthly(bulk.monthly_counts, granularity),
        )

    def calculate_growth_rate_from_bulk(self, bulk: Any) -> Dict[str, Any]:
        if len(bulk.monthly_counts) < 2:
            return {"has_data": False}
        monthly = native_pd.Series(dict(bulk.monthly_counts))
        mom_growth = monthly.pct_change().dropna()
        cumulative = monthly.cumsum()
        x = np.arange(len(monthly))
        y = monthly.to_numpy()
        slope, _ = np.polyfit(x, y, 1)
        overall_growth = ((y[-1] - y[0]) / y[0] * 100) if y[0] > 0 else 0
        return {
            "has_data": True, "monthly_counts": monthly, "cumulative": cumulative,
            "avg_monthly_growth": float(mom_growth.mean() * 100),
            "overall_growth_pct": float(overall_growth),
            "trend_direction": "growing" if slope > 0 else "declining",
            "trend_slope": float(slope),
        }

    def analyze_seasonality_from_bulk(self, bulk: Any) -> SeasonalityResult:
        valid_count = bulk.total_count - bulk.null_count
        if valid_count < 30:
            return SeasonalityResult(has_seasonality=False, confidence=0.0)

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekly_pattern = native_pd.Series(bulk.dow_counts, index=dow_names)

        ym_data: Dict[int, Dict[int, int]] = {}
        month_totals: Dict[int, int] = {}
        for ym, cnt in bulk.monthly_counts:
            y, m = int(ym.split("-")[0]), int(ym.split("-")[1])
            ym_data.setdefault(y, {})[m] = cnt
            month_totals[m] = month_totals.get(m, 0) + cnt

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if ym_data:
            all_months = sorted({m for y_dict in ym_data.values() for m in y_dict})
            pivot_data = {y: {m: ym_data[y].get(m, 0) for m in all_months} for y in sorted(ym_data)}
            monthly_pivot = native_pd.DataFrame(pivot_data).T
            monthly_pivot.columns = [month_names[m - 1] for m in all_months]
        else:
            monthly_pivot = native_pd.DataFrame()

        weekly_cv = float(weekly_pattern.std() / weekly_pattern.mean()) if weekly_pattern.mean() > 0 else 0
        monthly_totals_s = native_pd.Series(month_totals).sort_index()

        if len(monthly_totals_s) >= 3:
            peak_months = list(monthly_totals_s.nlargest(3).index)
            trough_months = list(monthly_totals_s.nsmallest(3).index)
            peak_periods = [month_names[m - 1] for m in peak_months if m <= 12]
            trough_periods = [month_names[m - 1] for m in trough_months if m <= 12]
        else:
            peak_periods, trough_periods = [], []

        has_seasonality = weekly_cv > 0.15
        dominant_period = "weekly" if weekly_cv > 0.2 else None
        return SeasonalityResult(
            has_seasonality=has_seasonality, dominant_period=dominant_period,
            peak_periods=peak_periods, trough_periods=trough_periods,
            monthly_pattern=monthly_pivot, weekly_pattern=weekly_pattern,
            confidence=min(1.0, weekly_cv * 2) if has_seasonality else 0.0,
            seasonal_strength=float(weekly_cv),
        )

    def analyze_all_from_bulk(
        self, bulk: Any, column_name: str, other_date_columns: Optional[List[str]] = None,
    ) -> tuple:
        analysis = self.analyze_from_bulk(bulk)
        growth = self.calculate_growth_rate_from_bulk(bulk)
        seasonality = self.analyze_seasonality_from_bulk(bulk)
        valid_count = bulk.total_count - bulk.null_count
        recommendations = self._build_recommendations(
            column_name, analysis, seasonality, growth,
            valid_count=valid_count, placeholder_count=bulk.placeholder_count,
            dow_counts=bulk.dow_counts, other_date_columns=other_date_columns,
        )
        return analysis, growth, seasonality, recommendations
