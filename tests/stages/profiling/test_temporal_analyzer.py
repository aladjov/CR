from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_analyzer import (
    SeasonalityResult,
    TemporalAnalysis,
    TemporalAnalyzer,
    TemporalGranularity,
    TemporalRecommendation,
    TemporalRecommendationType,
)


class TestTemporalGranularity:
    def test_granularity_values(self):
        assert TemporalGranularity.DAY.value == "day"
        assert TemporalGranularity.WEEK.value == "week"
        assert TemporalGranularity.MONTH.value == "month"
        assert TemporalGranularity.QUARTER.value == "quarter"
        assert TemporalGranularity.YEAR.value == "year"


class TestTemporalAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    @pytest.fixture
    def daily_dates(self):
        """Generate 90 days of dates."""
        base = datetime(2023, 1, 1)
        return pd.Series([base + timedelta(days=i) for i in range(90)])

    @pytest.fixture
    def monthly_dates(self):
        """Generate 18 months of dates (within 2-year threshold for monthly)."""
        dates = pd.date_range(start="2023-01-01", end="2024-06-30", freq="MS")
        return pd.Series(dates)

    @pytest.fixture
    def yearly_dates(self):
        """Generate 10 years of dates spread across years."""
        dates = []
        for year in range(2010, 2024):
            dates.extend([datetime(year, m, 15) for m in range(1, 13)])
        return pd.Series(dates)

    def test_detect_granularity_short_span_uses_day(self, analyzer, daily_dates):
        granularity = analyzer.detect_granularity(daily_dates)
        assert granularity == TemporalGranularity.DAY

    def test_detect_granularity_medium_span_uses_month(self, analyzer, monthly_dates):
        granularity = analyzer.detect_granularity(monthly_dates)
        assert granularity == TemporalGranularity.MONTH

    def test_detect_granularity_long_span_uses_year(self, analyzer, yearly_dates):
        granularity = analyzer.detect_granularity(yearly_dates)
        assert granularity in [TemporalGranularity.QUARTER, TemporalGranularity.YEAR]

    def test_aggregate_by_granularity_day(self, analyzer, daily_dates):
        result = analyzer.aggregate_by_granularity(daily_dates, TemporalGranularity.DAY)
        assert len(result) == 90
        assert "period" in result.columns
        assert "count" in result.columns

    def test_aggregate_by_granularity_month(self, analyzer, daily_dates):
        result = analyzer.aggregate_by_granularity(daily_dates, TemporalGranularity.MONTH)
        assert len(result) <= 4  # Jan, Feb, Mar, Apr at most
        assert result["count"].sum() == 90

    def test_aggregate_by_granularity_year(self, analyzer, yearly_dates):
        result = analyzer.aggregate_by_granularity(yearly_dates, TemporalGranularity.YEAR)
        assert len(result) == 14  # 2010-2023
        assert all(result["count"] == 12)  # 12 months per year

    def test_analyze_returns_temporal_analysis(self, analyzer, monthly_dates):
        analysis = analyzer.analyze(monthly_dates)
        assert isinstance(analysis, TemporalAnalysis)
        assert analysis.granularity in TemporalGranularity
        assert analysis.min_date is not None
        assert analysis.max_date is not None
        assert analysis.span_days > 0
        assert len(analysis.period_counts) > 0

    def test_analyze_handles_nulls(self, analyzer):
        dates = pd.Series([
            datetime(2023, 1, 1),
            None,
            datetime(2023, 2, 1),
            pd.NaT,
            datetime(2023, 3, 1),
        ])
        analysis = analyzer.analyze(dates)
        assert analysis.null_count == 2
        assert analysis.total_count == 5

    def test_analyze_with_forced_granularity(self, analyzer, daily_dates):
        analysis = analyzer.analyze(daily_dates, granularity=TemporalGranularity.WEEK)
        assert analysis.granularity == TemporalGranularity.WEEK
        assert len(analysis.period_counts) <= 14  # ~13 weeks in 90 days

    def test_period_labels_are_readable(self, analyzer, monthly_dates):
        analysis = analyzer.analyze(monthly_dates, granularity=TemporalGranularity.MONTH)
        # Month labels should be like "2023-01"
        assert all("-" in str(label) for label in analysis.period_counts["period"])

    def test_year_labels_are_integers(self, analyzer, yearly_dates):
        analysis = analyzer.analyze(yearly_dates, granularity=TemporalGranularity.YEAR)
        # Year labels should be integers like 2023
        assert all(isinstance(label, (int, np.integer)) or label.isdigit()
                   for label in analysis.period_counts["period"].astype(str))


class TestTemporalAnalysis:
    def test_temporal_analysis_dataclass(self):
        analysis = TemporalAnalysis(
            granularity=TemporalGranularity.MONTH,
            min_date=datetime(2023, 1, 1),
            max_date=datetime(2023, 12, 31),
            span_days=365,
            total_count=1000,
            null_count=50,
            period_counts=pd.DataFrame({"period": ["2023-01"], "count": [100]}),
        )
        assert analysis.granularity == TemporalGranularity.MONTH
        assert analysis.span_days == 365
        assert analysis.null_percentage == 5.0

    def test_null_percentage_calculation(self):
        analysis = TemporalAnalysis(
            granularity=TemporalGranularity.DAY,
            min_date=datetime(2023, 1, 1),
            max_date=datetime(2023, 1, 31),
            span_days=30,
            total_count=100,
            null_count=25,
            period_counts=pd.DataFrame(),
        )
        assert analysis.null_percentage == 25.0


class TestSeasonalityAnalysis:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    @pytest.fixture
    def multi_year_dates(self):
        return pd.Series(pd.date_range("2020-01-01", "2023-12-31", freq="D"))

    def test_returns_seasonality_result(self, analyzer, multi_year_dates):
        result = analyzer.analyze_seasonality(multi_year_dates)
        from customer_retention.stages.profiling.temporal_analyzer import SeasonalityResult
        assert isinstance(result, SeasonalityResult)

    def test_detects_peak_months(self, analyzer, multi_year_dates):
        result = analyzer.analyze_seasonality(multi_year_dates)
        assert len(result.peak_periods) > 0
        assert all(m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                  for m in result.peak_periods)

    def test_detects_trough_months(self, analyzer, multi_year_dates):
        result = analyzer.analyze_seasonality(multi_year_dates)
        assert len(result.trough_periods) > 0

    def test_returns_weekly_pattern(self, analyzer, multi_year_dates):
        result = analyzer.analyze_seasonality(multi_year_dates)
        assert result.weekly_pattern is not None
        assert len(result.weekly_pattern) == 7

    def test_returns_monthly_pattern_pivot(self, analyzer, multi_year_dates):
        result = analyzer.analyze_seasonality(multi_year_dates)
        assert result.monthly_pattern is not None
        assert len(result.monthly_pattern) == 4  # 4 years

    def test_handles_insufficient_data(self, analyzer):
        short_dates = pd.Series([datetime(2023, 1, i) for i in range(1, 10)])
        result = analyzer.analyze_seasonality(short_dates)
        assert result.has_seasonality is False
        assert result.confidence == 0.0


class TestYearOverYearComparison:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    def test_returns_pivot_dataframe(self, analyzer):
        dates = pd.Series(pd.date_range("2020-01-01", "2023-12-31", freq="D"))
        result = analyzer.year_over_year_comparison(dates)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 4 years as rows

    def test_columns_are_months(self, analyzer):
        dates = pd.Series(pd.date_range("2022-01-01", "2023-12-31", freq="D"))
        result = analyzer.year_over_year_comparison(dates)
        expected_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        assert list(result.columns) == expected_months

    def test_handles_empty_series(self, analyzer):
        result = analyzer.year_over_year_comparison(pd.Series([], dtype="datetime64[ns]"))
        assert result.empty


class TestGrowthRateCalculation:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    @pytest.fixture
    def growing_dates(self):
        # More records in later months
        dates = []
        for month in range(1, 13):
            count = month * 100  # 100 in Jan, 1200 in Dec
            dates.extend([datetime(2023, month, 15)] * count)
        return pd.Series(dates)

    def test_returns_growth_dict(self, analyzer, growing_dates):
        result = analyzer.calculate_growth_rate(growing_dates)
        assert result["has_data"] is True
        assert "trend_direction" in result
        assert "overall_growth_pct" in result
        assert "avg_monthly_growth" in result

    def test_detects_growing_trend(self, analyzer, growing_dates):
        result = analyzer.calculate_growth_rate(growing_dates)
        assert result["trend_direction"] == "growing"
        assert result["trend_slope"] > 0

    def test_detects_declining_trend(self, analyzer):
        # More records in earlier months
        dates = []
        for month in range(1, 13):
            count = (13 - month) * 100
            dates.extend([datetime(2023, month, 15)] * count)
        result = analyzer.calculate_growth_rate(pd.Series(dates))
        assert result["trend_direction"] == "declining"
        assert result["trend_slope"] < 0

    def test_returns_cumulative_series(self, analyzer, growing_dates):
        result = analyzer.calculate_growth_rate(growing_dates)
        assert "cumulative" in result
        cumulative = result["cumulative"]
        assert cumulative.is_monotonic_increasing

    def test_handles_insufficient_data(self, analyzer):
        result = analyzer.calculate_growth_rate(pd.Series([datetime(2023, 1, 1)]))
        assert result["has_data"] is False


class TestTemporalRecommendations:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    @pytest.fixture
    def multi_year_dates(self):
        return pd.Series(pd.date_range("2020-01-01", "2023-12-31", freq="D"))

    @pytest.fixture
    def growing_dates(self):
        dates = []
        for month in range(1, 13):
            count = month * 100
            dates.extend([datetime(2023, month, 15)] * count)
        return pd.Series(dates)

    def test_recommend_features_returns_list(self, analyzer, multi_year_dates):
        from customer_retention.stages.profiling.temporal_analyzer import TemporalRecommendation
        recs = analyzer.recommend_features(multi_year_dates, "signup_date")
        assert isinstance(recs, list)
        assert all(isinstance(r, TemporalRecommendation) for r in recs)

    def test_recommend_features_includes_recency(self, analyzer, multi_year_dates):
        recs = analyzer.recommend_features(multi_year_dates, "signup_date")
        categories = [r.category for r in recs]
        assert "recency" in categories

    def test_recommend_features_detects_seasonality(self, analyzer):
        # Create data with actual weekly seasonality (more records on weekdays)
        dates = []
        for year in range(2020, 2024):
            for month in range(1, 13):
                for day in range(1, 28):
                    d = datetime(year, month, day)
                    # Add more records on weekdays (Monday=0 to Friday=4)
                    count = 3 if d.weekday() < 5 else 1
                    dates.extend([d] * count)
        seasonal_dates = pd.Series(dates)
        recs = analyzer.recommend_features(seasonal_dates, "transaction_date")
        has_cyclical = any(r.category == "cyclical" for r in recs)
        assert has_cyclical

    def test_recommend_features_with_strong_trend(self, analyzer, growing_dates):
        recs = analyzer.recommend_features(growing_dates, "order_date")
        # Strong trend should recommend time-based split
        has_trend_warning = any("split" in r.reason.lower() or "trend" in r.reason.lower()
                               for r in recs)
        assert has_trend_warning

    def test_recommend_features_for_multiple_columns(self, analyzer):
        dates1 = pd.Series(pd.date_range("2020-01-01", "2023-12-31", freq="D"))
        dates2 = pd.Series(pd.date_range("2021-01-01", "2023-06-30", freq="D"))
        recs = analyzer.recommend_features(dates1, "first_date", other_date_columns=["last_date"])
        has_duration = any(r.category == "duration" for r in recs)
        assert has_duration

    def test_recommend_features_detects_placeholders(self, analyzer):
        from customer_retention.stages.profiling.temporal_analyzer import TemporalRecommendationType
        dates = pd.Series([datetime(2023, 1, 1)] * 100 + [datetime(1900, 1, 1)] * 10)
        recs = analyzer.recommend_features(dates, "created_at")
        # Should detect data quality issue with placeholder dates
        has_data_quality = any(r.recommendation_type == TemporalRecommendationType.DATA_QUALITY
                               for r in recs)
        assert has_data_quality

    def test_recommendation_to_dict(self, analyzer, multi_year_dates):
        recs = analyzer.recommend_features(multi_year_dates, "test_date")
        for rec in recs:
            d = rec.to_dict()
            assert "feature_name" in d
            assert "recommendation_type" in d
            assert "category" in d
            assert "reason" in d
            assert "priority" in d

    def test_recommend_features_long_history_suggests_tenure(self, analyzer):
        dates = pd.Series(pd.date_range("2015-01-01", "2023-12-31", freq="D"))
        recs = analyzer.recommend_features(dates, "account_created")
        has_tenure = any("tenure" in r.feature_name.lower() for r in recs)
        assert has_tenure


class TestActionDescription:

    def test_feature_engineering_description(self):
        rec = TemporalRecommendation(
            feature_name="days_since_signup",
            recommendation_type=TemporalRecommendationType.FEATURE_ENGINEERING,
            category="recency",
            reason="test",
            priority="medium",
        )
        assert rec.action_description == "Create feature: days_since_signup"

    def test_modeling_strategy_description(self):
        rec = TemporalRecommendation(
            feature_name="time_based_split",
            recommendation_type=TemporalRecommendationType.MODELING_STRATEGY,
            category="split",
            reason="test",
            priority="high",
        )
        assert rec.action_description == "Modeling: time_based_split"

    def test_data_quality_description(self):
        rec = TemporalRecommendation(
            feature_name="placeholder_flag",
            recommendation_type=TemporalRecommendationType.DATA_QUALITY,
            category="filter",
            reason="test",
            priority="high",
        )
        assert rec.action_description == "Data quality: placeholder_flag"


class TestEdgeCases:

    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    def test_detect_granularity_empty_dates(self, analyzer):
        dates = pd.Series([], dtype="datetime64[ns]")
        assert analyzer.detect_granularity(dates) == TemporalGranularity.MONTH

    def test_detect_granularity_all_nat(self, analyzer):
        dates = pd.Series([pd.NaT, pd.NaT])
        assert analyzer.detect_granularity(dates) == TemporalGranularity.MONTH

    def test_aggregate_by_granularity_empty(self, analyzer):
        dates = pd.Series([], dtype="datetime64[ns]")
        result = analyzer.aggregate_by_granularity(dates, TemporalGranularity.MONTH)
        assert result.empty

    def test_analyze_all_nulls(self, analyzer):
        dates = pd.Series([pd.NaT, None, pd.NaT])
        analysis = analyzer.analyze(dates)
        assert analysis.span_days == 0
        assert pd.isna(analysis.min_date)
        assert pd.isna(analysis.max_date)
        assert analysis.null_count == 3
        assert analysis.total_count == 3

    def test_null_percentage_zero_total(self):
        analysis = TemporalAnalysis(
            granularity=TemporalGranularity.MONTH,
            min_date=pd.NaT,
            max_date=pd.NaT,
            span_days=0,
            total_count=0,
            null_count=0,
            period_counts=pd.DataFrame(),
        )
        assert analysis.null_percentage == 0.0

    def test_growth_rate_single_month_data(self, analyzer):
        dates = pd.Series([datetime(2023, 1, 15)] * 100)
        result = analyzer.calculate_growth_rate(dates)
        assert result["has_data"] is False

    def test_growth_rate_zero_first_month(self, analyzer):
        # Two months with first month having 0 records - impossible directly
        # but test with first month count = 0 via small data
        dates = pd.Series([datetime(2023, 1, 1), datetime(2023, 3, 1)])
        result = analyzer.calculate_growth_rate(dates)
        assert result["has_data"] is True

    def test_recommend_features_empty_dates(self, analyzer):
        dates = pd.Series([], dtype="datetime64[ns]")
        recs = analyzer.recommend_features(dates, "test_col")
        assert recs == []

    def test_recommend_features_all_nat(self, analyzer):
        dates = pd.Series([pd.NaT, pd.NaT])
        recs = analyzer.recommend_features(dates, "test_col")
        assert recs == []

    def test_recommend_features_moderate_trend(self, analyzer):
        # Moderate growth (20-50%) should recommend time_aware_validation
        dates = []
        for month in range(1, 13):
            # Moderate growth: 100 -> 130 over the year
            count = 100 + month * 3
            dates.extend([datetime(2023, month, 15)] * count)
        recs = analyzer.recommend_features(pd.Series(dates), "order_date")
        split_recs = [r for r in recs if r.category == "split"]
        if split_recs:
            assert any("time_aware_validation" in r.feature_name for r in split_recs)

    def test_extract_period_quarter(self, analyzer):
        dates = pd.Series(pd.date_range("2020-01-01", "2023-12-31", freq="MS"))
        parsed = pd.to_datetime(dates)
        result = analyzer._extract_period(parsed, TemporalGranularity.QUARTER)
        assert len(result) == len(dates)

    def test_extract_period_week(self, analyzer):
        dates = pd.Series(pd.date_range("2023-01-01", periods=30, freq="D"))
        parsed = pd.to_datetime(dates)
        result = analyzer._extract_period(parsed, TemporalGranularity.WEEK)
        assert len(result) == 30

    def test_seasonality_low_month_count(self, analyzer):
        # Enough data but only 2 distinct months
        dates = pd.Series(
            pd.date_range("2023-01-01", periods=40, freq="D")
        )
        result = analyzer.analyze_seasonality(dates)
        # Should still produce peak/trough if >= 3 months
        assert isinstance(result, SeasonalityResult)

    def test_weekend_imbalance_recommendation(self, analyzer):
        # Create data with strong weekday bias
        dates = []
        for d in pd.date_range("2022-01-01", "2023-12-31", freq="D"):
            if d.weekday() < 5:
                dates.extend([d] * 10)  # 10 records per weekday
            else:
                dates.extend([d] * 2)   # 2 records per weekend day
        recs = analyzer.recommend_features(pd.Series(dates), "activity_date")
        weekend_recs = [r for r in recs if r.category == "extraction"]
        assert len(weekend_recs) >= 1
        assert "weekend" in weekend_recs[0].feature_name

    def test_seasonality_with_weekly_cv_above_threshold(self, analyzer):
        # Create data with high weekly CV
        dates = []
        for year in range(2020, 2023):
            for month in range(1, 13):
                for day in range(1, 28):
                    d = datetime(year, month, day)
                    if d.weekday() < 5:
                        dates.extend([d] * 5)
                    else:
                        dates.extend([d] * 1)
        result = analyzer.analyze_seasonality(pd.Series(dates))
        assert result.has_seasonality
        assert result.dominant_period == "weekly"
        assert result.confidence > 0

    def test_growth_rate_declining_overall(self, analyzer):
        dates = []
        for month in range(1, 7):
            count = 200 - month * 30
            dates.extend([datetime(2023, month, 15)] * max(count, 1))
        result = analyzer.calculate_growth_rate(pd.Series(dates))
        assert result["has_data"] is True
        assert result["overall_growth_pct"] < 0
