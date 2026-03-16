from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat.bulk_profiling import DatetimeAnalysisBulkResult
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


class TestGranularityFromSpan:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    def test_short_span_returns_day(self, analyzer):
        assert analyzer._granularity_from_span(30) == TemporalGranularity.DAY

    def test_medium_span_returns_week(self, analyzer):
        assert analyzer._granularity_from_span(200) == TemporalGranularity.WEEK

    def test_two_year_span_returns_month(self, analyzer):
        assert analyzer._granularity_from_span(600) == TemporalGranularity.MONTH

    def test_five_year_span_returns_quarter(self, analyzer):
        assert analyzer._granularity_from_span(1500) == TemporalGranularity.QUARTER

    def test_long_span_returns_year(self, analyzer):
        assert analyzer._granularity_from_span(3000) == TemporalGranularity.YEAR

    def test_zero_span_returns_day(self, analyzer):
        assert analyzer._granularity_from_span(0) == TemporalGranularity.DAY


class TestPeriodCountsFromMonthly:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    @pytest.fixture
    def monthly_data(self):
        return [("2023-01", 100), ("2023-02", 150), ("2023-03", 120),
                ("2023-04", 80), ("2023-05", 200), ("2023-06", 110)]

    def test_month_granularity_passthrough(self, analyzer, monthly_data):
        result = analyzer._period_counts_from_monthly(monthly_data, TemporalGranularity.MONTH)
        assert len(result) == 6
        assert list(result["period"]) == ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06"]
        assert result["count"].sum() == 760

    def test_quarter_aggregation(self, analyzer, monthly_data):
        result = analyzer._period_counts_from_monthly(monthly_data, TemporalGranularity.QUARTER)
        assert len(result) == 2
        assert "2023-Q1" in list(result["period"])
        assert "2023-Q2" in list(result["period"])
        assert result["count"].sum() == 760

    def test_year_aggregation(self, analyzer):
        data = [("2022-06", 50), ("2022-12", 60), ("2023-01", 100), ("2023-06", 90)]
        result = analyzer._period_counts_from_monthly(data, TemporalGranularity.YEAR)
        assert len(result) == 2
        assert result["count"].sum() == 300

    def test_empty_monthly_returns_empty(self, analyzer):
        result = analyzer._period_counts_from_monthly([], TemporalGranularity.MONTH)
        assert len(result) == 0

    def test_day_granularity_uses_monthly(self, analyzer, monthly_data):
        result = analyzer._period_counts_from_monthly(monthly_data, TemporalGranularity.DAY)
        assert len(result) == 6


class TestAnalyzeFromBulk:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    def test_returns_temporal_analysis(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(
            total_count=1000, null_count=50,
            min_date=datetime(2022, 1, 1), max_date=datetime(2023, 6, 30),
            span_days=545,
            monthly_counts=[("2022-01", 50), ("2022-06", 60), ("2023-01", 70), ("2023-06", 80)],
            dow_counts=[140, 150, 130, 160, 120, 80, 70],
        )
        result = analyzer.analyze_from_bulk(bulk)
        assert isinstance(result, TemporalAnalysis)
        assert result.granularity == TemporalGranularity.MONTH
        assert result.span_days == 545
        assert result.total_count == 1000
        assert result.null_count == 50
        assert len(result.period_counts) == 4

    def test_null_dates_return_empty(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(total_count=100, null_count=100)
        result = analyzer.analyze_from_bulk(bulk)
        assert result.span_days == 0
        assert pd.isna(result.min_date)
        assert len(result.period_counts) == 0

    def test_long_span_uses_quarter(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(
            total_count=5000, null_count=0,
            min_date=datetime(2019, 1, 1), max_date=datetime(2023, 12, 31),
            span_days=1826,
            monthly_counts=[(f"{y}-{m:02d}", 80) for y in range(2019, 2024) for m in range(1, 13)],
        )
        result = analyzer.analyze_from_bulk(bulk)
        assert result.granularity in (TemporalGranularity.QUARTER, TemporalGranularity.YEAR)

    def test_short_span_caps_to_month(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(
            total_count=50, null_count=0,
            min_date=datetime(2023, 1, 1), max_date=datetime(2023, 2, 28),
            span_days=58,
            monthly_counts=[("2023-01", 25), ("2023-02", 25)],
        )
        result = analyzer.analyze_from_bulk(bulk)
        assert result.granularity == TemporalGranularity.MONTH


class TestCalculateGrowthRateFromBulk:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    def test_growing_trend(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(
            total_count=7800, null_count=0,
            min_date=datetime(2023, 1, 1), max_date=datetime(2023, 12, 31),
            span_days=364,
            monthly_counts=[(f"2023-{m:02d}", m * 100) for m in range(1, 13)],
        )
        result = analyzer.calculate_growth_rate_from_bulk(bulk)
        assert result["has_data"] is True
        assert result["trend_direction"] == "growing"
        assert result["trend_slope"] > 0
        assert result["overall_growth_pct"] > 0
        assert "cumulative" in result

    def test_declining_trend(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(
            monthly_counts=[(f"2023-{m:02d}", (13 - m) * 100) for m in range(1, 13)],
        )
        result = analyzer.calculate_growth_rate_from_bulk(bulk)
        assert result["trend_direction"] == "declining"
        assert result["overall_growth_pct"] < 0

    def test_insufficient_data(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(monthly_counts=[("2023-01", 100)])
        result = analyzer.calculate_growth_rate_from_bulk(bulk)
        assert result["has_data"] is False

    def test_empty_monthly_counts(self, analyzer):
        bulk = DatetimeAnalysisBulkResult()
        result = analyzer.calculate_growth_rate_from_bulk(bulk)
        assert result["has_data"] is False


class TestAnalyzeSeasonalityFromBulk:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    def test_detects_weekly_seasonality(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(
            total_count=1000, null_count=0,
            dow_counts=[200, 190, 180, 195, 185, 30, 20],
            monthly_counts=[(f"2023-{m:02d}", 80) for m in range(1, 13)],
        )
        result = analyzer.analyze_seasonality_from_bulk(bulk)
        assert result.has_seasonality is True
        assert result.dominant_period == "weekly"
        assert result.seasonal_strength > 0.15

    def test_insufficient_data(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(total_count=20, null_count=0, dow_counts=[3] * 7)
        result = analyzer.analyze_seasonality_from_bulk(bulk)
        assert result.has_seasonality is False

    def test_returns_peak_and_trough_months(self, analyzer):
        monthly = [(f"2023-{m:02d}", m * 10) for m in range(1, 13)]
        bulk = DatetimeAnalysisBulkResult(
            total_count=780, null_count=0,
            dow_counts=[110, 112, 108, 115, 111, 112, 112],
            monthly_counts=monthly,
        )
        result = analyzer.analyze_seasonality_from_bulk(bulk)
        assert len(result.peak_periods) > 0
        assert len(result.trough_periods) > 0

    def test_weekly_pattern_has_7_values(self, analyzer):
        bulk = DatetimeAnalysisBulkResult(
            total_count=100, null_count=0,
            dow_counts=[15, 14, 16, 13, 15, 14, 13],
            monthly_counts=[("2023-01", 50), ("2023-02", 50)],
        )
        result = analyzer.analyze_seasonality_from_bulk(bulk)
        assert len(result.weekly_pattern) == 7

    def test_monthly_pivot_shape(self, analyzer):
        monthly = [(f"{y}-{m:02d}", 10) for y in [2022, 2023] for m in range(1, 13)]
        bulk = DatetimeAnalysisBulkResult(
            total_count=240, null_count=0,
            dow_counts=[34, 34, 34, 34, 34, 35, 35],
            monthly_counts=monthly,
        )
        result = analyzer.analyze_seasonality_from_bulk(bulk)
        assert result.monthly_pattern is not None
        assert len(result.monthly_pattern) == 2


class TestAnalyzeAllFromBulk:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    @pytest.fixture
    def multi_year_bulk(self):
        monthly = [(f"{y}-{m:02d}", m * 10 + (y - 2020) * 5)
                   for y in range(2020, 2024) for m in range(1, 13)]
        total = sum(c for _, c in monthly)
        return DatetimeAnalysisBulkResult(
            total_count=total + 50, null_count=50,
            min_date=datetime(2020, 1, 1), max_date=datetime(2023, 12, 31),
            span_days=1460,
            monthly_counts=monthly,
            dow_counts=[500, 480, 510, 490, 520, 200, 180],
            placeholder_count=10,
        )

    def test_returns_four_tuple(self, analyzer, multi_year_bulk):
        result = analyzer.analyze_all_from_bulk(multi_year_bulk, "created_at")
        assert len(result) == 4
        analysis, growth, seasonality, recommendations = result
        assert isinstance(analysis, TemporalAnalysis)
        assert isinstance(seasonality, SeasonalityResult)
        assert isinstance(growth, dict)
        assert isinstance(recommendations, list)

    def test_includes_recency_recommendation(self, analyzer, multi_year_bulk):
        _, _, _, recs = analyzer.analyze_all_from_bulk(multi_year_bulk, "signup_date")
        categories = [r.category for r in recs]
        assert "recency" in categories

    def test_includes_tenure_for_long_span(self, analyzer, multi_year_bulk):
        _, _, _, recs = analyzer.analyze_all_from_bulk(multi_year_bulk, "account_created")
        has_tenure = any("tenure" in r.feature_name for r in recs)
        assert has_tenure

    def test_includes_placeholder_quality_rec(self, analyzer, multi_year_bulk):
        _, _, _, recs = analyzer.analyze_all_from_bulk(multi_year_bulk, "date_col")
        quality_recs = [r for r in recs if r.recommendation_type == TemporalRecommendationType.DATA_QUALITY]
        assert len(quality_recs) >= 1
        assert "placeholder" in quality_recs[0].feature_name

    def test_includes_duration_with_other_cols(self, analyzer, multi_year_bulk):
        _, _, _, recs = analyzer.analyze_all_from_bulk(
            multi_year_bulk, "start_date", other_date_columns=["end_date"]
        )
        has_duration = any(r.category == "duration" for r in recs)
        assert has_duration

    def test_includes_weekend_rec_with_imbalance(self, analyzer, multi_year_bulk):
        _, _, _, recs = analyzer.analyze_all_from_bulk(multi_year_bulk, "activity_date")
        weekend_recs = [r for r in recs if r.category == "extraction"]
        assert len(weekend_recs) >= 1

    def test_parity_with_series_methods(self, analyzer):
        dates = []
        for year in range(2020, 2024):
            for month in range(1, 13):
                for day in range(1, 28):
                    d = datetime(year, month, day)
                    count = 3 if d.weekday() < 5 else 1
                    dates.extend([d] * count)
        series = pd.Series(dates)

        series_analysis = analyzer.analyze(series)
        series_growth = analyzer.calculate_growth_rate(series)
        series_seasonality = analyzer.analyze_seasonality(series)
        series_recs = analyzer.recommend_features(series, "test_col")

        from customer_retention.core.compat.bulk_profiling import bulk_datetime_analysis_stats
        df = pd.DataFrame({"test_col": series})
        bulk_stats = bulk_datetime_analysis_stats(df, ["test_col"])["test_col"]
        bulk_analysis, bulk_growth, bulk_seasonality, bulk_recs = analyzer.analyze_all_from_bulk(
            bulk_stats, "test_col"
        )

        assert series_analysis.span_days == bulk_analysis.span_days
        assert series_analysis.total_count == bulk_analysis.total_count
        assert series_analysis.null_count == bulk_analysis.null_count
        assert series_growth["trend_direction"] == bulk_growth["trend_direction"]
        assert series_seasonality.has_seasonality == bulk_seasonality.has_seasonality
        assert abs(series_growth["overall_growth_pct"] - bulk_growth["overall_growth_pct"]) < 0.01

        series_cats = sorted(set(r.category for r in series_recs))
        bulk_cats = sorted(set(r.category for r in bulk_recs))
        assert series_cats == bulk_cats


class TestBuildRecommendations:
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()

    def test_empty_valid_count_returns_empty(self, analyzer):
        analysis = TemporalAnalysis(
            granularity=TemporalGranularity.MONTH, min_date=pd.NaT,
            max_date=pd.NaT, span_days=0, total_count=0, null_count=0,
            period_counts=pd.DataFrame(),
        )
        seasonality = SeasonalityResult(has_seasonality=False)
        recs = analyzer._build_recommendations(
            "col", analysis, seasonality, {"has_data": False},
            valid_count=0, placeholder_count=0, dow_counts=[0] * 7,
        )
        assert recs == []

    def test_always_includes_recency(self, analyzer):
        analysis = TemporalAnalysis(
            granularity=TemporalGranularity.MONTH,
            min_date=datetime(2023, 1, 1), max_date=datetime(2023, 6, 30),
            span_days=180, total_count=100, null_count=0,
            period_counts=pd.DataFrame({"period": ["2023-01"], "count": [100]}),
        )
        seasonality = SeasonalityResult(has_seasonality=False)
        recs = analyzer._build_recommendations(
            "test_date", analysis, seasonality, {"has_data": False},
            valid_count=100, placeholder_count=0, dow_counts=[15, 14, 15, 14, 15, 14, 13],
        )
        assert any(r.category == "recency" for r in recs)

    def test_placeholder_generates_quality_rec(self, analyzer):
        analysis = TemporalAnalysis(
            granularity=TemporalGranularity.MONTH,
            min_date=datetime(2023, 1, 1), max_date=datetime(2023, 12, 31),
            span_days=364, total_count=110, null_count=0,
            period_counts=pd.DataFrame(),
        )
        recs = analyzer._build_recommendations(
            "dt", analysis, SeasonalityResult(has_seasonality=False),
            {"has_data": False}, valid_count=110, placeholder_count=10,
            dow_counts=[15] * 7,
        )
        quality = [r for r in recs if r.recommendation_type == TemporalRecommendationType.DATA_QUALITY]
        assert len(quality) == 1
        assert "10" in quality[0].reason
