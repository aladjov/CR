from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.utils import compute_effect_size
from customer_retention.stages.profiling.temporal_pattern_analyzer import (
    CohortDistribution,
    RecencyComparisonResult,
    RecencyResult,
    TemporalPatternAnalysis,
    TemporalPatternAnalyzer,
    TrendDirection,
    TrendRecommendation,
    TrendResult,
    analyze_cohort_distribution,
    compare_recency_by_target,
    compute_group_stats,
    generate_cohort_recommendations,
    generate_trend_recommendations,
)


@pytest.fixture
def trending_up_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    # Clear upward trend: base + day_number * slope
    values = 100 + np.arange(365) * 0.5 + np.random.normal(0, 10, 365)
    return pd.DataFrame({
        "date": dates,
        "value": values,
        "entity": ["E001"] * 365,
    })


@pytest.fixture
def trending_down_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    values = 300 - np.arange(365) * 0.5 + np.random.normal(0, 10, 365)
    return pd.DataFrame({
        "date": dates,
        "value": values,
        "entity": ["E001"] * 365,
    })


@pytest.fixture
def seasonal_weekly_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    # Higher values on weekends
    base = 100
    weekend_boost = [30 if d.dayofweek >= 5 else 0 for d in dates]
    values = base + np.array(weekend_boost) + np.random.normal(0, 5, 365)
    return pd.DataFrame({
        "date": dates,
        "value": values,
        "entity": ["E001"] * 365,
    })


@pytest.fixture
def cohort_data():
    np.random.seed(42)
    data = []

    # Cohort 1: Signed up in Jan 2023, high activity
    for i in range(100):
        signup = pd.Timestamp("2023-01-15")
        n_events = np.random.randint(5, 15)
        for j in range(n_events):
            data.append({
                "customer_id": f"C1_{i:03d}",
                "signup_date": signup,
                "event_date": signup + timedelta(days=np.random.randint(0, 180)),
                "retained": 1,
            })

    # Cohort 2: Signed up in Apr 2023, medium activity
    for i in range(100):
        signup = pd.Timestamp("2023-04-15")
        n_events = np.random.randint(2, 8)
        for j in range(n_events):
            data.append({
                "customer_id": f"C2_{i:03d}",
                "signup_date": signup,
                "event_date": signup + timedelta(days=np.random.randint(0, 120)),
                "retained": np.random.choice([0, 1], p=[0.3, 0.7]),
            })

    return pd.DataFrame(data)


@pytest.fixture
def recency_data():
    np.random.seed(42)
    data = []

    reference_date = pd.Timestamp("2023-12-31")

    for i in range(200):
        # Churned customers: last activity long ago
        if i < 80:
            last_activity = reference_date - timedelta(days=np.random.randint(60, 180))
            retained = 0
        # Retained customers: recent activity
        else:
            last_activity = reference_date - timedelta(days=np.random.randint(1, 30))
            retained = 1

        data.append({
            "customer_id": f"C{i:03d}",
            "last_event_date": last_activity,
            "retained": retained,
        })

    return pd.DataFrame(data)


class TestTrendDetection:

    def test_detects_upward_trend(self, trending_up_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(trending_up_data, value_column="value")

        assert isinstance(result, TrendResult)
        assert result.direction == TrendDirection.INCREASING
        assert result.strength > 0.5

    def test_detects_downward_trend(self, trending_down_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(trending_down_data, value_column="value")

        assert result.direction == TrendDirection.DECREASING
        assert result.strength > 0.5

    def test_stable_data_shows_stable_trend(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.normal(100, 5, 100),  # Constant mean
        })

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(df, value_column="value")

        assert result.direction == TrendDirection.STABLE

    def test_trend_result_has_slope(self, trending_up_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(trending_up_data, value_column="value")

        assert result.slope is not None
        assert result.slope > 0  # Upward slope


class TestSeasonalityDetection:

    def test_detects_weekly_pattern(self, seasonal_weekly_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_seasonality(seasonal_weekly_data, value_column="value")

        assert isinstance(result, list)
        # Should find weekly pattern
        periods = [r.period for r in result]
        assert any(6 <= p <= 8 for p in periods)  # ~7 days

    def test_returns_empty_for_non_seasonal_data(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.normal(100, 5, 365),
        })

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_seasonality(df, value_column="value")

        # Should return empty or low-strength results
        strong_patterns = [r for r in result if r.strength > 0.5]
        assert len(strong_patterns) == 0


class TestCohortAnalysis:

    def test_identifies_cohorts(self, cohort_data):
        analyzer = TemporalPatternAnalyzer(time_column="event_date")
        result = analyzer.analyze_cohorts(
            cohort_data,
            entity_column="customer_id",
            cohort_column="signup_date",
            period="M"
        )

        assert isinstance(result, pd.DataFrame)
        assert "cohort" in result.columns
        assert "entity_count" in result.columns

    def test_cohort_metrics(self, cohort_data):
        analyzer = TemporalPatternAnalyzer(time_column="event_date")
        result = analyzer.analyze_cohorts(
            cohort_data,
            entity_column="customer_id",
            cohort_column="signup_date",
            target_column="retained",
            period="M"
        )

        # Should have retention metrics
        assert "retention_rate" in result.columns or "avg_target" in result.columns


class TestRecencyAnalysis:

    def test_computes_recency_stats(self, recency_data):
        analyzer = TemporalPatternAnalyzer(time_column="last_event_date")
        result = analyzer.analyze_recency(
            recency_data,
            entity_column="customer_id",
            reference_date=pd.Timestamp("2023-12-31")
        )

        assert isinstance(result, RecencyResult)
        assert result.avg_recency_days is not None
        assert result.median_recency_days is not None

    def test_recency_target_correlation(self, recency_data):
        analyzer = TemporalPatternAnalyzer(time_column="last_event_date")
        result = analyzer.analyze_recency(
            recency_data,
            entity_column="customer_id",
            target_column="retained",
            reference_date=pd.Timestamp("2023-12-31")
        )

        # Should show negative correlation (recent = more likely retained)
        assert result.target_correlation is not None
        assert result.target_correlation < 0


class TestFullAnalysis:

    def test_analyze_returns_complete_result(self, trending_up_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.analyze(
            trending_up_data,
            value_column="value"
        )

        assert isinstance(result, TemporalPatternAnalysis)
        assert result.trend is not None

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "value"])

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.analyze(df, value_column="value")

        assert result.trend is None or result.trend.direction == TrendDirection.UNKNOWN


class TestEdgeCases:

    def test_single_row_data(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2023-01-01")],
            "value": [100],
        })

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.analyze(df, value_column="value")

        # Should not crash, return unknown/empty results
        assert result is not None

    def test_handles_null_values(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": [100, None, 102, None, 104, 105, None, 107, 108, 109],
        })

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.analyze(df, value_column="value")

        # Should handle nulls gracefully
        assert result is not None


class TestComputeCohensD:
    def test_large_effect_size(self):
        group1 = np.array([10, 11, 12, 13, 14])
        group2 = np.array([1, 2, 3, 4, 5])
        d, interp = compute_effect_size(group1, group2)
        assert d > 0.8
        assert interp == "Large effect"

    def test_negligible_effect_size(self):
        group1 = np.array([100, 101, 102, 103, 104])
        group2 = np.array([100.1, 101.1, 102.1, 103.1, 104.1])
        d, interp = compute_effect_size(group1, group2)
        assert abs(d) < 0.2
        assert interp == "Negligible"

    def test_zero_variance(self):
        group1 = np.array([5, 5, 5])
        group2 = np.array([5, 5, 5])
        d, interp = compute_effect_size(group1, group2)
        assert d == 0.0
        assert interp == "Negligible"


class TestComputeGroupStats:
    def test_basic_stats(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = compute_group_stats(values)
        assert stats.mean == 5.5
        assert stats.median == 5.5
        assert stats.count == 10
        assert stats.q25 == 3.25
        assert stats.q75 == 7.75


class TestGenerateTrendRecommendations:
    def test_strong_increasing_trend(self):
        trend = TrendResult(direction=TrendDirection.INCREASING, strength=0.5, slope=0.1, p_value=0.001)
        recs = generate_trend_recommendations(trend, mean_value=100.0)
        actions = [r.action for r in recs]
        assert "add_trend_features" in actions
        assert "time_based_split" in actions
        high_priority = [r for r in recs if r.priority == "high"]
        assert len(high_priority) >= 2

    def test_moderate_trend(self):
        trend = TrendResult(direction=TrendDirection.DECREASING, strength=0.2, slope=-0.05, p_value=0.03)
        recs = generate_trend_recommendations(trend, mean_value=100.0)
        actions = [r.action for r in recs]
        assert "add_trend_features" in actions
        medium_priority = [r for r in recs if r.priority == "medium"]
        assert len(medium_priority) >= 1

    def test_stable_trend(self):
        trend = TrendResult(direction=TrendDirection.STABLE, strength=0.01, slope=0.0, p_value=0.8)
        recs = generate_trend_recommendations(trend, mean_value=100.0)
        actions = [r.action for r in recs]
        assert "skip_trend_features" in actions

    def test_returns_trend_recommendation_objects(self):
        trend = TrendResult(direction=TrendDirection.INCREASING, strength=0.5, slope=0.1, p_value=0.001)
        recs = generate_trend_recommendations(trend)
        assert all(isinstance(r, TrendRecommendation) for r in recs)


class TestAnalyzeCohortDistribution:
    def test_basic_distribution(self):
        first_events = pd.DataFrame({
            "entity": ["A", "B", "C", "D", "E"],
            "first_event": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01", "2021-12-01"])
        })
        dist = analyze_cohort_distribution(first_events, "first_event")
        assert isinstance(dist, CohortDistribution)
        assert dist.total_entities == 5
        assert dist.num_years == 2
        assert 2021 in dist.year_counts

    def test_dominant_year_detection(self):
        first_events = pd.DataFrame({
            "entity": list("ABCDEFGHIJ"),
            "first_event": pd.to_datetime(["2020-01-01"]*8 + ["2021-01-01"]*2)
        })
        dist = analyze_cohort_distribution(first_events, "first_event")
        assert dist.dominant_year == 2020
        assert dist.dominant_pct == 80.0


class TestGenerateCohortRecommendations:
    def test_skewed_distribution_skips_features(self):
        dist = CohortDistribution(
            year_counts={2020: 900, 2021: 100}, total_entities=1000,
            dominant_year=2020, dominant_pct=90.0, num_years=2
        )
        recs = generate_cohort_recommendations(dist)
        actions = [r.action for r in recs]
        assert "skip_cohort_features" in actions

    def test_varied_distribution_adds_features(self):
        dist = CohortDistribution(
            year_counts={2019: 300, 2020: 350, 2021: 350}, total_entities=1000,
            dominant_year=2020, dominant_pct=35.0, num_years=3
        )
        recs = generate_cohort_recommendations(dist)
        actions = [r.action for r in recs]
        assert "add_cohort_features" in actions

    def test_retention_variation_adds_recommendation(self):
        dist = CohortDistribution(
            year_counts={2020: 500, 2021: 500}, total_entities=1000,
            dominant_year=2020, dominant_pct=50.0, num_years=2
        )
        recs = generate_cohort_recommendations(dist, retention_variation=0.15)
        actions = [r.action for r in recs]
        assert "investigate_cohort_retention" in actions


class TestCompareRecencyByTarget:
    @pytest.fixture
    def recency_comparison_data(self):
        np.random.seed(42)
        data = []
        ref_date = pd.Timestamp("2023-12-31")
        for i in range(100):
            if i < 50:
                last_event = ref_date - timedelta(days=np.random.randint(60, 120))
                target = 0
            else:
                last_event = ref_date - timedelta(days=np.random.randint(5, 30))
                target = 1
            data.append({"entity": f"E{i}", "event_date": last_event, "retained": target})
        return pd.DataFrame(data)

    def test_returns_comparison_result(self, recency_comparison_data):
        result = compare_recency_by_target(
            recency_comparison_data, "entity", "event_date", "retained",
            pd.Timestamp("2023-12-31")
        )
        assert isinstance(result, RecencyComparisonResult)
        assert result.retained_stats is not None
        assert result.churned_stats is not None

    def test_detects_churned_higher_recency(self, recency_comparison_data):
        result = compare_recency_by_target(
            recency_comparison_data, "entity", "event_date", "retained",
            pd.Timestamp("2023-12-31")
        )
        assert result.churned_higher is True
        assert result.churned_stats.median > result.retained_stats.median

    def test_generates_recommendations(self, recency_comparison_data):
        result = compare_recency_by_target(
            recency_comparison_data, "entity", "event_date", "retained",
            pd.Timestamp("2023-12-31")
        )
        assert len(result.recommendations) > 0
        assert "add_recency_features" in [r["action"] for r in result.recommendations]

    def test_returns_none_without_target(self):
        df = pd.DataFrame({
            "entity": ["A", "B"], "event_date": pd.to_datetime(["2023-01-01", "2023-02-01"])
        })
        result = compare_recency_by_target(df, "entity", "event_date", "missing_target")
        assert result is None
