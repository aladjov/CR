from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.utils import compute_effect_size
from customer_retention.stages.profiling.temporal_pattern_analyzer import (
    AnomalyDiagnostics,
    CohortDistribution,
    GroupStats,
    RecencyBucketStats,
    RecencyComparisonResult,
    RecencyResult,
    TemporalPatternAnalysis,
    TemporalPatternAnalyzer,
    TrendDirection,
    TrendRecommendation,
    TrendResult,
    _diagnose_anomaly_pattern,
    _effect_size_description,
    _generate_bucket_labels,
    _generate_enhanced_recommendations,
    analyze_cohort_distribution,
    classify_distribution_pattern,
    compare_recency_by_target,
    compute_group_stats,
    detect_inflection_bucket,
    generate_cohort_recommendations,
    generate_recency_insights,
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

    def test_returns_none_insufficient_data(self):
        """Too few retained or churned entities => None."""
        df = pd.DataFrame({
            "entity": ["A", "A", "B", "B"],
            "event_date": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]),
            "retained": [1, 1, 1, 1],
        })
        result = compare_recency_by_target(df, "entity", "event_date", "retained",
                                           pd.Timestamp("2023-12-31"))
        assert result is None

    def test_uses_default_reference_date(self, recency_comparison_data):
        """When no reference_date given, uses max date."""
        result = compare_recency_by_target(
            recency_comparison_data, "entity", "event_date", "retained"
        )
        assert result is not None

    def test_anomaly_diagnostics_when_churned_not_higher(self):
        """Anomaly path: churned have LOWER recency than retained."""
        np.random.seed(42)
        data = []
        ref_date = pd.Timestamp("2023-12-31")
        # Churned: very recent
        for i in range(50):
            last_event = ref_date - timedelta(days=np.random.randint(1, 10))
            data.append({"entity": f"E{i}", "event_date": last_event, "retained": 0})
        # Retained: longer ago
        for i in range(50, 100):
            last_event = ref_date - timedelta(days=np.random.randint(60, 120))
            data.append({"entity": f"E{i}", "event_date": last_event, "retained": 1})
        df = pd.DataFrame(data)
        result = compare_recency_by_target(df, "entity", "event_date", "retained", ref_date)
        assert result is not None
        assert result.churned_higher is False
        assert result.anomaly_diagnostics is not None


class TestDetectInflectionBucket:
    def test_less_than_two_buckets(self):
        buckets = [RecencyBucketStats("0-7d", (0, 7), 50, 0.8)]
        assert detect_inflection_bucket(buckets) is None

    def test_detects_inflection(self):
        buckets = [
            RecencyBucketStats("0-7d", (0, 7), 50, 0.9),
            RecencyBucketStats("8-30d", (8, 30), 50, 0.5),
            RecencyBucketStats("31-90d", (31, 90), 50, 0.4),
        ]
        result = detect_inflection_bucket(buckets)
        assert result == "8-30d"

    def test_no_inflection_when_small_drops(self):
        buckets = [
            RecencyBucketStats("0-7d", (0, 7), 50, 0.5),
            RecencyBucketStats("8-30d", (8, 30), 50, 0.49),
            RecencyBucketStats("31-90d", (31, 90), 50, 0.48),
        ]
        assert detect_inflection_bucket(buckets) is None


class TestClassifyDistributionPattern:
    def test_insufficient_data(self):
        buckets = [RecencyBucketStats("0-7d", (0, 7), 50, 0.8)]
        assert classify_distribution_pattern(buckets) == "insufficient_data"

    def test_flat_pattern(self):
        buckets = [
            RecencyBucketStats("0-7d", (0, 7), 50, 0.5),
            RecencyBucketStats("8-30d", (8, 30), 50, 0.5),
            RecencyBucketStats("31-90d", (31, 90), 50, 0.5),
        ]
        assert classify_distribution_pattern(buckets) == "flat_no_pattern"

    def test_monotonic_decline(self):
        buckets = [
            RecencyBucketStats("0-7d", (0, 7), 50, 0.9),
            RecencyBucketStats("8-30d", (8, 30), 50, 0.7),
            RecencyBucketStats("31-90d", (31, 90), 50, 0.5),
            RecencyBucketStats("91-180d", (91, 180), 50, 0.3),
        ]
        assert classify_distribution_pattern(buckets) == "monotonic_decline"

    def test_threshold_step(self):
        # Need max_drop > avg_drop*2 and max_drop >= 0.10
        # rates = [0.9, 0.3, 0.3] => drops=[0.6, 0.0], total_drop=0.6
        # avg_drop=0.3, max_drop=0.6 > 0.6 => need more buckets for the math
        # rates = [0.9, 0.2, 0.19, 0.19] => total_drop=0.71, avg_drop=0.2367
        # max_drop=0.7 > 0.4733 => True
        buckets = [
            RecencyBucketStats("0-7d", (0, 7), 50, 0.9),
            RecencyBucketStats("8-30d", (8, 30), 50, 0.2),
            RecencyBucketStats("31-90d", (31, 90), 50, 0.19),
            RecencyBucketStats("91-180d", (91, 180), 50, 0.19),
        ]
        assert classify_distribution_pattern(buckets) == "threshold_step"

    def test_variable_pattern(self):
        # Need: not flat, not threshold_step, not monotonic
        # rates=[0.7, 0.8, 0.6, 0.5] => total_drop=0.2, drops=[-0.1, 0.2, 0.1]
        # avg_drop=0.0667, max_drop=0.2 > 0.133 => threshold_step
        # Need max_drop <= avg_drop*2 AND some negative drop
        # rates=[0.8, 0.9, 0.6, 0.2] => total_drop=0.6, drops=[-0.1, 0.3, 0.4]
        # avg_drop=0.2, max_drop=0.4 > 0.4 => borderline. Use 5 buckets:
        # rates=[0.8, 0.9, 0.7, 0.5, 0.2] => total_drop=0.6, drops=[-0.1, 0.2, 0.2, 0.3]
        # avg_drop=0.15, max_drop=0.3 > 0.3 => borderline again
        # Let's make drops nearly equal with one negative
        # rates=[0.7, 0.8, 0.6, 0.4, 0.2] => total_drop=0.5, drops=[-0.1, 0.2, 0.2, 0.2]
        # avg_drop=0.125, max_drop=0.2 > 0.25 => False, not monotonic (-0.1 < -0.05) => variable
        buckets = [
            RecencyBucketStats("0-7d", (0, 7), 50, 0.7),
            RecencyBucketStats("8-30d", (8, 30), 50, 0.8),
            RecencyBucketStats("31-90d", (31, 90), 50, 0.6),
            RecencyBucketStats("91-180d", (91, 180), 50, 0.4),
            RecencyBucketStats(">180d", (180, 9999), 50, 0.2),
        ]
        assert classify_distribution_pattern(buckets) == "variable"


class TestDiagnoseAnomalyPattern:
    def test_basic_anomaly_diagnosis(self):
        df = pd.DataFrame({
            "entity": ["A", "A", "B", "B", "C", "C"],
            "time": pd.to_datetime(["2023-01-01", "2023-06-01",
                                    "2023-01-01", "2023-03-01",
                                    "2023-01-01", "2023-02-01"]),
            "target": [1, 1, 0, 0, 0, 0],
        })
        result = _diagnose_anomaly_pattern(df, "entity", "time", "target")
        assert isinstance(result, AnomalyDiagnostics)
        assert isinstance(result.target_1_is_minority, bool)
        assert result.target_1_pct > 0

    def test_tenure_explains_pattern(self):
        """Retained have much longer tenure => tenure_explains_pattern = True."""
        df = pd.DataFrame({
            "entity": ["A", "A", "B", "B"],
            "time": pd.to_datetime(["2023-01-01", "2023-12-01",
                                    "2023-06-01", "2023-06-10"]),
            "target": [1, 1, 0, 0],
        })
        result = _diagnose_anomaly_pattern(df, "entity", "time", "target")
        # Retained entity A: 334 days tenure, Churned entity B: 9 days
        assert result.tenure_explains_pattern is True

    def test_no_retained(self):
        """All churned => retained_median_tenure is None."""
        df = pd.DataFrame({
            "entity": ["A", "B"],
            "time": pd.to_datetime(["2023-01-01", "2023-06-01"]),
            "target": [0, 0],
        })
        result = _diagnose_anomaly_pattern(df, "entity", "time", "target")
        assert result.retained_median_tenure is None
        assert result.tenure_explains_pattern is False


class TestGenerateRecencyInsights:
    def _make_group_stats(self, median):
        return GroupStats(mean=median, median=median, std=10, q25=median - 10, q75=median + 10, count=50)

    def test_churned_higher_insights(self):
        result = RecencyComparisonResult(
            retained_stats=self._make_group_stats(10),
            churned_stats=self._make_group_stats(60),
            cohens_d=0.9, effect_interpretation="Large effect",
            churned_higher=True, recommendations=[],
        )
        result.inflection_bucket = "31-90d"
        result.distribution_pattern = "threshold_step"
        insights = generate_recency_insights(result)
        findings = [i.finding for i in insights]
        assert any("longer" in f for f in findings)
        # Verify inflection insight present when churned_higher is True
        assert len(insights) >= 3, f"Expected at least 3 insights, got {len(insights)}: {findings}"
        metric_names = [i.metric_name for i in insights]
        assert "inflection_point" in metric_names, f"Got metric names: {metric_names}, findings: {findings}"

    def test_anomaly_insights_minority_target(self):
        diag = AnomalyDiagnostics(
            target_1_is_minority=True, target_1_pct=20.0,
            retained_median_tenure=100, churned_median_tenure=50,
            tenure_explains_pattern=True,
        )
        result = RecencyComparisonResult(
            retained_stats=self._make_group_stats(60),
            churned_stats=self._make_group_stats(10),
            cohens_d=-0.5, effect_interpretation="Medium effect",
            churned_higher=False, recommendations=[],
            anomaly_diagnostics=diag,
        )
        insights = generate_recency_insights(result)
        assert any("minority" in i.finding.lower() for i in insights)
        assert any("Tenure gap" in i.finding for i in insights)

    def test_anomaly_insights_majority_target(self):
        diag = AnomalyDiagnostics(
            target_1_is_minority=False, target_1_pct=70.0,
            retained_median_tenure=100, churned_median_tenure=80,
            tenure_explains_pattern=False,
        )
        result = RecencyComparisonResult(
            retained_stats=self._make_group_stats(60),
            churned_stats=self._make_group_stats(10),
            cohens_d=-0.3, effect_interpretation="Small effect",
            churned_higher=False, recommendations=[],
            anomaly_diagnostics=diag,
        )
        insights = generate_recency_insights(result)
        assert any("majority" in i.finding.lower() for i in insights)

    def test_no_inflection_when_churned_not_higher(self):
        result = RecencyComparisonResult(
            retained_stats=self._make_group_stats(60),
            churned_stats=self._make_group_stats(10),
            cohens_d=-0.3, effect_interpretation="Small effect",
            churned_higher=False, recommendations=[],
            inflection_bucket="31-90d",
        )
        insights = generate_recency_insights(result)
        assert not any("inflection" in i.finding.lower() for i in insights)


class TestEffectSizeDescription:
    def test_large_effect(self):
        desc = _effect_size_description(0.9, "Large effect")
        assert "strongly" in desc

    def test_moderate_effect(self):
        desc = _effect_size_description(0.6, "Medium effect")
        assert "moderately" in desc

    def test_weak_effect(self):
        desc = _effect_size_description(0.3, "Small effect")
        assert "weakly" in desc

    def test_minimal_effect(self):
        desc = _effect_size_description(0.1, "Negligible")
        assert "minimal" in desc


class TestGenerateEnhancedRecommendations:
    def test_churned_not_higher_minority_target(self):
        diag = AnomalyDiagnostics(
            target_1_is_minority=True, target_1_pct=20.0,
        )
        recs = _generate_enhanced_recommendations(False, 0.5, None, "flat", [], diag)
        assert any(r["action"] == "invert_target_interpretation" for r in recs)

    def test_churned_not_higher_tenure_explains(self):
        diag = AnomalyDiagnostics(
            target_1_is_minority=False, target_1_pct=70.0,
            retained_median_tenure=200, churned_median_tenure=50,
            tenure_explains_pattern=True,
        )
        recs = _generate_enhanced_recommendations(False, 0.5, None, "flat", [], diag)
        assert any(r["action"] == "use_tenure_adjusted_recency" for r in recs)

    def test_churned_not_higher_unexplained(self):
        diag = AnomalyDiagnostics(
            target_1_is_minority=False, target_1_pct=70.0,
            tenure_explains_pattern=False,
        )
        recs = _generate_enhanced_recommendations(False, 0.5, None, "flat", [], diag)
        assert any(r["action"] == "investigate_further" for r in recs)
        assert any(r["action"] == "check_pre_churn_activity" for r in recs)

    def test_churned_higher_strong_effect(self):
        recs = _generate_enhanced_recommendations(True, 0.6, None, "flat", [])
        assert any(r["action"] == "add_recency_features" for r in recs)

    def test_churned_higher_threshold_step(self):
        recs = _generate_enhanced_recommendations(True, 0.6, "31-90d", "threshold_step", [])
        assert any(r["action"] == "create_activity_threshold_flag" for r in recs)

    def test_churned_higher_monotonic_decline(self):
        recs = _generate_enhanced_recommendations(True, 0.3, None, "monotonic_decline", [])
        assert any(r["action"] == "use_continuous_recency" for r in recs)

    def test_churned_higher_fallback_buckets(self):
        """When few recommendations and bucket_stats exist, add bucket rec."""
        bucket_stats = [RecencyBucketStats("0-7d", (0, 7), 50, 0.8)]
        recs = _generate_enhanced_recommendations(True, 0.3, None, "variable", bucket_stats)
        assert any(r["action"] == "add_recency_buckets" for r in recs)


class TestDetectTrendEdgeCases:
    def test_less_than_3_rows(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "value": [1, 2],
        })
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(df, "value")
        assert result.direction == TrendDirection.UNKNOWN

    def test_less_than_3_after_dropna(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            "value": [1, np.nan, np.nan, 4],
        })
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(df, "value")
        # Only 2 non-null values, so should return unknown
        assert result.direction == TrendDirection.UNKNOWN

    def test_zero_mean(self):
        """Values centered around zero => normalized_slope uses 0 path."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
        })
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(df, "value")
        assert result.direction in [TrendDirection.INCREASING, TrendDirection.STABLE, TrendDirection.DECREASING]


class TestSeasonalityEdgeCases:
    def test_less_than_14_rows(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": range(10),
        })
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_seasonality(df, "value")
        assert result == []

    def test_less_than_14_after_dropna(self):
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        values = [1.0, 2.0] + [np.nan] * 12 + [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        df = pd.DataFrame({"date": dates, "value": values})
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_seasonality(df, "value")
        assert result == []

    def test_lag_exceeds_half_data(self):
        """Lags bigger than len(values)//2 should be skipped."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        df = pd.DataFrame({"date": dates, "value": np.random.normal(0, 1, 20)})
        analyzer = TemporalPatternAnalyzer(time_column="date")
        # additional_lags=[15] but 15 >= 20//2=10, so skipped
        result = analyzer.detect_seasonality(df, "value", additional_lags=[15])
        # Just verify it doesn't crash
        assert isinstance(result, list)

    def test_zero_variance_autocorrelation(self):
        """All same values => variance = 0 => autocorrelation returns 0."""
        analyzer = TemporalPatternAnalyzer(time_column="date")
        series = np.array([5.0] * 20)
        assert analyzer._autocorrelation(series, 3) == 0.0

    def test_lag_exceeds_n(self):
        """Lag >= len(series) => 0.0."""
        analyzer = TemporalPatternAnalyzer(time_column="date")
        series = np.array([1, 2, 3])
        assert analyzer._autocorrelation(series, 5) == 0.0


class TestAnalyzeCohortsEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["customer_id", "event_date", "signup_date"])
        analyzer = TemporalPatternAnalyzer(time_column="event_date")
        result = analyzer.analyze_cohorts(df, "customer_id", "signup_date")
        assert len(result) == 0


class TestAnalyzeRecencyEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["customer_id", "event_date"])
        analyzer = TemporalPatternAnalyzer(time_column="event_date")
        result = analyzer.analyze_recency(df, "customer_id")
        assert result.avg_recency_days == 0
        assert result.median_recency_days == 0

    def test_default_reference_date(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "event_date": pd.to_datetime(["2023-01-01", "2023-06-01"]),
        })
        analyzer = TemporalPatternAnalyzer(time_column="event_date")
        result = analyzer.analyze_recency(df, "customer_id")
        assert result.avg_recency_days > 0

    def test_insufficient_data_for_correlation(self):
        """Only 2 entities => no correlation computed."""
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "event_date": pd.to_datetime(["2023-01-01", "2023-06-01"]),
            "target": [0, 1],
        })
        analyzer = TemporalPatternAnalyzer(time_column="event_date")
        result = analyzer.analyze_recency(df, "customer_id", target_column="target",
                                          reference_date=pd.Timestamp("2023-12-31"))
        assert result.target_correlation is None


class TestTrendResultProperties:
    def test_is_significant_true(self):
        t = TrendResult(direction=TrendDirection.INCREASING, strength=0.5, p_value=0.01)
        assert t.is_significant is True

    def test_is_significant_false(self):
        t = TrendResult(direction=TrendDirection.STABLE, strength=0.1, p_value=0.5)
        assert t.is_significant is False

    def test_is_significant_none_pvalue(self):
        t = TrendResult(direction=TrendDirection.UNKNOWN, strength=0.0, p_value=None)
        assert t.is_significant is False

    def test_has_direction_true(self):
        t = TrendResult(direction=TrendDirection.INCREASING, strength=0.5)
        assert t.has_direction is True

    def test_has_direction_false(self):
        t = TrendResult(direction=TrendDirection.STABLE, strength=0.0)
        assert t.has_direction is False


class TestGenerateTrendRecommendationsEdgeCases:
    def test_no_slope(self):
        """slope is None => daily_pct = 0."""
        trend = TrendResult(direction=TrendDirection.UNKNOWN, strength=0.0, slope=None, p_value=0.5)
        recs = generate_trend_recommendations(trend)
        assert isinstance(recs, list)

    def test_weak_non_significant_trend(self):
        """Has direction and strength > 0.1 but not significant."""
        trend = TrendResult(direction=TrendDirection.INCREASING, strength=0.2, slope=0.05, p_value=0.1)
        recs = generate_trend_recommendations(trend)
        # Neither strong nor moderate nor stable => empty
        assert len(recs) == 0


class TestGenerateCohortRecommendationsElse:
    def test_moderate_variation(self):
        """Two years, dominant_pct between 60 and 80 => consider_cohort_features."""
        dist = CohortDistribution(
            year_counts={2020: 600, 2021: 400}, total_entities=1000,
            dominant_year=2020, dominant_pct=60.0, num_years=2
        )
        recs = generate_cohort_recommendations(dist)
        actions = [r.action for r in recs]
        assert "consider_cohort_features" in actions

    def test_no_retention_variation(self):
        dist = CohortDistribution(
            year_counts={2020: 500, 2021: 500}, total_entities=1000,
            dominant_year=2020, dominant_pct=50.0, num_years=2
        )
        recs = generate_cohort_recommendations(dist, retention_variation=None)
        assert not any(r.action == "investigate_cohort_retention" for r in recs)

    def test_low_retention_variation(self):
        dist = CohortDistribution(
            year_counts={2020: 500, 2021: 500}, total_entities=1000,
            dominant_year=2020, dominant_pct=50.0, num_years=2
        )
        recs = generate_cohort_recommendations(dist, retention_variation=0.05)
        assert not any(r.action == "investigate_cohort_retention" for r in recs)


class TestGenerateBucketLabels:
    def test_generates_correct_labels(self):
        edges = [0, 7, 30, float("inf")]
        labels = _generate_bucket_labels(edges)
        assert labels == ["0-7d", "8-30d", ">30d"]
