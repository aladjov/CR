from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_pattern_analyzer import (
    RecencyBucketStats,
    RecencyInsight,
    classify_distribution_pattern,
    compare_recency_by_target,
    compute_recency_buckets,
    detect_inflection_bucket,
    generate_recency_insights,
)


@pytest.fixture
def classic_churn_data():
    np.random.seed(42)
    data = []
    ref_date = pd.Timestamp("2023-12-31")
    for i in range(200):
        if i < 80:  # Churned: last activity 60-180 days ago
            last_event = ref_date - timedelta(days=np.random.randint(60, 180))
            target = 0
        else:  # Retained: last activity 1-30 days ago
            last_event = ref_date - timedelta(days=np.random.randint(1, 30))
            target = 1
        data.append({"entity": f"E{i}", "event_date": last_event, "target": target})
    return pd.DataFrame(data)


@pytest.fixture
def monotonic_decline_data():
    np.random.seed(42)
    data = []
    ref_date = pd.Timestamp("2023-12-31")
    # Create data where target rate drops linearly with recency
    for bucket_start in [0, 30, 60, 90, 120]:
        n_entities = 50
        # Higher recency = lower target rate
        target_rate = 0.9 - (bucket_start / 150)
        for i in range(n_entities):
            days = bucket_start + np.random.randint(0, 30)
            target = 1 if np.random.random() < target_rate else 0
            data.append({
                "entity": f"E{bucket_start}_{i}",
                "event_date": ref_date - timedelta(days=days),
                "target": target
            })
    return pd.DataFrame(data)


@pytest.fixture
def threshold_at_30d_data():
    np.random.seed(42)
    data = []
    ref_date = pd.Timestamp("2023-12-31")
    for i in range(200):
        if i < 100:  # Recent (0-30 days): 80% target rate
            days = np.random.randint(1, 30)
            target = 1 if np.random.random() < 0.8 else 0
        else:  # Old (30-120 days): 20% target rate
            days = np.random.randint(30, 120)
            target = 1 if np.random.random() < 0.2 else 0
        data.append({"entity": f"E{i}", "event_date": ref_date - timedelta(days=days), "target": target})
    return pd.DataFrame(data)


@pytest.fixture
def flat_rate_data():
    data = []
    ref_date = pd.Timestamp("2023-12-31")
    # Create perfectly balanced data - each bucket has exactly 50% target rate
    # Bucket ranges: 0-7d, 8-30d, 31-90d, 91-180d
    bucket_day_ranges = [(1, 6), (10, 25), (40, 80), (100, 170)]
    for bucket_idx, (min_day, max_day) in enumerate(bucket_day_ranges):
        for i in range(40):
            days = min_day + (i % (max_day - min_day + 1))
            target = 1 if i < 20 else 0  # Exactly 50% in each bucket
            data.append({"entity": f"E{bucket_idx}_{i}", "event_date": ref_date - timedelta(days=days), "target": target})
    return pd.DataFrame(data)


class TestComputeRecencyBuckets:
    def test_returns_bucket_stats_list(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        assert isinstance(buckets, list)
        assert all(isinstance(b, RecencyBucketStats) for b in buckets)

    def test_bucket_stats_have_required_fields(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        for b in buckets:
            assert hasattr(b, "bucket_label")
            assert hasattr(b, "bucket_range")
            assert hasattr(b, "entity_count")
            assert hasattr(b, "target_rate")
            assert 0 <= b.target_rate <= 1

    def test_custom_bucket_edges(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        custom_edges = [0, 7, 30, 90, 180, float("inf")]
        buckets = compute_recency_buckets(
            classic_churn_data, "entity", "event_date", "target", ref_date,
            bucket_edges=custom_edges
        )
        # Only non-empty buckets are returned
        assert len(buckets) <= len(custom_edges) - 1
        assert len(buckets) >= 1

    def test_total_entities_matches_input(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        total = sum(b.entity_count for b in buckets)
        assert total == len(classic_churn_data)


class TestDetectInflectionBucket:
    def test_finds_inflection_in_threshold_data(self, threshold_at_30d_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            threshold_at_30d_data, "entity", "event_date", "target", ref_date
        )
        inflection = detect_inflection_bucket(buckets)
        assert inflection is not None
        # Should detect the sharp drop around 30 days
        assert "30" in inflection or "8-30" in inflection or "31" in inflection

    def test_returns_none_for_flat_data(self, flat_rate_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            flat_rate_data, "entity", "event_date", "target", ref_date
        )
        inflection = detect_inflection_bucket(buckets)
        # Flat data has no meaningful inflection
        assert inflection is None

    def test_returns_bucket_with_largest_rate_drop(self, monotonic_decline_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            monotonic_decline_data, "entity", "event_date", "target", ref_date
        )
        inflection = detect_inflection_bucket(buckets)
        # Should return the bucket label, not None (monotonic has drops)
        assert inflection is not None


class TestClassifyDistributionPattern:
    def test_monotonic_decline(self, monotonic_decline_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            monotonic_decline_data, "entity", "event_date", "target", ref_date
        )
        pattern = classify_distribution_pattern(buckets)
        assert "monotonic" in pattern.lower() or "declining" in pattern.lower()

    def test_threshold_pattern(self, threshold_at_30d_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            threshold_at_30d_data, "entity", "event_date", "target", ref_date
        )
        pattern = classify_distribution_pattern(buckets)
        assert "threshold" in pattern.lower() or "step" in pattern.lower()

    def test_flat_pattern(self, flat_rate_data):
        ref_date = pd.Timestamp("2023-12-31")
        buckets = compute_recency_buckets(
            flat_rate_data, "entity", "event_date", "target", ref_date
        )
        pattern = classify_distribution_pattern(buckets)
        assert "flat" in pattern.lower() or "stable" in pattern.lower() or "no_pattern" in pattern.lower()


class TestGenerateRecencyInsights:
    def test_returns_insight_list(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        insights = generate_recency_insights(result)
        assert isinstance(insights, list)
        assert all(isinstance(i, RecencyInsight) for i in insights)

    def test_insights_have_required_fields(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        insights = generate_recency_insights(result)
        for i in insights:
            assert hasattr(i, "finding")
            assert hasattr(i, "metric_value")
            assert hasattr(i, "metric_name")
            assert len(i.finding) > 0

    def test_generates_median_gap_insight(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        insights = generate_recency_insights(result)
        metric_names = [i.metric_name for i in insights]
        assert "median_gap_days" in metric_names

    def test_generates_effect_size_insight(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        insights = generate_recency_insights(result)
        metric_names = [i.metric_name for i in insights]
        assert "effect_size" in metric_names

    def test_insights_reflect_actual_data(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        insights = generate_recency_insights(result)
        median_insight = next(i for i in insights if i.metric_name == "median_gap_days")
        expected_gap = result.churned_stats.median - result.retained_stats.median
        assert abs(median_insight.metric_value - expected_gap) < 1


class TestCompareRecencyByTargetEnhanced:
    def test_includes_bucket_stats(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        assert hasattr(result, "bucket_stats")
        assert isinstance(result.bucket_stats, list)
        assert len(result.bucket_stats) > 0

    def test_includes_key_findings(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        assert hasattr(result, "key_findings")
        assert isinstance(result.key_findings, list)
        assert len(result.key_findings) >= 2

    def test_includes_inflection_bucket(self, threshold_at_30d_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            threshold_at_30d_data, "entity", "event_date", "target", ref_date
        )
        assert hasattr(result, "inflection_bucket")

    def test_includes_distribution_pattern(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        assert hasattr(result, "distribution_pattern")
        assert isinstance(result.distribution_pattern, str)

    def test_recommendations_are_actionable(self, classic_churn_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            classic_churn_data, "entity", "event_date", "target", ref_date
        )
        assert len(result.recommendations) >= 2
        assert len(result.recommendations) <= 3
        for rec in result.recommendations:
            assert "features" in rec or "action" in rec

    def test_recommendations_reference_observed_thresholds(self, threshold_at_30d_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(
            threshold_at_30d_data, "entity", "event_date", "target", ref_date
        )
        # At least one recommendation should mention the detected threshold
        rec_texts = " ".join(str(r) for r in result.recommendations)
        assert any(char.isdigit() for char in rec_texts)


class TestAnomalyPattern:
    @pytest.fixture
    def inverted_pattern_data(self):
        np.random.seed(42)
        data = []
        ref_date = pd.Timestamp("2023-12-31")
        for i in range(200):
            if i < 80:  # Churned (40%): RECENT activity (unusual)
                days = np.random.randint(5, 30)
                target = 0
            else:  # Retained (60%): OLD activity (unusual)
                days = np.random.randint(100, 300)
                target = 1
            data.append({"entity": f"E{i}", "event_date": ref_date - timedelta(days=days), "target": target})
        return pd.DataFrame(data)

    @pytest.fixture
    def minority_target_inverted_data(self):
        np.random.seed(42)
        data = []
        ref_date = pd.Timestamp("2023-12-31")
        for i in range(200):
            if i < 160:  # target=0 (80%): RECENT activity (low recency)
                days = np.random.randint(5, 30)
                target = 0
            else:  # target=1 (20%): OLD activity (high recency)
                days = np.random.randint(100, 300)
                target = 1
            data.append({"entity": f"E{i}", "event_date": ref_date - timedelta(days=days), "target": target})
        return pd.DataFrame(data)

    def test_detects_anomaly_pattern(self, inverted_pattern_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(inverted_pattern_data, "entity", "event_date", "target", ref_date)
        assert result.churned_higher is False
        assert result.anomaly_diagnostics is not None

    def test_anomaly_diagnostics_populated(self, inverted_pattern_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(inverted_pattern_data, "entity", "event_date", "target", ref_date)
        diag = result.anomaly_diagnostics
        assert diag is not None
        assert diag.target_1_pct > 50  # target=1 is majority in this fixture
        assert diag.target_1_is_minority is False

    def test_minority_target_detected(self, minority_target_inverted_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(minority_target_inverted_data, "entity", "event_date", "target", ref_date)
        diag = result.anomaly_diagnostics
        assert diag is not None
        assert diag.target_1_is_minority is True
        assert diag.target_1_pct < 50

    def test_anomaly_generates_diagnostic_insight(self, inverted_pattern_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(inverted_pattern_data, "entity", "event_date", "target", ref_date)
        anomaly_insights = [i for i in result.key_findings if i.metric_name == "pattern_anomaly"]
        assert len(anomaly_insights) == 1
        assert "Unusual" in anomaly_insights[0].finding
        assert "Target=1" in anomaly_insights[0].finding

    def test_minority_target_recommends_inversion(self, minority_target_inverted_data):
        ref_date = pd.Timestamp("2023-12-31")
        result = compare_recency_by_target(minority_target_inverted_data, "entity", "event_date", "target", ref_date)
        actions = [r["action"] for r in result.recommendations]
        assert "invert_target_interpretation" in actions


class TestEdgeCases:
    def test_single_bucket_all_same_recency(self):
        ref_date = pd.Timestamp("2023-12-31")
        data = pd.DataFrame({
            "entity": [f"E{i}" for i in range(50)],
            "event_date": [ref_date - timedelta(days=10)] * 50,
            "target": [1] * 25 + [0] * 25
        })
        result = compare_recency_by_target(data, "entity", "event_date", "target", ref_date)
        assert result is not None
        assert len(result.bucket_stats) >= 1

    def test_no_churned_entities(self):
        ref_date = pd.Timestamp("2023-12-31")
        data = pd.DataFrame({
            "entity": [f"E{i}" for i in range(50)],
            "event_date": [ref_date - timedelta(days=i) for i in range(50)],
            "target": [1] * 50  # All retained
        })
        result = compare_recency_by_target(data, "entity", "event_date", "target", ref_date)
        assert result is None  # Cannot compare without both groups

    def test_very_few_entities(self):
        ref_date = pd.Timestamp("2023-12-31")
        data = pd.DataFrame({
            "entity": ["E1", "E2", "E3"],
            "event_date": [ref_date - timedelta(days=d) for d in [5, 50, 100]],
            "target": [1, 0, 0]
        })
        result = compare_recency_by_target(data, "entity", "event_date", "target", ref_date)
        # Should handle gracefully - may return None or limited results
        if result is not None:
            assert len(result.bucket_stats) >= 1
