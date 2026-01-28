"""Tests for new leakage detection checks: domain patterns, temporal split, cross-entity."""

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics import LeakageDetector
from customer_retention.core.components.enums import Severity


class TestDomainTargetPatterns:
    """Tests for LD053 domain-specific target pattern detection."""

    @pytest.fixture
    def churn_feature_data(self):
        """Data with features that have churn/retention-related names."""
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        return pd.DataFrame({
            "churn_rate_30d": target * 0.8 + np.random.randn(n) * 0.1,  # High corr with target
            "cancel_probability": target * 0.5 + np.random.randn(n) * 0.3,  # Medium corr
            "retention_score": np.random.randn(n),  # Low corr
            "normal_feature": np.random.randn(n),
            "target": target,
        })

    def test_ld053_detects_high_correlation_domain_patterns(self, churn_feature_data):
        detector = LeakageDetector()
        X = churn_feature_data.drop(columns=["target"])
        y = churn_feature_data["target"]

        result = detector.check_domain_target_patterns(X, y)

        critical_features = {c.feature for c in result.checks if c.severity == Severity.CRITICAL}
        assert "churn_rate_30d" in critical_features

    def test_ld053_flags_medium_correlation_as_high(self, churn_feature_data):
        detector = LeakageDetector()
        X = churn_feature_data.drop(columns=["target"])
        y = churn_feature_data["target"]

        result = detector.check_domain_target_patterns(X, y)

        high_features = {c.feature for c in result.checks if c.severity == Severity.HIGH}
        assert "cancel_probability" in high_features

    def test_ld053_flags_low_correlation_as_medium(self):
        """Test that domain patterns with low correlation are flagged as MEDIUM."""
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        # Create a feature with domain name but very low correlation
        df = pd.DataFrame({
            "retention_score": np.random.randn(n) * 0.1 + target * 0.1,  # Low but non-zero corr
            "target": target,
        })

        detector = LeakageDetector()
        X = df.drop(columns=["target"])
        y = df["target"]

        result = detector.check_domain_target_patterns(X, y)

        # Should be flagged (domain pattern detected)
        flagged_features = {c.feature for c in result.checks}
        assert "retention_score" in flagged_features

    def test_ld053_ignores_non_domain_features(self, churn_feature_data):
        detector = LeakageDetector()
        X = churn_feature_data.drop(columns=["target"])
        y = churn_feature_data["target"]

        result = detector.check_domain_target_patterns(X, y)

        flagged_features = {c.feature for c in result.checks}
        assert "normal_feature" not in flagged_features

    def test_domain_patterns_included_in_run_all_checks(self, churn_feature_data):
        detector = LeakageDetector()
        X = churn_feature_data.drop(columns=["target"])
        y = churn_feature_data["target"]

        result = detector.run_all_checks(X, y, include_pit=False)

        ld053_checks = [c for c in result.checks if c.check_id == "LD053"]
        assert len(ld053_checks) > 0

    @pytest.mark.parametrize("pattern", [
        "churn_rate", "cancel_date", "unsubscribe_flag", "attrition_score",
        "lapse_indicator", "defection_risk", "convert_flag", "inactive_days",
        "leave_probability", "stay_score", "renewal_status", "expire_date",
        "terminate_reason", "close_date", "deactivation_flag"
    ])
    def test_domain_pattern_variations(self, pattern):
        """Test that various domain pattern variations are detected."""
        np.random.seed(42)
        n = 100
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            pattern: np.random.randn(n),
            "normal": np.random.randn(n),
        })

        detector = LeakageDetector()
        result = detector.check_domain_target_patterns(df, pd.Series(target))

        flagged_features = {c.feature for c in result.checks}
        assert pattern in flagged_features, f"Pattern '{pattern}' should be detected"


class TestTemporalSplitValidation:
    """Tests for LD061 temporal train/test split validation."""

    def test_ld061_detects_overlapping_splits(self):
        """Test detection of train/test temporal overlap."""
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"])
        test_dates = pd.to_datetime(["2024-02-01", "2024-02-15", "2024-03-01"])  # Overlaps with train

        detector = LeakageDetector()
        result = detector.check_temporal_split(train_dates, test_dates)

        assert not result.passed
        assert len(result.critical_issues) > 0
        assert result.critical_issues[0].check_id == "LD061"

    def test_ld061_passes_for_proper_temporal_split(self):
        """Test that proper temporal splits pass validation."""
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"])
        test_dates = pd.to_datetime(["2024-03-01", "2024-03-15", "2024-04-01"])  # All after train

        detector = LeakageDetector()
        result = detector.check_temporal_split(train_dates, test_dates)

        assert result.passed
        assert len(result.critical_issues) == 0

    def test_ld061_handles_exact_boundary(self):
        """Test that exact boundary (train_max == test_min) is flagged."""
        train_dates = pd.to_datetime(["2024-01-01", "2024-02-01"])
        test_dates = pd.to_datetime(["2024-02-01", "2024-03-01"])  # test_min == train_max

        detector = LeakageDetector()
        result = detector.check_temporal_split(train_dates, test_dates)

        assert not result.passed  # Should fail - boundary is not safe

    def test_ld061_handles_empty_timestamps(self):
        """Test graceful handling of empty timestamp series."""
        detector = LeakageDetector()

        result = detector.check_temporal_split(pd.Series(dtype="datetime64[ns]"), pd.Series(dtype="datetime64[ns]"))
        assert result.passed

    def test_ld061_reports_overlap_statistics(self):
        """Test that overlap statistics are reported in recommendation."""
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"])
        test_dates = pd.to_datetime(["2024-02-01"])

        detector = LeakageDetector()
        result = detector.check_temporal_split(train_dates, test_dates)

        recommendation = result.critical_issues[0].recommendation
        assert "overlap" in recommendation.lower()
        assert "%" in recommendation


class TestCrossEntityLeakage:
    """Tests for LD060 cross-entity aggregation leakage detection."""

    @pytest.fixture
    def cross_entity_feature_data(self):
        """Data with features that have cross-entity aggregation patterns."""
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        return pd.DataFrame({
            "global_avg_spend": target * 0.6 + np.random.randn(n) * 0.2,
            "population_percentile": np.random.randn(n),
            "cohort_avg_activity": np.random.randn(n),
            "normal_feature": np.random.randn(n),
            "target": target,
        })

    def test_ld060_detects_global_aggregation_patterns(self, cross_entity_feature_data):
        detector = LeakageDetector()
        X = cross_entity_feature_data.drop(columns=["target"])
        y = cross_entity_feature_data["target"]

        result = detector.check_cross_entity_leakage(X, y, "entity_id", "timestamp")

        flagged_features = {c.feature for c in result.checks}
        assert "global_avg_spend" in flagged_features

    def test_ld060_detects_population_patterns(self, cross_entity_feature_data):
        detector = LeakageDetector()
        X = cross_entity_feature_data.drop(columns=["target"])
        y = cross_entity_feature_data["target"]

        result = detector.check_cross_entity_leakage(X, y, "entity_id", "timestamp")

        flagged_features = {c.feature for c in result.checks}
        assert "population_percentile" in flagged_features

    def test_ld060_ignores_normal_features(self, cross_entity_feature_data):
        detector = LeakageDetector()
        X = cross_entity_feature_data.drop(columns=["target"])
        y = cross_entity_feature_data["target"]

        result = detector.check_cross_entity_leakage(X, y, "entity_id", "timestamp")

        flagged_features = {c.feature for c in result.checks}
        assert "normal_feature" not in flagged_features

    @pytest.mark.parametrize("pattern", [
        "global_mean", "population_avg", "all_users_median", "cross_entity_std",
        "market_avg_price", "cohort_avg_tenure", "overall_mean_spend",
        "overall_std_activity", "benchmark_score", "percentile_rank_spend"
    ])
    def test_cross_entity_pattern_variations(self, pattern):
        """Test that various cross-entity pattern variations are detected."""
        np.random.seed(42)
        n = 100
        target = np.random.choice([0, 1], n)
        df = pd.DataFrame({
            pattern: np.random.randn(n),
            "normal": np.random.randn(n),
        })

        detector = LeakageDetector()
        result = detector.check_cross_entity_leakage(df, pd.Series(target), "entity", "ts")

        flagged_features = {c.feature for c in result.checks}
        assert pattern in flagged_features, f"Pattern '{pattern}' should be detected"


class TestIntegration:
    """Integration tests for all new checks working together."""

    def test_run_all_checks_includes_domain_patterns(self):
        """Verify domain patterns are included in run_all_checks."""
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        df = pd.DataFrame({
            "churn_score": target + np.random.randn(n) * 0.1,
            "normal": np.random.randn(n),
        })

        detector = LeakageDetector()
        result = detector.run_all_checks(df, pd.Series(target), include_pit=False)

        check_ids = {c.check_id for c in result.checks}
        assert "LD053" in check_ids

    def test_multiple_leakage_types_detected_together(self):
        """Test that multiple leakage types are detected in same dataset."""
        np.random.seed(42)
        n = 200
        target = np.random.choice([0, 1], n, p=[0.3, 0.7])
        leaky = target + np.random.randn(n) * 0.01

        df = pd.DataFrame({
            "churn_indicator": leaky,  # Domain pattern + high correlation
            "target_sum_30d": leaky,  # Target-derived name
            "leaky_feature": leaky,  # High correlation
            "normal": np.random.randn(n),
        })

        detector = LeakageDetector()
        result = detector.run_all_checks(df, pd.Series(target), include_pit=False)

        # Should detect multiple types
        check_ids = {c.check_id for c in result.checks}
        assert len(check_ids) >= 2  # At least 2 different check types

        # Should have critical issues
        assert len(result.critical_issues) > 0

        # Should not pass
        assert not result.passed
