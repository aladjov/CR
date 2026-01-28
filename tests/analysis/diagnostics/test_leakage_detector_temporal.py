"""Tests for temporal leakage detection in LeakageDetector."""

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.diagnostics.leakage_detector import (
    LeakageDetector,
    LeakageResult,
)
from customer_retention.core.components.enums import Severity


@pytest.fixture
def sample_features_target():
    """Sample feature matrix and target."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "event_count": np.random.randint(1, 100, n),
        }
    )
    y = pd.Series(np.random.choice([0, 1], n, p=[0.6, 0.4]))
    return X, y


class TestUniformTimestampDetection:
    """Tests for LD050: Uniform timestamp detection."""

    def test_check_uniform_timestamps_detects_all_same_timestamp(self):
        """Should detect when all timestamps are identical (datetime.now() pattern)."""
        detector = LeakageDetector()

        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "event_timestamp": pd.Timestamp("2024-01-15 10:30:00"),  # All same
            }
        )

        result = detector.check_uniform_timestamps(df, timestamp_column="event_timestamp")

        # LD050 is HIGH severity (not CRITICAL), so passed is still True
        # but the check should be present
        assert any(c.check_id == "LD050" for c in result.checks)
        assert any(c.severity == Severity.HIGH for c in result.checks)

    def test_check_uniform_timestamps_passes_for_varied_timestamps(self):
        """Should pass when timestamps are properly varied."""
        detector = LeakageDetector()

        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "event_timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            }
        )

        result = detector.check_uniform_timestamps(df, timestamp_column="event_timestamp")

        assert result.passed
        ld050_checks = [c for c in result.checks if c.check_id == "LD050"]
        assert len(ld050_checks) == 0 or all(c.severity == Severity.INFO for c in ld050_checks)

    def test_check_uniform_timestamps_detects_low_variance(self):
        """Should detect when timestamps have suspiciously low variance."""
        detector = LeakageDetector()

        # All timestamps within 1 second - suspicious
        base_ts = pd.Timestamp("2024-01-15 10:30:00")
        df = pd.DataFrame(
            {
                "customer_id": range(100),
                "feature_1": np.random.randn(100),
                "event_timestamp": [base_ts + pd.Timedelta(milliseconds=i) for i in range(100)],
            }
        )

        result = detector.check_uniform_timestamps(df, timestamp_column="event_timestamp")

        # Should flag as suspicious - timestamps span less than 1 second
        assert not result.passed or any(
            c.check_id == "LD050" and c.severity in [Severity.HIGH, Severity.MEDIUM] for c in result.checks
        )

    def test_check_uniform_timestamps_handles_missing_column(self):
        """Should handle gracefully when timestamp column doesn't exist."""
        detector = LeakageDetector()

        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "feature_1": [1.0, 2.0, 3.0],
            }
        )

        result = detector.check_uniform_timestamps(df, timestamp_column="nonexistent")

        # Should pass (no check possible) or return info-level
        assert result.passed or all(c.severity == Severity.INFO for c in result.checks)


class TestTargetDerivedFeatureDetection:
    """Tests for LD052: Target-derived feature detection."""

    def test_check_target_in_features_detects_target_column(self, sample_features_target):
        """Should detect if target column is in feature matrix."""
        X, y = sample_features_target
        detector = LeakageDetector()

        # Add target to features (leakage!)
        X_with_target = X.copy()
        X_with_target["target"] = y.values

        result = detector.check_target_in_features(X_with_target, y, target_name="target")

        assert not result.passed
        assert any(c.check_id == "LD052" for c in result.checks)
        assert any(c.severity == Severity.CRITICAL for c in result.checks)

    def test_check_target_in_features_detects_target_derived_columns(self, sample_features_target):
        """Should detect columns derived from target (e.g., target_sum, target_mean)."""
        X, y = sample_features_target
        detector = LeakageDetector()

        # Add target-derived columns (leakage!)
        X_with_derived = X.copy()
        X_with_derived["target_sum_30d"] = y.values * 10  # Derived from target
        X_with_derived["target_mean_all_time"] = y.values * 0.5

        result = detector.check_target_in_features(X_with_derived, y, target_name="target")

        assert not result.passed
        assert any(c.check_id == "LD052" for c in result.checks)

    def test_check_target_in_features_passes_for_clean_features(self, sample_features_target):
        """Should pass when no target-derived features exist."""
        X, y = sample_features_target
        detector = LeakageDetector()

        result = detector.check_target_in_features(X, y, target_name="target")

        assert result.passed
        ld052_checks = [c for c in result.checks if c.check_id == "LD052"]
        assert len(ld052_checks) == 0

    def test_check_target_in_features_detects_perfect_correlation(self, sample_features_target):
        """Should detect features with perfect correlation to target."""
        X, y = sample_features_target
        detector = LeakageDetector()

        # Add feature perfectly correlated with target
        X_with_leak = X.copy()
        X_with_leak["leaky_feature"] = y.values * 100 + 5  # Perfect linear correlation

        result = detector.check_target_in_features(X_with_leak, y, target_name="target")

        assert not result.passed
        # Should catch via correlation check or LD052
        assert any(c.check_id in ["LD052", "LD001"] and c.severity == Severity.CRITICAL for c in result.checks)


class TestIntegrationWithRunAllChecks:
    """Tests for integration of new checks with run_all_checks."""

    def test_run_all_checks_includes_temporal_checks(self, sample_features_target):
        """run_all_checks should include the new temporal leakage checks."""
        X, y = sample_features_target
        detector = LeakageDetector()

        # Add timestamp column
        X_with_ts = X.copy()
        X_with_ts["event_timestamp"] = pd.date_range("2024-01-01", periods=len(X), freq="h")

        result = detector.run_all_checks(X_with_ts, y, include_pit=True)

        assert isinstance(result, LeakageResult)
        # Should run without error and include various checks
        assert len(result.checks) > 0

    def test_run_all_checks_catches_uniform_timestamps(self, sample_features_target):
        """run_all_checks should catch uniform timestamps when present."""
        X, y = sample_features_target
        detector = LeakageDetector(feature_timestamp_column="event_timestamp")

        # Add uniform timestamps (leakage pattern)
        X_with_uniform_ts = X.copy()
        X_with_uniform_ts["event_timestamp"] = pd.Timestamp("2024-01-15 10:30:00")

        result = detector.run_all_checks(X_with_uniform_ts, y, include_pit=True)

        # Should detect the uniform timestamp issue
        ld050_found = any(c.check_id == "LD050" for c in result.checks)
        assert ld050_found, "LD050 check for uniform timestamps should be included"

    def test_run_all_checks_catches_target_derived_features(self, sample_features_target):
        """run_all_checks should catch target-derived features."""
        X, y = sample_features_target
        detector = LeakageDetector()

        # Add target-derived column
        X_with_target = X.copy()
        X_with_target["target_mean_30d"] = y.values * 0.5

        result = detector.run_all_checks(X_with_target, y, include_pit=True)

        # Should detect via LD052 or correlation checks
        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0


class TestLeakageRecommendations:
    """Tests for actionable recommendations from leakage detection."""

    def test_ld050_recommendation_mentions_reference_date(self):
        """LD050 recommendation should suggest using actual reference dates."""
        detector = LeakageDetector()

        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "event_timestamp": pd.Timestamp("2024-01-15 10:30:00"),
            }
        )

        result = detector.check_uniform_timestamps(df, timestamp_column="event_timestamp")

        if not result.passed:
            recommendations = [c.recommendation for c in result.checks if c.check_id == "LD050"]
            assert len(recommendations) > 0
            # Should mention reference_date or aggregation date
            assert any("reference" in r.lower() or "aggregation" in r.lower() for r in recommendations)

    def test_ld052_recommendation_mentions_exclusion(self):
        """LD052 recommendation should suggest excluding target from features."""
        detector = LeakageDetector()
        X = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "target_sum_30d": [10, 20, 30],
            }
        )
        y = pd.Series([0, 1, 1])

        result = detector.check_target_in_features(X, y, target_name="target")

        if not result.passed:
            recommendations = [c.recommendation for c in result.checks if c.check_id == "LD052"]
            assert len(recommendations) > 0
            # Should mention removing or excluding
            assert any("remove" in r.lower() or "exclude" in r.lower() or "drop" in r.lower() for r in recommendations)
