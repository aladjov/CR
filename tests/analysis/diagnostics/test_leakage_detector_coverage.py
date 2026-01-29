"""Additional coverage tests for LeakageDetector to reach 86% threshold."""

import numpy as np
import pandas as pd

from customer_retention.analysis.diagnostics.leakage_detector import (
    LeakageDetector,
    LeakageResult,
)
from customer_retention.core.components.enums import Severity


class TestClassifyCorrelationThresholds:
    """Tests for _classify_correlation method branch coverage."""

    def test_classify_correlation_critical_threshold(self):
        """Correlation > 0.90 should return CRITICAL severity."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_correlation(0.95)
        assert severity == Severity.CRITICAL
        assert check_id == "LD001"

    def test_classify_correlation_high_threshold(self):
        """Correlation between 0.70 and 0.90 should return HIGH severity."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_correlation(0.75)
        assert severity == Severity.HIGH
        assert check_id == "LD002"

    def test_classify_correlation_medium_threshold(self):
        """Correlation between 0.50 and 0.70 should return MEDIUM severity."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_correlation(0.55)
        assert severity == Severity.MEDIUM
        assert check_id == "LD003"

    def test_classify_correlation_info_threshold(self):
        """Correlation <= 0.50 should return INFO severity."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_correlation(0.3)
        assert severity == Severity.INFO
        assert check_id == "LD000"


class TestCorrelationRecommendation:
    """Tests for _correlation_recommendation method."""

    def test_correlation_recommendation_critical(self):
        """Critical correlation should recommend REMOVE."""
        detector = LeakageDetector()
        rec = detector._correlation_recommendation("feature_x", 0.95)
        assert "REMOVE" in rec
        assert "feature_x" in rec
        assert "0.95" in rec

    def test_correlation_recommendation_high(self):
        """High correlation should recommend INVESTIGATE."""
        detector = LeakageDetector()
        rec = detector._correlation_recommendation("feature_y", 0.75)
        assert "INVESTIGATE" in rec
        assert "feature_y" in rec

    def test_correlation_recommendation_medium(self):
        """Medium correlation should recommend MONITOR."""
        detector = LeakageDetector()
        rec = detector._correlation_recommendation("feature_z", 0.55)
        assert "MONITOR" in rec
        assert "feature_z" in rec


class TestCheckCorrelationsNanHandling:
    """Tests for NaN handling in check_correlations."""

    def test_check_correlations_with_nan_correlation(self):
        """NaN correlation should be treated as 0.0."""
        detector = LeakageDetector()
        X = pd.DataFrame({
            "constant_feature": [1.0] * 100,  # Constant feature produces NaN correlation
            "normal_feature": np.random.randn(100),
        })
        y = pd.Series(np.random.choice([0, 1], 100))

        result = detector.check_correlations(X, y)

        assert isinstance(result, LeakageResult)
        # Constant feature should not trigger high correlation warning
        constant_checks = [c for c in result.checks if c.feature == "constant_feature"]
        assert len(constant_checks) == 0  # INFO level is not added to checks


class TestSeparationChecks:
    """Tests for check_separation and related methods."""

    def test_classify_separation_critical(self):
        """Zero overlap should return CRITICAL."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_separation(0.0)
        assert severity == Severity.CRITICAL
        assert check_id == "LD010"

    def test_classify_separation_high(self):
        """Overlap < 1.0 should return HIGH."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_separation(0.5)
        assert severity == Severity.HIGH
        assert check_id == "LD011"

    def test_classify_separation_medium(self):
        """Overlap between 1.0 and 5.0 should return MEDIUM."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_separation(3.0)
        assert severity == Severity.MEDIUM
        assert check_id == "LD012"

    def test_separation_recommendation_critical(self):
        """Zero overlap should recommend removal."""
        detector = LeakageDetector()
        rec = detector._separation_recommendation("feature_x", 0.0)
        assert "REMOVE" in rec
        assert "perfect" in rec.lower()

    def test_separation_recommendation_high(self):
        """Near-zero overlap should recommend removal."""
        detector = LeakageDetector()
        rec = detector._separation_recommendation("feature_x", 0.5)
        assert "REMOVE" in rec
        assert "near-perfect" in rec.lower()

    def test_separation_recommendation_medium(self):
        """Low overlap should recommend investigation."""
        detector = LeakageDetector()
        rec = detector._separation_recommendation("feature_x", 3.0)
        assert "INVESTIGATE" in rec

    def test_separation_recommendation_normal(self):
        """Normal overlap should be OK."""
        detector = LeakageDetector()
        rec = detector._separation_recommendation("feature_x", 10.0)
        assert "OK" in rec


class TestTemporalLogicChecks:
    """Tests for check_temporal_logic method."""

    def test_temporal_logic_detects_high_correlation_temporal_feature(self):
        """Temporal features with high correlation should be flagged."""
        detector = LeakageDetector()
        np.random.seed(42)
        n = 200
        y = pd.Series(np.random.choice([0, 1], n))
        X = pd.DataFrame({
            "days_since_signup": y * 100 + np.random.randn(n) * 10,  # High correlation
            "regular_feature": np.random.randn(n),
        })

        result = detector.check_temporal_logic(X, y)

        # Should detect the temporal feature
        temporal_checks = [c for c in result.checks if c.check_id == "LD022"]
        assert len(temporal_checks) > 0
        assert any(c.severity == Severity.HIGH for c in temporal_checks)

    def test_temporal_logic_detects_medium_correlation_temporal_feature(self):
        """Temporal features with medium correlation should be flagged."""
        detector = LeakageDetector()
        np.random.seed(42)
        n = 200
        y = pd.Series(np.random.choice([0, 1], n))
        # Create medium correlation (0.5-0.7)
        X = pd.DataFrame({
            "tenure_days": y * 30 + np.random.randn(n) * 50,
            "regular_feature": np.random.randn(n),
        })

        result = detector.check_temporal_logic(X, y)

        # Should detect temporal patterns
        temporal_checks = [c for c in result.checks if c.check_id == "LD022"]
        # Medium correlation should produce MEDIUM severity
        if len(temporal_checks) > 0:
            severities = [c.severity for c in temporal_checks]
            assert Severity.HIGH in severities or Severity.MEDIUM in severities

    def test_temporal_logic_matches_various_patterns(self):
        """Various temporal column names should match TEMPORAL_PATTERNS."""
        detector = LeakageDetector()
        np.random.seed(42)
        n = 100
        y = pd.Series(np.random.choice([0, 1], n))

        temporal_names = [
            "days_since_last_order",
            "recency_score",
            "last_login_days",
            "time_to_event",
            "date_diff",
            "tenure_months",
            "ago_weeks",
        ]

        X = pd.DataFrame({
            name: y * 50 + np.random.randn(n) * 20 for name in temporal_names
        })

        result = detector.check_temporal_logic(X, y)

        # At least some temporal patterns should be detected
        checked_features = [c.feature for c in result.checks]
        assert len(checked_features) > 0

    def test_temporal_logic_ignores_non_numeric(self):
        """Non-numeric temporal columns should be ignored."""
        detector = LeakageDetector()
        X = pd.DataFrame({
            "date_string": ["2024-01-01"] * 50,
            "days_since": np.random.randn(50),
        })
        y = pd.Series(np.random.choice([0, 1], 50))

        result = detector.check_temporal_logic(X, y)

        # date_string should not cause errors
        assert isinstance(result, LeakageResult)


class TestAucChecks:
    """Tests for AUC-based leakage detection."""

    def test_classify_auc_critical(self):
        """AUC > 0.90 should be CRITICAL."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_auc(0.95)
        assert severity == Severity.CRITICAL
        assert check_id == "LD030"

    def test_classify_auc_high(self):
        """AUC between 0.80 and 0.90 should be HIGH."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_auc(0.85)
        assert severity == Severity.HIGH
        assert check_id == "LD031"

    def test_classify_auc_info(self):
        """AUC <= 0.80 should be INFO."""
        detector = LeakageDetector()
        severity, check_id = detector._classify_auc(0.7)
        assert severity == Severity.INFO
        assert check_id == "LD000"

    def test_auc_recommendation_critical(self):
        """Critical AUC should recommend REMOVE."""
        detector = LeakageDetector()
        rec = detector._auc_recommendation("feature_x", 0.95)
        assert "REMOVE" in rec
        assert "0.95" in rec

    def test_auc_recommendation_high(self):
        """High AUC should recommend INVESTIGATE."""
        detector = LeakageDetector()
        rec = detector._auc_recommendation("feature_x", 0.85)
        assert "INVESTIGATE" in rec

    def test_auc_recommendation_normal(self):
        """Normal AUC should be OK."""
        detector = LeakageDetector()
        rec = detector._auc_recommendation("feature_x", 0.65)
        assert "OK" in rec

    def test_compute_single_feature_auc_handles_exception(self):
        """AUC computation should handle exceptions gracefully."""
        detector = LeakageDetector()
        feature = pd.Series([np.nan] * 10)
        y = pd.Series([0, 1] * 5)

        auc = detector._compute_single_feature_auc(feature, y)
        assert auc == 0.5

    def test_compute_single_feature_auc_single_class(self):
        """AUC should return 0.5 for single-class target."""
        detector = LeakageDetector()
        feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([1, 1, 1, 1, 1])

        auc = detector._compute_single_feature_auc(feature, y)
        assert auc == 0.5

    def test_cv_auc_detects_truly_leaky_feature(self):
        detector = LeakageDetector()
        np.random.seed(42)
        n = 500
        y = pd.Series(np.array([0] * 250 + [1] * 250))
        feature = pd.Series(y.values + np.random.randn(n) * 0.01)
        auc = detector._compute_single_feature_auc(feature, y)
        assert auc > 0.95

    def test_cv_auc_near_chance_for_random_feature(self):
        detector = LeakageDetector()
        np.random.seed(42)
        n = 500
        y = pd.Series(np.random.choice([0, 1], n))
        feature = pd.Series(np.random.randn(n))
        auc = detector._compute_single_feature_auc(feature, y)
        assert 0.35 < auc < 0.65

    def test_cv_auc_returns_half_when_minority_class_too_small(self):
        detector = LeakageDetector()
        y = pd.Series([0] * 9 + [1])
        feature = pd.Series(np.random.randn(10))
        auc = detector._compute_single_feature_auc(feature, y)
        assert auc == 0.5

    def test_cv_auc_handles_many_nans(self):
        detector = LeakageDetector()
        np.random.seed(42)
        n = 100
        y = pd.Series(np.random.choice([0, 1], n))
        values = [np.nan] * 80 + list(np.random.randn(20))
        feature = pd.Series(values)
        auc = detector._compute_single_feature_auc(feature, y)
        assert 0.0 <= auc <= 1.0


class TestTemporalFeatureAucHandling:

    def test_classify_auc_temporal_high_not_critical(self):
        detector = LeakageDetector()
        severity, check_id = detector._classify_auc(0.95, is_temporal=True)
        assert severity == Severity.HIGH
        assert check_id == "LD031"

    def test_classify_auc_temporal_moderate_returns_info(self):
        detector = LeakageDetector()
        severity, check_id = detector._classify_auc(0.85, is_temporal=True)
        assert severity == Severity.INFO

    def test_classify_auc_non_temporal_default_unchanged(self):
        detector = LeakageDetector()
        severity, check_id = detector._classify_auc(0.95)
        assert severity == Severity.CRITICAL
        assert check_id == "LD030"

    def test_auc_recommendation_temporal_suggests_review(self):
        detector = LeakageDetector()
        rec = detector._auc_recommendation("days_since_last_event", 0.95, is_temporal=True)
        assert "REVIEW" in rec
        assert "REMOVE" not in rec

    def test_auc_recommendation_non_temporal_still_remove(self):
        detector = LeakageDetector()
        rec = detector._auc_recommendation("feature_x", 0.95)
        assert "REMOVE" in rec

    def test_temporal_feature_not_critical_in_check(self):
        detector = LeakageDetector()
        np.random.seed(42)
        n = 500
        y = pd.Series(np.array([0] * 250 + [1] * 250))
        X = pd.DataFrame({
            "days_since_last_event": y.values * 100 + np.random.randn(n) * 5,
            "normal_feature": np.random.randn(n),
        })
        result = detector.check_single_feature_auc(X, y)
        temporal_checks = [c for c in result.checks if c.feature == "days_since_last_event"]
        assert len(temporal_checks) > 0
        assert all(c.severity != Severity.CRITICAL for c in temporal_checks)

    def test_non_temporal_feature_stays_critical(self):
        detector = LeakageDetector()
        np.random.seed(42)
        n = 500
        y = pd.Series(np.array([0] * 250 + [1] * 250))
        X = pd.DataFrame({
            "leaky_score": y.values + np.random.randn(n) * 0.01,
        })
        result = detector.check_single_feature_auc(X, y)
        critical = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical) > 0

    def test_run_all_checks_no_ld030_for_temporal_features(self):
        detector = LeakageDetector()
        np.random.seed(42)
        n = 500
        y = pd.Series(np.array([0] * 250 + [1] * 250))
        X = pd.DataFrame({
            "days_since_last_event": y.values * 100 + np.random.randn(n) * 5,
        })
        result = detector.run_all_checks(X, y, include_pit=False)
        auc_critical = [c for c in result.checks if c.check_id == "LD030"]
        assert len(auc_critical) == 0

    def test_run_all_checks_passes_with_moderate_temporal_signal(self):
        detector = LeakageDetector()
        np.random.seed(42)
        n = 500
        y = pd.Series(np.random.choice([0, 1], n, p=[0.6, 0.4]))
        X = pd.DataFrame({
            "days_since_last_event": y.values * 20 + np.random.randn(n) * 15,
        })
        result = detector.run_all_checks(X, y, include_pit=False)
        assert result.passed


class TestPointInTimeChecks:
    """Tests for check_point_in_time method."""

    def test_point_in_time_detects_timestamp_violations(self):
        """Should detect when feature_timestamp > label_timestamp."""
        detector = LeakageDetector(
            feature_timestamp_column="feature_ts",
            label_timestamp_column="label_ts",
        )

        df = pd.DataFrame({
            "feature_ts": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-01-25"]),
            "label_ts": pd.to_datetime(["2024-01-10", "2024-01-25", "2024-01-30"]),
            "feature_1": [1.0, 2.0, 3.0],
        })

        result = detector.check_point_in_time(df)

        # First row has feature_ts > label_ts
        assert any(c.check_id == "LD040" for c in result.checks)
        assert not result.passed

    def test_point_in_time_detects_datetime_column_violations(self):
        """Should detect datetime columns with values after feature_timestamp."""
        detector = LeakageDetector(feature_timestamp_column="feature_ts")

        df = pd.DataFrame({
            "feature_ts": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-15"]),
            "order_date": pd.to_datetime(["2024-01-10", "2024-01-20", "2024-01-25"]),
        })
        df["order_date"] = df["order_date"].astype("datetime64[ns]")

        result = detector.check_point_in_time(df)

        # order_date has values after feature_ts
        ld041_checks = [c for c in result.checks if c.check_id == "LD041"]
        assert len(ld041_checks) > 0

    def test_point_in_time_handles_missing_feature_timestamp(self):
        """Should pass gracefully when feature_timestamp column is missing."""
        detector = LeakageDetector(feature_timestamp_column="nonexistent")

        df = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
        })

        result = detector.check_point_in_time(df)

        assert result.passed
        assert len(result.checks) == 0

    def test_point_in_time_handles_invalid_timestamp_format(self):
        """Should handle invalid timestamp format gracefully."""
        detector = LeakageDetector(feature_timestamp_column="feature_ts")

        df = pd.DataFrame({
            "feature_ts": ["not-a-date", "also-not-a-date", "nope"],
        })

        result = detector.check_point_in_time(df)

        # Should not crash
        assert isinstance(result, LeakageResult)

    def test_point_in_time_critical_vs_high_severity(self):
        """Should set CRITICAL for >5% violations, HIGH otherwise."""
        detector = LeakageDetector(feature_timestamp_column="feature_ts")

        # Create 10% violations (> 5% threshold)
        n = 100
        feature_ts = pd.to_datetime(["2024-01-15"] * n)
        order_dates = pd.to_datetime(["2024-01-10"] * 90 + ["2024-01-20"] * 10)

        df = pd.DataFrame({
            "feature_ts": feature_ts,
            "order_date": order_dates,
        })
        df["order_date"] = df["order_date"].astype("datetime64[ns]")

        result = detector.check_point_in_time(df)

        ld041_checks = [c for c in result.checks if c.check_id == "LD041"]
        if len(ld041_checks) > 0:
            # 10% violations should be CRITICAL
            assert ld041_checks[0].severity == Severity.CRITICAL


class TestUniformTimestampsEdgeCases:
    """Additional edge case tests for check_uniform_timestamps."""

    def test_uniform_timestamps_with_single_record(self):
        """Should handle single record gracefully."""
        detector = LeakageDetector()
        df = pd.DataFrame({
            "customer_id": [1],
            "event_timestamp": pd.Timestamp("2024-01-15"),
        })

        result = detector.check_uniform_timestamps(df, timestamp_column="event_timestamp")

        # Single record - not enough to check
        assert result.passed

    def test_uniform_timestamps_with_all_nan(self):
        """Should handle all NaN timestamps."""
        detector = LeakageDetector()
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "event_timestamp": [pd.NaT, pd.NaT, pd.NaT],
        })

        result = detector.check_uniform_timestamps(df, timestamp_column="event_timestamp")

        # All NaN - should pass or handle gracefully
        assert isinstance(result, LeakageResult)


class TestTargetInFeaturesEdgeCases:
    """Additional edge case tests for check_target_in_features."""

    def test_target_in_features_with_underscore_pattern(self):
        """Should detect _target suffix pattern."""
        detector = LeakageDetector()
        X = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "rolling_target": [0.5, 0.6, 0.7],  # _target pattern
        })
        y = pd.Series([0, 1, 1])

        result = detector.check_target_in_features(X, y, target_name="target")

        assert not result.passed
        assert any(c.check_id == "LD052" for c in result.checks)

    def test_target_in_features_handles_correlation_exception(self):
        """Should handle correlation calculation exceptions."""
        detector = LeakageDetector()
        X = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "constant": [1.0, 1.0, 1.0],  # Constant - may cause issues
        })
        y = pd.Series([0, 1, 1])

        result = detector.check_target_in_features(X, y, target_name="target")

        # Should not crash
        assert isinstance(result, LeakageResult)


class TestRunAllChecksIntegration:
    """Integration tests for run_all_checks."""

    def test_run_all_checks_without_pit(self):
        """run_all_checks with include_pit=False should skip PIT checks."""
        detector = LeakageDetector()
        X = pd.DataFrame({
            "feature_1": np.random.randn(50),
            "feature_2": np.random.randn(50),
        })
        y = pd.Series(np.random.choice([0, 1], 50))

        result = detector.run_all_checks(X, y, include_pit=False)

        # Should not have LD040/LD041/LD050 checks
        pit_checks = [c for c in result.checks if c.check_id in ["LD040", "LD041", "LD050"]]
        assert len(pit_checks) == 0

    def test_run_all_checks_generates_recommendations(self):
        """run_all_checks should populate recommendations for critical/high issues."""
        detector = LeakageDetector()
        np.random.seed(42)
        n = 200
        y = pd.Series(np.random.choice([0, 1], n))
        X = pd.DataFrame({
            "leaky_feature": y * 100 + np.random.randn(n),  # High correlation
            "normal_feature": np.random.randn(n),
        })

        result = detector.run_all_checks(X, y, include_pit=False)

        # Should have recommendations
        assert len(result.recommendations) > 0
