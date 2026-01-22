import numpy as np
import pandas as pd
import pytest

from customer_retention.core.components.enums import Severity
from customer_retention.stages.validation import LeakageGate


class TestHighCorrelationDetection:
    @pytest.fixture
    def df_with_leaky_feature(self):
        np.random.seed(42)
        n = 1000
        target = np.random.choice([0, 1], n)
        # Feature perfectly correlated with target
        leaky = target + np.random.randn(n) * 0.05
        normal = np.random.randn(n)
        return pd.DataFrame({
            "leaky_feature": leaky,
            "normal_feature": normal,
            "target": target
        })

    def test_detects_high_correlation_lk001(self, df_with_leaky_feature):
        gate = LeakageGate(
            target_column="target",
            correlation_threshold_critical=0.90
        )
        result = gate.run(df_with_leaky_feature)

        assert not result.passed
        assert len(result.critical_issues) > 0
        # Should detect the leaky feature
        leaky_issues = [i for i in result.critical_issues if "leaky_feature" in str(i)]
        assert len(leaky_issues) > 0

    def test_suspicious_correlation_lk002(self):
        np.random.seed(42)
        n = 1000
        target = np.random.choice([0, 1], n)
        # Feature moderately correlated with target (stronger correlation)
        suspicious = target * 1.5 + np.random.randn(n) * 0.3
        df = pd.DataFrame({
            "suspicious_feature": suspicious,
            "target": target
        })

        gate = LeakageGate(
            target_column="target",
            correlation_threshold_high=0.70
        )
        result = gate.run(df)

        # Should flag high-risk items
        assert len(result.high_issues) > 0 or len(result.critical_issues) > 0


class TestPerfectSeparationDetection:
    def test_detects_perfect_separation_lk003(self):
        # Feature that perfectly separates classes
        df = pd.DataFrame({
            "separator": [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
            "normal": np.random.randn(10),
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })

        gate = LeakageGate(target_column="target")
        result = gate.run(df)

        assert not result.passed
        # Should detect perfect separation
        separation_issues = [i for i in result.critical_issues if "separator" in str(i) or "separation" in str(i).lower()]
        assert len(separation_issues) > 0 or "separator" in result.recommended_drops


class TestTemporalLeakageDetection:
    def test_detects_temporal_violation_lk004(self):
        # Feature with dates after reference date
        df = pd.DataFrame({
            "future_date": pd.to_datetime(["2024-08-01", "2024-08-15", "2024-09-01"]),
            "normal_feature": [1.0, 2.0, 3.0],
            "target": [0, 1, 0]
        })

        gate = LeakageGate(
            target_column="target",
            reference_date=pd.Timestamp("2024-07-01"),
            date_columns=["future_date"]
        )
        result = gate.run(df)

        assert not result.passed
        # Should flag temporal violation
        temporal_issues = [i for i in result.critical_issues if "future_date" in str(i) or "temporal" in str(i).lower()]
        assert len(temporal_issues) > 0


class TestNearConstantByClass:
    def test_detects_near_constant_by_class_lk008(self):
        # Feature nearly constant within each class
        df = pd.DataFrame({
            "near_constant": [1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "normal": np.random.randn(10),
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })

        gate = LeakageGate(target_column="target")
        result = gate.run(df)

        # Should flag this as suspicious
        assert len(result.high_issues) > 0 or len(result.critical_issues) > 0


class TestGateResult:
    @pytest.fixture
    def clean_df(self):
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.randn(n),
            "target": np.random.choice([0, 1], n)
        })

    def test_result_contains_passed_field(self, clean_df):
        gate = LeakageGate(target_column="target")
        result = gate.run(clean_df)

        assert hasattr(result, "passed")
        assert isinstance(result.passed, bool)

    def test_result_contains_issues_lists(self, clean_df):
        gate = LeakageGate(target_column="target")
        result = gate.run(clean_df)

        assert hasattr(result, "critical_issues")
        assert hasattr(result, "high_issues")
        assert isinstance(result.critical_issues, list)
        assert isinstance(result.high_issues, list)

    def test_result_contains_suspicious_features(self, clean_df):
        gate = LeakageGate(target_column="target")
        result = gate.run(clean_df)

        assert hasattr(result, "suspicious_features")
        assert isinstance(result.suspicious_features, list)

    def test_result_contains_recommended_drops(self, clean_df):
        gate = LeakageGate(target_column="target")
        result = gate.run(clean_df)

        assert hasattr(result, "recommended_drops")
        assert isinstance(result.recommended_drops, list)

    def test_clean_data_passes_gate(self, clean_df):
        gate = LeakageGate(target_column="target")
        result = gate.run(clean_df)

        assert result.passed
        assert len(result.critical_issues) == 0


class TestSeverity:
    def test_severity_levels_exist(self):
        assert hasattr(Severity, "CRITICAL")
        assert hasattr(Severity, "HIGH")
        assert hasattr(Severity, "MEDIUM")
        assert hasattr(Severity, "LOW")


class TestGateConfiguration:
    def test_custom_thresholds(self):
        gate = LeakageGate(
            target_column="target",
            correlation_threshold_critical=0.85,
            correlation_threshold_high=0.60
        )

        assert gate.correlation_threshold_critical == 0.85
        assert gate.correlation_threshold_high == 0.60

    def test_exclude_features_from_check(self):
        np.random.seed(42)
        n = 1000
        target = np.random.choice([0, 1], n)
        leaky = target + np.random.randn(n) * 0.05

        df = pd.DataFrame({
            "allowed_leaky": leaky,
            "target": target
        })

        gate = LeakageGate(
            target_column="target",
            exclude_features=["allowed_leaky"]
        )
        result = gate.run(df)

        # Should pass because the leaky feature is excluded
        assert result.passed


class TestLeakageReport:
    def test_result_has_leakage_report(self):
        np.random.seed(42)
        n = 100
        target = np.random.choice([0, 1], n)
        leaky = target + np.random.randn(n) * 0.05

        df = pd.DataFrame({
            "leaky": leaky,
            "normal": np.random.randn(n),
            "target": target
        })

        gate = LeakageGate(target_column="target")
        result = gate.run(df)

        assert hasattr(result, "leakage_report")
        assert isinstance(result.leakage_report, dict)

    def test_report_contains_correlation_info(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })

        gate = LeakageGate(target_column="target")
        result = gate.run(df)

        assert "correlations" in result.leakage_report


class TestGateBlocksOnCritical:
    def test_gate_fails_on_critical_issues(self):
        np.random.seed(42)
        n = 100
        target = np.random.choice([0, 1], n)
        # Nearly perfect correlation
        leaky = target.astype(float) + np.random.randn(n) * 0.01

        df = pd.DataFrame({
            "leaky": leaky,
            "target": target
        })

        gate = LeakageGate(target_column="target")
        result = gate.run(df)

        assert not result.passed
        assert len(result.critical_issues) > 0


class TestMultipleLeakageTypes:
    def test_detects_multiple_leakage_types(self):
        np.random.seed(42)
        n = 100
        target = np.random.choice([0, 1], n)

        df = pd.DataFrame({
            "high_corr": target.astype(float) + np.random.randn(n) * 0.05,
            "separator": [0] * 50 + [10] * 50,  # Perfect separation
            "normal": np.random.randn(n),
            "target": target
        })
        # Ensure perfect separation matches target
        df.loc[df["target"] == 0, "separator"] = 0
        df.loc[df["target"] == 1, "separator"] = 10

        gate = LeakageGate(target_column="target")
        result = gate.run(df)

        assert not result.passed
        # Should detect both issues
        assert len(result.critical_issues) >= 1


class TestPointInTimeViolationLK009:
    """Tests for LK009: feature_timestamp > label_timestamp violations."""

    def test_detects_point_in_time_violation(self):
        """LK009: Should detect when feature_timestamp > label_timestamp."""
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "feature_timestamp": pd.to_datetime(["2024-06-01", "2024-07-01", "2024-05-01"]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-08-01", "2024-06-01"]),
            "feature1": [100, 200, 300],
            "target": [1, 0, 1]
        })

        gate = LeakageGate(
            target_column="target",
            feature_timestamp_column="feature_timestamp",
            label_timestamp_column="label_timestamp"
        )
        result = gate.run(df)

        assert not result.passed
        lk009_issues = [i for i in result.critical_issues if "LK009" in str(i) or "point-in-time" in str(i).lower()]
        assert len(lk009_issues) > 0

    def test_passes_when_all_timestamps_valid(self):
        """LK009: Should pass when all feature_timestamps <= label_timestamps."""
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]),
            "feature1": np.random.randn(3),
            "target": [1, 0, 1]
        })

        gate = LeakageGate(
            target_column="target",
            feature_timestamp_column="feature_timestamp",
            label_timestamp_column="label_timestamp"
        )
        result = gate.run(df)

        lk009_issues = [i for i in result.critical_issues if "LK009" in str(i)]
        assert len(lk009_issues) == 0

    def test_skips_check_when_no_timestamp_columns(self):
        """LK009: Should skip check when timestamp columns not configured."""
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })

        gate = LeakageGate(target_column="target")
        result = gate.run(df)

        # Should not crash and not flag LK009 issues
        lk009_issues = [i for i in result.critical_issues if "LK009" in str(i)]
        assert len(lk009_issues) == 0


class TestFutureDateInFeaturesLK010:
    """Tests for LK010: feature dates beyond reference timestamp."""

    def test_detects_future_date_in_features(self):
        """LK010: Should detect when feature date columns have future dates."""
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "feature_timestamp": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-01"]),
            "label_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-06-01"]),
            "last_purchase_date": pd.to_datetime(["2024-04-01", "2024-04-15", "2024-02-01"]),  # 2 are after feature_ts
            "target": [1, 0, 1]
        })

        gate = LeakageGate(
            target_column="target",
            feature_timestamp_column="feature_timestamp",
            label_timestamp_column="label_timestamp"
        )
        result = gate.run(df)

        lk010_issues = [i for i in result.critical_issues if "LK010" in str(i) or "future" in str(i).lower()]
        assert len(lk010_issues) > 0

    def test_passes_when_all_feature_dates_valid(self):
        """LK010: Should pass when all feature dates <= feature_timestamp."""
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "feature_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-06-01"]),
            "label_timestamp": pd.to_datetime(["2024-09-01", "2024-09-01", "2024-09-01"]),
            "last_login_date": pd.to_datetime(["2024-05-01", "2024-05-15", "2024-05-20"]),  # All before feature_ts
            "target": [1, 0, 1]
        })

        gate = LeakageGate(
            target_column="target",
            feature_timestamp_column="feature_timestamp",
            label_timestamp_column="label_timestamp"
        )
        result = gate.run(df)

        lk010_issues = [i for i in result.critical_issues if "LK010" in str(i)]
        assert len(lk010_issues) == 0

    def test_detects_multiple_future_date_columns(self):
        """LK010: Should detect multiple columns with future dates."""
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "feature_timestamp": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            "label_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "last_login": pd.to_datetime(["2024-04-01", "2024-04-01"]),  # Future
            "last_purchase": pd.to_datetime(["2024-05-01", "2024-05-01"]),  # Future
            "signup_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),  # Valid
            "target": [1, 0]
        })

        gate = LeakageGate(
            target_column="target",
            feature_timestamp_column="feature_timestamp",
            label_timestamp_column="label_timestamp"
        )
        result = gate.run(df)

        lk010_issues = [i for i in result.critical_issues if "LK010" in str(i) or "future" in str(i).lower()]
        # Should detect at least 2 issues (last_login and last_purchase)
        assert len(lk010_issues) >= 2


class TestAntiPatternsIntegration:
    """Integration tests for anti-patterns identified in the leakage-safe refactoring plan."""

    def test_detects_joins_without_timestamps_causing_leakage(self):
        """Anti-pattern: Joins without timestamps can cause temporal leakage.

        This test simulates a scenario where feature aggregations from future
        data leak into training features.
        """
        np.random.seed(42)
        n = 100
        target = np.random.choice([0, 1], n, p=[0.7, 0.3])

        # Simulate a feature that was computed using future data
        # (e.g., average purchase amount including future purchases)
        future_aware_avg = target * 50 + np.random.randn(n) * 5  # Highly predictive

        df = pd.DataFrame({
            "customer_id": range(n),
            "future_aware_avg_purchase": future_aware_avg,
            "normal_feature": np.random.randn(n),
            "target": target
        })

        gate = LeakageGate(target_column="target")
        result = gate.run(df)

        # Should detect the leaky aggregation feature
        assert not result.passed
        leaky_issues = [i for i in result.critical_issues if "future_aware" in str(i).lower()]
        assert len(leaky_issues) > 0 or len(result.critical_issues) > 0

    def test_combined_point_in_time_and_correlation_leakage(self):
        """Integration test: both temporal and correlation leakage present."""
        n = 100
        target = np.array([0] * 50 + [1] * 50)

        df = pd.DataFrame({
            "customer_id": range(n),
            "feature_timestamp": pd.to_datetime(["2024-06-01"] * 50 + ["2024-05-01"] * 50),  # Some after label
            "label_timestamp": pd.to_datetime(["2024-04-01"] * n),
            "leaky_corr_feature": target.astype(float) + np.random.randn(n) * 0.05,
            "normal_feature": np.random.randn(n),
            "target": target
        })

        gate = LeakageGate(
            target_column="target",
            feature_timestamp_column="feature_timestamp",
            label_timestamp_column="label_timestamp"
        )
        result = gate.run(df)

        assert not result.passed
        # Should detect both types of issues
        assert len(result.critical_issues) >= 1

    def test_snapshot_based_training_prevents_leakage(self):
        """Verify that using snapshots with proper temporal filtering prevents leakage."""
        import tempfile
        from datetime import datetime
        from pathlib import Path

        from customer_retention.stages.temporal import ScenarioDetector, UnifiedDataPreparer

        np.random.seed(42)
        n = 100

        # Create production-style data
        base_date = datetime(2024, 1, 1)
        feature_timestamps = [base_date + pd.Timedelta(days=i) for i in range(n)]
        label_timestamps = [ft + pd.Timedelta(days=90) for ft in feature_timestamps]

        df = pd.DataFrame({
            "customer_id": range(n),
            "feature_timestamp": feature_timestamps,
            "label_timestamp": label_timestamps,
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "churned": np.random.choice([0, 1], n, p=[0.7, 0.3])
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            detector = ScenarioDetector()
            scenario, ts_config, discovery_result = detector.detect(df, "churned")

            preparer = UnifiedDataPreparer(tmp_path, ts_config)
            unified_df = preparer.prepare_from_raw(df, "churned", "customer_id")

            # Create snapshot
            cutoff = datetime(2024, 5, 1)
            snapshot_df, metadata = preparer.create_training_snapshot(unified_df, cutoff)

            # Verify snapshot has proper temporal constraints
            gate = LeakageGate(
                target_column="target",
                feature_timestamp_column="feature_timestamp",
                label_timestamp_column="label_timestamp"
            )

            # Run on snapshot - should not have point-in-time violations
            exclude_cols = ["entity_id", "target", "feature_timestamp", "label_timestamp", "label_available_flag"]
            numeric_cols = snapshot_df.select_dtypes(include=[np.number]).columns.tolist()
            check_cols = [c for c in numeric_cols if c not in exclude_cols]

            if check_cols:
                result = gate.run(snapshot_df[check_cols + ["target", "feature_timestamp", "label_timestamp"]])
                lk009_issues = [i for i in result.critical_issues if "LK009" in str(i)]
                # Snapshot should not have point-in-time violations
                assert len(lk009_issues) == 0
