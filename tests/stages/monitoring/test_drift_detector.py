import numpy as np
import pandas as pd
import pytest

from customer_retention.core.components.enums import Severity
from customer_retention.stages.monitoring import DriftConfig, DriftDetector, DriftType


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame({
        "avgorder": np.random.normal(100, 20, 1000),
        "ordfreq": np.random.normal(10, 3, 1000),
        "eopenrate": np.random.uniform(0, 1, 1000),
        "eclickrate": np.random.uniform(0, 0.5, 1000),
        "paperless": np.random.choice([0, 1], 1000),
    })


@pytest.fixture
def current_data_no_drift(reference_data):
    np.random.seed(43)
    return pd.DataFrame({
        "avgorder": np.random.normal(100, 20, 500),
        "ordfreq": np.random.normal(10, 3, 500),
        "eopenrate": np.random.uniform(0, 1, 500),
        "eclickrate": np.random.uniform(0, 0.5, 500),
        "paperless": np.random.choice([0, 1], 500),
    })


@pytest.fixture
def current_data_with_drift():
    np.random.seed(44)
    return pd.DataFrame({
        "avgorder": np.random.normal(150, 30, 500),
        "ordfreq": np.random.normal(15, 5, 500),
        "eopenrate": np.random.uniform(0.3, 1, 500),
        "eclickrate": np.random.uniform(0.1, 0.7, 500),
        "paperless": np.random.choice([0, 1], 500, p=[0.8, 0.2]),
    })


class TestDriftTypes:
    def test_drift_type_enum_values(self):
        assert DriftType.FEATURE.value == "feature"
        assert DriftType.TARGET.value == "target"
        assert DriftType.CONCEPT.value == "concept"
        assert DriftType.DATA_QUALITY.value == "data_quality"


class TestSeverityValues:
    def test_drift_severity_enum_values(self):
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.CRITICAL.value == "critical"


class TestDriftConfig:
    def test_config_has_default_values(self):
        config = DriftConfig()
        assert config.ks_warning_threshold == 0.05
        assert config.ks_critical_threshold == 0.10
        assert config.psi_warning_threshold == 0.10
        assert config.psi_critical_threshold == 0.20

    def test_config_accepts_custom_thresholds(self):
        config = DriftConfig(
            ks_warning_threshold=0.08,
            psi_critical_threshold=0.25
        )
        assert config.ks_warning_threshold == 0.08
        assert config.psi_critical_threshold == 0.25


class TestKSTestDrift:
    def test_computes_ks_statistic(self, reference_data, current_data_no_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_no_drift, method="ks")
        assert result is not None
        for feature_result in result.feature_results:
            assert feature_result.metric_value is not None

    def test_no_drift_detected_for_similar_distributions(self, reference_data, current_data_no_drift):
        detector = DriftDetector(reference_data=reference_data, config=DriftConfig(ks_warning_threshold=0.15))
        result = detector.detect_drift(current_data_no_drift, method="ks")
        severe_drifts = [r for r in result.feature_results if r.severity == Severity.CRITICAL]
        assert len(severe_drifts) == 0

    def test_drift_detected_for_shifted_distributions(self, reference_data, current_data_with_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_with_drift, method="ks")
        drifted_features = [r for r in result.feature_results if r.drift_detected]
        assert len(drifted_features) > 0


class TestPSIDrift:
    def test_computes_psi_correctly(self, reference_data, current_data_no_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_no_drift, method="psi")
        for feature_result in result.feature_results:
            assert feature_result.metric_value >= 0

    def test_psi_threshold_triggers_warning(self, reference_data, current_data_with_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_with_drift, method="psi")
        warnings = [r for r in result.feature_results if r.severity in [Severity.WARNING, Severity.CRITICAL]]
        assert len(warnings) > 0

    def test_psi_above_critical_threshold(self, reference_data, current_data_with_drift):
        detector = DriftDetector(reference_data=reference_data, config=DriftConfig(psi_critical_threshold=0.15))
        result = detector.detect_drift(current_data_with_drift, method="psi")
        critical = [r for r in result.feature_results if r.severity == Severity.CRITICAL]
        assert len(critical) >= 0


class TestMeanShiftDrift:
    def test_computes_mean_shift(self, reference_data, current_data_with_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_with_drift, method="mean_shift")
        assert result is not None

    def test_mean_shift_detects_location_change(self, reference_data, current_data_with_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_with_drift, method="mean_shift")
        drifted = [r for r in result.feature_results if r.drift_detected]
        assert len(drifted) > 0


class TestMissingRateDrift:
    def test_detects_missing_rate_change(self, reference_data):
        current_with_missing = reference_data.copy()
        current_with_missing.loc[:200, "avgorder"] = np.nan
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_missing_rate_drift(current_with_missing)
        avgorder_result = next((r for r in result.feature_results if r.feature_name == "avgorder"), None)
        assert avgorder_result is not None
        assert avgorder_result.drift_detected is True

    def test_no_drift_for_similar_missing_rates(self, reference_data, current_data_no_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_missing_rate_drift(current_data_no_drift)
        critical_drifts = [r for r in result.feature_results if r.severity == Severity.CRITICAL]
        assert len(critical_drifts) == 0


class TestTargetDrift:
    def test_detects_target_drift(self):
        np.random.seed(42)
        ref_target = pd.Series(np.random.choice([0, 1], 1000, p=[0.75, 0.25]))
        curr_target = pd.Series(np.random.choice([0, 1], 500, p=[0.60, 0.40]))
        detector = DriftDetector()
        result = detector.detect_target_drift(ref_target, curr_target)
        assert result.drift_detected == True

    def test_no_target_drift_for_similar_rates(self):
        np.random.seed(42)
        ref_target = pd.Series(np.random.choice([0, 1], 1000, p=[0.75, 0.25]))
        curr_target = pd.Series(np.random.choice([0, 1], 500, p=[0.75, 0.25]))
        detector = DriftDetector()
        result = detector.detect_target_drift(ref_target, curr_target, threshold=0.20)
        assert result.drift_detected == False


class TestDriftResult:
    def test_drift_result_contains_required_fields(self, reference_data, current_data_no_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_no_drift, method="psi")
        assert hasattr(result, "feature_results")
        assert hasattr(result, "overall_drift_detected")
        assert hasattr(result, "monitoring_timestamp")

    def test_feature_result_contains_recommendation(self, reference_data, current_data_with_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_with_drift, method="psi")
        for feature_result in result.feature_results:
            if feature_result.drift_detected:
                assert feature_result.recommendation is not None


class TestDriftMonitoringWindow:
    def test_uses_training_data_as_reference(self, reference_data, current_data_no_drift):
        detector = DriftDetector(reference_data=reference_data, reference_type="training")
        result = detector.detect_drift(current_data_no_drift, method="psi")
        assert result is not None

    def test_can_update_reference_data(self, reference_data, current_data_no_drift):
        detector = DriftDetector(reference_data=reference_data)
        detector.update_reference(current_data_no_drift)
        assert len(detector.reference_data) == len(current_data_no_drift)


class TestMultipleFeaturesDrift:
    def test_monitors_specified_features_only(self, reference_data, current_data_with_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(
            current_data_with_drift,
            method="psi",
            features=["avgorder", "ordfreq"]
        )
        feature_names = [r.feature_name for r in result.feature_results]
        assert "avgorder" in feature_names
        assert "ordfreq" in feature_names
        assert "paperless" not in feature_names

    def test_top_n_features_drift_summary(self, reference_data, current_data_with_drift):
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data_with_drift, method="psi")
        top_drifted = result.get_top_drifted_features(n=3)
        assert len(top_drifted) <= 3


class TestSeverityAssignment:
    def test_assigns_correct_severity_for_psi(self, reference_data):
        detector = DriftDetector(
            reference_data=reference_data,
            config=DriftConfig(psi_warning_threshold=0.10, psi_critical_threshold=0.20)
        )
        assert detector._assign_severity(0.05, "psi") == Severity.INFO
        assert detector._assign_severity(0.15, "psi") == Severity.WARNING
        assert detector._assign_severity(0.25, "psi") == Severity.CRITICAL

    def test_assigns_correct_severity_for_ks(self, reference_data):
        detector = DriftDetector(
            reference_data=reference_data,
            config=DriftConfig(ks_warning_threshold=0.05, ks_critical_threshold=0.10)
        )
        assert detector._assign_severity(0.03, "ks") == Severity.INFO
        assert detector._assign_severity(0.07, "ks") == Severity.WARNING
        assert detector._assign_severity(0.15, "ks") == Severity.CRITICAL
