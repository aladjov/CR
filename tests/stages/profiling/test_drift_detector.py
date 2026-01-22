import numpy as np
import pandas as pd
import pytest

from customer_retention.core.config import ColumnType
from customer_retention.stages.profiling import BaselineDriftChecker, ColumnProfile, ProfileResult, TypeInference
from customer_retention.stages.profiling.profile_result import NumericMetrics, UniversalMetrics


@pytest.fixture
def baseline_numeric_series():
    """Baseline numeric data with normal distribution."""
    np.random.seed(42)
    return pd.Series(np.random.normal(100, 15, 1000))


@pytest.fixture
def drifted_numeric_series():
    """Drifted numeric data with different mean."""
    np.random.seed(43)
    return pd.Series(np.random.normal(120, 15, 1000))


@pytest.fixture
def baseline_categorical_series():
    """Baseline categorical data."""
    np.random.seed(42)
    return pd.Series(np.random.choice(['A', 'B', 'C', 'D'], 1000, p=[0.4, 0.3, 0.2, 0.1]))


@pytest.fixture
def drifted_categorical_series():
    """Drifted categorical data with different distribution."""
    np.random.seed(43)
    return pd.Series(np.random.choice(['A', 'B', 'C', 'D'], 1000, p=[0.2, 0.3, 0.3, 0.2]))


@pytest.fixture
def new_category_series():
    """Categorical data with new category."""
    np.random.seed(44)
    return pd.Series(np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000, p=[0.3, 0.3, 0.2, 0.1, 0.1]))


class TestBaselineDriftCheckerBasic:
    def test_detector_initialization(self):
        detector = BaselineDriftChecker()
        assert detector is not None
        assert detector.baseline is None

    def test_set_baseline_from_series(self, baseline_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("test_col", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        assert detector.baseline is not None
        assert "test_col" in detector.baseline


class TestNumericDriftDetection:
    def test_ks_test_no_drift(self, baseline_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        # Same distribution should show no drift
        result = detector.detect_drift("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        assert result is not None
        assert result.column_name == "value"
        assert result.has_drift is False
        assert "ks_statistic" in result.metrics

    def test_ks_test_with_drift(self, baseline_numeric_series, drifted_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        result = detector.detect_drift("value", drifted_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        assert result.has_drift is True
        assert result.severity in ["high", "critical"]
        assert "ks_statistic" in result.metrics
        assert result.metrics["ks_statistic"] > 0.05

    def test_psi_calculation_numeric(self, baseline_numeric_series, drifted_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        result = detector.detect_drift("value", drifted_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        assert "psi" in result.metrics
        assert result.metrics["psi"] >= 0

    def test_mean_shift_detection(self, baseline_numeric_series, drifted_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        result = detector.detect_drift("value", drifted_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        assert "mean_shift" in result.metrics
        assert abs(result.metrics["mean_shift"]) > 0

    def test_variance_ratio(self, baseline_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        # Create series with higher variance
        high_var = pd.Series(np.random.normal(100, 30, 1000))
        result = detector.detect_drift("value", high_var, ColumnType.NUMERIC_CONTINUOUS)

        assert "variance_ratio" in result.metrics
        assert result.metrics["variance_ratio"] > 1.5


class TestCategoricalDriftDetection:
    def test_chi_square_no_drift(self, baseline_categorical_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("category", baseline_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        result = detector.detect_drift("category", baseline_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        assert result.has_drift is False
        assert "chi_square_statistic" in result.metrics

    def test_chi_square_with_drift(self, baseline_categorical_series, drifted_categorical_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("category", baseline_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        result = detector.detect_drift("category", drifted_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        assert result.has_drift is True
        assert "chi_square_statistic" in result.metrics

    def test_psi_calculation_categorical(self, baseline_categorical_series, drifted_categorical_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("category", baseline_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        result = detector.detect_drift("category", drifted_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        assert "psi" in result.metrics
        assert result.metrics["psi"] > 0.05

    def test_new_categories_detected(self, baseline_categorical_series, new_category_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("category", baseline_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        result = detector.detect_drift("category", new_category_series, ColumnType.CATEGORICAL_NOMINAL)

        assert "new_categories" in result.metrics
        assert len(result.metrics["new_categories"]) > 0
        assert 'E' in result.metrics["new_categories"]

    def test_missing_categories_detected(self, baseline_categorical_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("category", baseline_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        # Remove category D
        missing_cat = pd.Series(np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]))
        result = detector.detect_drift("category", missing_cat, ColumnType.CATEGORICAL_NOMINAL)

        assert "missing_categories" in result.metrics
        assert len(result.metrics["missing_categories"]) > 0
        assert 'D' in result.metrics["missing_categories"]


class TestDriftSeverity:
    def test_psi_severity_low(self, baseline_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        # Very similar distribution
        similar = baseline_numeric_series + np.random.normal(0, 1, len(baseline_numeric_series))
        result = detector.detect_drift("value", similar, ColumnType.NUMERIC_CONTINUOUS)

        if result.metrics["psi"] < 0.1:
            assert result.severity == "low" or result.has_drift is False

    def test_psi_severity_medium(self, baseline_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        # Moderate shift
        moderate = baseline_numeric_series + 5
        result = detector.detect_drift("value", moderate, ColumnType.NUMERIC_CONTINUOUS)

        if 0.1 <= result.metrics["psi"] < 0.2:
            assert result.severity == "medium"

    def test_psi_severity_high(self, baseline_numeric_series, drifted_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        result = detector.detect_drift("value", drifted_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        if 0.2 <= result.metrics["psi"] < 0.5:
            assert result.severity == "high"

    def test_psi_severity_critical(self, baseline_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        # Major shift
        major = pd.Series(np.random.normal(200, 30, 1000))
        result = detector.detect_drift("value", major, ColumnType.NUMERIC_CONTINUOUS)

        if result.metrics["psi"] >= 0.5:
            assert result.severity == "critical"


class TestBaselineManagement:
    def test_save_and_load_baseline(self, tmp_path, baseline_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        baseline_path = tmp_path / "baseline.json"
        detector.save_baseline(str(baseline_path))

        assert baseline_path.exists()

        new_detector = BaselineDriftChecker()
        new_detector.load_baseline(str(baseline_path))

        assert new_detector.baseline is not None
        assert "value" in new_detector.baseline

    def test_baseline_from_profile_result(self):
        # Create a mock ProfileResult
        column_profiles = {
            "test_col": ColumnProfile(
                column_name="test_col",
                configured_type=ColumnType.NUMERIC_CONTINUOUS,
                inferred_type=TypeInference(
                    inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence="high",
                    evidence=["numeric values"]
                ),
                universal_metrics=UniversalMetrics(
                    total_count=100,
                    null_count=0,
                    null_percentage=0.0,
                    distinct_count=100,
                    distinct_percentage=1.0
                ),
                numeric_metrics=NumericMetrics(
                    mean=100.0,
                    std=15.0,
                    min_value=50.0,
                    max_value=150.0,
                    range_value=100.0,
                    median=100.0,
                    q1=90.0,
                    q3=110.0,
                    iqr=20.0,
                    zero_count=0,
                    zero_percentage=0.0,
                    negative_count=0,
                    negative_percentage=0.0,
                    inf_count=0,
                    inf_percentage=0.0,
                    outlier_count_iqr=5,
                    outlier_count_zscore=3,
                    outlier_percentage=5.0
                )
            )
        }

        profile = ProfileResult(
            dataset_name="test",
            total_rows=100,
            total_columns=1,
            column_profiles=column_profiles,
            profiling_timestamp="2024-01-01T00:00:00",
            profiling_duration_seconds=1.0
        )

        detector = BaselineDriftChecker()
        detector.set_baseline_from_profile(profile)

        assert detector.baseline is not None
        assert "test_col" in detector.baseline


class TestDriftResult:
    def test_drift_result_structure(self, baseline_numeric_series, drifted_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        result = detector.detect_drift("value", drifted_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        assert hasattr(result, 'column_name')
        assert hasattr(result, 'has_drift')
        assert hasattr(result, 'severity')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'recommendations')
        assert isinstance(result.metrics, dict)

    def test_drift_recommendations_provided(self, baseline_numeric_series, drifted_numeric_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("value", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        result = detector.detect_drift("value", drifted_numeric_series, ColumnType.NUMERIC_CONTINUOUS)

        if result.has_drift:
            assert result.recommendations is not None
            assert len(result.recommendations) > 0


class TestMultiColumnDrift:
    def test_detect_drift_all_columns(self, baseline_numeric_series, baseline_categorical_series):
        detector = BaselineDriftChecker()
        detector.set_baseline("numeric_col", baseline_numeric_series, ColumnType.NUMERIC_CONTINUOUS)
        detector.set_baseline("cat_col", baseline_categorical_series, ColumnType.CATEGORICAL_NOMINAL)

        df = pd.DataFrame({
            "numeric_col": baseline_numeric_series,
            "cat_col": baseline_categorical_series
        })

        results = detector.detect_drift_all(df)

        assert len(results) == 2
        assert all(result.column_name in ["numeric_col", "cat_col"] for result in results)
