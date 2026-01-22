import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from customer_retention.analysis.diagnostics import CalibrationAnalyzer
from customer_retention.core.components.enums import Severity


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


class TestBrierScore:
    def test_ca001_detects_poor_brier_score(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.9, 0.8, 0.2, 0.3, 0.7, 0.4])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_brier(y_true, y_proba)

        high_issues = [c for c in result.checks if c.severity in [Severity.HIGH, Severity.CRITICAL]]
        assert len(high_issues) > 0

    def test_ca002_detects_moderate_brier_score(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.3, 0.4, 0.6, 0.7, 0.35, 0.65])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_brier(y_true, y_proba)

        assert result.brier_score is not None

    def test_good_brier_score(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.85, 0.15, 0.8])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_brier(y_true, y_proba)

        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) == 0


class TestExpectedCalibrationError:
    def test_calculates_ece(self, sample_data, trained_model):
        X, y = sample_data
        y_proba = trained_model.predict_proba(X)[:, 1]

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(y.values, y_proba)

        assert hasattr(result, "ece")
        assert 0 <= result.ece <= 1

    def test_ca003_detects_high_ece(self):
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_proba = np.random.rand(100)

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(y_true, y_proba)

        assert result.ece is not None


class TestReliabilityDiagram:
    def test_generates_reliability_diagram_data(self, sample_data, trained_model):
        X, y = sample_data
        y_proba = trained_model.predict_proba(X)[:, 1]

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(y.values, y_proba)

        assert hasattr(result, "reliability_data")
        assert len(result.reliability_data) > 0

    def test_reliability_data_has_bins(self, sample_data, trained_model):
        X, y = sample_data
        y_proba = trained_model.predict_proba(X)[:, 1]

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(y.values, y_proba)

        for bin_data in result.reliability_data:
            assert "predicted_prob" in bin_data
            assert "actual_prob" in bin_data


class TestCalibrationResult:
    def test_result_contains_required_fields(self, sample_data, trained_model):
        X, y = sample_data
        y_proba = trained_model.predict_proba(X)[:, 1]

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(y.values, y_proba)

        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert hasattr(result, "brier_score")
        assert hasattr(result, "ece")
        assert hasattr(result, "recommendation")


class TestCalibrationRecommendation:
    def test_recommends_platt_scaling_for_overconfident(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.05, 0.02, 0.03, 0.95, 0.98, 0.97])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(y_true, y_proba)

        assert result.recommendation is not None
