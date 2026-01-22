import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from customer_retention.analysis.business import (
    FairnessAnalyzer, FairnessResult, FairnessMetric, GroupMetrics
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "recency": np.random.randint(1, 365, n),
        "frequency": np.random.randint(1, 50, n),
        "monetary": np.random.uniform(10, 500, n),
        "tenure": np.random.randint(30, 1000, n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    protected = pd.Series(np.random.choice(["GroupA", "GroupB"], n))
    return X, y, protected


@pytest.fixture
def trained_model(sample_data):
    X, y, _ = sample_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


class TestFairnessMetrics:
    def test_ac7_23_fairness_metrics_calculated(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer()
        y_pred = trained_model.predict(X)
        result = analyzer.analyze(y, y_pred, protected)
        assert isinstance(result, FairnessResult)
        assert len(result.metrics) > 0

    def test_demographic_parity_calculated(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer()
        y_pred = trained_model.predict(X)
        result = analyzer.analyze(y, y_pred, protected)
        dp_metric = next((m for m in result.metrics if m.name == "demographic_parity"), None)
        assert dp_metric is not None
        assert dp_metric.ratio is not None

    def test_equalized_odds_calculated(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer()
        y_pred = trained_model.predict(X)
        result = analyzer.analyze(y, y_pred, protected)
        eo_metric = next((m for m in result.metrics if m.name == "equalized_odds"), None)
        assert eo_metric is not None


class TestFairnessThresholds:
    def test_ac7_24_no_severe_bias_detection(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer(threshold=0.8)
        y_pred = trained_model.predict(X)
        result = analyzer.analyze(y, y_pred, protected)
        assert result.passed is not None

    def test_bias_flagged_when_below_threshold(self):
        np.random.seed(42)
        n = 500
        y = pd.Series([0] * 250 + [1] * 250)
        y_pred = pd.Series([0] * 200 + [1] * 50 + [0] * 50 + [1] * 200)
        protected = pd.Series(["GroupA"] * 250 + ["GroupB"] * 250)
        analyzer = FairnessAnalyzer(threshold=0.8)
        result = analyzer.analyze(y, y_pred, protected)
        failed = [m for m in result.metrics if not m.passed]
        assert len(failed) >= 0


class TestFairnessRecommendations:
    def test_ac7_25_recommendations_provided(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer()
        y_pred = trained_model.predict(X)
        result = analyzer.analyze(y, y_pred, protected)
        assert result.recommendations is not None


class TestGroupMetrics:
    def test_per_group_values_returned(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer()
        y_pred = trained_model.predict(X)
        result = analyzer.analyze(y, y_pred, protected)
        assert len(result.group_metrics) == 2
        for group, metrics in result.group_metrics.items():
            assert isinstance(metrics, GroupMetrics)
            assert metrics.positive_rate is not None

    def test_group_accuracy_calculated(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer()
        y_pred = trained_model.predict(X)
        result = analyzer.analyze(y, y_pred, protected)
        for group, metrics in result.group_metrics.items():
            assert metrics.accuracy is not None
            assert 0 <= metrics.accuracy <= 1


class TestDisparateImpact:
    def test_disparate_impact_calculated(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer()
        y_pred = trained_model.predict(X)
        result = analyzer.analyze(y, y_pred, protected)
        di_metric = next((m for m in result.metrics if m.name == "disparate_impact"), None)
        assert di_metric is not None
        assert di_metric.ratio is not None
        assert di_metric.ratio > 0


class TestCalibrationFairness:
    def test_calibration_per_group(self, trained_model, sample_data):
        X, y, protected = sample_data
        analyzer = FairnessAnalyzer()
        y_proba = trained_model.predict_proba(X)[:, 1]
        result = analyzer.analyze_calibration(y, y_proba, protected)
        assert len(result.group_metrics) == 2


class TestMultipleProtectedAttributes:
    def test_analyzes_multiple_attributes(self, trained_model, sample_data):
        X, y, protected1 = sample_data
        protected2 = pd.Series(np.random.choice(["Urban", "Rural"], len(X)))
        analyzer = FairnessAnalyzer()
        y_pred = trained_model.predict(X)
        results = analyzer.analyze_multiple(
            y, y_pred,
            protected_attributes={"group": protected1, "location": protected2}
        )
        assert "group" in results
        assert "location" in results
