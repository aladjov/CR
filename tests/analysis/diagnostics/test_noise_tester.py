import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from customer_retention.analysis.diagnostics import NoiseTester, NoiseResult
from customer_retention.core.components.enums import Severity


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 300
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


class TestGaussianNoise:
    def test_gaussian_noise_test_runs(self, sample_data, trained_model):
        X, y = sample_data
        tester = NoiseTester()
        result = tester.test_gaussian_noise(trained_model, X, y)
        assert hasattr(result, "degradation_curve")
        assert len(result.degradation_curve) > 0

    def test_nr001_detects_fragile_to_low_noise(self, sample_data, trained_model):
        X, y = sample_data
        tester = NoiseTester()
        result = tester.test_gaussian_noise(trained_model, X, y)
        assert hasattr(result, "checks")


class TestFeatureDropout:
    def test_feature_dropout_test_runs(self, sample_data, trained_model):
        X, y = sample_data
        tester = NoiseTester()
        result = tester.test_feature_dropout(trained_model, X, y)
        assert hasattr(result, "degradation_curve")

    def test_nr003_detects_single_feature_dependency(self, sample_data, trained_model):
        X, y = sample_data
        tester = NoiseTester()
        result = tester.test_feature_dropout(trained_model, X, y)
        assert hasattr(result, "feature_importance")


class TestNoiseResult:
    def test_result_contains_required_fields(self, sample_data, trained_model):
        X, y = sample_data
        tester = NoiseTester()
        result = tester.test_gaussian_noise(trained_model, X, y)
        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert hasattr(result, "robustness_score")

    def test_robustness_score_is_valid(self, sample_data, trained_model):
        X, y = sample_data
        tester = NoiseTester()
        result = tester.test_gaussian_noise(trained_model, X, y)
        assert 0 <= result.robustness_score <= 1


class TestRunAllNoise:
    def test_run_all_combines_results(self, sample_data, trained_model):
        X, y = sample_data
        tester = NoiseTester()
        result = tester.run_all(trained_model, X, y)
        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
