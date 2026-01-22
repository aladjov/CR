import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from customer_retention.analysis.diagnostics import ErrorAnalyzer, ErrorAnalysisResult, ErrorPattern


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


class TestErrorCategories:
    def test_identifies_false_positives(self, sample_data, trained_model):
        X, y = sample_data
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)
        assert hasattr(result, "false_positives")
        assert result.false_positives is not None

    def test_identifies_false_negatives(self, sample_data, trained_model):
        X, y = sample_data
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)
        assert hasattr(result, "false_negatives")
        assert result.false_negatives is not None

    def test_identifies_high_confidence_errors(self, sample_data, trained_model):
        X, y = sample_data
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)
        assert hasattr(result, "high_confidence_fp")
        assert hasattr(result, "high_confidence_fn")


class TestErrorPatterns:
    def test_finds_common_error_characteristics(self, sample_data, trained_model):
        X, y = sample_data
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)
        assert hasattr(result, "error_patterns")

    def test_confidence_distribution_for_errors(self, sample_data, trained_model):
        X, y = sample_data
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)
        assert hasattr(result, "fp_confidence_dist")
        assert hasattr(result, "fn_confidence_dist")


class TestErrorAnalysisResult:
    def test_result_contains_required_fields(self, sample_data, trained_model):
        X, y = sample_data
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)
        assert hasattr(result, "total_errors")
        assert hasattr(result, "error_rate")
        assert hasattr(result, "fp_count")
        assert hasattr(result, "fn_count")

    def test_error_rate_is_valid(self, sample_data, trained_model):
        X, y = sample_data
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)
        assert 0 <= result.error_rate <= 1


class TestErrorHypothesis:
    def test_generates_hypotheses_for_errors(self, sample_data, trained_model):
        X, y = sample_data
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_errors(trained_model, X, y)
        assert hasattr(result, "hypotheses")
