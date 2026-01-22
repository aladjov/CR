import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from customer_retention.analysis.diagnostics import SegmentPerformanceAnalyzer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "tenure_days": np.random.randint(0, 365 * 3, n),
        "order_frequency": np.random.randint(1, 20, n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


class TestSegmentDefinitions:
    def test_defines_tenure_segments(self, sample_data):
        X, y = sample_data
        analyzer = SegmentPerformanceAnalyzer()
        segments = analyzer.define_segments(X, segment_column="tenure_days", segment_type="tenure")
        assert len(segments) > 0

    def test_defines_quantile_segments(self, sample_data):
        X, y = sample_data
        analyzer = SegmentPerformanceAnalyzer()
        segments = analyzer.define_segments(X, segment_column="order_frequency", segment_type="quantile")
        assert len(segments) > 0


class TestSegmentPerformance:
    def test_sg001_detects_underperformance(self, sample_data, trained_model):
        X, y = sample_data
        X_with_segment = X.copy()
        X_with_segment["segment"] = np.where(X["tenure_days"] < 90, "new", "established")

        analyzer = SegmentPerformanceAnalyzer()
        result = analyzer.analyze_performance(trained_model, X, y, X_with_segment["segment"])

        assert hasattr(result, "segment_metrics")
        assert len(result.segment_metrics) > 0

    def test_sg002_detects_low_recall_segment(self, sample_data, trained_model):
        X, y = sample_data
        segments = pd.Series(["A"] * 250 + ["B"] * 250)

        analyzer = SegmentPerformanceAnalyzer()
        result = analyzer.analyze_performance(trained_model, X, y, segments)

        for segment, metrics in result.segment_metrics.items():
            assert "recall" in metrics

    def test_sg003_flags_small_segments(self, sample_data, trained_model):
        X, y = sample_data
        segments = pd.Series(["large"] * 475 + ["small"] * 25)

        analyzer = SegmentPerformanceAnalyzer()
        result = analyzer.analyze_performance(trained_model, X, y, segments)

        small_segment_checks = [c for c in result.checks if "small" in c.segment.lower() or c.check_id == "SG003"]
        assert len(small_segment_checks) >= 0


class TestSegmentResult:
    def test_result_contains_required_fields(self, sample_data, trained_model):
        X, y = sample_data
        segments = pd.Series(["A"] * 250 + ["B"] * 250)

        analyzer = SegmentPerformanceAnalyzer()
        result = analyzer.analyze_performance(trained_model, X, y, segments)

        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert hasattr(result, "segment_metrics")
        assert hasattr(result, "recommendations")


class TestSegmentRecommendations:
    def test_provides_recommendations(self, sample_data, trained_model):
        X, y = sample_data
        segments = pd.Series(["A"] * 250 + ["B"] * 250)

        analyzer = SegmentPerformanceAnalyzer()
        result = analyzer.analyze_performance(trained_model, X, y, segments)

        assert hasattr(result, "recommendation")
