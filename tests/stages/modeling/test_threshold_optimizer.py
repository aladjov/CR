import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from customer_retention.stages.modeling import (
    ThresholdOptimizer, OptimizationObjective, ThresholdResult
)


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


class TestOptimizationObjective:
    def test_objective_enum_has_required_values(self):
        assert hasattr(OptimizationObjective, "MIN_COST")
        assert hasattr(OptimizationObjective, "MAX_F1")
        assert hasattr(OptimizationObjective, "MAX_F2")
        assert hasattr(OptimizationObjective, "TARGET_RECALL")
        assert hasattr(OptimizationObjective, "TARGET_PRECISION")


class TestMinCostOptimization:
    def test_finds_min_cost_threshold(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.MIN_COST,
            cost_fn=100,
            cost_fp=10,
        )
        result = optimizer.optimize(trained_model, X, y)

        assert result.optimal_threshold is not None
        assert 0 <= result.optimal_threshold <= 1

    def test_returns_cost_at_threshold(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.MIN_COST,
            cost_fn=100,
            cost_fp=10,
        )
        result = optimizer.optimize(trained_model, X, y)

        assert result.cost_at_threshold is not None


class TestMaxF1Optimization:
    def test_finds_max_f1_threshold(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(objective=OptimizationObjective.MAX_F1)
        result = optimizer.optimize(trained_model, X, y)

        assert result.optimal_threshold is not None
        assert 0 <= result.optimal_threshold <= 1


class TestMaxF2Optimization:
    def test_finds_max_f2_threshold(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(objective=OptimizationObjective.MAX_F2)
        result = optimizer.optimize(trained_model, X, y)

        assert result.optimal_threshold is not None


class TestTargetRecall:
    def test_finds_threshold_for_target_recall(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.TARGET_RECALL,
            target_recall=0.80,
        )
        result = optimizer.optimize(trained_model, X, y)

        assert result.optimal_threshold is not None


class TestThresholdResult:
    def test_result_contains_required_fields(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(objective=OptimizationObjective.MAX_F1)
        result = optimizer.optimize(trained_model, X, y)

        assert hasattr(result, "optimal_threshold")
        assert hasattr(result, "threshold_metrics")
        assert hasattr(result, "comparison_default")

    def test_threshold_metrics_at_optimal(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(objective=OptimizationObjective.MAX_F1)
        result = optimizer.optimize(trained_model, X, y)

        assert "precision" in result.threshold_metrics
        assert "recall" in result.threshold_metrics
        assert "f1" in result.threshold_metrics


class TestComparisonWithDefault:
    def test_compares_with_default_threshold(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(objective=OptimizationObjective.MAX_F1)
        result = optimizer.optimize(trained_model, X, y)

        assert result.comparison_default is not None
        assert "default_threshold" in result.comparison_default
        assert "default_f1" in result.comparison_default


class TestCostConfiguration:
    def test_custom_cost_values(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.MIN_COST,
            cost_fn=200,
            cost_fp=5,
        )
        result = optimizer.optimize(trained_model, X, y)

        assert result.cost_at_threshold is not None


class TestThresholdSearch:
    def test_searches_threshold_range(self, sample_data, trained_model):
        X, y = sample_data
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.MAX_F1,
            threshold_step=0.05,
        )
        result = optimizer.optimize(trained_model, X, y)

        assert result.optimal_threshold is not None
