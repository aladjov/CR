import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from customer_retention.stages.modeling import ModelEvaluator, EvaluationResult


@pytest.fixture
def binary_classification_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


@pytest.fixture
def trained_model(binary_classification_data):
    X, y = binary_classification_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


class TestPrimaryMetrics:
    def test_calculates_pr_auc(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "pr_auc" in result.metrics
        assert 0 <= result.metrics["pr_auc"] <= 1

    def test_calculates_average_precision(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "average_precision" in result.metrics
        assert 0 <= result.metrics["average_precision"] <= 1

    def test_calculates_f1_score(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "f1" in result.metrics
        assert 0 <= result.metrics["f1"] <= 1

    def test_calculates_recall(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "recall" in result.metrics
        assert 0 <= result.metrics["recall"] <= 1

    def test_calculates_precision(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "precision" in result.metrics
        assert 0 <= result.metrics["precision"] <= 1


class TestSecondaryMetrics:
    def test_calculates_roc_auc(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "roc_auc" in result.metrics
        assert 0 <= result.metrics["roc_auc"] <= 1

    def test_calculates_accuracy(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "accuracy" in result.metrics
        assert 0 <= result.metrics["accuracy"] <= 1

    def test_calculates_balanced_accuracy(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "balanced_accuracy" in result.metrics
        assert 0 <= result.metrics["balanced_accuracy"] <= 1

    def test_calculates_brier_score(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "brier_score" in result.metrics
        assert 0 <= result.metrics["brier_score"] <= 1

    def test_calculates_log_loss(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "log_loss" in result.metrics
        assert result.metrics["log_loss"] >= 0


class TestConfusionMatrix:
    def test_generates_confusion_matrix(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape == (2, 2)

    def test_confusion_matrix_sums_to_total(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert result.confusion_matrix.sum() == len(y)


class TestClassificationReport:
    def test_generates_classification_report(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert result.classification_report is not None
        assert isinstance(result.classification_report, dict)


class TestCurves:
    def test_generates_roc_curve_data(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "roc_curve" in result.curves
        assert "fpr" in result.curves["roc_curve"]
        assert "tpr" in result.curves["roc_curve"]
        assert "thresholds" in result.curves["roc_curve"]

    def test_generates_pr_curve_data(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "pr_curve" in result.curves
        assert "precision" in result.curves["pr_curve"]
        assert "recall" in result.curves["pr_curve"]
        assert "thresholds" in result.curves["pr_curve"]


class TestCustomThreshold:
    def test_evaluate_at_custom_threshold(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator(threshold=0.3)
        result = evaluator.evaluate(trained_model, X, y)

        assert result.threshold == 0.3

    def test_metrics_change_with_threshold(self, binary_classification_data, trained_model):
        X, y = binary_classification_data

        evaluator_high = ModelEvaluator(threshold=0.8)
        evaluator_low = ModelEvaluator(threshold=0.2)

        result_high = evaluator_high.evaluate(trained_model, X, y)
        result_low = evaluator_low.evaluate(trained_model, X, y)

        assert result_high.predictions.sum() != result_low.predictions.sum()


class TestEvaluationResult:
    def test_result_contains_required_fields(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert hasattr(result, "metrics")
        assert hasattr(result, "confusion_matrix")
        assert hasattr(result, "classification_report")
        assert hasattr(result, "curves")
        assert hasattr(result, "threshold")
        assert hasattr(result, "predictions")
        assert hasattr(result, "probabilities")


class TestPredictionOutput:
    def test_stores_predictions(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert len(result.predictions) == len(y)
        assert set(result.predictions).issubset({0, 1})

    def test_stores_probabilities(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert len(result.probabilities) == len(y)
        assert np.all(result.probabilities >= 0) and np.all(result.probabilities <= 1)


class TestTrainTestComparison:
    def test_evaluate_train_and_test(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()

        train_result = evaluator.evaluate(trained_model, X, y, dataset_name="train")
        test_result = evaluator.evaluate(trained_model, X, y, dataset_name="test")

        assert train_result.dataset_name == "train"
        assert test_result.dataset_name == "test"


class TestLiftAndGain:
    def test_calculates_lift_at_k(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "lift_at_10" in result.metrics
        assert "lift_at_20" in result.metrics

    def test_calculates_gain_at_k(self, binary_classification_data, trained_model):
        X, y = binary_classification_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(trained_model, X, y)

        assert "gain_at_10" in result.metrics
        assert "gain_at_20" in result.metrics
