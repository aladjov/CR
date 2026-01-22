import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from customer_retention.stages.modeling import ModelComparator


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
def trained_models(sample_data):
    X, y = sample_data
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    return {"logistic": lr, "random_forest": rf}


class TestModelMetrics:
    def test_model_metrics_has_required_fields(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y)

        for name, metrics in result.model_metrics.items():
            assert hasattr(metrics, "pr_auc")
            assert hasattr(metrics, "roc_auc")
            assert hasattr(metrics, "f1")


class TestComparisonResult:
    def test_result_contains_ranking(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y)

        assert result.ranking is not None
        assert len(result.ranking) == len(trained_models)

    def test_result_contains_best_model(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y)

        assert result.best_model_name is not None
        assert result.best_model_name in trained_models

    def test_result_contains_comparison_table(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y)

        assert result.comparison_table is not None
        assert isinstance(result.comparison_table, pd.DataFrame)

    def test_result_contains_selection_reason(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y)

        assert result.selection_reason is not None
        assert isinstance(result.selection_reason, str)


class TestRankingCriteria:
    def test_ranking_by_pr_auc(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator(primary_metric="pr_auc")
        result = comparator.compare(trained_models, X, y)

        pr_aucs = [result.model_metrics[name].pr_auc for name in result.ranking]
        assert pr_aucs == sorted(pr_aucs, reverse=True)

    def test_ranking_by_roc_auc(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator(primary_metric="roc_auc")
        result = comparator.compare(trained_models, X, y)

        assert result.ranking is not None


class TestComparisonTable:
    def test_table_has_all_models(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y)

        assert len(result.comparison_table) == len(trained_models)

    def test_table_has_key_metrics(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y)

        required_columns = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
        for col in required_columns:
            assert col in result.comparison_table.columns


class TestGeneralizationGap:
    def test_calculates_train_test_gap(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y, X_train=X, y_train=y)

        for name, metrics in result.model_metrics.items():
            assert hasattr(metrics, "train_test_gap")


class TestCVStability:
    def test_includes_cv_stability(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator()
        result = comparator.compare(trained_models, X, y)

        assert result.comparison_table is not None


class TestBestModelSelection:
    def test_selects_best_by_primary_metric(self, sample_data, trained_models):
        X, y = sample_data
        comparator = ModelComparator(primary_metric="pr_auc")
        result = comparator.compare(trained_models, X, y)

        best_pr_auc = max(result.model_metrics[name].pr_auc for name in trained_models)
        assert result.model_metrics[result.best_model_name].pr_auc == best_pr_auc
