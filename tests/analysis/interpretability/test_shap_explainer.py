import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from customer_retention.analysis.interpretability import (
    ShapExplainer, GlobalExplanation, FeatureImportance
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "recency": np.random.randint(1, 365, n),
        "frequency": np.random.randint(1, 50, n),
        "monetary": np.random.uniform(10, 500, n),
        "tenure": np.random.randint(30, 1000, n),
        "engagement": np.random.uniform(0, 1, n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


@pytest.fixture
def trained_rf_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_lr_model(sample_data):
    X, y = sample_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


class TestShapExplainerCreation:
    def test_creates_tree_explainer_for_tree_model(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        assert explainer.explainer_type == "tree"

    def test_creates_linear_explainer_for_linear_model(self, trained_lr_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_lr_model, X)
        assert explainer.explainer_type == "linear"


class TestGlobalExplanation:
    def test_ac7_1_shap_values_compute_without_error(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.explain_global(X.head(50))
        assert result is not None
        assert isinstance(result, GlobalExplanation)

    def test_ac7_2_feature_ranking_returned(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.explain_global(X.head(50))
        assert len(result.feature_importance) == X.shape[1]
        for fi in result.feature_importance:
            assert isinstance(fi, FeatureImportance)
            assert fi.feature_name in X.columns
            assert fi.importance >= 0

    def test_features_sorted_by_importance(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.explain_global(X.head(50))
        importances = [fi.importance for fi in result.feature_importance]
        assert importances == sorted(importances, reverse=True)

    def test_shap_values_matrix_has_correct_shape(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.explain_global(X.head(50))
        assert result.shap_values.shape == (50, X.shape[1])

    def test_expected_value_returned(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.explain_global(X.head(50))
        assert result.expected_value is not None


class TestBusinessTranslations:
    def test_ac7_4_business_translations_provided(self, trained_rf_model, sample_data):
        X, _ = sample_data
        translations = {
            "recency": "Days since last order",
            "frequency": "Order frequency",
            "monetary": "Average order value",
            "tenure": "Customer tenure",
            "engagement": "Email engagement score",
        }
        explainer = ShapExplainer(trained_rf_model, X, feature_translations=translations)
        result = explainer.explain_global(X.head(50))
        for fi in result.feature_importance:
            assert fi.business_description is not None
            assert fi.business_description == translations.get(fi.feature_name, fi.feature_name)


class TestTopNFeatures:
    def test_returns_top_n_features(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.explain_global(X.head(50), top_n=3)
        assert len(result.feature_importance) == 3

    def test_all_features_when_top_n_exceeds_count(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.explain_global(X.head(50), top_n=100)
        assert len(result.feature_importance) == X.shape[1]


class TestMeanAbsoluteShap:
    def test_mean_absolute_shap_calculated(self, trained_rf_model, sample_data):
        X, _ = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.explain_global(X.head(50))
        for fi in result.feature_importance:
            assert hasattr(fi, "mean_abs_shap")
            assert fi.mean_abs_shap >= 0


class TestPermutationImportance:
    def test_permutation_importance_calculated(self, trained_rf_model, sample_data):
        X, y = sample_data
        explainer = ShapExplainer(trained_rf_model, X)
        result = explainer.calculate_permutation_importance(X.head(100), y.head(100))
        assert len(result) == X.shape[1]
        for feature, importance in result.items():
            assert feature in X.columns
