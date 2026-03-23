import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from customer_retention.analysis.interpretability import (
    FeatureImportance,
    GlobalExplanation,
    ShapExplainer,
    select_risk_stratified_sample,
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


class TestSelectRiskStratifiedSample:
    def test_returns_all_when_n_samples_exceeds_total(self):
        probs = np.array([0.1, 0.5, 0.9])
        result = select_risk_stratified_sample(probs, n_samples=10)
        np.testing.assert_array_equal(result, np.arange(3))

    def test_returns_exact_n_samples(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 1000)
        result = select_risk_stratified_sample(probs, n_samples=200)
        assert len(result) == 200

    def test_indices_are_valid(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 500)
        result = select_risk_stratified_sample(probs, n_samples=100)
        assert all(0 <= i < 500 for i in result)

    def test_no_duplicate_indices(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 500)
        result = select_risk_stratified_sample(probs, n_samples=100)
        assert len(result) == len(set(result))

    def test_covers_all_risk_bins(self):
        probs = np.concatenate([
            np.full(200, 0.1),
            np.full(200, 0.3),
            np.full(200, 0.5),
            np.full(200, 0.7),
            np.full(200, 0.9),
        ])
        result = select_risk_stratified_sample(probs, n_samples=50, n_bins=5)
        selected_probs = probs[result]
        unique_values = set(selected_probs)
        assert len(unique_values) == 5

    def test_priority_mask_entities_always_included(self):
        probs = np.linspace(0, 1, 100)
        priority = np.zeros(100, dtype=bool)
        priority[[0, 50, 99]] = True
        result = select_risk_stratified_sample(probs, n_samples=20, priority_mask=priority)
        assert 0 in result
        assert 50 in result
        assert 99 in result

    def test_priority_exceeds_budget_caps_to_n_samples(self):
        probs = np.linspace(0, 1, 100)
        priority = np.ones(100, dtype=bool)
        result = select_risk_stratified_sample(probs, n_samples=10, priority_mask=priority)
        assert len(result) == 10

    def test_deterministic_with_same_inputs(self):
        probs = np.random.default_rng(0).uniform(0, 1, 500)
        r1 = select_risk_stratified_sample(probs, n_samples=50)
        r2 = select_risk_stratified_sample(probs, n_samples=50)
        np.testing.assert_array_equal(r1, r2)

    def test_sorted_output(self):
        probs = np.random.default_rng(0).uniform(0, 1, 500)
        result = select_risk_stratified_sample(probs, n_samples=100)
        np.testing.assert_array_equal(result, np.sort(result))

    def test_extreme_single_bin(self):
        probs = np.full(100, 0.5)
        result = select_risk_stratified_sample(probs, n_samples=20, n_bins=5)
        assert len(result) == 20

    def test_skewed_distribution_still_samples_tails(self):
        probs = np.concatenate([np.full(950, 0.1), np.full(50, 0.9)])
        result = select_risk_stratified_sample(probs, n_samples=100, n_bins=5)
        selected_probs = probs[result]
        assert np.any(selected_probs > 0.5)
        assert np.any(selected_probs < 0.5)

    def test_priority_plus_stratified_fill(self):
        probs = np.linspace(0, 1, 200)
        priority = np.zeros(200, dtype=bool)
        priority[[0, 1, 2]] = True
        result = select_risk_stratified_sample(probs, n_samples=50, priority_mask=priority)
        assert len(result) == 50
        assert all(i in result for i in [0, 1, 2])
