import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from customer_retention.analysis.interpretability import (
    CounterfactualGenerator, Counterfactual, CounterfactualChange
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
def trained_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def actionable_features():
    return ["frequency", "engagement"]


class TestCounterfactualGeneration:
    def test_ac7_11_counterfactuals_generate(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        result = generator.generate(X.iloc[0])
        assert isinstance(result, Counterfactual)

    def test_ac7_12_changes_are_actionable(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        result = generator.generate(X.iloc[0])
        for change in result.changes:
            assert change.feature_name in actionable_features

    def test_ac7_13_predictions_flip(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        high_risk_idx = None
        for i in range(len(X)):
            prob = trained_model.predict_proba(X.iloc[[i]])[0, 1]
            if prob > 0.6:
                high_risk_idx = i
                break
        if high_risk_idx is not None:
            result = generator.generate(X.iloc[high_risk_idx], target_class=0, max_iterations=500)
            if len(result.changes) > 0:
                assert result.counterfactual_prediction <= result.original_prediction


class TestCounterfactualChanges:
    def test_changes_contain_required_fields(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        result = generator.generate(X.iloc[0])
        for change in result.changes:
            assert isinstance(change, CounterfactualChange)
            assert change.feature_name is not None
            assert change.original_value is not None
            assert change.new_value is not None

    def test_changes_are_minimal(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        result = generator.generate(X.iloc[0])
        assert len(result.changes) <= len(actionable_features)


class TestFeasibility:
    def test_feasibility_score_calculated(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        result = generator.generate(X.iloc[0])
        assert 0 <= result.feasibility_score <= 1

    def test_changes_within_observed_range(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        result = generator.generate(X.iloc[0])
        for change in result.changes:
            feature = change.feature_name
            min_val = X[feature].min()
            max_val = X[feature].max()
            assert min_val <= change.new_value <= max_val


class TestBusinessInterpretation:
    def test_business_interpretation_provided(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        result = generator.generate(X.iloc[0])
        assert result.business_interpretation is not None
        assert len(result.business_interpretation) > 0


class TestMultipleCounterfactuals:
    def test_generates_diverse_counterfactuals(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        results = generator.generate_diverse(X.iloc[0], n=3)
        assert len(results) <= 3
        assert len(results) >= 1


class TestPrototypeCounterfactual:
    def test_generates_prototype_counterfactual(self, trained_model, sample_data, actionable_features):
        X, y = sample_data
        retained_mask = y == 1
        retained_X = X[retained_mask]
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable_features)
        result = generator.generate_prototype(X.iloc[0], retained_X)
        assert result is not None
        assert result.counterfactual_prediction <= result.original_prediction


class TestCounterfactualConstraints:
    def test_respects_custom_constraints(self, trained_model, sample_data, actionable_features):
        X, _ = sample_data
        constraints = {"frequency": {"min": 10, "max": 30}}
        generator = CounterfactualGenerator(
            trained_model, X,
            actionable_features=actionable_features,
            constraints=constraints
        )
        result = generator.generate(X.iloc[0])
        for change in result.changes:
            if change.feature_name == "frequency":
                assert 10 <= change.new_value <= 30
