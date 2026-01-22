import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from customer_retention.stages.modeling import HyperparameterTuner, SearchStrategy


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


class TestSearchStrategy:
    def test_strategy_enum_has_required_values(self):
        assert hasattr(SearchStrategy, "RANDOM_SEARCH")
        assert hasattr(SearchStrategy, "GRID_SEARCH")
        assert hasattr(SearchStrategy, "BAYESIAN")
        assert hasattr(SearchStrategy, "HALVING")


class TestRandomSearch:
    def test_random_search_finds_best_params(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        param_space = {
            "n_estimators": [10, 50],
            "max_depth": [3, 5],
        }

        tuner = HyperparameterTuner(
            strategy=SearchStrategy.RANDOM_SEARCH,
            param_space=param_space,
            n_iter=4,
            cv=3,
            random_state=42,
        )
        result = tuner.tune(model, X, y)

        assert result.best_params is not None
        assert "n_estimators" in result.best_params
        assert "max_depth" in result.best_params

    def test_random_search_returns_best_model(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        param_space = {"n_estimators": [10, 50], "max_depth": [3, 5]}

        tuner = HyperparameterTuner(
            strategy=SearchStrategy.RANDOM_SEARCH,
            param_space=param_space,
            n_iter=4,
            cv=3,
        )
        result = tuner.tune(model, X, y)

        assert result.best_model is not None
        assert hasattr(result.best_model, "predict")


class TestGridSearch:
    def test_grid_search_explores_all_combinations(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        param_space = {"n_estimators": [10, 20], "max_depth": [3, 5]}

        tuner = HyperparameterTuner(
            strategy=SearchStrategy.GRID_SEARCH,
            param_space=param_space,
            cv=3,
        )
        result = tuner.tune(model, X, y)

        assert len(result.cv_results) == 4


class TestTuningResult:
    def test_result_contains_required_fields(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        param_space = {"n_estimators": [10, 20]}

        tuner = HyperparameterTuner(
            strategy=SearchStrategy.RANDOM_SEARCH,
            param_space=param_space,
            n_iter=2,
            cv=3,
        )
        result = tuner.tune(model, X, y)

        assert hasattr(result, "best_params")
        assert hasattr(result, "best_score")
        assert hasattr(result, "best_model")
        assert hasattr(result, "cv_results")

    def test_best_score_is_valid(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        param_space = {"n_estimators": [10, 20]}

        tuner = HyperparameterTuner(
            strategy=SearchStrategy.RANDOM_SEARCH,
            param_space=param_space,
            n_iter=2,
            cv=3,
        )
        result = tuner.tune(model, X, y)

        assert 0 <= result.best_score <= 1


class TestCVResults:
    def test_cv_results_contains_all_trials(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        param_space = {"n_estimators": [10, 20], "max_depth": [3]}

        tuner = HyperparameterTuner(
            strategy=SearchStrategy.GRID_SEARCH,
            param_space=param_space,
            cv=3,
        )
        result = tuner.tune(model, X, y)

        assert len(result.cv_results) >= 2


class TestScoringMetric:
    def test_custom_scoring_metric(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        param_space = {"n_estimators": [10, 20]}

        tuner = HyperparameterTuner(
            strategy=SearchStrategy.RANDOM_SEARCH,
            param_space=param_space,
            n_iter=2,
            cv=3,
            scoring="roc_auc",
        )
        result = tuner.tune(model, X, y)

        assert result.scoring == "roc_auc"


class TestReproducibility:
    def test_tuning_reproducible_with_seed(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        param_space = {"n_estimators": [10, 20, 50]}

        tuner1 = HyperparameterTuner(
            strategy=SearchStrategy.RANDOM_SEARCH,
            param_space=param_space,
            n_iter=2,
            cv=3,
            random_state=42,
        )
        result1 = tuner1.tune(model, X, y)

        tuner2 = HyperparameterTuner(
            strategy=SearchStrategy.RANDOM_SEARCH,
            param_space=param_space,
            n_iter=2,
            cv=3,
            random_state=42,
        )
        result2 = tuner2.tune(model, X, y)

        assert result1.best_params == result2.best_params
