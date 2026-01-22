import pytest
import pandas as pd
import numpy as np
from customer_retention.stages.modeling import (
    BaselineTrainer, ModelType, TrainingConfig, TrainedModel
)


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


class TestTrainingConfig:
    def test_default_config_values(self):
        config = TrainingConfig()
        assert config.random_state == 42
        assert config.verbose is False

    def test_custom_config(self):
        config = TrainingConfig(random_state=123, verbose=True)
        assert config.random_state == 123
        assert config.verbose is True


class TestLogisticRegression:
    def test_logistic_regression_trains_successfully(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.LOGISTIC_REGRESSION)
        result = trainer.fit(X, y)

        assert result.model is not None
        assert result.model_type == ModelType.LOGISTIC_REGRESSION

    def test_logistic_regression_with_class_weight(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(
            model_type=ModelType.LOGISTIC_REGRESSION,
            class_weight="balanced"
        )
        result = trainer.fit(X, y)

        assert result.model is not None

    def test_logistic_regression_custom_params(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_params={"C": 0.5, "max_iter": 500}
        )
        result = trainer.fit(X, y)

        assert result.hyperparameters["C"] == 0.5


class TestRandomForest:
    def test_random_forest_trains_successfully(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.RANDOM_FOREST)
        result = trainer.fit(X, y)

        assert result.model is not None
        assert result.model_type == ModelType.RANDOM_FOREST

    def test_random_forest_with_class_weight(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(
            model_type=ModelType.RANDOM_FOREST,
            class_weight="balanced"
        )
        result = trainer.fit(X, y)

        assert result.model is not None

    def test_random_forest_custom_params(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(
            model_type=ModelType.RANDOM_FOREST,
            model_params={"n_estimators": 50, "max_depth": 5}
        )
        result = trainer.fit(X, y)

        assert result.hyperparameters["n_estimators"] == 50
        assert result.hyperparameters["max_depth"] == 5


class TestXGBoost:
    def test_xgboost_trains_successfully(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.XGBOOST)
        result = trainer.fit(X, y)

        assert result.model is not None
        assert result.model_type == ModelType.XGBOOST

    def test_xgboost_with_scale_pos_weight(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(
            model_type=ModelType.XGBOOST,
            model_params={"scale_pos_weight": 2.0}
        )
        result = trainer.fit(X, y)

        assert result.model is not None

    def test_xgboost_custom_params(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(
            model_type=ModelType.XGBOOST,
            model_params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05}
        )
        result = trainer.fit(X, y)

        assert result.hyperparameters["max_depth"] == 3


class TestPrediction:
    def test_model_can_predict(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.LOGISTIC_REGRESSION)
        result = trainer.fit(X, y)

        predictions = result.model.predict(X)
        assert len(predictions) == len(X)

    def test_model_can_predict_proba(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.LOGISTIC_REGRESSION)
        result = trainer.fit(X, y)

        probas = result.model.predict_proba(X)
        assert probas.shape == (len(X), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)


class TestTrainedModel:
    def test_trained_model_contains_required_fields(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.LOGISTIC_REGRESSION)
        result = trainer.fit(X, y)

        assert hasattr(result, "model")
        assert hasattr(result, "model_type")
        assert hasattr(result, "hyperparameters")
        assert hasattr(result, "training_time")
        assert hasattr(result, "feature_names")

    def test_feature_names_preserved(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.LOGISTIC_REGRESSION)
        result = trainer.fit(X, y)

        assert result.feature_names == list(X.columns)


class TestEarlyStopping:
    def test_xgboost_early_stopping(self, training_data):
        X_train, y_train = training_data
        X_val = X_train.copy()
        y_val = y_train.copy()

        trainer = BaselineTrainer(
            model_type=ModelType.XGBOOST,
            model_params={"n_estimators": 100, "early_stopping_rounds": 10}
        )
        result = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        assert result.model is not None


class TestMultipleModels:
    def test_train_multiple_models(self, training_data):
        X, y = training_data
        models_to_train = [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.RANDOM_FOREST,
            ModelType.XGBOOST,
        ]

        results = []
        for model_type in models_to_train:
            trainer = BaselineTrainer(model_type=model_type)
            result = trainer.fit(X, y)
            results.append(result)

        assert len(results) == 3
        assert all(r.model is not None for r in results)


class TestDefaultHyperparameters:
    def test_logistic_regression_defaults(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.LOGISTIC_REGRESSION)
        result = trainer.fit(X, y)

        assert result.hyperparameters.get("max_iter", 1000) >= 1000

    def test_random_forest_defaults(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.RANDOM_FOREST)
        result = trainer.fit(X, y)

        assert "n_estimators" in result.hyperparameters

    def test_xgboost_defaults(self, training_data):
        X, y = training_data
        trainer = BaselineTrainer(model_type=ModelType.XGBOOST)
        result = trainer.fit(X, y)

        assert "n_estimators" in result.hyperparameters


class TestReproducibility:
    def test_training_reproducible_with_seed(self, training_data):
        X, y = training_data

        trainer1 = BaselineTrainer(model_type=ModelType.RANDOM_FOREST, random_state=42)
        result1 = trainer1.fit(X, y)
        pred1 = result1.model.predict_proba(X)

        trainer2 = BaselineTrainer(model_type=ModelType.RANDOM_FOREST, random_state=42)
        result2 = trainer2.fit(X, y)
        pred2 = result2.model.predict_proba(X)

        np.testing.assert_array_almost_equal(pred1, pred2)
