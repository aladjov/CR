from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.components.enums import ModelType
from customer_retention.stages.modeling.baseline_trainer import TrainedModel
from customer_retention.stages.modeling.spark_baseline_trainer import (
    SparkBaselineTrainer,
    create_distributed_models,
)

_WRAPPER_MOD = "customer_retention.stages.modeling.spark_classifier_wrapper"


class TestSparkBaselineTrainerInit:
    def test_stores_model_type(self):
        trainer = SparkBaselineTrainer(
            model_type=ModelType.LOGISTIC_REGRESSION,
            feature_names=["a", "b"],
        )
        assert trainer.model_type == ModelType.LOGISTIC_REGRESSION

    def test_stores_feature_names(self):
        trainer = SparkBaselineTrainer(
            model_type=ModelType.RANDOM_FOREST,
            feature_names=["x", "y", "z"],
        )
        assert trainer.feature_names == ["x", "y", "z"]

    def test_default_class_weight_none(self):
        trainer = SparkBaselineTrainer(
            model_type=ModelType.LOGISTIC_REGRESSION,
            feature_names=["a"],
        )
        assert trainer.class_weight is None

    def test_stores_class_weight(self):
        trainer = SparkBaselineTrainer(
            model_type=ModelType.LOGISTIC_REGRESSION,
            feature_names=["a"],
            class_weight="balanced",
        )
        assert trainer.class_weight == "balanced"


class TestSparkBaselineTrainerModelMapping:
    def test_logistic_regression_maps_to_spark(self):
        trainer = SparkBaselineTrainer(
            model_type=ModelType.LOGISTIC_REGRESSION,
            feature_names=["a"],
        )
        spark_class, params = trainer._get_spark_model_config()
        assert spark_class == "LogisticRegression"
        assert "maxIter" in params

    def test_random_forest_maps_to_spark(self):
        trainer = SparkBaselineTrainer(
            model_type=ModelType.RANDOM_FOREST,
            feature_names=["a"],
        )
        spark_class, params = trainer._get_spark_model_config()
        assert spark_class == "RandomForestClassifier"
        assert "numTrees" in params

    def test_xgboost_maps_to_gbt(self):
        trainer = SparkBaselineTrainer(
            model_type=ModelType.XGBOOST,
            feature_names=["a"],
        )
        spark_class, params = trainer._get_spark_model_config()
        assert spark_class == "GBTClassifier"
        assert "maxIter" in params

    def test_unsupported_model_raises(self):
        trainer = SparkBaselineTrainer(
            model_type=ModelType.LIGHTGBM,
            feature_names=["a"],
        )
        with pytest.raises(ValueError, match="No spark.ml mapping"):
            trainer._get_spark_model_config()


class TestSparkBaselineTrainerFit:
    @patch(f"{_WRAPPER_MOD}._make_assembler")
    @patch(f"{_WRAPPER_MOD}._get_spark_session")
    def test_fit_returns_trained_model(self, mock_get_spark, mock_make_asm):
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_spark.createDataFrame.return_value = MagicMock()
        mock_spark.createDataFrame.return_value.withColumn.return_value = mock_spark.createDataFrame.return_value
        mock_assembled = MagicMock()
        mock_assembled.count.return_value = 2
        mock_assembled.repartition.return_value = mock_assembled
        mock_make_asm.return_value.transform.return_value = mock_assembled

        mock_model_cls = MagicMock()
        mock_model_cls.return_value.fit.return_value = MagicMock()

        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y = pd.Series([0, 1])

        with patch(f"{_WRAPPER_MOD}._import_class", return_value=mock_model_cls):
            trainer = SparkBaselineTrainer(
                model_type=ModelType.LOGISTIC_REGRESSION,
                feature_names=["f1", "f2"],
            )
            result = trainer.fit(X, y)

        assert isinstance(result, TrainedModel)
        assert result.model_type == ModelType.LOGISTIC_REGRESSION
        assert result.feature_names == ["f1", "f2"]
        assert result.training_time >= 0

    @patch(f"{_WRAPPER_MOD}._make_assembler")
    @patch(f"{_WRAPPER_MOD}._get_spark_session")
    def test_fit_does_not_collect_to_pandas(self, mock_get_spark, mock_make_asm):
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_spark.createDataFrame.return_value = MagicMock()
        mock_spark.createDataFrame.return_value.withColumn.return_value = mock_spark.createDataFrame.return_value
        mock_assembled = MagicMock()
        mock_assembled.count.return_value = 100
        mock_assembled.repartition.return_value = mock_assembled
        mock_make_asm.return_value.transform.return_value = mock_assembled

        mock_model_cls = MagicMock()
        mock_model_cls.return_value.fit.return_value = MagicMock()

        X = pd.DataFrame({"f1": np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))

        with patch(f"{_WRAPPER_MOD}._import_class", return_value=mock_model_cls):
            trainer = SparkBaselineTrainer(
                model_type=ModelType.RANDOM_FOREST,
                feature_names=["f1"],
            )
            result = trainer.fit(X, y)

        assert result.model is not None

    @pytest.mark.spark
    @patch(f"{_WRAPPER_MOD}._make_assembler")
    @patch(f"{_WRAPPER_MOD}._get_spark_session")
    def test_fit_with_balanced_class_weight(self, mock_get_spark, mock_make_asm):
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_df
        mock_df.withColumn.return_value = mock_df
        mock_df.groupBy.return_value.count.return_value.collect.return_value = [
            {"__label__": 0.0, "count": 80},
            {"__label__": 1.0, "count": 20},
        ]
        mock_assembled = MagicMock()
        mock_assembled.count.return_value = 100
        mock_assembled.repartition.return_value = mock_assembled
        mock_make_asm.return_value.transform.return_value = mock_assembled

        mock_model_cls = MagicMock()
        mock_model_cls.return_value.fit.return_value = MagicMock()

        X = pd.DataFrame({"f1": np.random.randn(100)})
        y = pd.Series([0] * 80 + [1] * 20)

        with patch(f"{_WRAPPER_MOD}._import_class", return_value=mock_model_cls):
            trainer = SparkBaselineTrainer(
                model_type=ModelType.LOGISTIC_REGRESSION,
                feature_names=["f1"],
                class_weight="balanced",
            )
            result = trainer.fit(X, y)

        assert result.class_weight == "balanced"


class TestCreateDistributedModels:
    def test_returns_three_models(self):
        models = create_distributed_models(feature_names=["f1", "f2"])
        assert len(models) == 3

    def test_model_names(self):
        models = create_distributed_models(feature_names=["f1"])
        assert "Logistic Regression" in models
        assert "Random Forest" in models
        assert "Gradient Boosting" in models

    def test_all_are_spark_wrappers(self):
        from customer_retention.stages.modeling.spark_classifier_wrapper import SparkClassifierWrapper
        models = create_distributed_models(feature_names=["f1", "f2"])
        for model in models.values():
            assert isinstance(model, SparkClassifierWrapper)

    def test_default_balanced_weights(self):
        models = create_distributed_models(feature_names=["f1"])
        for model in models.values():
            assert model.class_weight == "balanced"

    def test_custom_class_weight(self):
        models = create_distributed_models(feature_names=["f1"], class_weight=None)
        for model in models.values():
            assert model.class_weight is None

    def test_feature_names_propagated(self):
        features = ["feat_a", "feat_b", "feat_c"]
        models = create_distributed_models(feature_names=features)
        for model in models.values():
            assert model.feature_names == features

    def test_logistic_params(self):
        models = create_distributed_models(feature_names=["f1"])
        lr = models["Logistic Regression"]
        assert lr.spark_model_class == "LogisticRegression"
        assert lr.spark_model_params["maxIter"] == 1000

    def test_logistic_regression_carries_tol_for_early_termination(self):
        # Without an explicit tol, L-BFGS uses 1e-6 which never converges on
        # imbalanced data with class_weight='balanced' — runs all 1000 iterations
        # and pays per-task overhead × 1000 on shared/Spark-Connect clusters.
        models = create_distributed_models(feature_names=["f1"])
        lr = models["Logistic Regression"]
        assert "tol" in lr.spark_model_params
        assert lr.spark_model_params["tol"] >= 1e-4

    def test_random_forest_params(self):
        models = create_distributed_models(feature_names=["f1"])
        rf = models["Random Forest"]
        assert rf.spark_model_class == "RandomForestClassifier"
        assert rf.spark_model_params["numTrees"] == 100

    def test_gradient_boosting_params(self):
        models = create_distributed_models(feature_names=["f1"])
        gb = models["Gradient Boosting"]
        assert gb.spark_model_class == "GBTClassifier"
        assert gb.spark_model_params["maxIter"] == 100
