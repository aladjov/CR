from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.modeling.spark_classifier_wrapper import (
    _FEATURES_COL,
    _LABEL_COL,
    _MODEL_REGISTRY,
    _WEIGHT_COL,
    SparkClassifierWrapper,
)


@pytest.fixture
def binary_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name="target")
    return X, y


@pytest.fixture
def imbalanced_data():
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
    })
    y = pd.Series([0] * 250 + [1] * 50, name="target")
    return X, y



class TestSparkClassifierWrapperInit:
    def test_rejects_empty_feature_names(self):
        with pytest.raises(ValueError, match="at least one feature"):
            SparkClassifierWrapper(
                spark_model_class="LogisticRegression",
                spark_model_params={},
                feature_names=[],
            )

    def test_stores_feature_names(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=["f1", "f2", "f3"],
        )
        assert wrapper.feature_names == ["f1", "f2", "f3"]

    def test_stores_class_weight(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="RandomForestClassifier",
            spark_model_params={},
            feature_names=["a"],
            class_weight="balanced",
        )
        assert wrapper.class_weight == "balanced"

    def test_default_no_class_weight(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
        )
        assert wrapper.class_weight is None

    def test_stores_model_params(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="GBTClassifier",
            spark_model_params={"maxIter": 50, "maxDepth": 5},
            feature_names=["a", "b"],
        )
        assert wrapper.spark_model_params == {"maxIter": 50, "maxDepth": 5}

    def test_unfitted_model_is_none(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
        )
        assert wrapper._fitted_model is None


class TestModelRegistry:
    def test_logistic_regression_in_registry(self):
        assert "LogisticRegression" in _MODEL_REGISTRY

    def test_random_forest_in_registry(self):
        assert "RandomForestClassifier" in _MODEL_REGISTRY

    def test_gbt_in_registry(self):
        assert "GBTClassifier" in _MODEL_REGISTRY

    def test_registry_has_correct_module_paths(self):
        for name, fqn in _MODEL_REGISTRY.items():
            assert fqn.startswith("pyspark.ml.classification.")
            assert fqn.endswith(name)


class TestSparkClassifierWrapperClone:
    def test_clone_returns_unfitted(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=["f1", "f2"],
            class_weight="balanced",
        )
        wrapper._fitted_model = MagicMock()
        cloned = wrapper.clone()
        assert cloned._fitted_model is None

    def test_clone_preserves_params(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="RandomForestClassifier",
            spark_model_params={"numTrees": 20},
            feature_names=["f1", "f2", "f3"],
            class_weight="balanced",
        )
        cloned = wrapper.clone()
        assert cloned.spark_model_class == "RandomForestClassifier"
        assert cloned.spark_model_params == {"numTrees": 20}
        assert cloned.feature_names == ["f1", "f2", "f3"]
        assert cloned.class_weight == "balanced"

    def test_clone_does_not_share_params_dict(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=["a"],
        )
        cloned = wrapper.clone()
        cloned.spark_model_params["maxIter"] = 999
        assert wrapper.spark_model_params["maxIter"] == 10

    def test_clone_does_not_share_feature_names(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a", "b"],
        )
        cloned = wrapper.clone()
        cloned.feature_names.append("c")
        assert wrapper.feature_names == ["a", "b"]


class TestSparkClassifierWrapperProperties:
    def test_classes_default(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
        )
        np.testing.assert_array_equal(wrapper.classes_, np.array([0, 1]))

    def test_classes_after_fit(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
        )
        wrapper._classes = np.array([0, 1])
        np.testing.assert_array_equal(wrapper.classes_, np.array([0, 1]))

    def test_feature_importances_from_spark_attr(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="RandomForestClassifier",
            spark_model_params={},
            feature_names=["f1", "f2", "f3"],
        )
        mock_model = MagicMock()
        mock_model.featureImportances.toArray.return_value = np.array([0.5, 0.3, 0.2])
        wrapper._fitted_model = mock_model
        importances = wrapper.feature_importances_
        np.testing.assert_array_equal(importances, np.array([0.5, 0.3, 0.2]))

    def test_feature_importances_from_coefficients(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1", "f2"],
        )
        mock_model = MagicMock(spec=[])
        mock_model.coefficients = MagicMock()
        mock_model.coefficients.toArray.return_value = np.array([-0.5, 0.3])
        wrapper._fitted_model = mock_model
        importances = wrapper.feature_importances_
        np.testing.assert_array_equal(importances, np.array([0.5, 0.3]))

    def test_feature_importances_fallback_to_zeros(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1", "f2"],
        )
        mock_model = MagicMock(spec=[])
        wrapper._fitted_model = mock_model
        importances = wrapper.feature_importances_
        np.testing.assert_array_equal(importances, np.array([0.0, 0.0]))


class TestCreateSparkModel:
    def test_unknown_model_raises(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="UnknownModel",
            spark_model_params={},
            feature_names=["a"],
        )
        with pytest.raises(ValueError, match="Unknown spark model class"):
            wrapper._create_spark_model()

    @patch("customer_retention.stages.modeling.spark_classifier_wrapper._import_class")
    def test_creates_logistic_with_correct_params(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 1000},
            feature_names=["f1", "f2"],
        )
        wrapper._create_spark_model()
        mock_cls.assert_called_once_with(
            maxIter=1000,
            featuresCol=_FEATURES_COL,
            labelCol=_LABEL_COL,
        )

    @patch("customer_retention.stages.modeling.spark_classifier_wrapper._import_class")
    def test_creates_logistic_with_weight_col(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=["f1"],
            class_weight="balanced",
        )
        wrapper._create_spark_model()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["weightCol"] == _WEIGHT_COL

    @patch("customer_retention.stages.modeling.spark_classifier_wrapper._import_class")
    def test_gbt_no_weight_col(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        wrapper = SparkClassifierWrapper(
            spark_model_class="GBTClassifier",
            spark_model_params={"maxIter": 10},
            feature_names=["f1"],
            class_weight="balanced",
        )
        wrapper._create_spark_model()
        call_kwargs = mock_cls.call_args[1]
        assert "weightCol" not in call_kwargs

    @patch("customer_retention.stages.modeling.spark_classifier_wrapper._import_class")
    def test_random_forest_with_weight_col(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        wrapper = SparkClassifierWrapper(
            spark_model_class="RandomForestClassifier",
            spark_model_params={"numTrees": 100},
            feature_names=["f1"],
            class_weight="balanced",
        )
        wrapper._create_spark_model()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["weightCol"] == _WEIGHT_COL

    _WRAPPER_MOD = "customer_retention.stages.modeling.spark_classifier_wrapper"

    @patch(f"{_WRAPPER_MOD}.get_default_parallelism", return_value=16)
    @patch(f"{_WRAPPER_MOD}._import_class")
    def test_aggregation_depth_added_when_model_supports_it(self, mock_import, _mock_par):
        mock_cls = MagicMock()
        mock_cls.aggregationDepth = True
        mock_import.return_value = mock_cls
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=["f1"],
        )
        wrapper._create_spark_model()
        call_kwargs = mock_cls.call_args[1]
        assert "aggregationDepth" in call_kwargs
        assert call_kwargs["aggregationDepth"] >= 2

    @patch(f"{_WRAPPER_MOD}.get_default_parallelism", return_value=16)
    @patch(f"{_WRAPPER_MOD}._import_class")
    def test_aggregation_depth_skipped_when_model_lacks_param(self, mock_import, _mock_par):
        mock_cls = MagicMock(spec=[])
        mock_import.return_value = mock_cls
        wrapper = SparkClassifierWrapper(
            spark_model_class="RandomForestClassifier",
            spark_model_params={"numTrees": 100},
            feature_names=["f1"],
        )
        wrapper._create_spark_model()
        call_kwargs = mock_cls.call_args[1]
        assert "aggregationDepth" not in call_kwargs

    @patch(f"{_WRAPPER_MOD}.get_default_parallelism", return_value=16)
    @patch(f"{_WRAPPER_MOD}._import_class")
    def test_explicit_aggregation_depth_not_overridden(self, mock_import, _mock_par):
        mock_cls = MagicMock()
        mock_cls.aggregationDepth = True
        mock_import.return_value = mock_cls
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10, "aggregationDepth": 5},
            feature_names=["f1"],
        )
        wrapper._create_spark_model()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["aggregationDepth"] == 5


_MOD = "customer_retention.stages.modeling.spark_classifier_wrapper"


class TestFitIntegration:
    @patch(f"{_MOD}._make_assembler")
    @patch(f"{_MOD}._get_spark_session")
    def test_fit_calls_spark_model_fit(self, mock_get_spark, mock_make_asm, binary_data):
        X, y = binary_data
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark

        mock_spark_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_spark_df
        mock_spark_df.withColumn.return_value = mock_spark_df

        mock_assembled = MagicMock()
        mock_assembled.count.return_value = len(X)
        mock_assembled.repartition.return_value = mock_assembled
        mock_make_asm.return_value.transform.return_value = mock_assembled

        mock_model_cls = MagicMock()
        mock_fitted = MagicMock()
        mock_model_cls.return_value.fit.return_value = mock_fitted

        with patch(f"{_MOD}._import_class", return_value=mock_model_cls):
            wrapper = SparkClassifierWrapper(
                spark_model_class="LogisticRegression",
                spark_model_params={"maxIter": 10},
                feature_names=X.columns.tolist(),
            )
            result = wrapper.fit(X, y)

        assert result is wrapper
        assert wrapper._fitted_model is mock_fitted
        mock_model_cls.return_value.fit.assert_called_once_with(mock_assembled)

    @pytest.mark.spark
    @patch(f"{_MOD}._make_assembler")
    @patch(f"{_MOD}._get_spark_session")
    def test_fit_with_balanced_weight_adds_column(self, mock_get_spark, mock_make_asm, imbalanced_data):
        X, y = imbalanced_data
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark

        mock_spark_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_spark_df
        mock_spark_df.withColumn.return_value = mock_spark_df
        mock_spark_df.groupBy.return_value.count.return_value.collect.return_value = [
            {_LABEL_COL: 0.0, "count": 250},
            {_LABEL_COL: 1.0, "count": 50},
        ]

        mock_make_asm.return_value.transform.return_value = MagicMock()

        mock_model_cls = MagicMock()
        mock_model_cls.return_value.fit.return_value = MagicMock()

        with patch(f"{_MOD}._import_class", return_value=mock_model_cls):
            wrapper = SparkClassifierWrapper(
                spark_model_class="LogisticRegression",
                spark_model_params={"maxIter": 10},
                feature_names=X.columns.tolist(),
                class_weight="balanced",
            )
            wrapper.fit(X, y)

        weight_calls = [c for c in mock_spark_df.withColumn.call_args_list if c[0][0] == _WEIGHT_COL]
        assert len(weight_calls) >= 1


class TestPredictIntegration:
    @patch(f"{_MOD}._make_assembler")
    @patch(f"{_MOD}._get_spark_session")
    def test_predict_collects_predictions_only(self, mock_get_spark, mock_make_asm, binary_data):
        X, _ = binary_data
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_spark.createDataFrame.return_value = MagicMock()
        mock_make_asm.return_value.transform.return_value = MagicMock()

        mock_fitted = MagicMock()
        mock_transformed = MagicMock()
        mock_fitted.transform.return_value = mock_transformed
        mock_transformed.select.return_value.toPandas.return_value = pd.DataFrame(
            {"prediction": np.random.choice([0.0, 1.0], len(X))}
        )

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=X.columns.tolist(),
        )
        wrapper._fitted_model = mock_fitted
        with patch(f"{_MOD}.enable_arrow_optimization"):
            preds = wrapper.predict(X)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)
        mock_transformed.select.assert_called_once_with("prediction")

    @patch(f"{_MOD}._make_assembler")
    @patch(f"{_MOD}._get_spark_session")
    def test_predict_proba_uses_vector_to_array(self, mock_get_spark, mock_make_asm, binary_data):
        pytest.importorskip("pyspark")
        X, _ = binary_data
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_spark.createDataFrame.return_value = MagicMock()
        mock_make_asm.return_value.transform.return_value = MagicMock()

        mock_fitted = MagicMock()
        mock_transformed = MagicMock()
        mock_fitted.transform.return_value = mock_transformed

        n = len(X)
        mock_transformed.select.return_value.toPandas.return_value = pd.DataFrame(
            {"p0": np.full(n, 0.3), "p1": np.full(n, 0.7)}
        )

        # Mock pyspark imports used inside predict_proba
        mock_F = MagicMock()
        mock_vector_to_array = MagicMock()
        mock_prob_array = MagicMock()
        mock_vector_to_array.return_value = mock_prob_array

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=X.columns.tolist(),
        )
        wrapper._fitted_model = mock_fitted
        with patch(f"{_MOD}.enable_arrow_optimization"), \
             patch.dict("sys.modules", {
                 "pyspark.sql.functions": mock_F,
                 "pyspark.ml.functions": MagicMock(vector_to_array=mock_vector_to_array),
             }):
            proba = wrapper.predict_proba(X)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (n, 2)
        np.testing.assert_allclose(proba[:, 0], 0.3)
        np.testing.assert_allclose(proba[:, 1], 0.7)
        mock_transformed.select.assert_called_once()


class TestEvaluateDistributed:
    @pytest.fixture(autouse=True)
    def _require_pyspark(self):
        pytest.importorskip("pyspark")

    def test_uses_spark_evaluators_without_collecting(self):
        mock_fitted = MagicMock()
        mock_predictions = MagicMock()
        mock_fitted.transform.return_value = mock_predictions

        mock_roc_eval = MagicMock()
        mock_roc_eval.evaluate.return_value = 0.85
        mock_pr_eval = MagicMock()
        mock_pr_eval.evaluate.return_value = 0.72

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1", "f2"],
        )
        wrapper._fitted_model = mock_fitted

        call_count = [0]
        def make_evaluator(**kwargs):
            evaluator = mock_roc_eval if call_count[0] == 0 else mock_pr_eval
            call_count[0] += 1
            assert kwargs["labelCol"] == _LABEL_COL
            assert kwargs["rawPredictionCol"] == "rawPrediction"
            return evaluator

        with patch("pyspark.ml.evaluation.BinaryClassificationEvaluator", side_effect=make_evaluator):
            result = wrapper.evaluate_distributed(MagicMock(name="prepared_df"))

        assert result == {"roc_auc": 0.85, "pr_auc": 0.72}
        mock_fitted.transform.assert_called_once()
        mock_roc_eval.evaluate.assert_called_once_with(mock_predictions)
        mock_pr_eval.evaluate.assert_called_once_with(mock_predictions)

    def test_unfitted_model_raises_attribute_error(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1", "f2"],
        )
        with pytest.raises(AttributeError):
            wrapper.evaluate_distributed(MagicMock())

    def test_returns_only_roc_auc_and_pr_auc_keys(self):
        mock_fitted = MagicMock()
        mock_fitted.transform.return_value = MagicMock()

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1"],
        )
        wrapper._fitted_model = mock_fitted

        with patch("pyspark.ml.evaluation.BinaryClassificationEvaluator") as mock_eval_cls:
            mock_eval_cls.return_value.evaluate.return_value = 0.5
            result = wrapper.evaluate_distributed(MagicMock())

        assert set(result.keys()) == {"roc_auc", "pr_auc"}
        assert all(isinstance(v, float) for v in result.values())

    def test_transforms_prepared_df_not_raw_data(self):
        mock_fitted = MagicMock()
        mock_fitted.transform.return_value = MagicMock()
        prepared = MagicMock(name="prepared_df")

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1"],
        )
        wrapper._fitted_model = mock_fitted

        with patch("pyspark.ml.evaluation.BinaryClassificationEvaluator"):
            wrapper.evaluate_distributed(prepared)

        mock_fitted.transform.assert_called_once_with(prepared)


class TestAsPipelineModel:
    @pytest.fixture(autouse=True)
    def _require_pyspark(self):
        pytest.importorskip("pyspark")

    def test_unfitted_raises(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1", "f2"],
        )
        with pytest.raises(ValueError, match="unfitted"):
            wrapper.as_pipeline_model()

    @patch(f"{_MOD}._make_assembler")
    def test_returns_pipeline_with_assembler_and_model(self, mock_make_asm):
        mock_assembler = MagicMock()
        mock_make_asm.return_value = mock_assembler
        mock_pipeline_cls = MagicMock()

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=["f1", "f2"],
        )
        mock_fitted = MagicMock()
        wrapper._fitted_model = mock_fitted

        with patch("pyspark.ml.PipelineModel", mock_pipeline_cls):
            wrapper.as_pipeline_model()

        mock_pipeline_cls.assert_called_once()
        stages = mock_pipeline_cls.call_args[1].get("stages") or mock_pipeline_cls.call_args[0][0]
        assert stages[0] is mock_assembler
        assert stages[1] is mock_fitted


class TestFromPipelineModel:
    def test_reconstructs_wrapper(self):
        mock_pipeline = MagicMock()
        mock_fitted = MagicMock()
        mock_pipeline.stages = [MagicMock(), mock_fitted]

        wrapper = SparkClassifierWrapper.from_pipeline_model(
            mock_pipeline, "RandomForestClassifier",
            {"numTrees": 100}, ["f1", "f2", "f3"], "balanced",
        )

        assert wrapper.spark_model_class == "RandomForestClassifier"
        assert wrapper.spark_model_params == {"numTrees": 100}
        assert wrapper.feature_names == ["f1", "f2", "f3"]
        assert wrapper.class_weight == "balanced"
        assert wrapper._fitted_model is mock_fitted
        np.testing.assert_array_equal(wrapper._classes, np.array([0, 1]))

    def test_reconstructed_has_predict_and_importances(self):
        mock_pipeline = MagicMock()
        mock_fitted = MagicMock()
        mock_fitted.featureImportances.toArray.return_value = np.array([0.5, 0.5])
        mock_pipeline.stages = [MagicMock(), mock_fitted]

        wrapper = SparkClassifierWrapper.from_pipeline_model(
            mock_pipeline, "RandomForestClassifier", {}, ["f1", "f2"],
        )
        np.testing.assert_array_equal(wrapper.feature_importances_, np.array([0.5, 0.5]))


class TestFitFailsFastOnStaleFeatures:
    @patch(f"{_MOD}._get_spark_session")
    def test_fit_raises_on_missing_feature(self, mock_get_spark):
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y = pd.Series([0, 1], name="target")

        mock_get_spark.return_value = MagicMock()

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=["f1", "f2", "event_count_180d"],
        )
        with pytest.raises(KeyError):
            wrapper.fit(X, y)


class TestToSparkDfAlignment:
    @patch(f"{_MOD}._make_assembler")
    @patch(f"{_MOD}._get_spark_session")
    def test_misaligned_pandas_indexes_produce_no_nans(self, mock_get_spark, mock_make_asm):
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}, index=[10, 20, 30])
        y = pd.Series([0, 1, 0], index=[100, 200, 300], name="target")
        captured = {}

        def capture(df):
            captured["df"] = df.copy()
            mock_df = MagicMock()
            mock_df.withColumn.return_value = mock_df
            return mock_df

        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_spark.createDataFrame.side_effect = capture
        mock_make_asm.return_value.transform.return_value = MagicMock()

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1", "f2"],
        )
        wrapper._to_spark_df(X, y)

        combined = captured["df"]
        assert not combined.isna().any().any()
        assert len(combined) == 3
        np.testing.assert_array_equal(combined[_LABEL_COL].to_numpy(), [0, 1, 0])

    @patch(f"{_MOD}._make_assembler")
    @patch(f"{_MOD}._get_spark_session")
    def test_aligned_pandas_indexes_preserve_values(self, mock_get_spark, mock_make_asm):
        X = pd.DataFrame({"f1": [10.0, 20.0], "f2": [30.0, 40.0]})
        y = pd.Series([1, 0], name="target")
        captured = {}

        def capture(df):
            captured["df"] = df.copy()
            mock_df = MagicMock()
            mock_df.withColumn.return_value = mock_df
            return mock_df

        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_spark.createDataFrame.side_effect = capture
        mock_make_asm.return_value.transform.return_value = MagicMock()

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["f1", "f2"],
        )
        wrapper._to_spark_df(X, y)

        combined = captured["df"]
        assert list(combined.columns) == ["f1", "f2", _LABEL_COL]
        np.testing.assert_array_equal(combined["f1"].to_numpy(), [10.0, 20.0])
        np.testing.assert_array_equal(combined[_LABEL_COL].to_numpy(), [1, 0])

    @pytest.mark.spark
    @patch(f"{_MOD}._make_assembler")
    @patch(f"{_MOD}._get_spark_session")
    def test_pyspark_pandas_path_resets_indexes(self, mock_get_spark, mock_make_asm):
        mock_X = MagicMock()
        mock_X.__contains__ = MagicMock(return_value=True)
        mock_X_sel = MagicMock()
        mock_X.__getitem__ = MagicMock(return_value=mock_X_sel)
        mock_X_reset = MagicMock()
        mock_X_sel.reset_index = MagicMock(return_value=mock_X_reset)

        mock_y = MagicMock()
        mock_y.rename = MagicMock()
        mock_y_renamed = mock_y.rename.return_value
        mock_y_reset = MagicMock()
        mock_y_renamed.reset_index = MagicMock(return_value=mock_y_reset)

        mock_combined = MagicMock()
        mock_spark_df = MagicMock()
        mock_combined.to_spark.return_value = mock_spark_df
        mock_spark_df.withColumn.return_value = mock_spark_df
        mock_make_asm.return_value.transform.return_value = MagicMock()

        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch("pyspark.pandas.concat", return_value=mock_combined):
            wrapper = SparkClassifierWrapper(
                spark_model_class="LogisticRegression",
                spark_model_params={},
                feature_names=["f1", "f2"],
            )
            wrapper._to_spark_df(mock_X, mock_y)

        mock_X_sel.reset_index.assert_called_once_with(drop=True)
        mock_y_renamed.reset_index.assert_called_once_with(drop=True)


class TestTargetPartitionsRowAware:
    """Repartition target must be capped by row count to avoid the shared-cluster slowdown
    where get_default_parallelism() falls back to spark.sql.shuffle.partitions=200 and
    L-BFGS pays per-task scheduling overhead × 1000 iterations."""

    @patch(f"{_MOD}.get_default_parallelism", return_value=200)
    def test_small_data_does_not_overpartition_on_high_parallelism(self, _mock_par):
        # 60K rows on a 200-parallelism shared cluster should NOT become 400 partitions
        target = SparkClassifierWrapper._target_partitions(60_000)
        assert target == 12  # ceil(60000 / 5000)
        assert target < 50, "small data must never be over-partitioned"

    @patch(f"{_MOD}.get_default_parallelism", return_value=200)
    def test_medium_data_capped_by_row_count_not_parallelism(self, _mock_par):
        # 250K rows: target_by_size = 50, parallelism*2 = 400 → take 50
        target = SparkClassifierWrapper._target_partitions(250_000)
        assert target == 50

    @patch(f"{_MOD}.get_default_parallelism", return_value=16)
    def test_large_data_capped_by_parallelism(self, _mock_par):
        # 10M rows on a 16-core cluster: target_by_size = 2000, parallelism*2 = 32 → take 32
        target = SparkClassifierWrapper._target_partitions(10_000_000)
        assert target == 32

    @patch(f"{_MOD}.get_default_parallelism", return_value=16)
    def test_small_data_on_small_cluster_uses_row_count(self, _mock_par):
        # 60K rows on 16-core cluster: ceil(60000/5000)=12, parallelism*2=32 → take 12
        assert SparkClassifierWrapper._target_partitions(60_000) == 12

    @patch(f"{_MOD}.get_default_parallelism", return_value=200)
    def test_zero_rows_returns_one_partition(self, _mock_par):
        assert SparkClassifierWrapper._target_partitions(0) == 1

    @patch(f"{_MOD}.get_default_parallelism", return_value=0)
    def test_zero_parallelism_falls_back_to_one(self, _mock_par):
        # Defensive: get_default_parallelism can return 0 when spark isn't available
        target = SparkClassifierWrapper._target_partitions(60_000)
        # With parallelism = max(1, 0) = 1, target = min(2, 12) = 2
        assert target == 2


class TestRepartitionForTraining:
    """Verify repartition_for_training uses the row-aware target."""

    @patch(f"{_MOD}.get_default_parallelism", return_value=200)
    def test_repartition_target_matches_row_count_not_parallelism(self, _mock_par):
        spark_df = MagicMock()
        spark_df.count.return_value = 60_000
        spark_df.repartition.return_value = spark_df

        SparkClassifierWrapper._repartition_for_training(spark_df)

        spark_df.repartition.assert_called_once_with(12)
        spark_df.cache.assert_called_once()

    @patch(f"{_MOD}.get_default_parallelism", return_value=16)
    def test_repartition_uses_parallelism_cap_for_large_data(self, _mock_par):
        spark_df = MagicMock()
        spark_df.count.return_value = 10_000_000
        spark_df.repartition.return_value = spark_df

        SparkClassifierWrapper._repartition_for_training(spark_df)

        spark_df.repartition.assert_called_once_with(32)


class TestFitPropagatesRowCount:
    """fit() must pass n_rows down so _create_spark_model picks an aggregationDepth
    based on the actual partition count, not raw cluster parallelism."""

    @patch(f"{_MOD}._make_assembler")
    @patch(f"{_MOD}._get_spark_session")
    @patch(f"{_MOD}.get_default_parallelism", return_value=200)
    def test_fit_aggregation_depth_uses_row_aware_partition_count(
        self, _mock_par, mock_get_spark, mock_make_asm, binary_data,
    ):
        X, _ = binary_data
        mock_get_spark.return_value = MagicMock()

        mock_assembled = MagicMock()
        # Pretend the data has 100K rows; with parallelism=200 the row-aware
        # target is 20 (NOT 400) → aggregationDepth = ceil(log2(20)/log2(4)) = 3
        mock_assembled.count.return_value = 100_000
        mock_assembled.repartition.return_value = mock_assembled
        mock_make_asm.return_value.transform.return_value = mock_assembled

        mock_model_cls = MagicMock()
        mock_model_cls.aggregationDepth = True
        with patch(f"{_MOD}._import_class", return_value=mock_model_cls):
            wrapper = SparkClassifierWrapper(
                spark_model_class="LogisticRegression",
                spark_model_params={"maxIter": 10},
                feature_names=X.columns.tolist(),
            )
            y = pd.Series(np.random.randint(0, 2, len(X)))
            wrapper.fit(X, y)

        call_kwargs = mock_model_cls.call_args[1]
        assert "aggregationDepth" in call_kwargs
        # 20 row-aware partitions → log2(20)/log2(4) ≈ 2.16 → ceil = 3
        assert call_kwargs["aggregationDepth"] == 3, (
            "aggregationDepth must scale with the row-aware partition count, "
            "not raw cluster parallelism (which would give depth 5 here)"
        )
