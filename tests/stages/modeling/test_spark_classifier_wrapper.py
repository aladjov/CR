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


class TestClassWeightComputation:
    def test_balanced_weights_equal_classes(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
            class_weight="balanced",
        )
        y = pd.Series([0] * 50 + [1] * 50)
        weights = wrapper._compute_balanced_weights(y)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(1.0)

    def test_balanced_weights_imbalanced(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
            class_weight="balanced",
        )
        y = pd.Series([0] * 80 + [1] * 20)
        weights = wrapper._compute_balanced_weights(y)
        assert weights[0] == pytest.approx(100 / (2 * 80))
        assert weights[1] == pytest.approx(100 / (2 * 20))

    def test_balanced_weights_match_sklearn(self):
        from sklearn.utils.class_weight import compute_class_weight
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
            class_weight="balanced",
        )
        y = pd.Series([0] * 75 + [1] * 25)
        our_weights = wrapper._compute_balanced_weights(y)
        sklearn_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y.to_numpy())
        assert our_weights[0] == pytest.approx(sklearn_weights[0], rel=1e-6)
        assert our_weights[1] == pytest.approx(sklearn_weights[1], rel=1e-6)

    def test_balanced_weights_highly_imbalanced(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
            class_weight="balanced",
        )
        y = pd.Series([0] * 990 + [1] * 10)
        weights = wrapper._compute_balanced_weights(y)
        assert weights[1] > weights[0]
        assert weights[1] == pytest.approx(1000 / (2 * 10))
        assert weights[0] == pytest.approx(1000 / (2 * 990))

    def test_balanced_weights_with_numpy_input(self):
        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={},
            feature_names=["a"],
            class_weight="balanced",
        )
        y = np.array([0] * 60 + [1] * 40)
        weights = wrapper._compute_balanced_weights(y)
        assert weights[0] == pytest.approx(100 / (2 * 60))
        assert weights[1] == pytest.approx(100 / (2 * 40))


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
    def test_predict_proba_collects_probability_only(self, mock_get_spark, mock_make_asm, binary_data):
        X, _ = binary_data
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_spark.createDataFrame.return_value = MagicMock()
        mock_make_asm.return_value.transform.return_value = MagicMock()

        mock_fitted = MagicMock()
        mock_transformed = MagicMock()
        mock_fitted.transform.return_value = mock_transformed

        mock_dense_vectors = [MagicMock() for _ in range(len(X))]
        for v in mock_dense_vectors:
            v.toArray.return_value = np.array([0.3, 0.7])
        mock_transformed.select.return_value.toPandas.return_value = pd.DataFrame(
            {"probability": mock_dense_vectors}
        )

        wrapper = SparkClassifierWrapper(
            spark_model_class="LogisticRegression",
            spark_model_params={"maxIter": 10},
            feature_names=X.columns.tolist(),
        )
        wrapper._fitted_model = mock_fitted
        with patch(f"{_MOD}.enable_arrow_optimization"):
            proba = wrapper.predict_proba(X)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (len(X), 2)
        mock_transformed.select.assert_called_once_with("probability")


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
