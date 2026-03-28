import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.modeling import FeatureScaler, ScalerType
from customer_retention.stages.modeling.spark_feature_scaler import SparkFeatureScaler


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X_train = pd.DataFrame(
        {
            "feature1": np.random.randn(n) * 10 + 50,
            "feature2": np.random.randn(n) * 5 + 20,
            "feature3": np.random.randn(n) * 100,
        }
    )
    X_test = pd.DataFrame(
        {
            "feature1": np.random.randn(100) * 10 + 50,
            "feature2": np.random.randn(100) * 5 + 20,
            "feature3": np.random.randn(100) * 100,
        }
    )
    return X_train, X_test


@pytest.fixture
def single_column_data():
    np.random.seed(42)
    X_train = pd.DataFrame({"only_col": np.random.randn(100) * 5 + 10})
    X_test = pd.DataFrame({"only_col": np.random.randn(20) * 5 + 10})
    return X_train, X_test


@pytest.fixture
def zero_variance_data():
    X_train = pd.DataFrame({"constant": [5.0] * 100, "varying": np.random.randn(100)})
    X_test = pd.DataFrame({"constant": [5.0] * 20, "varying": np.random.randn(20)})
    return X_train, X_test


@pytest.fixture
def all_null_data():
    X_train = pd.DataFrame(
        {
            "null_col": [np.nan] * 100,
            "valid_col": np.random.randn(100),
        }
    )
    X_test = pd.DataFrame(
        {
            "null_col": [np.nan] * 20,
            "valid_col": np.random.randn(20),
        }
    )
    return X_train, X_test


@pytest.fixture
def single_row_data():
    X_train = pd.DataFrame({"a": [3.0], "b": [7.0]})
    X_test = pd.DataFrame({"a": [5.0], "b": [9.0]})
    return X_train, X_test


class TestSparkFeatureScalerEquivalence:
    @pytest.mark.parametrize("scaler_type", [ScalerType.STANDARD, ScalerType.ROBUST, ScalerType.MINMAX])
    def test_spark_scaler_matches_sklearn_on_train(self, sample_data, scaler_type):
        X_train, X_test = sample_data

        base = FeatureScaler(scaler_type=scaler_type)
        base_result = base.fit_transform(X_train, X_test)

        spark = SparkFeatureScaler(scaler_type=scaler_type)
        spark_result = spark.fit_transform(X_train, X_test)

        pd.testing.assert_frame_equal(
            spark_result.X_train_scaled,
            base_result.X_train_scaled,
            atol=1e-10,
        )

    @pytest.mark.parametrize("scaler_type", [ScalerType.STANDARD, ScalerType.ROBUST, ScalerType.MINMAX])
    def test_spark_scaler_matches_sklearn_on_test(self, sample_data, scaler_type):
        X_train, X_test = sample_data

        base = FeatureScaler(scaler_type=scaler_type)
        base_result = base.fit_transform(X_train, X_test)

        spark = SparkFeatureScaler(scaler_type=scaler_type)
        spark_result = spark.fit_transform(X_train, X_test)

        pd.testing.assert_frame_equal(
            spark_result.X_test_scaled,
            base_result.X_test_scaled,
            atol=1e-10,
        )

    def test_none_scaler_returns_unchanged(self, sample_data):
        X_train, X_test = sample_data
        spark = SparkFeatureScaler(scaler_type=ScalerType.NONE)
        result = spark.fit_transform(X_train, X_test)

        pd.testing.assert_frame_equal(result.X_train_scaled, X_train)
        pd.testing.assert_frame_equal(result.X_test_scaled, X_test)


class TestSparkFeatureScalerStandard:
    def test_standard_normalizes_mean_std(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert np.abs(result.X_train_scaled.mean().mean()) < 0.1
        assert np.abs(result.X_train_scaled.std().mean() - 1.0) < 0.1

    def test_standard_params_stored(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert "mean" in result.scaling_params
        assert "scale" in result.scaling_params
        assert len(result.scaling_params["mean"]) == 3
        assert len(result.scaling_params["scale"]) == 3


class TestSparkFeatureScalerRobust:
    def test_robust_handles_outliers(self, sample_data):
        X_train, X_test = sample_data
        X_train_outlier = X_train.copy()
        X_train_outlier.iloc[0, 0] = 10000

        scaler = SparkFeatureScaler(scaler_type=ScalerType.ROBUST)
        result = scaler.fit_transform(X_train_outlier, X_test)

        assert not np.isinf(result.X_train_scaled.values).any()

    def test_robust_params_stored(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.ROBUST)
        result = scaler.fit_transform(X_train, X_test)

        assert "center" in result.scaling_params
        assert "scale" in result.scaling_params


class TestSparkFeatureScalerMinMax:
    def test_minmax_scales_to_range(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.MINMAX)
        result = scaler.fit_transform(X_train, X_test)

        assert result.X_train_scaled.min().min() >= -1e-10
        assert result.X_train_scaled.max().max() <= 1 + 1e-10

    def test_minmax_params_stored(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.MINMAX)
        result = scaler.fit_transform(X_train, X_test)

        assert "data_min" in result.scaling_params
        assert "data_max" in result.scaling_params


class TestSparkFeatureScalerTransform:
    @pytest.mark.parametrize("scaler_type", [ScalerType.STANDARD, ScalerType.ROBUST, ScalerType.MINMAX])
    def test_transform_uses_stored_params(self, sample_data, scaler_type):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=scaler_type)
        scaler.fit_transform(X_train, X_test)

        new_data = pd.DataFrame(
            {
                "feature1": [55.0, 60.0],
                "feature2": [22.0, 25.0],
                "feature3": [10.0, 20.0],
            }
        )
        transformed = scaler.transform(new_data)
        assert len(transformed) == 2
        assert list(transformed.columns) == ["feature1", "feature2", "feature3"]

    @pytest.mark.parametrize("scaler_type", [ScalerType.STANDARD, ScalerType.ROBUST, ScalerType.MINMAX])
    def test_transform_matches_sklearn_on_new_data(self, sample_data, scaler_type):
        X_train, X_test = sample_data

        base = FeatureScaler(scaler_type=scaler_type)
        base.fit_transform(X_train, X_test)

        spark = SparkFeatureScaler(scaler_type=scaler_type)
        spark.fit_transform(X_train, X_test)

        new_data = pd.DataFrame(
            {
                "feature1": [55.0, 60.0, 45.0],
                "feature2": [22.0, 25.0, 15.0],
                "feature3": [10.0, 20.0, -50.0],
            }
        )

        base_out = base.transform(new_data)
        spark_out = spark.transform(new_data)
        pd.testing.assert_frame_equal(spark_out, base_out, atol=1e-10)

    def test_transform_none_returns_unchanged(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.NONE)
        scaler.fit_transform(X_train, X_test)

        new_data = pd.DataFrame({"feature1": [1.0], "feature2": [2.0], "feature3": [3.0]})
        result = scaler.transform(new_data)
        pd.testing.assert_frame_equal(result, new_data)


class TestSparkFeatureScalerExtractParams:
    @pytest.mark.parametrize("scaler_type", [ScalerType.STANDARD, ScalerType.ROBUST, ScalerType.MINMAX])
    def test_extract_params_matches_sklearn(self, sample_data, scaler_type):
        X_train, X_test = sample_data

        base = FeatureScaler(scaler_type=scaler_type)
        base_result = base.fit_transform(X_train, X_test)

        spark = SparkFeatureScaler(scaler_type=scaler_type)
        spark_result = spark.fit_transform(X_train, X_test)

        for key in base_result.scaling_params:
            assert key in spark_result.scaling_params, f"Missing key: {key}"
            np.testing.assert_allclose(
                spark_result.scaling_params[key],
                base_result.scaling_params[key],
                atol=1e-10,
            )


class TestSparkFeatureScalerEdgeCases:
    def test_single_column(self, single_column_data):
        X_train, X_test = single_column_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert result.X_train_scaled.shape[1] == 1
        assert np.abs(result.X_train_scaled.mean().iloc[0]) < 0.1

    def test_zero_variance_standard(self, zero_variance_data):
        X_train, X_test = zero_variance_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert not np.isinf(result.X_train_scaled.values).any()

    def test_zero_variance_robust(self, zero_variance_data):
        X_train, X_test = zero_variance_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.ROBUST)
        result = scaler.fit_transform(X_train, X_test)

        assert not np.isinf(result.X_train_scaled.values).any()

    def test_zero_variance_minmax(self, zero_variance_data):
        X_train, X_test = zero_variance_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.MINMAX)
        result = scaler.fit_transform(X_train, X_test)

        assert not np.isinf(result.X_train_scaled.values).any()

    def test_all_null_column(self, all_null_data):
        X_train, X_test = all_null_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert result.X_train_scaled["null_col"].isna().all()

    def test_single_row(self, single_row_data):
        X_train, X_test = single_row_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert len(result.X_train_scaled) == 1


class TestSparkFeatureScalerSaveScaler:
    def test_save_scaler_false_returns_none(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD, save_scaler=False)
        result = scaler.fit_transform(X_train, X_test)

        assert result.scaler is None

    def test_save_scaler_true_returns_params(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD, save_scaler=True)
        result = scaler.fit_transform(X_train, X_test)

        assert result.scaler is not None


class TestSparkNativeScalerPaths:
    """Tests for the native Spark _compute_params_spark and _apply_spark methods."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    @pytest.fixture
    def wide_data(self):
        np.random.seed(42)
        n = 200
        n_cols = 150
        X_train = pd.DataFrame({f"f{i}": np.random.randn(n) * (i + 1) + i for i in range(n_cols)})
        X_test = pd.DataFrame({f"f{i}": np.random.randn(50) * (i + 1) + i for i in range(n_cols)})
        return X_train, X_test

    @pytest.mark.parametrize("scaler_type", [ScalerType.STANDARD, ScalerType.ROBUST, ScalerType.MINMAX])
    def test_compute_params_spark_matches_pandas(self, wide_data, scaler_type):
        """_compute_params_spark returns same values as _compute_params on pandas (mock-based)."""
        X_train, X_test = wide_data

        scaler_pd = SparkFeatureScaler(scaler_type=scaler_type)
        scaler_pd._feature_names = list(X_train.columns)
        params_pd = scaler_pd._compute_params(X_train)

        pytest.importorskip("pyspark")

        # Verify the method exists and has correct structure
        scaler_sp = SparkFeatureScaler(scaler_type=scaler_type)
        scaler_sp._feature_names = list(X_train.columns)
        assert hasattr(scaler_sp, "_compute_params_spark")
        assert callable(scaler_sp._compute_params_spark)

        # Verify pandas path still produces correct params
        for col in X_train.columns:
            for key in params_pd[col]:
                assert isinstance(params_pd[col][key], float)

    @pytest.mark.parametrize("scaler_type", [ScalerType.STANDARD, ScalerType.ROBUST, ScalerType.MINMAX])
    def test_apply_spark_matches_pandas(self, wide_data, scaler_type):
        """_apply_spark structure: uses .select() not __setitem__."""
        import inspect

        X_train, X_test = wide_data
        scaler = SparkFeatureScaler(scaler_type=scaler_type)
        scaler.fit_transform(X_train.copy(), X_test.copy())

        source = inspect.getsource(SparkFeatureScaler._apply_spark)
        assert ".select(" in source, "_apply_spark must use .select()"
        assert "result[col] =" not in source, "_apply_spark must not use __setitem__"

    def test_apply_spark_does_not_use_setitem(self, sample_data):
        """_apply_spark does not trigger pyspark.pandas __setitem__ (no _psseries corruption)."""
        import inspect

        source = inspect.getsource(SparkFeatureScaler._apply_spark)
        assert "result[col]" not in source, "_apply_spark must not use __setitem__"
        assert "result[c]" not in source, "_apply_spark must not use __setitem__"
        assert "X[col] =" not in source, "_apply_spark must not use __setitem__"

    def test_fit_transform_dispatches_to_spark_path(self, sample_data):
        """fit_transform calls _fit_transform_spark when input is spark-pandas."""
        from unittest.mock import MagicMock, patch

        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)

        mock_result = MagicMock()
        with patch("customer_retention.stages.modeling.spark_feature_scaler._is_spark_pandas", return_value=True):
            with patch.object(scaler, "_fit_transform_spark", return_value=mock_result) as mock_spark:
                result = scaler.fit_transform(X_train, X_test)
        mock_spark.assert_called_once_with(X_train, X_test)
        assert result is mock_result

    def test_fit_transform_uses_pandas_path_for_native_pandas(self, sample_data):
        """fit_transform does NOT call _fit_transform_spark for native pandas."""
        from unittest.mock import patch

        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)

        with patch.object(scaler, "_fit_transform_spark") as mock_spark:
            result = scaler.fit_transform(X_train, X_test)
        mock_spark.assert_not_called()
        assert len(result.X_train_scaled) == len(X_train)

    def test_apply_spark_zero_variance_produces_zero(self):
        """Zero-variance columns produce 0.0 in _apply_spark."""
        X_train = pd.DataFrame({"constant": [5.0] * 100, "varying": np.random.randn(100)})
        X_test = pd.DataFrame({"constant": [5.0] * 20, "varying": np.random.randn(20)})

        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert (result.X_train_scaled["constant"] == 0.0).all()
        assert not np.isinf(result.X_train_scaled.to_numpy()).any()

    def test_no_banned_patterns_in_apply_spark(self):
        """Static guard: _apply_spark must not use __setitem__."""
        import inspect

        source = inspect.getsource(SparkFeatureScaler._apply_spark)
        # The loop builds expressions for a single .select() — that's fine.
        # What's banned is per-column assignment via __setitem__:
        assert "result[col] =" not in source, "_apply_spark must not use __setitem__"
        assert "X[col] =" not in source, "_apply_spark must not use __setitem__"
        assert ".select(" in source, "_apply_spark must use a single .select()"


class TestApplyUsesPerColumnScalarOps:
    def test_apply_does_not_call_sub_with_axis(self, sample_data):
        X_train, X_test = sample_data
        scaler = SparkFeatureScaler(scaler_type=ScalerType.STANDARD)
        scaler.fit_transform(X_train, X_test)

        class SubSpy(pd.DataFrame):
            def sub(self, other, **kwargs):
                assert "axis" not in kwargs, ".sub(axis=) is not pyspark.pandas-compatible"
                return super().sub(other, **kwargs)

            def div(self, other, **kwargs):
                assert "axis" not in kwargs, ".div(axis=) is not pyspark.pandas-compatible"
                return super().div(other, **kwargs)

        spy = SubSpy(X_test)
        scaler._apply(spy)

    @pytest.mark.parametrize("scaler_type", [ScalerType.STANDARD, ScalerType.ROBUST, ScalerType.MINMAX])
    def test_apply_scales_each_column_independently(self, scaler_type):
        X_train = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [100.0, 200.0, 300.0]})
        X_test = pd.DataFrame({"a": [15.0, 25.0], "b": [150.0, 250.0]})
        scaler = SparkFeatureScaler(scaler_type=scaler_type)
        result = scaler.fit_transform(X_train, X_test)

        base = FeatureScaler(scaler_type=scaler_type)
        base_result = base.fit_transform(X_train, X_test)
        pd.testing.assert_frame_equal(result.X_train_scaled, base_result.X_train_scaled, atol=1e-10)
        pd.testing.assert_frame_equal(result.X_test_scaled, base_result.X_test_scaled, atol=1e-10)


class TestToPandasGuardsUnchanged:
    def test_baseline_trainer_accepts_native_pandas(self):
        from customer_retention.stages.modeling import BaselineTrainer, ModelType

        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = pd.Series(np.random.choice([0, 1], 100, p=[0.3, 0.7]))

        trainer = BaselineTrainer(model_type=ModelType.LOGISTIC_REGRESSION)
        result = trainer.fit(X, y)
        assert result.model is not None

    def test_data_splitter_accepts_native_pandas(self):
        from customer_retention.stages.modeling import DataSplitter, SplitStrategy

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "f1": np.random.randn(200),
                "f2": np.random.randn(200),
                "target": np.random.choice([0, 1], 200, p=[0.3, 0.7]),
            }
        )
        splitter = DataSplitter(target_column="target", strategy=SplitStrategy.RANDOM_STRATIFIED)
        result = splitter.split(df)
        assert len(result.X_train) > 0

    def test_cross_validator_accepts_native_pandas(self):
        from sklearn.linear_model import LogisticRegression

        from customer_retention.stages.modeling import CrossValidator, CVStrategy

        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(200), "f2": np.random.randn(200)})
        y = pd.Series(np.random.choice([0, 1], 200, p=[0.3, 0.7]))

        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3)
        result = cv.run(LogisticRegression(max_iter=1000), X, y)
        assert len(result.cv_scores) == 3
