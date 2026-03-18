"""Tests for native Spark transform ops.

These verify parity with the pandas ops in ops.py.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

pytestmark = pytest.mark.spark

from customer_retention.transforms import spark_ops


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    try:
        return SparkSession.builder.master("local[1]").appName("test_spark_ops").getOrCreate()
    except RuntimeError:
        pytest.skip("local Spark session not available")


@pytest.fixture
def sample_sdf(spark):
    pdf = pd.DataFrame({
        "a": [1.0, 0.0, 5.0, 100.0],
        "b": [2.0, 4.0, 0.0, 8.0],
        "cat": ["x", "y", "x", "z"],
    })
    return spark.createDataFrame(pdf)


class TestSparkLogTransform:
    def test_applies_log1p(self, sample_sdf):
        result = spark_ops.spark_log_transform(sample_sdf, "a").toPandas()
        expected = np.log1p([1.0, 0.0, 5.0, 100.0])
        np.testing.assert_array_almost_equal(result["a"].values, expected)

    def test_missing_column_returns_unchanged(self, sample_sdf):
        result = spark_ops.spark_log_transform(sample_sdf, "missing")
        assert result.columns == sample_sdf.columns


class TestSparkSqrtTransform:
    def test_applies_sqrt_abs(self, sample_sdf):
        result = spark_ops.spark_sqrt_transform(sample_sdf, "a").toPandas()
        expected = np.sqrt([1.0, 0.0, 5.0, 100.0])
        np.testing.assert_array_almost_equal(result["a"].values, expected)


class TestSparkCapThenLog:
    def test_caps_at_q99_and_logs(self, sample_sdf):
        result = spark_ops.spark_cap_then_log(sample_sdf, "a", q99=50.0).toPandas()
        expected = np.log1p(np.clip([1.0, 0.0, 5.0, 100.0], 0, 50.0))
        np.testing.assert_array_almost_equal(result["a"].values, expected)


class TestSparkZeroInflation:
    def test_creates_is_zero_and_transforms(self, sample_sdf):
        result = spark_ops.spark_zero_inflation(sample_sdf, "a").toPandas()
        assert "a_is_zero" in result.columns
        assert list(result["a_is_zero"]) == [0, 1, 0, 0]
        assert result.loc[1, "a"] == 0.0
        assert result.loc[0, "a"] == np.log1p(1.0)


class TestSparkDerivedRatio:
    def test_ratio(self, sample_sdf):
        result = spark_ops.spark_derived_ratio(sample_sdf, "r", numerator="a", denominator="b").toPandas()
        assert "r" in result.columns
        assert result.loc[0, "r"] == pytest.approx(0.5)
        assert pd.isna(result.loc[2, "r"])


class TestSparkDerivedInteraction:
    def test_interaction(self, sample_sdf):
        result = spark_ops.spark_derived_interaction(sample_sdf, "ab", col_a="a", col_b="b").toPandas()
        assert list(result["ab"]) == [2.0, 0.0, 0.0, 800.0]


class TestSparkOneHotEncode:
    def test_encodes_categories(self, sample_sdf):
        result = spark_ops.spark_one_hot_encode(sample_sdf, "cat").toPandas()
        assert "cat" not in result.columns
        assert "cat_x" in result.columns
        assert "cat_y" in result.columns
        assert "cat_z" in result.columns


class TestSparkImpute:
    def test_fills_with_value(self, spark):
        sdf = spark.createDataFrame(pd.DataFrame({"x": [1.0, None, 3.0]}))
        result = spark_ops.spark_impute_null(sdf, "x", value=0).toPandas()
        assert list(result["x"]) == [1.0, 0.0, 3.0]


class TestSparkFeatureSelect:
    def test_drops_column(self, sample_sdf):
        result = spark_ops.spark_feature_select(sample_sdf, "a")
        assert "a" not in result.columns
        assert "b" in result.columns


class TestSparkStandardScale:
    def test_scales_to_zero_mean(self, spark):
        import numpy as np
        np.random.seed(42)
        pdf = pd.DataFrame({"v": np.random.randn(100) * 10 + 50})
        sdf = spark.createDataFrame(pdf)
        mean, std = float(pdf["v"].mean()), float(pdf["v"].std(ddof=0))
        result = spark_ops.spark_standard_scale(sdf, "v", mean=mean, scale=std).toPandas()
        assert abs(result["v"].mean()) < 0.1
        assert abs(result["v"].std() - 1.0) < 0.15

    def test_missing_column_noop(self, sample_sdf):
        result = spark_ops.spark_standard_scale(sample_sdf, "missing", mean=0.0, scale=1.0)
        assert result.columns == sample_sdf.columns

    def test_matches_fitted_scaler(self, spark):
        import numpy as np

        from customer_retention.transforms.fitted import FittedScaler
        np.random.seed(42)
        pdf = pd.DataFrame({"v": np.random.randn(100) * 10 + 50})
        scaler = FittedScaler("standard")
        pandas_result = scaler.fit_transform(pdf.copy(), "v", type("Store", (), {"register": lambda *a: None})())
        mean, scale = float(scaler._scaler.mean_[0]), float(scaler._scaler.scale_[0])
        sdf = spark.createDataFrame(pdf)
        spark_result = spark_ops.spark_standard_scale(sdf, "v", mean=mean, scale=scale).toPandas()
        np.testing.assert_array_almost_equal(spark_result["v"].values, pandas_result["v"].values)


class TestSparkMinmaxScale:
    def test_scales_to_0_1(self, spark):
        pdf = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        sdf = spark.createDataFrame(pdf)
        result = spark_ops.spark_minmax_scale(sdf, "v", scale=0.25, offset=-0.25).toPandas()
        assert result["v"].min() == pytest.approx(0.0)
        assert result["v"].max() == pytest.approx(1.0)

    def test_missing_column_noop(self, sample_sdf):
        result = spark_ops.spark_minmax_scale(sample_sdf, "missing", scale=1.0, offset=0.0)
        assert result.columns == sample_sdf.columns


class TestSparkYeoJohnson:
    def test_positive_values(self, spark):
        pdf = pd.DataFrame({"v": [0.0, 1.0, 2.0, 10.0]})
        sdf = spark.createDataFrame(pdf)
        result = spark_ops.spark_yeo_johnson(sdf, "v", lmbda=0.5, standardize=False).toPandas()
        expected = ((pdf["v"] + 1) ** 0.5 - 1) / 0.5
        np.testing.assert_array_almost_equal(result["v"].values, expected.values)

    def test_lambda_zero_uses_log1p(self, spark):
        pdf = pd.DataFrame({"v": [0.0, 1.0, 5.0]})
        sdf = spark.createDataFrame(pdf)
        result = spark_ops.spark_yeo_johnson(sdf, "v", lmbda=0.0, standardize=False).toPandas()
        expected = np.log1p(pdf["v"])
        np.testing.assert_array_almost_equal(result["v"].values, expected.values)

    def test_lambda_two_neg_uses_neg_log1p(self, spark):
        pdf = pd.DataFrame({"v": [-1.0, -2.0, -5.0]})
        sdf = spark.createDataFrame(pdf)
        result = spark_ops.spark_yeo_johnson(sdf, "v", lmbda=2.0, standardize=False).toPandas()
        expected = -np.log1p(-pdf["v"])
        np.testing.assert_array_almost_equal(result["v"].values, expected.values)

    def test_standardize(self, spark):
        pdf = pd.DataFrame({"v": [0.0, 1.0, 2.0, 3.0]})
        sdf = spark.createDataFrame(pdf)
        result = spark_ops.spark_yeo_johnson(sdf, "v", lmbda=1.5, std_mean=2.0, std_scale=0.5, standardize=True).toPandas()
        raw = ((pdf["v"] + 1) ** 1.5 - 1) / 1.5
        expected = (raw - 2.0) / 0.5
        np.testing.assert_array_almost_equal(result["v"].values, expected.values)

    def test_nulls_filled_with_zero(self, spark):
        pdf = pd.DataFrame({"v": [1.0, None, 3.0]})
        sdf = spark.createDataFrame(pdf)
        result = spark_ops.spark_yeo_johnson(sdf, "v", lmbda=0.5, standardize=False).toPandas()
        assert not result["v"].isna().any()
        zero_transformed = ((0.0 + 1) ** 0.5 - 1) / 0.5
        assert result["v"].iloc[1] == pytest.approx(zero_transformed)

    def test_missing_column_noop(self, sample_sdf):
        result = spark_ops.spark_yeo_johnson(sample_sdf, "missing", lmbda=0.5)
        assert result.columns == sample_sdf.columns

    def test_matches_fitted_power_transform(self, spark):
        from customer_retention.transforms.fitted import FittedPowerTransform
        np.random.seed(42)
        pdf = pd.DataFrame({"v": np.random.exponential(5, 200)})
        pt = FittedPowerTransform()
        pt.fit_from_local(pdf["v"])
        pandas_result = pt._apply_yj(pdf["v"].fillna(0))
        lmbda = float(pt._pt.lambdas_[0])
        std_mean = float(pt._pt._scaler.mean_[0])
        std_scale = float(pt._pt._scaler.scale_[0])
        sdf = spark.createDataFrame(pdf)
        spark_result = spark_ops.spark_yeo_johnson(sdf, "v", lmbda=lmbda, std_mean=std_mean, std_scale=std_scale, standardize=True).toPandas()
        np.testing.assert_array_almost_equal(spark_result["v"].values, pandas_result.values, decimal=6)
