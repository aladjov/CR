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
