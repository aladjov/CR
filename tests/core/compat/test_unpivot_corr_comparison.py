"""Verify batched F.corr() correlation implementations against pandas ground truth.

Requires PySpark — skipped on CI.
"""
import math
import time

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
pytestmark = pytest.mark.spark

import pandas as pd
from pyspark.sql import SparkSession

from customer_retention.core.compat import (
    _spark_batched_corr_with_target,
    _spark_batched_leakage_corr_combined,
)
from customer_retention.core.compat.spark_backend import _as_pandas_api


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[*]").appName("corr_compare").getOrCreate()


def _make_test_df(spark, n_rows, n_feature_cols, null_frac=0.1):
    rng = np.random.default_rng(42)
    data = {"target": rng.choice([0, 1], n_rows, p=[0.6, 0.4]).astype(float)}
    for i in range(n_feature_cols):
        col = rng.normal(0, 100, n_rows)
        if i % 5 == 0:
            mask = rng.random(n_rows) < null_frac
            col[mask] = np.nan
        if i % 7 == 0:
            col[:] = 42.0
        data[f"f{i:04d}"] = col
    pdf = pd.DataFrame(data)
    spark_df = spark.createDataFrame(pdf)
    ps_df = _as_pandas_api(spark_df)
    return ps_df, pdf, [c for c in pdf.columns if c != "target"]


class TestBatchedCorrWithTargetVsPandas:
    @pytest.mark.parametrize("n_cols", [50, 200, 500])
    def test_matches_pandas_corr(self, spark, n_cols):
        ps_df, pdf, columns = _make_test_df(spark, 2000, n_cols)
        t0 = time.monotonic()
        spark_result = _spark_batched_corr_with_target(ps_df, columns, "target")
        t_spark = time.monotonic() - t0
        for c in columns:
            expected = pdf[c].corr(pdf["target"])
            sv = spark_result[c]
            if math.isnan(expected):
                assert math.isnan(sv), f"{c}: expected=NaN, got={sv}"
            else:
                assert abs(sv - expected) < 1e-6, f"{c}: expected={expected}, got={sv}"
        print(f"\n  batched F.corr ({n_cols} cols): {t_spark:.2f}s")


class TestBatchedLeakageCorrVsPandas:
    @pytest.mark.parametrize("n_cols", [50, 200])
    def test_matches_pandas_corr(self, spark, n_cols):
        ps_df, pdf, columns = _make_test_df(spark, 2000, n_cols)
        t0 = time.monotonic()
        null_corrs, value_corrs = _spark_batched_leakage_corr_combined(ps_df, columns, "target")
        t_spark = time.monotonic() - t0
        target = pdf["target"]
        for c in columns:
            expected_val = pdf[c].corr(target)
            expected_null = pdf[c].isnull().astype(float).corr(target)
            sv, sn = value_corrs.get(c, math.nan), null_corrs.get(c, math.nan)
            if math.isnan(expected_val):
                assert math.isnan(sv), f"{c} val: expected=NaN, got={sv}"
            else:
                assert abs(sv - expected_val) < 1e-6, f"{c} val: expected={expected_val}, got={sv}"
            if math.isnan(expected_null):
                assert math.isnan(sn), f"{c} null: expected=NaN, got={sn}"
            else:
                assert abs(sn - expected_null) < 1e-6, f"{c} null: expected={expected_null}, got={sn}"
        print(f"\n  batched leakage corr ({n_cols} cols): {t_spark:.2f}s")
