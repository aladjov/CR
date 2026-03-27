"""Compare batched vs unpivot correlation implementations.

Both approaches must produce numerically identical results. This test runs both
on the same data, asserts equivalence, and prints wall-clock timing so the slower
implementation can be identified and removed.

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
    _spark_bulk_corr_with_target,
    _spark_leakage_corr_combined,
)
from customer_retention.core.compat.__init__ import (
    _spark_unpivot_corr_with_target,
    _spark_unpivot_leakage_corr_combined,
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
    return ps_df, [c for c in pdf.columns if c != "target"]


class TestBulkCorrWithTargetEquivalence:
    @pytest.mark.parametrize("n_cols", [50, 200, 500])
    def test_identical_results(self, spark, n_cols):
        ps_df, columns = _make_test_df(spark, 2000, n_cols)

        t0 = time.monotonic()
        batched = _spark_bulk_corr_with_target(ps_df, columns, "target")
        t_batched = time.monotonic() - t0

        t0 = time.monotonic()
        unpivot = _spark_unpivot_corr_with_target(ps_df, columns, "target")
        t_unpivot = time.monotonic() - t0

        assert set(batched.keys()) == set(unpivot.keys()), "Key sets differ"
        for c in batched:
            bv, uv = batched[c], unpivot[c]
            if math.isnan(bv):
                assert math.isnan(uv), f"{c}: batched=NaN, unpivot={uv}"
            else:
                assert abs(bv - uv) < 1e-10, f"{c}: batched={bv}, unpivot={uv}"
        print(f"\n  bulk_corr_with_target ({n_cols} cols): batched={t_batched:.2f}s, unpivot={t_unpivot:.2f}s")


class TestLeakageCorrCombinedEquivalence:
    @pytest.mark.parametrize("n_cols", [50, 200, 500])
    def test_identical_results(self, spark, n_cols):
        ps_df, columns = _make_test_df(spark, 2000, n_cols)

        t0 = time.monotonic()
        batched_null, batched_val = _spark_leakage_corr_combined(ps_df, columns, "target")
        t_batched = time.monotonic() - t0

        t0 = time.monotonic()
        unpivot_null, unpivot_val = _spark_unpivot_leakage_corr_combined(ps_df, columns, "target")
        t_unpivot = time.monotonic() - t0

        assert set(batched_null.keys()) == set(unpivot_null.keys()), "Null key sets differ"
        assert set(batched_val.keys()) == set(unpivot_val.keys()), "Value key sets differ"
        for c in batched_null:
            bn, un = batched_null[c], unpivot_null[c]
            if math.isnan(bn):
                assert math.isnan(un), f"{c} null: batched=NaN, unpivot={un}"
            else:
                assert abs(bn - un) < 1e-10, f"{c} null: batched={bn}, unpivot={un}"
        for c in batched_val:
            bv, uv = batched_val[c], unpivot_val[c]
            if math.isnan(bv):
                assert math.isnan(uv), f"{c} val: batched=NaN, unpivot={uv}"
            else:
                assert abs(bv - uv) < 1e-10, f"{c} val: batched={bv}, unpivot={uv}"
        print(f"\n  leakage_corr_combined ({n_cols} cols): batched={t_batched:.2f}s, unpivot={t_unpivot:.2f}s")
