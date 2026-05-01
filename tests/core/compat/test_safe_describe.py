import math

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import safe_describe


class TestSafeDescribe:
    def test_numeric_only(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = safe_describe(df)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "mean" in result.index

    def test_mixed_types(self):
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "ts": pd.date_range("2024-01-01", periods=3),
            "cat": ["a", "b", "c"],
        })
        result = safe_describe(df)
        assert len(result.columns) > 0

    def test_timestamp_only_fallback(self):
        df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3)})
        result = safe_describe(df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = safe_describe(df)
        assert isinstance(result, pd.DataFrame)

    def test_describe_failure_falls_back_to_numeric(self):
        class BrokenDescribeDF:
            def __init__(self):
                self._df = pd.DataFrame({"a": [1, 2, 3]})
                self.columns = self._df.columns

            def describe(self):
                raise RuntimeError("avg on TIMESTAMP_NTZ not supported")

            def select_dtypes(self, include=None):
                return self._df.select_dtypes(include=include)

        result = safe_describe(BrokenDescribeDF())
        assert "a" in result.columns
        assert "mean" in result.index


class TestSafeDescribeSparkBulk:
    """Spark path uses ``_spark_bulk_describe`` — two batched ``.agg()``
    calls regardless of column count, plan size O(N).

    Replaces pyspark.pandas's per-column describe path which scaled
    linearly in Spark jobs (200 cols ≈ 200+ jobs, ~45 min observed on
    SPS opportunity df_aggregated). The matrix-batched form keeps total
    job count bounded at ≈ 5-10 even for 300+ column inputs.
    """

    pyspark = pytest.importorskip("pyspark")
    pytestmark = pytest.mark.spark

    @pytest.fixture(scope="class")
    def spark(self):
        from pyspark.sql import SparkSession
        return (
            SparkSession.builder
            .master("local[2]")
            .appName("safe_describe_bulk")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )

    def _make_ps_df(self, spark, n_rows, n_cols, seed=42, null_frac=0.1):
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_rows, n_cols))
        if null_frac > 0:
            mask = rng.random((n_rows, n_cols)) < null_frac
            data[mask] = np.nan
        cols = [f"f{i:03d}" for i in range(n_cols)]
        pdf = pd.DataFrame(data, columns=cols)
        return _as_pandas_api(spark.createDataFrame(pdf)), pdf, cols

    def test_returns_describe_shape(self, spark):
        ps_df, pdf, cols = self._make_ps_df(spark, n_rows=200, n_cols=4)
        result = safe_describe(ps_df)
        assert list(result.index) == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        for c in cols:
            assert c in result.columns

    def test_count_mean_std_match_pandas(self, spark):
        ps_df, pdf, cols = self._make_ps_df(spark, n_rows=500, n_cols=10, null_frac=0.2)
        result = safe_describe(ps_df)
        expected = pdf.describe()
        for c in cols:
            assert result.loc["count", c] == pytest.approx(expected.loc["count", c])
            assert result.loc["mean", c] == pytest.approx(expected.loc["mean", c], abs=1e-9)
            assert result.loc["std", c] == pytest.approx(expected.loc["std", c], abs=1e-9)
            assert result.loc["min", c] == pytest.approx(expected.loc["min", c], abs=1e-12)
            assert result.loc["max", c] == pytest.approx(expected.loc["max", c], abs=1e-12)

    def test_quantiles_approx_match_pandas(self, spark):
        # percentile_approx is approximate; validate within Spark's default
        # relativeError (~1%) bound. Big enough sample to make approx tight.
        ps_df, pdf, cols = self._make_ps_df(spark, n_rows=2000, n_cols=5, null_frac=0.0)
        result = safe_describe(ps_df)
        expected = pdf.describe()
        for c in cols:
            for q in ("25%", "50%", "75%"):
                got = result.loc[q, c]
                exp = expected.loc[q, c]
                assert abs(got - exp) < 0.05, f"{c}/{q}: spark={got}, pandas={exp}"

    def test_wide_n_uses_bounded_jobs(self, spark):
        """Regression: 250-column describe must complete promptly using
        batched aggs, not 250 per-column jobs.

        Asserts shape + that count column matches non-null count, which
        proves the batched aggregation actually scanned all columns. The
        speed property is enforced implicitly by pytest-timeout (default
        in conftest) and by the reduced job count in the Spark UI.
        """
        ps_df, pdf, cols = self._make_ps_df(spark, n_rows=400, n_cols=250, null_frac=0.15)
        result = safe_describe(ps_df)
        assert result.shape == (8, 250)
        expected_counts = pdf.notna().sum()
        for c in cols[::25]:  # spot-check every 25th column
            assert result.loc["count", c] == pytest.approx(float(expected_counts[c]))

    def test_non_numeric_only_returns_empty(self, spark):
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        pdf = pd.DataFrame({"s": ["a", "b", "c"]})
        ps_df = _as_pandas_api(spark.createDataFrame(pdf))
        result = safe_describe(ps_df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_mixed_numeric_and_string(self, spark):
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "label": ["a", "b", "c", "d"]})
        ps_df = _as_pandas_api(spark.createDataFrame(pdf))
        result = safe_describe(ps_df)
        # Only numeric column appears
        assert list(result.columns) == ["x"]
        assert result.loc["count", "x"] == pytest.approx(4.0)
        assert result.loc["mean", "x"] == pytest.approx(2.5)

    def test_all_null_column_emits_nan_stats(self, spark):
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        pdf = pd.DataFrame({"v": [np.nan, np.nan, np.nan]})
        ps_df = _as_pandas_api(spark.createDataFrame(pdf))
        result = safe_describe(ps_df)
        assert result.loc["count", "v"] == pytest.approx(0.0)
        assert math.isnan(result.loc["mean", "v"])
        assert math.isnan(result.loc["std", "v"])
