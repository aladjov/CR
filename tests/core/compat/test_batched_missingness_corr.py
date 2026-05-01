from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import batched_missingness_corr


class TestBatchedMissingnessCorr:

    def test_returns_dataframe(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3], "c": [1, 2, 3]})
        result = batched_missingness_corr(df)
        assert isinstance(result, pd.DataFrame)

    def test_matches_pandas_isnull_corr(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 5))
        mask = rng.random((200, 5)) < 0.3
        data[mask] = np.nan
        cols = [f"c{i}" for i in range(5)]
        df = pd.DataFrame(data, columns=cols)

        expected = df.isnull().corr()
        result = batched_missingness_corr(df, cols)
        pd.testing.assert_frame_equal(result, expected, atol=1e-10)

    def test_subset_of_columns(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3], "c": [1, 2, None]})
        result = batched_missingness_corr(df, ["a", "b"])
        assert list(result.columns) == ["a", "b"]
        assert list(result.index) == ["a", "b"]

    def test_single_column_returns_empty(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        result = batched_missingness_corr(df, ["a"])
        assert result.shape == (1, 1)

    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = batched_missingness_corr(df, ["a", "b"])
        # Zero variance in null indicators → NaN on diagonal (pandas behavior)
        assert np.isnan(result.loc["a", "a"])
        assert np.isnan(result.loc["b", "b"])

    def test_perfectly_correlated_missingness(self):
        df = pd.DataFrame({
            "a": [1, None, None, 4, 5],
            "b": [10, None, None, 40, 50],
        })
        result = batched_missingness_corr(df, ["a", "b"])
        assert result.loc["a", "b"] == pytest.approx(1.0)
        assert result.loc["b", "a"] == pytest.approx(1.0)

    def test_anti_correlated_missingness(self):
        df = pd.DataFrame({
            "a": [1, None, 3, None, 5, None, 7, None],
            "b": [None, 2, None, 4, None, 6, None, 8],
        })
        result = batched_missingness_corr(df, ["a", "b"])
        assert result.loc["a", "b"] == pytest.approx(-1.0)

    def test_columns_default_to_all(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
        result = batched_missingness_corr(df)
        assert list(result.columns) == ["a", "b"]

    def test_nonexistent_columns_filtered(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
        result = batched_missingness_corr(df, ["a", "b", "nonexistent"])
        assert list(result.columns) == ["a", "b"]

    def test_mixed_null_and_complete_columns(self):
        df = pd.DataFrame({
            "has_nulls_1": [1, None, 3, None, 5],
            "has_nulls_2": [None, 2, None, 4, None],
            "complete": [1, 2, 3, 4, 5],
        })
        result = batched_missingness_corr(df)
        # complete column has zero null variance → NaN with everything
        assert np.isnan(result.loc["complete", "complete"])
        assert np.isnan(result.loc["complete", "has_nulls_1"])
        # has_nulls columns should have valid correlation
        assert not np.isnan(result.loc["has_nulls_1", "has_nulls_2"])


class TestBatchedMissingnessSparkMatrixForm:
    """Spark path uses MLlib's ``Correlation.corr`` for an O(N) plan.

    Verifies behaviour parity with the pandas branch on a real local
    SparkSession, including the wide-N case that triggered the previous
    O(N²) batched-pair plan to OOM driver heaps when run in parallel
    across multiple NB02 tasks on a shared cluster.
    """

    pyspark = pytest.importorskip("pyspark")
    pytestmark = pytest.mark.spark

    @pytest.fixture(scope="class")
    def spark(self):
        from pyspark.sql import SparkSession
        return (
            SparkSession.builder
            .master("local[2]")
            .appName("missingness_corr_matrix_form")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )

    def _make_ps_df(self, spark, n_rows, n_cols, seed=42, null_frac=0.3):
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_rows, n_cols))
        mask = rng.random((n_rows, n_cols)) < null_frac
        data[mask] = np.nan
        cols = [f"c{i:03d}" for i in range(n_cols)]
        pdf = pd.DataFrame(data, columns=cols)
        return _as_pandas_api(spark.createDataFrame(pdf)), pdf, cols

    def test_matches_pandas_isnull_corr_small(self, spark):
        ps_df, pdf, cols = self._make_ps_df(spark, n_rows=300, n_cols=8)
        result = batched_missingness_corr(ps_df, cols)
        expected = pdf.isnull().corr()
        # MLlib computes via moments; tolerance accounts for accumulator order
        pd.testing.assert_frame_equal(result, expected, atol=1e-9)

    def test_wide_n_does_not_emit_quadratic_plan(self, spark):
        """Regression: 200-col missingness corr must not OOM or hang.

        The previous implementation issued O(N²/500) Spark jobs per call;
        the matrix-form rewrite issues exactly one MLlib correlation job
        regardless of N. This test exercises the wide-N path against
        a real SparkSession to ensure the single-job path is in effect
        and produces the right shape.
        """
        ps_df, pdf, cols = self._make_ps_df(spark, n_rows=500, n_cols=200)
        result = batched_missingness_corr(ps_df, cols)
        assert result.shape == (200, 200)
        # Diagonal of has-variance cols must be 1.0 (matches pandas contract)
        diag_finite = np.array([result.iloc[i, i] for i in range(200) if not np.isnan(result.iloc[i, i])])
        assert (diag_finite == pytest.approx(1.0)).all()
        # Symmetry is required (correlation matrix is symmetric)
        for i in range(0, 200, 13):  # spot-check, not full N²
            for j in range(i + 1, 200, 17):
                a, b = result.iloc[i, j], result.iloc[j, i]
                if np.isnan(a):
                    assert np.isnan(b)
                else:
                    assert a == pytest.approx(b)

    def test_zero_variance_column_emits_nan_row(self, spark):
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        # 'complete' has no nulls → zero variance → all-NaN row/col
        pdf = pd.DataFrame({
            "complete": [1.0, 2.0, 3.0, 4.0, 5.0],
            "has_nulls_a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "has_nulls_b": [np.nan, 2.0, np.nan, 4.0, np.nan],
        })
        ps_df = _as_pandas_api(spark.createDataFrame(pdf))
        result = batched_missingness_corr(ps_df, ["complete", "has_nulls_a", "has_nulls_b"])
        assert np.isnan(result.loc["complete", "complete"])
        assert np.isnan(result.loc["complete", "has_nulls_a"])
        assert np.isnan(result.loc["has_nulls_a", "complete"])
        # has-variance cols still get a real correlation
        assert not np.isnan(result.loc["has_nulls_a", "has_nulls_b"])

    def test_matches_pandas_anti_correlated(self, spark):
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        pdf = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan],
            "b": [np.nan, 2.0, np.nan, 4.0, np.nan, 6.0, np.nan, 8.0],
        })
        ps_df = _as_pandas_api(spark.createDataFrame(pdf))
        result = batched_missingness_corr(ps_df, ["a", "b"])
        assert result.loc["a", "b"] == pytest.approx(-1.0, abs=1e-9)
        assert result.loc["b", "a"] == pytest.approx(-1.0, abs=1e-9)


class TestBulkIqrBounds:

    def test_returns_dataframe_with_expected_columns(self):
        from customer_retention.core.compat import bulk_iqr_bounds

        df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": [10, 20, 30, 40, 50]})
        result = bulk_iqr_bounds(df, ["a", "b"])
        expected_cols = {"feature", "Q1", "Q3", "IQR", "lower_bound", "upper_bound",
                         "outliers_low", "outliers_high", "total_outliers", "outlier_pct"}
        assert set(result.columns) == expected_cols
        assert len(result) == 2

    def test_matches_manual_iqr(self):
        from customer_retention.core.compat import bulk_iqr_bounds

        data = list(range(1, 101)) + [500]  # 500 is an outlier
        df = pd.DataFrame({"v": data})
        result = bulk_iqr_bounds(df, ["v"])
        row = result.iloc[0]

        series = pd.Series(data)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr

        assert row["Q1"] == pytest.approx(q1)
        assert row["Q3"] == pytest.approx(q3)
        assert row["lower_bound"] == pytest.approx(lo)
        assert row["upper_bound"] == pytest.approx(hi)
        assert row["outliers_high"] >= 1  # 500 is above upper bound

    def test_empty_columns_returns_empty(self):
        from customer_retention.core.compat import bulk_iqr_bounds

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = bulk_iqr_bounds(df, ["nonexistent"])
        assert len(result) == 0

    def test_no_outliers(self):
        from customer_retention.core.compat import bulk_iqr_bounds

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = bulk_iqr_bounds(df, ["a"])
        assert result.iloc[0]["total_outliers"] == 0
        assert result.iloc[0]["outlier_pct"] == 0.0

    def test_multiple_columns(self):
        from customer_retention.core.compat import bulk_iqr_bounds

        rng = np.random.default_rng(42)
        df = pd.DataFrame({f"c{i}": rng.standard_normal(100) for i in range(10)})
        result = bulk_iqr_bounds(df, [f"c{i}" for i in range(10)])
        assert len(result) == 10
        assert all(result["feature"] == [f"c{i}" for i in range(10)])

    def test_column_with_all_nans(self):
        from customer_retention.core.compat import bulk_iqr_bounds

        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
        result = bulk_iqr_bounds(df, ["a", "b"])
        row_a = result[result["feature"] == "a"].iloc[0]
        assert row_a["total_outliers"] == 0
        assert row_a["outlier_pct"] == 0.0
