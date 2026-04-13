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


class TestBatchedMissingnessSparkCheckpointed:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_spark_path_calls_localcheckpoint(self):
        from unittest.mock import MagicMock, patch

        cols = ["a", "b"]
        mock_df = MagicMock()
        mock_df.columns = cols

        mock_spark_df = MagicMock()
        mock_null_df = MagicMock()
        mock_checkpointed = MagicMock()
        mock_spark_df.select.return_value = mock_null_df
        mock_null_df.localCheckpoint.return_value = mock_checkpointed
        null_count_row = MagicMock()
        null_count_row.__getitem__ = lambda self, k: 0
        mock_checkpointed.agg.return_value.head.return_value = null_count_row

        with (
            patch("customer_retention.core.compat._is_spark_pandas", return_value=True),
            patch("customer_retention.core.compat.as_spark_df", return_value=mock_spark_df),
        ):
            batched_missingness_corr(mock_df, cols)

        mock_null_df.localCheckpoint.assert_called_once_with(eager=True)
        mock_checkpointed.agg.assert_called()


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
