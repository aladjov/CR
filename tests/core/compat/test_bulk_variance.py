from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import bulk_variance


class TestBulkVariance:

    def test_matches_pandas_var(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100), "c": rng.standard_normal(100)})
        result = bulk_variance(df, ["a", "b", "c"])
        expected = df[["a", "b", "c"]].var()
        pd.testing.assert_series_equal(result, expected)

    def test_empty_columns_returns_empty(self):
        df = pd.DataFrame({"a": [1.0, 2, 3]})
        result = bulk_variance(df, [])
        assert len(result) == 0

    def test_single_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        result = bulk_variance(df, ["x"])
        assert result["x"] == pytest.approx(df["x"].var())

    def test_constant_column_zero_variance(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0, 4.0]})
        result = bulk_variance(df, ["a", "b"])
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] > 0

    def test_all_nan_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
        result = bulk_variance(df, ["a", "b"])
        assert np.isnan(result["a"])
        assert result["b"] == pytest.approx(df["b"].var())

    def test_preserves_column_order(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"z": rng.standard_normal(50), "a": rng.standard_normal(50), "m": rng.standard_normal(50)})
        result = bulk_variance(df, ["z", "a", "m"])
        assert list(result.index) == ["z", "a", "m"]

    def test_many_columns_matches_pandas(self):
        """Verify batching does not lose columns when count exceeds _AGG_BATCH_SIZE."""
        rng = np.random.default_rng(123)
        ncols = 600
        df = pd.DataFrame({f"f_{i}": rng.standard_normal(50) for i in range(ncols)})
        cols = list(df.columns)
        result = bulk_variance(df, cols)
        expected = df.var()
        pd.testing.assert_series_equal(result, expected)
