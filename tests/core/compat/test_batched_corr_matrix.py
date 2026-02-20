from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import batched_corr_matrix


class TestBatchedCorrMatrix:

    def test_returns_correlation_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [4.0, 3, 2, 1], "c": [1.0, 1, 1, 1]})
        result = batched_corr_matrix(df, ["a", "b"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert list(result.index) == ["a", "b"]

    def test_matches_pandas_corr(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.standard_normal(100),
            "y": rng.standard_normal(100),
            "z": rng.standard_normal(100),
        })
        expected = df[["x", "y", "z"]].corr()
        result = batched_corr_matrix(df, ["x", "y", "z"])
        pd.testing.assert_frame_equal(result, expected)

    def test_perfect_negative_correlation(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [5.0, 4, 3, 2, 1]})
        result = batched_corr_matrix(df, ["a", "b"])
        assert result.loc["a", "b"] == pytest.approx(-1.0)

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6]})
        result = batched_corr_matrix(df, ["a", "b", "missing"])
        assert list(result.columns) == ["a", "b"]

    def test_single_column_returns_empty(self):
        df = pd.DataFrame({"a": [1.0, 2, 3]})
        result = batched_corr_matrix(df, ["a"])
        assert result.shape == (1, 1)

    def test_no_valid_columns_returns_empty(self):
        df = pd.DataFrame({"a": [1.0, 2, 3]})
        result = batched_corr_matrix(df, ["missing1", "missing2"])
        assert result.empty

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(99)
        df = pd.DataFrame({"a": rng.standard_normal(50), "b": rng.standard_normal(50), "c": rng.standard_normal(50)})
        result = batched_corr_matrix(df, ["a", "b", "c"])
        for col in ["a", "b", "c"]:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"p": rng.standard_normal(80), "q": rng.standard_normal(80)})
        result = batched_corr_matrix(df, ["p", "q"])
        assert result.loc["p", "q"] == pytest.approx(result.loc["q", "p"])
