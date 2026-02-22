from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import bulk_corr_with_target


class TestBulkCorrWithTarget:

    def test_returns_dict(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [4.0, 3, 2, 1], "t": [1.0, 0, 1, 0]})
        result = bulk_corr_with_target(df, ["a", "b"], "t")
        assert isinstance(result, dict)

    def test_matches_pairwise_corr(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.standard_normal(100),
            "y": rng.standard_normal(100),
            "target": rng.integers(0, 2, size=100).astype(float),
        })
        result = bulk_corr_with_target(df, ["x", "y"], "target")
        for col in ["x", "y"]:
            expected = df[col].corr(df["target"])
            assert result[col] == pytest.approx(expected, abs=1e-10)

    def test_perfect_positive_correlation(self):
        df = pd.DataFrame({"a": [0.0, 1, 2, 3, 4], "t": [0.0, 1, 2, 3, 4]})
        result = bulk_corr_with_target(df, ["a"], "t")
        assert result["a"] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        df = pd.DataFrame({"a": [4.0, 3, 2, 1, 0], "t": [0.0, 1, 2, 3, 4]})
        result = bulk_corr_with_target(df, ["a"], "t")
        assert result["a"] == pytest.approx(-1.0)

    def test_zero_variance_column_returns_nan(self):
        df = pd.DataFrame({"const": [5.0, 5, 5, 5], "t": [0.0, 1, 0, 1]})
        result = bulk_corr_with_target(df, ["const"], "t")
        assert math.isnan(result["const"])

    def test_zero_variance_target_returns_nan(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4], "t": [1.0, 1, 1, 1]})
        result = bulk_corr_with_target(df, ["a"], "t")
        assert math.isnan(result["a"])

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1.0, 2, 3], "t": [0.0, 1, 0]})
        result = bulk_corr_with_target(df, ["a", "nonexistent"], "t")
        assert "a" in result
        assert "nonexistent" not in result

    def test_empty_columns_returns_empty_dict(self):
        df = pd.DataFrame({"t": [0.0, 1, 0]})
        result = bulk_corr_with_target(df, [], "t")
        assert result == {}

    def test_target_not_in_df_returns_empty_dict(self):
        df = pd.DataFrame({"a": [1.0, 2, 3]})
        result = bulk_corr_with_target(df, ["a"], "missing_target")
        assert result == {}

    def test_scattered_nulls_pairwise(self):
        rng = np.random.default_rng(42)
        n = 100
        target = rng.integers(0, 2, size=n).astype(float)
        feat_a = target * 3 + rng.standard_normal(n) * 0.5
        feat_b = -target * 2 + rng.standard_normal(n) * 0.5
        df = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "t": target})
        df.loc[0:20, "feat_a"] = np.nan
        df.loc[30:50, "feat_b"] = np.nan
        result = bulk_corr_with_target(df, ["feat_a", "feat_b"], "t")
        for col in ["feat_a", "feat_b"]:
            expected = df[col].corr(df["t"])
            assert result[col] == pytest.approx(expected, abs=1e-10)

    def test_all_nan_column_returns_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "t": [0.0, 1, 0]})
        result = bulk_corr_with_target(df, ["a"], "t")
        assert math.isnan(result["a"])

    def test_many_columns_single_call(self):
        rng = np.random.default_rng(7)
        n = 50
        data = {f"col_{i}": rng.standard_normal(n) for i in range(20)}
        data["target"] = rng.integers(0, 2, size=n).astype(float)
        df = pd.DataFrame(data)
        cols = [f"col_{i}" for i in range(20)]
        result = bulk_corr_with_target(df, cols, "target")
        assert len(result) == 20
        for col in cols:
            expected = df[col].corr(df["target"])
            assert result[col] == pytest.approx(expected, abs=1e-10)

    def test_does_not_include_target_in_result(self):
        df = pd.DataFrame({"a": [1.0, 2, 3], "t": [0.0, 1, 0]})
        result = bulk_corr_with_target(df, ["a", "t"], "t")
        assert "t" not in result
        assert "a" in result
