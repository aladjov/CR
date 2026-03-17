from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import bulk_max, bulk_skew


class TestBulkSkew:

    def test_returns_dict(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [5.0, 4, 3, 2, 1]})
        result = bulk_skew(df, ["a", "b"])
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_matches_pandas_skew(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({f"c_{i}": rng.standard_normal(100) for i in range(20)})
        cols = list(df.columns)
        result = bulk_skew(df, cols)
        expected = df.skew()
        for c in cols:
            assert result[c] == pytest.approx(expected[c], abs=1e-10)

    def test_skewed_distribution_detected(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"normal": rng.standard_normal(500), "skewed": rng.exponential(1, 500)})
        result = bulk_skew(df, ["normal", "skewed"])
        assert abs(result["normal"]) < 0.5
        assert result["skewed"] > 1.0

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1.0, 2, 3]})
        result = bulk_skew(df, ["a", "nonexistent"])
        assert "a" in result
        assert "nonexistent" not in result

    def test_empty_columns_returns_empty(self):
        df = pd.DataFrame({"a": [1.0, 2, 3]})
        assert bulk_skew(df, []) == {}

    def test_string_column_returns_nan(self):
        df = pd.DataFrame({"cat": ["a", "b", "c", "d"], "num": [1.0, 2, 3, 4]})
        result = bulk_skew(df, ["cat", "num"])
        assert math.isnan(result["cat"])
        assert not math.isnan(result["num"])

    def test_constant_column(self):
        df = pd.DataFrame({"const": [5.0] * 10, "var": np.arange(10, dtype=float)})
        result = bulk_skew(df, ["const", "var"])
        assert result["const"] == pytest.approx(0.0, abs=1e-10) or math.isnan(result["const"])

    def test_many_columns(self):
        rng = np.random.default_rng(99)
        df = pd.DataFrame({f"f_{i}": rng.standard_normal(50) for i in range(200)})
        result = bulk_skew(df, list(df.columns))
        assert len(result) == 200


class TestBulkMax:

    def test_returns_dict(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        result = bulk_max(df, ["a", "b"])
        assert result["a"] == 3
        assert result["b"] == 30

    def test_datetime_columns(self):
        df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=5)})
        result = bulk_max(df, ["d"])
        assert result["d"] == pd.Timestamp("2024-01-05")

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = bulk_max(df, ["a", "missing"])
        assert "a" in result
        assert "missing" not in result

    def test_empty_columns(self):
        assert bulk_max(pd.DataFrame({"a": [1]}), []) == {}

    def test_all_null_column_excluded(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
        result = bulk_max(df, ["a", "b"])
        assert "b" in result
