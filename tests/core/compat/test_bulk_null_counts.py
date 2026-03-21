from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat.bulk_profiling import bulk_null_counts


class TestBulkNullCounts:

    def test_returns_dict(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, 1.0]})
        result = bulk_null_counts(df, ["a", "b"])
        assert isinstance(result, dict)

    def test_counts_nulls_per_column(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, 1.0], "c": [1, 2, 3]})
        result = bulk_null_counts(df, ["a", "b", "c"])
        assert result == {"a": 1, "b": 2, "c": 0}

    def test_all_null_column(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        assert bulk_null_counts(df, ["x"]) == {"x": 3}

    def test_no_null_column(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        assert bulk_null_counts(df, ["x"]) == {"x": 0}

    def test_empty_columns_list(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert bulk_null_counts(df, []) == {}

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [np.nan, 1.0], "b": [1, 2]})
        result = bulk_null_counts(df, ["a", "nonexistent", "b"])
        assert "a" in result
        assert "b" in result
        assert "nonexistent" not in result

    def test_many_columns_single_call(self):
        rng = np.random.default_rng(42)
        n = 200
        data = {}
        expected = {}
        for i in range(50):
            col = rng.standard_normal(n)
            null_idx = rng.choice(n, size=rng.integers(0, n // 2), replace=False)
            col[null_idx] = np.nan
            data[f"col_{i}"] = col
            expected[f"col_{i}"] = int(np.isnan(col).sum())
        df = pd.DataFrame(data)
        result = bulk_null_counts(df, list(data.keys()))
        assert result == expected

    def test_returns_int_values(self):
        df = pd.DataFrame({"a": [np.nan, 1.0], "b": [1.0, 2.0]})
        result = bulk_null_counts(df, ["a", "b"])
        for v in result.values():
            assert isinstance(v, int)

    def test_none_columns_defaults_to_all(self):
        df = pd.DataFrame({"a": [np.nan, 1.0], "b": [1.0, np.nan]})
        result = bulk_null_counts(df)
        assert result == {"a": 1, "b": 1}

    def test_mixed_dtypes(self):
        df = pd.DataFrame({
            "num": [1.0, np.nan, 3.0],
            "txt": ["a", None, "c"],
            "dt": pd.to_datetime(["2020-01-01", pd.NaT, "2020-03-01"]),
        })
        result = bulk_null_counts(df, ["num", "txt", "dt"])
        assert result == {"num": 1, "txt": 1, "dt": 1}
