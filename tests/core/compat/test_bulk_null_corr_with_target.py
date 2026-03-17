from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import bulk_null_corr_with_target


class TestBulkNullCorrWithTarget:

    def test_returns_dict(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, 1.0, 2.0],
            "t": [0, 0, 1, 1],
        })
        result = bulk_null_corr_with_target(df, ["a"], "t")
        assert isinstance(result, dict)

    def test_perfect_null_correlation_detected(self):
        df = pd.DataFrame({
            "leaker": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
            "t": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_null_corr_with_target(df, ["leaker"], "t")
        assert abs(result["leaker"]) >= 0.8

    def test_no_nulls_returns_nan(self):
        df = pd.DataFrame({
            "complete": [1.0, 2, 3, 4, 5, 6],
            "t": [0, 1, 0, 1, 0, 1],
        })
        result = bulk_null_corr_with_target(df, ["complete"], "t")
        assert math.isnan(result["complete"])

    def test_all_null_returns_nan(self):
        df = pd.DataFrame({
            "empty": [np.nan, np.nan, np.nan, np.nan],
            "t": [0, 1, 0, 1],
        })
        result = bulk_null_corr_with_target(df, ["empty"], "t")
        assert math.isnan(result["empty"])

    def test_multiple_columns(self):
        df = pd.DataFrame({
            "null_leaker": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
            "clean": [10, 20, 30, 10, 20, 30],
            "t": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_null_corr_with_target(df, ["null_leaker", "clean"], "t")
        assert abs(result["null_leaker"]) >= 0.8
        assert math.isnan(result["clean"])

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3], "t": [0, 1, 0]})
        result = bulk_null_corr_with_target(df, ["a", "nonexistent"], "t")
        assert "a" in result
        assert "nonexistent" not in result

    def test_empty_columns_returns_empty(self):
        df = pd.DataFrame({"t": [0, 1, 0]})
        assert bulk_null_corr_with_target(df, [], "t") == {}

    def test_target_not_in_df_returns_empty(self):
        df = pd.DataFrame({"a": [np.nan, 1.0]})
        assert bulk_null_corr_with_target(df, ["a"], "missing") == {}

    def test_inverse_null_pattern(self):
        df = pd.DataFrame({
            "inv": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
            "t": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_null_corr_with_target(df, ["inv"], "t")
        assert result["inv"] >= 0.8

    def test_matches_manual_null_indicator_corr(self):
        rng = np.random.default_rng(42)
        n = 100
        target = rng.integers(0, 2, size=n).astype(float)
        feat = rng.standard_normal(n)
        feat[target == 1] = np.where(rng.random(int(target.sum())) < 0.7, np.nan, feat[target == 1])
        df = pd.DataFrame({"feat": feat, "t": target})
        result = bulk_null_corr_with_target(df, ["feat"], "t")
        expected = df["feat"].isna().astype(float).corr(df["t"])
        assert result["feat"] == pytest.approx(expected, abs=1e-10)

    def test_many_columns_single_call(self):
        rng = np.random.default_rng(7)
        n = 100
        data = {f"col_{i}": rng.standard_normal(n) for i in range(20)}
        for i in range(20):
            nulls = rng.choice(n, size=rng.integers(5, 30), replace=False)
            data[f"col_{i}"][nulls] = np.nan
        data["t"] = rng.integers(0, 2, size=n).astype(float)
        df = pd.DataFrame(data)
        cols = [f"col_{i}" for i in range(20)]
        result = bulk_null_corr_with_target(df, cols, "t")
        assert len(result) == 20
        for col in cols:
            expected = df[col].isna().astype(float).corr(df["t"])
            assert result[col] == pytest.approx(expected, abs=1e-10)

    def test_excludes_target_from_result(self):
        df = pd.DataFrame({
            "a": [np.nan, 1.0, np.nan, 1.0],
            "t": [0, 1, 0, 1],
        })
        result = bulk_null_corr_with_target(df, ["a", "t"], "t")
        assert "t" not in result
        assert "a" in result
