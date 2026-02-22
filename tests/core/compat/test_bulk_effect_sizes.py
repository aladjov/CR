from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import BulkEffectSizeResult, bulk_effect_sizes


class TestBulkEffectSizes:

    def test_returns_bulk_effect_size_result(self):
        df = pd.DataFrame({
            "a": [1.0, 2, 3, 4, 5, 6],
            "target": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_effect_sizes(df, ["a"], "target")
        assert isinstance(result, BulkEffectSizeResult)
        assert isinstance(result.effect_sizes, dict)
        assert isinstance(result.class_stats, dict)

    def test_matches_manual_cohens_d(self):
        rng = np.random.default_rng(42)
        n = 200
        target = np.array([0] * 100 + [1] * 100)
        x = np.where(target == 0, rng.normal(5, 2, n), rng.normal(8, 2, n))
        df = pd.DataFrame({"x": x, "target": target})

        result = bulk_effect_sizes(df, ["x"], "target")

        g0 = df[df["target"] == 0]["x"]
        g1 = df[df["target"] == 1]["x"]
        pooled = np.sqrt(
            ((len(g0) - 1) * g0.std() ** 2 + (len(g1) - 1) * g1.std() ** 2)
            / (len(g0) + len(g1) - 2)
        )
        expected_d = float((g1.mean() - g0.mean()) / pooled)
        assert result.effect_sizes["x"] == pytest.approx(expected_d, abs=1e-10)

    def test_multiple_columns(self):
        rng = np.random.default_rng(7)
        n = 200
        target = np.array([0] * 100 + [1] * 100)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
            "target": target,
        })
        result = bulk_effect_sizes(df, ["a", "b", "c"], "target")
        assert len(result.effect_sizes) == 3
        assert len(result.class_stats) == 3
        for col in ["a", "b", "c"]:
            assert col in result.effect_sizes
            assert col in result.class_stats

    def test_zero_variance_returns_zero(self):
        df = pd.DataFrame({
            "const": [5.0, 5, 5, 5, 5, 5],
            "target": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_effect_sizes(df, ["const"], "target")
        assert result.effect_sizes["const"] == 0.0

    def test_empty_group_returns_zero(self):
        df = pd.DataFrame({
            "a": [1.0, 2, 3],
            "target": [1, 1, 1],
        })
        result = bulk_effect_sizes(df, ["a"], "target")
        assert result.effect_sizes["a"] == 0.0

    def test_single_row_per_group_returns_zero(self):
        df = pd.DataFrame({
            "a": [1.0, 5.0],
            "target": [0, 1],
        })
        result = bulk_effect_sizes(df, ["a"], "target")
        assert result.effect_sizes["a"] == 0.0

    def test_all_nan_column_returns_zero(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, np.nan, np.nan],
            "target": [0, 0, 1, 1],
        })
        result = bulk_effect_sizes(df, ["a"], "target")
        assert result.effect_sizes["a"] == 0.0

    def test_class_stats_contain_expected_keys(self):
        df = pd.DataFrame({
            "a": [1.0, 2, 3, 4, 5, 6],
            "target": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_effect_sizes(df, ["a"], "target")
        stats = result.class_stats["a"]
        for key in ("mean_0", "std_0", "count_0", "mean_1", "std_1", "count_1", "median_0", "median_1"):
            assert key in stats

    def test_class_stats_values_correct(self):
        df = pd.DataFrame({
            "a": [10.0, 20, 30, 40, 50, 60],
            "target": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_effect_sizes(df, ["a"], "target")
        stats = result.class_stats["a"]
        assert stats["mean_0"] == pytest.approx(20.0)
        assert stats["mean_1"] == pytest.approx(50.0)
        assert stats["count_0"] == 3
        assert stats["count_1"] == 3
        assert stats["median_0"] == pytest.approx(20.0)
        assert stats["median_1"] == pytest.approx(50.0)

    def test_skips_missing_columns(self):
        df = pd.DataFrame({
            "a": [1.0, 2, 3],
            "target": [0, 0, 1],
        })
        result = bulk_effect_sizes(df, ["a", "nonexistent"], "target")
        assert "a" in result.effect_sizes
        assert "nonexistent" not in result.effect_sizes

    def test_empty_columns_returns_empty(self):
        df = pd.DataFrame({"target": [0, 1, 0]})
        result = bulk_effect_sizes(df, [], "target")
        assert result.effect_sizes == {}
        assert result.class_stats == {}

    def test_target_not_in_df_returns_empty(self):
        df = pd.DataFrame({"a": [1.0, 2, 3]})
        result = bulk_effect_sizes(df, ["a"], "missing_target")
        assert result.effect_sizes == {}
        assert result.class_stats == {}

    def test_sign_of_cohens_d(self):
        df = pd.DataFrame({
            "higher_in_retained": [1.0, 2, 3, 10, 11, 12],
            "lower_in_retained": [10.0, 11, 12, 1, 2, 3],
            "target": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_effect_sizes(df, ["higher_in_retained", "lower_in_retained"], "target")
        assert result.effect_sizes["higher_in_retained"] > 0
        assert result.effect_sizes["lower_in_retained"] < 0

    def test_matches_recommender_per_column_calculation(self):
        rng = np.random.default_rng(99)
        n = 500
        target = rng.integers(0, 2, size=n)
        df = pd.DataFrame({
            f"f{i}": rng.normal(i * 0.5, 1 + i * 0.1, n) for i in range(10)
        })
        df["target"] = target

        result = bulk_effect_sizes(df, [f"f{i}" for i in range(10)], "target")

        for col in [f"f{i}" for i in range(10)]:
            g0 = df[df["target"] == 0][col].dropna()
            g1 = df[df["target"] == 1][col].dropna()
            pooled = np.sqrt(
                ((len(g0) - 1) * g0.std() ** 2 + (len(g1) - 1) * g1.std() ** 2)
                / (len(g0) + len(g1) - 2)
            )
            expected = float((g1.mean() - g0.mean()) / pooled) if pooled > 0 else 0.0
            assert result.effect_sizes[col] == pytest.approx(expected, abs=1e-10)

    def test_with_nan_values_matches_dropna(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3, 4, np.nan, 6],
            "target": [0, 0, 0, 1, 1, 1],
        })
        result = bulk_effect_sizes(df, ["a"], "target")
        g0 = df[df["target"] == 0]["a"].dropna()
        g1 = df[df["target"] == 1]["a"].dropna()
        pooled = np.sqrt(
            ((len(g0) - 1) * g0.std() ** 2 + (len(g1) - 1) * g1.std() ** 2)
            / (len(g0) + len(g1) - 2)
        )
        expected = float((g1.mean() - g0.mean()) / pooled) if pooled > 0 else 0.0
        assert result.effect_sizes["a"] == pytest.approx(expected, abs=1e-10)
