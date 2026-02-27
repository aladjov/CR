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

    def test_all_nulls_returns_nan_matrix(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [np.nan, np.nan, np.nan]})
        result = batched_corr_matrix(df, ["a", "b"])
        assert list(result.columns) == ["a", "b"]
        assert list(result.index) == ["a", "b"]
        assert result.isna().all().all()

    def test_partial_nulls_still_computes(self):
        df = pd.DataFrame({"a": [1.0, 2, np.nan, 4], "b": [4.0, np.nan, 2, 1]})
        result = batched_corr_matrix(df, ["a", "b"])
        assert not result.isna().all().all()

    def test_scattered_nulls_pairwise_correlation(self):
        """When nulls are scattered across many columns, pairwise correlation
        must still produce correct values — each pair uses rows where BOTH
        columns are non-null, not only rows where ALL columns are non-null."""
        rng = np.random.default_rng(42)
        n = 100
        target = rng.integers(0, 2, size=n).astype(float)
        feature_a = target * 3 + rng.standard_normal(n) * 0.5
        feature_b = -target * 2 + rng.standard_normal(n) * 0.5
        noise = rng.standard_normal(n)
        df = pd.DataFrame({
            "target": target,
            "feat_a": feature_a,
            "feat_b": feature_b,
            "noise": noise,
        })
        df.loc[0:20, "feat_a"] = np.nan
        df.loc[30:50, "feat_b"] = np.nan
        df.loc[60:70, "noise"] = np.nan

        result = batched_corr_matrix(df, ["feat_a", "feat_b", "noise", "target"])
        expected = df[["feat_a", "feat_b", "noise", "target"]].corr()
        pd.testing.assert_frame_equal(result, expected, atol=1e-10)

    def test_zero_variance_column_produces_nan(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "const": [7.0, 7, 7, 7, 7], "b": [5.0, 4, 3, 2, 1]})
        result = batched_corr_matrix(df, ["a", "const", "b"])
        expected = df[["a", "const", "b"]].corr()
        pd.testing.assert_frame_equal(result, expected)

    def test_pairwise_zero_variance_in_overlap(self):
        """Columns have global variance but zero variance in their non-null
        overlap — must return NaN, not raise DIVIDE_BY_ZERO."""
        df = pd.DataFrame({
            "a": [1.0, 1.0, 2.0, 3.0],
            "b": [5.0, 6.0, np.nan, np.nan],
        })
        result = batched_corr_matrix(df, ["a", "b"])
        expected = df[["a", "b"]].corr()
        pd.testing.assert_frame_equal(result, expected)

    def test_all_zero_variance_columns(self):
        df = pd.DataFrame({"x": [3.0] * 10, "y": [5.0] * 10})
        result = batched_corr_matrix(df, ["x", "y"])
        assert result.isna().all().all()

    def test_zero_variance_mixed_with_nulls(self):
        df = pd.DataFrame({
            "a": [1.0, 2, 3, np.nan, 5],
            "const": [4.0, 4, 4, 4, 4],
            "b": [np.nan, 2, 3, 4, 5],
        })
        result = batched_corr_matrix(df, ["a", "const", "b"])
        expected = df[["a", "const", "b"]].corr()
        pd.testing.assert_frame_equal(result, expected)

    def test_target_correlation_with_scattered_nulls(self):
        """Feature-target correlations must be correct even when different
        features have nulls in different rows (the VectorAssembler bug)."""
        rng = np.random.default_rng(99)
        n = 200
        target = rng.integers(0, 2, size=n).astype(float)
        strong_feature = target * 5 + rng.standard_normal(n)
        df = pd.DataFrame({
            "target": target,
            "strong": strong_feature,
            **{f"col_{i}": rng.standard_normal(n) for i in range(10)},
        })
        for i in range(10):
            nulls = rng.choice(n, size=20, replace=False)
            df.loc[nulls, f"col_{i}"] = np.nan

        result = batched_corr_matrix(
            df, ["strong"] + [f"col_{i}" for i in range(10)] + ["target"],
        )
        assert result.loc["strong", "target"] > 0.8

    def test_string_columns_return_nan(self):
        df = pd.DataFrame({
            "label": ["a", "b", "c", "d", "e"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "score": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        result = batched_corr_matrix(df, ["label", "value", "score"])
        assert result.loc["value", "score"] == pytest.approx(-1.0)
        assert np.isnan(result.loc["label", "value"])
        assert np.isnan(result.loc["label", "score"])
