import numpy as np
import pandas as pd
import pytest

from customer_retention.transforms.ops import (
    apply_cap_outlier,
    apply_cap_then_log,
    apply_derived_composite,
    apply_derived_interaction,
    apply_derived_ratio,
    apply_drop_column,
    apply_feature_select,
    apply_impute_null,
    apply_log_transform,
    apply_one_hot_encode,
    apply_segment_aware_cap,
    apply_sqrt_transform,
    apply_type_cast,
    apply_winsorize,
    apply_zero_inflation_handling,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 100.0],
        "b": [0.0, 10.0, 20.0, 30.0, 40.0],
        "cat": ["x", "y", "x", "z", "y"],
    })


class TestImputeNull:
    def test_fills_with_zero(self, sample_df):
        result = apply_impute_null(sample_df, "a", value=0)
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[2] == 0.0

    def test_fills_with_median(self, sample_df):
        result = apply_impute_null(sample_df, "a", value="median")
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[2] == 3.0

    def test_missing_column_noop(self, sample_df):
        result = apply_impute_null(sample_df, "nonexistent", value=0)
        assert list(result.columns) == list(sample_df.columns)


class TestCapOutlier:
    def test_clips_values(self, sample_df):
        result = apply_cap_outlier(sample_df, "a", lower=0, upper=50)
        assert result["a"].max() <= 50

    def test_missing_column_noop(self, sample_df):
        result = apply_cap_outlier(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestTypeCast:
    def test_casts_to_int(self, sample_df):
        df = sample_df.copy()
        df["a"] = df["a"].fillna(0)
        result = apply_type_cast(df, "a", dtype="int")
        assert result["a"].dtype == int

    def test_missing_column_noop(self, sample_df):
        result = apply_type_cast(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestDropColumn:
    def test_drops_existing(self, sample_df):
        result = apply_drop_column(sample_df, "a")
        assert "a" not in result.columns

    def test_drops_missing_is_noop(self, sample_df):
        result = apply_drop_column(sample_df, "nonexistent")
        assert list(result.columns) == list(sample_df.columns)


class TestWinsorize:
    def test_clips_values(self, sample_df):
        result = apply_winsorize(sample_df, "a", lower_bound=1, upper_bound=10)
        assert result["a"].dropna().min() >= 1
        assert result["a"].dropna().max() <= 10

    def test_missing_column_noop(self, sample_df):
        result = apply_winsorize(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestSegmentAwareCap:
    def test_caps_per_segment(self):
        np.random.seed(42)
        df = pd.DataFrame({"v": np.concatenate([
            np.random.normal(10, 2, 50),
            np.random.normal(100, 5, 50),
            [500],
        ])})
        result = apply_segment_aware_cap(df, "v", n_segments=2)
        assert result["v"].max() < 500

    def test_missing_column_noop(self, sample_df):
        result = apply_segment_aware_cap(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_too_few_samples_noop(self):
        df = pd.DataFrame({"v": [42.0]})
        result = apply_segment_aware_cap(df, "v", n_segments=2)
        pd.testing.assert_frame_equal(result, df)


class TestLogTransform:
    def test_log1p_applied(self, sample_df):
        result = apply_log_transform(sample_df.copy(), "b")
        expected = np.log1p(sample_df["b"].clip(lower=0))
        np.testing.assert_array_almost_equal(result["b"], expected)

    def test_missing_column_noop(self, sample_df):
        result = apply_log_transform(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestSqrtTransform:
    def test_sqrt_applied(self, sample_df):
        result = apply_sqrt_transform(sample_df.copy(), "b")
        expected = np.sqrt(sample_df["b"].clip(lower=0))
        np.testing.assert_array_almost_equal(result["b"], expected)

    def test_missing_column_noop(self, sample_df):
        result = apply_sqrt_transform(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestZeroInflationHandling:
    def test_creates_indicator_and_log_column(self):
        df = pd.DataFrame({"v": [0, 0, 5, 10, 0]})
        result = apply_zero_inflation_handling(df, "v")
        assert "v_is_zero" in result.columns
        assert "v_log" in result.columns
        assert result["v_is_zero"].sum() == 3
        assert result.loc[2, "v_log"] == pytest.approx(np.log1p(5))
        assert result.loc[3, "v_log"] == pytest.approx(np.log1p(10))

    def test_original_column_preserved(self):
        df = pd.DataFrame({"v": [0, 0, 5, 10, 0]})
        original = df["v"].copy()
        result = apply_zero_inflation_handling(df, "v")
        pd.testing.assert_series_equal(result["v"], original, check_names=False)

    def test_nan_values_preserved_in_log(self):
        df = pd.DataFrame({"v": [0, np.nan, 5, 10, 0]})
        original = df["v"].copy()
        result = apply_zero_inflation_handling(df, "v")
        assert pd.isna(result["v_log"].iloc[1])
        assert result["v_is_zero"].iloc[1] == 0
        assert result["v_log"].iloc[0] == 0
        assert result["v_log"].iloc[2] == pytest.approx(np.log1p(5))
        pd.testing.assert_series_equal(result["v"], original, check_names=False)

    def test_negative_values_clipped_in_log(self):
        df = pd.DataFrame({"v": [0, -3, 5, 10]})
        original = df["v"].copy()
        result = apply_zero_inflation_handling(df, "v")
        assert result["v_log"].iloc[0] == 0
        assert result["v_log"].iloc[1] == pytest.approx(np.log1p(0))
        assert result["v_log"].iloc[2] == pytest.approx(np.log1p(5))
        pd.testing.assert_series_equal(result["v"], original, check_names=False)

    def test_missing_column_noop(self, sample_df):
        result = apply_zero_inflation_handling(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestCapThenLog:
    def test_caps_and_logs(self):
        df = pd.DataFrame({"v": list(range(100)) + [10000]})
        result = apply_cap_then_log(df, "v")
        assert result["v"].max() < np.log1p(10000)

    def test_precomputed_q99(self):
        df = pd.DataFrame({"v": list(range(100)) + [10000]})
        q99 = df["v"].quantile(0.99)
        result = apply_cap_then_log(df.copy(), "v", _precomputed_q99=q99)
        expected = apply_cap_then_log(df.copy(), "v")
        np.testing.assert_array_almost_equal(result["v"].to_numpy(), expected["v"].to_numpy())

    def test_missing_column_noop(self, sample_df):
        result = apply_cap_then_log(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestOneHotEncode:
    def test_creates_dummies(self, sample_df):
        result = apply_one_hot_encode(sample_df, "cat")
        assert "cat" not in result.columns
        assert "cat_x" in result.columns
        assert "cat_y" in result.columns
        assert "cat_z" in result.columns

    def test_missing_column_noop(self, sample_df):
        result = apply_one_hot_encode(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestFeatureSelect:
    def test_drops_column(self, sample_df):
        result = apply_feature_select(sample_df, "a")
        assert "a" not in result.columns


class TestDerivedRatio:
    def test_creates_ratio(self, sample_df):
        result = apply_derived_ratio(sample_df, "ratio", numerator="a", denominator="b")
        assert "ratio" in result.columns
        assert pd.isna(result["ratio"].iloc[0])
        assert result["ratio"].iloc[1] == pytest.approx(2.0 / 10.0)

    def test_missing_columns_noop(self, sample_df):
        result = apply_derived_ratio(sample_df, "ratio", numerator="x", denominator="y")
        assert "ratio" not in result.columns

    def test_raises_when_denominator_all_zero(self):
        """100%-zero denominator → every ratio row is NaN → useless feature.
        Fail-fast is better than silently generating a column that will be
        median-imputed to a constant and dropped at variance downstream.
        """
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "denom": [0.0, 0.0, 0.0]})
        with pytest.raises(ValueError, match=r"denominator.*denom.*all-zero|100%"):
            apply_derived_ratio(df, "ratio", numerator="num", denominator="denom")

    def test_accepts_partially_zero_denominator(self):
        """Mixed-zero denominator is legitimate — NaN rows get median-imputed
        downstream, non-zero rows carry the actual ratio signal."""
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "denom": [0.0, 5.0, 0.0, 10.0]})
        result = apply_derived_ratio(df, "ratio", numerator="num", denominator="denom")
        assert "ratio" in result.columns
        assert pd.isna(result["ratio"].iloc[0])
        assert result["ratio"].iloc[1] == pytest.approx(2.0 / 5.0)

    def test_accepts_nonzero_denominator(self):
        df = pd.DataFrame({"num": [1.0, 2.0], "denom": [5.0, 10.0]})
        result = apply_derived_ratio(df, "ratio", numerator="num", denominator="denom")
        assert result["ratio"].iloc[0] == pytest.approx(0.2)
        assert result["ratio"].iloc[1] == pytest.approx(0.2)

    def test_empty_df_noop(self):
        """Empty DataFrame — cannot compute zero-rate, treat as no-op (no ratio
        column added, no raise). Downstream will detect missing columns."""
        df = pd.DataFrame({"num": [], "denom": []})
        result = apply_derived_ratio(df, "ratio", numerator="num", denominator="denom")
        # Zero-length series has no zero entries; ratio created as empty series
        assert "ratio" in result.columns
        assert len(result["ratio"]) == 0

    def test_null_denominator_not_counted_as_zero(self):
        """NaN denom rows stay NaN in ratio — they're not zero, just unknown."""
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "denom": [5.0, float("nan"), 10.0]})
        result = apply_derived_ratio(df, "ratio", numerator="num", denominator="denom")
        assert result["ratio"].iloc[0] == pytest.approx(0.2)
        assert pd.isna(result["ratio"].iloc[1])
        assert result["ratio"].iloc[2] == pytest.approx(0.3)


class TestDerivedInteraction:
    def test_creates_product(self, sample_df):
        result = apply_derived_interaction(sample_df, "prod", col_a="a", col_b="b")
        assert "prod" in result.columns
        assert result["prod"].iloc[1] == pytest.approx(2.0 * 10.0)

    def test_missing_column_noop(self, sample_df):
        result = apply_derived_interaction(sample_df, "prod", col_a="x", col_b="y")
        assert "prod" not in result.columns


class TestDerivedComposite:
    def test_creates_mean(self, sample_df):
        result = apply_derived_composite(sample_df, "comp", columns=["a", "b"])
        assert "comp" in result.columns
        assert result["comp"].iloc[1] == pytest.approx(6.0)

    def test_no_valid_columns_noop(self, sample_df):
        result = apply_derived_composite(sample_df, "comp", columns=["x", "y"])
        assert "comp" not in result.columns


# ── Batch ops ──────────────────────────────────────────────────────


class TestBatchLogTransform:
    def test_matches_sequential(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
        expected = df.copy()
        for c in ["a", "b", "c"]:
            expected = apply_log_transform(expected, c)
        from customer_retention.transforms.ops import apply_batch_log_transform
        result = apply_batch_log_transform(df.copy(), ["a", "b", "c"])
        pd.testing.assert_frame_equal(result, expected)

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        from customer_retention.transforms.ops import apply_batch_log_transform
        result = apply_batch_log_transform(df.copy(), ["a", "missing"])
        expected = apply_log_transform(df.copy(), "a")
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_columns_noop(self):
        df = pd.DataFrame({"a": [1.0]})
        from customer_retention.transforms.ops import apply_batch_log_transform
        result = apply_batch_log_transform(df.copy(), [])
        pd.testing.assert_frame_equal(result, df)

    def test_negative_values_clipped(self):
        df = pd.DataFrame({"a": [-5.0, 0.0, 3.0], "b": [-1.0, 2.0, 4.0]})
        from customer_retention.transforms.ops import apply_batch_log_transform
        result = apply_batch_log_transform(df.copy(), ["a", "b"])
        expected = df.copy()
        for c in ["a", "b"]:
            expected = apply_log_transform(expected, c)
        pd.testing.assert_frame_equal(result, expected)

    def test_preserves_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
        from customer_retention.transforms.ops import apply_batch_log_transform
        result = apply_batch_log_transform(df.copy(), ["a", "b"])
        expected = df.copy()
        for c in ["a", "b"]:
            expected = apply_log_transform(expected, c)
        pd.testing.assert_frame_equal(result, expected)


class TestBatchSqrtTransform:
    def test_matches_sequential(self):
        df = pd.DataFrame({"a": [1.0, 4.0, 9.0], "b": [16.0, 25.0, 36.0]})
        expected = df.copy()
        for c in ["a", "b"]:
            expected = apply_sqrt_transform(expected, c)
        from customer_retention.transforms.ops import apply_batch_sqrt_transform
        result = apply_batch_sqrt_transform(df.copy(), ["a", "b"])
        pd.testing.assert_frame_equal(result, expected)

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [4.0, 9.0]})
        from customer_retention.transforms.ops import apply_batch_sqrt_transform
        result = apply_batch_sqrt_transform(df.copy(), ["a", "gone"])
        expected = apply_sqrt_transform(df.copy(), "a")
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_columns_noop(self):
        df = pd.DataFrame({"a": [4.0]})
        from customer_retention.transforms.ops import apply_batch_sqrt_transform
        result = apply_batch_sqrt_transform(df.copy(), [])
        pd.testing.assert_frame_equal(result, df)


class TestBatchZeroInflation:
    def test_matches_sequential(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 5.0], "b": [3.0, 0.0, 0.0]})
        expected = df.copy()
        for c in ["a", "b"]:
            expected = apply_zero_inflation_handling(expected, c)
        from customer_retention.transforms.ops import apply_batch_zero_inflation
        result = apply_batch_zero_inflation(df.copy(), ["a", "b"])
        pd.testing.assert_frame_equal(result, expected)

    def test_creates_all_flags_and_log_columns(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "y": [2.0, 0.0], "z": [0.0, 0.0]})
        from customer_retention.transforms.ops import apply_batch_zero_inflation
        result = apply_batch_zero_inflation(df.copy(), ["x", "y", "z"])
        for c in ["x", "y", "z"]:
            assert f"{c}_is_zero" in result.columns
            assert f"{c}_log" in result.columns
        assert result["x_is_zero"].sum() == 1
        assert result["y_is_zero"].sum() == 1
        assert result["z_is_zero"].sum() == 2

    def test_originals_preserved(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "y": [2.0, 0.0]})
        original = df.copy()
        from customer_retention.transforms.ops import apply_batch_zero_inflation
        result = apply_batch_zero_inflation(df.copy(), ["x", "y"])
        pd.testing.assert_series_equal(result["x"], original["x"])
        pd.testing.assert_series_equal(result["y"], original["y"])

    def test_preserves_nan(self):
        df = pd.DataFrame({"a": [0.0, np.nan, 5.0], "b": [np.nan, 0.0, 3.0]})
        expected = df.copy()
        for c in ["a", "b"]:
            expected = apply_zero_inflation_handling(expected, c)
        from customer_retention.transforms.ops import apply_batch_zero_inflation
        result = apply_batch_zero_inflation(df.copy(), ["a", "b"])
        pd.testing.assert_frame_equal(result, expected)

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [0.0, 5.0]})
        from customer_retention.transforms.ops import apply_batch_zero_inflation
        result = apply_batch_zero_inflation(df.copy(), ["a", "nope"])
        expected = apply_zero_inflation_handling(df.copy(), "a")
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_columns_noop(self):
        df = pd.DataFrame({"a": [1.0]})
        from customer_retention.transforms.ops import apply_batch_zero_inflation
        result = apply_batch_zero_inflation(df.copy(), [])
        pd.testing.assert_frame_equal(result, df)


class TestBatchCapThenLog:
    def test_matches_sequential(self):
        df = pd.DataFrame({"a": list(range(100)) + [10000.0], "b": list(range(50)) + [5000.0] * 51})
        expected = df.copy()
        expected = apply_cap_then_log(expected, "a")
        expected = apply_cap_then_log(expected, "b")
        from customer_retention.transforms.ops import apply_batch_cap_then_log
        result = apply_batch_cap_then_log(df.copy(), [("a", None), ("b", None)])
        pd.testing.assert_frame_equal(result, expected)

    def test_precomputed_q99_used(self):
        df = pd.DataFrame({"a": list(range(100)) + [10000.0]})
        q99 = df["a"].quantile(0.99)
        from customer_retention.transforms.ops import apply_batch_cap_then_log
        result = apply_batch_cap_then_log(df.copy(), [("a", q99)])
        expected = apply_cap_then_log(df.copy(), "a")
        np.testing.assert_array_almost_equal(result["a"].to_numpy(), expected["a"].to_numpy())

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        from customer_retention.transforms.ops import apply_batch_cap_then_log
        result = apply_batch_cap_then_log(df.copy(), [("a", None), ("missing", None)])
        expected = apply_cap_then_log(df.copy(), "a")
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_noop(self):
        df = pd.DataFrame({"a": [1.0]})
        from customer_retention.transforms.ops import apply_batch_cap_then_log
        result = apply_batch_cap_then_log(df.copy(), [])
        pd.testing.assert_frame_equal(result, df)
