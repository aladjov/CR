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
    def test_creates_indicator_and_logs(self):
        df = pd.DataFrame({"v": [0, 0, 5, 10, 0]})
        result = apply_zero_inflation_handling(df, "v")
        assert "v_is_zero" in result.columns
        assert result["v_is_zero"].sum() == 3
        assert result.loc[2, "v"] == pytest.approx(np.log1p(5))

    def test_missing_column_noop(self, sample_df):
        result = apply_zero_inflation_handling(sample_df, "nonexistent")
        pd.testing.assert_frame_equal(result, sample_df)


class TestCapThenLog:
    def test_caps_and_logs(self):
        df = pd.DataFrame({"v": list(range(100)) + [10000]})
        result = apply_cap_then_log(df, "v")
        assert result["v"].max() < np.log1p(10000)

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
