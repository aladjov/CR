import numpy as np
import pandas as pd
import pytest

from customer_retention.transforms.artifact_store import ArtifactStore
from customer_retention.transforms.fitted import (
    FittedEncoder,
    FittedPowerTransform,
    FittedScaler,
)


@pytest.fixture
def artifact_store(tmp_path):
    return ArtifactStore(str(tmp_path / "artifacts"))


@pytest.fixture
def numeric_df():
    np.random.seed(42)
    return pd.DataFrame({"val": np.random.randn(100) * 10 + 50})


@pytest.fixture
def categorical_df():
    return pd.DataFrame({"cat": ["a", "b", "c", "a", "b", "c", "a"]})


class TestFittedScaler:
    def test_standard_fit_transform(self, numeric_df, artifact_store):
        scaler = FittedScaler("standard")
        result = scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        assert result["val"].mean() == pytest.approx(0, abs=0.1)
        assert result["val"].std() == pytest.approx(1, abs=0.1)
        assert artifact_store.has("val_scaler")

    def test_standard_matches_sklearn(self, numeric_df, artifact_store):
        from sklearn.preprocessing import StandardScaler as SkStandard
        scaler = FittedScaler("standard")
        result = scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        expected = SkStandard().fit_transform(numeric_df["val"].to_numpy().reshape(-1, 1)).ravel()
        np.testing.assert_array_almost_equal(result["val"].to_numpy(), expected)

    def test_minmax_matches_sklearn(self, numeric_df, artifact_store):
        from sklearn.preprocessing import MinMaxScaler as SkMinMax
        scaler = FittedScaler("minmax")
        result = scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        expected = SkMinMax().fit_transform(numeric_df["val"].to_numpy().reshape(-1, 1)).ravel()
        np.testing.assert_array_almost_equal(result["val"].to_numpy(), expected)

    def test_standard_transform(self, numeric_df, artifact_store):
        scaler = FittedScaler("standard")
        scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        scaler2 = FittedScaler("standard")
        result = scaler2.transform(numeric_df.copy(), "val", artifact_store)
        assert result["val"].mean() == pytest.approx(0, abs=0.1)

    def test_minmax_fit_transform(self, numeric_df, artifact_store):
        scaler = FittedScaler("minmax")
        result = scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        assert result["val"].min() >= -0.01
        assert result["val"].max() <= 1.01

    def test_constant_column_standard(self, artifact_store):
        df = pd.DataFrame({"val": [5.0] * 20})
        scaler = FittedScaler("standard")
        result = scaler.fit_transform(df.copy(), "val", artifact_store)
        assert (result["val"] == 0).all()

    def test_constant_column_minmax(self, artifact_store):
        df = pd.DataFrame({"val": [5.0] * 20})
        scaler = FittedScaler("minmax")
        result = scaler.fit_transform(df.copy(), "val", artifact_store)
        assert (result["val"] == 0).all()

    def test_missing_column_noop(self, numeric_df, artifact_store):
        scaler = FittedScaler("standard")
        result = scaler.fit_transform(numeric_df.copy(), "nonexistent", artifact_store)
        pd.testing.assert_frame_equal(result, numeric_df)

    def test_transform_missing_column_noop(self, numeric_df, artifact_store):
        scaler = FittedScaler("standard")
        scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        result = scaler.transform(numeric_df.copy(), "nonexistent", artifact_store)
        pd.testing.assert_frame_equal(result, numeric_df)


class TestFittedEncoder:
    def test_fit_transform(self, categorical_df, artifact_store):
        encoder = FittedEncoder()
        result = encoder.fit_transform(categorical_df.copy(), "cat", artifact_store)
        assert result["cat"].dtype in (np.int64, np.intp, int)
        assert set(result["cat"].unique()) == {0, 1, 2}
        assert artifact_store.has("cat_encoder")

    def test_transform_known_classes(self, categorical_df, artifact_store):
        encoder = FittedEncoder()
        encoder.fit_transform(categorical_df.copy(), "cat", artifact_store)
        encoder2 = FittedEncoder()
        result = encoder2.transform(categorical_df.copy(), "cat", artifact_store)
        assert set(result["cat"].unique()) == {0, 1, 2}

    def test_transform_unknown_class_fallback(self, categorical_df, artifact_store):
        encoder = FittedEncoder()
        encoder.fit_transform(categorical_df.copy(), "cat", artifact_store)
        df_new = pd.DataFrame({"cat": ["a", "unknown"]})
        encoder2 = FittedEncoder()
        result = encoder2.transform(df_new, "cat", artifact_store)
        assert result["cat"].iloc[1] == 0

    def test_missing_column_noop(self, categorical_df, artifact_store):
        encoder = FittedEncoder()
        result = encoder.fit_transform(categorical_df.copy(), "nonexistent", artifact_store)
        pd.testing.assert_frame_equal(result, categorical_df)

    def test_transform_missing_column_noop(self, categorical_df, artifact_store):
        encoder = FittedEncoder()
        encoder.fit_transform(categorical_df.copy(), "cat", artifact_store)
        result = encoder.transform(categorical_df.copy(), "nonexistent", artifact_store)
        pd.testing.assert_frame_equal(result, categorical_df)


class TestFittedPowerTransform:
    def test_fit_transform(self, numeric_df, artifact_store):
        pt = FittedPowerTransform()
        result = pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        assert abs(result["val"].skew()) < abs(numeric_df["val"].skew()) + 1
        assert artifact_store.has("val_power_transformer")

    def test_matches_sklearn(self, numeric_df, artifact_store):
        from sklearn.preprocessing import PowerTransformer as SkPower
        pt = FittedPowerTransform()
        result = pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        expected = SkPower(method="yeo-johnson").fit_transform(
            numeric_df["val"].to_numpy().reshape(-1, 1)
        ).ravel()
        np.testing.assert_array_almost_equal(result["val"].to_numpy(), expected, decimal=10)

    def test_with_negative_values(self, artifact_store):
        np.random.seed(99)
        df = pd.DataFrame({"val": np.random.randn(200) * 5})
        pt = FittedPowerTransform()
        result = pt.fit_transform(df.copy(), "val", artifact_store)
        assert not result["val"].isna().any()
        from sklearn.preprocessing import PowerTransformer as SkPower
        expected = SkPower(method="yeo-johnson").fit_transform(df["val"].to_numpy().reshape(-1, 1)).ravel()
        np.testing.assert_array_almost_equal(result["val"].to_numpy(), expected, decimal=10)

    def test_transform(self, numeric_df, artifact_store):
        pt = FittedPowerTransform()
        pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        pt2 = FittedPowerTransform()
        result = pt2.transform(numeric_df.copy(), "val", artifact_store)
        assert not result["val"].isna().any()

    def test_transform_matches_fit_transform(self, numeric_df, artifact_store):
        pt = FittedPowerTransform()
        fit_result = pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        pt2 = FittedPowerTransform()
        transform_result = pt2.transform(numeric_df.copy(), "val", artifact_store)
        np.testing.assert_array_almost_equal(
            fit_result["val"].to_numpy(), transform_result["val"].to_numpy()
        )

    def test_missing_column_noop(self, numeric_df, artifact_store):
        pt = FittedPowerTransform()
        result = pt.fit_transform(numeric_df.copy(), "nonexistent", artifact_store)
        pd.testing.assert_frame_equal(result, numeric_df)

    def test_transform_missing_column_noop(self, numeric_df, artifact_store):
        pt = FittedPowerTransform()
        pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        result = pt.transform(numeric_df.copy(), "nonexistent", artifact_store)
        pd.testing.assert_frame_equal(result, numeric_df)

    def test_sample_fit_reasonable(self, artifact_store):
        np.random.seed(42)
        large_df = pd.DataFrame({"val": np.random.exponential(5, 10_000)})
        pt_full = FittedPowerTransform()
        full_result = pt_full.fit_transform(large_df.copy(), "val", artifact_store)
        full_lambda = pt_full._pt.lambdas_[0]
        store2 = ArtifactStore(str(artifact_store._dir.parent / "artifacts2"))
        pt_sample = FittedPowerTransform()
        sample = large_df.sample(n=500, random_state=42)
        pt_sample._pt.fit(sample["val"].to_numpy().reshape(-1, 1))
        assert abs(pt_sample._pt.lambdas_[0] - full_lambda) < 0.2


class TestFittedScalerSparkPath:
    """Cover _fit_from_stats (Spark path) for StandardScaler and MinMaxScaler."""

    @pytest.fixture
    def _force_spark(self, monkeypatch):
        monkeypatch.setattr(
            "customer_retention.transforms.fitted._is_spark_pandas",
            lambda _: True,
        )

    def test_standard_fit_from_stats(self, numeric_df, artifact_store, _force_spark):
        scaler = FittedScaler("standard")
        result = scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        assert result["val"].mean() == pytest.approx(0, abs=0.15)
        assert result["val"].std() == pytest.approx(1, abs=0.15)
        assert scaler._scaler.n_features_in_ == 1

    def test_minmax_fit_from_stats(self, numeric_df, artifact_store, _force_spark):
        scaler = FittedScaler("minmax")
        result = scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        assert result["val"].min() == pytest.approx(0, abs=0.01)
        assert result["val"].max() == pytest.approx(1, abs=0.01)

    def test_minmax_fit_from_stats_zero_range(self, artifact_store, _force_spark):
        df = pd.DataFrame({"val": [5.0] * 20})
        scaler = FittedScaler("minmax")
        result = scaler.fit_transform(df.copy(), "val", artifact_store)
        # zero range: scale=1, min_=-col_min → result = 5*1 + (-5) = 0
        assert (result["val"] == 0).all()
        assert scaler._scaler.scale_[0] == 1.0
        assert scaler._scaler.data_range_[0] == 0.0


class TestFittedPowerTransformSparkPath:
    """Cover _fit_from_sample (Spark path) for PowerTransformer."""

    def test_fit_from_sample(self, numeric_df, artifact_store, monkeypatch):
        monkeypatch.setattr(
            "customer_retention.transforms.fitted._is_spark_pandas",
            lambda _: True,
        )
        # _fit_from_sample does a local import from customer_retention.core.compat;
        # patch at the source so the local import picks up stubs.
        monkeypatch.setattr(
            "customer_retention.core.compat.collect_for_sklearn",
            lambda s: s,
        )
        monkeypatch.setattr(
            "customer_retention.core.compat.safe_sample",
            lambda df, n: df,
        )

        pt = FittedPowerTransform()
        result = pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        assert not result["val"].isna().any()
        assert hasattr(pt._pt, "lambdas_")


class TestApplyYjEdgeCases:
    """Cover _apply_yj edge cases: lambda~0, lambda~2, standardize."""

    @pytest.fixture
    def _fitted_pt(self):
        """Return a FittedPowerTransform with a manually set lambda."""
        pt = FittedPowerTransform()
        # Fit on dummy data so internal scaler is populated (standardize=True)
        pt._pt.fit(np.array([1, 2, 3, 4, 5]).reshape(-1, 1))
        return pt

    def test_lambda_zero_log1p(self, _fitted_pt):
        _fitted_pt._pt.standardize = False
        _fitted_pt._pt.lambdas_ = np.array([0.0])
        series = pd.Series([0.0, 1.0, 2.0, 10.0])
        result = _fitted_pt._apply_yj(series)
        expected = np.log1p(series)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected.to_numpy())

    def test_lambda_two_neg_log1p(self, _fitted_pt):
        _fitted_pt._pt.standardize = False
        _fitted_pt._pt.lambdas_ = np.array([2.0])
        series = pd.Series([-1.0, -2.0, -5.0])
        result = _fitted_pt._apply_yj(series)
        expected = -np.log1p(-series)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected.to_numpy()
        )

    def test_standardize_applied(self, _fitted_pt):
        _fitted_pt._pt.standardize = True
        _fitted_pt._pt.lambdas_ = np.array([1.5])
        # Set up known scaler params
        _fitted_pt._pt._scaler.mean_ = np.array([2.0])
        _fitted_pt._pt._scaler.scale_ = np.array([0.5])
        series = pd.Series([1.0, 2.0, 3.0])
        result = _fitted_pt._apply_yj(series)
        # Manually compute: pos_vals = ((x+1)^1.5 - 1)/1.5, then standardize
        raw = ((series + 1) ** 1.5 - 1) / 1.5
        expected = (raw - 2.0) / 0.5
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected.to_numpy()
        )
