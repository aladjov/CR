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

    def test_transform(self, numeric_df, artifact_store):
        pt = FittedPowerTransform()
        pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        pt2 = FittedPowerTransform()
        result = pt2.transform(numeric_df.copy(), "val", artifact_store)
        assert not result["val"].isna().any()

    def test_missing_column_noop(self, numeric_df, artifact_store):
        pt = FittedPowerTransform()
        result = pt.fit_transform(numeric_df.copy(), "nonexistent", artifact_store)
        pd.testing.assert_frame_equal(result, numeric_df)

    def test_transform_missing_column_noop(self, numeric_df, artifact_store):
        pt = FittedPowerTransform()
        pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        result = pt.transform(numeric_df.copy(), "nonexistent", artifact_store)
        pd.testing.assert_frame_equal(result, numeric_df)
