import pytest
import pandas as pd
import numpy as np
from customer_retention.stages.modeling import (
    FeatureScaler, ScalerType, ScalingResult
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X_train = pd.DataFrame({
        "feature1": np.random.randn(n) * 10 + 50,
        "feature2": np.random.randn(n) * 5 + 20,
        "feature3": np.random.randn(n) * 100,
    })
    X_test = pd.DataFrame({
        "feature1": np.random.randn(100) * 10 + 50,
        "feature2": np.random.randn(100) * 5 + 20,
        "feature3": np.random.randn(100) * 100,
    })
    return X_train, X_test


class TestScalerType:
    def test_scaler_type_enum_has_required_values(self):
        assert hasattr(ScalerType, "STANDARD")
        assert hasattr(ScalerType, "ROBUST")
        assert hasattr(ScalerType, "MINMAX")
        assert hasattr(ScalerType, "NONE")


class TestStandardScaler:
    def test_standard_scaler_normalizes_mean_std(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert np.abs(result.X_train_scaled.mean().mean()) < 0.1
        assert np.abs(result.X_train_scaled.std().mean() - 1.0) < 0.1

    def test_standard_scaler_uses_train_params_for_test(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert result.X_test_scaled is not None


class TestRobustScaler:
    def test_robust_scaler_handles_outliers(self, sample_data):
        X_train, X_test = sample_data
        X_train_with_outliers = X_train.copy()
        X_train_with_outliers.iloc[0, 0] = 10000

        scaler = FeatureScaler(scaler_type=ScalerType.ROBUST)
        result = scaler.fit_transform(X_train_with_outliers, X_test)

        assert result.X_train_scaled is not None
        assert not np.isinf(result.X_train_scaled.values).any()


class TestMinMaxScaler:
    def test_minmax_scaler_scales_to_range(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.MINMAX)
        result = scaler.fit_transform(X_train, X_test)

        assert result.X_train_scaled.min().min() >= 0
        assert result.X_train_scaled.max().max() <= 1


class TestNoScaling:
    def test_none_scaler_returns_unchanged(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.NONE)
        result = scaler.fit_transform(X_train, X_test)

        pd.testing.assert_frame_equal(result.X_train_scaled, X_train)
        pd.testing.assert_frame_equal(result.X_test_scaled, X_test)


class TestFitOnTrainOnly:
    def test_fit_on_train_only_prevents_leakage(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.STANDARD, fit_on_train_only=True)
        result = scaler.fit_transform(X_train, X_test)

        assert result.scaling_params is not None
        assert "mean" in result.scaling_params
        assert "scale" in result.scaling_params


class TestScalingResult:
    def test_result_contains_required_fields(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        assert hasattr(result, "scaler")
        assert hasattr(result, "X_train_scaled")
        assert hasattr(result, "X_test_scaled")
        assert hasattr(result, "scaling_params")


class TestScalerPersistence:
    def test_scaler_can_be_saved(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.STANDARD, save_scaler=True)
        result = scaler.fit_transform(X_train, X_test)

        assert result.scaler is not None

    def test_scaler_can_transform_new_data(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        new_data = pd.DataFrame({
            "feature1": [55, 60],
            "feature2": [22, 25],
            "feature3": [10, 20],
        })
        transformed = scaler.transform(new_data)
        assert len(transformed) == 2


class TestAC5_7_ScalingAppliedCorrectly:
    def test_ac5_7_verify_mean_std_after_scaling(self, sample_data):
        X_train, X_test = sample_data
        scaler = FeatureScaler(scaler_type=ScalerType.STANDARD)
        result = scaler.fit_transform(X_train, X_test)

        for col in result.X_train_scaled.columns:
            assert np.abs(result.X_train_scaled[col].mean()) < 0.1
            assert np.abs(result.X_train_scaled[col].std() - 1.0) < 0.1
