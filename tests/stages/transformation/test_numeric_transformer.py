import pytest
import pandas as pd
import numpy as np
from customer_retention.stages.transformation import (
    NumericTransformer, ScalingStrategy, PowerTransform, NumericTransformResult
)


class TestScalingStrategies:
    @pytest.fixture
    def numeric_series(self):
        return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

    def test_standard_scaling_mean_zero(self, numeric_series):
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(numeric_series)

        assert result.series.mean() == pytest.approx(0.0, abs=1e-10)

    def test_standard_scaling_std_one(self, numeric_series):
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(numeric_series)

        assert result.series.std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_minmax_scaling_range(self, numeric_series):
        transformer = NumericTransformer(scaling=ScalingStrategy.MINMAX)
        result = transformer.fit_transform(numeric_series)

        assert result.series.min() == pytest.approx(0.0)
        assert result.series.max() == pytest.approx(1.0)

    def test_robust_scaling(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.ROBUST)
        result = transformer.fit_transform(series)

        assert result.series.median() == pytest.approx(0.0, abs=0.1)

    def test_maxabs_scaling(self, numeric_series):
        transformer = NumericTransformer(scaling=ScalingStrategy.MAXABS)
        result = transformer.fit_transform(numeric_series)

        assert result.series.abs().max() == pytest.approx(1.0)

    def test_none_scaling(self, numeric_series):
        transformer = NumericTransformer(scaling=ScalingStrategy.NONE)
        result = transformer.fit_transform(numeric_series)

        pd.testing.assert_series_equal(result.series, numeric_series)


class TestPowerTransformations:
    @pytest.fixture
    def skewed_series(self):
        return pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])

    def test_log_transform(self, skewed_series):
        transformer = NumericTransformer(power_transform=PowerTransform.LOG)
        result = transformer.fit_transform(skewed_series)

        expected = np.log(skewed_series)
        pd.testing.assert_series_equal(result.series, expected)

    def test_log1p_transform(self):
        series = pd.Series([0.0, 1.0, 2.0, 10.0])
        transformer = NumericTransformer(power_transform=PowerTransform.LOG1P)
        result = transformer.fit_transform(series)

        expected = np.log1p(series)
        pd.testing.assert_series_equal(result.series, expected)

    def test_sqrt_transform(self):
        series = pd.Series([1.0, 4.0, 9.0, 16.0])
        transformer = NumericTransformer(power_transform=PowerTransform.SQRT)
        result = transformer.fit_transform(series)

        expected = np.sqrt(series)
        pd.testing.assert_series_equal(result.series, expected)

    def test_log_transform_errors_on_zero(self):
        series = pd.Series([0.0, 1.0, 2.0])
        transformer = NumericTransformer(power_transform=PowerTransform.LOG)

        with pytest.raises(ValueError, match="positive"):
            transformer.fit_transform(series)

    def test_log_transform_errors_on_negative(self):
        series = pd.Series([-1.0, 1.0, 2.0])
        transformer = NumericTransformer(power_transform=PowerTransform.LOG)

        with pytest.raises(ValueError, match="positive"):
            transformer.fit_transform(series)


class TestCombinedTransformations:
    def test_power_then_scale(self):
        series = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
        transformer = NumericTransformer(
            power_transform=PowerTransform.LOG1P,
            scaling=ScalingStrategy.STANDARD
        )
        result = transformer.fit_transform(series)

        assert result.series.mean() == pytest.approx(0.0, abs=1e-10)
        assert result.transformations_applied == [PowerTransform.LOG1P, ScalingStrategy.STANDARD]


class TestFitTransformSeparation:
    def test_fit_stores_parameters(self):
        train = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        transformer.fit(train)

        assert transformer._mean is not None
        assert transformer._std is not None

    def test_transform_uses_fitted_parameters(self):
        train = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        test = pd.Series([15.0, 25.0, 60.0])

        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        transformer.fit(train)
        result = transformer.transform(test)

        train_mean = train.mean()
        train_std = train.std(ddof=0)
        expected = (test - train_mean) / train_std

        pd.testing.assert_series_equal(result.series, expected)

    def test_transform_without_fit_raises_error(self):
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        with pytest.raises(ValueError, match="not fitted"):
            transformer.transform(pd.Series([1.0, 2.0, 3.0]))


class TestInverseTransform:
    def test_inverse_standard_scaling(self):
        original = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(original)

        inversed = transformer.inverse_transform(result.series)
        pd.testing.assert_series_equal(inversed, original, atol=1e-10)

    def test_inverse_minmax_scaling(self):
        original = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.MINMAX)
        result = transformer.fit_transform(original)

        inversed = transformer.inverse_transform(result.series)
        pd.testing.assert_series_equal(inversed, original, atol=1e-10)


class TestResultOutput:
    def test_result_contains_original_stats(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(series)

        assert result.original_mean == pytest.approx(30.0)
        assert result.original_std is not None
        assert result.original_min == pytest.approx(10.0)
        assert result.original_max == pytest.approx(50.0)

    def test_result_contains_transformed_stats(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(series)

        assert result.transformed_mean == pytest.approx(0.0, abs=1e-10)
        assert result.transformed_std == pytest.approx(1.0, abs=1e-10)


class TestEdgeCases:
    def test_handles_nulls(self):
        series = pd.Series([10.0, None, 30.0, None, 50.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(series)

        assert pd.isna(result.series[1])
        assert pd.isna(result.series[3])
        assert not pd.isna(result.series[0])

    def test_constant_series_standard_scaling(self):
        series = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(series)

        assert (result.series == 0.0).all()

    def test_single_value_series(self):
        series = pd.Series([42.0])
        transformer = NumericTransformer(scaling=ScalingStrategy.MINMAX)
        result = transformer.fit_transform(series)

        assert len(result.series) == 1
