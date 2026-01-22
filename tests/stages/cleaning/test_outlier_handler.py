import pytest
import pandas as pd
import numpy as np
from customer_retention.stages.cleaning import (
    OutlierHandler, OutlierDetectionMethod, OutlierTreatmentStrategy, OutlierResult
)


class TestOutlierDetection:
    @pytest.fixture
    def series_with_outliers(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, -50]
        return pd.Series(data, dtype=float)

    @pytest.fixture
    def normal_series(self):
        np.random.seed(42)
        return pd.Series(np.random.normal(0, 1, 100))

    def test_iqr_detection(self, series_with_outliers):
        handler = OutlierHandler(detection_method=OutlierDetectionMethod.IQR)
        result = handler.detect(series_with_outliers)

        assert result.outliers_detected > 0
        assert 100 in series_with_outliers[result.outlier_mask].values
        assert -50 in series_with_outliers[result.outlier_mask].values

    def test_zscore_detection(self, series_with_outliers):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.ZSCORE, zscore_threshold=2.0
        )
        result = handler.detect(series_with_outliers)

        assert result.outliers_detected > 0

    def test_percentile_detection(self, series_with_outliers):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.PERCENTILE,
            percentile_lower=5, percentile_upper=95
        )
        result = handler.detect(series_with_outliers)

        assert result.lower_bound is not None
        assert result.upper_bound is not None

    def test_iqr_bounds_calculation(self, series_with_outliers):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR, iqr_multiplier=1.5
        )
        result = handler.detect(series_with_outliers)

        q1 = series_with_outliers.quantile(0.25)
        q3 = series_with_outliers.quantile(0.75)
        iqr = q3 - q1
        expected_lower = q1 - 1.5 * iqr
        expected_upper = q3 + 1.5 * iqr

        assert result.lower_bound == pytest.approx(expected_lower)
        assert result.upper_bound == pytest.approx(expected_upper)

    def test_modified_zscore_detection(self, series_with_outliers):
        handler = OutlierHandler(detection_method=OutlierDetectionMethod.MODIFIED_ZSCORE)
        result = handler.detect(series_with_outliers)

        assert result.outliers_detected > 0
        assert result.lower_bound is not None
        assert result.upper_bound is not None


class TestOutlierTreatment:
    @pytest.fixture
    def series_with_outliers(self):
        return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -50.0])

    def test_cap_iqr_treatment(self, series_with_outliers):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.CAP_IQR
        )
        result = handler.fit_transform(series_with_outliers)

        assert result.series.max() <= result.upper_bound
        assert result.series.min() >= result.lower_bound
        assert result.outliers_treated > 0

    def test_cap_percentile_treatment(self, series_with_outliers):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.PERCENTILE,
            treatment_strategy=OutlierTreatmentStrategy.CAP_PERCENTILE,
            percentile_lower=10, percentile_upper=90
        )
        result = handler.fit_transform(series_with_outliers)

        assert result.series.max() <= result.upper_bound
        assert result.series.min() >= result.lower_bound

    def test_clip_treatment(self, series_with_outliers):
        handler = OutlierHandler(
            treatment_strategy=OutlierTreatmentStrategy.CLIP,
            clip_min=0.0, clip_max=10.0
        )
        result = handler.fit_transform(series_with_outliers)

        assert result.series.max() <= 10.0
        assert result.series.min() >= 0.0

    def test_none_treatment_keeps_outliers(self, series_with_outliers):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.NONE
        )
        result = handler.fit_transform(series_with_outliers)

        pd.testing.assert_series_equal(result.series, series_with_outliers)
        assert result.outliers_treated == 0

    def test_log_transform_treatment(self):
        series = pd.Series([1.0, 2.0, 10.0, 100.0, 1000.0])
        handler = OutlierHandler(treatment_strategy=OutlierTreatmentStrategy.LOG_TRANSFORM)
        result = handler.fit_transform(series)

        assert result.series.max() < series.max()
        assert (result.series == np.log1p(series)).all()

    def test_sqrt_transform_treatment(self):
        series = pd.Series([1.0, 4.0, 9.0, 16.0, 100.0])
        handler = OutlierHandler(treatment_strategy=OutlierTreatmentStrategy.SQRT_TRANSFORM)
        result = handler.fit_transform(series)

        assert result.series[0] == pytest.approx(1.0)
        assert result.series[1] == pytest.approx(2.0)
        assert result.series[4] == pytest.approx(10.0)

    def test_drop_treatment_returns_mask(self, series_with_outliers):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.DROP
        )
        result = handler.fit_transform(series_with_outliers)

        assert result.rows_dropped > 0
        assert result.drop_mask is not None

    def test_winsorize_treatment_same_as_cap_iqr(self, series_with_outliers):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.WINSORIZE
        )
        result = handler.fit_transform(series_with_outliers)

        assert result.series.max() <= result.upper_bound
        assert result.series.min() >= result.lower_bound
        assert result.outliers_treated > 0


class TestOutlierIndicator:
    def test_creates_indicator_when_configured(self):
        series = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0])
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.INDICATOR
        )
        result = handler.fit_transform(series)

        assert result.indicator_column is not None
        assert result.indicator_column[3] == 1
        pd.testing.assert_series_equal(result.series, series)


class TestFitTransformSeparation:
    def test_fit_stores_bounds(self):
        train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        handler = OutlierHandler(detection_method=OutlierDetectionMethod.IQR)
        handler.fit(train)

        assert handler._lower_bound is not None
        assert handler._upper_bound is not None

    def test_transform_uses_fitted_bounds(self):
        train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        test = pd.Series([0.0, 3.0, 100.0])

        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.CAP_IQR
        )
        handler.fit(train)
        result = handler.transform(test)

        assert result.series.max() <= handler._upper_bound
        assert result.series.min() >= handler._lower_bound

    def test_transform_without_fit_raises_error(self):
        handler = OutlierHandler()
        with pytest.raises(ValueError, match="not fitted"):
            handler.transform(pd.Series([1.0, 2.0, 3.0]))


class TestEdgeCases:
    def test_no_outliers_returns_unchanged(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.CAP_IQR
        )
        result = handler.fit_transform(series)

        assert result.outliers_detected == 0
        pd.testing.assert_series_equal(result.series, series)

    def test_handles_series_with_nulls(self):
        series = pd.Series([1.0, 2.0, None, 4.0, 100.0])
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.CAP_IQR
        )
        result = handler.fit_transform(series)

        assert pd.isna(result.series[2])
        assert result.series[4] <= result.upper_bound

    def test_log_transform_handles_zeros(self):
        series = pd.Series([0.0, 1.0, 2.0, 10.0])
        handler = OutlierHandler(treatment_strategy=OutlierTreatmentStrategy.LOG_TRANSFORM)
        result = handler.fit_transform(series)

        assert np.isfinite(result.series[0])

    def test_log_transform_errors_on_negatives(self):
        series = pd.Series([-1.0, 1.0, 2.0])
        handler = OutlierHandler(treatment_strategy=OutlierTreatmentStrategy.LOG_TRANSFORM)

        with pytest.raises(ValueError, match="negative"):
            handler.fit_transform(series)


class TestOutlierResultOutput:
    def test_result_contains_all_fields(self):
        series = pd.Series([1.0, 2.0, 100.0])
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.CAP_IQR
        )
        result = handler.fit_transform(series)

        assert hasattr(result, 'series')
        assert hasattr(result, 'method_used')
        assert hasattr(result, 'strategy_used')
        assert hasattr(result, 'outliers_detected')
        assert hasattr(result, 'outliers_treated')
        assert hasattr(result, 'lower_bound')
        assert hasattr(result, 'upper_bound')
