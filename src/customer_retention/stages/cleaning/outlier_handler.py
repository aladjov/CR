from enum import Enum
from typing import Optional
from dataclasses import dataclass
import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series


class OutlierDetectionMethod(str, Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    PERCENTILE = "percentile"


class OutlierTreatmentStrategy(str, Enum):
    NONE = "none"
    CAP_IQR = "cap_iqr"
    CAP_PERCENTILE = "cap_percentile"
    WINSORIZE = "winsorize"
    CLIP = "clip"
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    DROP = "drop"
    INDICATOR = "indicator"


@dataclass
class OutlierResult:
    series: Series
    method_used: OutlierDetectionMethod
    strategy_used: OutlierTreatmentStrategy
    outliers_detected: int
    outliers_treated: int
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    outlier_mask: Optional[Series] = None
    indicator_column: Optional[Series] = None
    rows_dropped: int = 0
    drop_mask: Optional[list[bool]] = None


class OutlierHandler:
    def __init__(
        self,
        detection_method: OutlierDetectionMethod = OutlierDetectionMethod.IQR,
        treatment_strategy: OutlierTreatmentStrategy = OutlierTreatmentStrategy.CAP_IQR,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        percentile_lower: float = 1,
        percentile_upper: float = 99,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ):
        self.detection_method = detection_method
        self.treatment_strategy = treatment_strategy
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._lower_bound: Optional[float] = None
        self._upper_bound: Optional[float] = None
        self._is_fitted = False

    def detect(self, series: Series) -> OutlierResult:
        clean = series.dropna()
        lower, upper = self._compute_bounds(clean)
        mask = (series < lower) | (series > upper)
        mask = mask.fillna(False)

        return OutlierResult(
            series=series, method_used=self.detection_method,
            strategy_used=self.treatment_strategy,
            outliers_detected=int(mask.sum()), outliers_treated=0,
            lower_bound=lower, upper_bound=upper, outlier_mask=mask
        )

    def fit(self, series: Series) -> "OutlierHandler":
        clean = series.dropna()
        self._lower_bound, self._upper_bound = self._compute_bounds(clean)
        self._is_fitted = True
        return self

    def transform(self, series: Series) -> OutlierResult:
        if not self._is_fitted:
            raise ValueError("Handler not fitted. Call fit() or fit_transform() first.")
        return self._apply_treatment(series, self._lower_bound, self._upper_bound)

    def fit_transform(self, series: Series) -> OutlierResult:
        self.fit(series)
        return self._apply_treatment(series, self._lower_bound, self._upper_bound)

    def _compute_bounds(self, clean: Series) -> tuple[float, float]:
        if self.detection_method == OutlierDetectionMethod.IQR:
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            return q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr

        if self.detection_method == OutlierDetectionMethod.ZSCORE:
            mean, std = clean.mean(), clean.std()
            return mean - self.zscore_threshold * std, mean + self.zscore_threshold * std

        if self.detection_method == OutlierDetectionMethod.MODIFIED_ZSCORE:
            median = clean.median()
            mad = np.abs(clean - median).median()
            k = 1.4826
            return median - 3.5 * k * mad, median + 3.5 * k * mad

        if self.detection_method == OutlierDetectionMethod.PERCENTILE:
            return clean.quantile(self.percentile_lower / 100), clean.quantile(self.percentile_upper / 100)

        return clean.min(), clean.max()

    def _apply_treatment(self, series: Series, lower: float, upper: float) -> OutlierResult:
        mask = ((series < lower) | (series > upper)) & series.notna()
        outliers_detected = int(mask.sum())
        result_series = series.copy()

        if self.treatment_strategy == OutlierTreatmentStrategy.NONE:
            return OutlierResult(
                series=result_series, method_used=self.detection_method,
                strategy_used=self.treatment_strategy,
                outliers_detected=outliers_detected, outliers_treated=0,
                lower_bound=lower, upper_bound=upper, outlier_mask=mask
            )

        if self.treatment_strategy == OutlierTreatmentStrategy.INDICATOR:
            indicator = mask.astype(int)
            return OutlierResult(
                series=result_series, method_used=self.detection_method,
                strategy_used=self.treatment_strategy,
                outliers_detected=outliers_detected, outliers_treated=0,
                lower_bound=lower, upper_bound=upper, outlier_mask=mask,
                indicator_column=indicator
            )

        if self.treatment_strategy == OutlierTreatmentStrategy.DROP:
            return OutlierResult(
                series=result_series, method_used=self.detection_method,
                strategy_used=self.treatment_strategy,
                outliers_detected=outliers_detected, outliers_treated=outliers_detected,
                lower_bound=lower, upper_bound=upper, outlier_mask=mask,
                rows_dropped=outliers_detected, drop_mask=mask.tolist()
            )

        if self.treatment_strategy in [OutlierTreatmentStrategy.CAP_IQR, OutlierTreatmentStrategy.WINSORIZE]:
            result_series = result_series.clip(lower=lower, upper=upper)
            return OutlierResult(
                series=result_series, method_used=self.detection_method,
                strategy_used=self.treatment_strategy,
                outliers_detected=outliers_detected, outliers_treated=outliers_detected,
                lower_bound=lower, upper_bound=upper, outlier_mask=mask
            )

        if self.treatment_strategy == OutlierTreatmentStrategy.CAP_PERCENTILE:
            result_series = result_series.clip(lower=lower, upper=upper)
            return OutlierResult(
                series=result_series, method_used=self.detection_method,
                strategy_used=self.treatment_strategy,
                outliers_detected=outliers_detected, outliers_treated=outliers_detected,
                lower_bound=lower, upper_bound=upper, outlier_mask=mask
            )

        if self.treatment_strategy == OutlierTreatmentStrategy.CLIP:
            clip_lower = self.clip_min if self.clip_min is not None else lower
            clip_upper = self.clip_max if self.clip_max is not None else upper
            result_series = result_series.clip(lower=clip_lower, upper=clip_upper)
            return OutlierResult(
                series=result_series, method_used=self.detection_method,
                strategy_used=self.treatment_strategy,
                outliers_detected=outliers_detected, outliers_treated=outliers_detected,
                lower_bound=clip_lower, upper_bound=clip_upper, outlier_mask=mask
            )

        if self.treatment_strategy == OutlierTreatmentStrategy.LOG_TRANSFORM:
            if (series.dropna() < 0).any():
                raise ValueError("Log transform requires non-negative values")
            result_series = np.log1p(series)
            return OutlierResult(
                series=result_series, method_used=self.detection_method,
                strategy_used=self.treatment_strategy,
                outliers_detected=0, outliers_treated=0,
                lower_bound=None, upper_bound=None
            )

        if self.treatment_strategy == OutlierTreatmentStrategy.SQRT_TRANSFORM:
            result_series = np.sqrt(series)
            return OutlierResult(
                series=result_series, method_used=self.detection_method,
                strategy_used=self.treatment_strategy,
                outliers_detected=0, outliers_treated=0,
                lower_bound=None, upper_bound=None
            )

        return OutlierResult(
            series=result_series, method_used=self.detection_method,
            strategy_used=self.treatment_strategy,
            outliers_detected=outliers_detected, outliers_treated=0,
            lower_bound=lower, upper_bound=upper, outlier_mask=mask
        )
