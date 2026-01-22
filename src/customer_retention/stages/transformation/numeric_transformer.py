from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from customer_retention.core.compat import Series, pd


class ScalingStrategy(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    NONE = "none"


class PowerTransform(str, Enum):
    LOG = "log"
    LOG1P = "log1p"
    SQRT = "sqrt"
    BOXCOX = "boxcox"
    YEOJOHNSON = "yeojohnson"
    NONE = "none"


@dataclass
class NumericTransformResult:
    series: Series
    original_mean: float
    original_std: float
    original_min: float
    original_max: float
    transformed_mean: float
    transformed_std: float
    transformations_applied: list = field(default_factory=list)
    scaler_params: dict = field(default_factory=dict)


class NumericTransformer:
    def __init__(
        self,
        scaling: ScalingStrategy = ScalingStrategy.NONE,
        power_transform: PowerTransform = PowerTransform.NONE
    ):
        self.scaling = scaling
        self.power_transform = power_transform
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._median: Optional[float] = None
        self._iqr: Optional[float] = None
        self._max_abs: Optional[float] = None
        self._is_fitted = False

    def fit(self, series: Series) -> "NumericTransformer":
        clean = series.dropna()
        transformed = self._apply_power_transform(clean)

        self._mean = float(transformed.mean())
        self._std = float(transformed.std(ddof=0))
        self._min = float(transformed.min())
        self._max = float(transformed.max())
        self._median = float(transformed.median())
        q1, q3 = transformed.quantile(0.25), transformed.quantile(0.75)
        self._iqr = float(q3 - q1)
        self._max_abs = float(transformed.abs().max())
        self._is_fitted = True
        return self

    def transform(self, series: Series) -> NumericTransformResult:
        if not self._is_fitted:
            raise ValueError("Transformer not fitted. Call fit() or fit_transform() first.")
        return self._apply_transformations(series)

    def fit_transform(self, series: Series) -> NumericTransformResult:
        self.fit(series)
        return self._apply_transformations(series)

    def inverse_transform(self, series: Series) -> Series:
        result = series.copy()

        if self.scaling == ScalingStrategy.STANDARD:
            result = result * self._std + self._mean
        elif self.scaling == ScalingStrategy.MINMAX:
            result = result * (self._max - self._min) + self._min
        elif self.scaling == ScalingStrategy.ROBUST:
            result = result * self._iqr + self._median
        elif self.scaling == ScalingStrategy.MAXABS:
            result = result * self._max_abs

        if self.power_transform == PowerTransform.LOG:
            result = np.exp(result)
        elif self.power_transform == PowerTransform.LOG1P:
            result = np.expm1(result)
        elif self.power_transform == PowerTransform.SQRT:
            result = result ** 2

        return result

    def _apply_power_transform(self, series: Series) -> Series:
        if self.power_transform == PowerTransform.NONE:
            return series

        if self.power_transform == PowerTransform.LOG:
            if (series <= 0).any():
                raise ValueError("Log transform requires positive values")
            return np.log(series)

        if self.power_transform == PowerTransform.LOG1P:
            if (series < 0).any():
                raise ValueError("Log1p transform requires non-negative values")
            return np.log1p(series)

        if self.power_transform == PowerTransform.SQRT:
            if (series < 0).any():
                raise ValueError("Sqrt transform requires non-negative values")
            return np.sqrt(series)

        return series

    def _apply_scaling(self, series: Series) -> Series:
        if self.scaling == ScalingStrategy.NONE:
            return series

        if self.scaling == ScalingStrategy.STANDARD:
            if self._std == 0:
                return series - self._mean
            return (series - self._mean) / self._std

        if self.scaling == ScalingStrategy.MINMAX:
            range_val = self._max - self._min
            if range_val == 0:
                return pd.Series(0.0, index=series.index)
            return (series - self._min) / range_val

        if self.scaling == ScalingStrategy.ROBUST:
            if self._iqr == 0:
                return series - self._median
            return (series - self._median) / self._iqr

        if self.scaling == ScalingStrategy.MAXABS:
            if self._max_abs == 0:
                return series
            return series / self._max_abs

        return series

    def _apply_transformations(self, series: Series) -> NumericTransformResult:
        original_clean = series.dropna()
        original_mean = float(original_clean.mean())
        original_std = float(original_clean.std(ddof=0))
        original_min = float(original_clean.min())
        original_max = float(original_clean.max())

        transformations = []

        mask = series.notna()
        result = series.copy()

        if self.power_transform != PowerTransform.NONE:
            result.loc[mask] = self._apply_power_transform(series[mask])
            transformations.append(self.power_transform)

        if self.scaling != ScalingStrategy.NONE:
            result.loc[mask] = self._apply_scaling(result[mask])
            transformations.append(self.scaling)

        result_clean = result.dropna()
        transformed_mean = float(result_clean.mean()) if len(result_clean) > 0 else 0.0
        transformed_std = float(result_clean.std(ddof=0)) if len(result_clean) > 0 else 0.0

        return NumericTransformResult(
            series=result,
            original_mean=original_mean, original_std=original_std,
            original_min=original_min, original_max=original_max,
            transformed_mean=transformed_mean, transformed_std=transformed_std,
            transformations_applied=transformations,
            scaler_params={"mean": self._mean, "std": self._std, "min": self._min, "max": self._max}
        )
