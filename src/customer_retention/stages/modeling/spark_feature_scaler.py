from typing import Any, Dict, Tuple

import numpy as np

from customer_retention.core.compat import DataFrame

from .feature_scaler import FeatureScaler, ScalerType, ScalingResult


class SparkFeatureScaler(FeatureScaler):

    def __init__(
        self,
        scaler_type: ScalerType = ScalerType.ROBUST,
        fit_on_train_only: bool = True,
        save_scaler: bool = True,
    ):
        super().__init__(
            scaler_type=scaler_type,
            fit_on_train_only=fit_on_train_only,
            save_scaler=save_scaler,
        )
        self._params: Dict[str, Dict[str, float]] = {}

    def fit_transform(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
    ) -> ScalingResult:
        self._feature_names = list(X_train.columns)

        if self.scaler_type == ScalerType.NONE:
            return ScalingResult(
                scaler=None,
                X_train_scaled=X_train,
                X_test_scaled=X_test,
                scaling_params={},
            )

        self._params = self._compute_params(X_train)
        X_train_scaled = self._apply(X_train)
        X_test_scaled = self._apply(X_test)

        scaling_params = self._extract_params()

        return ScalingResult(
            scaler=self._params if self.save_scaler else None,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            scaling_params=scaling_params,
        )

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._params:
            return X
        return self._apply(X)

    def _compute_params(self, X: DataFrame) -> Dict[str, Dict[str, float]]:
        params: Dict[str, Dict[str, float]] = {}

        if self.scaler_type == ScalerType.STANDARD:
            means = X.mean()
            stds = X.std(axis=0, ddof=0)
            for col in X.columns:
                params[col] = {"mean": float(means[col]), "std": float(stds[col])}

        elif self.scaler_type == ScalerType.ROBUST:
            medians = X.median()
            q25 = X.quantile(0.25)
            q75 = X.quantile(0.75)
            for col in X.columns:
                params[col] = {
                    "center": float(medians[col]),
                    "iqr": float(q75[col]) - float(q25[col]),
                }

        elif self.scaler_type == ScalerType.MINMAX:
            mins = X.min()
            maxs = X.max()
            for col in X.columns:
                params[col] = {"min": float(mins[col]), "max": float(maxs[col])}

        return params

    def _apply(self, X: DataFrame) -> DataFrame:
        offsets, scales = self._as_vectors()
        zero_mask = scales == 0
        safe_scales = np.where(zero_mask, 1.0, scales)
        result = X.copy()
        for i, col in enumerate(self._feature_names):
            if zero_mask[i]:
                result[col] = 0.0
            else:
                result[col] = (X[col] - offsets[i]) / safe_scales[i]
        return result

    def _as_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self._feature_names)
        offsets = np.empty(n)
        scales = np.empty(n)
        for i, c in enumerate(self._feature_names):
            p = self._params[c]
            if self.scaler_type == ScalerType.STANDARD:
                offsets[i], scales[i] = p["mean"], p["std"]
            elif self.scaler_type == ScalerType.ROBUST:
                offsets[i], scales[i] = p["center"], p["iqr"]
            elif self.scaler_type == ScalerType.MINMAX:
                offsets[i] = p["min"]
                scales[i] = p["max"] - p["min"]
        return offsets, scales

    def _extract_params(self) -> Dict[str, Any]:
        if not self._params:
            return {}

        if self.scaler_type == ScalerType.STANDARD:
            return {
                "mean": [self._params[c]["mean"] for c in self._feature_names],
                "scale": [self._params[c]["std"] for c in self._feature_names],
            }

        if self.scaler_type == ScalerType.ROBUST:
            return {
                "center": [self._params[c]["center"] for c in self._feature_names],
                "scale": [self._params[c]["iqr"] for c in self._feature_names],
            }

        if self.scaler_type == ScalerType.MINMAX:
            scales = []
            for c in self._feature_names:
                r = self._params[c]["max"] - self._params[c]["min"]
                scales.append(1.0 / r if r != 0 else 0.0)
            return {
                "data_min": [self._params[c]["min"] for c in self._feature_names],
                "data_max": [self._params[c]["max"] for c in self._feature_names],
                "scale": scales,
            }

        return {}
