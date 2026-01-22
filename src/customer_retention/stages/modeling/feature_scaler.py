"""Feature scaling for model training."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any

from customer_retention.core.compat import pd, DataFrame, Series
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class ScalerType(Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


@dataclass
class ScalingResult:
    scaler: Optional[Any]
    X_train_scaled: DataFrame
    X_test_scaled: DataFrame
    scaling_params: Dict[str, Any]


class FeatureScaler:
    def __init__(
        self,
        scaler_type: ScalerType = ScalerType.ROBUST,
        fit_on_train_only: bool = True,
        save_scaler: bool = True,
    ):
        self.scaler_type = scaler_type
        self.fit_on_train_only = fit_on_train_only
        self.save_scaler = save_scaler
        self._scaler = None
        self._feature_names = None

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

        self._scaler = self._create_scaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        scaling_params = self._extract_params()

        return ScalingResult(
            scaler=self._scaler if self.save_scaler else None,
            X_train_scaled=DataFrame(X_train_scaled, columns=self._feature_names, index=X_train.index),
            X_test_scaled=DataFrame(X_test_scaled, columns=self._feature_names, index=X_test.index),
            scaling_params=scaling_params,
        )

    def transform(self, X: DataFrame) -> DataFrame:
        if self._scaler is None:
            return X
        X_scaled = self._scaler.transform(X)
        return DataFrame(X_scaled, columns=self._feature_names, index=X.index)

    def _create_scaler(self):
        if self.scaler_type == ScalerType.STANDARD:
            return StandardScaler()
        if self.scaler_type == ScalerType.ROBUST:
            return RobustScaler()
        if self.scaler_type == ScalerType.MINMAX:
            return MinMaxScaler()
        return None

    def _extract_params(self) -> Dict[str, Any]:
        if self._scaler is None:
            return {}

        params = {}
        if hasattr(self._scaler, "mean_"):
            params["mean"] = self._scaler.mean_.tolist()
        if hasattr(self._scaler, "scale_"):
            params["scale"] = self._scaler.scale_.tolist()
        if hasattr(self._scaler, "center_"):
            params["center"] = self._scaler.center_.tolist()
        if hasattr(self._scaler, "data_min_"):
            params["data_min"] = self._scaler.data_min_.tolist()
        if hasattr(self._scaler, "data_max_"):
            params["data_max"] = self._scaler.data_max_.tolist()

        return params
