"""Partial Dependence Plot generation."""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from sklearn.inspection import partial_dependence

from customer_retention.core.compat import DataFrame


@dataclass
class PDPResult:
    feature_name: str
    grid_values: np.ndarray
    pdp_values: np.ndarray
    feature_min: float
    feature_max: float
    average_prediction: float
    ice_values: Optional[List[np.ndarray]] = None


@dataclass
class InteractionResult:
    feature1_name: str
    feature2_name: str
    grid1_values: np.ndarray
    grid2_values: np.ndarray
    pdp_matrix: np.ndarray


class PDPGenerator:
    def __init__(self, model: Any):
        self.model = model

    def generate(self, X: DataFrame, feature: str, grid_resolution: int = 50,
                 include_ice: bool = False, ice_lines: int = 100) -> PDPResult:
        feature_idx = list(X.columns).index(feature)
        pd_result = partial_dependence(
            self.model, X, [feature_idx], kind="average", grid_resolution=grid_resolution
        )
        grid_values = pd_result["grid_values"][0]
        pdp_values = pd_result["average"][0]
        ice_values = None
        if include_ice:
            ice_values = self._calculate_ice(X, feature, grid_values, ice_lines)
        return PDPResult(
            feature_name=feature,
            grid_values=grid_values,
            pdp_values=pdp_values,
            feature_min=float(X[feature].min()),
            feature_max=float(X[feature].max()),
            average_prediction=float(np.mean(pdp_values)),
            ice_values=ice_values
        )

    def _calculate_ice(self, X: DataFrame, feature: str,
                       grid_values: np.ndarray, n_samples: int) -> List[np.ndarray]:
        sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        feature_idx = list(X.columns).index(feature)
        X_sample = X.iloc[sample_indices].values
        ice_lines = []
        for row_vals in X_sample:
            row = row_vals.copy().reshape(1, -1)
            predictions = np.empty(len(grid_values))
            for j, val in enumerate(grid_values):
                row[0, feature_idx] = val
                predictions[j] = self.model.predict_proba(row)[0, 1]
            ice_lines.append(predictions)
        return ice_lines

    def generate_multiple(self, X: DataFrame, features: List[str],
                          grid_resolution: int = 50) -> List[PDPResult]:
        return [self.generate(X, feature, grid_resolution) for feature in features]

    def generate_top_features(self, X: DataFrame, n_features: int = 5,
                              grid_resolution: int = 50) -> List[PDPResult]:
        X_vals = X.values.copy()
        original_pred = self.model.predict_proba(X_vals)[:, 1].mean()
        importances = {}
        for i, feature in enumerate(X.columns):
            saved_col = X_vals[:, i].copy()
            X_vals[:, i] = np.random.permutation(saved_col)
            shuffled_pred = self.model.predict_proba(X_vals)[:, 1].mean()
            X_vals[:, i] = saved_col
            importances[feature] = abs(original_pred - shuffled_pred)
        top_features = sorted(importances, key=importances.get, reverse=True)[:n_features]
        return self.generate_multiple(X, top_features, grid_resolution)

    def generate_interaction(self, X: DataFrame, feature1: str, feature2: str,
                             grid_resolution: int = 20) -> InteractionResult:
        feature1_idx = list(X.columns).index(feature1)
        feature2_idx = list(X.columns).index(feature2)
        pd_result = partial_dependence(
            self.model, X, [(feature1_idx, feature2_idx)], kind="average", grid_resolution=grid_resolution
        )
        grid1 = pd_result["grid_values"][0]
        grid2 = pd_result["grid_values"][1]
        pdp_matrix = pd_result["average"][0]
        return InteractionResult(
            feature1_name=feature1,
            feature2_name=feature2,
            grid1_values=grid1,
            grid2_values=grid2,
            pdp_matrix=pdp_matrix
        )
