"""SHAP-based model explainability."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import shap
from sklearn.inspection import permutation_importance

from customer_retention.core.compat import DataFrame, Series


def select_risk_stratified_sample(
    probabilities: np.ndarray, n_samples: int = 200,
    n_bins: int = 5, priority_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return sorted indices for risk-stratified SHAP sampling.

    Bins entities by predicted probability quantiles and samples proportionally
    from each bin. Priority entities (via boolean mask) are always included.
    """
    n_total = len(probabilities)
    if n_total <= n_samples:
        return np.arange(n_total)

    rng = np.random.default_rng(42)
    priority_idx = np.where(priority_mask)[0] if priority_mask is not None else np.array([], dtype=int)

    if len(priority_idx) >= n_samples:
        return np.sort(rng.choice(priority_idx, n_samples, replace=False))

    selected = set(priority_idx.tolist())
    remaining_budget = n_samples - len(selected)
    candidates = np.array([i for i in range(n_total) if i not in selected])

    bin_edges = np.quantile(probabilities[candidates], np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    bin_ids = np.digitize(probabilities[candidates], bin_edges[1:-1])

    bins: dict[int, list[int]] = {}
    for pos, idx in enumerate(candidates):
        bins.setdefault(bin_ids[pos], []).append(idx)

    n_active_bins = len(bins)
    per_bin = remaining_budget // max(n_active_bins, 1)
    extra = remaining_budget - per_bin * n_active_bins

    for i, b in enumerate(sorted(bins)):
        members = np.array(bins[b])
        alloc = per_bin + (1 if i < extra else 0)
        take = min(alloc, len(members))
        if take > 0:
            selected.update(rng.choice(members, take, replace=False).tolist())

    return np.sort(np.array(list(selected), dtype=int))


@dataclass
class FeatureImportance:
    feature_name: str
    importance: float
    mean_abs_shap: float
    business_description: Optional[str] = None


@dataclass
class GlobalExplanation:
    feature_importance: List[FeatureImportance]
    shap_values: np.ndarray
    expected_value: float
    feature_names: List[str] = field(default_factory=list)


class ShapExplainer:
    def __init__(self, model: Any, background_data: DataFrame,
                 feature_translations: Optional[Dict[str, str]] = None, max_samples: int = 100):
        self.model = model
        self.background_data = background_data.head(max_samples)
        self.feature_translations = feature_translations or {}
        self.explainer_type = self._determine_explainer_type()
        self._explainer = self._create_explainer()

    def _determine_explainer_type(self) -> str:
        model_type = type(self.model).__name__
        tree_models = ["RandomForestClassifier", "GradientBoostingClassifier",
                       "XGBClassifier", "LGBMClassifier", "DecisionTreeClassifier", "RandomForestRegressor"]
        linear_models = ["LogisticRegression", "LinearRegression", "Ridge", "Lasso"]
        if model_type in tree_models:
            return "tree"
        if model_type in linear_models:
            return "linear"
        return "kernel"

    def _create_explainer(self) -> shap.Explainer:
        if self.explainer_type == "tree":
            return shap.TreeExplainer(self.model)
        if self.explainer_type == "linear":
            return shap.LinearExplainer(self.model, self.background_data)
        return shap.KernelExplainer(self.model.predict_proba, self.background_data)

    def explain_global(self, X: DataFrame, top_n: Optional[int] = None) -> GlobalExplanation:
        shap_values = self._extract_shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        if top_n:
            sorted_indices = sorted_indices[:top_n]
        feature_importance = []
        for idx in sorted_indices:
            feature_name = X.columns[idx]
            importance_val = mean_abs_shap[idx]
            if hasattr(importance_val, '__len__') and len(importance_val) == 1:
                importance_val = importance_val[0]
            feature_importance.append(FeatureImportance(
                feature_name=feature_name,
                importance=float(importance_val),
                mean_abs_shap=float(importance_val),
                business_description=self.feature_translations.get(feature_name, feature_name)
            ))
        expected_value = self._get_expected_value()
        return GlobalExplanation(
            feature_importance=feature_importance,
            shap_values=shap_values,
            expected_value=float(expected_value),
            feature_names=list(X.columns)
        )

    def _extract_shap_values(self, X: DataFrame) -> np.ndarray:
        shap_values = self._explainer.shap_values(X)
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        return shap_values

    def _get_expected_value(self) -> float:
        expected_value = self._explainer.expected_value
        if hasattr(expected_value, '__len__'):
            if len(expected_value) > 1:
                return float(expected_value[1])
            return float(expected_value[0])
        return float(expected_value)

    def calculate_permutation_importance(self, X: DataFrame, y: Series,
                                         n_repeats: int = 10) -> Dict[str, float]:
        result = permutation_importance(self.model, X, y, n_repeats=n_repeats, random_state=42)
        return {feature: float(importance) for feature, importance in zip(X.columns, result.importances_mean)}

    def get_shap_values(self, X: DataFrame) -> np.ndarray:
        return self._extract_shap_values(X)
