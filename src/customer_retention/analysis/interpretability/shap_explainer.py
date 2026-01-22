"""SHAP-based model explainability."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import shap

from customer_retention.core.compat import pd, DataFrame, Series
from sklearn.inspection import permutation_importance


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
