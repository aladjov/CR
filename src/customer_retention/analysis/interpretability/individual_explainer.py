"""Individual customer explanation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import numpy as np
import shap

from customer_retention.core.compat import pd, DataFrame, Series
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RiskContribution:
    feature_name: str
    contribution: float
    current_value: float
    direction: str


@dataclass
class IndividualExplanation:
    customer_id: Optional[str]
    churn_probability: float
    base_value: float
    shap_values: np.ndarray
    top_positive_factors: List[RiskContribution]
    top_negative_factors: List[RiskContribution]
    confidence: Confidence
    feature_names: List[str] = field(default_factory=list)


class IndividualExplainer:
    def __init__(self, model: Any, background_data: DataFrame, max_samples: int = 100):
        self.model = model
        self.background_data = background_data.head(max_samples)
        self.feature_names = list(background_data.columns)
        self._explainer = self._create_explainer()

    def _create_explainer(self) -> shap.Explainer:
        model_type = type(self.model).__name__
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier"]:
            return shap.TreeExplainer(self.model)
        if model_type in ["LogisticRegression", "LinearRegression"]:
            return shap.LinearExplainer(self.model, self.background_data)
        return shap.KernelExplainer(self.model.predict_proba, self.background_data)

    def explain(self, instance: Series, customer_id: Optional[str] = None,
                top_n: int = 3) -> IndividualExplanation:
        instance_df = instance.to_frame().T
        shap_values = self._extract_shap_values(instance_df)
        churn_prob = float(self.model.predict_proba(instance_df)[0, 1])
        expected_value = self._get_expected_value()
        positive_factors = self._extract_factors(instance, shap_values, top_n, positive=True)
        negative_factors = self._extract_factors(instance, shap_values, top_n, positive=False)
        confidence = self._assess_confidence(churn_prob)
        return IndividualExplanation(
            customer_id=customer_id,
            churn_probability=churn_prob,
            base_value=float(expected_value),
            shap_values=shap_values,
            top_positive_factors=positive_factors,
            top_negative_factors=negative_factors,
            confidence=confidence,
            feature_names=self.feature_names
        )

    def _extract_shap_values(self, X: DataFrame) -> np.ndarray:
        shap_values = self._explainer.shap_values(X)
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        return shap_values.flatten()

    def _get_expected_value(self) -> float:
        expected_value = self._explainer.expected_value
        if hasattr(expected_value, '__len__'):
            if len(expected_value) > 1:
                return float(expected_value[1])
            return float(expected_value[0])
        return float(expected_value)

    def _extract_factors(self, instance: Series, shap_values: np.ndarray,
                         top_n: int, positive: bool) -> List[RiskContribution]:
        if positive:
            indices = np.argsort(shap_values)[::-1]
            values = [(i, shap_values[i]) for i in indices if shap_values[i] > 0]
        else:
            indices = np.argsort(shap_values)
            values = [(i, shap_values[i]) for i in indices if shap_values[i] < 0]
        factors = []
        for idx, contrib in values[:top_n]:
            feature_name = self.feature_names[idx]
            factors.append(RiskContribution(
                feature_name=feature_name,
                contribution=float(contrib),
                current_value=float(instance[feature_name]),
                direction="increases risk" if contrib > 0 else "decreases risk"
            ))
        return factors

    def _assess_confidence(self, probability: float) -> Confidence:
        if probability < 0.2 or probability > 0.8:
            return Confidence.HIGH
        if 0.4 < probability < 0.6:
            return Confidence.LOW
        return Confidence.MEDIUM

    def find_similar_customers(self, instance: Series, X: DataFrame,
                               y: Series, k: int = 5) -> List[Dict]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        instance_scaled = scaler.transform(instance.to_frame().T)
        knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(instance_scaled)
        similar = []
        for dist, idx in zip(distances[0][1:], indices[0][1:]):
            similar.append({
                "index": int(idx),
                "distance": float(dist),
                "outcome": int(y.iloc[idx]),
                "features": X.iloc[idx].to_dict()
            })
        return similar

    def explain_batch(self, X: DataFrame,
                      customer_ids: Optional[List[str]] = None) -> List[IndividualExplanation]:
        customer_ids = customer_ids or [None] * len(X)
        return [self.explain(X.iloc[i], customer_ids[i]) for i in range(len(X))]
