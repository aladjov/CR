"""Counterfactual explanation generation."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series


@dataclass
class CounterfactualChange:
    feature_name: str
    original_value: float
    new_value: float
    change_magnitude: float


@dataclass
class Counterfactual:
    original_prediction: float
    counterfactual_prediction: float
    changes: List[CounterfactualChange]
    feasibility_score: float
    business_interpretation: str


class CounterfactualGenerator:
    def __init__(self, model: Any, reference_data: DataFrame,
                 actionable_features: Optional[List[str]] = None,
                 constraints: Optional[Dict[str, Dict[str, float]]] = None):
        self.model = model
        self.reference_data = reference_data
        self.actionable_features = actionable_features or list(reference_data.columns)
        self.constraints = constraints or {}
        self._feature_bounds = self._calculate_bounds()

    def _calculate_bounds(self) -> Dict[str, Dict[str, float]]:
        bounds = {}
        for col in self.reference_data.columns:
            bounds[col] = {
                "min": float(self.reference_data[col].min()),
                "max": float(self.reference_data[col].max()),
                "mean": float(self.reference_data[col].mean()),
                "std": float(self.reference_data[col].std())
            }
        return bounds

    def generate(self, instance: Series, target_class: int = 0,
                 max_iterations: int = 100) -> Counterfactual:
        instance_df = instance.to_frame().T
        original_pred = float(self.model.predict_proba(instance_df)[0, 1])
        best_cf = instance.copy()
        best_pred = original_pred
        best_changes = []
        target_pred = 0.3 if target_class == 0 else 0.7
        for _ in range(max_iterations):
            candidate = self._perturb_instance(instance, best_cf)
            candidate_df = candidate.to_frame().T
            pred = float(self.model.predict_proba(candidate_df)[0, 1])
            improved = (target_class == 0 and pred < best_pred) or (target_class == 1 and pred > best_pred)
            if improved:
                best_cf = candidate
                best_pred = pred
                best_changes = self._compute_changes(instance, best_cf)
            if (target_class == 0 and best_pred < target_pred) or (target_class == 1 and best_pred > target_pred):
                break
        feasibility = self._calculate_feasibility(instance, best_cf)
        interpretation = self._generate_interpretation(best_changes, original_pred, best_pred)
        return Counterfactual(
            original_prediction=original_pred,
            counterfactual_prediction=best_pred,
            changes=best_changes,
            feasibility_score=feasibility,
            business_interpretation=interpretation
        )

    def _perturb_instance(self, original: Series, current: Series) -> Series:
        candidate = current.copy()
        feature = np.random.choice(self.actionable_features)
        bounds = self._get_feature_bounds(feature)
        current_val = candidate[feature]
        step = (bounds["max"] - bounds["min"]) * 0.1
        direction = np.random.choice([-1, 1])
        new_val = current_val + direction * step * np.random.uniform(0.5, 1.5)
        new_val = np.clip(new_val, bounds["min"], bounds["max"])
        candidate[feature] = new_val
        return candidate

    def _get_feature_bounds(self, feature: str) -> Dict[str, float]:
        if feature in self.constraints:
            constraint = self.constraints[feature]
            return {
                "min": constraint.get("min", self._feature_bounds[feature]["min"]),
                "max": constraint.get("max", self._feature_bounds[feature]["max"])
            }
        return self._feature_bounds[feature]

    def _compute_changes(self, original: Series, counterfactual: Series) -> List[CounterfactualChange]:
        changes = []
        for feature in self.actionable_features:
            if abs(original[feature] - counterfactual[feature]) > 1e-6:
                changes.append(CounterfactualChange(
                    feature_name=feature,
                    original_value=float(original[feature]),
                    new_value=float(counterfactual[feature]),
                    change_magnitude=float(abs(original[feature] - counterfactual[feature]))
                ))
        return changes

    def _calculate_feasibility(self, original: Series, counterfactual: Series) -> float:
        total_change = 0
        max_change = 0
        for feature in self.actionable_features:
            bounds = self._feature_bounds[feature]
            range_size = bounds["max"] - bounds["min"]
            if range_size > 0:
                normalized_change = abs(original[feature] - counterfactual[feature]) / range_size
                total_change += normalized_change
                max_change += 1
        if max_change == 0:
            return 1.0
        feasibility = 1 - (total_change / max_change)
        return max(0.0, min(1.0, feasibility))

    def _generate_interpretation(self, changes: List[CounterfactualChange],
                                 original_pred: float, new_pred: float) -> str:
        if not changes:
            return "No changes needed to achieve target prediction."
        change_strs = []
        for c in changes[:3]:
            direction = "increase" if c.new_value > c.original_value else "decrease"
            change_strs.append(f"{direction} {c.feature_name} from {c.original_value:.2f} to {c.new_value:.2f}")
        changes_text = ", ".join(change_strs)
        return f"To reduce churn risk from {original_pred:.1%} to {new_pred:.1%}: {changes_text}"

    def generate_diverse(self, instance: Series, n: int = 3) -> List[Counterfactual]:
        counterfactuals = []
        used_features = set()
        for _ in range(n):
            available = [f for f in self.actionable_features if f not in used_features]
            if not available:
                available = self.actionable_features
            temp_generator = CounterfactualGenerator(
                self.model, self.reference_data,
                actionable_features=available,
                constraints=self.constraints
            )
            cf = temp_generator.generate(instance)
            counterfactuals.append(cf)
            for change in cf.changes:
                used_features.add(change.feature_name)
        return counterfactuals

    def generate_prototype(self, instance: Series, prototype_data: DataFrame) -> Counterfactual:
        instance_df = instance.to_frame().T
        original_pred = float(self.model.predict_proba(instance_df)[0, 1])
        prototype = prototype_data.mean()
        best_cf = instance.copy()
        for feature in self.actionable_features:
            bounds = self._get_feature_bounds(feature)
            target_val = np.clip(prototype[feature], bounds["min"], bounds["max"])
            best_cf[feature] = instance[feature] + 0.5 * (target_val - instance[feature])
        cf_df = best_cf.to_frame().T
        new_pred = float(self.model.predict_proba(cf_df)[0, 1])
        changes = self._compute_changes(instance, best_cf)
        feasibility = self._calculate_feasibility(instance, best_cf)
        interpretation = self._generate_interpretation(changes, original_pred, new_pred)
        return Counterfactual(
            original_prediction=original_pred,
            counterfactual_prediction=new_pred,
            changes=changes,
            feasibility_score=feasibility,
            business_interpretation=interpretation
        )
