"""Threshold optimization for classification models."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score

from customer_retention.core.compat import DataFrame, Series


class OptimizationObjective(Enum):
    MIN_COST = "min_cost"
    MAX_F1 = "max_f1"
    MAX_F2 = "max_f2"
    TARGET_RECALL = "target_recall"
    TARGET_PRECISION = "target_precision"


@dataclass
class ThresholdResult:
    optimal_threshold: float
    threshold_metrics: Dict[str, float]
    cost_at_threshold: Optional[float]
    comparison_default: Dict[str, Any]


class ThresholdOptimizer:
    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.MAX_F1,
        cost_fn: float = 100,
        cost_fp: float = 10,
        target_recall: Optional[float] = None,
        target_precision: Optional[float] = None,
        threshold_step: float = 0.01,
    ):
        self.objective = objective
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.target_recall = target_recall
        self.target_precision = target_precision
        self.threshold_step = threshold_step

    def optimize(self, model, X: DataFrame, y: Series) -> ThresholdResult:
        probabilities = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.01, 1.0, self.threshold_step)

        best_threshold = 0.5
        best_score = float("-inf") if self.objective != OptimizationObjective.MIN_COST else float("inf")

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            score = self._calculate_score(y, predictions, probabilities, threshold)

            if self._is_better_score(score, best_score):
                best_score = score
                best_threshold = threshold

        optimal_predictions = (probabilities >= best_threshold).astype(int)
        threshold_metrics = self._calculate_metrics(y, optimal_predictions)
        cost_at_threshold = self._calculate_cost(y, optimal_predictions)
        comparison_default = self._compare_with_default(y, probabilities, best_threshold)

        return ThresholdResult(
            optimal_threshold=best_threshold,
            threshold_metrics=threshold_metrics,
            cost_at_threshold=cost_at_threshold,
            comparison_default=comparison_default,
        )

    def _calculate_score(self, y_true, y_pred, y_proba, threshold) -> float:
        if self.objective == OptimizationObjective.MIN_COST:
            return self._calculate_cost(y_true, y_pred)

        if self.objective == OptimizationObjective.MAX_F1:
            return f1_score(y_true, y_pred, zero_division=0)

        if self.objective == OptimizationObjective.MAX_F2:
            return fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        if self.objective == OptimizationObjective.TARGET_RECALL:
            recall = recall_score(y_true, y_pred, zero_division=0)
            if recall >= self.target_recall:
                return precision_score(y_true, y_pred, zero_division=0)
            return float("-inf")

        if self.objective == OptimizationObjective.TARGET_PRECISION:
            precision = precision_score(y_true, y_pred, zero_division=0)
            if precision >= self.target_precision:
                return recall_score(y_true, y_pred, zero_division=0)
            return float("-inf")

        return f1_score(y_true, y_pred, zero_division=0)

    def _is_better_score(self, score: float, best_score: float) -> bool:
        if self.objective == OptimizationObjective.MIN_COST:
            return score < best_score
        return score > best_score

    def _calculate_cost(self, y_true, y_pred) -> float:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return fn * self.cost_fn + fp * self.cost_fp

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        }

    def _compare_with_default(
        self,
        y_true: Series,
        y_proba: np.ndarray,
        optimal_threshold: float,
    ) -> Dict[str, Any]:
        default_threshold = 0.5
        default_preds = (y_proba >= default_threshold).astype(int)
        optimal_preds = (y_proba >= optimal_threshold).astype(int)

        return {
            "default_threshold": default_threshold,
            "default_f1": f1_score(y_true, default_preds, zero_division=0),
            "default_cost": self._calculate_cost(y_true, default_preds),
            "optimal_f1": f1_score(y_true, optimal_preds, zero_division=0),
            "optimal_cost": self._calculate_cost(y_true, optimal_preds),
        }
