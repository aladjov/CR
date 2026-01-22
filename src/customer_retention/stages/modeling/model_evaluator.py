"""Model evaluation metrics for customer retention prediction."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from customer_retention.core.compat import DataFrame, Series


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    curves: Dict[str, Dict[str, np.ndarray]]
    threshold: float
    predictions: np.ndarray
    probabilities: np.ndarray
    dataset_name: Optional[str] = None


class ModelEvaluator:
    def __init__(self, threshold: float = 0.5, positive_class: int = 1):
        self.threshold = threshold
        self.positive_class = positive_class

    def evaluate(
        self,
        model,
        X: DataFrame,
        y: Series,
        dataset_name: Optional[str] = None,
    ) -> EvaluationResult:
        probabilities = model.predict_proba(X)[:, self.positive_class]
        predictions = (probabilities >= self.threshold).astype(int)

        metrics = self._compute_metrics(y, predictions, probabilities)
        cm = confusion_matrix(y, predictions)
        report = classification_report(y, predictions, output_dict=True)
        curves = self._compute_curves(y, probabilities)

        return EvaluationResult(
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=report,
            curves=curves,
            threshold=self.threshold,
            predictions=predictions,
            probabilities=probabilities,
            dataset_name=dataset_name,
        )

    def _compute_metrics(
        self,
        y_true: Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "average_precision": average_precision_score(y_true, y_proba),
            "brier_score": brier_score_loss(y_true, y_proba),
            "log_loss": log_loss(y_true, y_proba),
        }

        lift_gain = self._compute_lift_gain(y_true, y_proba)
        metrics.update(lift_gain)

        return metrics

    def _compute_curves(
        self,
        y_true: Series,
        y_proba: np.ndarray,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

        return {
            "roc_curve": {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": roc_thresholds,
            },
            "pr_curve": {
                "precision": precision,
                "recall": recall,
                "thresholds": pr_thresholds,
            },
        }

    def _compute_lift_gain(
        self,
        y_true: Series,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        y_true = np.array(y_true)
        sorted_indices = np.argsort(y_proba)[::-1]
        y_sorted = y_true[sorted_indices]

        n_total = len(y_true)
        n_positive = y_true.sum()
        baseline_rate = n_positive / n_total

        metrics = {}
        for k in [10, 20]:
            top_k_idx = int(n_total * k / 100)
            top_k_positive = y_sorted[:top_k_idx].sum()

            lift = (top_k_positive / top_k_idx) / baseline_rate if top_k_idx > 0 else 0
            gain = top_k_positive / n_positive if n_positive > 0 else 0

            metrics[f"lift_at_{k}"] = lift
            metrics[f"gain_at_{k}"] = gain

        return metrics
