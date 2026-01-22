"""Model comparison and selection for customer retention prediction."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from customer_retention.core.compat import pd, DataFrame, Series
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)


@dataclass
class ModelMetrics:
    pr_auc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    train_test_gap: Optional[float] = None
    cv_std: Optional[float] = None


@dataclass
class ComparisonResult:
    model_metrics: Dict[str, ModelMetrics]
    ranking: List[str]
    best_model_name: str
    comparison_table: DataFrame
    selection_reason: str


class ModelComparator:
    def __init__(
        self,
        primary_metric: str = "pr_auc",
        weights: Optional[Dict[str, float]] = None,
    ):
        self.primary_metric = primary_metric
        self.weights = weights or {
            "pr_auc": 0.40,
            "generalization_gap": 0.20,
            "cv_stability": 0.15,
            "business_cost": 0.15,
            "training_time": 0.05,
            "interpretability": 0.05,
        }

    def compare(
        self,
        models: Dict[str, Any],
        X_test: DataFrame,
        y_test: Series,
        X_train: Optional[DataFrame] = None,
        y_train: Optional[Series] = None,
    ) -> ComparisonResult:
        model_metrics = {}

        for name, model in models.items():
            metrics = self._evaluate_model(model, X_test, y_test, X_train, y_train)
            model_metrics[name] = metrics

        ranking = self._rank_models(model_metrics)
        best_model_name = ranking[0]
        comparison_table = self._build_comparison_table(model_metrics, ranking)
        selection_reason = self._generate_selection_reason(best_model_name, model_metrics)

        return ComparisonResult(
            model_metrics=model_metrics,
            ranking=ranking,
            best_model_name=best_model_name,
            comparison_table=comparison_table,
            selection_reason=selection_reason,
        )

    def _evaluate_model(
        self,
        model,
        X_test: DataFrame,
        y_test: Series,
        X_train: Optional[DataFrame],
        y_train: Optional[Series],
    ) -> ModelMetrics:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        train_test_gap = None
        if X_train is not None and y_train is not None:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, y_train_proba)
            train_test_gap = train_pr_auc - pr_auc

        return ModelMetrics(
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            f1=f1_score(y_test, y_pred, zero_division=0),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            accuracy=accuracy_score(y_test, y_pred),
            train_test_gap=train_test_gap,
        )

    def _rank_models(self, model_metrics: Dict[str, ModelMetrics]) -> List[str]:
        scores = {}
        for name, metrics in model_metrics.items():
            scores[name] = getattr(metrics, self.primary_metric)

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    def _build_comparison_table(
        self,
        model_metrics: Dict[str, ModelMetrics],
        ranking: List[str],
    ) -> DataFrame:
        rows = []
        for name in ranking:
            metrics = model_metrics[name]
            rows.append({
                "model": name,
                "pr_auc": metrics.pr_auc,
                "roc_auc": metrics.roc_auc,
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "accuracy": metrics.accuracy,
                "train_test_gap": metrics.train_test_gap,
            })

        return DataFrame(rows).set_index("model")

    def _generate_selection_reason(
        self,
        best_model_name: str,
        model_metrics: Dict[str, ModelMetrics],
    ) -> str:
        metrics = model_metrics[best_model_name]
        return (
            f"Selected {best_model_name} based on highest {self.primary_metric} "
            f"({getattr(metrics, self.primary_metric):.4f})"
        )
