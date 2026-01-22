from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass
class ModelFeedback:
    iteration_id: str
    model_type: str
    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    error_analysis: Optional[Dict[str, Any]] = None
    collected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration_id": self.iteration_id,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
            "confusion_matrix": self.confusion_matrix,
            "error_analysis": self.error_analysis,
            "collected_at": self.collected_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelFeedback":
        return cls(
            iteration_id=data["iteration_id"],
            model_type=data["model_type"],
            metrics=data["metrics"],
            feature_importances=data["feature_importances"],
            confusion_matrix=data.get("confusion_matrix"),
            error_analysis=data.get("error_analysis"),
            collected_at=datetime.fromisoformat(data["collected_at"]) if "collected_at" in data else datetime.now()
        )


@dataclass
class FeatureInsight:
    feature_name: str
    importance_rank: int
    importance_score: float
    recommendation_to_drop: bool
    recommendation_to_engineer: Optional[str] = None


class ModelFeedbackCollector:
    def __init__(self, drop_threshold: float = 0.01):
        self.drop_threshold = drop_threshold

    def create_from_sklearn(self, model, iteration_id: str,
                            feature_names: List[str],
                            metrics: Dict[str, float]) -> ModelFeedback:
        model_type = type(model).__name__

        feature_importances = {}
        if hasattr(model, 'feature_importances_'):
            for name, imp in zip(feature_names, model.feature_importances_):
                feature_importances[name] = float(imp)
        elif hasattr(model, 'coef_'):
            import numpy as np
            coefs = np.abs(model.coef_).flatten()
            for name, coef in zip(feature_names, coefs):
                feature_importances[name] = float(coef)

        return ModelFeedback(
            iteration_id=iteration_id,
            model_type=model_type,
            metrics=metrics,
            feature_importances=feature_importances
        )

    def analyze_feature_importance(self, feedback: ModelFeedback) -> List[FeatureInsight]:
        sorted_features = sorted(
            feedback.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        insights = []
        for rank, (feature, score) in enumerate(sorted_features, 1):
            should_drop = score < self.drop_threshold
            recommendation = None
            if should_drop:
                recommendation = f"Consider removing {feature} (importance: {score:.4f})"

            insights.append(FeatureInsight(
                feature_name=feature,
                importance_rank=rank,
                importance_score=score,
                recommendation_to_drop=should_drop,
                recommendation_to_engineer=recommendation
            ))

        return insights

    def suggest_next_actions(self, feedback: ModelFeedback) -> List[str]:
        actions = []

        roc_auc = feedback.metrics.get("roc_auc", 0)
        pr_auc = feedback.metrics.get("pr_auc", 0)

        if roc_auc >= 0.90 or pr_auc >= 0.85:
            actions.append(f"Model performance is excellent (ROC-AUC: {roc_auc:.2f}). Consider deploying.")
        elif roc_auc >= 0.80:
            actions.append(f"Model performance is good (ROC-AUC: {roc_auc:.2f}). Consider feature engineering for improvement.")
        else:
            actions.append(f"Model performance needs improvement (ROC-AUC: {roc_auc:.2f}). Review feature engineering and data quality.")

        low_importance = [
            name for name, score in feedback.feature_importances.items()
            if score < self.drop_threshold
        ]
        if low_importance:
            actions.append(f"Consider dropping {len(low_importance)} low-importance features: {', '.join(low_importance[:5])}")

        if len(feedback.feature_importances) < 5:
            actions.append("Feature set is small. Consider engineering additional features.")

        top_features = self.get_top_features(feedback, n=3)
        if top_features:
            top_names = [f[0] for f in top_features]
            actions.append(f"Top performing features: {', '.join(top_names)}. Consider creating derived features from these.")

        return actions

    def compare_feedback(self, previous: ModelFeedback,
                        current: ModelFeedback) -> Dict[str, Any]:
        metric_improvements = {}
        for metric in current.metrics:
            if metric in previous.metrics:
                metric_improvements[metric] = current.metrics[metric] - previous.metrics[metric]

        avg_improvement = sum(metric_improvements.values()) / len(metric_improvements) if metric_improvements else 0

        if avg_improvement > 0.02:
            trend = "improved"
        elif avg_improvement < -0.02:
            trend = "degraded"
        else:
            trend = "stable"

        feature_changes = {}
        for feature, score in current.feature_importances.items():
            if feature in previous.feature_importances:
                feature_changes[feature] = score - previous.feature_importances[feature]

        return {
            "metric_improvements": metric_improvements,
            "overall_trend": trend,
            "feature_importance_changes": feature_changes,
            "previous_iteration": previous.iteration_id,
            "current_iteration": current.iteration_id
        }

    def get_top_features(self, feedback: ModelFeedback,
                        n: int = 5) -> List[Tuple[str, float]]:
        sorted_features = sorted(
            feedback.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]

    def get_low_importance_features(self, feedback: ModelFeedback) -> List[str]:
        return [
            name for name, score in feedback.feature_importances.items()
            if score < self.drop_threshold
        ]

    def save_feedback(self, feedback: ModelFeedback, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(feedback.to_dict(), f, default_flow_style=False, sort_keys=False)

    def load_feedback(self, path: str) -> ModelFeedback:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return ModelFeedback.from_dict(data)
