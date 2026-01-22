from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    precision_score, recall_score
)
import time

from customer_retention.core.compat import pd, DataFrame, Series


class ModelRole(Enum):
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    SHADOW = "shadow"


@dataclass
class PromotionCriteria:
    min_pr_auc_improvement: float = 0.02
    max_fairness_regression: float = 0.0
    max_latency_ratio: float = 2.0
    requires_validation_pass: bool = True
    requires_business_approval: bool = True
    requires_rollback_plan: bool = True


@dataclass
class ComparisonResult:
    champion_metrics: Dict[str, Any]
    challenger_metrics: Dict[str, Any]
    pr_auc_improvement: float
    recommendation: str
    meets_promotion_criteria: bool
    weighted_score_champion: Optional[float] = None
    weighted_score_challenger: Optional[float] = None
    fairness_comparison: Optional[Dict] = None
    latency_comparison: Optional[Dict] = None
    comparison_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RollbackPlan:
    current_model_name: str
    current_version: str
    rollback_model_name: str
    rollback_version: str
    estimated_duration_minutes: float = 5.0
    steps: List[str] = field(default_factory=list)


@dataclass
class RollbackResult:
    success: bool
    from_version: str
    to_version: str
    duration_seconds: float
    error: Optional[str] = None


class ChampionChallenger:
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 promotion_criteria: Optional[PromotionCriteria] = None):
        self.weights = weights or {
            "pr_auc": 0.40,
            "stability": 0.20,
            "business_roi": 0.25,
            "latency": 0.10,
            "fairness": 0.05
        }
        self.promotion_criteria = promotion_criteria or PromotionCriteria()
        self.champion = None
        self.champion_name = None
        self.champion_version = None
        self.challenger = None
        self.challenger_name = None
        self.challenger_version = None
        self.shadow_models: List[Dict] = []
        self._comparison_history: List[ComparisonResult] = []

    def set_champion(self, model: Any, model_name: str, version: str):
        self.champion = model
        self.champion_name = model_name
        self.champion_version = version

    def set_challenger(self, model: Any, model_name: str, version: str):
        self.challenger = model
        self.challenger_name = model_name
        self.challenger_version = version

    def add_shadow(self, model: Any, model_name: str, version: str):
        self.shadow_models.append({
            "model": model,
            "name": model_name,
            "version": version
        })

    def compare(self, X: DataFrame, y: Series,
                protected_attribute: Optional[Series] = None,
                include_stability: bool = False,
                include_latency: bool = False) -> ComparisonResult:
        champion_metrics = self._evaluate_model(self.champion, X, y, include_latency)
        challenger_metrics = self._evaluate_model(self.challenger, X, y, include_latency)
        if include_stability:
            champion_metrics["cv_std"] = self._compute_stability(self.champion, X, y)
            challenger_metrics["cv_std"] = self._compute_stability(self.challenger, X, y)
        pr_auc_improvement = challenger_metrics["pr_auc"] - champion_metrics["pr_auc"]
        fairness_comparison = None
        if protected_attribute is not None:
            fairness_comparison = self._compare_fairness(X, y, protected_attribute)
            champion_metrics["fairness_metrics"] = fairness_comparison.get("champion")
            challenger_metrics["fairness_metrics"] = fairness_comparison.get("challenger")
        latency_comparison = None
        if include_latency:
            latency_comparison = {
                "champion_ms": champion_metrics.get("latency_ms"),
                "challenger_ms": challenger_metrics.get("latency_ms")
            }
        weighted_champion = self._compute_weighted_score(champion_metrics)
        weighted_challenger = self._compute_weighted_score(challenger_metrics)
        meets_criteria = self._check_promotion_criteria(
            pr_auc_improvement, fairness_comparison, latency_comparison, champion_metrics, challenger_metrics
        )
        if meets_criteria and pr_auc_improvement >= self.promotion_criteria.min_pr_auc_improvement:
            recommendation = "promote_challenger"
        else:
            recommendation = "keep_champion"
        result = ComparisonResult(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            pr_auc_improvement=pr_auc_improvement,
            recommendation=recommendation,
            meets_promotion_criteria=meets_criteria,
            weighted_score_champion=weighted_champion,
            weighted_score_challenger=weighted_challenger,
            fairness_comparison=fairness_comparison,
            latency_comparison=latency_comparison
        )
        self._comparison_history.append(result)
        return result

    def _evaluate_model(self, model: Any, X: DataFrame, y: Series,
                        include_latency: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        y_prob = model.predict_proba(X)[:, 1]
        latency_ms = (time.time() - start_time) * 1000
        y_pred = (y_prob >= 0.5).astype(int)
        precision, recall, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y, y_prob)
        metrics = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred)
        }
        if include_latency:
            metrics["latency_ms"] = latency_ms
        return metrics

    def _compute_stability(self, model: Any, X: DataFrame, y: Series, n_splits: int = 5) -> float:
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=n_splits, scoring="roc_auc")
        return scores.std()

    def _compare_fairness(self, X: DataFrame, y: Series,
                          protected: Series) -> Dict:
        champion_probs = self.champion.predict_proba(X)[:, 1]
        challenger_probs = self.challenger.predict_proba(X)[:, 1]
        groups = protected.unique()
        champion_fairness = {}
        challenger_fairness = {}
        for group in groups:
            mask = protected == group
            champion_fairness[group] = np.mean(champion_probs[mask])
            challenger_fairness[group] = np.mean(challenger_probs[mask])
        return {
            "champion": champion_fairness,
            "challenger": challenger_fairness
        }

    def _compute_weighted_score(self, metrics: Dict[str, Any]) -> float:
        score = 0.0
        if "pr_auc" in metrics:
            score += self.weights.get("pr_auc", 0) * metrics["pr_auc"]
        if "cv_std" in metrics:
            stability_score = 1 - min(metrics["cv_std"] * 10, 1)
            score += self.weights.get("stability", 0) * stability_score
        return score

    def _check_promotion_criteria(self, pr_auc_improvement: float,
                                   fairness_comparison: Optional[Dict],
                                   latency_comparison: Optional[Dict],
                                   champion_metrics: Dict,
                                   challenger_metrics: Dict) -> bool:
        if pr_auc_improvement < self.promotion_criteria.min_pr_auc_improvement:
            return False
        if latency_comparison:
            champ_latency = latency_comparison.get("champion_ms", 1)
            chall_latency = latency_comparison.get("challenger_ms", 1)
            if champ_latency and chall_latency:
                if chall_latency > champ_latency * self.promotion_criteria.max_latency_ratio:
                    return False
        return True

    def score_with_shadow(self, X: DataFrame) -> Dict[str, Any]:
        champion_preds = self.champion.predict_proba(X)[:, 1]
        shadow_preds = {}
        for shadow in self.shadow_models:
            shadow_preds[shadow["name"]] = shadow["model"].predict_proba(X)[:, 1]
        return {
            "champion_predictions": champion_preds,
            "shadow_predictions": shadow_preds,
            "active_predictions": champion_preds
        }

    def get_comparison_history(self) -> List[ComparisonResult]:
        return self._comparison_history.copy()

    def generate_report(self) -> str:
        if not self._comparison_history:
            return "No comparisons performed"
        latest = self._comparison_history[-1]
        report = f"""Champion vs Challenger Report
=============================
Champion Metrics:
  PR-AUC: {latest.champion_metrics.get('pr_auc', 'N/A'):.4f}
  ROC-AUC: {latest.champion_metrics.get('roc_auc', 'N/A'):.4f}

Challenger Metrics:
  PR-AUC: {latest.challenger_metrics.get('pr_auc', 'N/A'):.4f}
  ROC-AUC: {latest.challenger_metrics.get('roc_auc', 'N/A'):.4f}

PR-AUC Improvement: {latest.pr_auc_improvement:.4f}
Meets Promotion Criteria: {latest.meets_promotion_criteria}
Recommendation: {latest.recommendation}
"""
        return report


class RollbackManager:
    def __init__(self, notify_on_rollback: bool = True):
        self.notify_on_rollback = notify_on_rollback

    def create_plan(self, current_model_name: str, current_version: str,
                    rollback_model_name: str, rollback_version: str) -> RollbackPlan:
        steps = [
            "1. Stop scoring with current model",
            "2. Load rollback model from registry",
            "3. Validate rollback model",
            "4. Switch production pointer to rollback model",
            "5. Notify stakeholders",
            "6. Document incident"
        ]
        return RollbackPlan(
            current_model_name=current_model_name,
            current_version=current_version,
            rollback_model_name=rollback_model_name,
            rollback_version=rollback_version,
            estimated_duration_minutes=5.0,
            steps=steps
        )

    def execute_rollback(self, model_name: str, from_version: str,
                         to_version: str) -> RollbackResult:
        start_time = time.time()
        try:
            from customer_retention.stages.deployment.model_registry import ModelRegistry, ModelStage
            registry = ModelRegistry()
            registry.transition_stage(model_name, to_version, ModelStage.PRODUCTION)
            registry.transition_stage(model_name, from_version, ModelStage.ARCHIVED)
            if self.notify_on_rollback:
                from customer_retention.stages.monitoring.alert_manager import AlertManager, Alert, AlertLevel
                alert_manager = AlertManager()
                alert = Alert(
                    alert_id=f"rollback_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    condition_id="ROLLBACK",
                    level=AlertLevel.CRITICAL,
                    message=f"Model rollback executed: {model_name} from v{from_version} to v{to_version}",
                    timestamp=datetime.now()
                )
                alert_manager.send_alert(alert)
            duration = time.time() - start_time
            return RollbackResult(
                success=True,
                from_version=from_version,
                to_version=to_version,
                duration_seconds=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            return RollbackResult(
                success=False,
                from_version=from_version,
                to_version=to_version,
                duration_seconds=duration,
                error=str(e)
            )
