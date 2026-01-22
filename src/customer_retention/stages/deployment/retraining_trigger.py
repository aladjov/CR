from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta


class RetrainingTriggerType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SIGNIFICANT_DRIFT = "significant_drift"
    SCHEDULED = "scheduled"
    DATA_VOLUME_INCREASE = "data_volume_increase"
    BUSINESS_REQUEST = "business_request"
    NEW_FEATURES = "new_features"


class TriggerPriority(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class RetrainingConfig:
    performance_drop_threshold: float = 0.15
    drift_psi_threshold: float = 0.20
    scheduled_interval_days: int = 90
    data_volume_increase_threshold: float = 0.50
    training_data_window_days: int = 365
    validation_split: float = 0.20
    min_performance_lift: float = 0.02
    auto_deploy: bool = False
    approval_required: bool = True


@dataclass
class RetrainingDecision:
    should_retrain: bool
    trigger_type: Optional[RetrainingTriggerType] = None
    priority: Optional[TriggerPriority] = None
    reason: Optional[str] = None
    action: Optional[str] = None
    requires_approval: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResult:
    triggered_conditions: List[RetrainingDecision]
    final_decision: RetrainingDecision


class RetrainingTrigger:
    def __init__(self, config: Optional[RetrainingConfig] = None):
        self.config = config or RetrainingConfig()
        self._history: List[RetrainingDecision] = []

    def evaluate_performance(self, metrics: Dict[str, Dict[str, float]]) -> RetrainingDecision:
        if "pr_auc" in metrics:
            current = metrics["pr_auc"].get("current", 0)
            baseline = metrics["pr_auc"].get("baseline", 0)
            drop = baseline - current
            if drop >= self.config.performance_drop_threshold:
                decision = RetrainingDecision(
                    should_retrain=True,
                    trigger_type=RetrainingTriggerType.PERFORMANCE_DEGRADATION,
                    priority=TriggerPriority.HIGH,
                    reason=f"PR-AUC dropped by {drop:.2%} (from {baseline:.3f} to {current:.3f})",
                    action="immediate_retrain",
                    requires_approval=self.config.approval_required
                )
                self._history.append(decision)
                return decision
        decision = RetrainingDecision(
            should_retrain=False,
            reason="Performance within acceptable range"
        )
        self._history.append(decision)
        return decision

    def evaluate_drift(self, drift_metrics: Dict[str, Dict[str, float]]) -> RetrainingDecision:
        max_psi = 0
        drifted_features = []
        for feature, metrics in drift_metrics.items():
            psi = metrics.get("psi", 0)
            if psi > max_psi:
                max_psi = psi
            if psi >= self.config.drift_psi_threshold:
                drifted_features.append(feature)
        if drifted_features:
            decision = RetrainingDecision(
                should_retrain=True,
                trigger_type=RetrainingTriggerType.SIGNIFICANT_DRIFT,
                priority=TriggerPriority.HIGH,
                reason=f"Significant drift detected in features: {', '.join(drifted_features)} (max PSI: {max_psi:.3f})",
                action="immediate_retrain",
                requires_approval=self.config.approval_required
            )
            self._history.append(decision)
            return decision
        decision = RetrainingDecision(
            should_retrain=False,
            reason="No significant drift detected"
        )
        self._history.append(decision)
        return decision

    def evaluate_schedule(self, last_training_date: datetime) -> RetrainingDecision:
        days_since_training = (datetime.now() - last_training_date).days
        if days_since_training >= self.config.scheduled_interval_days:
            decision = RetrainingDecision(
                should_retrain=True,
                trigger_type=RetrainingTriggerType.SCHEDULED,
                priority=TriggerPriority.MEDIUM,
                reason=f"Scheduled retraining: {days_since_training} days since last training",
                action="scheduled_retrain",
                requires_approval=self.config.approval_required
            )
            self._history.append(decision)
            return decision
        return RetrainingDecision(
            should_retrain=False,
            reason=f"Next scheduled retraining in {self.config.scheduled_interval_days - days_since_training} days"
        )

    def evaluate_data_volume(self, training_data_size: int, current_data_size: int) -> RetrainingDecision:
        increase_ratio = (current_data_size - training_data_size) / training_data_size
        if increase_ratio >= self.config.data_volume_increase_threshold:
            decision = RetrainingDecision(
                should_retrain=True,
                trigger_type=RetrainingTriggerType.DATA_VOLUME_INCREASE,
                priority=TriggerPriority.MEDIUM,
                reason=f"Data volume increased by {increase_ratio:.1%} ({training_data_size} -> {current_data_size})",
                action="retrain_with_new_data",
                requires_approval=self.config.approval_required
            )
            self._history.append(decision)
            return decision
        return RetrainingDecision(
            should_retrain=False,
            reason=f"Data volume increase ({increase_ratio:.1%}) below threshold"
        )

    def trigger_manual(self, reason: str) -> RetrainingDecision:
        decision = RetrainingDecision(
            should_retrain=True,
            trigger_type=RetrainingTriggerType.BUSINESS_REQUEST,
            priority=TriggerPriority.LOW,
            reason=f"Business request: {reason}",
            action="manual_retrain",
            requires_approval=self.config.approval_required
        )
        self._history.append(decision)
        return decision

    def evaluate_new_features(self, current_features: List[str], new_features: List[str]) -> RetrainingDecision:
        if new_features:
            decision = RetrainingDecision(
                should_retrain=True,
                trigger_type=RetrainingTriggerType.NEW_FEATURES,
                priority=TriggerPriority.LOW,
                reason=f"New features available: {', '.join(new_features)}",
                action="retrain_with_new_features",
                requires_approval=self.config.approval_required
            )
            self._history.append(decision)
            return decision
        return RetrainingDecision(
            should_retrain=False,
            reason="No new features available"
        )

    def make_decision(self, performance_degraded: bool, drift_detected: bool) -> RetrainingDecision:
        if performance_degraded and drift_detected:
            return RetrainingDecision(
                should_retrain=True,
                priority=TriggerPriority.HIGH,
                action="immediate_retrain",
                reason="Both performance degradation and drift detected",
                requires_approval=not self.config.auto_deploy
            )
        elif drift_detected:
            return RetrainingDecision(
                should_retrain=False,
                priority=TriggerPriority.MEDIUM,
                action="investigate_and_prepare",
                reason="Drift detected but performance OK - investigate and prepare for retraining",
                requires_approval=True
            )
        elif performance_degraded:
            return RetrainingDecision(
                should_retrain=False,
                priority=TriggerPriority.MEDIUM,
                action="investigate_possible_retrain",
                reason="Performance degraded without drift - investigate root cause",
                requires_approval=True
            )
        else:
            return RetrainingDecision(
                should_retrain=False,
                priority=TriggerPriority.LOW,
                action="continue_monitoring",
                reason="Performance and drift within acceptable ranges",
                requires_approval=False
            )

    def evaluate_all(self, performance_metrics: Optional[Dict] = None,
                     drift_metrics: Optional[Dict] = None,
                     last_training_date: Optional[datetime] = None,
                     training_data_size: Optional[int] = None,
                     current_data_size: Optional[int] = None) -> EvaluationResult:
        triggered = []
        if performance_metrics:
            result = self.evaluate_performance(performance_metrics)
            if result.should_retrain:
                triggered.append(result)
        if drift_metrics:
            result = self.evaluate_drift(drift_metrics)
            if result.should_retrain:
                triggered.append(result)
        if last_training_date:
            result = self.evaluate_schedule(last_training_date)
            if result.should_retrain:
                triggered.append(result)
        if training_data_size and current_data_size:
            result = self.evaluate_data_volume(training_data_size, current_data_size)
            if result.should_retrain:
                triggered.append(result)
        if triggered:
            triggered.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.priority.value, 3))
            final = triggered[0]
        else:
            final = RetrainingDecision(
                should_retrain=False,
                action="continue_monitoring",
                reason="No retraining triggers activated"
            )
        return EvaluationResult(
            triggered_conditions=triggered,
            final_decision=final
        )

    def get_trigger_history(self, trigger_type: Optional[RetrainingTriggerType] = None) -> List[RetrainingDecision]:
        if trigger_type:
            return [h for h in self._history if h.trigger_type == trigger_type]
        return self._history.copy()
