from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from customer_retention.core.compat import DataFrame

if TYPE_CHECKING:
    from .context import IterationTrigger


class IterationSignal(Enum):
    DRIFT_CRITICAL = "drift_critical"
    DRIFT_WARNING = "drift_warning"
    PERFORMANCE_CRITICAL = "performance_critical"
    PERFORMANCE_WARNING = "performance_warning"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    SCHEDULED_RETRAIN = "scheduled_retrain"
    MANUAL_TRIGGER = "manual_trigger"


@dataclass
class SignalEvent:
    signal_type: IterationSignal
    source: str
    severity: str
    details: Dict[str, Any]
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "source": self.source,
            "severity": self.severity,
            "details": self.details,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalEvent":
        return cls(
            signal_type=IterationSignal(data["signal_type"]),
            source=data["source"],
            severity=data["severity"],
            details=data["details"],
            recommended_action=data["recommended_action"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        )


class SignalAggregator:
    def __init__(self, drift_detector=None, performance_monitor=None):
        self.drift_detector = drift_detector
        self.performance_monitor = performance_monitor
        self._pending_signals: List[SignalEvent] = []

    def check_drift_signals(self, current_data: DataFrame) -> List[SignalEvent]:
        if self.drift_detector is None:
            return []

        signals = []
        try:
            drift_result = self.drift_detector.detect_drift(current_data, method="psi")

            for feature_result in drift_result.feature_results:
                if not feature_result.drift_detected:
                    continue

                from customer_retention.core.components.enums import Severity
                if feature_result.severity == Severity.CRITICAL:
                    signal_type = IterationSignal.DRIFT_CRITICAL
                    severity = "critical"
                    action = "retrain"
                else:
                    signal_type = IterationSignal.DRIFT_WARNING
                    severity = "warning"
                    action = "monitor"

                signals.append(SignalEvent(
                    signal_type=signal_type,
                    source="drift_detector",
                    severity=severity,
                    details={
                        "feature": feature_result.feature_name,
                        "metric": feature_result.metric_name,
                        "value": feature_result.metric_value
                    },
                    recommended_action=action
                ))
        except Exception:
            pass

        self._pending_signals.extend(signals)
        return signals

    def check_performance_signals(self, current_metrics: Dict[str, float]) -> List[SignalEvent]:
        if self.performance_monitor is None:
            return []

        signals = []
        result = self.performance_monitor.compare_metrics(current_metrics)

        from customer_retention.stages.monitoring.performance_monitor import PerformanceStatus
        if result.status == PerformanceStatus.CRITICAL:
            signals.append(SignalEvent(
                signal_type=IterationSignal.PERFORMANCE_CRITICAL,
                source="performance_monitor",
                severity="critical",
                details={
                    "current_metrics": current_metrics,
                    "baseline_metrics": result.baseline_metrics,
                    "comparison": result.comparison
                },
                recommended_action="retrain"
            ))
        elif result.status == PerformanceStatus.WARNING:
            signals.append(SignalEvent(
                signal_type=IterationSignal.PERFORMANCE_WARNING,
                source="performance_monitor",
                severity="warning",
                details={
                    "current_metrics": current_metrics,
                    "baseline_metrics": result.baseline_metrics,
                    "comparison": result.comparison
                },
                recommended_action="investigate"
            ))

        self._pending_signals.extend(signals)
        return signals

    def add_manual_signal(self, reason: str, details: Dict[str, Any]) -> SignalEvent:
        event = SignalEvent(
            signal_type=IterationSignal.MANUAL_TRIGGER,
            source="user",
            severity="info",
            details={"reason": reason, **details},
            recommended_action="retrain"
        )
        self._pending_signals.append(event)
        return event

    def add_scheduled_signal(self, schedule_name: str) -> SignalEvent:
        event = SignalEvent(
            signal_type=IterationSignal.SCHEDULED_RETRAIN,
            source="scheduler",
            severity="info",
            details={"schedule": schedule_name},
            recommended_action="retrain"
        )
        self._pending_signals.append(event)
        return event

    def check_all_signals(self, current_data: Optional[DataFrame] = None,
                          current_metrics: Optional[Dict[str, float]] = None) -> List[SignalEvent]:
        all_signals = []

        if current_data is not None:
            all_signals.extend(self.check_drift_signals(current_data))

        if current_metrics is not None:
            all_signals.extend(self.check_performance_signals(current_metrics))

        return all_signals

    def get_pending_signals(self) -> List[SignalEvent]:
        return self._pending_signals.copy()

    def clear_signals(self) -> None:
        self._pending_signals.clear()

    def should_trigger_iteration(self) -> Tuple[bool, Optional["IterationTrigger"]]:
        from .context import IterationTrigger

        if not self._pending_signals:
            return False, None

        for signal in self._pending_signals:
            if signal.signal_type == IterationSignal.DRIFT_CRITICAL:
                return True, IterationTrigger.DRIFT_DETECTED
            if signal.signal_type == IterationSignal.PERFORMANCE_CRITICAL:
                return True, IterationTrigger.PERFORMANCE_DROP

        for signal in self._pending_signals:
            if signal.signal_type == IterationSignal.MANUAL_TRIGGER:
                return True, IterationTrigger.MANUAL
            if signal.signal_type == IterationSignal.SCHEDULED_RETRAIN:
                return True, IterationTrigger.SCHEDULED

        critical_count = sum(1 for s in self._pending_signals if "critical" in s.severity.lower())
        warning_count = sum(1 for s in self._pending_signals if "warning" in s.severity.lower())

        if critical_count > 0:
            return True, IterationTrigger.DRIFT_DETECTED
        if warning_count >= 3:
            return True, IterationTrigger.DRIFT_DETECTED

        return False, None

    def get_signal_summary(self) -> Dict[str, Any]:
        return {
            "total": len(self._pending_signals),
            "critical": sum(1 for s in self._pending_signals if "critical" in s.severity.lower()),
            "warning": sum(1 for s in self._pending_signals if "warning" in s.severity.lower()),
            "info": sum(1 for s in self._pending_signals if "info" in s.severity.lower()),
            "signals_by_type": {
                signal_type.value: sum(1 for s in self._pending_signals if s.signal_type == signal_type)
                for signal_type in IterationSignal
            }
        }
