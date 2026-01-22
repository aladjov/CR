import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .early_warning_model import WarningResult
from .event_schema import Event


class StreamTriggerType(Enum):
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    TIME_BASED = "time_based"
    COMPOSITE = "composite"


class ActionType(Enum):
    IMMEDIATE_ALERT = "immediate_alert"
    QUEUE_FOR_OUTREACH = "queue_for_outreach"
    ADD_TO_CAMPAIGN = "add_to_campaign"
    RETENTION_OFFER = "retention_offer"
    DISCOUNT_EMAIL = "discount_email"
    ALERT_CS = "alert_cs"
    FRAUD_CHECK = "fraud_check"
    EMAIL = "email"
    PHONE_CALL = "phone_call"


@dataclass
class TriggerConfig:
    evaluation_interval_seconds: int = 60
    cooldown_period_seconds: int = 3600
    max_triggers_per_customer_per_day: int = 3


@dataclass
class TriggerContext:
    customer_id: str
    current_activity: float = 0.0
    baseline_activity: float = 0.0
    activity_drop_percent: float = 0.0
    current_spending: float = 0.0
    baseline_spending: float = 0.0
    spending_deviation_zscore: float = 0.0


@dataclass
class TriggerResult:
    triggered: bool
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str = ""
    trigger_type: Optional[StreamTriggerType] = None
    trigger_name: str = ""
    trigger_time: datetime = field(default_factory=datetime.now)
    action: Optional[ActionType] = None
    priority: int = 3
    context: Dict[str, Any] = field(default_factory=dict)
    cooldown_active: bool = False

    def to_alert(self):
        from customer_retention.stages.monitoring import Alert, AlertLevel
        level = AlertLevel.WARNING
        if self.action == ActionType.IMMEDIATE_ALERT:
            level = AlertLevel.CRITICAL
        return Alert(
            alert_id=self.trigger_id,
            condition_id=f"TRIGGER_{self.trigger_name}",
            level=level,
            message=f"Trigger {self.trigger_name} fired for customer {self.customer_id}",
            timestamp=self.trigger_time
        )


@dataclass
class TriggerDefinition:
    name: str
    action: ActionType
    trigger_type: StreamTriggerType = field(default=StreamTriggerType.THRESHOLD)
    priority: int = 3
    cooldown_seconds: int = 3600


@dataclass
class ThresholdTrigger(TriggerDefinition):
    threshold: float = 0.80
    trigger_type: StreamTriggerType = field(default=StreamTriggerType.THRESHOLD)

    @classmethod
    def from_alert_condition(cls, condition) -> "ThresholdTrigger":
        return cls(
            name=condition.name,
            threshold=condition.threshold,
            action=ActionType.IMMEDIATE_ALERT,
            priority=1 if condition.level.value == "critical" else 3
        )


@dataclass
class PatternTrigger(TriggerDefinition):
    pattern: List[str] = field(default_factory=list)
    window_minutes: int = 60
    trigger_type: StreamTriggerType = field(default=StreamTriggerType.PATTERN)


@dataclass
class AnomalyTrigger(TriggerDefinition):
    anomaly_threshold: float = 0.80
    zscore_threshold: float = 3.0
    window_hours: int = 24
    trigger_type: StreamTriggerType = field(default=StreamTriggerType.ANOMALY)


@dataclass
class CompositeTrigger(TriggerDefinition):
    conditions: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    logic: str = "AND"
    trigger_type: StreamTriggerType = field(default=StreamTriggerType.COMPOSITE)


class TriggerEngine:
    def __init__(self, config: Optional[TriggerConfig] = None):
        self._config = config or TriggerConfig()
        self._triggers: List[TriggerDefinition] = []
        self._cooldowns: Dict[str, Dict[str, datetime]] = {}
        self._trigger_counts: Dict[str, Dict[str, int]] = {}
        self._action_executor: Optional[Any] = None

    @property
    def config(self) -> TriggerConfig:
        return self._config

    def register_trigger(self, trigger: TriggerDefinition):
        self._triggers.append(trigger)
        self._triggers.sort(key=lambda t: t.priority)

    def set_action_executor(self, executor):
        self._action_executor = executor

    def evaluate(self, warning: WarningResult) -> TriggerResult:
        for trigger in self._triggers:
            if isinstance(trigger, ThresholdTrigger):
                if self._check_cooldown(warning.customer_id, trigger.name):
                    return TriggerResult(
                        triggered=False,
                        customer_id=warning.customer_id,
                        cooldown_active=True
                    )
                if self._check_daily_limit(warning.customer_id):
                    return TriggerResult(triggered=False, customer_id=warning.customer_id)
                if warning.warning_score >= trigger.threshold:
                    self._set_cooldown(warning.customer_id, trigger.name, trigger.cooldown_seconds)
                    self._increment_daily_count(warning.customer_id)
                    return TriggerResult(
                        triggered=True,
                        customer_id=warning.customer_id,
                        trigger_type=StreamTriggerType.THRESHOLD,
                        trigger_name=trigger.name,
                        action=trigger.action,
                        priority=trigger.priority,
                        context={"warning_score": warning.warning_score}
                    )
        return TriggerResult(triggered=False, customer_id=warning.customer_id)

    def evaluate_all(self, warning: WarningResult) -> List[TriggerResult]:
        results = []
        for trigger in self._triggers:
            if isinstance(trigger, ThresholdTrigger):
                if warning.warning_score >= trigger.threshold:
                    results.append(TriggerResult(
                        triggered=True,
                        customer_id=warning.customer_id,
                        trigger_type=StreamTriggerType.THRESHOLD,
                        trigger_name=trigger.name,
                        action=trigger.action,
                        priority=trigger.priority,
                        context={"warning_score": warning.warning_score}
                    ))
        results.sort(key=lambda r: r.priority)
        return results

    def evaluate_first_match(self, warning: WarningResult) -> TriggerResult:
        results = self.evaluate_all(warning)
        return results[0] if results else TriggerResult(triggered=False, customer_id=warning.customer_id)

    def evaluate_pattern(self, events: List[Event], customer_id: str) -> TriggerResult:
        for trigger in self._triggers:
            if isinstance(trigger, PatternTrigger):
                if self._match_pattern(events, trigger):
                    return TriggerResult(
                        triggered=True,
                        customer_id=customer_id,
                        trigger_type=StreamTriggerType.PATTERN,
                        trigger_name=trigger.name,
                        action=trigger.action,
                        priority=trigger.priority
                    )
        return TriggerResult(triggered=False, customer_id=customer_id)

    def evaluate_anomaly(self, context: TriggerContext) -> TriggerResult:
        for trigger in self._triggers:
            if isinstance(trigger, AnomalyTrigger):
                if context.activity_drop_percent >= trigger.anomaly_threshold:
                    return TriggerResult(
                        triggered=True,
                        customer_id=context.customer_id,
                        trigger_type=StreamTriggerType.ANOMALY,
                        trigger_name=trigger.name,
                        action=trigger.action,
                        priority=trigger.priority
                    )
                if context.spending_deviation_zscore >= trigger.zscore_threshold:
                    return TriggerResult(
                        triggered=True,
                        customer_id=context.customer_id,
                        trigger_type=StreamTriggerType.ANOMALY,
                        trigger_name=trigger.name,
                        action=trigger.action,
                        priority=trigger.priority
                    )
        return TriggerResult(triggered=False, customer_id=context.customer_id)

    def evaluate_composite(self, warning: WarningResult, trigger: CompositeTrigger) -> TriggerResult:
        results = []
        for condition_type, params in trigger.conditions:
            if condition_type == "threshold":
                results.append(warning.warning_score >= params.get("threshold", 0.5))
            elif condition_type == "signal":
                signal = params.get("signal")
                results.append(signal in warning.warning_signals)
        if trigger.logic == "AND":
            triggered = all(results)
        else:
            triggered = any(results)
        return TriggerResult(
            triggered=triggered,
            customer_id=warning.customer_id,
            trigger_type=StreamTriggerType.COMPOSITE,
            trigger_name=trigger.name,
            action=trigger.action if triggered else None,
            priority=trigger.priority
        )

    def evaluate_and_execute(self, warning: WarningResult):
        result = self.evaluate(warning)
        if result.triggered and self._action_executor:
            self._action_executor.execute(result)
        return result

    def _match_pattern(self, events: List[Event], trigger: PatternTrigger) -> bool:
        cutoff = datetime.now() - timedelta(minutes=trigger.window_minutes)
        recent_events = [e for e in events if e.event_timestamp >= cutoff]
        pattern_index = 0
        for event in sorted(recent_events, key=lambda e: e.event_timestamp):
            pattern_element = trigger.pattern[pattern_index]
            if ":" in pattern_element:
                event_type, qualifier = pattern_element.split(":", 1)
                if event.event_type.value == event_type:
                    page = event.event_properties.get("page", "")
                    query = event.event_properties.get("query", "")
                    if qualifier in page or qualifier in query:
                        pattern_index += 1
            else:
                if event.event_type.value == pattern_element:
                    pattern_index += 1
            if pattern_index >= len(trigger.pattern):
                return True
        return False

    def _check_cooldown(self, customer_id: str, trigger_name: str) -> bool:
        if customer_id not in self._cooldowns:
            return False
        if trigger_name not in self._cooldowns[customer_id]:
            return False
        cooldown_until = self._cooldowns[customer_id][trigger_name]
        return datetime.now() < cooldown_until

    def _set_cooldown(self, customer_id: str, trigger_name: str, seconds: int):
        if customer_id not in self._cooldowns:
            self._cooldowns[customer_id] = {}
        self._cooldowns[customer_id][trigger_name] = datetime.now() + timedelta(seconds=seconds)

    def _check_daily_limit(self, customer_id: str) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        count = self._trigger_counts.get(customer_id, {}).get(today, 0)
        return count >= self._config.max_triggers_per_customer_per_day

    def _increment_daily_count(self, customer_id: str):
        today = datetime.now().strftime("%Y-%m-%d")
        if customer_id not in self._trigger_counts:
            self._trigger_counts[customer_id] = {}
        self._trigger_counts[customer_id][today] = self._trigger_counts[customer_id].get(today, 0) + 1
