import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AlertLevel(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    DASHBOARD = "dashboard"
    TICKET = "ticket"


@dataclass
class AlertConfig:
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.EMAIL, AlertChannel.SLACK])
    critical_recipients: List[str] = field(default_factory=lambda: ["oncall@company.com"])
    warning_recipients: List[str] = field(default_factory=lambda: ["team@company.com"])
    aggregation_window_minutes: int = 60
    repeat_interval_minutes: int = 240
    slack_webhook_url: Optional[str] = None
    pagerduty_key: Optional[str] = None


@dataclass
class AlertCondition:
    alert_id: str
    name: str
    condition_type: str
    metric: str
    threshold: float
    comparison: str
    level: AlertLevel
    owner: Optional[str] = None


@dataclass
class Alert:
    alert_id: str
    condition_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    current_value: Optional[float] = None
    baseline_value: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AlertResult:
    alerts_triggered: List[Alert]
    alerts_sent: int
    alerts_aggregated: int


class EmailSender:
    def send(self, recipients: List[str], subject: str, body: str):
        pass


class SlackSender:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str, channel: Optional[str] = None):
        pass


class AlertManager:
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.conditions: List[AlertCondition] = []
        self._history: List[Alert] = []
        self._last_sent: Dict[Tuple[str, AlertLevel], datetime] = {}
        self._email_sender = EmailSender()
        self._slack_sender = SlackSender(self.config.slack_webhook_url) if self.config.slack_webhook_url else None

    def add_condition(self, condition: AlertCondition):
        self.conditions.append(condition)

    def load_predefined_conditions(self):
        predefined = [
            AlertCondition(
                alert_id="AL001",
                name="PR-AUC Critical Drop",
                condition_type="metric_threshold",
                metric="pr_auc",
                threshold=0.15,
                comparison="drop_greater_than",
                level=AlertLevel.CRITICAL,
                owner="Data Science Lead"
            ),
            AlertCondition(
                alert_id="AL002",
                name="PSI Critical",
                condition_type="drift_threshold",
                metric="psi",
                threshold=0.20,
                comparison="greater_than",
                level=AlertLevel.CRITICAL,
                owner="Data Science Team"
            ),
            AlertCondition(
                alert_id="AL003",
                name="Scoring Pipeline Failure",
                condition_type="pipeline_status",
                metric="pipeline_status",
                threshold=1,
                comparison="equals",
                level=AlertLevel.CRITICAL,
                owner="MLOps Engineer"
            ),
            AlertCondition(
                alert_id="AL004",
                name="PR-AUC Warning Drop",
                condition_type="metric_threshold",
                metric="pr_auc",
                threshold=0.10,
                comparison="drop_greater_than",
                level=AlertLevel.WARNING,
                owner="Data Scientist"
            ),
            AlertCondition(
                alert_id="AL005",
                name="Multiple Features Drift",
                condition_type="drift_count",
                metric="drifted_features",
                threshold=3,
                comparison="greater_than_or_equal",
                level=AlertLevel.WARNING,
                owner="Data Scientist"
            ),
            AlertCondition(
                alert_id="AL006",
                name="Churn Rate Change",
                condition_type="rate_change",
                metric="churn_rate",
                threshold=0.20,
                comparison="change_greater_than",
                level=AlertLevel.WARNING,
                owner="Business Analyst"
            ),
            AlertCondition(
                alert_id="AL007",
                name="Missing Data Rate",
                condition_type="data_quality",
                metric="missing_rate",
                threshold=0.10,
                comparison="greater_than",
                level=AlertLevel.WARNING,
                owner="Data Engineer"
            ),
            AlertCondition(
                alert_id="AL008",
                name="Score Distribution Shift",
                condition_type="distribution",
                metric="score_distribution",
                threshold=0.10,
                comparison="ks_greater_than",
                level=AlertLevel.INFO,
                owner="Data Scientist"
            ),
            AlertCondition(
                alert_id="AL009",
                name="Single Feature Drift",
                condition_type="drift_threshold",
                metric="psi",
                threshold=0.10,
                comparison="greater_than",
                level=AlertLevel.INFO,
                owner="Data Scientist"
            )
        ]
        self.conditions.extend(predefined)

    def create_alert(self, condition: AlertCondition, current_value: Optional[float] = None,
                     baseline_value: Optional[float] = None, message: Optional[str] = None) -> Alert:
        return Alert(
            alert_id=str(uuid.uuid4()),
            condition_id=condition.alert_id,
            level=condition.level,
            message=message or f"Alert triggered: {condition.name}",
            timestamp=datetime.now(),
            current_value=current_value,
            baseline_value=baseline_value
        )

    def evaluate(self, metrics: Dict[str, Dict[str, float]]) -> List[Alert]:
        alerts = []
        for condition in self.conditions:
            if condition.condition_type != "metric_threshold":
                continue
            if condition.metric not in metrics:
                continue
            metric_data = metrics[condition.metric]
            current = metric_data.get("current", 0)
            baseline = metric_data.get("baseline", 0)
            triggered = False
            if condition.comparison == "drop_greater_than":
                drop = baseline - current
                triggered = drop >= condition.threshold
            elif condition.comparison == "greater_than":
                triggered = current > condition.threshold
            elif condition.comparison == "less_than":
                triggered = current < condition.threshold
            if triggered:
                alert = self.create_alert(
                    condition,
                    current_value=current,
                    baseline_value=baseline,
                    message=f"{condition.name}: {condition.metric} = {current:.4f} (baseline: {baseline:.4f})"
                )
                alerts.append(alert)
        return alerts

    def evaluate_drift(self, drift_metrics: Dict[str, Dict[str, float]]) -> List[Alert]:
        alerts = []
        for condition in self.conditions:
            if condition.condition_type != "drift_threshold":
                continue
            for feature, metrics in drift_metrics.items():
                psi = metrics.get("psi", 0)
                triggered = False
                if condition.comparison == "greater_than":
                    triggered = psi > condition.threshold
                if triggered:
                    alert = self.create_alert(
                        condition,
                        current_value=psi,
                        message=f"{condition.name}: Feature '{feature}' PSI = {psi:.4f}"
                    )
                    alerts.append(alert)
        return alerts

    def get_channels_for_alert(self, alert: Alert) -> List[AlertChannel]:
        channels = []
        if alert.level == AlertLevel.CRITICAL:
            channels = [c for c in self.config.channels]
        elif alert.level == AlertLevel.WARNING:
            channels = [c for c in self.config.channels if c != AlertChannel.PAGERDUTY]
        else:
            channels = [AlertChannel.DASHBOARD]
        return channels

    def send_alert(self, alert: Alert):
        channels = self.get_channels_for_alert(alert)
        if AlertChannel.EMAIL in channels:
            recipients = (self.config.critical_recipients if alert.level == AlertLevel.CRITICAL
                          else self.config.warning_recipients)
            self._email_sender.send(
                recipients=recipients,
                subject=f"[{alert.level.value}] {alert.message}",
                body=self._format_email_body(alert)
            )
        if AlertChannel.SLACK in channels and self._slack_sender:
            self._slack_sender.send(self._format_slack_message(alert))
        self._last_sent[(alert.condition_id, alert.level)] = alert.timestamp
        self.record_alert(alert)

    def should_send_alert(self, alert: Alert) -> bool:
        key = (alert.condition_id, alert.level)
        if key not in self._last_sent:
            return True
        last_sent = self._last_sent[key]
        minutes_since = (alert.timestamp - last_sent).total_seconds() / 60
        return minutes_since >= self.config.repeat_interval_minutes

    def aggregate_alerts(self, alerts: List[Alert]) -> List[Alert]:
        if not alerts:
            return []
        grouped: Dict[str, List[Alert]] = {}
        for alert in alerts:
            key = alert.condition_id
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(alert)
        aggregated = []
        for condition_id, group in grouped.items():
            if len(group) == 1:
                aggregated.append(group[0])
            else:
                representative = group[0]
                representative.message = f"{representative.message} (and {len(group) - 1} similar alerts)"
                aggregated.append(representative)
        return aggregated

    def record_alert(self, alert: Alert):
        self._history.append(alert)

    def add_alert(self, alert: Alert):
        """Add an alert to the pending queue and record it."""
        self.record_alert(alert)

    def get_pending_alerts(self) -> List[Alert]:
        """Get all pending alerts (alerts that haven't been sent yet)."""
        return [a for a in self._history if (a.condition_id, a.level) not in self._last_sent]

    def get_alert_history(self, level: Optional[AlertLevel] = None,
                          since: Optional[datetime] = None) -> List[Alert]:
        history = self._history
        if level:
            history = [a for a in history if a.level == level]
        if since:
            history = [a for a in history if a.timestamp >= since]
        return history

    def _format_email_body(self, alert: Alert) -> str:
        return f"""
Alert Details:
- Level: {alert.level.value}
- Condition ID: {alert.condition_id}
- Time: {alert.timestamp.isoformat()}
- Message: {alert.message}
- Current Value: {alert.current_value}
- Baseline Value: {alert.baseline_value}
"""

    def _format_slack_message(self, alert: Alert) -> str:
        emoji = {"CRITICAL": ":red_circle:", "WARNING": ":warning:", "INFO": ":information_source:"}
        return f"{emoji.get(alert.level.value, '')} *{alert.level.value}*: {alert.message}"
