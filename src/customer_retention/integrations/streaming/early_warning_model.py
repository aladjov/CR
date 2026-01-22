import pickle
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class WarningLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SignalType(Enum):
    ACTIVITY_DROP = "activity_drop"
    DORMANT_RISK = "dormant_risk"
    SUPPORT_SPIKE = "support_spike"
    PAYMENT_ISSUE = "payment_issue"
    EXPLICIT_SIGNAL = "explicit_signal"


@dataclass
class EarlyWarningConfig:
    activity_drop_threshold: float = 0.50
    dormant_days_threshold: int = 14
    support_spike_count: int = 3
    support_spike_window_days: int = 7
    low_threshold: float = 0.30
    medium_threshold: float = 0.30
    high_threshold: float = 0.50
    critical_threshold: float = 0.90


@dataclass
class WarningResult:
    customer_id: str
    warning_score: float
    warning_level: WarningLevel
    warning_signals: List[SignalType]
    primary_signal: Optional[SignalType]
    timestamp: datetime
    recommended_action: Optional[str]

    def to_alert(self):
        from customer_retention.stages.monitoring import Alert, AlertLevel
        level_mapping = {
            WarningLevel.LOW: AlertLevel.INFO,
            WarningLevel.MEDIUM: AlertLevel.WARNING,
            WarningLevel.HIGH: AlertLevel.WARNING,
            WarningLevel.CRITICAL: AlertLevel.CRITICAL
        }
        return Alert(
            alert_id=f"streaming_warning_{self.customer_id}_{self.timestamp.isoformat()}",
            condition_id="STREAMING_WARNING",
            level=level_mapping.get(self.warning_level, AlertLevel.INFO),
            message=f"Early warning for customer {self.customer_id}: {self.warning_level.value} risk (score: {self.warning_score:.2f})",
            timestamp=self.timestamp
        )


class SignalDetector:
    def __init__(self, activity_drop_threshold: float = 0.50, dormant_days_threshold: int = 14,
                 support_spike_threshold: int = 3):
        self._activity_threshold = activity_drop_threshold
        self._dormant_threshold = dormant_days_threshold
        self._support_threshold = support_spike_threshold

    def detect(self, features: Dict[str, float]) -> List[SignalType]:
        signals = []
        if features.get("activity_drop_7d", 0) >= self._activity_threshold:
            signals.append(SignalType.ACTIVITY_DROP)
        if features.get("days_since_last_order", 0) >= self._dormant_threshold:
            signals.append(SignalType.DORMANT_RISK)
        if features.get("support_tickets_7d", 0) >= self._support_threshold:
            signals.append(SignalType.SUPPORT_SPIKE)
        if features.get("payment_failure", 0) > 0:
            signals.append(SignalType.PAYMENT_ISSUE)
        if features.get("email_unsubscribe", 0) > 0:
            signals.append(SignalType.EXPLICIT_SIGNAL)
        return signals


class EarlyWarningModel:
    def __init__(self, config: Optional[EarlyWarningConfig] = None):
        self._config = config or EarlyWarningConfig()
        self._signal_detector = SignalDetector(
            activity_drop_threshold=self._config.activity_drop_threshold,
            dormant_days_threshold=self._config.dormant_days_threshold,
            support_spike_threshold=self._config.support_spike_count
        )
        self._weights = {
            "activity_drop_7d": 0.25,
            "days_since_last_order": 0.20,
            "support_tickets_7d": 0.20,
            "email_unsubscribe": 0.15,
            "payment_failure": 0.30,
            "session_abandon_rate": 0.10,
            "negative_review": 0.10
        }
        self._trained = False
        self._model = None

    def predict(self, customer_id: str, features: Dict[str, float]) -> WarningResult:
        score = self._compute_score(features)
        level = self.score_to_level(score)
        signals = self._signal_detector.detect(features)
        primary = self._get_primary_signal(features, signals)
        action = self._get_recommended_action(primary, level)
        return WarningResult(
            customer_id=customer_id,
            warning_score=score,
            warning_level=level,
            warning_signals=signals,
            primary_signal=primary,
            timestamp=datetime.now(),
            recommended_action=action
        )

    def predict_batch(self, customers: Dict[str, Dict[str, float]]) -> Dict[str, WarningResult]:
        return {cust_id: self.predict(cust_id, features) for cust_id, features in customers.items()}

    def score_to_level(self, score: float) -> WarningLevel:
        if score >= self._config.critical_threshold:
            return WarningLevel.CRITICAL
        elif score >= self._config.high_threshold:
            return WarningLevel.HIGH
        elif score >= self._config.medium_threshold:
            return WarningLevel.MEDIUM
        return WarningLevel.LOW

    def get_feature_importance(self) -> Dict[str, float]:
        return self._weights.copy()

    def train(self, training_data: List[Tuple[Dict[str, float], int]]):
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            feature_names = sorted(self._weights.keys())
            X = []
            y = []
            for features, label in training_data:
                row = [features.get(f, 0.0) for f in feature_names]
                X.append(row)
                y.append(label)
            self._model = LogisticRegression()
            self._model.fit(np.array(X), np.array(y))
            self._trained = True
            for i, name in enumerate(feature_names):
                self._weights[name] = abs(self._model.coef_[0][i])
            total = sum(self._weights.values())
            self._weights = {k: v / total for k, v in self._weights.items()}
        except ImportError:
            pass

    def to_bytes(self) -> bytes:
        return pickle.dumps({
            "config": self._config,
            "weights": self._weights,
            "model": self._model,
            "trained": self._trained
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> "EarlyWarningModel":
        loaded = pickle.loads(data)
        model = cls(config=loaded["config"])
        model._weights = loaded["weights"]
        model._model = loaded["model"]
        model._trained = loaded["trained"]
        return model

    def _compute_score(self, features: Dict[str, float]) -> float:
        if self._trained and self._model:
            try:
                import numpy as np
                feature_names = sorted(self._weights.keys())
                X = [[features.get(f, 0.0) for f in feature_names]]
                return float(self._model.predict_proba(np.array(X))[0][1])
            except Exception:
                pass
        score = 0.0
        normalized_features = self._normalize_features(features)
        for feature_name, weight in self._weights.items():
            value = normalized_features.get(feature_name, 0.0)
            score += weight * value
        return min(max(score, 0.0), 1.0)

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        normalized["activity_drop_7d"] = min(features.get("activity_drop_7d", 0), 1.0)
        days = features.get("days_since_last_order", 0)
        normalized["days_since_last_order"] = min(days / 30.0, 1.0)
        tickets = features.get("support_tickets_7d", 0)
        normalized["support_tickets_7d"] = min(tickets / 5.0, 1.0)
        normalized["email_unsubscribe"] = min(features.get("email_unsubscribe", 0), 1.0)
        normalized["payment_failure"] = min(features.get("payment_failure", 0), 1.0)
        normalized["session_abandon_rate"] = min(features.get("session_abandon_rate", 0), 1.0)
        normalized["negative_review"] = min(features.get("negative_review", 0), 1.0)
        return normalized

    def _get_primary_signal(self, features: Dict[str, float], signals: List[SignalType]) -> Optional[SignalType]:
        if not signals:
            return None
        priority = [
            SignalType.PAYMENT_ISSUE,
            SignalType.EXPLICIT_SIGNAL,
            SignalType.ACTIVITY_DROP,
            SignalType.SUPPORT_SPIKE,
            SignalType.DORMANT_RISK
        ]
        for signal in priority:
            if signal in signals:
                return signal
        return signals[0]

    def _get_recommended_action(self, primary_signal: Optional[SignalType], level: WarningLevel) -> Optional[str]:
        if not primary_signal:
            return None
        action_mapping = {
            SignalType.PAYMENT_ISSUE: "phone_call",
            SignalType.EXPLICIT_SIGNAL: "immediate_escalation",
            SignalType.ACTIVITY_DROP: "email_campaign",
            SignalType.SUPPORT_SPIKE: "cs_followup",
            SignalType.DORMANT_RISK: "re_engagement_email"
        }
        return action_mapping.get(primary_signal)
