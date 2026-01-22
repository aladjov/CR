import time
from typing import Dict

import pytest

from customer_retention.integrations.streaming import (
    EarlyWarningConfig,
    EarlyWarningModel,
    SignalDetector,
    SignalType,
    WarningLevel,
)


@pytest.fixture
def early_warning_model():
    config = EarlyWarningConfig(
        activity_drop_threshold=0.50,
        dormant_days_threshold=14,
        support_spike_count=3,
        support_spike_window_days=7
    )
    return EarlyWarningModel(config=config)


@pytest.fixture
def healthy_customer_features() -> Dict[str, float]:
    return {
        "activity_drop_7d": 0.1,
        "days_since_last_order": 5,
        "support_tickets_7d": 0,
        "email_unsubscribe": 0,
        "payment_failure": 0,
        "session_abandon_rate": 0.1,
        "negative_review": 0,
        "page_views_1h": 10.0,
        "orders_7d": 2.0
    }


@pytest.fixture
def at_risk_customer_features() -> Dict[str, float]:
    return {
        "activity_drop_7d": 0.60,
        "days_since_last_order": 20,
        "support_tickets_7d": 4,
        "email_unsubscribe": 0,
        "payment_failure": 0,
        "session_abandon_rate": 0.4,
        "negative_review": 1,
        "page_views_1h": 1.0,
        "orders_7d": 0.0
    }


@pytest.fixture
def critical_risk_features() -> Dict[str, float]:
    return {
        "activity_drop_7d": 0.85,
        "days_since_last_order": 45,
        "support_tickets_7d": 5,
        "email_unsubscribe": 1,
        "payment_failure": 1,
        "session_abandon_rate": 0.8,
        "negative_review": 1,
        "page_views_1h": 0.0,
        "orders_7d": 0.0
    }


class TestEarlyWarningModel:
    def test_ac9_13_model_scores_correctly_healthy(self, early_warning_model, healthy_customer_features):
        result = early_warning_model.predict(
            customer_id="CUST001",
            features=healthy_customer_features
        )
        assert result.warning_score < 0.3
        assert result.warning_level == WarningLevel.LOW

    def test_ac9_13_model_scores_correctly_at_risk(self, early_warning_model, at_risk_customer_features):
        result = early_warning_model.predict(
            customer_id="CUST002",
            features=at_risk_customer_features
        )
        assert result.warning_score >= 0.5
        assert result.warning_level in [WarningLevel.MEDIUM, WarningLevel.HIGH]

    def test_ac9_13_model_scores_correctly_critical(self, early_warning_model, critical_risk_features):
        result = early_warning_model.predict(
            customer_id="CUST003",
            features=critical_risk_features
        )
        assert result.warning_score >= 0.8
        assert result.warning_level == WarningLevel.CRITICAL

    def test_warning_score_range(self, early_warning_model, healthy_customer_features):
        result = early_warning_model.predict("CUST001", healthy_customer_features)
        assert 0.0 <= result.warning_score <= 1.0

    def test_warning_level_mapping(self, early_warning_model):
        assert early_warning_model.score_to_level(0.1) == WarningLevel.LOW
        assert early_warning_model.score_to_level(0.4) == WarningLevel.MEDIUM
        assert early_warning_model.score_to_level(0.7) == WarningLevel.HIGH
        assert early_warning_model.score_to_level(0.95) == WarningLevel.CRITICAL


class TestWarningSignals:
    def test_ac9_14_activity_drop_signal(self, early_warning_model):
        features = {
            "activity_drop_7d": 0.60,
            "days_since_last_order": 5,
            "support_tickets_7d": 0,
            "email_unsubscribe": 0,
            "payment_failure": 0
        }
        result = early_warning_model.predict("CUST001", features)
        assert SignalType.ACTIVITY_DROP in result.warning_signals

    def test_ac9_14_dormant_risk_signal(self, early_warning_model):
        features = {
            "activity_drop_7d": 0.2,
            "days_since_last_order": 18,
            "support_tickets_7d": 0,
            "email_unsubscribe": 0,
            "payment_failure": 0
        }
        result = early_warning_model.predict("CUST001", features)
        assert SignalType.DORMANT_RISK in result.warning_signals

    def test_ac9_14_support_spike_signal(self, early_warning_model):
        features = {
            "activity_drop_7d": 0.1,
            "days_since_last_order": 3,
            "support_tickets_7d": 4,
            "email_unsubscribe": 0,
            "payment_failure": 0
        }
        result = early_warning_model.predict("CUST001", features)
        assert SignalType.SUPPORT_SPIKE in result.warning_signals

    def test_ac9_14_payment_issue_signal(self, early_warning_model):
        features = {
            "activity_drop_7d": 0.1,
            "days_since_last_order": 3,
            "support_tickets_7d": 0,
            "email_unsubscribe": 0,
            "payment_failure": 1
        }
        result = early_warning_model.predict("CUST001", features)
        assert SignalType.PAYMENT_ISSUE in result.warning_signals

    def test_ac9_14_explicit_signal_unsubscribe(self, early_warning_model):
        features = {
            "activity_drop_7d": 0.1,
            "days_since_last_order": 3,
            "support_tickets_7d": 0,
            "email_unsubscribe": 1,
            "payment_failure": 0
        }
        result = early_warning_model.predict("CUST001", features)
        assert SignalType.EXPLICIT_SIGNAL in result.warning_signals

    def test_multiple_signals_detected(self, early_warning_model, critical_risk_features):
        result = early_warning_model.predict("CUST001", critical_risk_features)
        assert len(result.warning_signals) >= 3

    def test_no_signals_for_healthy_customer(self, early_warning_model, healthy_customer_features):
        result = early_warning_model.predict("CUST001", healthy_customer_features)
        assert len(result.warning_signals) == 0


class TestWarningLatency:
    def test_ac9_15_prediction_latency_under_100ms(self, early_warning_model, at_risk_customer_features):
        start = time.time()
        for _ in range(100):
            early_warning_model.predict("CUST001", at_risk_customer_features)
        elapsed_ms = (time.time() - start) * 1000 / 100
        assert elapsed_ms < 100

    def test_batch_prediction_latency(self, early_warning_model, healthy_customer_features):
        customers = {f"CUST{i:03d}": healthy_customer_features.copy() for i in range(100)}
        start = time.time()
        results = early_warning_model.predict_batch(customers)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 1000
        assert len(results) == 100


class TestSignalDetector:
    def test_detect_activity_drop(self):
        detector = SignalDetector()
        features = {"activity_drop_7d": 0.55}
        signals = detector.detect(features)
        assert SignalType.ACTIVITY_DROP in signals

    def test_detect_dormant(self):
        detector = SignalDetector(dormant_days_threshold=14)
        features = {"days_since_last_order": 16}
        signals = detector.detect(features)
        assert SignalType.DORMANT_RISK in signals

    def test_custom_thresholds(self):
        detector = SignalDetector(
            activity_drop_threshold=0.30,
            dormant_days_threshold=7,
            support_spike_threshold=2
        )
        features = {
            "activity_drop_7d": 0.35,
            "days_since_last_order": 8,
            "support_tickets_7d": 2
        }
        signals = detector.detect(features)
        assert SignalType.ACTIVITY_DROP in signals
        assert SignalType.DORMANT_RISK in signals
        assert SignalType.SUPPORT_SPIKE in signals


class TestWarningResult:
    def test_result_has_all_fields(self, early_warning_model, at_risk_customer_features):
        result = early_warning_model.predict("CUST001", at_risk_customer_features)
        assert result.customer_id == "CUST001"
        assert result.warning_score is not None
        assert result.warning_level is not None
        assert result.warning_signals is not None
        assert result.primary_signal is not None
        assert result.timestamp is not None
        assert result.recommended_action is not None

    def test_primary_signal_selection(self, early_warning_model):
        features = {
            "activity_drop_7d": 0.60,
            "days_since_last_order": 20,
            "support_tickets_7d": 4,
            "payment_failure": 1,
            "email_unsubscribe": 0
        }
        result = early_warning_model.predict("CUST001", features)
        assert result.primary_signal == SignalType.PAYMENT_ISSUE

    def test_recommended_action_mapping(self, early_warning_model):
        features_payment = {"payment_failure": 1, "activity_drop_7d": 0.0, "days_since_last_order": 1, "support_tickets_7d": 0, "email_unsubscribe": 0}
        result = early_warning_model.predict("CUST001", features_payment)
        assert "phone" in result.recommended_action.lower() or "call" in result.recommended_action.lower()


class TestFeatureWeighting:
    def test_high_weight_features_impact_score(self, early_warning_model):
        base_features = {
            "activity_drop_7d": 0.0,
            "days_since_last_order": 1,
            "support_tickets_7d": 0,
            "email_unsubscribe": 0,
            "payment_failure": 0,
            "session_abandon_rate": 0.0,
            "negative_review": 0
        }
        payment_features = base_features.copy()
        payment_features["payment_failure"] = 1
        base_result = early_warning_model.predict("CUST001", base_features)
        payment_result = early_warning_model.predict("CUST002", payment_features)
        assert payment_result.warning_score > base_result.warning_score + 0.2

    def test_feature_importance_available(self, early_warning_model):
        importance = early_warning_model.get_feature_importance()
        assert "payment_failure" in importance
        assert "activity_drop_7d" in importance
        assert importance["payment_failure"] > importance.get("session_abandon_rate", 0)


class TestModelTraining:
    def test_train_on_labeled_data(self):
        training_data = [
            ({"activity_drop_7d": 0.1, "days_since_last_order": 3, "support_tickets_7d": 0, "payment_failure": 0, "email_unsubscribe": 0}, 0),
            ({"activity_drop_7d": 0.7, "days_since_last_order": 30, "support_tickets_7d": 5, "payment_failure": 1, "email_unsubscribe": 1}, 1),
        ] * 50
        model = EarlyWarningModel()
        model.train(training_data)
        low_risk = model.predict("TEST1", {"activity_drop_7d": 0.1, "days_since_last_order": 3, "support_tickets_7d": 0, "payment_failure": 0, "email_unsubscribe": 0})
        high_risk = model.predict("TEST2", {"activity_drop_7d": 0.7, "days_since_last_order": 30, "support_tickets_7d": 5, "payment_failure": 1, "email_unsubscribe": 1})
        assert high_risk.warning_score > low_risk.warning_score

    def test_model_serialization(self, early_warning_model):
        serialized = early_warning_model.to_bytes()
        loaded_model = EarlyWarningModel.from_bytes(serialized)
        features = {"activity_drop_7d": 0.5, "days_since_last_order": 10, "support_tickets_7d": 2, "payment_failure": 0, "email_unsubscribe": 0}
        original_result = early_warning_model.predict("CUST001", features)
        loaded_result = loaded_model.predict("CUST001", features)
        assert abs(original_result.warning_score - loaded_result.warning_score) < 0.001


class TestEdgeCases:
    def test_missing_features_handled(self, early_warning_model):
        partial_features = {
            "activity_drop_7d": 0.3,
            "days_since_last_order": 5
        }
        result = early_warning_model.predict("CUST001", partial_features)
        assert result is not None
        assert result.warning_score >= 0

    def test_zero_all_features(self, early_warning_model):
        zero_features = {
            "activity_drop_7d": 0.0,
            "days_since_last_order": 0,
            "support_tickets_7d": 0,
            "email_unsubscribe": 0,
            "payment_failure": 0
        }
        result = early_warning_model.predict("CUST001", zero_features)
        assert result.warning_level == WarningLevel.LOW

    def test_extreme_values_handled(self, early_warning_model):
        extreme_features = {
            "activity_drop_7d": 1.0,
            "days_since_last_order": 365,
            "support_tickets_7d": 100,
            "email_unsubscribe": 1,
            "payment_failure": 1
        }
        result = early_warning_model.predict("CUST001", extreme_features)
        assert result.warning_score <= 1.0
        assert result.warning_level == WarningLevel.CRITICAL


class TestIntegrationWithChapter8Alerts:
    def test_warning_to_alert_conversion(self, early_warning_model, critical_risk_features):
        from customer_retention.stages.monitoring import Alert
        from customer_retention.stages.monitoring import AlertLevel as MonitoringAlertLevel
        result = early_warning_model.predict("CUST001", critical_risk_features)
        alert = result.to_alert()
        assert isinstance(alert, Alert)
        assert alert.level == MonitoringAlertLevel.CRITICAL
        assert "CUST001" in alert.message

    def test_warning_integrates_with_alert_manager(self, early_warning_model, at_risk_customer_features):
        from customer_retention.stages.monitoring import AlertManager
        result = early_warning_model.predict("CUST001", at_risk_customer_features)
        alert = result.to_alert()
        manager = AlertManager()
        manager.add_alert(alert)
        pending = manager.get_pending_alerts()
        assert len(pending) >= 1
