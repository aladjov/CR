import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from customer_retention.integrations.streaming import (
    ActionType,
    AnomalyTrigger,
    CompositeTrigger,
    Event,
    EventSource,
    EventType,
    PatternTrigger,
    SignalType,
    StreamTriggerType,
    ThresholdTrigger,
    TriggerConfig,
    TriggerContext,
    TriggerEngine,
    WarningLevel,
    WarningResult,
)


@pytest.fixture
def trigger_engine():
    config = TriggerConfig(
        evaluation_interval_seconds=60,
        cooldown_period_seconds=3600,
        max_triggers_per_customer_per_day=3
    )
    return TriggerEngine(config=config)


@pytest.fixture
def high_risk_warning():
    return WarningResult(
        customer_id="CUST001",
        warning_score=0.85,
        warning_level=WarningLevel.HIGH,
        warning_signals=[SignalType.ACTIVITY_DROP, SignalType.SUPPORT_SPIKE],
        primary_signal=SignalType.ACTIVITY_DROP,
        timestamp=datetime.now(),
        recommended_action="phone_call"
    )


@pytest.fixture
def critical_risk_warning():
    return WarningResult(
        customer_id="CUST001",
        warning_score=0.95,
        warning_level=WarningLevel.CRITICAL,
        warning_signals=[SignalType.PAYMENT_ISSUE, SignalType.EXPLICIT_SIGNAL],
        primary_signal=SignalType.PAYMENT_ISSUE,
        timestamp=datetime.now(),
        recommended_action="immediate_escalation"
    )


class TestThresholdTriggers:
    def test_ac9_20_high_risk_immediate_trigger(self, trigger_engine, critical_risk_warning):
        trigger = ThresholdTrigger(
            name="HIGH_RISK_IMMEDIATE",
            threshold=0.90,
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate(critical_risk_warning)
        assert result.triggered is True
        assert result.trigger_name == "HIGH_RISK_IMMEDIATE"

    def test_high_risk_threshold_trigger(self, trigger_engine, high_risk_warning):
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.QUEUE_FOR_OUTREACH
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate(high_risk_warning)
        assert result.triggered is True

    def test_medium_risk_threshold_trigger(self, trigger_engine):
        warning = WarningResult(
            customer_id="CUST001",
            warning_score=0.65,
            warning_level=WarningLevel.MEDIUM,
            warning_signals=[],
            primary_signal=None,
            timestamp=datetime.now(),
            recommended_action="email"
        )
        trigger = ThresholdTrigger(
            name="MEDIUM_RISK",
            threshold=0.60,
            action=ActionType.ADD_TO_CAMPAIGN
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate(warning)
        assert result.triggered is True
        assert result.action == ActionType.ADD_TO_CAMPAIGN

    def test_below_threshold_no_trigger(self, trigger_engine):
        warning = WarningResult(
            customer_id="CUST001",
            warning_score=0.45,
            warning_level=WarningLevel.LOW,
            warning_signals=[],
            primary_signal=None,
            timestamp=datetime.now(),
            recommended_action=None
        )
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate(warning)
        assert result.triggered is False


class TestPatternTriggers:
    def test_ac9_20_cancel_flow_pattern(self, trigger_engine):
        events = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, datetime.now() - timedelta(minutes=10),
                  EventSource.WEBSITE, {"page": "/cancel"}),
            Event("evt_2", "CUST001", EventType.SUPPORT_TICKET, datetime.now(),
                  EventSource.SUPPORT, {"topic": "cancellation"})
        ]
        trigger = PatternTrigger(
            name="CANCEL_FLOW",
            pattern=["page_view:/cancel", "support_ticket"],
            window_minutes=60,
            action=ActionType.RETENTION_OFFER
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate_pattern(events, customer_id="CUST001")
        assert result.triggered is True
        assert result.trigger_name == "CANCEL_FLOW"
        assert result.action == ActionType.RETENTION_OFFER

    def test_competitor_research_pattern(self, trigger_engine):
        events = [
            Event("evt_1", "CUST001", EventType.SEARCH, datetime.now() - timedelta(minutes=5),
                  EventSource.WEBSITE, {"query": "competitor product"}),
            Event("evt_2", "CUST001", EventType.PAGE_VIEW, datetime.now(),
                  EventSource.WEBSITE, {"page": "/pricing"})
        ]
        trigger = PatternTrigger(
            name="COMPETITOR_RESEARCH",
            pattern=["search:competitor", "page_view:/pricing"],
            window_minutes=30,
            action=ActionType.DISCOUNT_EMAIL
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate_pattern(events, customer_id="CUST001")
        assert result.triggered is True

    def test_pattern_not_matched(self, trigger_engine):
        events = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, datetime.now(),
                  EventSource.WEBSITE, {"page": "/products"})
        ]
        trigger = PatternTrigger(
            name="CANCEL_FLOW",
            pattern=["page_view:/cancel", "support_ticket"],
            window_minutes=60,
            action=ActionType.RETENTION_OFFER
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate_pattern(events, customer_id="CUST001")
        assert result.triggered is False


class TestAnomalyTriggers:
    def test_ac9_20_usage_crash_trigger(self, trigger_engine):
        context = TriggerContext(
            customer_id="CUST001",
            current_activity=2.0,
            baseline_activity=10.0,
            activity_drop_percent=0.80
        )
        trigger = AnomalyTrigger(
            name="USAGE_CRASH",
            anomaly_threshold=0.80,
            window_hours=24,
            action=ActionType.ALERT_CS
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate_anomaly(context)
        assert result.triggered is True
        assert result.action == ActionType.ALERT_CS

    def test_spending_spike_trigger(self, trigger_engine):
        context = TriggerContext(
            customer_id="CUST001",
            current_spending=500.0,
            baseline_spending=50.0,
            spending_deviation_zscore=5.0
        )
        trigger = AnomalyTrigger(
            name="SPENDING_SPIKE",
            zscore_threshold=3.0,
            action=ActionType.FRAUD_CHECK
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate_anomaly(context)
        assert result.triggered is True
        assert result.action == ActionType.FRAUD_CHECK

    def test_normal_activity_no_trigger(self, trigger_engine):
        context = TriggerContext(
            customer_id="CUST001",
            current_activity=9.0,
            baseline_activity=10.0,
            activity_drop_percent=0.10
        )
        trigger = AnomalyTrigger(
            name="USAGE_CRASH",
            anomaly_threshold=0.80,
            window_hours=24,
            action=ActionType.ALERT_CS
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate_anomaly(context)
        assert result.triggered is False


class TestCompositeTriggers:
    def test_composite_and_trigger(self, trigger_engine, high_risk_warning):
        trigger = CompositeTrigger(
            name="HIGH_RISK_WITH_PATTERN",
            conditions=[
                ("threshold", {"threshold": 0.70}),
                ("signal", {"signal": SignalType.ACTIVITY_DROP})
            ],
            logic="AND",
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate_composite(high_risk_warning, trigger)
        assert result.triggered is True

    def test_composite_or_trigger(self, trigger_engine):
        warning = WarningResult(
            customer_id="CUST001",
            warning_score=0.55,
            warning_level=WarningLevel.MEDIUM,
            warning_signals=[SignalType.PAYMENT_ISSUE],
            primary_signal=SignalType.PAYMENT_ISSUE,
            timestamp=datetime.now(),
            recommended_action="call"
        )
        trigger = CompositeTrigger(
            name="PAYMENT_OR_HIGH_RISK",
            conditions=[
                ("threshold", {"threshold": 0.80}),
                ("signal", {"signal": SignalType.PAYMENT_ISSUE})
            ],
            logic="OR",
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate_composite(warning, trigger)
        assert result.triggered is True


class TestCooldownPeriod:
    def test_ac9_21_cooldown_respected(self, trigger_engine, high_risk_warning):
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT,
            cooldown_seconds=2
        )
        trigger_engine.register_trigger(trigger)
        result1 = trigger_engine.evaluate(high_risk_warning)
        assert result1.triggered is True
        result2 = trigger_engine.evaluate(high_risk_warning)
        assert result2.triggered is False
        assert result2.cooldown_active is True

    def test_cooldown_expires(self, trigger_engine, high_risk_warning):
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT,
            cooldown_seconds=1
        )
        trigger_engine.register_trigger(trigger)
        result1 = trigger_engine.evaluate(high_risk_warning)
        assert result1.triggered is True
        time.sleep(1.5)
        result2 = trigger_engine.evaluate(high_risk_warning)
        assert result2.triggered is True

    def test_cooldown_per_customer(self, trigger_engine):
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT,
            cooldown_seconds=3600
        )
        trigger_engine.register_trigger(trigger)
        warning1 = WarningResult(
            customer_id="CUST001", warning_score=0.85, warning_level=WarningLevel.HIGH,
            warning_signals=[], primary_signal=None, timestamp=datetime.now(), recommended_action=None
        )
        warning2 = WarningResult(
            customer_id="CUST002", warning_score=0.85, warning_level=WarningLevel.HIGH,
            warning_signals=[], primary_signal=None, timestamp=datetime.now(), recommended_action=None
        )
        trigger_engine.evaluate(warning1)
        result = trigger_engine.evaluate(warning2)
        assert result.triggered is True


class TestMaxTriggersLimit:
    def test_max_triggers_per_day_enforced(self, trigger_engine):
        trigger_engine.config.max_triggers_per_customer_per_day = 2
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.50,
            action=ActionType.EMAIL,
            cooldown_seconds=0
        )
        trigger_engine.register_trigger(trigger)
        results = []
        for i in range(5):
            warning = WarningResult(
                customer_id="CUST001", warning_score=0.6 + i * 0.05,
                warning_level=WarningLevel.MEDIUM, warning_signals=[],
                primary_signal=None, timestamp=datetime.now(), recommended_action=None
            )
            results.append(trigger_engine.evaluate(warning))
        triggered_count = sum(1 for r in results if r.triggered)
        assert triggered_count <= 2


class TestActionExecution:
    def test_ac9_22_action_executed(self, trigger_engine, high_risk_warning):
        action_executor = MagicMock()
        trigger_engine.set_action_executor(action_executor)
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        trigger_engine.evaluate_and_execute(high_risk_warning)
        action_executor.execute.assert_called_once()

    def test_action_not_executed_when_not_triggered(self, trigger_engine):
        action_executor = MagicMock()
        trigger_engine.set_action_executor(action_executor)
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        warning = WarningResult(
            customer_id="CUST001", warning_score=0.50, warning_level=WarningLevel.LOW,
            warning_signals=[], primary_signal=None, timestamp=datetime.now(), recommended_action=None
        )
        trigger_engine.evaluate_and_execute(warning)
        action_executor.execute.assert_not_called()


class TestTriggerResult:
    def test_result_has_all_fields(self, trigger_engine, high_risk_warning):
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate(high_risk_warning)
        assert result.trigger_id is not None
        assert result.customer_id == "CUST001"
        assert result.trigger_type == StreamTriggerType.THRESHOLD
        assert result.trigger_name == "HIGH_RISK"
        assert result.trigger_time is not None
        assert result.action == ActionType.IMMEDIATE_ALERT
        assert result.priority is not None

    def test_result_includes_context(self, trigger_engine, high_risk_warning):
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate(high_risk_warning)
        assert result.context is not None
        assert "warning_score" in result.context


class TestTriggerPriority:
    def test_higher_priority_trigger_evaluated_first(self, trigger_engine):
        low_trigger = ThresholdTrigger(name="LOW", threshold=0.50, action=ActionType.EMAIL, priority=5)
        high_trigger = ThresholdTrigger(name="HIGH", threshold=0.50, action=ActionType.IMMEDIATE_ALERT, priority=1)
        trigger_engine.register_trigger(low_trigger)
        trigger_engine.register_trigger(high_trigger)
        warning = WarningResult(
            customer_id="CUST001", warning_score=0.55, warning_level=WarningLevel.MEDIUM,
            warning_signals=[], primary_signal=None, timestamp=datetime.now(), recommended_action=None
        )
        results = trigger_engine.evaluate_all(warning)
        assert results[0].trigger_name == "HIGH"

    def test_priority_affects_action_selection(self, trigger_engine):
        trigger_engine.register_trigger(ThresholdTrigger(name="T1", threshold=0.50, action=ActionType.EMAIL, priority=3))
        trigger_engine.register_trigger(ThresholdTrigger(name="T2", threshold=0.50, action=ActionType.IMMEDIATE_ALERT, priority=1))
        warning = WarningResult(
            customer_id="CUST001", warning_score=0.55, warning_level=WarningLevel.MEDIUM,
            warning_signals=[], primary_signal=None, timestamp=datetime.now(), recommended_action=None
        )
        result = trigger_engine.evaluate_first_match(warning)
        assert result.action == ActionType.IMMEDIATE_ALERT


class TestTriggerIntegrationWithAlertManager:
    def test_trigger_creates_alert(self, trigger_engine, high_risk_warning):
        from customer_retention.stages.monitoring import AlertManager
        trigger = ThresholdTrigger(
            name="HIGH_RISK",
            threshold=0.80,
            action=ActionType.IMMEDIATE_ALERT
        )
        trigger_engine.register_trigger(trigger)
        result = trigger_engine.evaluate(high_risk_warning)
        alert_manager = AlertManager()
        alert = result.to_alert()
        alert_manager.add_alert(alert)
        pending = alert_manager.get_pending_alerts()
        assert len(pending) >= 1

    def test_trigger_uses_existing_alert_conditions(self, trigger_engine):
        from customer_retention.stages.monitoring import AlertCondition, AlertLevel
        condition = AlertCondition(
            alert_id="AL_STREAMING_001",
            name="High Risk Customer",
            condition_type="threshold",
            metric="warning_score",
            threshold=0.80,
            comparison="greater_than",
            level=AlertLevel.WARNING
        )
        trigger = ThresholdTrigger.from_alert_condition(condition)
        assert trigger.threshold == 0.80
        assert trigger.name == "High Risk Customer"
