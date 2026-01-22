import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from customer_retention.stages.monitoring import (
    AlertManager, Alert, AlertLevel, AlertChannel,
    AlertConfig, AlertCondition, AlertResult
)


class TestAlertLevel:
    def test_alert_level_enum_values(self):
        assert AlertLevel.CRITICAL.value == "CRITICAL"
        assert AlertLevel.WARNING.value == "WARNING"
        assert AlertLevel.INFO.value == "INFO"


class TestAlertChannel:
    def test_alert_channel_enum_values(self):
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.SLACK.value == "slack"
        assert AlertChannel.PAGERDUTY.value == "pagerduty"
        assert AlertChannel.DASHBOARD.value == "dashboard"


class TestAlertConfig:
    def test_config_has_default_values(self):
        config = AlertConfig()
        assert AlertChannel.EMAIL in config.channels
        assert config.aggregation_window_minutes == 60
        assert config.repeat_interval_minutes == 240

    def test_config_accepts_custom_recipients(self):
        config = AlertConfig(
            critical_recipients=["oncall@company.com"],
            warning_recipients=["team@company.com"]
        )
        assert "oncall@company.com" in config.critical_recipients
        assert "team@company.com" in config.warning_recipients


class TestAlertCondition:
    def test_creates_pr_auc_drop_condition(self):
        condition = AlertCondition(
            alert_id="AL001",
            name="PR-AUC Critical Drop",
            condition_type="metric_threshold",
            metric="pr_auc",
            threshold=0.15,
            comparison="drop_greater_than",
            level=AlertLevel.CRITICAL
        )
        assert condition.alert_id == "AL001"
        assert condition.level == AlertLevel.CRITICAL

    def test_creates_psi_threshold_condition(self):
        condition = AlertCondition(
            alert_id="AL002",
            name="Feature PSI Critical",
            condition_type="drift_threshold",
            metric="psi",
            threshold=0.20,
            comparison="greater_than",
            level=AlertLevel.CRITICAL
        )
        assert condition.metric == "psi"
        assert condition.threshold == 0.20


class TestAlertCreation:
    def test_creates_alert_from_condition(self):
        manager = AlertManager()
        condition = AlertCondition(
            alert_id="AL001",
            name="Test Alert",
            condition_type="metric_threshold",
            metric="pr_auc",
            threshold=0.15,
            comparison="drop_greater_than",
            level=AlertLevel.WARNING
        )
        alert = manager.create_alert(
            condition=condition,
            current_value=0.40,
            baseline_value=0.55,
            message="PR-AUC dropped below threshold"
        )
        assert alert.level == AlertLevel.WARNING
        assert alert.condition_id == "AL001"

    def test_alert_includes_timestamp(self):
        manager = AlertManager()
        condition = AlertCondition(
            alert_id="AL001",
            name="Test Alert",
            condition_type="metric_threshold",
            metric="pr_auc",
            threshold=0.15,
            comparison="drop_greater_than",
            level=AlertLevel.INFO
        )
        alert = manager.create_alert(condition=condition, current_value=0.50, baseline_value=0.55)
        assert alert.timestamp is not None


class TestAlertEvaluation:
    def test_evaluates_pr_auc_drop(self):
        manager = AlertManager()
        manager.add_condition(AlertCondition(
            alert_id="AL001",
            name="PR-AUC Drop",
            condition_type="metric_threshold",
            metric="pr_auc",
            threshold=0.15,
            comparison="drop_greater_than",
            level=AlertLevel.CRITICAL
        ))
        metrics = {"pr_auc": {"current": 0.40, "baseline": 0.55}}
        alerts = manager.evaluate(metrics)
        assert len(alerts) > 0
        assert alerts[0].level == AlertLevel.CRITICAL

    def test_evaluates_psi_threshold(self):
        manager = AlertManager()
        manager.add_condition(AlertCondition(
            alert_id="AL002",
            name="PSI Critical",
            condition_type="drift_threshold",
            metric="psi",
            threshold=0.20,
            comparison="greater_than",
            level=AlertLevel.CRITICAL
        ))
        drift_metrics = {"feature1": {"psi": 0.25}, "feature2": {"psi": 0.18}}
        alerts = manager.evaluate_drift(drift_metrics)
        assert len([a for a in alerts if a.level == AlertLevel.CRITICAL]) > 0

    def test_no_alert_when_under_threshold(self):
        manager = AlertManager()
        manager.add_condition(AlertCondition(
            alert_id="AL001",
            name="PR-AUC Drop",
            condition_type="metric_threshold",
            metric="pr_auc",
            threshold=0.15,
            comparison="drop_greater_than",
            level=AlertLevel.CRITICAL
        ))
        metrics = {"pr_auc": {"current": 0.52, "baseline": 0.55}}
        alerts = manager.evaluate(metrics)
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical_alerts) == 0


class TestAlertRouting:
    def test_routes_critical_to_pagerduty(self):
        config = AlertConfig(
            channels=[AlertChannel.EMAIL, AlertChannel.PAGERDUTY],
            critical_recipients=["oncall@company.com"]
        )
        manager = AlertManager(config=config)
        alert = Alert(
            alert_id="alert_123",
            condition_id="AL001",
            level=AlertLevel.CRITICAL,
            message="Critical issue",
            timestamp=datetime.now()
        )
        channels = manager.get_channels_for_alert(alert)
        assert AlertChannel.PAGERDUTY in channels

    def test_routes_warning_to_email(self):
        config = AlertConfig(
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            warning_recipients=["team@company.com"]
        )
        manager = AlertManager(config=config)
        alert = Alert(
            alert_id="alert_123",
            condition_id="AL004",
            level=AlertLevel.WARNING,
            message="Warning issue",
            timestamp=datetime.now()
        )
        channels = manager.get_channels_for_alert(alert)
        assert AlertChannel.EMAIL in channels

    def test_routes_info_to_dashboard_only(self):
        config = AlertConfig(
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD]
        )
        manager = AlertManager(config=config)
        alert = Alert(
            alert_id="alert_123",
            condition_id="AL009",
            level=AlertLevel.INFO,
            message="Info message",
            timestamp=datetime.now()
        )
        channels = manager.get_channels_for_alert(alert)
        assert AlertChannel.DASHBOARD in channels


class TestAlertSending:
    def test_sends_email_alert(self):
        with patch("customer_retention.stages.monitoring.alert_manager.EmailSender") as mock_sender:
            config = AlertConfig(channels=[AlertChannel.EMAIL])
            manager = AlertManager(config=config)
            alert = Alert(
                alert_id="alert_123",
                condition_id="AL001",
                level=AlertLevel.WARNING,
                message="Test alert",
                timestamp=datetime.now()
            )
            manager.send_alert(alert)
            mock_sender.return_value.send.assert_called()

    def test_sends_slack_alert(self):
        with patch("customer_retention.stages.monitoring.alert_manager.SlackSender") as mock_sender:
            config = AlertConfig(channels=[AlertChannel.SLACK], slack_webhook_url="https://hooks.slack.com/test")
            manager = AlertManager(config=config)
            alert = Alert(
                alert_id="alert_123",
                condition_id="AL001",
                level=AlertLevel.WARNING,
                message="Test alert",
                timestamp=datetime.now()
            )
            manager.send_alert(alert)
            mock_sender.return_value.send.assert_called()


class TestAlertAggregation:
    def test_aggregates_similar_alerts(self):
        config = AlertConfig(aggregation_window_minutes=60)
        manager = AlertManager(config=config)
        alerts = [
            Alert(alert_id=f"alert_{i}", condition_id="AL009", level=AlertLevel.INFO,
                  message="Drift detected", timestamp=datetime.now())
            for i in range(5)
        ]
        aggregated = manager.aggregate_alerts(alerts)
        assert len(aggregated) < len(alerts)

    def test_does_not_aggregate_different_conditions(self):
        config = AlertConfig(aggregation_window_minutes=60)
        manager = AlertManager(config=config)
        alerts = [
            Alert(alert_id="alert_1", condition_id="AL001", level=AlertLevel.CRITICAL,
                  message="PR-AUC drop", timestamp=datetime.now()),
            Alert(alert_id="alert_2", condition_id="AL002", level=AlertLevel.CRITICAL,
                  message="PSI threshold", timestamp=datetime.now()),
        ]
        aggregated = manager.aggregate_alerts(alerts)
        assert len(aggregated) == 2


class TestAlertRepeat:
    def test_respects_repeat_interval(self):
        config = AlertConfig(repeat_interval_minutes=240)
        manager = AlertManager(config=config)
        alert = Alert(
            alert_id="alert_1",
            condition_id="AL001",
            level=AlertLevel.WARNING,
            message="Test",
            timestamp=datetime.now()
        )
        manager.send_alert(alert)
        should_send = manager.should_send_alert(Alert(
            alert_id="alert_2",
            condition_id="AL001",
            level=AlertLevel.WARNING,
            message="Test",
            timestamp=datetime.now()
        ))
        assert should_send is False

    def test_sends_after_repeat_interval(self):
        config = AlertConfig(repeat_interval_minutes=240)
        manager = AlertManager(config=config)
        old_alert = Alert(
            alert_id="alert_1",
            condition_id="AL001",
            level=AlertLevel.WARNING,
            message="Test",
            timestamp=datetime.now() - timedelta(minutes=250)
        )
        manager._last_sent[("AL001", AlertLevel.WARNING)] = old_alert.timestamp
        should_send = manager.should_send_alert(Alert(
            alert_id="alert_2",
            condition_id="AL001",
            level=AlertLevel.WARNING,
            message="Test",
            timestamp=datetime.now()
        ))
        assert should_send is True


class TestPredefinedAlerts:
    def test_loads_predefined_alert_conditions(self):
        manager = AlertManager()
        manager.load_predefined_conditions()
        assert len(manager.conditions) > 0
        condition_ids = [c.alert_id for c in manager.conditions]
        assert "AL001" in condition_ids
        assert "AL002" in condition_ids
        assert "AL003" in condition_ids

    def test_predefined_alert_levels_correct(self):
        manager = AlertManager()
        manager.load_predefined_conditions()
        al001 = next((c for c in manager.conditions if c.alert_id == "AL001"), None)
        assert al001 is not None
        assert al001.level == AlertLevel.CRITICAL


class TestAlertHistory:
    def test_stores_alert_history(self):
        manager = AlertManager()
        alert = Alert(
            alert_id="alert_1",
            condition_id="AL001",
            level=AlertLevel.WARNING,
            message="Test",
            timestamp=datetime.now()
        )
        manager.record_alert(alert)
        history = manager.get_alert_history()
        assert len(history) == 1

    def test_queries_alerts_by_level(self):
        manager = AlertManager()
        for i, level in enumerate([AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.WARNING]):
            manager.record_alert(Alert(
                alert_id=f"alert_{i}",
                condition_id="AL001",
                level=level,
                message="Test",
                timestamp=datetime.now()
            ))
        critical_alerts = manager.get_alert_history(level=AlertLevel.CRITICAL)
        assert len(critical_alerts) == 1

    def test_queries_alerts_by_time_range(self):
        manager = AlertManager()
        base_time = datetime.now()
        for i in range(5):
            manager.record_alert(Alert(
                alert_id=f"alert_{i}",
                condition_id="AL001",
                level=AlertLevel.INFO,
                message="Test",
                timestamp=base_time - timedelta(days=i)
            ))
        recent = manager.get_alert_history(since=base_time - timedelta(days=2, hours=1))
        assert len(recent) == 3
