import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from customer_retention.stages.deployment import (
    RetrainingTrigger, RetrainingTriggerType, TriggerPriority,
    RetrainingDecision, RetrainingConfig
)


class TestRetrainingTriggerType:
    def test_trigger_type_enum_values(self):
        assert RetrainingTriggerType.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert RetrainingTriggerType.SIGNIFICANT_DRIFT.value == "significant_drift"
        assert RetrainingTriggerType.SCHEDULED.value == "scheduled"
        assert RetrainingTriggerType.DATA_VOLUME_INCREASE.value == "data_volume_increase"
        assert RetrainingTriggerType.BUSINESS_REQUEST.value == "business_request"
        assert RetrainingTriggerType.NEW_FEATURES.value == "new_features"


class TestTriggerPriority:
    def test_trigger_priority_enum_values(self):
        assert TriggerPriority.HIGH.value == "HIGH"
        assert TriggerPriority.MEDIUM.value == "MEDIUM"
        assert TriggerPriority.LOW.value == "LOW"


class TestRetrainingConfig:
    def test_config_has_default_values(self):
        config = RetrainingConfig()
        assert config.performance_drop_threshold == 0.15
        assert config.drift_psi_threshold == 0.20
        assert config.training_data_window_days == 365
        assert config.min_performance_lift == 0.02

    def test_config_accepts_custom_values(self):
        config = RetrainingConfig(
            performance_drop_threshold=0.10,
            auto_deploy=True
        )
        assert config.performance_drop_threshold == 0.10
        assert config.auto_deploy is True


class TestPerformanceDegradationTrigger:
    def test_triggers_on_pr_auc_drop(self):
        trigger = RetrainingTrigger()
        metrics = {
            "pr_auc": {"current": 0.40, "baseline": 0.55},
            "roc_auc": {"current": 0.70, "baseline": 0.75}
        }
        result = trigger.evaluate_performance(metrics)
        assert result.should_retrain is True
        assert result.trigger_type == RetrainingTriggerType.PERFORMANCE_DEGRADATION
        assert result.priority == TriggerPriority.HIGH

    def test_no_trigger_for_acceptable_performance(self):
        trigger = RetrainingTrigger()
        metrics = {
            "pr_auc": {"current": 0.52, "baseline": 0.55},
            "roc_auc": {"current": 0.73, "baseline": 0.75}
        }
        result = trigger.evaluate_performance(metrics)
        assert result.should_retrain is False


class TestDriftTrigger:
    def test_triggers_on_high_psi(self):
        trigger = RetrainingTrigger()
        drift_metrics = {
            "feature1": {"psi": 0.25},
            "feature2": {"psi": 0.15}
        }
        result = trigger.evaluate_drift(drift_metrics)
        assert result.should_retrain is True
        assert result.trigger_type == RetrainingTriggerType.SIGNIFICANT_DRIFT
        assert result.priority == TriggerPriority.HIGH

    def test_no_trigger_for_low_drift(self):
        trigger = RetrainingTrigger()
        drift_metrics = {
            "feature1": {"psi": 0.08},
            "feature2": {"psi": 0.05}
        }
        result = trigger.evaluate_drift(drift_metrics)
        assert result.should_retrain is False


class TestScheduledTrigger:
    def test_triggers_on_schedule(self):
        config = RetrainingConfig(scheduled_interval_days=90)
        trigger = RetrainingTrigger(config=config)
        last_training = datetime.now() - timedelta(days=100)
        result = trigger.evaluate_schedule(last_training_date=last_training)
        assert result.should_retrain is True
        assert result.trigger_type == RetrainingTriggerType.SCHEDULED
        assert result.priority == TriggerPriority.MEDIUM

    def test_no_trigger_before_schedule(self):
        config = RetrainingConfig(scheduled_interval_days=90)
        trigger = RetrainingTrigger(config=config)
        last_training = datetime.now() - timedelta(days=30)
        result = trigger.evaluate_schedule(last_training_date=last_training)
        assert result.should_retrain is False


class TestDataVolumeTrigger:
    def test_triggers_on_volume_increase(self):
        config = RetrainingConfig(data_volume_increase_threshold=0.50)
        trigger = RetrainingTrigger(config=config)
        result = trigger.evaluate_data_volume(
            training_data_size=10000,
            current_data_size=16000
        )
        assert result.should_retrain is True
        assert result.trigger_type == RetrainingTriggerType.DATA_VOLUME_INCREASE
        assert result.priority == TriggerPriority.MEDIUM

    def test_no_trigger_for_small_volume_increase(self):
        config = RetrainingConfig(data_volume_increase_threshold=0.50)
        trigger = RetrainingTrigger(config=config)
        result = trigger.evaluate_data_volume(
            training_data_size=10000,
            current_data_size=12000
        )
        assert result.should_retrain is False


class TestBusinessRequestTrigger:
    def test_triggers_on_business_request(self):
        trigger = RetrainingTrigger()
        result = trigger.trigger_manual(reason="New product launch requires model update")
        assert result.should_retrain is True
        assert result.trigger_type == RetrainingTriggerType.BUSINESS_REQUEST
        assert result.priority == TriggerPriority.LOW


class TestNewFeaturesTrigger:
    def test_triggers_on_new_features(self):
        trigger = RetrainingTrigger()
        result = trigger.evaluate_new_features(
            current_features=["f1", "f2", "f3", "f4", "f5"],
            new_features=["f6", "f7"]
        )
        assert result.should_retrain is True
        assert result.trigger_type == RetrainingTriggerType.NEW_FEATURES
        assert result.priority == TriggerPriority.LOW

    def test_no_trigger_without_new_features(self):
        trigger = RetrainingTrigger()
        result = trigger.evaluate_new_features(
            current_features=["f1", "f2", "f3"],
            new_features=[]
        )
        assert result.should_retrain is False


class TestRetrainingDecisionMatrix:
    def test_immediate_retrain_on_degraded_with_drift(self):
        trigger = RetrainingTrigger()
        decision = trigger.make_decision(
            performance_degraded=True,
            drift_detected=True
        )
        assert decision.action == "immediate_retrain"
        assert decision.priority == TriggerPriority.HIGH

    def test_investigate_on_performance_ok_with_drift(self):
        trigger = RetrainingTrigger()
        decision = trigger.make_decision(
            performance_degraded=False,
            drift_detected=True
        )
        assert decision.action == "investigate_and_prepare"

    def test_investigate_on_degraded_without_drift(self):
        trigger = RetrainingTrigger()
        decision = trigger.make_decision(
            performance_degraded=True,
            drift_detected=False
        )
        assert decision.action == "investigate_possible_retrain"

    def test_continue_monitoring_when_ok(self):
        trigger = RetrainingTrigger()
        decision = trigger.make_decision(
            performance_degraded=False,
            drift_detected=False
        )
        assert decision.action == "continue_monitoring"


class TestRetrainingDecision:
    def test_decision_contains_required_fields(self):
        trigger = RetrainingTrigger()
        decision = trigger.make_decision(
            performance_degraded=True,
            drift_detected=True
        )
        assert hasattr(decision, "should_retrain")
        assert hasattr(decision, "action")
        assert hasattr(decision, "priority")
        assert hasattr(decision, "reason")
        assert hasattr(decision, "timestamp")

    def test_decision_includes_reason(self):
        trigger = RetrainingTrigger()
        metrics = {
            "pr_auc": {"current": 0.35, "baseline": 0.55}
        }
        result = trigger.evaluate_performance(metrics)
        assert result.reason is not None
        assert "PR-AUC" in result.reason or "performance" in result.reason.lower()


class TestCombinedEvaluation:
    def test_evaluates_all_triggers(self):
        trigger = RetrainingTrigger()
        performance_metrics = {"pr_auc": {"current": 0.40, "baseline": 0.55}}
        drift_metrics = {"feature1": {"psi": 0.25}}
        last_training = datetime.now() - timedelta(days=100)
        result = trigger.evaluate_all(
            performance_metrics=performance_metrics,
            drift_metrics=drift_metrics,
            last_training_date=last_training,
            training_data_size=10000,
            current_data_size=12000
        )
        assert len(result.triggered_conditions) > 0
        assert result.final_decision is not None

    def test_prioritizes_highest_priority_trigger(self):
        trigger = RetrainingTrigger()
        performance_metrics = {"pr_auc": {"current": 0.40, "baseline": 0.55}}
        drift_metrics = {"feature1": {"psi": 0.08}}
        last_training = datetime.now() - timedelta(days=100)
        result = trigger.evaluate_all(
            performance_metrics=performance_metrics,
            drift_metrics=drift_metrics,
            last_training_date=last_training
        )
        assert result.final_decision.priority == TriggerPriority.HIGH


class TestApprovalWorkflow:
    def test_requires_approval_by_default(self):
        config = RetrainingConfig(approval_required=True)
        trigger = RetrainingTrigger(config=config)
        decision = trigger.make_decision(performance_degraded=True, drift_detected=True)
        assert decision.requires_approval is True

    def test_can_skip_approval_if_configured(self):
        config = RetrainingConfig(approval_required=False, auto_deploy=True)
        trigger = RetrainingTrigger(config=config)
        decision = trigger.make_decision(performance_degraded=True, drift_detected=True)
        assert decision.requires_approval is False


class TestTriggerHistory:
    def test_stores_trigger_history(self):
        trigger = RetrainingTrigger()
        metrics = {"pr_auc": {"current": 0.40, "baseline": 0.55}}
        trigger.evaluate_performance(metrics)
        trigger.evaluate_performance(metrics)
        history = trigger.get_trigger_history()
        assert len(history) == 2

    def test_queries_triggers_by_type(self):
        trigger = RetrainingTrigger()
        trigger.evaluate_performance({"pr_auc": {"current": 0.40, "baseline": 0.55}})
        trigger.evaluate_drift({"feature1": {"psi": 0.25}})
        perf_triggers = trigger.get_trigger_history(trigger_type=RetrainingTriggerType.PERFORMANCE_DEGRADATION)
        assert len(perf_triggers) == 1
