import pytest
from datetime import datetime
import pandas as pd
import numpy as np


class TestIterationSignal:
    def test_signal_values(self):
        from customer_retention.integrations.iteration.signals import IterationSignal
        assert IterationSignal.DRIFT_CRITICAL.value == "drift_critical"
        assert IterationSignal.DRIFT_WARNING.value == "drift_warning"
        assert IterationSignal.PERFORMANCE_CRITICAL.value == "performance_critical"
        assert IterationSignal.PERFORMANCE_WARNING.value == "performance_warning"
        assert IterationSignal.DATA_QUALITY_ISSUE.value == "data_quality_issue"
        assert IterationSignal.SCHEDULED_RETRAIN.value == "scheduled_retrain"
        assert IterationSignal.MANUAL_TRIGGER.value == "manual_trigger"


class TestSignalEvent:
    def test_create_signal_event(self):
        from customer_retention.integrations.iteration.signals import SignalEvent, IterationSignal
        event = SignalEvent(
            signal_type=IterationSignal.DRIFT_CRITICAL,
            source="drift_detector",
            severity="critical",
            details={"feature": "age", "psi": 0.25},
            recommended_action="retrain"
        )
        assert event.signal_type == IterationSignal.DRIFT_CRITICAL
        assert event.source == "drift_detector"
        assert event.recommended_action == "retrain"
        assert event.timestamp is not None

    def test_signal_event_to_dict(self):
        from customer_retention.integrations.iteration.signals import SignalEvent, IterationSignal
        event = SignalEvent(
            signal_type=IterationSignal.PERFORMANCE_WARNING,
            source="performance_monitor",
            severity="warning",
            details={"roc_auc_drop": 0.05},
            recommended_action="investigate"
        )
        data = event.to_dict()
        assert data["signal_type"] == "performance_warning"
        assert data["source"] == "performance_monitor"
        assert "timestamp" in data

    def test_signal_event_from_dict(self):
        from customer_retention.integrations.iteration.signals import SignalEvent, IterationSignal
        data = {
            "signal_type": "drift_warning",
            "source": "drift_detector",
            "severity": "warning",
            "details": {"feature": "income"},
            "recommended_action": "monitor",
            "timestamp": "2024-01-15T10:00:00"
        }
        event = SignalEvent.from_dict(data)
        assert event.signal_type == IterationSignal.DRIFT_WARNING
        assert event.source == "drift_detector"


class TestSignalAggregator:
    @pytest.fixture
    def sample_reference_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            "age": np.random.normal(35, 10, 100),
            "income": np.random.normal(60000, 15000, 100),
            "tenure": np.random.normal(24, 12, 100)
        })

    @pytest.fixture
    def sample_current_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            "age": np.random.normal(35, 10, 100),
            "income": np.random.normal(60000, 15000, 100),
            "tenure": np.random.normal(24, 12, 100)
        })

    @pytest.fixture
    def drifted_current_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            "age": np.random.normal(45, 15, 100),  # Significant drift
            "income": np.random.normal(80000, 20000, 100),  # Significant drift
            "tenure": np.random.normal(24, 12, 100)
        })

    def test_create_aggregator(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator
        aggregator = SignalAggregator()
        assert aggregator is not None

    def test_create_aggregator_with_monitors(self, sample_reference_data):
        from customer_retention.integrations.iteration.signals import SignalAggregator
        from customer_retention.stages.monitoring.drift_detector import DriftDetector
        from customer_retention.stages.monitoring.performance_monitor import PerformanceMonitor

        drift_detector = DriftDetector(reference_data=sample_reference_data)
        perf_monitor = PerformanceMonitor(baseline_metrics={"roc_auc": 0.85})

        aggregator = SignalAggregator(
            drift_detector=drift_detector,
            performance_monitor=perf_monitor
        )
        assert aggregator.drift_detector is not None
        assert aggregator.performance_monitor is not None

    def test_check_drift_signals_no_drift(self, sample_reference_data, sample_current_data):
        from customer_retention.integrations.iteration.signals import SignalAggregator
        from customer_retention.stages.monitoring.drift_detector import DriftDetector

        drift_detector = DriftDetector(reference_data=sample_reference_data)
        aggregator = SignalAggregator(drift_detector=drift_detector)

        signals = aggregator.check_drift_signals(sample_current_data)
        critical_signals = [s for s in signals if "critical" in s.severity.lower()]
        assert len(critical_signals) == 0

    def test_check_drift_signals_with_drift(self, sample_reference_data, drifted_current_data):
        from customer_retention.integrations.iteration.signals import SignalAggregator, IterationSignal
        from customer_retention.stages.monitoring.drift_detector import DriftDetector

        drift_detector = DriftDetector(reference_data=sample_reference_data)
        aggregator = SignalAggregator(drift_detector=drift_detector)

        signals = aggregator.check_drift_signals(drifted_current_data)
        # Should detect drift in age and income
        assert len(signals) > 0

    def test_check_performance_signals_ok(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator
        from customer_retention.stages.monitoring.performance_monitor import PerformanceMonitor

        perf_monitor = PerformanceMonitor(baseline_metrics={"roc_auc": 0.85, "pr_auc": 0.70})
        aggregator = SignalAggregator(performance_monitor=perf_monitor)

        current_metrics = {"roc_auc": 0.84, "pr_auc": 0.69}
        signals = aggregator.check_performance_signals(current_metrics)
        # Small drop, should be OK
        critical = [s for s in signals if "critical" in s.severity.lower()]
        assert len(critical) == 0

    def test_check_performance_signals_warning(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator, IterationSignal
        from customer_retention.stages.monitoring.performance_monitor import PerformanceMonitor

        perf_monitor = PerformanceMonitor(baseline_metrics={"roc_auc": 0.85, "pr_auc": 0.70})
        aggregator = SignalAggregator(performance_monitor=perf_monitor)

        current_metrics = {"roc_auc": 0.75, "pr_auc": 0.60}  # 10%+ drop
        signals = aggregator.check_performance_signals(current_metrics)
        assert len(signals) > 0
        warning_signals = [s for s in signals if s.signal_type == IterationSignal.PERFORMANCE_WARNING]
        assert len(warning_signals) > 0

    def test_check_performance_signals_critical(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator, IterationSignal
        from customer_retention.stages.monitoring.performance_monitor import PerformanceMonitor

        perf_monitor = PerformanceMonitor(baseline_metrics={"roc_auc": 0.85})
        aggregator = SignalAggregator(performance_monitor=perf_monitor)

        current_metrics = {"roc_auc": 0.68}  # 20%+ drop
        signals = aggregator.check_performance_signals(current_metrics)
        critical = [s for s in signals if s.signal_type == IterationSignal.PERFORMANCE_CRITICAL]
        assert len(critical) > 0

    def test_add_manual_signal(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator, IterationSignal
        aggregator = SignalAggregator()

        aggregator.add_manual_signal(
            reason="User requested model update",
            details={"user": "data_scientist"}
        )
        events = aggregator.get_pending_signals()
        assert len(events) == 1
        assert events[0].signal_type == IterationSignal.MANUAL_TRIGGER

    def test_add_scheduled_signal(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator, IterationSignal
        aggregator = SignalAggregator()

        aggregator.add_scheduled_signal(schedule_name="weekly_retrain")
        events = aggregator.get_pending_signals()
        assert len(events) == 1
        assert events[0].signal_type == IterationSignal.SCHEDULED_RETRAIN

    def test_should_trigger_iteration_no_signals(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator
        aggregator = SignalAggregator()

        should_trigger, trigger = aggregator.should_trigger_iteration()
        assert should_trigger is False
        assert trigger is None

    def test_should_trigger_iteration_with_critical(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator, SignalEvent, IterationSignal
        from customer_retention.integrations.iteration.context import IterationTrigger
        aggregator = SignalAggregator()

        # Add a critical signal
        event = SignalEvent(
            signal_type=IterationSignal.DRIFT_CRITICAL,
            source="drift_detector",
            severity="critical",
            details={},
            recommended_action="retrain"
        )
        aggregator._pending_signals.append(event)

        should_trigger, trigger = aggregator.should_trigger_iteration()
        assert should_trigger is True
        assert trigger == IterationTrigger.DRIFT_DETECTED

    def test_should_trigger_iteration_performance_drop(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator, SignalEvent, IterationSignal
        from customer_retention.integrations.iteration.context import IterationTrigger
        aggregator = SignalAggregator()

        event = SignalEvent(
            signal_type=IterationSignal.PERFORMANCE_CRITICAL,
            source="performance_monitor",
            severity="critical",
            details={},
            recommended_action="retrain"
        )
        aggregator._pending_signals.append(event)

        should_trigger, trigger = aggregator.should_trigger_iteration()
        assert should_trigger is True
        assert trigger == IterationTrigger.PERFORMANCE_DROP

    def test_clear_signals(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator
        aggregator = SignalAggregator()

        aggregator.add_manual_signal("Test", {})
        aggregator.add_manual_signal("Test 2", {})
        assert len(aggregator.get_pending_signals()) == 2

        aggregator.clear_signals()
        assert len(aggregator.get_pending_signals()) == 0

    def test_check_all_signals(self, sample_reference_data, drifted_current_data):
        from customer_retention.integrations.iteration.signals import SignalAggregator
        from customer_retention.stages.monitoring.drift_detector import DriftDetector
        from customer_retention.stages.monitoring.performance_monitor import PerformanceMonitor

        drift_detector = DriftDetector(reference_data=sample_reference_data)
        perf_monitor = PerformanceMonitor(baseline_metrics={"roc_auc": 0.85})

        aggregator = SignalAggregator(
            drift_detector=drift_detector,
            performance_monitor=perf_monitor
        )

        signals = aggregator.check_all_signals(
            current_data=drifted_current_data,
            current_metrics={"roc_auc": 0.70}
        )
        # Should have both drift and performance signals
        assert len(signals) > 0

    def test_get_signal_summary(self):
        from customer_retention.integrations.iteration.signals import SignalAggregator, SignalEvent, IterationSignal
        aggregator = SignalAggregator()

        aggregator._pending_signals.append(SignalEvent(
            signal_type=IterationSignal.DRIFT_WARNING,
            source="drift", severity="warning", details={}, recommended_action="monitor"
        ))
        aggregator._pending_signals.append(SignalEvent(
            signal_type=IterationSignal.DRIFT_CRITICAL,
            source="drift", severity="critical", details={}, recommended_action="retrain"
        ))
        aggregator._pending_signals.append(SignalEvent(
            signal_type=IterationSignal.PERFORMANCE_WARNING,
            source="perf", severity="warning", details={}, recommended_action="investigate"
        ))

        summary = aggregator.get_signal_summary()
        assert summary["total"] == 3
        assert summary["critical"] == 1
        assert summary["warning"] == 2
