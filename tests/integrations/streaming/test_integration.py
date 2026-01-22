import pytest
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock, patch

from customer_retention.integrations.streaming import (
    Event, EventType, EventSource, EventValidator,
    WindowAggregator, TumblingWindow, FeatureComputer,
    OnlineFeatureStore, FeatureStoreConfig, FeatureRecord,
    EarlyWarningModel, EarlyWarningConfig, WarningLevel,
    RealtimeScorer, ScoringRequest, ScoringConfig,
    TriggerEngine, TriggerConfig, ThresholdTrigger, ActionType,
    BatchStreamingBridge, ScoreCombinationStrategy
)


@pytest.fixture
def event_stream() -> List[Event]:
    base_time = datetime.now() - timedelta(hours=1)
    events = []
    for i in range(20):
        events.append(Event(
            event_id=f"evt_{i}",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=base_time + timedelta(minutes=i * 3),
            event_source=EventSource.WEBSITE,
            event_properties={"page": f"/page_{i % 5}"}
        ))
    events.append(Event(
        event_id="evt_order",
        customer_id="CUST001",
        event_type=EventType.ORDER,
        event_timestamp=base_time + timedelta(minutes=30),
        event_source=EventSource.PURCHASE,
        event_properties={"order_id": "ORD001", "amount": 150.0}
    ))
    events.append(Event(
        event_id="evt_support",
        customer_id="CUST001",
        event_type=EventType.SUPPORT_TICKET,
        event_timestamp=base_time + timedelta(minutes=45),
        event_source=EventSource.SUPPORT,
        event_properties={"ticket_id": "TKT001", "severity": "medium"}
    ))
    return events


@pytest.fixture
def feature_store():
    config = FeatureStoreConfig(backend="simulation")
    return OnlineFeatureStore(config=config)


@pytest.fixture
def early_warning_model():
    return EarlyWarningModel(config=EarlyWarningConfig())


class TestEndToEndEventProcessing:
    def test_event_to_features_to_scoring(self, event_stream, feature_store, early_warning_model):
        validator = EventValidator()
        batch_result = validator.validate_batch(event_stream)
        assert batch_result.valid_count == len(event_stream)
        computer = FeatureComputer()
        features = computer.compute_all_features(event_stream, customer_id="CUST001")
        assert features.features["page_views_1h"] > 0
        for name, value in features.features.items():
            feature_store.write(FeatureRecord(
                customer_id="CUST001",
                feature_name=name,
                feature_value=value
            ))
        warning_features = {
            "activity_drop_7d": 0.0,
            "days_since_last_order": 0,
            "support_tickets_7d": 1,
            "email_unsubscribe": 0,
            "payment_failure": 0
        }
        warning_result = early_warning_model.predict("CUST001", warning_features)
        assert warning_result.warning_level in [WarningLevel.LOW, WarningLevel.MEDIUM]

    def test_full_pipeline_high_risk_customer(self, feature_store, early_warning_model):
        high_risk_features = {
            "page_views_1h": 0.0,
            "activity_drop_7d": 0.75,
            "days_since_last_order": 30,
            "support_tickets_7d": 5,
            "email_unsubscribe": 1,
            "payment_failure": 1
        }
        for name, value in high_risk_features.items():
            feature_store.write(FeatureRecord(
                customer_id="CUST_HIGH_RISK",
                feature_name=name,
                feature_value=float(value)
            ))
        warning_result = early_warning_model.predict("CUST_HIGH_RISK", high_risk_features)
        assert warning_result.warning_level in [WarningLevel.HIGH, WarningLevel.CRITICAL]
        trigger_engine = TriggerEngine(config=TriggerConfig())
        trigger_engine.register_trigger(ThresholdTrigger(
            name="HIGH_RISK_ALERT",
            threshold=0.70,
            action=ActionType.IMMEDIATE_ALERT
        ))
        trigger_result = trigger_engine.evaluate(warning_result)
        assert trigger_result.triggered is True


class TestBatchStreamingIntegration:
    def test_combine_batch_and_streaming_scores(self):
        bridge = BatchStreamingBridge()
        batch_score = 0.65
        streaming_score = 0.75
        combined = bridge.combine_scores(
            batch_score=batch_score,
            streaming_score=streaming_score,
            strategy=ScoreCombinationStrategy.MAXIMUM
        )
        assert combined == 0.75

    def test_weighted_average_combination(self):
        bridge = BatchStreamingBridge()
        combined = bridge.combine_scores(
            batch_score=0.60,
            streaming_score=0.80,
            strategy=ScoreCombinationStrategy.ENSEMBLE,
            weights={"batch": 0.4, "streaming": 0.6}
        )
        expected = 0.60 * 0.4 + 0.80 * 0.6
        assert combined == pytest.approx(expected, abs=0.01)

    def test_streaming_override_when_fresher(self):
        bridge = BatchStreamingBridge()
        batch_timestamp = datetime.now() - timedelta(hours=6)
        streaming_timestamp = datetime.now() - timedelta(minutes=5)
        combined = bridge.combine_scores(
            batch_score=0.60,
            streaming_score=0.80,
            strategy=ScoreCombinationStrategy.STREAMING_OVERRIDE,
            batch_timestamp=batch_timestamp,
            streaming_timestamp=streaming_timestamp,
            freshness_threshold_hours=1
        )
        assert combined == 0.80

    def test_fallback_to_batch_when_streaming_unavailable(self):
        bridge = BatchStreamingBridge()
        combined = bridge.combine_scores(
            batch_score=0.65,
            streaming_score=None,
            strategy=ScoreCombinationStrategy.STREAMING_OVERRIDE
        )
        assert combined == 0.65


class TestFeatureMapping:
    def test_batch_to_streaming_feature_mapping(self):
        bridge = BatchStreamingBridge()
        batch_features = {
            "tenure_days": 365,
            "days_since_last_order": 10,
            "email_engagement_score": 0.75,
            "order_frequency": 2.5,
            "service_adoption_score": 0.8
        }
        streaming_features = {
            "minutes_since_last_order": 300,
            "email_opens_7d": 5,
            "emails_sent_7d": 7,
            "orders_7d": 2
        }
        mapped = bridge.map_features(batch_features, streaming_features)
        assert "tenure_days" in mapped
        assert "minutes_since_last_order" in mapped

    def test_streaming_overrides_batch_recency(self):
        bridge = BatchStreamingBridge()
        batch_features = {"days_since_last_order": 10}
        streaming_features = {"minutes_since_last_order": 30}
        mapped = bridge.map_features(batch_features, streaming_features, prefer_streaming_recency=True)
        assert mapped["days_since_last_order"] == pytest.approx(30 / (24 * 60), abs=0.01)


class TestFallbackMechanisms:
    def test_fallback_to_batch_score_on_store_failure(self):
        failing_store = MagicMock()
        failing_store.read_batch = MagicMock(side_effect=Exception("Store unavailable"))
        model = MagicMock()
        model.predict_proba = MagicMock(return_value=[[0.3, 0.7]])
        batch_scores = {"CUST001": 0.55}
        scorer = RealtimeScorer(
            model=model,
            feature_store=failing_store,
            fallback_scores=batch_scores
        )
        response = scorer.score(ScoringRequest(customer_id="CUST001"))
        assert response.churn_probability == 0.55
        assert response.is_fallback is True

    def test_fallback_to_cached_features(self, feature_store):
        feature_store.write(FeatureRecord(
            customer_id="CUST001",
            feature_name="page_views_1h",
            feature_value=15.0
        ))
        cached = feature_store.read("CUST001", "page_views_1h")
        assert cached.feature_value == 15.0

    def test_graceful_degradation_chain(self):
        bridge = BatchStreamingBridge()
        result = bridge.get_best_available_score(
            realtime_score=None,
            streaming_score=None,
            batch_score=0.65,
            cached_score=0.60
        )
        assert result.score == 0.65
        assert result.source == "batch"


class TestAlertManagerIntegration:
    def test_streaming_warnings_create_alerts(self, early_warning_model):
        from customer_retention.stages.monitoring import AlertManager, AlertLevel
        features = {
            "activity_drop_7d": 0.80,
            "days_since_last_order": 25,
            "support_tickets_7d": 4,
            "email_unsubscribe": 1,
            "payment_failure": 0
        }
        warning = early_warning_model.predict("CUST001", features)
        assert warning.warning_level in [WarningLevel.HIGH, WarningLevel.CRITICAL]
        alert_manager = AlertManager()
        alert = warning.to_alert()
        alert_manager.add_alert(alert)
        pending = alert_manager.get_pending_alerts()
        assert len(pending) >= 1
        assert any("CUST001" in str(a.message) for a in pending)

    def test_trigger_results_integrate_with_alerts(self):
        from customer_retention.stages.monitoring import AlertManager
        from customer_retention.integrations.streaming import WarningResult, SignalType
        warning = WarningResult(
            customer_id="CUST002",
            warning_score=0.92,
            warning_level=WarningLevel.CRITICAL,
            warning_signals=[SignalType.PAYMENT_ISSUE],
            primary_signal=SignalType.PAYMENT_ISSUE,
            timestamp=datetime.now(),
            recommended_action="immediate_call"
        )
        trigger_engine = TriggerEngine(config=TriggerConfig())
        trigger_engine.register_trigger(ThresholdTrigger(
            name="CRITICAL_ALERT",
            threshold=0.90,
            action=ActionType.IMMEDIATE_ALERT
        ))
        result = trigger_engine.evaluate(warning)
        assert result.triggered is True
        alert_manager = AlertManager()
        alert = result.to_alert()
        alert_manager.add_alert(alert)
        critical_alerts = [a for a in alert_manager.get_pending_alerts() if "CRITICAL" in str(a.level)]
        assert len(critical_alerts) >= 1


class TestStreamProcessor:
    def test_process_event_batch(self, event_stream):
        from customer_retention.integrations.streaming import StreamProcessor, ProcessingConfig
        config = ProcessingConfig(
            checkpoint_interval_seconds=60,
            watermark_delay_minutes=10
        )
        processor = StreamProcessor(config=config)
        results = processor.process_batch(event_stream)
        assert results.events_processed == len(event_stream)
        assert results.features_computed > 0

    def test_incremental_processing(self):
        from customer_retention.integrations.streaming import StreamProcessor, ProcessingConfig
        processor = StreamProcessor(config=ProcessingConfig())
        batch1 = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, datetime.now() - timedelta(minutes=10),
                  EventSource.WEBSITE, {})
        ]
        processor.process_batch(batch1)
        state1 = processor.get_state("CUST001")
        assert state1["page_views_1h"] == 1
        batch2 = [
            Event("evt_2", "CUST001", EventType.PAGE_VIEW, datetime.now(),
                  EventSource.WEBSITE, {})
        ]
        processor.process_batch(batch2)
        state2 = processor.get_state("CUST001")
        assert state2["page_views_1h"] == 2


class TestDatabricksCompatibility:
    def test_spark_schema_generation(self):
        pytest.importorskip("pyspark", reason="PySpark not installed")
        from customer_retention.integrations.streaming import Event, EventSchema
        spark_schema = Event.to_spark_schema()
        assert "event_id" in [f.name for f in spark_schema.fields]
        assert "customer_id" in [f.name for f in spark_schema.fields]
        assert "event_timestamp" in [f.name for f in spark_schema.fields]

    def test_feature_table_schema(self, feature_store):
        schema = feature_store.get_feature_table_schema()
        required_cols = ["customer_id", "feature_name", "feature_value", "updated_at"]
        for col in required_cols:
            assert col in schema

    def test_delta_lake_checkpoint_format(self):
        from customer_retention.integrations.streaming import StreamState
        state = StreamState()
        state.update_customer_state("CUST001", {"page_views_1h": 10})
        delta_format = state.to_delta_checkpoint_format()
        assert delta_format is not None
        assert "customer_id" in delta_format.columns if hasattr(delta_format, 'columns') else True


class TestMonitoringMetrics:
    def test_processing_latency_tracked(self, event_stream):
        from customer_retention.integrations.streaming import StreamProcessor, ProcessingConfig
        processor = StreamProcessor(config=ProcessingConfig())
        processor.process_batch(event_stream)
        metrics = processor.get_metrics()
        assert metrics.avg_processing_latency_ms >= 0
        assert metrics.events_per_second >= 0

    def test_feature_freshness_tracked(self, feature_store):
        feature_store.write(FeatureRecord(
            customer_id="CUST001",
            feature_name="page_views_1h",
            feature_value=10.0
        ))
        metrics = feature_store.get_freshness_metrics()
        assert metrics.avg_freshness_seconds >= 0

    def test_scoring_metrics_tracked(self):
        model = MagicMock()
        model.predict_proba = MagicMock(return_value=[[0.3, 0.7]])
        store = MagicMock()
        store.read_batch = MagicMock(return_value={"page_views_1h": 10.0})
        scorer = RealtimeScorer(model=model, feature_store=store)
        for i in range(10):
            scorer.score(ScoringRequest(customer_id=f"CUST{i:03d}"))
        metrics = scorer.get_metrics()
        assert metrics.total_requests == 10
        assert metrics.avg_latency_ms >= 0
