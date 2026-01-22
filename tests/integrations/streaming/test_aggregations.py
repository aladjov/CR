from datetime import datetime, timedelta
from typing import List

import pytest

from customer_retention.integrations.streaming import (
    Event,
    EventSource,
    EventType,
    FeatureComputer,
    SessionWindow,
    SlidingWindow,
    StreamState,
    TumblingWindow,
    WatermarkConfig,
    WindowAggregator,
    WindowType,
)


@pytest.fixture
def sample_events() -> List[Event]:
    base_time = datetime(2025, 1, 8, 10, 0, 0)
    return [
        Event(
            event_id=f"evt_{i}",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=base_time + timedelta(minutes=i * 5),
            event_source=EventSource.WEBSITE,
            event_properties={"page": f"/page_{i}"}
        )
        for i in range(12)
    ]


@pytest.fixture
def multi_customer_events() -> List[Event]:
    base_time = datetime(2025, 1, 8, 10, 0, 0)
    events = []
    for cust_id in ["CUST001", "CUST002", "CUST003"]:
        for i in range(5):
            events.append(Event(
                event_id=f"evt_{cust_id}_{i}",
                customer_id=cust_id,
                event_type=EventType.PAGE_VIEW,
                event_timestamp=base_time + timedelta(minutes=i * 10),
                event_source=EventSource.WEBSITE,
                event_properties={}
            ))
    return events


class TestTumblingWindow:
    def test_ac9_5_1_hour_tumbling_window_count(self, sample_events):
        window = TumblingWindow(duration_minutes=60)
        aggregator = WindowAggregator(window=window)
        result = aggregator.aggregate(sample_events, aggregation="count")
        assert result.window_type == WindowType.TUMBLING
        assert result.aggregated_value == 12

    def test_tumbling_window_groups_correctly(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event(
                event_id=f"evt_{i}",
                customer_id="CUST001",
                event_type=EventType.PAGE_VIEW,
                event_timestamp=base_time + timedelta(minutes=i * 20),
                event_source=EventSource.WEBSITE,
                event_properties={}
            )
            for i in range(6)
        ]
        window = TumblingWindow(duration_minutes=60)
        aggregator = WindowAggregator(window=window)
        results = aggregator.aggregate_by_window(events, aggregation="count")
        assert len(results) == 2
        assert results[0].aggregated_value == 3
        assert results[1].aggregated_value == 3

    def test_tumbling_window_per_customer(self, multi_customer_events):
        window = TumblingWindow(duration_minutes=60)
        aggregator = WindowAggregator(window=window)
        results = aggregator.aggregate_by_customer(multi_customer_events, aggregation="count")
        assert len(results) == 3
        for cust_id, result in results.items():
            assert result.aggregated_value == 5


class TestSlidingWindow:
    def test_ac9_5_sliding_window_24h(self, sample_events):
        window = SlidingWindow(duration_minutes=60, slide_minutes=30)
        aggregator = WindowAggregator(window=window)
        results = aggregator.aggregate_by_window(sample_events, aggregation="count")
        assert len(results) > 1
        for result in results:
            assert result.window_type == WindowType.SLIDING

    def test_sliding_window_overlap(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event(
                event_id=f"evt_{i}",
                customer_id="CUST001",
                event_type=EventType.PAGE_VIEW,
                event_timestamp=base_time + timedelta(minutes=i * 10),
                event_source=EventSource.WEBSITE,
                event_properties={}
            )
            for i in range(6)
        ]
        window = SlidingWindow(duration_minutes=30, slide_minutes=10)
        aggregator = WindowAggregator(window=window)
        results = aggregator.aggregate_by_window(events, aggregation="count")
        assert len(results) >= 3


class TestSessionWindow:
    def test_session_window_detection(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, base_time, EventSource.WEBSITE, {}),
            Event("evt_2", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=5), EventSource.WEBSITE, {}),
            Event("evt_3", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=10), EventSource.WEBSITE, {}),
            Event("evt_4", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=60), EventSource.WEBSITE, {}),
            Event("evt_5", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=65), EventSource.WEBSITE, {}),
        ]
        window = SessionWindow(gap_minutes=30)
        aggregator = WindowAggregator(window=window)
        results = aggregator.aggregate_by_window(events, aggregation="count")
        assert len(results) == 2
        assert results[0].aggregated_value == 3
        assert results[1].aggregated_value == 2

    def test_session_duration_calculation(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, base_time, EventSource.WEBSITE, {}),
            Event("evt_2", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=15), EventSource.WEBSITE, {}),
        ]
        window = SessionWindow(gap_minutes=30)
        aggregator = WindowAggregator(window=window)
        result = aggregator.compute_session_metrics(events)
        assert result.session_duration_minutes == 15


class TestWatermarkHandling:
    def test_ac9_6_late_events_within_watermark_included(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        watermark_config = WatermarkConfig(delay_minutes=10)
        events = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, base_time, EventSource.WEBSITE, {}),
            Event("evt_2", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=5), EventSource.WEBSITE, {}),
        ]
        late_event = Event(
            "evt_late",
            "CUST001",
            EventType.PAGE_VIEW,
            base_time - timedelta(minutes=5),
            EventSource.WEBSITE,
            {}
        )
        window = TumblingWindow(duration_minutes=60)
        aggregator = WindowAggregator(window=window, watermark_config=watermark_config)
        result = aggregator.aggregate(events + [late_event], aggregation="count")
        assert result.aggregated_value == 3
        assert result.late_events_count == 1

    def test_late_events_beyond_watermark_dropped(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        watermark_config = WatermarkConfig(delay_minutes=10)
        events = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, base_time, EventSource.WEBSITE, {}),
        ]
        very_late_event = Event(
            "evt_very_late",
            "CUST001",
            EventType.PAGE_VIEW,
            base_time - timedelta(minutes=30),
            EventSource.WEBSITE,
            {}
        )
        window = TumblingWindow(duration_minutes=60)
        aggregator = WindowAggregator(window=window, watermark_config=watermark_config)
        result = aggregator.aggregate(events + [very_late_event], aggregation="count")
        assert result.aggregated_value == 1
        assert result.dropped_events_count == 1


class TestStateManagement:
    def test_ac9_7_state_persisted(self):
        state = StreamState()
        state.update_customer_state("CUST001", {"page_views_1h": 10})
        state.save_checkpoint("/tmp/checkpoint")
        new_state = StreamState()
        new_state.load_checkpoint("/tmp/checkpoint")
        assert new_state.get_customer_state("CUST001")["page_views_1h"] == 10

    def test_state_incremental_update(self):
        state = StreamState()
        state.update_customer_state("CUST001", {"page_views_1h": 5})
        state.increment_customer_state("CUST001", "page_views_1h", 3)
        assert state.get_customer_state("CUST001")["page_views_1h"] == 8

    def test_state_window_expiration(self):
        state = StreamState()
        old_time = datetime.now() - timedelta(hours=2)
        state.update_customer_state("CUST001", {"page_views_1h": 10}, timestamp=old_time)
        state.expire_old_windows(max_age_minutes=60)
        cust_state = state.get_customer_state("CUST001")
        assert cust_state.get("page_views_1h") is None or cust_state.get("page_views_1h") == 0


class TestFeatureComputer:
    def test_count_features(self, sample_events):
        computer = FeatureComputer()
        features = computer.compute_count_features(sample_events, customer_id="CUST001")
        assert "page_views_1h" in features
        assert features["page_views_1h"] == 12

    def test_recency_features(self, sample_events):
        computer = FeatureComputer()
        features = computer.compute_recency_features(sample_events, customer_id="CUST001")
        assert "minutes_since_last_visit" in features
        assert features["minutes_since_last_visit"] >= 0

    def test_velocity_features(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event(f"evt_{i}", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=i * 5),
                  EventSource.WEBSITE, {})
            for i in range(12)
        ]
        computer = FeatureComputer()
        features = computer.compute_velocity_features(events, customer_id="CUST001")
        assert "visit_velocity_1h" in features
        assert features["visit_velocity_1h"] == 12.0

    def test_session_features(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event(f"evt_{i}", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=i * 3),
                  EventSource.WEBSITE, {"page": f"/page_{i}"})
            for i in range(5)
        ]
        computer = FeatureComputer()
        features = computer.compute_session_features(events, session_gap_minutes=30)
        assert "session_duration_minutes" in features
        assert "session_page_count" in features
        assert features["session_page_count"] == 5

    def test_anomaly_features(self):
        computer = FeatureComputer()
        baseline = {"page_views_1h": 10.0, "page_views_1h_std": 2.0}
        current = {"page_views_1h": 2.0}
        features = computer.compute_anomaly_features(current, baseline)
        assert "activity_anomaly_score" in features
        assert features["activity_anomaly_score"] < 0


class TestFeatureFreshness:
    def test_ac9_8_feature_freshness_under_5_minutes(self, sample_events):
        computer = FeatureComputer()
        result = computer.compute_all_features(sample_events, customer_id="CUST001")
        freshness_seconds = (datetime.now() - result.computed_at).total_seconds()
        assert freshness_seconds < 300

    def test_feature_timestamp_tracking(self, sample_events):
        computer = FeatureComputer()
        result = computer.compute_all_features(sample_events, customer_id="CUST001")
        assert result.computed_at is not None
        assert result.feature_freshness_seconds < 300


class TestAggregationTypes:
    def test_sum_aggregation(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event(f"evt_{i}", "CUST001", EventType.ORDER, base_time + timedelta(hours=i),
                  EventSource.PURCHASE, {"amount": 50.0 + i * 10})
            for i in range(3)
        ]
        window = TumblingWindow(duration_minutes=24 * 60)
        aggregator = WindowAggregator(window=window)
        result = aggregator.aggregate(events, aggregation="sum", property_key="amount")
        assert result.aggregated_value == 180.0

    def test_avg_aggregation(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event(f"evt_{i}", "CUST001", EventType.ORDER, base_time + timedelta(hours=i),
                  EventSource.PURCHASE, {"amount": 100.0})
            for i in range(4)
        ]
        window = TumblingWindow(duration_minutes=24 * 60)
        aggregator = WindowAggregator(window=window)
        result = aggregator.aggregate(events, aggregation="avg", property_key="amount")
        assert result.aggregated_value == 100.0

    def test_max_aggregation(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event("evt_1", "CUST001", EventType.ORDER, base_time, EventSource.PURCHASE, {"amount": 50.0}),
            Event("evt_2", "CUST001", EventType.ORDER, base_time + timedelta(hours=1), EventSource.PURCHASE, {"amount": 150.0}),
            Event("evt_3", "CUST001", EventType.ORDER, base_time + timedelta(hours=2), EventSource.PURCHASE, {"amount": 100.0}),
        ]
        window = TumblingWindow(duration_minutes=24 * 60)
        aggregator = WindowAggregator(window=window)
        result = aggregator.aggregate(events, aggregation="max", property_key="amount")
        assert result.aggregated_value == 150.0

    def test_min_aggregation(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event("evt_1", "CUST001", EventType.ORDER, base_time, EventSource.PURCHASE, {"amount": 50.0}),
            Event("evt_2", "CUST001", EventType.ORDER, base_time + timedelta(hours=1), EventSource.PURCHASE, {"amount": 150.0}),
        ]
        window = TumblingWindow(duration_minutes=24 * 60)
        aggregator = WindowAggregator(window=window)
        result = aggregator.aggregate(events, aggregation="min", property_key="amount")
        assert result.aggregated_value == 50.0


class TestMultiEventTypeAggregation:
    def test_aggregate_by_event_type(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, base_time, EventSource.WEBSITE, {}),
            Event("evt_2", "CUST001", EventType.PAGE_VIEW, base_time + timedelta(minutes=5), EventSource.WEBSITE, {}),
            Event("evt_3", "CUST001", EventType.CLICK, base_time + timedelta(minutes=10), EventSource.WEBSITE, {}),
            Event("evt_4", "CUST001", EventType.ORDER, base_time + timedelta(minutes=15), EventSource.PURCHASE, {}),
        ]
        window = TumblingWindow(duration_minutes=60)
        aggregator = WindowAggregator(window=window)
        results = aggregator.aggregate_by_event_type(events)
        assert results[EventType.PAGE_VIEW].aggregated_value == 2
        assert results[EventType.CLICK].aggregated_value == 1
        assert results[EventType.ORDER].aggregated_value == 1

    def test_filter_events_by_source(self):
        base_time = datetime(2025, 1, 8, 10, 0, 0)
        events = [
            Event("evt_1", "CUST001", EventType.PAGE_VIEW, base_time, EventSource.WEBSITE, {}),
            Event("evt_2", "CUST001", EventType.ORDER, base_time + timedelta(hours=1), EventSource.PURCHASE, {}),
            Event("evt_3", "CUST001", EventType.EMAIL_OPEN, base_time + timedelta(hours=2), EventSource.EMAIL, {}),
        ]
        window = TumblingWindow(duration_minutes=24 * 60)
        aggregator = WindowAggregator(window=window)
        result = aggregator.aggregate(events, aggregation="count", source_filter=EventSource.WEBSITE)
        assert result.aggregated_value == 1
