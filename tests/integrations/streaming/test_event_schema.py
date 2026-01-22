import pytest
from datetime import datetime, timedelta
from dataclasses import asdict

from customer_retention.integrations.streaming import (
    Event, EventType, EventSource, EventSchema, EventValidator,
    ValidationResult, SchemaRegistry
)


class TestEventDataclass:
    def test_create_website_event(self):
        event = Event(
            event_id="evt_001",
            customer_id="CUST123",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=datetime.now(),
            event_source=EventSource.WEBSITE,
            event_properties={"page": "/products", "duration_sec": 30},
            session_id="sess_abc",
            device_type="desktop"
        )
        assert event.event_id == "evt_001"
        assert event.event_source == EventSource.WEBSITE

    def test_create_mobile_event(self):
        event = Event(
            event_id="evt_002",
            customer_id="CUST456",
            event_type=EventType.APP_ACTION,
            event_timestamp=datetime.now(),
            event_source=EventSource.MOBILE_APP,
            event_properties={"action": "add_to_cart", "item_id": "SKU001"},
            session_id="sess_xyz",
            device_type="mobile"
        )
        assert event.event_source == EventSource.MOBILE_APP

    def test_create_email_event(self):
        event = Event(
            event_id="evt_003",
            customer_id="CUST789",
            event_type=EventType.EMAIL_OPEN,
            event_timestamp=datetime.now(),
            event_source=EventSource.EMAIL,
            event_properties={"campaign_id": "camp_123"},
            device_type="mobile"
        )
        assert event.event_source == EventSource.EMAIL

    def test_create_support_event(self):
        event = Event(
            event_id="evt_004",
            customer_id="CUST101",
            event_type=EventType.SUPPORT_TICKET,
            event_timestamp=datetime.now(),
            event_source=EventSource.SUPPORT,
            event_properties={"ticket_id": "TKT001", "severity": "high"}
        )
        assert event.event_source == EventSource.SUPPORT

    def test_create_purchase_event(self):
        event = Event(
            event_id="evt_005",
            customer_id="CUST202",
            event_type=EventType.ORDER,
            event_timestamp=datetime.now(),
            event_source=EventSource.PURCHASE,
            event_properties={"order_id": "ORD001", "amount": 150.0}
        )
        assert event.event_source == EventSource.PURCHASE

    def test_ingestion_timestamp_auto_set(self):
        before = datetime.now()
        event = Event(
            event_id="evt_006",
            customer_id="CUST303",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=datetime.now(),
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        after = datetime.now()
        assert before <= event.ingestion_timestamp <= after


class TestEventTypes:
    def test_all_website_event_types(self):
        website_types = [EventType.PAGE_VIEW, EventType.CLICK, EventType.SEARCH]
        for et in website_types:
            assert et.source_category == EventSource.WEBSITE

    def test_all_mobile_event_types(self):
        mobile_types = [EventType.APP_SESSION, EventType.APP_ACTION, EventType.APP_CRASH]
        for et in mobile_types:
            assert et.source_category == EventSource.MOBILE_APP

    def test_all_email_event_types(self):
        email_types = [EventType.EMAIL_OPEN, EventType.EMAIL_CLICK, EventType.EMAIL_UNSUBSCRIBE]
        for et in email_types:
            assert et.source_category == EventSource.EMAIL

    def test_all_support_event_types(self):
        support_types = [EventType.SUPPORT_TICKET, EventType.SUPPORT_CHAT, EventType.SUPPORT_CALL]
        for et in support_types:
            assert et.source_category == EventSource.SUPPORT

    def test_all_purchase_event_types(self):
        purchase_types = [EventType.ORDER, EventType.RETURN, EventType.REFUND]
        for et in purchase_types:
            assert et.source_category == EventSource.PURCHASE


class TestEventValidator:
    def test_ac9_2_valid_event_passes_validation(self):
        validator = EventValidator()
        event = Event(
            event_id="evt_valid",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=datetime.now(),
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        result = validator.validate(event)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_event_id_fails_validation(self):
        validator = EventValidator()
        event = Event(
            event_id="",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=datetime.now(),
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        result = validator.validate(event)
        assert result.is_valid is False
        assert "event_id" in result.errors[0].lower()

    def test_missing_customer_id_fails_validation(self):
        validator = EventValidator()
        event = Event(
            event_id="evt_001",
            customer_id="",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=datetime.now(),
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        result = validator.validate(event)
        assert result.is_valid is False
        assert "customer_id" in result.errors[0].lower()

    def test_future_timestamp_fails_validation(self):
        validator = EventValidator()
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=datetime.now() + timedelta(days=1),
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        result = validator.validate(event)
        assert result.is_valid is False
        assert "timestamp" in result.errors[0].lower()

    def test_very_old_timestamp_fails_validation(self):
        validator = EventValidator(max_age_days=30)
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=datetime.now() - timedelta(days=60),
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        result = validator.validate(event)
        assert result.is_valid is False

    def test_mismatched_event_type_source_fails(self):
        validator = EventValidator()
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.EMAIL_OPEN,
            event_timestamp=datetime.now(),
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        result = validator.validate(event)
        assert result.is_valid is False


class TestSchemaRegistry:
    def test_register_schema(self):
        registry = SchemaRegistry()
        schema = EventSchema(
            name="website_event",
            version="v1",
            required_properties=["page"],
            optional_properties=["duration_sec", "referrer"]
        )
        registry.register(schema)
        assert registry.get("website_event", "v1") is not None

    def test_get_latest_version(self):
        registry = SchemaRegistry()
        registry.register(EventSchema(name="website_event", version="v1", required_properties=[]))
        registry.register(EventSchema(name="website_event", version="v2", required_properties=["page"]))
        latest = registry.get_latest("website_event")
        assert latest.version == "v2"

    def test_validate_against_schema(self):
        registry = SchemaRegistry()
        registry.register(EventSchema(
            name="purchase_event",
            version="v1",
            required_properties=["order_id", "amount"]
        ))
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.ORDER,
            event_timestamp=datetime.now(),
            event_source=EventSource.PURCHASE,
            event_properties={"order_id": "ORD001", "amount": 99.99}
        )
        result = registry.validate_event(event, "purchase_event", "v1")
        assert result.is_valid is True

    def test_missing_required_property_fails(self):
        registry = SchemaRegistry()
        registry.register(EventSchema(
            name="purchase_event",
            version="v1",
            required_properties=["order_id", "amount"]
        ))
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.ORDER,
            event_timestamp=datetime.now(),
            event_source=EventSource.PURCHASE,
            event_properties={"order_id": "ORD001"}
        )
        result = registry.validate_event(event, "purchase_event", "v1")
        assert result.is_valid is False
        assert "amount" in result.errors[0].lower()


class TestEventSerialization:
    def test_event_to_dict(self):
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=datetime(2025, 1, 8, 10, 30, 0),
            event_source=EventSource.WEBSITE,
            event_properties={"page": "/home"}
        )
        d = event.to_dict()
        assert d["event_id"] == "evt_001"
        assert d["event_type"] == "page_view"
        assert d["event_source"] == "website"

    def test_event_from_dict(self):
        d = {
            "event_id": "evt_002",
            "customer_id": "CUST002",
            "event_type": "order",
            "event_timestamp": "2025-01-08T10:30:00",
            "event_source": "purchase",
            "event_properties": {"order_id": "ORD001"}
        }
        event = Event.from_dict(d)
        assert event.event_id == "evt_002"
        assert event.event_type == EventType.ORDER
        assert event.event_source == EventSource.PURCHASE

    def test_event_to_json(self):
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.CLICK,
            event_timestamp=datetime.now(),
            event_source=EventSource.WEBSITE,
            event_properties={"element": "button_buy"}
        )
        json_str = event.to_json()
        assert '"event_id": "evt_001"' in json_str
        assert '"element": "button_buy"' in json_str


class TestEventLatency:
    def test_ac9_3_latency_calculation(self):
        event_time = datetime.now() - timedelta(seconds=30)
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=event_time,
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        latency_seconds = event.ingestion_latency_seconds
        assert 29 <= latency_seconds <= 32

    def test_latency_within_requirement(self):
        event_time = datetime.now() - timedelta(seconds=45)
        event = Event(
            event_id="evt_001",
            customer_id="CUST001",
            event_type=EventType.PAGE_VIEW,
            event_timestamp=event_time,
            event_source=EventSource.WEBSITE,
            event_properties={}
        )
        assert event.ingestion_latency_seconds < 60


class TestEventBatch:
    def test_validate_batch_all_valid(self):
        validator = EventValidator()
        events = [
            Event(
                event_id=f"evt_{i}",
                customer_id=f"CUST{i}",
                event_type=EventType.PAGE_VIEW,
                event_timestamp=datetime.now(),
                event_source=EventSource.WEBSITE,
                event_properties={}
            )
            for i in range(100)
        ]
        results = validator.validate_batch(events)
        assert results.total_count == 100
        assert results.valid_count == 100
        assert results.invalid_count == 0

    def test_ac9_4_no_data_loss_in_validation(self):
        validator = EventValidator()
        events = [
            Event(
                event_id=f"evt_{i}",
                customer_id=f"CUST{i}",
                event_type=EventType.PAGE_VIEW,
                event_timestamp=datetime.now(),
                event_source=EventSource.WEBSITE,
                event_properties={}
            )
            for i in range(1000)
        ]
        results = validator.validate_batch(events)
        assert results.valid_count + results.invalid_count == len(events)

    def test_batch_with_some_invalid(self):
        validator = EventValidator()
        valid_events = [
            Event(
                event_id=f"evt_{i}",
                customer_id=f"CUST{i}",
                event_type=EventType.PAGE_VIEW,
                event_timestamp=datetime.now(),
                event_source=EventSource.WEBSITE,
                event_properties={}
            )
            for i in range(90)
        ]
        invalid_events = [
            Event(
                event_id="",
                customer_id=f"CUST{i}",
                event_type=EventType.PAGE_VIEW,
                event_timestamp=datetime.now(),
                event_source=EventSource.WEBSITE,
                event_properties={}
            )
            for i in range(10)
        ]
        results = validator.validate_batch(valid_events + invalid_events)
        assert results.valid_count == 90
        assert results.invalid_count == 10
        assert len(results.invalid_events) == 10
