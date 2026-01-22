import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class EventSource(Enum):
    WEBSITE = "website"
    MOBILE_APP = "mobile_app"
    EMAIL = "email"
    SUPPORT = "support"
    PURCHASE = "purchase"


class EventType(Enum):
    PAGE_VIEW = "page_view"
    CLICK = "click"
    SEARCH = "search"
    APP_SESSION = "app_session"
    APP_ACTION = "app_action"
    APP_CRASH = "app_crash"
    EMAIL_OPEN = "email_open"
    EMAIL_CLICK = "email_click"
    EMAIL_UNSUBSCRIBE = "email_unsubscribe"
    SUPPORT_TICKET = "support_ticket"
    SUPPORT_CHAT = "support_chat"
    SUPPORT_CALL = "support_call"
    ORDER = "order"
    RETURN = "return"
    REFUND = "refund"

    @property
    def source_category(self) -> EventSource:
        mapping = {
            EventType.PAGE_VIEW: EventSource.WEBSITE,
            EventType.CLICK: EventSource.WEBSITE,
            EventType.SEARCH: EventSource.WEBSITE,
            EventType.APP_SESSION: EventSource.MOBILE_APP,
            EventType.APP_ACTION: EventSource.MOBILE_APP,
            EventType.APP_CRASH: EventSource.MOBILE_APP,
            EventType.EMAIL_OPEN: EventSource.EMAIL,
            EventType.EMAIL_CLICK: EventSource.EMAIL,
            EventType.EMAIL_UNSUBSCRIBE: EventSource.EMAIL,
            EventType.SUPPORT_TICKET: EventSource.SUPPORT,
            EventType.SUPPORT_CHAT: EventSource.SUPPORT,
            EventType.SUPPORT_CALL: EventSource.SUPPORT,
            EventType.ORDER: EventSource.PURCHASE,
            EventType.RETURN: EventSource.PURCHASE,
            EventType.REFUND: EventSource.PURCHASE,
        }
        return mapping[self]


@dataclass
class Event:
    event_id: str
    customer_id: str
    event_type: EventType
    event_timestamp: datetime
    event_source: EventSource
    event_properties: Dict[str, Any]
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    ingestion_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def ingestion_latency_seconds(self) -> float:
        return (self.ingestion_timestamp - self.event_timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "customer_id": self.customer_id,
            "event_type": self.event_type.value,
            "event_timestamp": self.event_timestamp.isoformat(),
            "event_source": self.event_source.value,
            "event_properties": self.event_properties,
            "session_id": self.session_id,
            "device_type": self.device_type,
            "ingestion_timestamp": self.ingestion_timestamp.isoformat()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        event_type = EventType(data["event_type"]) if isinstance(data["event_type"], str) else data["event_type"]
        event_source = EventSource(data["event_source"]) if isinstance(data["event_source"], str) else data["event_source"]
        event_timestamp = datetime.fromisoformat(data["event_timestamp"]) if isinstance(data["event_timestamp"], str) else data["event_timestamp"]
        ingestion_timestamp = datetime.fromisoformat(data.get("ingestion_timestamp", datetime.now().isoformat())) if isinstance(data.get("ingestion_timestamp"), str) else data.get("ingestion_timestamp", datetime.now())
        return cls(
            event_id=data["event_id"],
            customer_id=data["customer_id"],
            event_type=event_type,
            event_timestamp=event_timestamp,
            event_source=event_source,
            event_properties=data.get("event_properties", {}),
            session_id=data.get("session_id"),
            device_type=data.get("device_type"),
            ingestion_timestamp=ingestion_timestamp
        )

    @staticmethod
    def to_spark_schema():
        try:
            from pyspark.sql.types import MapType, StringType, StructField, StructType, TimestampType
            return StructType([
                StructField("event_id", StringType(), False),
                StructField("customer_id", StringType(), False),
                StructField("event_type", StringType(), False),
                StructField("event_timestamp", TimestampType(), False),
                StructField("event_source", StringType(), False),
                StructField("event_properties", MapType(StringType(), StringType()), True),
                StructField("session_id", StringType(), True),
                StructField("device_type", StringType(), True),
                StructField("ingestion_timestamp", TimestampType(), False)
            ])
        except ImportError:
            return None


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class BatchValidationResult:
    total_count: int
    valid_count: int
    invalid_count: int
    invalid_events: List[Event] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class EventValidator:
    def __init__(self, max_age_days: int = 30, allow_future: bool = False):
        self._max_age_days = max_age_days
        self._allow_future = allow_future

    def validate(self, event: Event) -> ValidationResult:
        errors = []
        if not event.event_id or event.event_id.strip() == "":
            errors.append("event_id is required")
        if not event.customer_id or event.customer_id.strip() == "":
            errors.append("customer_id is required")
        if not self._allow_future and event.event_timestamp > datetime.now():
            errors.append("event_timestamp cannot be in the future")
        if event.event_timestamp < datetime.now() - timedelta(days=self._max_age_days):
            errors.append(f"event_timestamp is older than {self._max_age_days} days")
        if event.event_type.source_category != event.event_source:
            errors.append(f"event_type {event.event_type.value} does not match event_source {event.event_source.value}")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_batch(self, events: List[Event]) -> BatchValidationResult:
        valid_count = 0
        invalid_count = 0
        invalid_events = []
        all_errors = []
        for event in events:
            result = self.validate(event)
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_events.append(event)
                all_errors.extend(result.errors)
        return BatchValidationResult(
            total_count=len(events),
            valid_count=valid_count,
            invalid_count=invalid_count,
            invalid_events=invalid_events,
            errors=all_errors
        )


@dataclass
class EventSchema:
    name: str
    version: str
    required_properties: List[str]
    optional_properties: List[str] = field(default_factory=list)


class SchemaRegistry:
    def __init__(self):
        self._schemas: Dict[str, Dict[str, EventSchema]] = {}

    def register(self, schema: EventSchema):
        if schema.name not in self._schemas:
            self._schemas[schema.name] = {}
        self._schemas[schema.name][schema.version] = schema

    def get(self, name: str, version: str) -> Optional[EventSchema]:
        return self._schemas.get(name, {}).get(version)

    def get_latest(self, name: str) -> Optional[EventSchema]:
        if name not in self._schemas:
            return None
        versions = sorted(self._schemas[name].keys())
        return self._schemas[name][versions[-1]] if versions else None

    def validate_event(self, event: Event, schema_name: str, version: str) -> ValidationResult:
        schema = self.get(schema_name, version)
        if not schema:
            return ValidationResult(is_valid=False, errors=[f"Schema {schema_name}:{version} not found"])
        errors = []
        for prop in schema.required_properties:
            if prop not in event.event_properties:
                errors.append(f"Required property '{prop}' is missing")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
