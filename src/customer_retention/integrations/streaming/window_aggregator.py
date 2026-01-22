import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from customer_retention.core.compat import DataFrame, pd

from .event_schema import Event, EventSource, EventType


class WindowType(Enum):
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    GLOBAL = "global"


@dataclass
class Window:
    window_type: WindowType = field(default=WindowType.GLOBAL)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class TumblingWindow(Window):
    duration_minutes: int = 60
    window_type: WindowType = field(default=WindowType.TUMBLING)


@dataclass
class SlidingWindow(Window):
    duration_minutes: int = 60
    slide_minutes: int = 30
    window_type: WindowType = field(default=WindowType.SLIDING)


@dataclass
class SessionWindow(Window):
    gap_minutes: int = 30
    window_type: WindowType = field(default=WindowType.SESSION)


@dataclass
class WatermarkConfig:
    delay_minutes: int = 60
    enabled: bool = True


@dataclass
class AggregationResult:
    window_type: WindowType
    aggregated_value: float
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    event_count: int = 0
    late_events_count: int = 0
    dropped_events_count: int = 0


@dataclass
class SessionMetrics:
    session_duration_minutes: float = 0.0
    session_page_count: int = 0
    session_action_count: int = 0
    session_idle_time: float = 0.0


class WindowAggregator:
    def __init__(self, window: Window, watermark_config: Optional[WatermarkConfig] = None):
        self._window = window
        self._watermark_config = watermark_config or WatermarkConfig()

    def aggregate(self, events: List[Event], aggregation: str = "count",
                  property_key: Optional[str] = None, source_filter: Optional[EventSource] = None) -> AggregationResult:
        filtered_events = events
        if source_filter:
            filtered_events = [e for e in events if e.event_source == source_filter]
        valid_events, late_events, dropped_events = self._separate_by_watermark(filtered_events)
        value = self._compute_aggregation(valid_events + late_events, aggregation, property_key)
        return AggregationResult(
            window_type=self._window.window_type,
            aggregated_value=value,
            event_count=len(valid_events) + len(late_events),
            late_events_count=len(late_events),
            dropped_events_count=len(dropped_events)
        )

    def aggregate_by_window(self, events: List[Event], aggregation: str = "count") -> List[AggregationResult]:
        if isinstance(self._window, TumblingWindow):
            return self._aggregate_tumbling(events, aggregation)
        elif isinstance(self._window, SlidingWindow):
            return self._aggregate_sliding(events, aggregation)
        elif isinstance(self._window, SessionWindow):
            return self._aggregate_session(events, aggregation)
        return [self.aggregate(events, aggregation)]

    def aggregate_by_customer(self, events: List[Event], aggregation: str = "count") -> Dict[str, AggregationResult]:
        by_customer = defaultdict(list)
        for event in events:
            by_customer[event.customer_id].append(event)
        return {cust: self.aggregate(evts, aggregation) for cust, evts in by_customer.items()}

    def aggregate_by_event_type(self, events: List[Event]) -> Dict[EventType, AggregationResult]:
        by_type = defaultdict(list)
        for event in events:
            by_type[event.event_type].append(event)
        return {etype: self.aggregate(evts, "count") for etype, evts in by_type.items()}

    def compute_session_metrics(self, events: List[Event]) -> SessionMetrics:
        if not events:
            return SessionMetrics()
        sorted_events = sorted(events, key=lambda e: e.event_timestamp)
        duration = (sorted_events[-1].event_timestamp - sorted_events[0].event_timestamp).total_seconds() / 60
        page_count = sum(1 for e in events if e.event_type == EventType.PAGE_VIEW)
        action_count = sum(1 for e in events if e.event_type in [EventType.CLICK, EventType.APP_ACTION])
        idle_time = self._compute_idle_time(sorted_events)
        return SessionMetrics(
            session_duration_minutes=duration,
            session_page_count=page_count,
            session_action_count=action_count,
            session_idle_time=idle_time
        )

    def _separate_by_watermark(self, events: List[Event]):
        if not events:
            return [], [], []
        if not self._watermark_config.enabled:
            return events, [], []
        max_timestamp = max(e.event_timestamp for e in events)
        watermark = max_timestamp - timedelta(minutes=self._watermark_config.delay_minutes)
        valid, late, dropped = [], [], []
        for event in events:
            if event.event_timestamp > watermark:
                valid.append(event)
            elif event.event_timestamp >= watermark - timedelta(minutes=self._watermark_config.delay_minutes):
                late.append(event)
            else:
                dropped.append(event)
        return valid, late, dropped

    def _compute_aggregation(self, events: List[Event], aggregation: str, property_key: Optional[str]) -> float:
        if not events:
            return 0.0
        if aggregation == "count":
            return float(len(events))
        values = []
        for e in events:
            if property_key and property_key in e.event_properties:
                val = e.event_properties[property_key]
                if isinstance(val, (int, float)):
                    values.append(val)
        if not values:
            return 0.0
        if aggregation == "sum":
            return sum(values)
        elif aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "min":
            return min(values)
        return 0.0

    def _aggregate_tumbling(self, events: List[Event], aggregation: str) -> List[AggregationResult]:
        if not events:
            return []
        duration = timedelta(minutes=self._window.duration_minutes)
        sorted_events = sorted(events, key=lambda e: e.event_timestamp)
        min_time = sorted_events[0].event_timestamp
        windows = defaultdict(list)
        for event in sorted_events:
            window_idx = int((event.event_timestamp - min_time) / duration)
            windows[window_idx].append(event)
        results = []
        for idx in sorted(windows.keys()):
            window_events = windows[idx]
            value = self._compute_aggregation(window_events, aggregation, None)
            results.append(AggregationResult(
                window_type=WindowType.TUMBLING,
                aggregated_value=value,
                window_start=min_time + idx * duration,
                window_end=min_time + (idx + 1) * duration,
                event_count=len(window_events)
            ))
        return results

    def _aggregate_sliding(self, events: List[Event], aggregation: str) -> List[AggregationResult]:
        if not events:
            return []
        duration = timedelta(minutes=self._window.duration_minutes)
        slide = timedelta(minutes=self._window.slide_minutes)
        sorted_events = sorted(events, key=lambda e: e.event_timestamp)
        min_time = sorted_events[0].event_timestamp
        max_time = sorted_events[-1].event_timestamp
        results = []
        current_start = min_time
        while current_start <= max_time:
            current_end = current_start + duration
            window_events = [e for e in sorted_events if current_start <= e.event_timestamp < current_end]
            if window_events:
                value = self._compute_aggregation(window_events, aggregation, None)
                results.append(AggregationResult(
                    window_type=WindowType.SLIDING,
                    aggregated_value=value,
                    window_start=current_start,
                    window_end=current_end,
                    event_count=len(window_events)
                ))
            current_start += slide
        return results

    def _aggregate_session(self, events: List[Event], aggregation: str) -> List[AggregationResult]:
        if not events:
            return []
        gap = timedelta(minutes=self._window.gap_minutes)
        sorted_events = sorted(events, key=lambda e: e.event_timestamp)
        sessions = []
        current_session = [sorted_events[0]]
        for event in sorted_events[1:]:
            if event.event_timestamp - current_session[-1].event_timestamp > gap:
                sessions.append(current_session)
                current_session = [event]
            else:
                current_session.append(event)
        sessions.append(current_session)
        results = []
        for session_events in sessions:
            value = self._compute_aggregation(session_events, aggregation, None)
            results.append(AggregationResult(
                window_type=WindowType.SESSION,
                aggregated_value=value,
                window_start=session_events[0].event_timestamp,
                window_end=session_events[-1].event_timestamp,
                event_count=len(session_events)
            ))
        return results

    def _compute_idle_time(self, sorted_events: List[Event], idle_threshold_seconds: int = 30) -> float:
        if len(sorted_events) < 2:
            return 0.0
        idle = 0.0
        for i in range(1, len(sorted_events)):
            gap = (sorted_events[i].event_timestamp - sorted_events[i-1].event_timestamp).total_seconds()
            if gap > idle_threshold_seconds:
                idle += gap
        return idle / 60.0


class StreamState:
    def __init__(self):
        self._state: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, Dict[str, datetime]] = {}

    def update_customer_state(self, customer_id: str, features: Dict[str, Any], timestamp: Optional[datetime] = None):
        if customer_id not in self._state:
            self._state[customer_id] = {}
            self._timestamps[customer_id] = {}
        ts = timestamp or datetime.now()
        for key, value in features.items():
            self._state[customer_id][key] = value
            self._timestamps[customer_id][key] = ts

    def increment_customer_state(self, customer_id: str, feature_name: str, increment: float):
        if customer_id not in self._state:
            self._state[customer_id] = {}
            self._timestamps[customer_id] = {}
        current = self._state[customer_id].get(feature_name, 0)
        self._state[customer_id][feature_name] = current + increment
        self._timestamps[customer_id][feature_name] = datetime.now()

    def get_customer_state(self, customer_id: str) -> Dict[str, Any]:
        return self._state.get(customer_id, {}).copy()

    def expire_old_windows(self, max_age_minutes: int):
        cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
        for customer_id in list(self._state.keys()):
            for feature in list(self._state[customer_id].keys()):
                if self._timestamps.get(customer_id, {}).get(feature, datetime.now()) < cutoff:
                    del self._state[customer_id][feature]
                    del self._timestamps[customer_id][feature]

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "wb") as f:
            pickle.dump({"state": self._state, "timestamps": self._timestamps}, f)

    def load_checkpoint(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self._state = data["state"]
                self._timestamps = data["timestamps"]

    def to_delta_checkpoint_format(self) -> DataFrame:
        rows = []
        for cust_id, features in self._state.items():
            for feature_name, value in features.items():
                rows.append({
                    "customer_id": cust_id,
                    "feature_name": feature_name,
                    "feature_value": value,
                    "updated_at": self._timestamps.get(cust_id, {}).get(feature_name, datetime.now())
                })
        return pd.DataFrame(rows)


@dataclass
class StreamingFeature:
    name: str
    window_type: WindowType
    aggregation: str
    window_duration_minutes: int
    property_key: Optional[str] = None


@dataclass
class FeatureComputeResult:
    features: Dict[str, float]
    computed_at: datetime
    feature_freshness_seconds: float = 0.0


class FeatureComputer:
    def compute_count_features(self, events: List[Event], customer_id: str) -> Dict[str, float]:
        customer_events = [e for e in events if e.customer_id == customer_id]
        page_views = [e for e in customer_events if e.event_type == EventType.PAGE_VIEW]
        orders = [e for e in customer_events if e.event_type == EventType.ORDER]
        support = [e for e in customer_events if e.event_type == EventType.SUPPORT_TICKET]
        email_opens = [e for e in customer_events if e.event_type == EventType.EMAIL_OPEN]
        return {
            "page_views_1h": float(len(page_views)),
            "page_views_24h": float(len(page_views)),
            "orders_7d": float(len(orders)),
            "support_tickets_30d": float(len(support)),
            "email_opens_7d": float(len(email_opens))
        }

    def compute_recency_features(self, events: List[Event], customer_id: str) -> Dict[str, float]:
        customer_events = [e for e in events if e.customer_id == customer_id]
        if not customer_events:
            return {"minutes_since_last_visit": float("inf")}
        latest = max(e.event_timestamp for e in customer_events)
        minutes_since = (datetime.now() - latest).total_seconds() / 60
        return {"minutes_since_last_visit": max(0.0, minutes_since)}

    def compute_velocity_features(self, events: List[Event], customer_id: str) -> Dict[str, float]:
        customer_events = [e for e in events if e.customer_id == customer_id]
        visits = [e for e in customer_events if e.event_type in [EventType.PAGE_VIEW, EventType.APP_SESSION]]
        return {"visit_velocity_1h": float(len(visits))}

    def compute_session_features(self, events: List[Event], session_gap_minutes: int = 30) -> Dict[str, float]:
        if not events:
            return {"session_duration_minutes": 0.0, "session_page_count": 0}
        window = SessionWindow(gap_minutes=session_gap_minutes)
        aggregator = WindowAggregator(window=window)
        metrics = aggregator.compute_session_metrics(events)
        return {
            "session_duration_minutes": metrics.session_duration_minutes,
            "session_page_count": metrics.session_page_count,
            "session_action_count": metrics.session_action_count,
            "session_idle_time": metrics.session_idle_time
        }

    def compute_anomaly_features(self, current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
        if "page_views_1h" not in current or "page_views_1h" not in baseline:
            return {"activity_anomaly_score": 0.0}
        current_val = current["page_views_1h"]
        baseline_mean = baseline["page_views_1h"]
        baseline_std = baseline.get("page_views_1h_std", 1.0)
        if baseline_std == 0:
            baseline_std = 1.0
        zscore = (current_val - baseline_mean) / baseline_std
        return {"activity_anomaly_score": zscore}

    def compute_all_features(self, events: List[Event], customer_id: str) -> FeatureComputeResult:
        start = datetime.now()
        features = {}
        features.update(self.compute_count_features(events, customer_id))
        features.update(self.compute_recency_features(events, customer_id))
        features.update(self.compute_velocity_features(events, customer_id))
        customer_events = [e for e in events if e.customer_id == customer_id]
        features.update(self.compute_session_features(customer_events))
        computed_at = datetime.now()
        freshness = (computed_at - start).total_seconds()
        return FeatureComputeResult(
            features=features,
            computed_at=computed_at,
            feature_freshness_seconds=freshness
        )
