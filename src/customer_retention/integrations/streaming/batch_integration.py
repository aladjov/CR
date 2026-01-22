from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class ScoreCombinationStrategy(Enum):
    BATCH_ONLY = "batch_only"
    STREAMING_OVERRIDE = "streaming_override"
    ENSEMBLE = "ensemble"
    MAXIMUM = "maximum"
    SIGNAL_BOOST = "signal_boost"


@dataclass
class ScoreResult:
    score: float
    source: str
    timestamp: Optional[datetime] = None


class BatchStreamingBridge:
    def __init__(self):
        self._feature_mapping = {
            "days_since_last_order": "minutes_since_last_order",
            "email_engagement_score": ("email_opens_7d", "emails_sent_7d"),
            "order_frequency": "orders_7d"
        }

    def combine_scores(self, batch_score: Optional[float], streaming_score: Optional[float],
                       strategy: ScoreCombinationStrategy = ScoreCombinationStrategy.MAXIMUM,
                       weights: Optional[Dict[str, float]] = None,
                       batch_timestamp: Optional[datetime] = None,
                       streaming_timestamp: Optional[datetime] = None,
                       freshness_threshold_hours: int = 1) -> float:
        if streaming_score is None and batch_score is None:
            return 0.0
        if streaming_score is None:
            return batch_score
        if batch_score is None:
            return streaming_score
        if strategy == ScoreCombinationStrategy.BATCH_ONLY:
            return batch_score
        elif strategy == ScoreCombinationStrategy.STREAMING_OVERRIDE:
            if streaming_timestamp and batch_timestamp:
                streaming_age = (datetime.now() - streaming_timestamp).total_seconds() / 3600
                if streaming_age < freshness_threshold_hours:
                    return streaming_score
            return streaming_score if streaming_score is not None else batch_score
        elif strategy == ScoreCombinationStrategy.ENSEMBLE:
            w = weights or {"batch": 0.5, "streaming": 0.5}
            return batch_score * w.get("batch", 0.5) + streaming_score * w.get("streaming", 0.5)
        elif strategy == ScoreCombinationStrategy.MAXIMUM:
            return max(batch_score, streaming_score)
        elif strategy == ScoreCombinationStrategy.SIGNAL_BOOST:
            boost = 0.1 if streaming_score > batch_score else 0.0
            return min(batch_score + boost, 1.0)
        return batch_score

    def map_features(self, batch_features: Dict[str, float], streaming_features: Dict[str, float],
                     prefer_streaming_recency: bool = False) -> Dict[str, float]:
        result = batch_features.copy()
        result.update(streaming_features)
        if prefer_streaming_recency and "minutes_since_last_order" in streaming_features:
            result["days_since_last_order"] = streaming_features["minutes_since_last_order"] / (24 * 60)
        return result

    def get_best_available_score(self, realtime_score: Optional[float] = None,
                                 streaming_score: Optional[float] = None,
                                 batch_score: Optional[float] = None,
                                 cached_score: Optional[float] = None) -> ScoreResult:
        if realtime_score is not None:
            return ScoreResult(score=realtime_score, source="realtime")
        if streaming_score is not None:
            return ScoreResult(score=streaming_score, source="streaming")
        if batch_score is not None:
            return ScoreResult(score=batch_score, source="batch")
        if cached_score is not None:
            return ScoreResult(score=cached_score, source="cached")
        return ScoreResult(score=0.0, source="default")


@dataclass
class ProcessingConfig:
    checkpoint_interval_seconds: int = 60
    watermark_delay_minutes: int = 10
    trigger_interval_seconds: int = 60


@dataclass
class ProcessingResult:
    events_processed: int = 0
    features_computed: int = 0
    errors: int = 0
    processing_time_ms: float = 0.0


@dataclass
class ProcessingMetrics:
    avg_processing_latency_ms: float = 0.0
    events_per_second: float = 0.0


class StreamProcessor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self._config = config or ProcessingConfig()
        self._state: Dict[str, Dict[str, float]] = {}
        self._processing_times: List[float] = []
        self._events_processed = 0
        self._start_time = datetime.now()

    def process_batch(self, events: List) -> ProcessingResult:
        import time
        start = time.time()
        from .window_aggregator import FeatureComputer
        computer = FeatureComputer()
        features_computed = 0
        by_customer: Dict[str, List] = {}
        for event in events:
            cust_id = event.customer_id
            if cust_id not in by_customer:
                by_customer[cust_id] = []
            by_customer[cust_id].append(event)
        for customer_id, customer_events in by_customer.items():
            result = computer.compute_all_features(customer_events, customer_id)
            if customer_id not in self._state:
                self._state[customer_id] = {}
            for feature_name, value in result.features.items():
                self._state[customer_id][feature_name] = self._state[customer_id].get(feature_name, 0) + value
            features_computed += len(result.features)
        elapsed = (time.time() - start) * 1000
        self._processing_times.append(elapsed)
        self._events_processed += len(events)
        return ProcessingResult(
            events_processed=len(events),
            features_computed=features_computed,
            processing_time_ms=elapsed
        )

    def get_state(self, customer_id: str) -> Dict[str, float]:
        return self._state.get(customer_id, {}).copy()

    def get_metrics(self) -> ProcessingMetrics:
        import statistics
        elapsed_seconds = max((datetime.now() - self._start_time).total_seconds(), 1)
        return ProcessingMetrics(
            avg_processing_latency_ms=statistics.mean(self._processing_times) if self._processing_times else 0.0,
            events_per_second=self._events_processed / elapsed_seconds
        )
