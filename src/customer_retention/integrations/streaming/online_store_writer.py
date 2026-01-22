import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from customer_retention.core.compat import DataFrame, pd


@dataclass
class FeatureStoreConfig:
    backend: str = "simulation"
    read_timeout_ms: int = 100
    write_timeout_ms: int = 200


@dataclass
class TTLConfig:
    default_ttl_seconds: int = 86400


@dataclass
class FeatureRecord:
    customer_id: str
    feature_name: str
    feature_value: float
    updated_at: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[int] = None


@dataclass
class FeatureWriteResult:
    success: bool
    features_written: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class BatchSyncResult:
    success: bool
    customers_synced: int = 0
    features_synced: int = 0
    error: Optional[str] = None


@dataclass
class FeatureStoreMetrics:
    avg_read_latency_ms: float = 0.0
    p99_read_latency_ms: float = 0.0
    avg_write_latency_ms: float = 0.0
    p99_write_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    total_reads: int = 0
    total_writes: int = 0


@dataclass
class FreshnessMetrics:
    avg_freshness_seconds: float = 0.0


@dataclass
class FeatureStoreSchema:
    columns: List[str] = field(default_factory=lambda: ["customer_id", "feature_name", "feature_value", "updated_at", "ttl"])


class OnlineFeatureStore:
    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        self._config = config or FeatureStoreConfig()
        self._store: Dict[str, Dict[str, FeatureRecord]] = defaultdict(dict)
        self._history: Dict[str, Dict[str, List[FeatureRecord]]] = defaultdict(lambda: defaultdict(list))
        self._ttl_config = TTLConfig()
        self._read_latencies: List[float] = []
        self._write_latencies: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0

    def write(self, record: FeatureRecord) -> FeatureWriteResult:
        start = time.time()
        try:
            record.updated_at = datetime.now()
            self._store[record.customer_id][record.feature_name] = record
            self._history[record.customer_id][record.feature_name].append(record)
            latency = (time.time() - start) * 1000
            self._write_latencies.append(latency)
            return FeatureWriteResult(success=True, features_written=1, latency_ms=latency)
        except Exception as e:
            return FeatureWriteResult(success=False, error=str(e))

    def write_batch(self, customer_id: str, features: Dict[str, float]) -> FeatureWriteResult:
        start = time.time()
        try:
            for name, value in features.items():
                record = FeatureRecord(
                    customer_id=customer_id,
                    feature_name=name,
                    feature_value=value,
                    updated_at=datetime.now()
                )
                self._store[customer_id][name] = record
                self._history[customer_id][name].append(record)
            latency = (time.time() - start) * 1000
            self._write_latencies.append(latency)
            return FeatureWriteResult(success=True, features_written=len(features), latency_ms=latency)
        except Exception as e:
            return FeatureWriteResult(success=False, error=str(e))

    def read(self, customer_id: str, feature_name: str) -> Optional[FeatureRecord]:
        start = time.time()
        record = self._store.get(customer_id, {}).get(feature_name)
        if record and record.ttl_seconds:
            age = (datetime.now() - record.updated_at).total_seconds()
            if age > record.ttl_seconds:
                del self._store[customer_id][feature_name]
                record = None
        latency = (time.time() - start) * 1000
        self._read_latencies.append(latency)
        if record:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return record

    def read_batch(self, customer_id: str, feature_names: List[str]) -> Dict[str, float]:
        start = time.time()
        result = {}
        for name in feature_names:
            record = self.read(customer_id, name)
            if record:
                result[name] = record.feature_value
        latency = (time.time() - start) * 1000
        self._read_latencies.append(latency)
        return result

    def set_ttl_config(self, config: TTLConfig):
        self._ttl_config = config

    def sync_from_batch(self, offline_features: Dict[str, Dict[str, float]], merge_mode: str = "overwrite") -> BatchSyncResult:
        try:
            customers_synced = 0
            features_synced = 0
            for customer_id, features in offline_features.items():
                for name, value in features.items():
                    if merge_mode == "preserve_streaming" and customer_id in self._store and name in self._store[customer_id]:
                        continue
                    record = FeatureRecord(
                        customer_id=customer_id,
                        feature_name=name,
                        feature_value=value,
                        updated_at=datetime.now()
                    )
                    self._store[customer_id][name] = record
                    features_synced += 1
                customers_synced += 1
            return BatchSyncResult(success=True, customers_synced=customers_synced, features_synced=features_synced)
        except Exception as e:
            return BatchSyncResult(success=False, error=str(e))

    def cleanup_expired(self) -> int:
        expired_count = 0
        for customer_id in list(self._store.keys()):
            for feature_name in list(self._store[customer_id].keys()):
                record = self._store[customer_id][feature_name]
                if record.ttl_seconds:
                    age = (datetime.now() - record.updated_at).total_seconds()
                    if age > record.ttl_seconds:
                        del self._store[customer_id][feature_name]
                        expired_count += 1
        return expired_count

    def get_feature_history(self, customer_id: str, feature_name: str, limit: int = 10) -> List[FeatureRecord]:
        history = self._history.get(customer_id, {}).get(feature_name, [])
        return history[-limit:]

    def read_at_time(self, customer_id: str, feature_name: str, timestamp: datetime) -> Optional[FeatureRecord]:
        history = self._history.get(customer_id, {}).get(feature_name, [])
        for record in reversed(history):
            if record.updated_at <= timestamp:
                return record
        return history[0] if history else None

    def get_metrics(self) -> FeatureStoreMetrics:
        read_lat = self._read_latencies or [0]
        write_lat = self._write_latencies or [0]
        total_cache = self._cache_hits + self._cache_misses
        return FeatureStoreMetrics(
            avg_read_latency_ms=statistics.mean(read_lat),
            p99_read_latency_ms=sorted(read_lat)[int(len(read_lat) * 0.99)] if len(read_lat) > 1 else read_lat[0],
            avg_write_latency_ms=statistics.mean(write_lat),
            p99_write_latency_ms=sorted(write_lat)[int(len(write_lat) * 0.99)] if len(write_lat) > 1 else write_lat[0],
            cache_hit_rate=self._cache_hits / total_cache if total_cache > 0 else 0.0,
            total_reads=len(self._read_latencies),
            total_writes=len(self._write_latencies)
        )

    def get_freshness_metrics(self) -> FreshnessMetrics:
        all_ages = []
        now = datetime.now()
        for customer_features in self._store.values():
            for record in customer_features.values():
                age = (now - record.updated_at).total_seconds()
                all_ages.append(age)
        return FreshnessMetrics(
            avg_freshness_seconds=statistics.mean(all_ages) if all_ages else 0.0
        )

    def get_schema(self) -> FeatureStoreSchema:
        return FeatureStoreSchema()

    def get_feature_table_schema(self) -> List[str]:
        return ["customer_id", "feature_name", "feature_value", "updated_at"]

    def to_delta_dataframe(self) -> DataFrame:
        rows = []
        for customer_id, features in self._store.items():
            for feature_name, record in features.items():
                rows.append({
                    "customer_id": customer_id,
                    "feature_name": feature_name,
                    "feature_value": record.feature_value,
                    "updated_at": record.updated_at
                })
        return pd.DataFrame(rows)

    def import_from_feature_table(self, feature_table: Dict[str, Dict[str, float]]) -> BatchSyncResult:
        return self.sync_from_batch(feature_table)


class FeatureLookup:
    def __init__(self, feature_store: OnlineFeatureStore, feature_names: List[str],
                 defaults: Optional[Dict[str, float]] = None):
        self._store = feature_store
        self._feature_names = feature_names
        self._defaults = defaults or {}

    def get_features(self, customer_id: str) -> Dict[str, float]:
        result = {}
        for name in self._feature_names:
            record = self._store.read(customer_id, name)
            if record:
                result[name] = record.feature_value
            elif name in self._defaults:
                result[name] = self._defaults[name]
        return result

    def get_features_batch(self, customer_ids: List[str]) -> Dict[str, Dict[str, float]]:
        return {cust_id: self.get_features(cust_id) for cust_id in customer_ids}
