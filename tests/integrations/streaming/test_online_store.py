import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from customer_retention.integrations.streaming import (
    OnlineFeatureStore, FeatureRecord, FeatureStoreConfig,
    BatchSyncResult, TTLConfig, FeatureLookup, FeatureWriteResult
)


@pytest.fixture
def feature_store():
    config = FeatureStoreConfig(
        backend="simulation",
        read_timeout_ms=100,
        write_timeout_ms=200
    )
    return OnlineFeatureStore(config=config)


@pytest.fixture
def sample_features() -> Dict[str, float]:
    return {
        "page_views_1h": 15.0,
        "page_views_24h": 45.0,
        "orders_7d": 3.0,
        "minutes_since_last_visit": 30.0,
        "visit_velocity_1h": 15.0,
        "activity_anomaly_score": -0.5
    }


class TestFeatureWrite:
    def test_write_single_feature(self, feature_store):
        record = FeatureRecord(
            customer_id="CUST001",
            feature_name="page_views_1h",
            feature_value=10.0
        )
        result = feature_store.write(record)
        assert result.success is True

    def test_write_multiple_features(self, feature_store, sample_features):
        result = feature_store.write_batch(
            customer_id="CUST001",
            features=sample_features
        )
        assert result.success is True
        assert result.features_written == len(sample_features)

    def test_ac9_10_write_latency_under_50ms(self, feature_store, sample_features):
        start = time.time()
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 50

    def test_write_updates_timestamp(self, feature_store):
        before = datetime.now()
        record = FeatureRecord(
            customer_id="CUST001",
            feature_name="page_views_1h",
            feature_value=10.0
        )
        feature_store.write(record)
        stored = feature_store.read("CUST001", "page_views_1h")
        assert stored.updated_at >= before


class TestFeatureRead:
    def test_read_single_feature(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        result = feature_store.read("CUST001", "page_views_1h")
        assert result.feature_value == 15.0

    def test_read_multiple_features(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        features_to_read = ["page_views_1h", "orders_7d", "visit_velocity_1h"]
        result = feature_store.read_batch("CUST001", features_to_read)
        assert len(result) == 3
        assert result["page_views_1h"] == 15.0
        assert result["orders_7d"] == 3.0

    def test_ac9_9_read_latency_under_10ms(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        start = time.time()
        feature_store.read_batch("CUST001", list(sample_features.keys()))
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 10

    def test_read_nonexistent_customer_returns_none(self, feature_store):
        result = feature_store.read("NONEXISTENT", "page_views_1h")
        assert result is None

    def test_read_nonexistent_feature_returns_none(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        result = feature_store.read("CUST001", "nonexistent_feature")
        assert result is None


class TestFeatureLookup:
    def test_lookup_for_scoring(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        lookup = FeatureLookup(
            feature_store=feature_store,
            feature_names=["page_views_1h", "orders_7d", "activity_anomaly_score"]
        )
        result = lookup.get_features("CUST001")
        assert len(result) == 3
        assert result["page_views_1h"] == 15.0

    def test_lookup_with_defaults(self, feature_store):
        lookup = FeatureLookup(
            feature_store=feature_store,
            feature_names=["page_views_1h", "orders_7d"],
            defaults={"page_views_1h": 0.0, "orders_7d": 0.0}
        )
        result = lookup.get_features("NONEXISTENT")
        assert result["page_views_1h"] == 0.0
        assert result["orders_7d"] == 0.0

    def test_lookup_multiple_customers(self, feature_store, sample_features):
        for cust_id in ["CUST001", "CUST002", "CUST003"]:
            feature_store.write_batch(customer_id=cust_id, features=sample_features)
        lookup = FeatureLookup(
            feature_store=feature_store,
            feature_names=["page_views_1h"]
        )
        results = lookup.get_features_batch(["CUST001", "CUST002", "CUST003"])
        assert len(results) == 3
        for cust_id, features in results.items():
            assert features["page_views_1h"] == 15.0


class TestBatchSync:
    def test_ac9_11_batch_sync_from_offline_store(self, feature_store):
        offline_features = {
            "CUST001": {"tenure_days": 365, "total_orders": 50, "clv_score": 0.8},
            "CUST002": {"tenure_days": 180, "total_orders": 20, "clv_score": 0.5},
            "CUST003": {"tenure_days": 30, "total_orders": 5, "clv_score": 0.2},
        }
        result = feature_store.sync_from_batch(offline_features)
        assert result.success is True
        assert result.customers_synced == 3
        assert result.features_synced == 9

    def test_batch_sync_updates_existing(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        offline_features = {
            "CUST001": {"page_views_1h": 100.0, "tenure_days": 365}
        }
        feature_store.sync_from_batch(offline_features)
        result = feature_store.read("CUST001", "page_views_1h")
        assert result.feature_value == 100.0
        result2 = feature_store.read("CUST001", "tenure_days")
        assert result2.feature_value == 365

    def test_batch_sync_preserves_streaming_features(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        offline_features = {"CUST001": {"tenure_days": 365}}
        feature_store.sync_from_batch(offline_features, merge_mode="preserve_streaming")
        result = feature_store.read("CUST001", "page_views_1h")
        assert result.feature_value == 15.0


class TestTTL:
    def test_ac9_12_ttl_expiration(self, feature_store):
        config = TTLConfig(default_ttl_seconds=1)
        feature_store.set_ttl_config(config)
        record = FeatureRecord(
            customer_id="CUST001",
            feature_name="page_views_1h",
            feature_value=10.0,
            ttl_seconds=1
        )
        feature_store.write(record)
        time.sleep(1.5)
        result = feature_store.read("CUST001", "page_views_1h")
        assert result is None

    def test_ttl_feature_specific(self, feature_store):
        feature_store.write(FeatureRecord(
            customer_id="CUST001",
            feature_name="short_lived",
            feature_value=1.0,
            ttl_seconds=1
        ))
        feature_store.write(FeatureRecord(
            customer_id="CUST001",
            feature_name="long_lived",
            feature_value=2.0,
            ttl_seconds=3600
        ))
        time.sleep(1.5)
        short = feature_store.read("CUST001", "short_lived")
        long = feature_store.read("CUST001", "long_lived")
        assert short is None
        assert long is not None
        assert long.feature_value == 2.0

    def test_ttl_cleanup(self, feature_store):
        config = TTLConfig(default_ttl_seconds=1)
        feature_store.set_ttl_config(config)
        for i in range(10):
            feature_store.write(FeatureRecord(
                customer_id=f"CUST{i:03d}",
                feature_name="temp_feature",
                feature_value=float(i),
                ttl_seconds=1
            ))
        time.sleep(1.5)
        expired_count = feature_store.cleanup_expired()
        assert expired_count == 10


class TestFeatureStoreMetrics:
    def test_track_read_latency(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        for _ in range(10):
            feature_store.read("CUST001", "page_views_1h")
        metrics = feature_store.get_metrics()
        assert metrics.avg_read_latency_ms < 10
        assert metrics.p99_read_latency_ms < 20

    def test_track_write_latency(self, feature_store, sample_features):
        for i in range(10):
            feature_store.write_batch(customer_id=f"CUST{i:03d}", features=sample_features)
        metrics = feature_store.get_metrics()
        assert metrics.avg_write_latency_ms < 50
        assert metrics.p99_write_latency_ms < 100

    def test_track_cache_hit_rate(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        for _ in range(5):
            feature_store.read("CUST001", "page_views_1h")
        metrics = feature_store.get_metrics()
        assert metrics.cache_hit_rate > 0


class TestFeatureVersioning:
    def test_feature_update_history(self, feature_store):
        for i in range(5):
            feature_store.write(FeatureRecord(
                customer_id="CUST001",
                feature_name="page_views_1h",
                feature_value=float(i * 10)
            ))
        history = feature_store.get_feature_history("CUST001", "page_views_1h", limit=5)
        assert len(history) == 5
        assert history[-1].feature_value == 40.0

    def test_point_in_time_lookup(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features={"page_views_1h": 10.0})
        t1 = datetime.now()
        time.sleep(0.1)
        feature_store.write_batch(customer_id="CUST001", features={"page_views_1h": 20.0})
        t2 = datetime.now()
        result_t1 = feature_store.read_at_time("CUST001", "page_views_1h", timestamp=t1)
        result_t2 = feature_store.read_at_time("CUST001", "page_views_1h", timestamp=t2)
        assert result_t1.feature_value == 10.0
        assert result_t2.feature_value == 20.0


class TestDatabricksCompatibility:
    def test_schema_compatible_with_feature_table(self, feature_store):
        schema = feature_store.get_schema()
        required_columns = ["customer_id", "feature_name", "feature_value", "updated_at", "ttl"]
        for col in required_columns:
            assert col in schema.columns

    def test_export_to_delta_format(self, feature_store, sample_features):
        feature_store.write_batch(customer_id="CUST001", features=sample_features)
        feature_store.write_batch(customer_id="CUST002", features=sample_features)
        delta_df = feature_store.to_delta_dataframe()
        assert delta_df is not None
        assert len(delta_df) > 0

    def test_import_from_feature_engineering_table(self, feature_store):
        mock_feature_table = {
            "CUST001": {"tenure_days": 365, "avg_order_value": 75.50},
            "CUST002": {"tenure_days": 180, "avg_order_value": 50.25},
        }
        result = feature_store.import_from_feature_table(mock_feature_table)
        assert result.success is True
        stored = feature_store.read("CUST001", "tenure_days")
        assert stored.feature_value == 365
