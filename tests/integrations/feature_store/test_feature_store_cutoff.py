from datetime import datetime

import pandas as pd
import pytest

from customer_retention.integrations.feature_store.definitions import FeatureComputationType, TemporalFeatureDefinition
from customer_retention.integrations.feature_store.manager import FeastBackend, FeatureStoreManager
from customer_retention.integrations.feature_store.registry import FeatureRegistry

PASSTHROUGH_FEATURE = FeatureComputationType.PASSTHROUGH


class TestFeastBackendCutoff:
    @pytest.fixture
    def backend(self, tmp_path):
        return FeastBackend(repo_path=str(tmp_path / "feast_repo"))

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "entity_id": ["A", "B", "C"],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "tenure_months": [12, 24, 6],
            "total_spend": [100.0, 200.0, 50.0],
        })

    def test_create_table_stores_cutoff_date(self, backend):
        cutoff = datetime(2024, 6, 1)
        backend.create_feature_table(
            name="test_features",
            entity_key="entity_id",
            timestamp_column="feature_timestamp",
            schema={"entity_id": "string", "feature_timestamp": "datetime", "value": "float64"},
            cutoff_date=cutoff,
        )

        stored_cutoff = backend.get_table_cutoff_date("test_features")
        assert stored_cutoff.date() == cutoff.date()

    def test_cutoff_consistency_validation_first_table(self, backend):
        cutoff = datetime(2024, 6, 1)
        is_valid, message = backend.validate_cutoff_consistency(cutoff)

        assert is_valid
        assert "First feature table" in message

    def test_cutoff_consistency_validation_matching(self, backend):
        cutoff = datetime(2024, 6, 1)
        backend.create_feature_table(
            name="table_1", entity_key="id", timestamp_column="ts",
            schema={"id": "string"}, cutoff_date=cutoff,
        )

        is_valid, message = backend.validate_cutoff_consistency(cutoff)
        assert is_valid
        assert "matches reference" in message

    def test_cutoff_consistency_validation_mismatch(self, backend):
        backend.create_feature_table(
            name="table_1", entity_key="id", timestamp_column="ts",
            schema={"id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )

        is_valid, message = backend.validate_cutoff_consistency(datetime(2024, 7, 1))
        assert not is_valid
        assert "mismatch" in message

    def test_write_features_stores_data_hash(self, backend, sample_df):
        cutoff = datetime(2024, 6, 1)
        backend.create_feature_table(
            name="test_features",
            entity_key="entity_id",
            timestamp_column="feature_timestamp",
            schema={"entity_id": "string", "feature_timestamp": "datetime", "tenure_months": "int"},
            cutoff_date=cutoff,
        )

        backend.write_features("test_features", sample_df[["entity_id", "feature_timestamp", "tenure_months"]], cutoff_date=cutoff)

        assert backend._tables["test_features"].get("data_hash") is not None
        assert backend._tables["test_features"].get("row_count") == 3

    def test_metadata_persists_across_instances(self, tmp_path):
        repo_path = str(tmp_path / "feast_repo")
        cutoff = datetime(2024, 6, 1)

        backend_1 = FeastBackend(repo_path=repo_path)
        backend_1.create_feature_table(
            name="persisted_table", entity_key="id", timestamp_column="ts",
            schema={"id": "string"}, cutoff_date=cutoff,
        )

        backend_2 = FeastBackend(repo_path=repo_path)
        stored_cutoff = backend_2.get_table_cutoff_date("persisted_table")

        assert stored_cutoff is not None
        assert stored_cutoff.date() == cutoff.date()


class TestFeatureStoreManagerCutoff:
    @pytest.fixture
    def manager(self, tmp_path):
        return FeatureStoreManager.create(
            backend="feast",
            repo_path=str(tmp_path / "feast_repo"),
            output_path=str(tmp_path / "output"),
        )

    @pytest.fixture
    def registry(self):
        reg = FeatureRegistry()
        reg.register(TemporalFeatureDefinition(
            name="tenure_months",
            description="Customer tenure in months",
            entity_key="entity_id",
            data_type="int64",
            computation_type=PASSTHROUGH_FEATURE,
        ))
        return reg

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "entity_id": ["A", "B", "C"],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "tenure_months": [12, 24, 6],
        })

    def test_publish_features_with_cutoff(self, manager, registry, sample_df):
        cutoff = datetime(2024, 6, 1)

        table_name = manager.publish_features(
            df=sample_df,
            registry=registry,
            table_name="customer_features",
            cutoff_date=cutoff,
        )

        assert table_name == "customer_features"
        stored_cutoff = manager.backend.get_table_cutoff_date("customer_features")
        assert stored_cutoff.date() == cutoff.date()

    def test_publish_features_uses_registry_cutoff(self, manager, registry, sample_df, tmp_path):
        cutoff = datetime(2024, 6, 1)
        manager.pit_registry.register_snapshot(
            dataset_name="test_dataset",
            snapshot_id="snap_001",
            cutoff_date=cutoff,
            source_path="/data/test.csv",
            row_count=100,
        )

        manager.publish_features(
            df=sample_df,
            registry=registry,
            table_name="customer_features",
        )

        stored_cutoff = manager.backend.get_table_cutoff_date("customer_features")
        assert stored_cutoff.date() == cutoff.date()

    def test_publish_features_rejects_inconsistent_cutoff(self, manager, registry, sample_df):
        manager.pit_registry.register_snapshot(
            dataset_name="existing_dataset",
            snapshot_id="snap_001",
            cutoff_date=datetime(2024, 6, 1),
            source_path="/data/existing.csv",
            row_count=100,
        )

        with pytest.raises(ValueError, match="consistency error"):
            manager.publish_features(
                df=sample_df,
                registry=registry,
                table_name="customer_features",
                cutoff_date=datetime(2024, 7, 1),
            )

    def test_multiple_tables_same_cutoff(self, manager, sample_df):
        cutoff = datetime(2024, 6, 1)

        reg_1 = FeatureRegistry()
        reg_1.register(TemporalFeatureDefinition(
            name="tenure_months", description="Tenure", entity_key="entity_id",
            data_type="int64", computation_type=PASSTHROUGH_FEATURE,
        ))

        reg_2 = FeatureRegistry()
        reg_2.register(TemporalFeatureDefinition(
            name="tenure_months", description="Tenure", entity_key="entity_id",
            data_type="int64", computation_type=PASSTHROUGH_FEATURE,
        ))

        manager.publish_features(df=sample_df, registry=reg_1, table_name="features_1", cutoff_date=cutoff)
        manager.publish_features(df=sample_df, registry=reg_2, table_name="features_2", cutoff_date=cutoff)

        cutoff_1 = manager.backend.get_table_cutoff_date("features_1")
        cutoff_2 = manager.backend.get_table_cutoff_date("features_2")

        assert cutoff_1.date() == cutoff_2.date() == cutoff.date()
