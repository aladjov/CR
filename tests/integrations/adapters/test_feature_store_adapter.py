import pytest
import pandas as pd
from pathlib import Path
from abc import ABC

try:
    import deltalake
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False

requires_delta = pytest.mark.skipif(not DELTA_AVAILABLE, reason="deltalake not installed")


class TestFeatureStoreAdapterInterface:
    def test_feature_store_adapter_is_abstract(self):
        from customer_retention.integrations.adapters.feature_store import FeatureStoreAdapter
        assert issubclass(FeatureStoreAdapter, ABC)

    def test_local_feature_store_implements_interface(self):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore, FeatureStoreAdapter
        assert issubclass(LocalFeatureStore, FeatureStoreAdapter)

    def test_databricks_feature_store_implements_interface(self):
        from customer_retention.integrations.adapters.feature_store import DatabricksFeatureStore, FeatureStoreAdapter
        assert issubclass(DatabricksFeatureStore, FeatureStoreAdapter)


@requires_delta
class TestLocalFeatureStoreCreateTable:
    def test_create_table_returns_adapter_result(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        from customer_retention.integrations.adapters.base import AdapterResult
        store = LocalFeatureStore(base_path=str(tmp_path))
        result = store.create_table("test_features", schema={"id": "int", "feature1": "float"}, primary_keys=["id"])
        assert isinstance(result, AdapterResult)
        assert result.success is True

    def test_create_table_creates_metadata(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        store = LocalFeatureStore(base_path=str(tmp_path))
        store.create_table("test_features", schema={"id": "int"}, primary_keys=["id"])
        assert (Path(tmp_path) / "registry.json").exists()


@requires_delta
class TestLocalFeatureStoreWriteRead:
    def test_write_table_stores_data(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        store = LocalFeatureStore(base_path=str(tmp_path))
        store.create_table("features", schema={"id": "int", "value": "float"}, primary_keys=["id"])
        df = pd.DataFrame({"id": [1, 2], "value": [1.5, 2.5]})
        result = store.write_table("features", df)
        assert result.success is True

    def test_read_table_returns_dataframe(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        store = LocalFeatureStore(base_path=str(tmp_path))
        store.create_table("features", schema={"id": "int", "value": "float"}, primary_keys=["id"])
        df = pd.DataFrame({"id": [1, 2], "value": [1.5, 2.5]})
        store.write_table("features", df)
        result = store.read_table("features")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_write_merge_mode_updates(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        store = LocalFeatureStore(base_path=str(tmp_path))
        store.create_table("features", schema={"id": "int", "value": "float"}, primary_keys=["id"])
        df1 = pd.DataFrame({"id": [1, 2], "value": [1.0, 2.0]})
        store.write_table("features", df1)
        df2 = pd.DataFrame({"id": [2, 3], "value": [2.5, 3.0]})
        store.write_table("features", df2, mode="merge")
        result = store.read_table("features")
        assert len(result) == 3


@requires_delta
class TestLocalFeatureStoreListTables:
    def test_list_tables_returns_list(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        store = LocalFeatureStore(base_path=str(tmp_path))
        tables = store.list_tables()
        assert isinstance(tables, list)

    def test_list_tables_includes_created_tables(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        store = LocalFeatureStore(base_path=str(tmp_path))
        store.create_table("table1", schema={"id": "int"}, primary_keys=["id"])
        store.create_table("table2", schema={"id": "int"}, primary_keys=["id"])
        tables = store.list_tables()
        assert "table1" in tables
        assert "table2" in tables


@requires_delta
class TestLocalFeatureStoreMetadata:
    def test_get_table_metadata_returns_dict(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        store = LocalFeatureStore(base_path=str(tmp_path))
        store.create_table("features", schema={"id": "int"}, primary_keys=["id"])
        metadata = store.get_table_metadata("features")
        assert isinstance(metadata, dict)
        assert "primary_keys" in metadata
        assert metadata["primary_keys"] == ["id"]


class TestDatabricksFeatureStoreMocked:
    def test_databricks_store_requires_spark(self):
        from customer_retention.integrations.adapters.feature_store import DatabricksFeatureStore
        from customer_retention.core.compat.detection import is_spark_available
        if not is_spark_available():
            with pytest.raises(ImportError):
                DatabricksFeatureStore()


class TestFeatureViewConfig:
    def test_feature_view_config_creation(self):
        from customer_retention.integrations.adapters.feature_store import FeatureViewConfig
        config = FeatureViewConfig(name="test_view", entity_key="customer_id", features=["feat1", "feat2"])
        assert config.name == "test_view"
        assert config.entity_key == "customer_id"
        assert config.features == ["feat1", "feat2"]
        assert config.ttl_days is None
        assert config.tags == {}

    def test_feature_view_config_with_optional_fields(self):
        from customer_retention.integrations.adapters.feature_store import FeatureViewConfig
        config = FeatureViewConfig(name="test_view", entity_key="id", features=["f1"], ttl_days=7, tags={"env": "prod"})
        assert config.ttl_days == 7
        assert config.tags == {"env": "prod"}


try:
    import feast
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False

requires_feast = pytest.mark.skipif(not FEAST_AVAILABLE, reason="feast not installed")


class TestFeastAdapterInterface:
    def test_feast_adapter_implements_interface(self):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter, FeatureStoreAdapter
        assert issubclass(FeastAdapter, FeatureStoreAdapter)

    def test_feast_adapter_has_register_feature_view(self):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        assert hasattr(FeastAdapter, "register_feature_view")

    def test_feast_adapter_has_get_historical_features(self):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        assert hasattr(FeastAdapter, "get_historical_features")

    def test_feast_adapter_has_materialize(self):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        assert hasattr(FeastAdapter, "materialize")

    def test_feast_adapter_has_get_online_features(self):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        assert hasattr(FeastAdapter, "get_online_features")


class TestFeastAdapterInit:
    def test_lazy_store_initialization(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        adapter = FeastAdapter(repo_path=str(tmp_path))
        assert adapter._store is None

    def test_default_repo_path(self):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        adapter = FeastAdapter()
        assert adapter._repo_path == "./feature_store/feature_repo"

    def test_custom_repo_path(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        adapter = FeastAdapter(repo_path=str(tmp_path / "custom"))
        assert adapter._repo_path == str(tmp_path / "custom")


@requires_feast
class TestFeastAdapterRegisterFeatureView:
    def test_register_feature_view_returns_string(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter, FeatureViewConfig
        from unittest.mock import MagicMock, patch
        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter._store = MagicMock()
        config = FeatureViewConfig(name="test_view", entity_key="customer_id", features=["feat1"])
        df = pd.DataFrame({"customer_id": [1, 2], "feat1": [1.0, 2.0]})
        result = adapter.register_feature_view(config, df)
        assert isinstance(result, str)

    def test_register_feature_view_with_ttl(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter, FeatureViewConfig
        from unittest.mock import MagicMock
        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter._store = MagicMock()
        config = FeatureViewConfig(name="test_view", entity_key="id", features=["f1"], ttl_days=30)
        df = pd.DataFrame({"id": [1], "f1": [1.0]})
        result = adapter.register_feature_view(config, df)
        assert isinstance(result, str)


@requires_feast
class TestFeastAdapterHistoricalFeatures:
    def test_get_historical_features_returns_dataframe(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        from unittest.mock import MagicMock
        adapter = FeastAdapter(repo_path=str(tmp_path))
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame({"customer_id": [1], "feat1": [1.0]})
        adapter._store = MagicMock()
        adapter._store.get_historical_features.return_value = mock_result
        entity_df = pd.DataFrame({"customer_id": [1], "event_timestamp": [pd.Timestamp.now()]})
        result = adapter.get_historical_features(entity_df, ["test_view:feat1"])
        assert isinstance(result, pd.DataFrame)

    def test_get_historical_features_calls_store(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        from unittest.mock import MagicMock
        adapter = FeastAdapter(repo_path=str(tmp_path))
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame()
        adapter._store = MagicMock()
        adapter._store.get_historical_features.return_value = mock_result
        entity_df = pd.DataFrame({"customer_id": [1]})
        adapter.get_historical_features(entity_df, ["view:feat1"])
        adapter._store.get_historical_features.assert_called_once()


@requires_feast
class TestFeastAdapterMaterialize:
    def test_materialize_calls_store_materialize(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        from unittest.mock import MagicMock
        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter._store = MagicMock()
        adapter.materialize(["view1"], "2024-01-01", "2024-01-31")
        adapter._store.materialize.assert_called_once()


@requires_feast
class TestFeastAdapterOnlineFeatures:
    def test_get_online_features_returns_dict(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter
        from unittest.mock import MagicMock
        adapter = FeastAdapter(repo_path=str(tmp_path))
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"customer_id": [1], "feat1": [1.0]}
        adapter._store = MagicMock()
        adapter._store.get_online_features.return_value = mock_result
        result = adapter.get_online_features({"customer_id": [1]}, ["view:feat1"])
        assert isinstance(result, dict)
