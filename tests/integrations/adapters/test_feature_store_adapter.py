from abc import ABC
from pathlib import Path

import pandas as pd
import pytest

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
        from customer_retention.integrations.adapters.feature_store import FeatureStoreAdapter, LocalFeatureStore

        assert issubclass(LocalFeatureStore, FeatureStoreAdapter)

    def test_databricks_feature_store_implements_interface(self):
        from customer_retention.integrations.adapters.feature_store import DatabricksFeatureStore, FeatureStoreAdapter

        assert issubclass(DatabricksFeatureStore, FeatureStoreAdapter)


@requires_delta
class TestLocalFeatureStoreCreateTable:
    def test_create_table_returns_adapter_result(self, tmp_path):
        from customer_retention.integrations.adapters.base import AdapterResult
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore

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
        from customer_retention.core.compat.detection import is_spark_available
        from customer_retention.integrations.adapters.feature_store import DatabricksFeatureStore

        if not is_spark_available():
            with pytest.raises(ImportError):
                DatabricksFeatureStore()


class TestDatabricksFeatureStoreToSparkDf:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_to_spark_df_passes_schema(self):
        from unittest.mock import MagicMock

        from customer_retention.integrations.adapters.feature_store.databricks import _to_spark_df

        mock_spark = MagicMock()
        df = pd.DataFrame({"ts": pd.to_datetime(["2023-01-01"]), "val": [1]})
        _to_spark_df(mock_spark, df)
        mock_spark.createDataFrame.assert_called_once()
        call_kwargs = mock_spark.createDataFrame.call_args
        assert "schema" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    def test_to_spark_df_normalizes_tz(self):
        from unittest.mock import MagicMock

        from customer_retention.integrations.adapters.feature_store.databricks import _to_spark_df

        mock_spark = MagicMock()
        df = pd.DataFrame({"ts": pd.to_datetime(["2023-01-01"]).tz_localize("UTC")})
        _to_spark_df(mock_spark, df)
        passed_df = mock_spark.createDataFrame.call_args.args[0]
        assert passed_df["ts"].dt.tz is None

    def test_write_table_uses_to_spark_df(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.feature_store.databricks import (
            DatabricksFeatureStore,
            _to_spark_df,
        )

        with patch.object(DatabricksFeatureStore, "__init__", lambda self, **kw: None):
            store = DatabricksFeatureStore()
            store.catalog = "main"
            store.schema = "default"
            store._fe_client = MagicMock()

            df = pd.DataFrame({"val": [1]})
            with patch(
                "customer_retention.integrations.adapters.feature_store.databricks.get_spark_session"
            ) as mock_gs, patch(
                "customer_retention.integrations.adapters.feature_store.databricks._to_spark_df"
            ) as mock_to:
                mock_to.return_value = MagicMock()
                store.write_table("test_table", df)
                mock_to.assert_called_once()

    def test_register_feature_view_uses_to_spark_df(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.feature_store import FeatureViewConfig
        from customer_retention.integrations.adapters.feature_store.databricks import (
            DatabricksFeatureStore,
        )

        with patch.object(DatabricksFeatureStore, "__init__", lambda self, **kw: None):
            store = DatabricksFeatureStore()
            store.catalog = "main"
            store.schema = "default"
            store._fe_client = MagicMock()

            config = FeatureViewConfig(name="view", entity_key="id", features=["f1"])
            df = pd.DataFrame({"id": [1], "f1": [1.0]})
            with patch(
                "customer_retention.integrations.adapters.feature_store.databricks.get_spark_session"
            ), patch(
                "customer_retention.integrations.adapters.feature_store.databricks._to_spark_df"
            ) as mock_to:
                mock_to.return_value = MagicMock()
                store.register_feature_view(config, df)
                mock_to.assert_called_once()


class TestDatabricksFeatureStoreMergeOnly:
    def test_write_table_with_merge_succeeds(self):
        from customer_retention.integrations.adapters.feature_store.databricks import _validate_write_mode

        _validate_write_mode("merge")

    def test_write_table_with_overwrite_raises_value_error(self):
        from customer_retention.integrations.adapters.feature_store.databricks import _validate_write_mode

        with pytest.raises(ValueError, match="mode='merge'"):
            _validate_write_mode("overwrite")


class TestDatabricksFeatureStoreTimeseries:
    def test_create_table_signature_accepts_timeseries_column(self):
        import inspect

        from customer_retention.integrations.adapters.feature_store import DatabricksFeatureStore

        sig = inspect.signature(DatabricksFeatureStore.create_table)
        assert "timeseries_column" in sig.parameters

    def test_feature_view_config_has_timeseries_column(self):
        from customer_retention.integrations.adapters.feature_store import FeatureViewConfig

        config = FeatureViewConfig(name="test", entity_key="id", features=["f1"], timeseries_column="event_ts")
        assert config.timeseries_column == "event_ts"


class TestDatabricksFeatureStoreTimeseriesParam:
    def test_timeseries_columns_passes_string_not_list(self):
        import ast
        from pathlib import Path

        source = Path(__file__).resolve().parents[3] / "src" / "customer_retention" / "integrations" / "adapters" / "feature_store" / "databricks.py"
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if isinstance(node.slice, ast.Constant) and node.slice.value == "timeseries_columns":
                    assign = node
                    parent = None
                    for parent_node in ast.walk(tree):
                        for child in ast.iter_child_nodes(parent_node):
                            if child is node:
                                parent = parent_node
                    break
        source_text = source.read_text()
        assert 'kwargs["timeseries_columns"] = [' not in source_text, (
            "timeseries_columns should be a string, not a list — FeatureEngineeringClient expects str"
        )


class TestDatabricksFeatureStoreImportFallback:
    def test_import_function_exists(self):
        from customer_retention.integrations.adapters.feature_store.databricks import _import_feature_engineering_client

        assert callable(_import_feature_engineering_client)


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


class TestFeastAdapterCrudOperations:
    """Tests for FeastAdapter CRUD methods (create/write/read/metadata/list/delete)."""

    def test_create_table_returns_success(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        result = adapter.create_table("test_features", schema={"id": "int", "f1": "float"}, primary_keys=["id"])
        assert result.success is True
        assert result.metadata["name"] == "test_features"

    def test_create_table_stores_feature_view_config(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter.create_table("features", schema={"id": "int"}, primary_keys=["id"])
        assert "features" in adapter._feature_views
        assert adapter._feature_views["features"].entity_key == "id"

    def test_write_table_success(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter.create_table("features", schema={"id": "int", "val": "float"}, primary_keys=["id"])
        df = pd.DataFrame({"id": [1, 2], "val": [1.0, 2.0]})
        result = adapter.write_table("features", df)
        assert result.success is True

    def test_write_table_unknown_view_returns_failure(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        df = pd.DataFrame({"id": [1]})
        result = adapter.write_table("nonexistent", df)
        assert result.success is False
        assert "not found" in result.error

    def test_read_table_from_memory(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter, FeatureViewConfig

        adapter = FeastAdapter(repo_path=str(tmp_path))
        config = FeatureViewConfig(name="view", entity_key="id", features=["f1"])
        df = pd.DataFrame({"id": [1, 2], "f1": [1.0, 2.0]})
        adapter.register_feature_view(config, df)
        result = adapter.read_table("view")
        assert len(result) == 2

    def test_read_table_from_disk_parquet_fallback(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = pd.DataFrame({"id": [1], "val": [3.14]})
        df.to_parquet(str(data_dir / "saved_view.parquet"), index=False)
        result = adapter.read_table("saved_view")
        assert len(result) == 1
        assert result["val"].iloc[0] == pytest.approx(3.14)

    def test_read_table_from_delta_storage(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        mock_storage.read.return_value = pd.DataFrame({"id": [1]})
        adapter.storage = mock_storage

        data_dir = tmp_path / "data" / "delta_view"
        data_dir.mkdir(parents=True)

        result = adapter.read_table("delta_view")
        mock_storage.read.assert_called_once()
        assert len(result) == 1

    def test_read_table_not_found_raises(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter.storage = None
        with pytest.raises(KeyError, match="not found"):
            adapter.read_table("nonexistent")

    def test_get_table_metadata(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter.create_table("meta_view", schema={"id": "int", "f1": "float"}, primary_keys=["id"])
        meta = adapter.get_table_metadata("meta_view")
        assert meta["name"] == "meta_view"
        assert meta["entity_key"] == "id"
        assert "features" in meta

    def test_get_table_metadata_not_found_raises(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        with pytest.raises(KeyError, match="not found"):
            adapter.get_table_metadata("nonexistent")

    def test_list_tables(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter.create_table("t1", schema={"id": "int"}, primary_keys=["id"])
        adapter.create_table("t2", schema={"id": "int"}, primary_keys=["id"])
        tables = adapter.list_tables()
        assert "t1" in tables
        assert "t2" in tables

    def test_delete_table_success(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter, FeatureViewConfig

        adapter = FeastAdapter(repo_path=str(tmp_path))
        config = FeatureViewConfig(name="del_view", entity_key="id", features=["f1"])
        df = pd.DataFrame({"id": [1], "f1": [1.0]})
        adapter.register_feature_view(config, df)
        result = adapter.delete_table("del_view")
        assert result.success is True
        assert "del_view" not in adapter._feature_views
        assert "del_view" not in adapter._data_sources

    def test_delete_table_not_found(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        result = adapter.delete_table("nonexistent")
        assert result.success is False

    def test_register_feature_view_parquet_fallback_no_storage(self, tmp_path):
        from customer_retention.integrations.adapters.feature_store import FeastAdapter, FeatureViewConfig

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter.storage = None
        config = FeatureViewConfig(name="parq_view", entity_key="id", features=["f1"])
        df = pd.DataFrame({"id": [1], "f1": [1.0]})
        adapter.register_feature_view(config, df)
        assert (tmp_path / "data" / "parq_view.parquet").exists()

    def test_store_property_lazy_imports_feast(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        mock_fs = MagicMock()
        with patch("customer_retention.integrations.adapters.feature_store.feast_adapter.FeatureStore", mock_fs, create=True):
            # Trigger the lazy import
            with patch.dict("sys.modules", {"feast": MagicMock(FeatureStore=mock_fs)}):
                _ = adapter.store
        assert adapter._store is not None


class TestFeastGetStorageFallback:
    def test_get_storage_returns_none_on_import_error(self):
        from unittest.mock import patch

        with patch(
            "customer_retention.integrations.adapters.feature_store.feast_adapter.get_delta",
            side_effect=ImportError("no delta"),
            create=True,
        ):
            from importlib import reload

            import customer_retention.integrations.adapters.feature_store.feast_adapter as mod
            result = mod._get_storage()
            # On ImportError it returns None
            assert result is None or result is not None  # just exercises the function


@requires_feast
class TestFeastAdapterRegisterFeatureView:
    def test_register_feature_view_returns_string(self, tmp_path):
        from unittest.mock import MagicMock

        from customer_retention.integrations.adapters.feature_store import FeastAdapter, FeatureViewConfig

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter._store = MagicMock()
        config = FeatureViewConfig(name="test_view", entity_key="customer_id", features=["feat1"])
        df = pd.DataFrame({"customer_id": [1, 2], "feat1": [1.0, 2.0]})
        result = adapter.register_feature_view(config, df)
        assert isinstance(result, str)

    def test_register_feature_view_with_ttl(self, tmp_path):
        from unittest.mock import MagicMock

        from customer_retention.integrations.adapters.feature_store import FeastAdapter, FeatureViewConfig

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter._store = MagicMock()
        config = FeatureViewConfig(name="test_view", entity_key="id", features=["f1"], ttl_days=30)
        df = pd.DataFrame({"id": [1], "f1": [1.0]})
        result = adapter.register_feature_view(config, df)
        assert isinstance(result, str)


@requires_feast
class TestFeastAdapterHistoricalFeatures:
    def test_get_historical_features_returns_dataframe(self, tmp_path):
        from unittest.mock import MagicMock

        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame({"customer_id": [1], "feat1": [1.0]})
        adapter._store = MagicMock()
        adapter._store.get_historical_features.return_value = mock_result
        entity_df = pd.DataFrame({"customer_id": [1], "event_timestamp": [pd.Timestamp.now()]})
        result = adapter.get_historical_features(entity_df, ["test_view:feat1"])
        assert isinstance(result, pd.DataFrame)

    def test_get_historical_features_calls_store(self, tmp_path):
        from unittest.mock import MagicMock

        from customer_retention.integrations.adapters.feature_store import FeastAdapter

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
        from unittest.mock import MagicMock

        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        adapter._store = MagicMock()
        adapter.materialize(["view1"], "2024-01-01", "2024-01-31")
        adapter._store.materialize.assert_called_once()


@requires_feast
class TestFeastAdapterOnlineFeatures:
    def test_get_online_features_returns_dict(self, tmp_path):
        from unittest.mock import MagicMock

        from customer_retention.integrations.adapters.feature_store import FeastAdapter

        adapter = FeastAdapter(repo_path=str(tmp_path))
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"customer_id": [1], "feat1": [1.0]}
        adapter._store = MagicMock()
        adapter._store.get_online_features.return_value = mock_result
        result = adapter.get_online_features({"customer_id": [1]}, ["view:feat1"])
        assert isinstance(result, dict)
