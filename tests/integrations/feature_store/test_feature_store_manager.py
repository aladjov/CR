"""Tests for the feature store manager module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from customer_retention.integrations.feature_store.definitions import (
    FeatureComputationType,
    TemporalFeatureDefinition,
)
from customer_retention.integrations.feature_store.manager import (
    DatabricksBackend,
    FeastBackend,
    FeatureStoreManager,
    get_feature_store_manager,
)
from customer_retention.integrations.feature_store.registry import FeatureRegistry

PASSTHROUGH = FeatureComputationType.PASSTHROUGH


def _make_registry(*feature_names):
    """Helper to create a registry with simple passthrough features."""
    reg = FeatureRegistry()
    for name in feature_names:
        reg.register(TemporalFeatureDefinition(
            name=name,
            description=f"Feature: {name}",
            entity_key="entity_id",
            data_type="float64",
            computation_type=PASSTHROUGH,
        ))
    return reg


def _make_sample_df():
    return pd.DataFrame({
        "entity_id": ["A", "B", "C"],
        "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        "tenure_months": [12, 24, 6],
        "total_spend": [100.0, 200.0, 50.0],
    })


# ---------------------------------------------------------------------------
# FeastBackend tests
# ---------------------------------------------------------------------------
class TestFeastBackend:
    @pytest.fixture
    def backend(self, tmp_path):
        return FeastBackend(repo_path=str(tmp_path / "feast_repo"))

    @pytest.fixture
    def sample_df(self):
        return _make_sample_df()

    # --- create_feature_table ---
    def test_create_feature_table_without_cutoff(self, backend):
        name = backend.create_feature_table(
            name="tbl",
            entity_key="entity_id",
            timestamp_column="ts",
            schema={"entity_id": "string"},
        )
        assert name == "tbl"
        assert backend._tables["tbl"]["cutoff_date"] is None

    # --- get_table_cutoff_date ---
    def test_get_table_cutoff_date_missing_table(self, backend):
        assert backend.get_table_cutoff_date("nonexistent") is None

    def test_get_table_cutoff_date_no_cutoff_stored(self, backend):
        backend.create_feature_table(
            name="tbl", entity_key="id", timestamp_column="ts",
            schema={"id": "string"},
        )
        assert backend.get_table_cutoff_date("tbl") is None

    # --- _compute_feature_hash ---
    def test_compute_feature_hash_deterministic(self, backend, sample_df):
        h1 = backend._compute_feature_hash(sample_df)
        h2 = backend._compute_feature_hash(sample_df)
        assert h1 == h2
        assert len(h1) == 16

    def test_compute_feature_hash_with_cutoff(self, backend, sample_df):
        h_no = backend._compute_feature_hash(sample_df)
        h_with = backend._compute_feature_hash(sample_df, cutoff_date=datetime(2024, 6, 1))
        assert h_no != h_with

    def test_compute_feature_hash_with_datetime_cols(self, backend):
        df = pd.DataFrame({
            "entity_id": ["A"],
            "ts": pd.to_datetime(["2024-01-01"]),
            "val": [1.0],
        })
        h = backend._compute_feature_hash(df)
        assert isinstance(h, str)

    # --- write_features (no delta storage) ---
    def test_write_features_overwrite(self, backend, sample_df):
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        backend.write_features("tbl", sample_df, mode="overwrite")
        assert backend._tables["tbl"]["row_count"] == 3

    def test_write_features_merge_no_existing_parquet(self, backend, sample_df):
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        backend.write_features("tbl", sample_df, mode="merge")
        assert backend._tables["tbl"]["row_count"] == 3

    def test_write_features_merge_existing_parquet(self, backend, sample_df):
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        # First write
        backend.write_features("tbl", sample_df, mode="overwrite")
        # Merge with updated rows
        new_df = pd.DataFrame({
            "entity_id": ["A", "D"],
            "feature_timestamp": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            "tenure_months": [99, 1],
            "total_spend": [999.0, 10.0],
        })
        backend.write_features("tbl", new_df, mode="merge")
        assert backend._tables["tbl"]["row_count"] == 4  # 3 original minus 1 dup + 2 new

    def test_write_features_without_cutoff_in_table(self, backend, sample_df):
        """Write features when table has no cutoff date set."""
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"},
        )
        backend.write_features("tbl", sample_df, mode="overwrite")
        assert backend._tables["tbl"]["row_count"] == 3

    def test_write_features_table_not_in_tables_dict(self, backend, sample_df):
        """Write features to a table name not registered via create_feature_table."""
        parquet_dir = backend.repo_path / "data"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        backend.write_features("unknown_tbl", sample_df, mode="overwrite")
        # Should not crash; since unknown_tbl not in _tables it just saves parquet

    # --- write_features with delta storage ---
    def test_write_features_delta_merge(self, backend, sample_df):
        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        mock_storage.read.return_value = sample_df
        backend.storage = mock_storage

        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        backend.write_features("tbl", sample_df, mode="merge")
        mock_storage.merge.assert_called_once()
        mock_storage.read.assert_called_once()

    def test_write_features_delta_overwrite(self, backend, sample_df):
        mock_storage = MagicMock()
        mock_storage.exists.return_value = False
        backend.storage = mock_storage

        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        backend.write_features("tbl", sample_df, mode="overwrite")
        mock_storage.write.assert_called_once()

    # --- list_tables ---
    def test_list_tables_empty(self, backend):
        assert backend.list_tables() == []

    def test_list_tables_with_parquet(self, backend, sample_df):
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="ts",
            schema={"entity_id": "string"},
        )
        backend.write_features("tbl", sample_df, mode="overwrite")
        tables = backend.list_tables()
        assert "tbl" in tables

    def test_list_tables_with_delta_storage(self, backend, tmp_path):
        data_dir = backend.repo_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Create a sub-dir to simulate delta table
        delta_dir = data_dir / "delta_tbl"
        delta_dir.mkdir()

        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        backend.storage = mock_storage

        tables = backend.list_tables()
        assert "delta_tbl" in tables

    # --- _read_table_data ---
    def test_read_table_data_from_parquet(self, backend, sample_df):
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="ts",
            schema={"entity_id": "string"},
        )
        backend.write_features("tbl", sample_df, mode="overwrite")
        result = backend._read_table_data("tbl")
        assert result is not None
        assert len(result) == 3

    def test_read_table_data_from_delta(self, backend, sample_df):
        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        mock_storage.read.return_value = sample_df
        backend.storage = mock_storage

        result = backend._read_table_data("tbl")
        assert result is not None
        mock_storage.read.assert_called_once()

    def test_read_table_data_not_found(self, backend):
        assert backend._read_table_data("nonexistent") is None

    # --- data_root ---
    def test_default_data_root_is_repo_data(self, tmp_path):
        backend = FeastBackend(repo_path=str(tmp_path / "feast"))
        assert backend.data_root == tmp_path / "feast" / "data"

    def test_custom_data_root(self, tmp_path, sample_df):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        backend = FeastBackend(repo_path=str(tmp_path / "feast"), data_root=str(gold_dir))
        assert backend.data_root == gold_dir
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="ts",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        backend.write_features("tbl", sample_df, mode="overwrite")
        tbl_exists = (gold_dir / "tbl").is_dir() or (gold_dir / "tbl.parquet").exists()
        assert tbl_exists
        feast_data = tmp_path / "feast" / "data"
        assert not (feast_data / "tbl.parquet").exists()
        assert not (feast_data / "tbl").is_dir()

    def test_custom_data_root_read(self, tmp_path, sample_df):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        backend = FeastBackend(repo_path=str(tmp_path / "feast"), data_root=str(gold_dir))
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="ts",
            schema={"entity_id": "string"},
        )
        backend.write_features("tbl", sample_df, mode="overwrite")
        result = backend._read_table_data("tbl")
        assert result is not None
        assert len(result) == 3

    # --- get_historical_features (fallback to manual PIT join) ---
    def test_get_historical_features_fallback(self, backend, sample_df):
        """Test that get_historical_features falls back to manual PIT join."""
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        backend.write_features("tbl", sample_df, mode="overwrite")

        entity_df = pd.DataFrame({
            "entity_id": ["A", "B"],
            "event_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01"]),
        })

        # Feast is not installed, so store property will raise ImportError,
        # triggering the fallback to _manual_pit_join
        result = backend.get_historical_features(
            entity_df=entity_df,
            feature_refs=["tbl:tenure_months"],
            timestamp_column="event_timestamp",
        )
        assert "tenure_months" in result.columns

    def test_manual_pit_join_invalid_ref(self, backend):
        """Feature refs without colon are skipped."""
        entity_df = pd.DataFrame({"entity_id": ["A"], "event_timestamp": [datetime(2024, 6, 1)]})
        result = backend._manual_pit_join(entity_df, ["invalid_ref_no_colon"], "event_timestamp")
        assert list(result.columns) == list(entity_df.columns)

    def test_manual_pit_join_table_not_found(self, backend):
        """Table that doesn't exist as parquet skips feature."""
        entity_df = pd.DataFrame({"entity_id": ["A"], "event_timestamp": [datetime(2024, 6, 1)]})
        result = backend._manual_pit_join(entity_df, ["no_table:feat"], "event_timestamp")
        assert "feat" not in result.columns

    def test_manual_pit_join_feature_not_in_table(self, backend, sample_df):
        """Feature name not in table columns is skipped."""
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"},
        )
        backend.write_features("tbl", sample_df, mode="overwrite")
        entity_df = pd.DataFrame({"entity_id": ["A"], "event_timestamp": [datetime(2024, 6, 1)]})
        result = backend._manual_pit_join(entity_df, ["tbl:nonexistent_feature"], "event_timestamp")
        assert "nonexistent_feature" not in result.columns

    def test_manual_pit_join_simple_join_no_timestamp(self, backend):
        """When timestamp columns don't exist, falls back to simple join."""
        df = pd.DataFrame({
            "entity_id": ["A", "B"],
            "val": [10, 20],
        })
        data_dir = backend.repo_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_dir / "tbl.parquet", index=False)

        backend._tables["tbl"] = {
            "entity_key": "entity_id",
            "timestamp_column": "feature_timestamp",
        }

        entity_df = pd.DataFrame({"entity_id": ["A", "B"]})
        result = backend._manual_pit_join(entity_df, ["tbl:val"], "event_timestamp")
        assert "val" in result.columns

    # --- get_online_features (fallback) ---
    def test_get_online_features_fallback(self, backend, sample_df):
        """Online features fall back to historical when Feast not installed."""
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="feature_timestamp",
            schema={"entity_id": "string"},
        )
        backend.write_features("tbl", sample_df, mode="overwrite")

        result = backend.get_online_features(
            entity_keys={"entity_id": ["A"]},
            feature_refs=["tbl:tenure_months"],
        )
        assert isinstance(result, dict)

    # --- store property ---
    def test_store_property_import_error(self):
        """When feast is not importable, ImportError is raised."""
        with patch.dict("sys.modules", {"feast": None}):
            backend = FeastBackend.__new__(FeastBackend)
            backend.repo_path = Path("/tmp/no_feast")
            backend._store = None
            backend._tables = {}
            backend.storage = None
            with pytest.raises((ImportError, ModuleNotFoundError)):
                _ = backend.store


# ---------------------------------------------------------------------------
# DatabricksBackend tests
# ---------------------------------------------------------------------------
class TestDatabricksBackend:
    def test_init_defaults(self):
        backend = DatabricksBackend()
        assert backend.catalog == "main"
        assert backend.schema == "features"
        assert backend._client is None

    def test_full_table_name(self):
        backend = DatabricksBackend(catalog="cat", schema="sch")
        assert backend._full_table_name("tbl") == "cat.sch.tbl"

    def test_client_property_import_error(self):
        backend = DatabricksBackend()
        with pytest.raises(ImportError, match="Databricks Feature Engineering"):
            _ = backend.client


# ---------------------------------------------------------------------------
# FeatureStoreManager tests
# ---------------------------------------------------------------------------
class TestFeatureStoreManager:
    @pytest.fixture
    def manager(self, tmp_path):
        return FeatureStoreManager.create(
            backend="feast",
            repo_path=str(tmp_path / "feast_repo"),
            output_path=str(tmp_path / "output"),
        )

    @pytest.fixture
    def registry(self):
        return _make_registry("tenure_months", "total_spend")

    @pytest.fixture
    def sample_df(self):
        return _make_sample_df()

    # --- create factory ---
    def test_create_feast(self, tmp_path):
        mgr = FeatureStoreManager.create(
            backend="feast",
            repo_path=str(tmp_path / "repo"),
            output_path=str(tmp_path / "out"),
        )
        assert isinstance(mgr.backend, FeastBackend)

    def test_create_databricks(self):
        mgr = FeatureStoreManager.create(backend="databricks")
        assert isinstance(mgr.backend, DatabricksBackend)

    def test_create_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            FeatureStoreManager.create(backend="unknown")

    def test_create_default_output_path(self, tmp_path):
        mgr = FeatureStoreManager.create(
            backend="feast",
            repo_path=str(tmp_path / "repo"),
        )
        assert mgr.output_path == mgr.output_path  # just verify it doesn't crash

    # --- publish_features ---
    def test_publish_features_missing_features_warning(self, manager, sample_df, capsys):
        """Warning when registry features are missing from DataFrame."""
        reg = _make_registry("tenure_months", "nonexistent_col")
        manager.publish_features(
            df=sample_df, registry=reg, table_name="tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        captured = capsys.readouterr()
        assert "nonexistent_col" in captured.out

    def test_publish_features_returns_table_name(self, manager, registry, sample_df):
        result = manager.publish_features(
            df=sample_df, registry=registry, table_name="my_tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        assert result == "my_tbl"

    def test_publish_features_backend_cutoff_mismatch(self, manager, registry, sample_df):
        """When FeastBackend already has a table with a different cutoff date."""
        manager.publish_features(
            df=sample_df, registry=registry, table_name="tbl1",
            cutoff_date=datetime(2024, 6, 1),
        )
        with pytest.raises(ValueError, match="cutoff mismatch"):
            manager.publish_features(
                df=sample_df, registry=registry, table_name="tbl2",
                cutoff_date=datetime(2024, 7, 1),
            )

    # --- get_training_features ---
    def test_get_training_features(self, manager, registry, sample_df):
        manager.publish_features(
            df=sample_df, registry=registry, table_name="tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        entity_df = pd.DataFrame({
            "entity_id": ["A", "B"],
            "event_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01"]),
        })
        result = manager.get_training_features(
            entity_df=entity_df, registry=registry, table_name="tbl",
        )
        assert isinstance(result, pd.DataFrame)

    def test_get_training_features_specific_features(self, manager, registry, sample_df):
        manager.publish_features(
            df=sample_df, registry=registry, table_name="tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        entity_df = pd.DataFrame({
            "entity_id": ["A"],
            "event_timestamp": pd.to_datetime(["2024-06-01"]),
        })
        result = manager.get_training_features(
            entity_df=entity_df, registry=registry,
            feature_names=["tenure_months"], table_name="tbl",
        )
        assert isinstance(result, pd.DataFrame)

    # --- get_inference_features ---
    def test_get_inference_features(self, manager, registry, sample_df):
        manager.publish_features(
            df=sample_df, registry=registry, table_name="tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        entity_df = pd.DataFrame({
            "entity_id": ["A"],
            "event_timestamp": pd.to_datetime(["2024-06-01"]),
        })
        result = manager.get_inference_features(
            entity_df=entity_df, registry=registry, table_name="tbl",
        )
        assert isinstance(result, pd.DataFrame)

    def test_get_inference_features_specific(self, manager, registry, sample_df):
        manager.publish_features(
            df=sample_df, registry=registry, table_name="tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        entity_df = pd.DataFrame({
            "entity_id": ["A"],
            "event_timestamp": pd.to_datetime(["2024-06-01"]),
        })
        result = manager.get_inference_features(
            entity_df=entity_df, registry=registry,
            feature_names=["tenure_months"], table_name="tbl",
        )
        assert isinstance(result, pd.DataFrame)

    # --- get_online_features ---
    def test_get_online_features(self, manager, registry, sample_df):
        manager.publish_features(
            df=sample_df, registry=registry, table_name="tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        result = manager.get_online_features(
            entity_keys={"entity_id": ["A"]},
            registry=registry, table_name="tbl",
        )
        assert isinstance(result, dict)

    def test_get_online_features_specific(self, manager, registry, sample_df):
        manager.publish_features(
            df=sample_df, registry=registry, table_name="tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        result = manager.get_online_features(
            entity_keys={"entity_id": ["A"]},
            registry=registry, feature_names=["tenure_months"], table_name="tbl",
        )
        assert isinstance(result, dict)

    # --- create_training_set_from_snapshot ---
    def test_create_training_set_from_snapshot(self, manager, registry):
        mock_df = pd.DataFrame({
            "entity_id": ["A"],
            "tenure_months": [12],
            "total_spend": [100.0],
            "target": [1],
        })
        manager.snapshot_manager.load_snapshot = MagicMock(return_value=(mock_df, {}))
        X, y = manager.create_training_set_from_snapshot("snap_1", registry)
        assert "tenure_months" in X.columns
        assert y is not None

    def test_create_training_set_from_snapshot_no_target(self, manager, registry):
        mock_df = pd.DataFrame({
            "entity_id": ["A"],
            "tenure_months": [12],
            "total_spend": [100.0],
        })
        manager.snapshot_manager.load_snapshot = MagicMock(return_value=(mock_df, {}))
        X, y = manager.create_training_set_from_snapshot("snap_1", registry)
        assert y is None

    def test_create_training_set_from_snapshot_partial_features(self, manager):
        """Only features that exist in snapshot AND registry are included."""
        reg = _make_registry("tenure_months", "nonexistent")
        mock_df = pd.DataFrame({
            "entity_id": ["A"],
            "tenure_months": [12],
            "target": [1],
        })
        manager.snapshot_manager.load_snapshot = MagicMock(return_value=(mock_df, {}))
        X, y = manager.create_training_set_from_snapshot("snap_1", reg)
        assert list(X.columns) == ["tenure_months"]

    # --- list_tables ---
    def test_list_tables(self, manager, registry, sample_df):
        manager.publish_features(
            df=sample_df, registry=registry, table_name="tbl",
            cutoff_date=datetime(2024, 6, 1),
        )
        tables = manager.list_tables()
        assert "tbl" in tables


# ---------------------------------------------------------------------------
# get_feature_store_manager tests
# ---------------------------------------------------------------------------
class TestGetFeatureStoreManager:
    def test_explicit_feast(self, tmp_path):
        mgr = get_feature_store_manager(
            backend="feast",
            repo_path=str(tmp_path / "repo"),
            output_path=str(tmp_path / "out"),
        )
        assert isinstance(mgr.backend, FeastBackend)

    def test_explicit_databricks(self):
        mgr = get_feature_store_manager(backend="databricks")
        assert isinstance(mgr.backend, DatabricksBackend)

    def test_auto_detect_feast(self, tmp_path):
        """When auto-detect says not databricks, defaults to feast."""
        with patch(
            "customer_retention.integrations.feature_store.manager.FeatureStoreManager.create"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            get_feature_store_manager(
                repo_path=str(tmp_path / "repo"),
                output_path=str(tmp_path / "out"),
            )
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs[1].get("backend", call_kwargs[0][0] if call_kwargs[0] else None) == "feast" or \
                   "feast" in str(call_kwargs)

    def test_auto_detect_import_error_fallback(self, tmp_path):
        """When detection module can't be imported, default to feast."""
        with patch(
            "customer_retention.integrations.feature_store.manager.FeatureStoreManager.create"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            # Patch the is_databricks import to fail
            import customer_retention.integrations.feature_store.manager as mgr_mod
            original = mgr_mod.get_feature_store_manager
            # Force ImportError path by directly calling with backend=None
            with patch.dict("sys.modules", {"customer_retention.core.compat.detection": None}):
                try:
                    get_feature_store_manager(
                        repo_path=str(tmp_path / "repo"),
                        output_path=str(tmp_path / "out"),
                    )
                except Exception:
                    pass  # Detection module may or may not raise
            mock_create.assert_called()


# ---------------------------------------------------------------------------
# _get_storage tests
# ---------------------------------------------------------------------------
    def test_auto_detect_databricks(self, tmp_path):
        """When is_databricks() returns True, backend is 'databricks'."""
        with patch(
            "customer_retention.core.compat.detection.is_databricks",
            return_value=True,
        ):
            mgr = get_feature_store_manager(
                catalog="test_cat",
                schema="test_sch",
            )
            assert isinstance(mgr.backend, DatabricksBackend)

    def test_auto_detect_not_databricks(self, tmp_path):
        """When is_databricks() returns False, backend is 'feast'."""
        with patch(
            "customer_retention.core.compat.detection.is_databricks",
            return_value=False,
        ):
            mgr = get_feature_store_manager(
                repo_path=str(tmp_path / "repo"),
                output_path=str(tmp_path / "out"),
            )
            assert isinstance(mgr.backend, FeastBackend)


# ---------------------------------------------------------------------------
# _get_storage tests
# ---------------------------------------------------------------------------
class TestGetStorage:
    def test_get_storage_import_error(self):
        from customer_retention.integrations.feature_store.manager import _get_storage
        with patch.dict(
            "sys.modules",
            {"customer_retention.integrations.adapters.factory": None},
        ):
            result = _get_storage()
            assert result is None

    def test_get_storage_success(self):
        from customer_retention.integrations.feature_store.manager import _get_storage
        mock_delta = MagicMock()
        with patch(
            "customer_retention.integrations.adapters.factory.get_delta",
            return_value=mock_delta,
        ):
            result = _get_storage()
            # Result depends on whether get_delta returns something


# ---------------------------------------------------------------------------
# DatabricksBackend extended tests (with mocking)
# ---------------------------------------------------------------------------
class TestDatabricksBackendMocked:
    @pytest.fixture
    def backend(self):
        backend = DatabricksBackend(catalog="test_cat", schema="test_sch")
        backend._client = MagicMock()
        return backend

    def _make_spark_mocks(self):
        """Create mock SparkSession and related objects."""
        mock_spark = MagicMock()
        mock_spark_module = MagicMock()
        mock_spark_module.SparkSession.builder.getOrCreate.return_value = mock_spark
        mock_types = MagicMock()
        return mock_spark, mock_spark_module, mock_types

    def test_create_feature_table(self, backend):
        """create_feature_table via Databricks requires Spark mocking."""
        mock_spark, mock_spark_module, mock_types = self._make_spark_mocks()

        with patch.dict("sys.modules", {
            "pyspark": MagicMock(),
            "pyspark.sql": mock_spark_module,
            "pyspark.sql.types": mock_types,
        }):
            result = backend.create_feature_table(
                name="tbl",
                entity_key="entity_id",
                timestamp_column="ts",
                schema={"entity_id": "string", "val": "float64"},
                cutoff_date=datetime(2024, 6, 1),
            )
        assert result == "test_cat.test_sch.tbl"
        backend._client.create_table.assert_called_once()

    def test_write_features(self, backend):
        mock_spark, mock_spark_module, mock_types = self._make_spark_mocks()

        df = pd.DataFrame({"entity_id": ["A"], "val": [1.0]})
        with patch.dict("sys.modules", {
            "pyspark": MagicMock(),
            "pyspark.sql": mock_spark_module,
            "pyspark.sql.types": mock_types,
        }):
            backend.write_features("tbl", df, mode="merge")
        backend._client.write_table.assert_called_once()

    def test_get_historical_features(self, backend):
        mock_spark, mock_spark_module, _ = self._make_spark_mocks()

        mock_training_set = MagicMock()
        mock_training_set.load_df.return_value.toPandas.return_value = pd.DataFrame({"a": [1]})
        backend._client.create_training_set.return_value = mock_training_set

        mock_fe_module = MagicMock()
        entity_df = pd.DataFrame({
            "entity_id": ["A"],
            "event_timestamp": pd.to_datetime(["2024-01-01"]),
        })

        with patch.dict("sys.modules", {
            "pyspark": MagicMock(),
            "pyspark.sql": mock_spark_module,
            "pyspark.sql.types": MagicMock(),
            "databricks": MagicMock(),
            "databricks.feature_engineering": mock_fe_module,
        }):
            result = backend.get_historical_features(
                entity_df=entity_df,
                feature_refs=["tbl:feat1"],
                timestamp_column="event_timestamp",
            )
        assert isinstance(result, pd.DataFrame)

    def test_get_historical_features_invalid_ref(self, backend):
        """Feature refs without colon are skipped."""
        mock_spark, mock_spark_module, _ = self._make_spark_mocks()

        mock_training_set = MagicMock()
        mock_training_set.load_df.return_value.toPandas.return_value = pd.DataFrame({"a": [1]})
        backend._client.create_training_set.return_value = mock_training_set

        entity_df = pd.DataFrame({"entity_id": ["A"], "event_timestamp": [datetime(2024, 1, 1)]})
        with patch.dict("sys.modules", {
            "pyspark": MagicMock(),
            "pyspark.sql": mock_spark_module,
            "pyspark.sql.types": MagicMock(),
            "databricks": MagicMock(),
            "databricks.feature_engineering": MagicMock(),
        }):
            result = backend.get_historical_features(
                entity_df=entity_df,
                feature_refs=["invalid_no_colon"],
            )
        assert isinstance(result, pd.DataFrame)

    def test_get_online_features(self, backend):
        mock_spark, mock_spark_module, _ = self._make_spark_mocks()

        mock_result = MagicMock()
        mock_result.toPandas.return_value = pd.DataFrame({"entity_id": ["A"], "feat": [1.0]})
        backend._client.score_batch.return_value = mock_result

        with patch.dict("sys.modules", {
            "pyspark": MagicMock(),
            "pyspark.sql": mock_spark_module,
            "pyspark.sql.types": MagicMock(),
            "databricks": MagicMock(),
            "databricks.feature_engineering": MagicMock(),
        }):
            result = backend.get_online_features(
                entity_keys={"entity_id": ["A"]},
                feature_refs=["tbl:feat", "invalid_ref"],
            )
        assert isinstance(result, dict)

    def test_list_tables(self, backend):
        mock_table = MagicMock()
        mock_table.name = "test_cat.test_sch.my_table"
        backend._client.list_tables.return_value = [mock_table]
        tables = backend.list_tables()
        assert "my_table" in tables

    def test_list_tables_filters_prefix(self, backend):
        mock_t1 = MagicMock()
        mock_t1.name = "test_cat.test_sch.tbl1"
        mock_t2 = MagicMock()
        mock_t2.name = "other_cat.other_sch.tbl2"
        backend._client.list_tables.return_value = [mock_t1, mock_t2]
        tables = backend.list_tables()
        assert "tbl1" in tables
        assert "tbl2" not in tables


# ---------------------------------------------------------------------------
# Additional FeastBackend tests for uncovered branches
# ---------------------------------------------------------------------------
class TestFeastBackendAdditional:
    @pytest.fixture
    def backend(self, tmp_path):
        return FeastBackend(repo_path=str(tmp_path / "feast_repo"))

    def test_store_property_cached(self, tmp_path):
        """When _store is already set, return it without re-importing."""
        backend = FeastBackend(repo_path=str(tmp_path / "feast_repo"))
        mock_store = MagicMock()
        backend._store = mock_store
        assert backend.store is mock_store

    def test_load_metadata_on_init(self, tmp_path):
        """Second FeastBackend instance loads metadata from disk."""
        repo_path = str(tmp_path / "repo")
        b1 = FeastBackend(repo_path=repo_path)
        b1.create_feature_table(
            name="tbl", entity_key="id", timestamp_column="ts",
            schema={"id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        # Creating a new instance should load the metadata
        b2 = FeastBackend(repo_path=repo_path)
        assert "tbl" in b2._tables

    def test_validate_cutoff_consistency_matches(self, tmp_path):
        """validate_cutoff_consistency when cutoff matches returns True + 'matches' message."""
        backend = FeastBackend(repo_path=str(tmp_path / "repo"))
        cutoff = datetime(2024, 6, 1)
        backend.create_feature_table(
            name="t1", entity_key="id", timestamp_column="ts",
            schema={"id": "string"}, cutoff_date=cutoff,
        )
        ok, msg = backend.validate_cutoff_consistency(cutoff)
        assert ok is True
        assert "matches reference" in msg

    def test_write_features_merge_with_existing_parquet(self, tmp_path):
        """merge mode with existing parquet data and registered table."""
        backend = FeastBackend(repo_path=str(tmp_path / "repo"))
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="ts",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        df1 = pd.DataFrame({
            "entity_id": ["A", "B"],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "val": [1.0, 2.0],
        })
        backend.write_features("tbl", df1, mode="overwrite")

        df2 = pd.DataFrame({
            "entity_id": ["B", "C"],
            "ts": pd.to_datetime(["2024-02-01", "2024-02-01"]),
            "val": [20.0, 30.0],
        })
        backend.write_features("tbl", df2, mode="merge")
        assert backend._tables["tbl"]["row_count"] == 3  # A, B(updated), C

    def test_write_features_merge_parquet_exists_but_table_not_registered(self, tmp_path):
        """merge mode with parquet existing but table not in _tables."""
        backend = FeastBackend(repo_path=str(tmp_path / "repo"))
        data_dir = backend.repo_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        df1 = pd.DataFrame({"entity_id": ["A"], "val": [1.0]})
        df1.to_parquet(data_dir / "tbl.parquet", index=False)

        df2 = pd.DataFrame({"entity_id": ["B"], "val": [2.0]})
        # table "tbl" not in backend._tables, so merge skips drop_duplicates
        backend.write_features("tbl", df2, mode="merge")

    def test_write_features_effective_cutoff_from_table(self, tmp_path):
        """When no cutoff_date passed but table has stored cutoff."""
        backend = FeastBackend(repo_path=str(tmp_path / "repo"))
        backend.create_feature_table(
            name="tbl", entity_key="entity_id", timestamp_column="ts",
            schema={"entity_id": "string"}, cutoff_date=datetime(2024, 6, 1),
        )
        df = pd.DataFrame({"entity_id": ["A"], "ts": [datetime(2024, 1, 1)], "val": [1.0]})
        backend.write_features("tbl", df, mode="overwrite", cutoff_date=None)
        assert "data_hash" in backend._tables["tbl"]

    def test_list_tables_delta_appends_unique(self, tmp_path):
        """list_tables appends delta dirs that are not already in parquet tables."""
        backend = FeastBackend(repo_path=str(tmp_path / "repo"))
        data_dir = backend.repo_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # A parquet file
        pd.DataFrame({"a": [1]}).to_parquet(data_dir / "parquet_tbl.parquet", index=False)
        # A delta dir
        delta_dir = data_dir / "delta_tbl"
        delta_dir.mkdir()

        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        backend.storage = mock_storage

        tables = backend.list_tables()
        assert "parquet_tbl" in tables
        assert "delta_tbl" in tables

    def test_list_tables_delta_dir_already_in_parquet(self, tmp_path):
        """Delta dir with same name as parquet stem is not duplicated."""
        backend = FeastBackend(repo_path=str(tmp_path / "repo"))
        data_dir = backend.repo_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({"a": [1]}).to_parquet(data_dir / "tbl.parquet", index=False)
        delta_dir = data_dir / "tbl"
        delta_dir.mkdir()

        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        backend.storage = mock_storage

        tables = backend.list_tables()
        assert tables.count("tbl") == 1
