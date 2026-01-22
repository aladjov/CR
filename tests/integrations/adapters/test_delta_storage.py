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


class TestDeltaStorageInterface:
    def test_delta_storage_is_abstract(self):
        from customer_retention.integrations.adapters.storage import DeltaStorage
        assert issubclass(DeltaStorage, ABC)

    def test_local_delta_implements_interface(self):
        from customer_retention.integrations.adapters.storage import LocalDelta, DeltaStorage
        assert issubclass(LocalDelta, DeltaStorage)

    def test_databricks_delta_implements_interface(self):
        from customer_retention.integrations.adapters.storage import DatabricksDelta, DeltaStorage
        assert issubclass(DatabricksDelta, DeltaStorage)


class TestAdapterResult:
    def test_adapter_result_creation(self):
        from customer_retention.integrations.adapters.base import AdapterResult
        result = AdapterResult(success=True)
        assert result.success is True
        assert result.data is None
        assert result.error is None

    def test_adapter_result_with_data(self):
        from customer_retention.integrations.adapters.base import AdapterResult
        result = AdapterResult(success=True, data={"key": "value"})
        assert result.data == {"key": "value"}

    def test_adapter_result_with_error(self):
        from customer_retention.integrations.adapters.base import AdapterResult
        result = AdapterResult(success=False, error="Something failed")
        assert result.success is False
        assert result.error == "Something failed"

    def test_adapter_result_with_metadata(self):
        from customer_retention.integrations.adapters.base import AdapterResult
        result = AdapterResult(success=True, metadata={"version": 1})
        assert result.metadata == {"version": 1}


@requires_delta
class TestLocalDeltaRead:
    def test_read_returns_dataframe(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        storage.write(df, delta_path)
        result = storage.read(delta_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_read_with_version(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        storage.write(df1, delta_path)
        storage.write(df2, delta_path, mode="append")
        result_v0 = storage.read(delta_path, version=0)
        assert len(result_v0) == 2


@requires_delta
class TestLocalDeltaWrite:
    def test_write_creates_table(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        storage.write(df, delta_path)
        assert Path(delta_path).exists()

    def test_write_overwrite_mode(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4, 5]})
        storage.write(df1, delta_path)
        storage.write(df2, delta_path, mode="overwrite")
        result = storage.read(delta_path)
        assert len(result) == 3

    def test_write_append_mode(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2]})
        storage.write(df, delta_path)
        storage.write(df, delta_path, mode="append")
        result = storage.read(delta_path)
        assert len(result) == 4

    def test_write_with_partition(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2, 3], "part": ["x", "y", "x"]})
        storage.write(df, delta_path, partition_by=["part"])
        assert Path(delta_path).exists()


@requires_delta
class TestLocalDeltaMerge:
    def test_merge_updates_existing(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        storage.write(df1, delta_path)
        df2 = pd.DataFrame({"id": [2, 3], "value": [25, 30]})
        storage.merge(df2, delta_path, condition="source.id = target.id", update_cols=["value"])
        result = storage.read(delta_path)
        assert len(result) == 3
        assert result[result["id"] == 2]["value"].values[0] == 25


@requires_delta
class TestLocalDeltaHistory:
    def test_history_returns_list(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2]})
        storage.write(df, delta_path)
        history = storage.history(delta_path)
        assert isinstance(history, list)
        assert len(history) >= 1

    def test_history_grows_with_writes(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2]})
        storage.write(df, delta_path)
        storage.write(df, delta_path, mode="append")
        history = storage.history(delta_path)
        assert len(history) >= 2


@requires_delta
class TestLocalDeltaVacuum:
    def test_vacuum_runs_without_error(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2]})
        storage.write(df, delta_path)
        storage.vacuum(delta_path, retention_hours=0)


class TestDatabricksDeltaMocked:
    def test_databricks_delta_requires_spark(self):
        from customer_retention.integrations.adapters.storage import DatabricksDelta
        from customer_retention.core.compat.detection import is_spark_available
        if not is_spark_available():
            with pytest.raises(ImportError):
                storage = DatabricksDelta()
                storage.read("path")
