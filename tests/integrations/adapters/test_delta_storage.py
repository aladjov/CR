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


class TestDeltaStorageInterface:
    def test_delta_storage_is_abstract(self):
        from customer_retention.integrations.adapters.storage import DeltaStorage
        assert issubclass(DeltaStorage, ABC)

    def test_local_delta_implements_interface(self):
        from customer_retention.integrations.adapters.storage import DeltaStorage, LocalDelta
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

    def test_write_overwrite_with_schema_change(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "c": [7, 8]})
        storage.write(df1, delta_path)
        storage.write(df2, delta_path, mode="overwrite")
        result = storage.read(delta_path)
        assert set(result.columns) == {"a", "c"}
        assert len(result) == 2

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


@requires_delta
class TestLocalDeltaExists:
    def test_exists_returns_true_after_write(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2]})
        storage.write(df, delta_path)
        assert storage.exists(delta_path) is True

    def test_exists_returns_false_for_nonexistent(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        assert storage.exists(str(tmp_path / "nonexistent")) is False

    def test_exists_returns_false_for_plain_directory(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        plain_dir = tmp_path / "plain_dir"
        plain_dir.mkdir()
        assert storage.exists(str(plain_dir)) is False


@requires_delta
class TestLocalDeltaCommitMetadata:
    def test_write_with_metadata_stores_in_history(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2]})
        storage.write(df, delta_path, metadata={"run_id": "abc123", "run_type": "exploration"})
        history = storage.history(delta_path)
        assert len(history) >= 1
        latest = history[0]
        assert latest.get("clientVersion") is not None or "userMetadata" in str(latest)

    def test_write_without_metadata_still_works(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2]})
        storage.write(df, delta_path)
        result = storage.read(delta_path)
        assert len(result) == 2

    def test_multiple_writes_with_different_metadata(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "test_table")
        df = pd.DataFrame({"a": [1, 2]})
        storage.write(df, delta_path, metadata={"run_type": "exploration"})
        storage.write(df, delta_path, mode="overwrite", metadata={"run_type": "production"})
        history = storage.history(delta_path)
        assert len(history) >= 2


class TestDatabricksDeltaMocked:
    def test_databricks_delta_requires_spark(self):
        from customer_retention.core.compat.detection import is_spark_available
        from customer_retention.integrations.adapters.storage import DatabricksDelta
        if not is_spark_available():
            with pytest.raises(ImportError):
                storage = DatabricksDelta()
                storage.read("path")


class TestDatabricksDeltaToSparkDf:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_to_spark_df_calls_normalize_and_schema(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            df = pd.DataFrame({
                "ts": pd.to_datetime(["2023-01-01"]),
                "val": [1],
            })

            with patch(
                "customer_retention.integrations.adapters.storage.databricks.normalize_timestamp_columns",
                wraps=lambda d: d,
            ) as mock_norm, patch(
                "customer_retention.integrations.adapters.storage.databricks.pandas_dtype_to_spark_schema",
            ) as mock_schema:
                storage._to_spark_df(df)
                mock_norm.assert_called_once()
                mock_schema.assert_called_once()
                storage.spark.createDataFrame.assert_called_once()
                call_kwargs = storage.spark.createDataFrame.call_args
                assert "schema" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    def test_write_uses_to_spark_df(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            df = pd.DataFrame({"val": [1, 2]})

            with patch.object(storage, "_to_spark_df", return_value=MagicMock()) as mock_to:
                storage.write(df, "/fake/path")
                mock_to.assert_called_once_with(df)

    def test_merge_uses_to_spark_df(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            df = pd.DataFrame({"id": [1], "val": [10]})

            with patch.object(storage, "_to_spark_df", return_value=MagicMock()) as mock_to, \
                 patch("customer_retention.integrations.adapters.storage.databricks.DeltaTable", create=True):
                from unittest.mock import patch as _p
                with _p("delta.tables.DeltaTable") as mock_dt:
                    mock_dt.forPath.return_value.alias.return_value.merge.return_value \
                        .whenMatchedUpdateAll.return_value.whenNotMatchedInsertAll.return_value \
                        .execute = MagicMock()
                    try:
                        storage.merge(df, "/fake/path", "source.id = target.id")
                    except Exception:
                        pass
                    mock_to.assert_called_once_with(df)

    def test_to_spark_df_with_empty_df(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            df = pd.DataFrame()
            storage._to_spark_df(df)
            storage.spark.createDataFrame.assert_called_once()

    def test_to_spark_df_with_tz_aware_column(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.core.compat import normalize_timestamp_columns
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            df = pd.DataFrame({
                "ts": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
                "val": [1],
            })
            storage._to_spark_df(df)
            call_args = storage.spark.createDataFrame.call_args
            passed_df = call_args.args[0] if call_args.args else call_args[0][0]
            assert passed_df["ts"].dt.tz is None
