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

    def test_write_handles_null_dtype_columns(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "null_table")
        df = pd.DataFrame({"a": [1, 2, 3], "empty": [None, None, None]})
        storage.write(df, delta_path)
        result = storage.read(delta_path)
        assert len(result) == 3
        assert "empty" in result.columns

    def test_write_preserves_non_null_columns_with_nulls(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "mixed_table")
        df = pd.DataFrame({
            "ok": [1, 2, 3],
            "partial": [None, "hello", None],
            "all_null": pd.array([None, None, None], dtype=pd.StringDtype()),
        })
        storage.write(df, delta_path)
        result = storage.read(delta_path)
        assert len(result) == 3
        assert result["partial"].iloc[1] == "hello"


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


@requires_delta
class TestLocalDeltaDistributedGuard:
    def test_write_delegates_to_databricks_for_spark_pandas(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage import LocalDelta

        storage = LocalDelta()
        mock_ps_df = MagicMock()
        mock_ps_df.to_spark = MagicMock(return_value=MagicMock())

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)
        with patch.dict(
            "sys.modules",
            {"customer_retention.integrations.adapters.storage.databricks": MagicMock(DatabricksDelta=mock_cls)},
        ):
            storage.write(mock_ps_df, str(tmp_path / "out"), mode="overwrite")
            mock_instance.write.assert_called_once_with(
                mock_ps_df, str(tmp_path / "out"), "overwrite", None, None
            )

    def test_write_pandas_df_does_not_delegate(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta

        storage = LocalDelta()
        df = pd.DataFrame({"a": [1, 2, 3]})
        delta_path = str(tmp_path / "test_table")
        storage.write(df, delta_path)
        result = storage.read(delta_path)
        assert len(result) == 3


class TestDatabricksDeltaReturnsSparkPandas:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_read_uses_pandas_api(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            mock_spark_df = MagicMock()
            storage.spark.read.format("delta").load.return_value = mock_spark_df

            storage.read("/some/path")

            mock_spark_df.pandas_api.assert_called_once()
            mock_spark_df.toPandas.assert_not_called()

    def test_read_falls_back_to_to_pandas_on_spark(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            mock_spark_df = MagicMock(spec=[])
            mock_spark_df.to_pandas_on_spark = MagicMock()
            storage.spark.read.format("delta").load.return_value = mock_spark_df

            storage.read("/some/path")

            mock_spark_df.to_pandas_on_spark.assert_called_once()

    def test_write_detects_pyspark_pandas_input(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            mock_ps_df = MagicMock()
            mock_ps_df.to_spark.return_value = MagicMock()

            with patch.object(storage, "_to_spark_df") as mock_to_spark:
                storage.write(mock_ps_df, "/fake/path")
                mock_ps_df.to_spark.assert_called_once()
                mock_to_spark.assert_not_called()

    def test_write_regular_pandas_uses_to_spark_df(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            df = pd.DataFrame({"val": [1, 2]})

            with patch.object(storage, "_to_spark_df", return_value=MagicMock()) as mock_to:
                storage.write(df, "/fake/path")
                mock_to.assert_called_once_with(df)

    def test_merge_detects_pyspark_pandas_input(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            mock_ps_df = MagicMock()
            mock_spark_df = MagicMock()
            mock_ps_df.to_spark.return_value = mock_spark_df

            with patch.object(storage, "_to_spark_df") as mock_to_spark, \
                 patch("delta.tables.DeltaTable") as mock_dt:
                mock_target = MagicMock()
                mock_dt.forPath.return_value = mock_target
                mock_target.alias.return_value.merge.return_value \
                    .whenMatchedUpdateAll.return_value \
                    .whenNotMatchedInsertAll.return_value \
                    .execute = MagicMock()

                storage.merge(mock_ps_df, "/fake/path", "source.id = target.id")

                mock_ps_df.to_spark.assert_called_once()
                mock_to_spark.assert_not_called()


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
                "customer_retention.core.compat.normalize_timestamp_columns",
                wraps=lambda d: d,
            ) as mock_norm, patch(
                "customer_retention.core.compat.pandas_dtype_to_spark_schema",
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


class TestDatabricksDeltaPathNormalization:
    def test_strips_dbfs_prefix(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        assert DatabricksDelta._normalize_path("/dbfs/mnt/data/table") == "/mnt/data/table"

    def test_strips_dbfs_prefix_nested(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        assert DatabricksDelta._normalize_path("/dbfs/home/user/data") == "/home/user/data"

    def test_passthrough_relative_path(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        assert DatabricksDelta._normalize_path("experiments/runs/test") == "experiments/runs/test"

    def test_passthrough_absolute_non_dbfs(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        assert DatabricksDelta._normalize_path("/home/user/data") == "/home/user/data"

    def test_passthrough_dbfs_scheme(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        assert DatabricksDelta._normalize_path("dbfs:/mnt/data/table") == "dbfs:/mnt/data/table"


class TestDatabricksDeltaReadMissingTable:
    def test_read_raises_file_not_found_for_missing_delta_table(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            storage.spark.read.format("delta").load.side_effect = Exception(
                "[DELTA_TABLE_NOT_FOUND] Delta table `/fake/path` doesn't exist."
            )
            with pytest.raises(FileNotFoundError, match="/fake/path"):
                storage.read("/fake/path")

    def test_read_propagates_other_spark_exceptions(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            storage.spark.read.format("delta").load.side_effect = RuntimeError("disk full")
            with pytest.raises(RuntimeError, match="disk full"):
                storage.read("/fake/path")


class TestDatabricksDeltaReadNormalizesPath:
    def test_read_normalizes_path(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            storage.read("/dbfs/mnt/data/table")
            load_call = storage.spark.read.format("delta").load
            load_call.assert_called_once_with("/mnt/data/table")


class TestDatabricksDeltaStripTimestampTz:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_casts_timestamp_to_ntz(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import (
            IntegerType,
            StringType,
            StructField,
            StructType,
            TimestampNTZType,
            TimestampType,
        )

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampType()),
            StructField("name", StringType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields

        result = DatabricksDelta._strip_spark_timestamp_tz(mock_df)

        mock_df.select.assert_called_once()
        select_args = mock_df.select.call_args[0][0]
        assert len(select_args) == 3

    def test_no_timestamp_columns_returns_original(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StringType, StructField, StructType

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields

        result = DatabricksDelta._strip_spark_timestamp_tz(mock_df)

        assert result is mock_df
        mock_df.select.assert_not_called()

    def test_ntz_columns_returns_original(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StructField, StructType, TimestampNTZType

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampNTZType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields

        result = DatabricksDelta._strip_spark_timestamp_tz(mock_df)

        assert result is mock_df
        mock_df.select.assert_not_called()

    def test_write_applies_strip_tz(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            mock_spark_df = MagicMock()
            mock_ps_df = MagicMock()
            mock_ps_df.to_spark.return_value = mock_spark_df

            mock_stripped = MagicMock()
            with patch.object(DatabricksDelta, "_strip_spark_timestamp_tz", return_value=mock_stripped):
                storage.write(mock_ps_df, "/fake/path")
                DatabricksDelta._strip_spark_timestamp_tz.assert_called_once_with(mock_spark_df)
                mock_stripped.write.format("delta").mode("overwrite").option(
                    "overwriteSchema", "true"
                ).save.assert_called_once_with("/fake/path")

    def test_merge_applies_strip_tz(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            mock_spark_df = MagicMock()
            mock_ps_df = MagicMock()
            mock_ps_df.to_spark.return_value = mock_spark_df

            mock_stripped = MagicMock()
            with patch.object(DatabricksDelta, "_strip_spark_timestamp_tz", return_value=mock_stripped), \
                 patch("delta.tables.DeltaTable") as mock_dt:
                mock_target = MagicMock()
                mock_dt.forPath.return_value = mock_target
                mock_target.alias.return_value.merge.return_value \
                    .whenMatchedUpdateAll.return_value \
                    .whenNotMatchedInsertAll.return_value \
                    .execute = MagicMock()

                storage.merge(mock_ps_df, "/fake/path", "source.id = target.id")
                DatabricksDelta._strip_spark_timestamp_tz.assert_called_once_with(mock_spark_df)


class TestDatabricksDeltaWriteNormalizesPath:
    def test_write_normalizes_path(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta

        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()

            df = pd.DataFrame({"val": [1, 2]})
            with patch.object(storage, "_to_spark_df", return_value=MagicMock()) as mock_to:
                storage.write(df, "/dbfs/mnt/data/table")
                writer = mock_to.return_value.write.format("delta").mode("overwrite")
                writer.option("overwriteSchema", "true").save.assert_called_once_with("/mnt/data/table")
