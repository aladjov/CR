from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    import deltalake
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False

requires_delta = pytest.mark.skipif(not DELTA_AVAILABLE, reason="deltalake not installed")


@requires_delta
class TestLocalDeltaOptimizeCompact:
    def test_optimize_compact_preserves_data(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "compact_table")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        storage.write(df, delta_path)
        storage.optimize(delta_path)
        result = storage.read(delta_path)
        pd.testing.assert_frame_equal(result.sort_values("a").reset_index(drop=True), df)

    def test_optimize_compact_after_multiple_writes(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "multi_write")
        for i in range(3):
            storage.write(pd.DataFrame({"x": [i]}), delta_path, mode="append" if i > 0 else "overwrite")
        storage.optimize(delta_path)
        result = storage.read(delta_path)
        assert len(result) == 3


@requires_delta
class TestLocalDeltaOptimizeZOrder:
    def test_z_order_preserves_data(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "zorder_table")
        df = pd.DataFrame({"entity_id": [1, 2, 3], "as_of_date": ["2024-01-01", "2024-01-02", "2024-01-03"], "val": [10, 20, 30]})
        storage.write(df, delta_path)
        storage.optimize(delta_path, z_order_columns=["entity_id", "as_of_date"])
        result = storage.read(delta_path)
        pd.testing.assert_frame_equal(result.sort_values("entity_id").reset_index(drop=True), df)

    def test_z_order_single_column(self, tmp_path):
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = LocalDelta()
        delta_path = str(tmp_path / "zorder_single")
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        storage.write(df, delta_path)
        storage.optimize(delta_path, z_order_columns=["id"])
        result = storage.read(delta_path)
        assert len(result) == 2


class TestDatabricksDeltaOptimizeCompaction:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_calls_execute_compaction(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()
            with patch("delta.tables.DeltaTable") as mock_dt_cls:
                mock_dt = MagicMock()
                mock_dt_cls.forPath.return_value = mock_dt
                storage.optimize("/some/path")
                mock_dt_cls.forPath.assert_called_once_with(storage.spark, "/some/path")
                mock_dt.optimize.return_value.executeCompaction.assert_called_once()

    def test_normalizes_dbfs_path(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()
            with patch("delta.tables.DeltaTable") as mock_dt_cls:
                mock_dt = MagicMock()
                mock_dt_cls.forPath.return_value = mock_dt
                storage.optimize("/dbfs/mnt/data/table")
                mock_dt_cls.forPath.assert_called_once_with(storage.spark, "/mnt/data/table")


class TestDatabricksDeltaOptimizeZOrder:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_calls_execute_z_order_by(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()
            with patch("delta.tables.DeltaTable") as mock_dt_cls:
                mock_dt = MagicMock()
                mock_dt_cls.forPath.return_value = mock_dt
                storage.optimize("/path", z_order_columns=["entity_id", "as_of_date"])
                mock_dt.optimize.return_value.executeZOrderBy.assert_called_once_with(["entity_id", "as_of_date"])
                mock_dt.optimize.return_value.executeCompaction.assert_not_called()

    def test_empty_z_order_columns_falls_back_to_compact(self):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        with patch.object(DatabricksDelta, "__init__", lambda self: None):
            storage = DatabricksDelta()
            storage._spark = MagicMock()
            with patch("delta.tables.DeltaTable") as mock_dt_cls:
                mock_dt = MagicMock()
                mock_dt_cls.forPath.return_value = mock_dt
                storage.optimize("/path", z_order_columns=[])
                mock_dt.optimize.return_value.executeCompaction.assert_called_once()
