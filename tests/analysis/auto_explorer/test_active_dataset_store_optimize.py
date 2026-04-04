from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace


@pytest.fixture()
def namespace(tmp_path):
    ns = RunNamespace(root=tmp_path, run_id="proj-opt1234")
    ns.setup()
    return ns


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"customer_id": [1, 2, 3], "revenue": [100.0, 200.0, 300.0]})


class TestOptimizeDelta:
    def test_calls_optimize_with_z_order(self, tmp_path):
        from customer_retention.analysis.auto_explorer.active_dataset_store import optimize_delta
        with patch("customer_retention.analysis.auto_explorer.active_dataset_store.get_delta") as mock_get:
            mock_delta = MagicMock()
            mock_get.return_value = mock_delta
            optimize_delta(str(tmp_path / "table"), ["entity_id", "as_of_date"])
            mock_delta.optimize.assert_called_once_with(str(tmp_path / "table"), ["entity_id", "as_of_date"])

    def test_calls_optimize_compact_when_no_columns(self, tmp_path):
        from customer_retention.analysis.auto_explorer.active_dataset_store import optimize_delta
        with patch("customer_retention.analysis.auto_explorer.active_dataset_store.get_delta") as mock_get:
            mock_delta = MagicMock()
            mock_get.return_value = mock_delta
            optimize_delta(str(tmp_path / "table"))
            mock_delta.optimize.assert_called_once_with(str(tmp_path / "table"), None)

    def test_calls_optimize_compact_when_empty_list(self, tmp_path):
        from customer_retention.analysis.auto_explorer.active_dataset_store import optimize_delta
        with patch("customer_retention.analysis.auto_explorer.active_dataset_store.get_delta") as mock_get:
            mock_delta = MagicMock()
            mock_get.return_value = mock_delta
            optimize_delta(str(tmp_path / "table"), [])
            mock_delta.optimize.assert_called_once_with(str(tmp_path / "table"), None)


class TestSaveActiveDatasetZOrder:
    def test_save_applies_z_order_via_write(self, namespace):
        import deltalake

        from customer_retention.analysis.auto_explorer.active_dataset_store import save_active_dataset
        df = pd.DataFrame({"customer_id": [1, 2, 3], "revenue": [100.0, 200.0, 300.0]})
        path = save_active_dataset(namespace, "customers", df, z_order_columns=["customer_id"])
        dt = deltalake.DeltaTable(str(path))
        ops = [entry["operation"] for entry in dt.history()]
        assert "OPTIMIZE" in ops

    def test_save_skips_optimize_without_z_order(self, namespace, sample_df):
        import deltalake

        from customer_retention.analysis.auto_explorer.active_dataset_store import save_active_dataset
        save_active_dataset(namespace, "customers", sample_df)
        dt = deltalake.DeltaTable(str(namespace.landing_table_dir("customers")))
        ops = [entry["operation"] for entry in dt.history()]
        assert "OPTIMIZE" not in ops


class TestSaveAggregatedDatasetZOrder:
    def test_save_applies_z_order_via_write(self, namespace):
        import deltalake

        from customer_retention.analysis.auto_explorer.active_dataset_store import save_aggregated_dataset
        df = pd.DataFrame({"customer_id": [1, 2, 3], "revenue": [100.0, 200.0, 300.0]})
        path = save_aggregated_dataset(namespace, "events", df, z_order_columns=["customer_id"])
        dt = deltalake.DeltaTable(str(path))
        ops = [entry["operation"] for entry in dt.history()]
        assert "OPTIMIZE" in ops

    def test_save_skips_optimize_without_z_order(self, namespace, sample_df):
        import deltalake

        from customer_retention.analysis.auto_explorer.active_dataset_store import save_aggregated_dataset
        save_aggregated_dataset(namespace, "events", sample_df)
        dt = deltalake.DeltaTable(str(namespace.bronze_table_dir("events")))
        ops = [entry["operation"] for entry in dt.history()]
        assert "OPTIMIZE" not in ops


class TestDeltaWriteSummary:
    def test_returns_file_count(self, namespace, sample_df):
        from customer_retention.analysis.auto_explorer.active_dataset_store import (
            delta_write_summary,
            save_active_dataset,
        )
        save_active_dataset(namespace, "customers", sample_df)
        ws = delta_write_summary(str(namespace.landing_table_dir("customers")))
        assert isinstance(ws.get("files"), int)
        assert ws["files"] >= 1

    def test_detects_optimize_in_history(self, namespace):
        from customer_retention.analysis.auto_explorer.active_dataset_store import (
            delta_write_summary,
            save_active_dataset,
        )
        df = pd.DataFrame({"customer_id": [1, 2], "revenue": [10.0, 20.0]})
        save_active_dataset(namespace, "customers", df, z_order_columns=["customer_id"])
        ws = delta_write_summary(str(namespace.landing_table_dir("customers")))
        assert ws.get("optimize_verified") is True

    def test_no_optimize_when_no_z_order(self, namespace, sample_df):
        from customer_retention.analysis.auto_explorer.active_dataset_store import (
            delta_write_summary,
            save_active_dataset,
        )
        save_active_dataset(namespace, "customers", sample_df)
        ws = delta_write_summary(str(namespace.landing_table_dir("customers")))
        assert ws.get("optimize_verified") is None


class TestPrintWriteReport:
    def test_prints_via_console(self, namespace, sample_df, capsys, monkeypatch):
        import customer_retention.analysis.visualization.console as console_mod
        monkeypatch.setattr(console_mod, "HAS_IPYTHON", False)

        from customer_retention.analysis.auto_explorer.active_dataset_store import (
            print_write_report,
            save_active_dataset,
        )
        save_active_dataset(namespace, "customers", sample_df)
        path = str(namespace.landing_table_dir("customers"))
        print_write_report("Test Report", path, 3, 2, ["customer_id"])
        captured = capsys.readouterr()
        assert "TEST REPORT" in captured.out
        assert "Rows" in captured.out
        assert "3" in captured.out
        assert "Delta files" in captured.out
        assert "customer_id" in captured.out
