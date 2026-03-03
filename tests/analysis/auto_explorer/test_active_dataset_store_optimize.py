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


class TestSaveActiveDatasetOptimize:
    def test_save_calls_optimize_with_z_order(self, namespace, sample_df):
        with patch("customer_retention.analysis.auto_explorer.active_dataset_store.optimize_delta") as mock_opt:
            from customer_retention.analysis.auto_explorer.active_dataset_store import save_active_dataset
            path = save_active_dataset(namespace, "customers", sample_df, z_order_columns=["customer_id"])
            mock_opt.assert_called_once_with(str(path), ["customer_id"])

    def test_save_skips_optimize_without_z_order(self, namespace, sample_df):
        with patch("customer_retention.analysis.auto_explorer.active_dataset_store.optimize_delta") as mock_opt:
            from customer_retention.analysis.auto_explorer.active_dataset_store import save_active_dataset
            save_active_dataset(namespace, "customers", sample_df)
            mock_opt.assert_not_called()


class TestSaveAggregatedDatasetOptimize:
    def test_save_calls_optimize_with_z_order(self, namespace, sample_df):
        with patch("customer_retention.analysis.auto_explorer.active_dataset_store.optimize_delta") as mock_opt:
            from customer_retention.analysis.auto_explorer.active_dataset_store import save_aggregated_dataset
            path = save_aggregated_dataset(namespace, "events", sample_df, z_order_columns=["customer_id"])
            mock_opt.assert_called_once_with(str(path), ["customer_id"])

    def test_save_skips_optimize_without_z_order(self, namespace, sample_df):
        with patch("customer_retention.analysis.auto_explorer.active_dataset_store.optimize_delta") as mock_opt:
            from customer_retention.analysis.auto_explorer.active_dataset_store import save_aggregated_dataset
            save_aggregated_dataset(namespace, "events", sample_df)
            mock_opt.assert_not_called()
