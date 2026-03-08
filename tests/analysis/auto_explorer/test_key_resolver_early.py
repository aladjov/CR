from __future__ import annotations

import pytest

from customer_retention.analysis.auto_explorer.active_dataset_store import save_active_dataset
from customer_retention.analysis.auto_explorer.key_resolver import (
    _apply_resolution_step,
    resolve_single_dataset_keys,
)
from customer_retention.analysis.auto_explorer.project_context import KeyResolutionStep
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.compat import pd


class TestApplyResolutionStepIdempotent:
    def test_returns_unchanged_when_resolve_column_exists(self):
        df = pd.DataFrame({"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]})
        bridge_df = pd.DataFrame({"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]})
        step = KeyResolutionStep(
            bridge_dataset="case",
            source_key="CASE_ID",
            bridge_key="CASE_ID",
            resolve_column="ACCOUNT_ID",
        )
        result = _apply_resolution_step(df, "target", step, {"case": bridge_df})
        assert list(result.columns) == ["CASE_ID", "ACCOUNT_ID"]
        assert len(result) == 2

    def test_returns_unchanged_when_resolve_column_exists_case_insensitive(self):
        df = pd.DataFrame({"case_id": [1, 2], "account_id": ["A1", "A2"]})
        bridge_df = pd.DataFrame({"case_id": [1, 2], "account_id": ["A1", "A2"]})
        step = KeyResolutionStep(
            bridge_dataset="case",
            source_key="case_id",
            bridge_key="case_id",
            resolve_column="ACCOUNT_ID",
        )
        result = _apply_resolution_step(df, "target", step, {"case": bridge_df})
        assert len(result) == 2
        assert "account_id" in result.columns


class TestApplyResolutionStepCaseInsensitive:
    def test_joins_when_column_cases_differ(self):
        df = pd.DataFrame({"case_id": [1, 2, 3], "status": ["Open", "Closed", "Open"]})
        bridge_df = pd.DataFrame({"CASE_ID": [1, 2, 3], "ACCOUNT_ID": ["A1", "A2", "A3"]})
        step = KeyResolutionStep(
            bridge_dataset="case",
            source_key="case_id",
            bridge_key="CASE_ID",
            resolve_column="ACCOUNT_ID",
        )
        result = _apply_resolution_step(df, "target", step, {"case": bridge_df})
        assert "ACCOUNT_ID" in result.columns
        assert list(result["ACCOUNT_ID"]) == ["A1", "A2", "A3"]

    def test_joins_when_bridge_key_and_resolve_column_lowercase(self):
        df = pd.DataFrame({"CASE_ID": [1, 2], "STATUS": ["Open", "Closed"]})
        bridge_df = pd.DataFrame({"case_id": [1, 2], "account_id": ["A1", "A2"]})
        step = KeyResolutionStep(
            bridge_dataset="case",
            source_key="CASE_ID",
            bridge_key="CASE_ID",
            resolve_column="ACCOUNT_ID",
        )
        result = _apply_resolution_step(df, "target", step, {"case": bridge_df})
        assert "account_id" in result.columns
        assert list(result["account_id"]) == ["A1", "A2"]


class TestResolveSingleDatasetKeys:
    @pytest.fixture()
    def namespace(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="early-resolve-test")
        ns.setup()
        return ns

    def test_resolves_via_bridge_in_landing(self, namespace):
        bridge_df = pd.DataFrame({"CASE_ID": [1, 2, 3], "ACCOUNT_ID": ["A1", "A2", "A3"]})
        save_active_dataset(namespace, "case", bridge_df)
        source_df = pd.DataFrame({"CASE_ID": [1, 2, 3], "STATUS": ["Open", "Closed", "Open"]})
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        result = resolve_single_dataset_keys(source_df, steps, namespace)
        assert "ACCOUNT_ID" in result.columns
        assert list(result["ACCOUNT_ID"]) == ["A1", "A2", "A3"]

    def test_multi_hop_resolution(self, namespace):
        item_df = pd.DataFrame({"ITEM_ID": [10, 20], "ORDER_ID": [100, 200]})
        order_df = pd.DataFrame({"ORDER_ID": [100, 200], "ACCOUNT_ID": ["A1", "A2"]})
        save_active_dataset(namespace, "item", item_df)
        save_active_dataset(namespace, "order", order_df)
        source_df = pd.DataFrame({"ITEM_ID": [10, 20], "VALUE": [5.0, 10.0]})
        steps = [
            KeyResolutionStep(
                bridge_dataset="item",
                source_key="ITEM_ID",
                bridge_key="ITEM_ID",
                resolve_column="ORDER_ID",
            ),
            KeyResolutionStep(
                bridge_dataset="order",
                source_key="ORDER_ID",
                bridge_key="ORDER_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        result = resolve_single_dataset_keys(source_df, steps, namespace)
        assert "ACCOUNT_ID" in result.columns
        assert list(result["ACCOUNT_ID"]) == ["A1", "A2"]

    def test_bridge_missing_raises_clear_error(self, namespace):
        source_df = pd.DataFrame({"CASE_ID": [1, 2]})
        steps = [
            KeyResolutionStep(
                bridge_dataset="missing_bridge",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        with pytest.raises(FileNotFoundError, match="missing_bridge"):
            resolve_single_dataset_keys(source_df, steps, namespace)

    def test_idempotent_when_already_resolved(self, namespace):
        bridge_df = pd.DataFrame({"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]})
        save_active_dataset(namespace, "case", bridge_df)
        source_df = pd.DataFrame(
            {"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"], "STATUS": ["Open", "Closed"]}
        )
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        result = resolve_single_dataset_keys(source_df, steps, namespace)
        assert list(result.columns) == ["CASE_ID", "ACCOUNT_ID", "STATUS"]
        assert len(result) == 2
