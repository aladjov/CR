from __future__ import annotations

import logging

import pytest

from customer_retention.analysis.auto_explorer.active_dataset_store import save_active_dataset
from customer_retention.analysis.auto_explorer.key_resolver import (
    _column_exists,
    _is_empty,
    _StaleStepError,
    resolve_entity_keys,
    resolve_sample_ids_via_bridge,
    resolve_single_dataset_keys,
    suggest_key_resolutions,
)
from customer_retention.analysis.auto_explorer.project_context import KeyResolutionStep
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.compat import pd


def _case_frames():
    case_history = pd.DataFrame(
        {"CASE_ID": [1, 2, 3], "STATUS": ["Open", "Closed", "Open"]}
    )
    case = pd.DataFrame(
        {"CASE_ID": [1, 2, 3], "ACCOUNT_ID": ["A1", "A2", "A3"]}
    )
    return {"case_history": case_history, "case": case}


class TestResolveEntityKeys:
    def test_single_hop_resolution(self):
        frames = _case_frames()
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="case",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        result = resolve_entity_keys(frames, resolutions)
        assert "ACCOUNT_ID" in result["case_history"].columns
        assert list(result["case_history"]["ACCOUNT_ID"]) == ["A1", "A2", "A3"]

    def test_multi_step_resolution(self):
        source = pd.DataFrame({"ITEM_ID": [10, 20]})
        item = pd.DataFrame({"ITEM_ID": [10, 20], "ORDER_ID": [100, 200]})
        order = pd.DataFrame({"ORDER_ID": [100, 200], "ACCOUNT_ID": ["A1", "A2"]})
        frames = {"source": source, "item": item, "order": order}
        resolutions = {
            "source": [
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
            ],
        }
        result = resolve_entity_keys(frames, resolutions)
        assert "ORDER_ID" in result["source"].columns
        assert "ACCOUNT_ID" in result["source"].columns
        assert list(result["source"]["ACCOUNT_ID"]) == ["A1", "A2"]

    def test_noop_when_no_resolutions(self):
        frames = _case_frames()
        result = resolve_entity_keys(frames, {})
        assert result["case_history"].columns.tolist() == ["CASE_ID", "STATUS"]

    def test_fail_fast_missing_bridge_dataset(self):
        frames = {"case_history": pd.DataFrame({"CASE_ID": [1]})}
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="nonexistent",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_entity_keys(frames, resolutions)

    def test_fail_fast_missing_source_key_column(self):
        frames = {
            "case_history": pd.DataFrame({"STATUS": ["Open"]}),
            "case": pd.DataFrame({"CASE_ID": [1], "ACCOUNT_ID": ["A1"]}),
        }
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="case",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        with pytest.raises(KeyError, match="CASE_ID"):
            resolve_entity_keys(frames, resolutions)

    def test_fail_fast_missing_bridge_key_column(self):
        frames = {
            "case_history": pd.DataFrame({"CASE_ID": [1]}),
            "case": pd.DataFrame({"ID": [1], "ACCOUNT_ID": ["A1"]}),
        }
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="case",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        with pytest.raises(KeyError, match="CASE_ID"):
            resolve_entity_keys(frames, resolutions)

    def test_fail_fast_missing_resolve_column(self):
        frames = {
            "case_history": pd.DataFrame({"CASE_ID": [1]}),
            "case": pd.DataFrame({"CASE_ID": [1], "NAME": ["foo"]}),
        }
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="case",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        with pytest.raises(KeyError, match="ACCOUNT_ID"):
            resolve_entity_keys(frames, resolutions)

    def test_orphan_rows_dropped_by_inner_join(self):
        case_history = pd.DataFrame(
            {"CASE_ID": [1, 2, 99], "STATUS": ["Open", "Closed", "Open"]}
        )
        case = pd.DataFrame(
            {"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]}
        )
        frames = {"case_history": case_history, "case": case}
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="case",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        result = resolve_entity_keys(frames, resolutions)
        assert len(result["case_history"]) == 2
        assert list(result["case_history"]["ACCOUNT_ID"]) == ["A1", "A2"]

    def test_duplicate_bridge_keys(self):
        case_history = pd.DataFrame({"CASE_ID": [1, 2]})
        case = pd.DataFrame(
            {"CASE_ID": [1, 1, 2], "ACCOUNT_ID": ["A1", "A1", "A2"]}
        )
        frames = {"case_history": case_history, "case": case}
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="case",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        result = resolve_entity_keys(frames, resolutions)
        assert "ACCOUNT_ID" in result["case_history"].columns

    def test_preserves_original_columns(self):
        frames = _case_frames()
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="case",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        result = resolve_entity_keys(frames, resolutions)
        assert "CASE_ID" in result["case_history"].columns
        assert "STATUS" in result["case_history"].columns
        assert "ACCOUNT_ID" in result["case_history"].columns

    def test_all_rows_orphaned_fails_fast(self):
        case_history = pd.DataFrame({"CASE_ID": [99, 100]})
        case = pd.DataFrame({"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]})
        frames = {"case_history": case_history, "case": case}
        resolutions = {
            "case_history": [
                KeyResolutionStep(
                    bridge_dataset="case",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        with pytest.raises(ValueError, match="empty"):
            resolve_entity_keys(frames, resolutions)


class TestSuggestKeyResolutions:
    def test_detects_single_hop_bridge(self):
        case_history = pd.DataFrame(
            {"CASE_ID": [1, 2, 3], "STATUS": ["Open", "Closed", "Open"]}
        )
        case = pd.DataFrame(
            {"CASE_ID": [1, 2, 3], "ACCOUNT_ID": ["A1", "A2", "A3"]}
        )
        frames = {"case_history": case_history, "case": case}
        suggestions = suggest_key_resolutions(frames, "ACCOUNT_ID")
        assert "case_history" in suggestions
        steps = suggestions["case_history"]
        assert len(steps) == 1
        assert steps[0].bridge_dataset == "case"
        assert steps[0].source_key == "CASE_ID"
        assert steps[0].resolve_column == "ACCOUNT_ID"

    def test_returns_empty_for_datasets_with_entity_column(self):
        df = pd.DataFrame({"ACCOUNT_ID": ["A1"], "VALUE": [100]})
        frames = {"accounts": df}
        suggestions = suggest_key_resolutions(frames, "ACCOUNT_ID")
        assert suggestions == {}

    def test_returns_empty_when_no_bridge_path(self):
        orphan = pd.DataFrame({"X": [1], "Y": [2]})
        frames = {"orphan": orphan}
        suggestions = suggest_key_resolutions(frames, "ACCOUNT_ID")
        assert suggestions == {}

    def test_scores_by_coverage(self):
        source = pd.DataFrame({"CASE_ID": [1, 2, 3, 4, 5]})
        bridge_good = pd.DataFrame(
            {"CASE_ID": [1, 2, 3, 4, 5], "ACCOUNT_ID": ["A1", "A2", "A3", "A4", "A5"]}
        )
        bridge_bad = pd.DataFrame(
            {"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]}
        )
        frames = {"source": source, "bridge_good": bridge_good, "bridge_bad": bridge_bad}
        suggestions = suggest_key_resolutions(frames, "ACCOUNT_ID")
        assert "source" in suggestions
        assert suggestions["source"][0].bridge_dataset == "bridge_good"

    def test_id_like_column_detection(self):
        source = pd.DataFrame({"ORDER_KEY": ["K1", "K2"], "VALUE": [10, 20]})
        bridge = pd.DataFrame(
            {"ORDER_KEY": ["K1", "K2"], "ACCOUNT_ID": ["A1", "A2"]}
        )
        frames = {"source": source, "bridge": bridge}
        suggestions = suggest_key_resolutions(frames, "ACCOUNT_ID")
        assert "source" in suggestions
        assert suggestions["source"][0].source_key == "ORDER_KEY"

    def test_multiple_datasets_resolved(self):
        case_history = pd.DataFrame({"CASE_ID": [1, 2]})
        opp_product = pd.DataFrame({"OPP_ID": [10, 20]})
        case = pd.DataFrame(
            {"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]}
        )
        opportunity = pd.DataFrame(
            {"OPP_ID": [10, 20], "ACCOUNT_ID": ["A1", "A2"]}
        )
        frames = {
            "case_history": case_history,
            "opp_product": opp_product,
            "case": case,
            "opportunity": opportunity,
        }
        suggestions = suggest_key_resolutions(frames, "ACCOUNT_ID")
        assert "case_history" in suggestions
        assert "opp_product" in suggestions

    def test_does_not_suggest_for_bridge_datasets(self):
        case = pd.DataFrame(
            {"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]}
        )
        frames = {"case": case}
        suggestions = suggest_key_resolutions(frames, "ACCOUNT_ID")
        assert "case" not in suggestions


class TestResolveSampleIdsViaBridge:
    @pytest.fixture()
    def namespace(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="bridge-test")
        ns.setup()
        return ns

    def test_single_hop(self, namespace):
        bridge_df = pd.DataFrame(
            {"CASE_ID": [1, 2, 3], "ACCOUNT_ID": ["A1", "A2", "A3"]}
        )
        save_active_dataset(namespace, "case", bridge_df)
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        local_key, local_ids = resolve_sample_ids_via_bridge(
            namespace, steps, ["A1", "A3"],
        )
        assert local_key == "CASE_ID"
        assert local_ids == {1, 3}

    def test_multi_hop(self, namespace):
        item_df = pd.DataFrame(
            {"ITEM_ID": [10, 20, 30], "ORDER_ID": [100, 200, 300]}
        )
        order_df = pd.DataFrame(
            {"ORDER_ID": [100, 200, 300], "ACCOUNT_ID": ["A1", "A2", "A3"]}
        )
        save_active_dataset(namespace, "item", item_df)
        save_active_dataset(namespace, "order", order_df)
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
        local_key, local_ids = resolve_sample_ids_via_bridge(
            namespace, steps, ["A1", "A3"],
        )
        assert local_key == "ITEM_ID"
        assert local_ids == {10, 30}

    def test_empty_sample_ids(self, namespace):
        bridge_df = pd.DataFrame(
            {"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]}
        )
        save_active_dataset(namespace, "case", bridge_df)
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        local_key, local_ids = resolve_sample_ids_via_bridge(
            namespace, steps, [],
        )
        assert local_key == "CASE_ID"
        assert local_ids == set()

    def test_no_matching_ids(self, namespace):
        bridge_df = pd.DataFrame(
            {"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]}
        )
        save_active_dataset(namespace, "case", bridge_df)
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        local_key, local_ids = resolve_sample_ids_via_bridge(
            namespace, steps, ["Z99"],
        )
        assert local_key == "CASE_ID"
        assert local_ids == set()

    def test_bridge_not_saved_raises(self, namespace):
        steps = [
            KeyResolutionStep(
                bridge_dataset="missing_bridge",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        with pytest.raises(FileNotFoundError):
            resolve_sample_ids_via_bridge(namespace, steps, ["A1"])

    def test_many_to_one(self, namespace):
        bridge_df = pd.DataFrame(
            {"CASE_ID": [1, 2, 3, 4], "ACCOUNT_ID": ["A1", "A1", "A2", "A2"]}
        )
        save_active_dataset(namespace, "case", bridge_df)
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        local_key, local_ids = resolve_sample_ids_via_bridge(
            namespace, steps, ["A1"],
        )
        assert local_key == "CASE_ID"
        assert local_ids == {1, 2}

    def test_bridge_key_differs_from_source_key(self, namespace):
        bridge_df = pd.DataFrame(
            {"TICKET_ID": [1, 2, 3], "ACCOUNT_ID": ["A1", "A2", "A3"]}
        )
        save_active_dataset(namespace, "ticket", bridge_df)
        steps = [
            KeyResolutionStep(
                bridge_dataset="ticket",
                source_key="CASE_ID",
                bridge_key="TICKET_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        local_key, local_ids = resolve_sample_ids_via_bridge(
            namespace, steps, ["A1", "A2"],
        )
        assert local_key == "CASE_ID"
        assert local_ids == {1, 2}

    def test_case_insensitive_column_match(self, namespace):
        bridge_df = pd.DataFrame(
            {"case_id": [1, 2, 3], "account_id": ["A1", "A2", "A3"]}
        )
        save_active_dataset(namespace, "case", bridge_df)
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        local_key, local_ids = resolve_sample_ids_via_bridge(
            namespace, steps, ["A1", "A3"],
        )
        assert local_key == "CASE_ID"
        assert local_ids == {1, 3}

    def test_missing_column_raises(self, namespace):
        bridge_df = pd.DataFrame(
            {"UNRELATED_KEY": [1, 2], "UNRELATED_COL": ["X", "Y"]}
        )
        save_active_dataset(namespace, "case", bridge_df)
        steps = [
            KeyResolutionStep(
                bridge_dataset="case",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        with pytest.raises(KeyError, match="ACCOUNT_ID"):
            resolve_sample_ids_via_bridge(namespace, steps, ["A1"])


class TestStaleStepHandling:
    def test_stale_step_error_is_key_error(self):
        assert issubclass(_StaleStepError, KeyError)

    def test_column_exists_true(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert _column_exists(df.columns, "A") is True

    def test_column_exists_false(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert _column_exists(df.columns, "Z") is False

    def test_column_exists_case_insensitive(self):
        df = pd.DataFrame({"account_id": [1]})
        assert _column_exists(df.columns, "ACCOUNT_ID") is True

    def test_is_empty_true(self):
        df = pd.DataFrame({"A": pd.Series([], dtype=int)})
        assert _is_empty(df) is True

    def test_is_empty_false(self):
        df = pd.DataFrame({"A": [1]})
        assert _is_empty(df) is False

    def test_resolve_entity_keys_raises_key_error_for_stale_step(self):
        frames = {
            "source": pd.DataFrame({"X": [1]}),
            "bridge": pd.DataFrame({"Y": [1], "Z": ["A"]}),
        }
        resolutions = {
            "source": [
                KeyResolutionStep(
                    bridge_dataset="bridge",
                    source_key="MISSING_COL",
                    bridge_key="Y",
                    resolve_column="Z",
                ),
            ],
        }
        with pytest.raises(KeyError):
            resolve_entity_keys(frames, resolutions)

    def test_skip_when_resolve_column_already_present(self):
        frames = {
            "source": pd.DataFrame({"CASE_ID": [1], "ACCOUNT_ID": ["A1"]}),
            "bridge": pd.DataFrame({"CASE_ID": [1], "ACCOUNT_ID": ["A1"]}),
        }
        resolutions = {
            "source": [
                KeyResolutionStep(
                    bridge_dataset="bridge",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        }
        result = resolve_entity_keys(frames, resolutions)
        assert list(result["source"]["ACCOUNT_ID"]) == ["A1"]


class TestResolveSingleDatasetKeysStaleStep:
    @pytest.fixture()
    def namespace(self, tmp_path):
        ns = RunNamespace(root=tmp_path, run_id="stale-test")
        ns.setup()
        return ns

    def test_skips_stale_step_with_warning(self, namespace, caplog):
        bridge_df = pd.DataFrame({"Y": [1], "Z": ["A"]})
        save_active_dataset(namespace, "bridge", bridge_df)
        source_df = pd.DataFrame({"X": [1, 2]})
        steps = [
            KeyResolutionStep(
                bridge_dataset="bridge",
                source_key="MISSING_COL",
                bridge_key="Y",
                resolve_column="Z",
            ),
        ]
        with caplog.at_level(logging.WARNING):
            result = resolve_single_dataset_keys(source_df, steps, namespace)
        assert len(result) == 2
        assert "Skipping stale key resolution step" in caplog.text

    def test_applies_valid_step(self, namespace):
        bridge_df = pd.DataFrame({"CASE_ID": [1, 2], "ACCOUNT_ID": ["A1", "A2"]})
        save_active_dataset(namespace, "bridge", bridge_df)
        source_df = pd.DataFrame({"CASE_ID": [1, 2], "VALUE": [10, 20]})
        steps = [
            KeyResolutionStep(
                bridge_dataset="bridge",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        result = resolve_single_dataset_keys(source_df, steps, namespace)
        assert "ACCOUNT_ID" in result.columns
        assert list(result["ACCOUNT_ID"]) == ["A1", "A2"]

    def test_skips_when_entity_column_already_present(self, namespace):
        bridge_df = pd.DataFrame({"CASE_ID": [1], "ACCOUNT_ID": ["A1"]})
        save_active_dataset(namespace, "bridge", bridge_df)
        source_df = pd.DataFrame(
            {"CASE_ID": [1, 2], "VALUE": [10, 20], "ACCOUNT_ID": ["A1", "A2"]}
        )
        steps = [
            KeyResolutionStep(
                bridge_dataset="bridge",
                source_key="CASE_ID",
                bridge_key="CASE_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        result = resolve_single_dataset_keys(source_df, steps, namespace)
        assert len(result) == 2
        assert list(result["ACCOUNT_ID"]) == ["A1", "A2"]
