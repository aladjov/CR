from __future__ import annotations

import logging

import pytest

from customer_retention.analysis.auto_explorer.key_resolver import (
    resolve_entity_keys,
    suggest_key_resolutions,
)
from customer_retention.analysis.auto_explorer.project_context import KeyResolutionStep
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

    def test_orphan_rows_warns_and_proceeds(self, caplog):
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
        with caplog.at_level(logging.WARNING):
            result = resolve_entity_keys(frames, resolutions)
        assert len(result["case_history"]) == 2
        assert "1" in caplog.text

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
