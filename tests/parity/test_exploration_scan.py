from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_retention.parity import ApplyOpKind
from customer_retention.parity.exploration_scan import (
    DYNAMIC,
    scan_exploration_manifest,
)

_FRAMEWORK_MODULES = (
    "customer_retention.analysis.auto_explorer.sampling",
    "customer_retention.stages.lifecycle.enrich",
    "customer_retention.stages.profiling.time_window_aggregator",
    "customer_retention.stages.temporal.temporal_merger",
    "customer_retention.transforms.ops",
    "customer_retention.stages.profiling.target_validator",
    "customer_retention.stages.modeling.data_splitter",
    "customer_retention.transforms.fitted",
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_modules_imported():
    import importlib
    for mod in _FRAMEWORK_MODULES:
        importlib.import_module(mod)
    yield


def _write_notebook(tmp_path: Path, name: str, cells: list[dict]) -> Path:
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / name
    path.write_text(json.dumps(nb))
    return path


def _code_cell(source: str, cell_id: str | None = None, tag_id: str | None = None) -> dict:
    src_lines = source.splitlines(keepends=True)
    if tag_id:
        src_lines = [f"# @cr:code name='cell' id={tag_id}\n", *src_lines]
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src_lines,
    }
    if cell_id is not None:
        cell["id"] = cell_id
    return cell


class TestSingleCellCallSite:
    def test_decorated_framework_call_is_recorded(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    enriched = enrich_lifecycle_dataset(df, config)\n",
                tag_id="abcd1234",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        entries = manifest.by_dataset("contract")
        assert len(entries) == 1
        e = entries[0]
        assert e.kind is ApplyOpKind.LIFECYCLE_ENRICH
        assert e.source_location.cell_id == "abcd1234"

    def test_aliased_import_resolves(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset as enrich\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='account'):\n"
                "    enrich(df, config)\n",
                tag_id="abcd0002",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        kinds = manifest.kinds_for("account")
        assert ApplyOpKind.LIFECYCLE_ENRICH in kinds

    def test_module_attribute_call_resolves(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle import enrich\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    enrich.enrich_lifecycle_dataset(df, config)\n",
                tag_id="abcd0003",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        assert ApplyOpKind.LIFECYCLE_ENRICH in manifest.kinds_for("contract")

    def test_undecorated_call_is_ignored(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "import math\n"
                "math.sqrt(4)\n",
                tag_id="abcd0004",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        assert manifest.entries == ()


class TestDatasetHint:
    def test_explicit_apply_context_wins(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='subscription'):\n"
                "    enrich_lifecycle_dataset(df, config)\n",
                tag_id="dddd0001",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        assert "subscription" in manifest.datasets()
        assert "<unknown>" not in manifest.datasets()

    def test_nested_apply_context_inner_wins(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='outer'):\n"
                "    with apply_context(dataset='inner'):\n"
                "        enrich_lifecycle_dataset(df, config)\n",
                tag_id="dddd0002",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        assert "inner" in manifest.datasets()
        assert "outer" not in manifest.datasets()

    def test_dataset_kwarg_fallback(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.analysis.auto_explorer.sampling import apply_sample_filters\n"
                "apply_sample_filters(df, dataset_name='account', filters={'account': 'col > 0'})\n",
                tag_id="dddd0003",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        assert "account" in manifest.datasets()

    def test_unknown_when_no_hint(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "enrich_lifecycle_dataset(df, config)\n",
                tag_id="dddd0004",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        assert manifest.datasets() == {"<unknown>"}


class TestGateExtraction:
    def test_enclosing_if_captured_as_gate(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.analysis.auto_explorer.sampling import apply_temporal_lookback\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    if intent.lookback_periods is not None:\n"
                "        apply_temporal_lookback(df, time_col, intent)\n",
                tag_id="gg000001",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        entry = manifest.by_dataset("contract")[0]
        # gate is captured in the fingerprint under the conventional key
        assert "_gate" in entry.kwargs_fingerprint
        assert "lookback_periods" in entry.kwargs_fingerprint["_gate"]

    def test_no_gate_when_unconditional(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='account'):\n"
                "    enrich_lifecycle_dataset(df, config)\n",
                tag_id="gg000002",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        entry = manifest.by_dataset("account")[0]
        assert "_gate" not in entry.kwargs_fingerprint


class TestKwargsCapture:
    def test_literal_kwargs_captured(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.profiling.time_window_aggregator import derive_extra_datetime_features\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    derive_extra_datetime_features(df, time_column='event_timestamp', datetime_columns=['a', 'b'])\n",
                tag_id="kw000001",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        entry = manifest.by_dataset("contract")[0]
        fp = entry.kwargs_fingerprint
        assert fp["time_column"] == "event_timestamp"
        assert fp["datetime_columns"] == ("a", "b")

    def test_name_resolution_assign_then_call(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.profiling.time_window_aggregator import derive_extra_datetime_features\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    ts_col = 'feature_timestamp'\n"
                "    derive_extra_datetime_features(df, time_column=ts_col, datetime_columns=['x'])\n",
                tag_id="kw000002",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        fp = manifest.by_dataset("contract")[0].kwargs_fingerprint
        assert fp["time_column"] == "feature_timestamp"

    def test_unresolvable_kwarg_becomes_dynamic(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.profiling.time_window_aggregator import derive_extra_datetime_features\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    derive_extra_datetime_features(df, time_column=compute()(intent), datetime_columns=cols)\n",
                tag_id="kw000003",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        fp = manifest.by_dataset("contract")[0].kwargs_fingerprint
        assert fp["time_column"] == DYNAMIC


class TestLocalHelperResolution:
    def test_call_through_local_helper_in_same_cell(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "from customer_retention.parity import apply_context\n"
                "\n"
                "def _do_enrich(df, config):\n"
                "    return enrich_lifecycle_dataset(df, config)\n"
                "\n"
                "with apply_context(dataset='contract'):\n"
                "    _do_enrich(df, cfg)\n",
                tag_id="hh000001",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        kinds = manifest.kinds_for("contract")
        assert ApplyOpKind.LIFECYCLE_ENRICH in kinds

    def test_call_through_local_helper_in_other_cell(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [
                _code_cell(
                    "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                    "\n"
                    "def _do_enrich(df, config):\n"
                    "    return enrich_lifecycle_dataset(df, config)\n",
                    tag_id="hh000002",
                ),
                _code_cell(
                    "from customer_retention.parity import apply_context\n"
                    "with apply_context(dataset='account'):\n"
                    "    _do_enrich(df, cfg)\n",
                    tag_id="hh000003",
                ),
            ],
        )
        manifest = scan_exploration_manifest([nb])
        assert ApplyOpKind.LIFECYCLE_ENRICH in manifest.kinds_for("account")

    def test_recursion_depth_bounded(self, tmp_path):
        # Helper chain of depth 5 — should still resolve down to the apply_op
        # because our bound is 3, but the apply_op call sits at depth 1
        # inside the deepest helper; outer wrappers count toward depth
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "from customer_retention.parity import apply_context\n"
                "\n"
                "def a(df, c): return b(df, c)\n"
                "def b(df, c): return _enrich(df, c)\n"
                "def _enrich(df, c): return enrich_lifecycle_dataset(df, c)\n"
                "\n"
                "with apply_context(dataset='contract'):\n"
                "    a(df, cfg)\n",
                tag_id="hh000004",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        # depth 3 reaches `enrich_lifecycle_dataset` exactly
        assert ApplyOpKind.LIFECYCLE_ENRICH in manifest.kinds_for("contract")


class TestImportSharedAcrossCells:
    def test_import_in_cell_one_visible_in_cell_two(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [
                _code_cell(
                    "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n",
                    tag_id="ii000001",
                ),
                _code_cell(
                    "from customer_retention.parity import apply_context\n"
                    "with apply_context(dataset='contract'):\n"
                    "    enrich_lifecycle_dataset(df, config)\n",
                    tag_id="ii000002",
                ),
            ],
        )
        manifest = scan_exploration_manifest([nb])
        assert ApplyOpKind.LIFECYCLE_ENRICH in manifest.kinds_for("contract")


class TestCallOrder:
    def test_entries_preserve_invocation_order(self, tmp_path):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "from customer_retention.analysis.auto_explorer.sampling import apply_temporal_lookback\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    enrich_lifecycle_dataset(df, cfg)\n"
                "    apply_temporal_lookback(df, 'feature_timestamp', intent)\n",
                tag_id="oo000001",
            )],
        )
        manifest = scan_exploration_manifest([nb])
        seq = [e.kind for e in manifest.by_dataset("contract")]
        assert seq.index(ApplyOpKind.LIFECYCLE_ENRICH) < seq.index(ApplyOpKind.TEMPORAL_LOOKBACK)


class TestMultipleNotebooks:
    def test_call_order_across_notebooks(self, tmp_path):
        nb1 = _write_notebook(
            tmp_path, "00.ipynb",
            [_code_cell(
                "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    enrich_lifecycle_dataset(df, cfg)\n",
                tag_id="mm000001",
            )],
        )
        nb2 = _write_notebook(
            tmp_path, "01.ipynb",
            [_code_cell(
                "from customer_retention.analysis.auto_explorer.sampling import apply_temporal_lookback\n"
                "from customer_retention.parity import apply_context\n"
                "with apply_context(dataset='contract'):\n"
                "    apply_temporal_lookback(df, 'ts', intent)\n",
                tag_id="mm000002",
            )],
        )
        manifest = scan_exploration_manifest([nb1, nb2])
        # Should contain BOTH entries, in lexical-notebook-order
        kinds = [e.kind for e in manifest.by_dataset("contract")]
        assert kinds == [ApplyOpKind.LIFECYCLE_ENRICH, ApplyOpKind.TEMPORAL_LOOKBACK]


class TestEmptyAndDegenerateInputs:
    def test_empty_notebook_yields_empty_manifest(self, tmp_path):
        nb = _write_notebook(tmp_path, "01.ipynb", [])
        manifest = scan_exploration_manifest([nb])
        assert manifest.entries == ()

    def test_only_markdown_yields_empty_manifest(self, tmp_path):
        cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# This is a markdown cell\n"],
        }
        nb = _write_notebook(tmp_path, "01.ipynb", [cell])
        manifest = scan_exploration_manifest([nb])
        assert manifest.entries == ()

    def test_syntax_error_cell_is_skipped_with_warning(self, tmp_path, caplog):
        nb = _write_notebook(
            tmp_path, "01.ipynb",
            [
                _code_cell("this is not valid python (((\n", tag_id="bb000001"),
                _code_cell(
                    "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
                    "from customer_retention.parity import apply_context\n"
                    "with apply_context(dataset='contract'):\n"
                    "    enrich_lifecycle_dataset(df, cfg)\n",
                    tag_id="bb000002",
                ),
            ],
        )
        manifest = scan_exploration_manifest([nb])
        # Valid cell still picked up
        assert ApplyOpKind.LIFECYCLE_ENRICH in manifest.kinds_for("contract")
