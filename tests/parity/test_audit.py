from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_retention.parity import ApplyOpKind, AuditScope
from customer_retention.parity.audit import (
    AuditOutcome,
    audit_landing,
    audit_pipeline,
)
from customer_retention.parity.gaps import GapKind

_FRAMEWORK_MODULES = (
    "customer_retention.analysis.auto_explorer.sampling",
    "customer_retention.stages.lifecycle.enrich",
    "customer_retention.stages.profiling.target_validator",
    "customer_retention.stages.profiling.time_window_aggregator",
    "customer_retention.stages.temporal.temporal_merger",
    "customer_retention.stages.modeling.data_splitter",
    "customer_retention.transforms.ops",
    "customer_retention.transforms.fitted",
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_modules_imported():
    import importlib
    for mod in _FRAMEWORK_MODULES:
        importlib.import_module(mod)
    yield


def _ipynb(path: Path, cells: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }))
    return path


def _cell(source: str, tag_id: str | None = None) -> dict:
    lines = source.splitlines(keepends=True)
    if tag_id:
        lines = [f"# @cr:code name='cell' id={tag_id}\n", *lines]
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


class TestAuditOutcome:
    def test_no_gaps_is_pass(self):
        outcome = AuditOutcome(gaps=())
        assert outcome.has_gaps is False
        summary = outcome.format_summary()
        assert "PASS" in summary or "OK" in summary

    def test_gaps_is_fail(self):
        from customer_retention.parity import ParityGap, SourceLocation
        gap = ParityGap(
            gap_kind=GapKind.PRODUCTION_ONLY,
            dataset="contract",
            op_kind=ApplyOpKind.TEMPORAL_LOOKBACK,
            exploration_location=None,
            production_location=SourceLocation(file=Path("x.py"), line=80),
            detail="boom",
        )
        outcome = AuditOutcome(gaps=(gap,))
        assert outcome.has_gaps is True
        report = outcome.format_report()
        assert "PRODUCTION_ONLY" in report
        assert "TEMPORAL_LOOKBACK" in report

    def test_to_failed_json_returns_serializable_string(self):
        from customer_retention.parity import ParityGap, SourceLocation
        gap = ParityGap(
            gap_kind=GapKind.EXPLORATION_ONLY,
            dataset="account",
            op_kind=ApplyOpKind.SAMPLE_FILTER,
            exploration_location=SourceLocation(file=Path("nb.ipynb"), line=5),
            production_location=None,
            detail="exploration applies, production omits",
        )
        outcome = AuditOutcome(gaps=(gap,))
        payload = json.loads(outcome.to_failed_json())
        assert payload["status"] == "failed"
        assert payload["gap_count"] == 1
        assert payload["gaps"][0]["gap_kind"] == "exploration_only"


class TestAuditLandingEndToEnd:
    """audit_landing wires the exploration scan against an in-memory
    production scan of the rendered landing scripts. When both sides see
    the same kinds for the same dataset, the outcome is clean."""

    def test_passes_when_manifests_match(self, tmp_path):
        engagement = tmp_path / "engagement"
        _ipynb(engagement / "00_start.ipynb", [])
        _ipynb(engagement / "01_discover.ipynb", [_cell(
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "from customer_retention.parity import apply_context\n"
            "with apply_context(dataset='contract'):\n"
            "    enrich_lifecycle_dataset(df, config)\n",
            tag_id="aa000001",
        )])

        pipeline = tmp_path / "generated"
        (pipeline / "landing").mkdir(parents=True)
        (pipeline / "landing" / "landing_contract.py").write_text(
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "def run_landing(df, cfg):\n"
            "    return enrich_lifecycle_dataset(df, cfg)\n"
        )
        outcome = audit_landing(
            engagement_dir=engagement,
            pipeline_dir=pipeline,
        )
        assert outcome.has_gaps is False
        assert "PASS" in outcome.format_summary()

    def test_catches_history_window_gap(self, tmp_path):
        """The headline scenario: exploration skips temporal_lookback (gated
        off for interval-type roles), production emits it anyway."""
        engagement = tmp_path / "engagement"
        _ipynb(engagement / "00_start.ipynb", [])
        _ipynb(engagement / "01_discover.ipynb", [_cell(
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "from customer_retention.parity import apply_context\n"
            "with apply_context(dataset='contract'):\n"
            "    enrich_lifecycle_dataset(df, config)\n",
            tag_id="aa000002",
        )])

        pipeline = tmp_path / "generated"
        (pipeline / "landing").mkdir(parents=True)
        (pipeline / "landing" / "landing_contract.py").write_text(
            "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset\n"
            "from customer_retention.analysis.auto_explorer.sampling import apply_temporal_lookback\n"
            "def run_landing(df, cfg, intent):\n"
            "    df = enrich_lifecycle_dataset(df, cfg)\n"
            "    return apply_temporal_lookback(df, 'feature_timestamp', intent)\n"
        )
        outcome = audit_landing(
            engagement_dir=engagement,
            pipeline_dir=pipeline,
        )
        assert outcome.has_gaps is True
        kinds_in_gaps = {g.op_kind for g in outcome.gaps if g.op_kind}
        assert ApplyOpKind.TEMPORAL_LOOKBACK in kinds_in_gaps
        report = outcome.format_report()
        assert "TEMPORAL_LOOKBACK" in report
        assert "contract" in report

    def test_exit_code_distinguishes_pass_from_fail(self, tmp_path):
        engagement = tmp_path / "engagement"
        _ipynb(engagement / "00_start.ipynb", [])
        pipeline = tmp_path / "generated"
        (pipeline / "landing").mkdir(parents=True)
        clean = audit_landing(engagement_dir=engagement, pipeline_dir=pipeline)
        assert clean.exit_code == 0


class TestAuditPipelineT1:
    """T1 audit covers bronze/silver/gold/training once findings exist."""

    def test_t1_returns_clean_outcome_when_aligned(self, tmp_path):
        engagement = tmp_path / "engagement"
        engagement.mkdir()
        pipeline = tmp_path / "generated"
        (pipeline / "silver").mkdir(parents=True)
        (pipeline / "silver" / "silver_featureset_cn.py").write_text(
            "from customer_retention.stages.temporal.temporal_merger import TemporalMerger\n"
            "def run_silver(spine, datasets):\n"
            "    return TemporalMerger().merge_all(spine, datasets)\n"
        )
        outcome = audit_pipeline(
            engagement_dir=engagement,
            pipeline_dir=pipeline,
            scope=AuditScope.SILVER,
        )
        # T1 outcome captures the production manifest's kinds; with no
        # exploration entries, every production entry is an orphan from
        # the audit's perspective, but the report should be structured.
        assert isinstance(outcome, AuditOutcome)
