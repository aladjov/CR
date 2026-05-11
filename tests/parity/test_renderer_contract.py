"""Renderer-contract regression: run the live PipelineGenerator end-to-end
and lock in the invariants that a clean parity audit checks.

The fixture-driven approach lets us add new cartesian variations (granularity
× role × intent setting) without changing the test code — each fixture is a
small `findings_dir` whose expected audit outcome is asserted here. The
existing `sps_mini` fixture covers the post-fix engagement state; new
fixtures slot in next to it.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from customer_retention.generators.pipeline_generator import PipelineGenerator
from customer_retention.parity import (
    ApplyOpKind,
    AuditScope,
    audit_pipeline,
    scan_generated_pipeline,
)

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
def _ensure_decoration():
    for mod in _FRAMEWORK_MODULES:
        importlib.import_module(mod)
    yield


@pytest.fixture
def generated_sps_mini(tmp_path):
    findings_dir = Path("tests/fixtures/user_extensions/sps_mini")
    out = tmp_path / "generated"
    PipelineGenerator(
        findings_dir=str(findings_dir),
        output_dir=str(out),
        pipeline_name="parity_contract",
    ).generate()
    return out


class TestRendererContractInvariants:
    def test_history_window_invariant_no_lookback_emitted(self, generated_sps_mini):
        """The headline invariant: no `TEMPORAL_LOOKBACK` for any dataset
        when the intent does not configure `lookback_periods` (sps_mini).
        Reverting the `should_apply_lookback` gate would fail this test."""
        manifest = scan_generated_pipeline(generated_sps_mini, scope=AuditScope.ALL)
        kinds = {e.kind for e in manifest.entries}
        assert ApplyOpKind.TEMPORAL_LOOKBACK not in kinds

    def test_landing_emits_three_derive_ops_per_event_source(self, generated_sps_mini):
        """Every event-level landing script must emit
        `derive_feature_timestamp` + `derive_label_timestamp` +
        `derive_label_available_flag`. Their absence indicates a template
        regression (e.g. dropped a render block by accident)."""
        manifest = scan_generated_pipeline(generated_sps_mini, scope=AuditScope.LANDING)
        request_kinds = manifest.kinds_for("request")
        assert {
            ApplyOpKind.FEATURE_TIMESTAMP_DERIVE,
            ApplyOpKind.LABEL_TIMESTAMP_DERIVE,
            ApplyOpKind.LABEL_AVAILABLE_FLAG,
        }.issubset(request_kinds)

    def test_silver_emits_holdout_mask(self, generated_sps_mini):
        manifest = scan_generated_pipeline(generated_sps_mini, scope=AuditScope.SILVER)
        kinds = {e.kind for e in manifest.entries}
        assert ApplyOpKind.SILVER_HOLDOUT_MASK in kinds

    def test_gold_emits_transformation_and_encoding(self, generated_sps_mini):
        manifest = scan_generated_pipeline(generated_sps_mini, scope=AuditScope.GOLD)
        kinds = {e.kind for e in manifest.entries}
        assert ApplyOpKind.GOLD_TRANSFORMATION in kinds
        assert ApplyOpKind.GOLD_ENCODING in kinds


class TestAuditEndToEndOnLiveRenderer:
    def test_audit_pipeline_runs_to_completion(self, tmp_path, generated_sps_mini):
        """Smoke test: audit_pipeline against the live-rendered output
        completes without exceptions and returns a structured outcome."""
        engagement = tmp_path / "engagement"
        engagement.mkdir()
        outcome = audit_pipeline(
            engagement_dir=engagement,
            pipeline_dir=generated_sps_mini,
            scope=AuditScope.ALL,
        )
        # Empty engagement → every production emit is a gap, which is
        # expected. We just verify the audit machinery doesn't crash and
        # returns a stable shape.
        assert outcome.scope is AuditScope.ALL
        assert outcome.production_manifest is not None
        assert len(outcome.production_manifest.entries) > 0
