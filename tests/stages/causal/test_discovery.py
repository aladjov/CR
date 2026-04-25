"""Tests for ``stages/causal/interpretation/discovery.py``.

Covers the failure modes Cycle 013 surfaced:
  - namespace discovery returns None → bundle is empty + warning emitted
  - namespace resolved but sidecars are empty → per-sidecar warnings
  - all sidecars populated → ``fully_populated`` is True, no warnings
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

from customer_retention.stages.causal.interpretation.discovery import (
    InterpretationSidecars,
    discover_interpretation_sidecars,
)


def _ns_stub() -> SimpleNamespace:
    """Stand-in RunNamespace exposing only the dir attributes the warnings reference."""
    return SimpleNamespace(
        feature_meta_dir="/x/feature_meta",
        feature_population_stats_dir="/x/feature_population_stats",
        column_descriptions_dir="/x/column_descriptions",
        run_id="run-test",
    )


class TestDiscoverWithNamespace:
    def test_full_bundle_warns_nothing(self):
        ns = _ns_stub()
        with patch(
            "customer_retention.stages.causal.interpretation.sidecars.load_feature_meta_sidecar",
            return_value={"f": object()},
        ), patch(
            "customer_retention.stages.causal.interpretation.sidecars.load_population_stats_sidecar",
            return_value={"f": object()},
        ), patch(
            "customer_retention.stages.causal.interpretation.sidecars.load_column_descriptions_sidecar",
            return_value={"c": object()},
        ):
            bundle = discover_interpretation_sidecars(namespace=ns)
        assert bundle.namespace is ns
        assert bundle.feature_meta and bundle.population_stats and bundle.column_descriptions
        assert bundle.warnings == []
        assert bundle.fully_populated is True

    def test_empty_sidecars_emit_per_sidecar_warnings(self):
        ns = _ns_stub()
        with patch(
            "customer_retention.stages.causal.interpretation.sidecars.load_feature_meta_sidecar",
            return_value={},
        ), patch(
            "customer_retention.stages.causal.interpretation.sidecars.load_population_stats_sidecar",
            return_value={},
        ), patch(
            "customer_retention.stages.causal.interpretation.sidecars.load_column_descriptions_sidecar",
            return_value={},
        ):
            bundle = discover_interpretation_sidecars(namespace=ns)
        # Three sidecars empty → three warnings, one per sidecar
        assert bundle.namespace is ns
        assert len(bundle.warnings) == 3
        joined = "\n".join(bundle.warnings)
        assert "feature_meta sidecar empty" in joined
        assert "feature_population_stats sidecar empty" in joined
        assert "column_descriptions sidecar empty" in joined
        assert bundle.fully_populated is False


class TestDiscoverWithoutNamespace:
    def test_namespace_discovery_failure_attaches_warning(self):
        with patch(
            "customer_retention.analysis.auto_explorer.run_namespace.RunNamespace.from_env_or_latest",
            return_value=None,
        ):
            bundle = discover_interpretation_sidecars(namespace=None)
        assert bundle.namespace is None
        assert bundle.fully_populated is False
        assert any("RunNamespace discovery failed" in w for w in bundle.warnings)

    def test_namespace_discovery_exception_caught_and_warned(self):
        with patch(
            "customer_retention.analysis.auto_explorer.run_namespace.RunNamespace.from_env_or_latest",
            side_effect=RuntimeError("boom"),
        ):
            bundle = discover_interpretation_sidecars(namespace=None)
        assert bundle.namespace is None
        # First entry: from_env_or_latest raised → exception text captured
        assert any("RuntimeError: boom" in w for w in bundle.warnings)


class TestEmitWarnings:
    def test_emits_via_logger(self, caplog):
        bundle = InterpretationSidecars(
            namespace=None,
            warnings=["sidecar empty: foo", "namespace not found"],
        )
        with caplog.at_level(logging.WARNING):
            bundle.emit_warnings()
        messages = [r.getMessage() for r in caplog.records]
        assert any("sidecar empty: foo" in m for m in messages)
        assert any("namespace not found" in m for m in messages)

    def test_no_warnings_no_log(self, caplog):
        bundle = InterpretationSidecars(namespace=_ns_stub())
        with caplog.at_level(logging.WARNING):
            bundle.emit_warnings()
        assert not [r for r in caplog.records if "[interpretation]" in r.getMessage()]


class TestFullyPopulated:
    def test_requires_namespace_and_all_three_sidecars(self):
        ns = _ns_stub()
        empty = InterpretationSidecars(namespace=ns)
        assert empty.fully_populated is False

        partial = InterpretationSidecars(
            namespace=ns,
            feature_meta={"a": 1},
            population_stats={"a": 1},
            column_descriptions={},
        )
        assert partial.fully_populated is False

        full = InterpretationSidecars(
            namespace=ns,
            feature_meta={"a": 1},
            population_stats={"a": 1},
            column_descriptions={"a": 1},
        )
        assert full.fully_populated is True
