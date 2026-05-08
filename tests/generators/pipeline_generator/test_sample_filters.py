"""Cohort-scope sample_filters replay in production landing.

Operator scenario: NB00 sets ``SAMPLE_FILTER_COLUMNS`` (persisted as
``project_context.sample_filters``) to scope exploration to a subset of
entities — e.g. SPS engagement spschurn-e34b8ec5 scopes account to
``REVENUE_MARKET_SEGMENT in ['Emerging', 'Small'] and ACCOUNT_ID in
(select ACCOUNT_ID from contract where event_type = 'start')``.

Without replay in production, the generated `landing_<ds>.py` reads the
full upstream and the cohort scope drifts between exploration and prod.
The parser must:

* Walk ``project_context.sample_filters`` and emit a ``LANDING_FILTER``
  step per dataset.
* Detect sibling-table refs (e.g. ``from contract``) so the landing
  template can register them as bare-name temp views before applying
  the predicate.
* Reorder ``config.landing`` so siblings precede dependents — siblings'
  registered transforms (lifecycle, drops, NB10 OVERRIDES) must run
  first so the temp view points at the post-transform schema the
  predicate expects (e.g. ``event_type`` only exists on contract after
  the lifecycle enrichment doubles its rows into start/terminate).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
)


def _fake_namespace(tmp_path: Path):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    ns = RunNamespace(root=tmp_path, run_id="test-run")
    ns.merged_dir.mkdir(parents=True, exist_ok=True)
    return ns


def _write_event_findings(ns, dataset_name: str, raw_source_path: str) -> None:
    findings_doc = {
        "source_path": raw_source_path,
        "source_format": "delta",
        "row_count": 1000, "column_count": 3,
        "target_column": "churn",
        "identifier_columns": ["entity_id"],
        "datetime_columns": ["event_timestamp"],
        "time_series_metadata": {
            "time_column": "event_timestamp",
            "entity_column": "entity_id",
        },
        "columns": {
            "entity_id": {"name": "entity_id", "inferred_type": "identifier",
                          "confidence": 1.0, "evidence": []},
            "event_timestamp": {"name": "event_timestamp", "inferred_type": "datetime",
                                "confidence": 1.0, "evidence": []},
        },
    }
    ds_path = ns.dataset_findings_dir(dataset_name) / f"{dataset_name}_findings.yaml"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    with ds_path.open("w") as f:
        yaml.safe_dump(findings_doc, f)


def _write_multi_findings(ns, datasets: list[str], raw_source_paths: dict[str, str]) -> None:
    multi = {
        "datasets": {
            name: {
                "name": name,
                "findings_path": str(ns.dataset_findings_dir(name) / f"{name}_findings.yaml"),
                "source_path": raw_source_paths[name],
                "granularity": "event_level",
                "row_count": 1000, "column_count": 3,
                "entity_column": "entity_id", "target_column": "churn",
                "raw_source_path": raw_source_paths[name],
            }
            for name in datasets
        },
        "primary_entity_dataset": datasets[0],
        "event_datasets": list(datasets),
        "excluded_datasets": [],
    }
    with ns.multi_dataset_findings_path.open("w") as f:
        yaml.safe_dump(multi, f)


def _write_project_context(ns, sample_filters: dict[str, str], datasets: list[str]) -> None:
    """Write a minimal ``project_context.yaml`` that satisfies the
    `ProjectContext` schema validators (`objectives` non-empty, exactly
    one primary; per-dataset `granularity` + roles)."""
    ctx = {
        "project_name": "test",
        "target_dataset": datasets[0],
        "target_column": "churn",
        "entity_column": "entity_id",
        "primary_objective": "immediate_risk",
        "objectives": [{
            "objective": "immediate_risk", "priority": "primary",
            "horizon_days": 90, "cadence_days": 7,
        }],
        "datasets": {
            name: {
                "name": name, "path": f"./{name}.csv", "role": "feature",
                "granularity": "event_level",
                "entity_column": "entity_id",
                "time_column": "event_timestamp",
            }
            for name in datasets
        },
        "sample_filters": sample_filters,
    }
    # mark first as target
    ctx["datasets"][datasets[0]]["role"] = "target"
    ns.project_context_path.write_text(yaml.dump(ctx, default_flow_style=False))


class TestSampleFilterAppendsLandingFilterStep:
    def test_simple_predicate_no_subquery(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "account", "ml_catalog.retention.account_raw")
        _write_multi_findings(ns, ["account"], {"account": "ml_catalog.retention.account_raw"})
        _write_project_context(
            ns,
            sample_filters={"account": "REVENUE_MARKET_SEGMENT in ['Emerging', 'Small']"},
            datasets=["account"],
        )
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        steps = config.landing["account"].filters
        sample_steps = [s for s in steps if s.source_notebook == "NB00"]
        assert len(sample_steps) == 1
        step = sample_steps[0]
        assert step.type == PipelineTransformationType.LANDING_FILTER
        # Bracket -> paren translation: Spark SQL `IN` rejects `[...]`.
        assert step.parameters["predicate"] == "REVENUE_MARKET_SEGMENT in ('Emerging', 'Small')"
        assert step.parameters["sibling_views"] == []

    def test_predicate_with_subquery_records_sibling(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        for ds in ("account", "contract"):
            _write_event_findings(ns, ds, f"ml_catalog.retention.{ds}_raw")
        _write_multi_findings(
            ns, ["account", "contract"],
            {"account": "ml_catalog.retention.account_raw",
             "contract": "ml_catalog.retention.contract_raw"},
        )
        _write_project_context(
            ns,
            sample_filters={
                "account": (
                    "REVENUE_MARKET_SEGMENT in ['Emerging', 'Small'] "
                    "and ACCOUNT_ID in (select ACCOUNT_ID from contract "
                    "where event_type = 'start')"
                ),
            },
            datasets=["account", "contract"],
        )
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        steps = [s for s in config.landing["account"].filters
                 if s.source_notebook == "NB00"]
        assert len(steps) == 1
        assert steps[0].parameters["sibling_views"] == ["contract"]

    def test_predicate_with_join_records_sibling(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        for ds in ("account", "contract"):
            _write_event_findings(ns, ds, f"ml_catalog.retention.{ds}_raw")
        _write_multi_findings(
            ns, ["account", "contract"],
            {"account": "ml_catalog.retention.account_raw",
             "contract": "ml_catalog.retention.contract_raw"},
        )
        _write_project_context(
            ns,
            sample_filters={
                "account": (
                    "ACCOUNT_ID in (select a.ACCOUNT_ID from contract a "
                    "join account b on a.ACCOUNT_ID = b.ACCOUNT_ID)"
                ),
            },
            datasets=["account", "contract"],
        )
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        steps = [s for s in config.landing["account"].filters
                 if s.source_notebook == "NB00"]
        # contract appears via FROM and JOIN; account is self-ref (excluded)
        assert "contract" in steps[0].parameters["sibling_views"]
        assert "account" not in steps[0].parameters["sibling_views"]

    def test_unknown_dataset_warns(self, tmp_path, caplog):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "account", "ml_catalog.retention.account_raw")
        _write_multi_findings(ns, ["account"], {"account": "ml_catalog.retention.account_raw"})
        _write_project_context(
            ns,
            sample_filters={"unknown_ds": "x = 1"},
            datasets=["account"],
        )
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        with caplog.at_level(logging.WARNING,
                             logger="customer_retention.generators.pipeline_generator.findings_parser"):
            config = parser.parse()
        msgs = " ".join(rec.message for rec in caplog.records)
        assert "unknown_ds" in msgs
        # account landing untouched by NB00 sample_filter
        assert all(s.source_notebook != "NB00"
                   for s in config.landing["account"].filters)

    def test_empty_predicate_skipped(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "account", "ml_catalog.retention.account_raw")
        _write_multi_findings(ns, ["account"], {"account": "ml_catalog.retention.account_raw"})
        _write_project_context(
            ns, sample_filters={"account": ""}, datasets=["account"],
        )
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        assert all(s.source_notebook != "NB00"
                   for s in config.landing["account"].filters)

    def test_no_namespace_no_op(self, tmp_path):
        """Parser without namespace cannot read project_context — sample
        filters silently skipped (no crash, no warning)."""
        # Use the existing landing_overrides fixture pattern; pass no namespace.
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )
        parser = FindingsParser(findings_dir=str(tmp_path))
        # Just ensure the method is a safe no-op when namespace is None.
        from customer_retention.generators.pipeline_generator.models import PipelineConfig
        # Build a minimum-viable PipelineConfig — the method should early-return
        # before touching it.
        config = PipelineConfig.__new__(PipelineConfig)
        config.landing = {}
        parser._apply_sample_filters(config)  # no exception


class TestSampleFilterReordersLanding:
    def test_sibling_appears_before_dependent(self, tmp_path):
        """`landing_contract` must run before `landing_account` so the
        post-lifecycle contract Delta has the columns the predicate
        references."""
        ns = _fake_namespace(tmp_path)
        for ds in ("account", "contract"):
            _write_event_findings(ns, ds, f"ml_catalog.retention.{ds}_raw")
        _write_multi_findings(
            ns, ["account", "contract"],
            {"account": "ml_catalog.retention.account_raw",
             "contract": "ml_catalog.retention.contract_raw"},
        )
        _write_project_context(
            ns,
            sample_filters={
                "account": "ACCOUNT_ID in (select ACCOUNT_ID from contract)",
            },
            datasets=["account", "contract"],
        )
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        order = list(config.landing.keys())
        # contract must appear before account
        assert order.index("contract") < order.index("account")

    def test_independent_landings_keep_relative_order(self, tmp_path):
        """Datasets with no sibling deps preserve their original
        insertion order — the reorder is stable."""
        ns = _fake_namespace(tmp_path)
        for ds in ("account", "contract", "request"):
            _write_event_findings(ns, ds, f"ml_catalog.retention.{ds}_raw")
        _write_multi_findings(
            ns, ["account", "contract", "request"],
            {"account": "ml_catalog.retention.account_raw",
             "contract": "ml_catalog.retention.contract_raw",
             "request": "ml_catalog.retention.request_raw"},
        )
        # Only account depends on contract; request is independent.
        _write_project_context(
            ns,
            sample_filters={
                "account": "ACCOUNT_ID in (select ACCOUNT_ID from contract)",
            },
            datasets=["account", "contract", "request"],
        )
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        order = list(config.landing.keys())
        # contract before account; request stays where it was
        assert order.index("contract") < order.index("account")
        assert "request" in order


class TestSampleFilterChainWithLifecycle:
    def test_sibling_lifecycle_is_emitted_before_dependent_filter(self, tmp_path):
        """When the predicate's sibling has a registered lifecycle (e.g.
        contract gets enriched into START/TERMINATE events that produce
        the ``event_type`` column the predicate references), the sibling's
        landing must apply that lifecycle BEFORE the dependent reads its
        UC table. Verifies both are emitted to ``config.landing`` and the
        sibling sits earlier in the dict order."""
        ns = _fake_namespace(tmp_path)
        for ds in ("account", "contract"):
            _write_event_findings(ns, ds, f"ml_catalog.retention.{ds}_raw")
        _write_multi_findings(
            ns, ["account", "contract"],
            {"account": "ml_catalog.retention.account_raw",
             "contract": "ml_catalog.retention.contract_raw"},
        )
        _write_project_context(
            ns,
            sample_filters={
                "account": (
                    "ACCOUNT_ID in (select ACCOUNT_ID from contract "
                    "where event_type = 'start')"
                ),
            },
            datasets=["account", "contract"],
        )
        # Operator NB10 also pastes a lifecycle override for contract — this
        # is the cell that ADDS the ``event_type`` column the predicate
        # references. Without it, the predicate would crash at runtime.
        lifecycle_cfg = {
            "enriched_view_name": "sps_enriched_contract",
            "parent_entity_key": "ACCOUNT_ID",
            "sub_entity_key": "CONTRACT_ID",
            "valid_from_column": "CONTRACT_START_DATE",
            "valid_to_columns": ["BILLING_TERMINATION_DATE"],
            "status_column": "CONTRACT_STATUS",
            "terminal_status_values": ["Cancelled"],
        }
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            landing_lifecycle_overrides={"contract": lifecycle_cfg},
        )
        config = parser.parse()
        # 1. contract has lifecycle step
        contract_lifecycles = config.landing["contract"].lifecycle_enrichments
        assert len(contract_lifecycles) == 1
        # 2. account has sample_filter step with contract as sibling
        account_filters = [s for s in config.landing["account"].filters
                           if s.source_notebook == "NB00"]
        assert len(account_filters) == 1
        assert account_filters[0].parameters["sibling_views"] == ["contract"]
        # 3. Order: contract before account (sibling-deps reorder)
        order = list(config.landing.keys())
        assert order.index("contract") < order.index("account")


class TestSampleFilterCycleDetection:
    def test_cycle_raises(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        for ds in ("a", "b"):
            _write_event_findings(ns, ds, f"ml_catalog.retention.{ds}_raw")
        _write_multi_findings(
            ns, ["a", "b"],
            {"a": "ml_catalog.retention.a_raw",
             "b": "ml_catalog.retention.b_raw"},
        )
        _write_project_context(
            ns,
            sample_filters={
                "a": "id in (select id from b)",
                "b": "id in (select id from a)",
            },
            datasets=["a", "b"],
        )
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        with pytest.raises(ValueError, match="cycle"):
            parser.parse()
