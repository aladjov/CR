"""``landing_lifecycle_overrides`` / ``landing_filter_overrides`` — NB10's
escape hatch for pre-migration NB00s.

Operator scenario: NB00 still uses imperative `@cr:user_code` cells, so
the registered overrides chain in `recommendations.yaml.landing` never
received the `add_landing_filter` / `add_landing_lifecycle_enrichment`
calls. Generated landing then fails downstream because no filter or
lifecycle enrichment is applied to the upstream raw table. NB10 lets
the operator paste the per-dataset config directly without re-running
NB00 -> NB05.

Tests cover:
* Default (no overrides) leaves landing.filters / lifecycle_enrichments
  unchanged.
* Override appends a TransformationStep with the right type + parameters
  for the matching dataset.
* Unknown-dataset entries are skipped with a warning (not a hard error,
  so a stale paste-fallback doesn't block generation).
* Generators forward both overrides into FindingsParser.
"""
from __future__ import annotations

import logging
from pathlib import Path

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
    """Write per-dataset findings + multi_dataset_findings for a single
    event-level dataset so the parser builds a `LandingLayerConfig` for it."""
    case_findings = {
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
        yaml.safe_dump(case_findings, f)
    multi = {
        "datasets": {dataset_name: {
            "name": dataset_name, "findings_path": str(ds_path),
            "source_path": raw_source_path,
            "granularity": "event_level",
            "row_count": 1000, "column_count": 3,
            "entity_column": "entity_id", "target_column": "churn",
            "raw_source_path": raw_source_path,
        }},
        "primary_entity_dataset": dataset_name,
        "event_datasets": [dataset_name],
        "excluded_datasets": [],
    }
    with ns.multi_dataset_findings_path.open("w") as f:
        yaml.safe_dump(multi, f)


_LIFECYCLE_CFG = {
    "enriched_view_name": "sps_enriched_contract",
    "parent_entity_key": "ACCOUNT_ID",
    "sub_entity_key": "CONTRACT_ID",
    "valid_from_column": "CONTRACT_START_DATE",
    "valid_to_columns": ["CONTRACT_END_DATE"],
}


class TestParserConstruction:
    def test_default_empty(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        assert parser.landing_lifecycle_overrides == {}
        assert parser.landing_filter_overrides == {}

    def test_explicit_overrides_accepted(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path),
            landing_lifecycle_overrides={"contract": _LIFECYCLE_CFG},
            landing_filter_overrides={"request": "ACCOUNT_ID IS NOT NULL"},
        )
        assert parser.landing_lifecycle_overrides == {"contract": _LIFECYCLE_CFG}
        assert parser.landing_filter_overrides == {"request": "ACCOUNT_ID IS NOT NULL"}

    def test_property_returns_copy(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path),
            landing_lifecycle_overrides={"contract": _LIFECYCLE_CFG},
            landing_filter_overrides={"request": "ACCOUNT_ID IS NOT NULL"},
        )
        ret_lc = parser.landing_lifecycle_overrides
        ret_lc["mutated"] = {}
        ret_f = parser.landing_filter_overrides
        ret_f["mutated"] = "x"
        assert "mutated" not in parser.landing_lifecycle_overrides
        assert "mutated" not in parser.landing_filter_overrides


class TestApplyLandingOverrides:
    def test_lifecycle_override_appends_transformation_step(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "contract", "ml_catalog.retention.contract_raw")
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            landing_lifecycle_overrides={"contract": _LIFECYCLE_CFG},
        )
        config = parser.parse()
        contract_landing = config.landing["contract"]
        assert len(contract_landing.lifecycle_enrichments) == 1
        step = contract_landing.lifecycle_enrichments[0]
        assert step.type == PipelineTransformationType.LANDING_LIFECYCLE_ENRICHMENT
        assert step.parameters["config"] == _LIFECYCLE_CFG
        assert step.source_notebook == "NB10"

    def test_filter_override_appends_transformation_step(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "request", "ml_catalog.retention.request_raw")
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            landing_filter_overrides={"request": "ACCOUNT_ID IS NOT NULL AND CREATED_DATE IS NOT NULL"},
        )
        config = parser.parse()
        request_landing = config.landing["request"]
        assert len(request_landing.filters) == 1
        step = request_landing.filters[0]
        assert step.type == PipelineTransformationType.LANDING_FILTER
        assert step.parameters["predicate"] == "ACCOUNT_ID IS NOT NULL AND CREATED_DATE IS NOT NULL"
        assert step.source_notebook == "NB10"

    def test_no_override_leaves_landing_steps_untouched(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "contract", "ml_catalog.retention.contract_raw")
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        contract_landing = config.landing["contract"]
        assert contract_landing.lifecycle_enrichments == []
        assert contract_landing.filters == []

    def test_unknown_dataset_warns_but_does_not_raise(self, tmp_path, caplog):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "contract", "ml_catalog.retention.contract_raw")
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            landing_lifecycle_overrides={"unknown_dataset": _LIFECYCLE_CFG},
            landing_filter_overrides={"also_unknown": "x = 1"},
        )
        with caplog.at_level(logging.WARNING, logger="customer_retention.generators.pipeline_generator.findings_parser"):
            config = parser.parse()
        assert config is not None
        msgs = " ".join(rec.message for rec in caplog.records)
        assert "unknown_dataset" in msgs
        assert "also_unknown" in msgs
        # Known dataset must remain clean
        assert config.landing["contract"].lifecycle_enrichments == []
        assert config.landing["contract"].filters == []

    def test_empty_lifecycle_value_is_skipped(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "contract", "ml_catalog.retention.contract_raw")
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            landing_lifecycle_overrides={"contract": {}},
        )
        config = parser.parse()
        assert config.landing["contract"].lifecycle_enrichments == []

    def test_empty_filter_value_is_skipped(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "request", "ml_catalog.retention.request_raw")
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            landing_filter_overrides={"request": ""},
        )
        config = parser.parse()
        assert config.landing["request"].filters == []

    def test_lifecycle_override_accepts_dataclass_instance(self, tmp_path):
        """Operators paste `LifecycleEnrichmentConfig(...)` verbatim from
        `sps_notebook_registered_overrides.md`. The override must coerce
        a dataclass instance the same as a plain dict — otherwise the
        natural paste path raises `cannot convert dictionary update
        sequence element #0 to a sequence` from `dict(<dataclass>)`."""
        from customer_retention.stages.lifecycle.config import LifecycleEnrichmentConfig

        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "contract", "ml_catalog.retention.contract_raw")
        cfg = LifecycleEnrichmentConfig(
            enriched_view_name="sps_enriched_contract",
            parent_entity_key="ACCOUNT_ID",
            sub_entity_key="CONTRACT_ID",
            valid_from_column="CONTRACT_START_DATE",
            valid_to_columns=("CONTRACT_END_DATE",),
        )
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            landing_lifecycle_overrides={"contract": cfg},
        )
        config = parser.parse()
        steps = config.landing["contract"].lifecycle_enrichments
        assert len(steps) == 1
        coerced = steps[0].parameters["config"]
        assert isinstance(coerced, dict)
        assert coerced["enriched_view_name"] == "sps_enriched_contract"
        assert coerced["valid_from_column"] == "CONTRACT_START_DATE"
        # tuples are normalized via dataclasses.asdict -> tuple stays tuple,
        # but to_dict() lists them. Either is acceptable downstream because
        # `LifecycleEnrichmentConfig.from_dict(...)` re-tuples them.
        assert list(coerced["valid_to_columns"]) == ["CONTRACT_END_DATE"]

    def test_lifecycle_override_rejects_non_mappable(self, tmp_path):
        import pytest

        ns = _fake_namespace(tmp_path)
        _write_event_findings(ns, "contract", "ml_catalog.retention.contract_raw")
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            landing_lifecycle_overrides={"contract": ["not", "a", "dict"]},
        )
        with pytest.raises(TypeError, match="LifecycleEnrichmentConfig"):
            parser.parse()


class TestGeneratorForwardsLandingOverrides:
    def test_pipeline_generator_forwards(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        gen = PipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p",
            landing_lifecycle_overrides={"contract": _LIFECYCLE_CFG},
            landing_filter_overrides={"request": "x=1"},
        )
        assert gen._parser.landing_lifecycle_overrides == {"contract": _LIFECYCLE_CFG}
        assert gen._parser.landing_filter_overrides == {"request": "x=1"}

    def test_databricks_generator_forwards(self, tmp_path):
        from customer_retention.generators.pipeline_generator.databricks_generator import (
            DatabricksPipelineGenerator,
        )

        gen = DatabricksPipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p",
            landing_lifecycle_overrides={"contract": _LIFECYCLE_CFG},
            landing_filter_overrides={"request": "x=1"},
        )
        assert gen._parser.landing_lifecycle_overrides == {"contract": _LIFECYCLE_CFG}
        assert gen._parser.landing_filter_overrides == {"request": "x=1"}

    def test_pipeline_generator_default_is_empty(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        gen = PipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p",
        )
        assert gen._parser.landing_lifecycle_overrides == {}
        assert gen._parser.landing_filter_overrides == {}
