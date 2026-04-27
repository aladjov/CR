"""``raw_source_path_overrides`` — NB10's per-dataset path-rewire mechanism.

Operator scenario: a stale findings file from a pre-fix NB00 run carries
`raw_source_path: global_temp.sps_filtered_case`. Re-running NB00 -> NB05
to regenerate findings is expensive. NB10 already resolves a `datasets`
dict from `_namespace.original_datasets` (or paste-fallback
`DATASETS_ORIGINAL_FALLBACK`). Threading that dict to the generators
as `raw_source_path_overrides` lets the in-flight generation use the
upstream persistent paths WITHOUT mutating the on-disk findings.

Tests cover:
* Default (no overrides) preserves the findings-recorded path.
* Override swaps `LandingLayerConfig.raw_source_path` and
  `SourceConfig.raw_source_path` for matching dataset names.
* Override applies even when the findings path is `global_temp.*`
  (the prime field repro from `spschurn-e4ad6e1b`).
* Generators forward the override into FindingsParser.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser


def _fake_namespace(tmp_path: Path):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    ns = RunNamespace(root=tmp_path, run_id="test-run")
    ns.merged_dir.mkdir(parents=True, exist_ok=True)
    return ns


def _write_findings(ns, dataset_name: str, raw_source_path: str) -> None:
    case_findings = {
        "source_path": raw_source_path,
        "source_format": "delta",
        "row_count": 100, "column_count": 2,
        "target_column": "churn",
        "identifier_columns": ["entity_id"],
        "datetime_columns": ["event_timestamp"],
        "columns": {
            "entity_id": {"name": "entity_id", "inferred_type": "identifier",
                          "confidence": 1.0, "evidence": []},
            "event_timestamp": {"name": "event_timestamp", "inferred_type": "datetime",
                                "confidence": 1.0, "evidence": []},
        },
    }
    path = ns.dataset_findings_dir(dataset_name) / f"{dataset_name}_findings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(case_findings, f)
    multi = {
        "datasets": {dataset_name: {
            "name": dataset_name, "findings_path": str(path),
            "source_path": raw_source_path, "granularity": "entity_level",
            "row_count": 100, "column_count": 2,
            "entity_column": "entity_id", "target_column": "churn",
            "raw_source_path": raw_source_path,
        }},
        "primary_entity_dataset": dataset_name,
        "event_datasets": [],
        "excluded_datasets": [],
    }
    with ns.multi_dataset_findings_path.open("w") as f:
        yaml.safe_dump(multi, f)


class TestParserConstruction:
    def test_default_empty(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        assert parser.raw_source_path_overrides == {}

    def test_explicit_dict_accepted(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path),
            raw_source_path_overrides={"case": "ml_catalog.retention.case_raw"},
        )
        assert parser.raw_source_path_overrides == {"case": "ml_catalog.retention.case_raw"}

    def test_property_returns_copy(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path),
            raw_source_path_overrides={"case": "ml_catalog.retention.case_raw"},
        )
        ret = parser.raw_source_path_overrides
        ret["mutated"] = "by_caller"
        assert "mutated" not in parser.raw_source_path_overrides


class TestResolveRawSource:
    def test_no_override_falls_back_to_argument(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        result = parser._resolve_raw_source("case", "ml_catalog.retention.case_raw")
        assert result == "ml_catalog.retention.case_raw"

    def test_override_takes_precedence(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path),
            raw_source_path_overrides={"case": "ml_catalog.retention.case_persistent"},
        )
        result = parser._resolve_raw_source("case", "global_temp.sps_filtered_case")
        assert result == "ml_catalog.retention.case_persistent"

    def test_override_for_other_dataset_doesnt_match(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path),
            raw_source_path_overrides={"case": "ml_catalog.retention.case_raw"},
        )
        result = parser._resolve_raw_source("contract", "global_temp.sps_enriched_contract")
        assert result == "global_temp.sps_enriched_contract"

    def test_none_override_returns_none(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        result = parser._resolve_raw_source("case", None)
        assert result is None


class TestParseAppliesOverrides:
    def test_source_config_raw_source_path_overridden(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_findings(ns, "case", "global_temp.sps_filtered_case")
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            raw_source_path_overrides={"case": "ml_catalog.retention.case_raw"},
        )
        config = parser.parse()
        case_source = next(s for s in config.sources if s.name == "case")
        assert case_source.raw_source_path == "ml_catalog.retention.case_raw"
        assert "global_temp." not in (case_source.raw_source_path or "")

    def test_landing_config_raw_source_path_overridden_for_event_level(self, tmp_path):
        """Discovered event-level datasets (the SPS shape — entity-level in
        multi_dataset_findings, event_level in per-dataset findings) get their
        landing entry built by `_build_discovered_landing_configs`. The
        override must apply there too."""
        ns = _fake_namespace(tmp_path)
        # Build per-dataset findings WITH event-level metadata so the
        # discovery path triggers.
        case_findings = {
            "source_path": "global_temp.sps_filtered_case",
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
        ds_path = ns.dataset_findings_dir("case") / "case_findings.yaml"
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        with ds_path.open("w") as f:
            yaml.safe_dump(case_findings, f)
        multi = {
            "datasets": {"case": {
                "name": "case", "findings_path": str(ds_path),
                "source_path": "global_temp.sps_filtered_case",
                "granularity": "event_level",
                "row_count": 1000, "column_count": 3,
                "entity_column": "entity_id", "target_column": "churn",
                "raw_source_path": "global_temp.sps_filtered_case",
            }},
            "primary_entity_dataset": "case",
            "event_datasets": ["case"],
            "excluded_datasets": [],
        }
        with ns.multi_dataset_findings_path.open("w") as f:
            yaml.safe_dump(multi, f)

        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            raw_source_path_overrides={"case": "ml_catalog.retention.case_raw"},
        )
        config = parser.parse()
        assert "case" in config.landing
        assert config.landing["case"].raw_source_path == "ml_catalog.retention.case_raw"
        case_source = next(s for s in config.sources if s.name == "case")
        assert case_source.raw_source_path == "ml_catalog.retention.case_raw"

    def test_no_override_preserves_findings_path(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_findings(ns, "case", "ml_catalog.retention.case_raw")
        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        case_source = next(s for s in config.sources if s.name == "case")
        assert case_source.raw_source_path == "ml_catalog.retention.case_raw"


class TestGeneratorForwardsOverrides:
    def test_pipeline_generator_forwards(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        gen = PipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p",
            raw_source_path_overrides={"case": "ml_catalog.retention.case_raw"},
        )
        assert gen._parser.raw_source_path_overrides == {"case": "ml_catalog.retention.case_raw"}

    def test_databricks_generator_forwards(self, tmp_path):
        from customer_retention.generators.pipeline_generator.databricks_generator import (
            DatabricksPipelineGenerator,
        )

        gen = DatabricksPipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p",
            raw_source_path_overrides={"contract": "ml_catalog.retention.contract_raw"},
        )
        assert gen._parser.raw_source_path_overrides == {
            "contract": "ml_catalog.retention.contract_raw",
        }

    def test_pipeline_generator_default_is_empty(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        gen = PipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p",
        )
        assert gen._parser.raw_source_path_overrides == {}
