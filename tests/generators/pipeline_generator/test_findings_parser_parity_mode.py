"""Parity-gate strict/warn toggle.

A FeatureSpec parity violation can be either a hard FAIL (default — catches
exploration/generation drift at generation time) or a soft WARN (escape hatch
when the user knowingly accepts drift, e.g. during migration or when the
runtime-generation parity for a feature category is still being validated).

Resolution order: explicit constructor arg > `CR_FEATURE_SPEC_PARITY_MODE`
env var > "strict". Unknown values fail fast at construction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pytest
import yaml

from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
)


def _fake_namespace(tmp_path: Path):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    ns = RunNamespace(root=tmp_path, run_id="test-run")
    ns.merged_dir.mkdir(parents=True, exist_ok=True)
    return ns


def _build_spec(selected: List[str]) -> FeatureSpec:
    return FeatureSpec(
        spec_version=1, exploration_run_id="test-run",
        target_column="churn", entity_column="entity_id",
        timestamp_column="as_of_date", horizon_days=90,
        selected_features=list(selected),
        fitted_transforms=[
            FittedTransform(column=c, action="impute", method="median") for c in selected
        ],
    )


class TestParityModeConstruction:
    def test_default_mode_is_strict(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        assert parser.parity_mode == "strict"

    def test_explicit_warn_accepted(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path), parity_mode="warn")
        assert parser.parity_mode == "warn"

    def test_explicit_strict_accepted(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path), parity_mode="strict")
        assert parser.parity_mode == "strict"

    def test_unknown_mode_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="parity_mode"):
            FindingsParser(findings_dir=str(tmp_path), parity_mode="silent")

    def test_env_var_sets_mode_when_arg_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_FEATURE_SPEC_PARITY_MODE", "warn")
        parser = FindingsParser(findings_dir=str(tmp_path))
        assert parser.parity_mode == "warn"

    def test_env_var_normalized_to_lowercase(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_FEATURE_SPEC_PARITY_MODE", "WARN")
        parser = FindingsParser(findings_dir=str(tmp_path))
        assert parser.parity_mode == "warn"

    def test_explicit_arg_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_FEATURE_SPEC_PARITY_MODE", "warn")
        parser = FindingsParser(findings_dir=str(tmp_path), parity_mode="strict")
        assert parser.parity_mode == "strict"


class TestParityModeBehavior:
    def test_strict_mode_raises_on_missing_feature(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path), parity_mode="strict")
        spec = _build_spec(["missing_feature", "known"])
        pipeline = {"entity_id", "as_of_date", "churn", "known"}
        with pytest.raises(ValueError, match="parity violation"):
            parser._collect_allowlist_drops(spec, pipeline, target_column="churn")

    def test_warn_mode_logs_and_continues(self, tmp_path, caplog):
        parser = FindingsParser(findings_dir=str(tmp_path), parity_mode="warn")
        spec = _build_spec(["missing_feature", "known"])
        pipeline = {"entity_id", "as_of_date", "churn", "known"}
        with caplog.at_level(logging.WARNING, logger="customer_retention.generators.pipeline_generator.findings_parser"):
            drops = parser._collect_allowlist_drops(spec, pipeline, target_column="churn")
        assert drops == set()
        assert any("parity violation" in rec.message for rec in caplog.records)
        assert any("missing_feature" in rec.message for rec in caplog.records)

    def test_warn_mode_no_log_when_all_selected_present(self, tmp_path, caplog):
        parser = FindingsParser(findings_dir=str(tmp_path), parity_mode="warn")
        spec = _build_spec(["a", "b"])
        pipeline = {"entity_id", "as_of_date", "churn", "a", "b", "extra"}
        with caplog.at_level(logging.WARNING, logger="customer_retention.generators.pipeline_generator.findings_parser"):
            drops = parser._collect_allowlist_drops(spec, pipeline, target_column="churn")
        assert drops == {"extra"}
        assert not any("parity violation" in rec.message for rec in caplog.records)

    def test_warn_mode_end_to_end_parse_does_not_raise(self, tmp_path, caplog):
        ns = _fake_namespace(tmp_path)
        case_findings = {
            "source_path": "/data/case.csv", "source_format": "csv",
            "row_count": 100, "column_count": 2,
            "target_column": "churn",
            "identifier_columns": ["entity_id"],
            "datetime_columns": [],
            "columns": {
                "entity_id": {"name": "entity_id", "inferred_type": "identifier",
                              "confidence": 1.0, "evidence": []},
                "amount": {"name": "amount", "inferred_type": "numeric_continuous",
                           "confidence": 1.0, "evidence": []},
            },
        }
        path = ns.dataset_findings_dir("case") / "case_findings.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(case_findings, f)

        multi = {
            "datasets": {"case": {
                "name": "case", "findings_path": str(path),
                "source_path": "/data/case.csv", "granularity": "entity_level",
                "row_count": 100, "column_count": 2,
                "entity_column": "entity_id", "target_column": "churn",
            }},
            "primary_entity_dataset": "case",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        with ns.multi_dataset_findings_path.open("w") as f:
            yaml.safe_dump(multi, f)

        _build_spec(["amount", "feature_that_cannot_exist"]).save(ns.feature_spec_path)

        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns, parity_mode="warn",
        )
        with caplog.at_level(logging.WARNING, logger="customer_retention.generators.pipeline_generator.findings_parser"):
            config = parser.parse()
        assert config is not None
        assert any("feature_that_cannot_exist" in rec.message for rec in caplog.records)


class TestGeneratorConstructorPropagatesMode:
    def test_pipeline_generator_forwards_parity_mode(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        gen = PipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p", parity_mode="warn",
        )
        assert gen._parser.parity_mode == "warn"

    def test_databricks_pipeline_generator_forwards_parity_mode(self, tmp_path):
        from customer_retention.generators.pipeline_generator.databricks_generator import (
            DatabricksPipelineGenerator,
        )

        gen = DatabricksPipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p", parity_mode="warn",
        )
        assert gen._parser.parity_mode == "warn"

    def test_pipeline_generator_default_is_strict(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        gen = PipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p",
        )
        assert gen._parser.parity_mode == "strict"
