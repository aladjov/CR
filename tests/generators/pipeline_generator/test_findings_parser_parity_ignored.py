"""``parity_ignored_features`` escape hatch.

When NB10 hits a `FeatureSpec parity violation` for a feature the user has
knowingly accepted as drift, they can list it in NB10's `PARITY_IGNORED_FEATURES`
and rerun without first re-running NB08. The on-disk `feature_spec.yaml` in
the run namespace is NEVER mutated; the generator writes a stripped copy into
``output_dir/findings/feature_spec.yaml`` and repoints ``feature_spec_path`` at
it so the runtime gate inside the generated training step also sees the
stripped spec.
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


def _write_minimal_fixture(ns, dataset_name: str = "case") -> None:
    case_findings = {
        "source_path": f"/data/{dataset_name}.csv", "source_format": "csv",
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
    path = ns.dataset_findings_dir(dataset_name) / f"{dataset_name}_findings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(case_findings, f)
    multi = {
        "datasets": {dataset_name: {
            "name": dataset_name, "findings_path": str(path),
            "source_path": f"/data/{dataset_name}.csv", "granularity": "entity_level",
            "row_count": 100, "column_count": 2,
            "entity_column": "entity_id", "target_column": "churn",
        }},
        "primary_entity_dataset": dataset_name,
        "event_datasets": [],
        "excluded_datasets": [],
    }
    with ns.multi_dataset_findings_path.open("w") as f:
        yaml.safe_dump(multi, f)


class TestParityIgnoredFeaturesConstruction:
    def test_default_empty(self, tmp_path):
        parser = FindingsParser(findings_dir=str(tmp_path))
        assert parser.parity_ignored_features == frozenset()

    def test_explicit_iterable_accepted(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path), parity_ignored_features=["a", "b"],
        )
        assert parser.parity_ignored_features == frozenset({"a", "b"})

    def test_empty_list_treated_as_disabled(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path), parity_ignored_features=[],
        )
        assert parser.parity_ignored_features == frozenset()

    def test_none_treated_as_disabled(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path), parity_ignored_features=None,
        )
        assert parser.parity_ignored_features == frozenset()

    def test_blank_entries_filtered(self, tmp_path):
        parser = FindingsParser(
            findings_dir=str(tmp_path), parity_ignored_features=["", "a", None],
        )
        assert parser.parity_ignored_features == frozenset({"a"})


class TestParityIgnoredAtSpecLoad:
    def test_load_strips_ignored_features_and_transforms(self, tmp_path, caplog):
        ns = _fake_namespace(tmp_path)
        _build_spec(["keep", "drop_me"]).save(ns.feature_spec_path)
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            parity_ignored_features=["drop_me"],
        )
        with caplog.at_level(logging.WARNING, logger="customer_retention.generators.pipeline_generator.findings_parser"):
            spec = parser._load_feature_spec()
        assert spec is not None
        assert spec.selected_features == ["keep"]
        assert {ft.column for ft in spec.fitted_transforms} == {"keep"}
        assert any("PARITY_IGNORED_FEATURES" in rec.message for rec in caplog.records)

    def test_on_disk_spec_unchanged(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _build_spec(["keep", "drop_me"]).save(ns.feature_spec_path)
        original_yaml = ns.feature_spec_path.read_text()
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            parity_ignored_features=["drop_me"],
        )
        parser._load_feature_spec()
        assert ns.feature_spec_path.read_text() == original_yaml

    def test_no_op_when_ignored_not_in_spec(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _build_spec(["keep_a", "keep_b"]).save(ns.feature_spec_path)
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            parity_ignored_features=["unknown"],
        )
        spec = parser._load_feature_spec()
        assert spec.selected_features == ["keep_a", "keep_b"]


class TestParityIgnoredEndToEnd:
    def test_strict_mode_passes_when_only_violation_is_ignored(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_minimal_fixture(ns)
        _build_spec(["amount", "missing_in_pipeline"]).save(ns.feature_spec_path)
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            parity_ignored_features=["missing_in_pipeline"],
        )
        config = parser.parse()
        assert config is not None

    def test_strict_mode_still_raises_on_unignored_violation(self, tmp_path):
        ns = _fake_namespace(tmp_path)
        _write_minimal_fixture(ns)
        _build_spec(["amount", "still_missing"]).save(ns.feature_spec_path)
        parser = FindingsParser(
            findings_dir=str(ns.merged_dir), namespace=ns,
            parity_ignored_features=["something_else"],
        )
        with pytest.raises(ValueError, match="parity violation"):
            parser.parse()


class TestGeneratorMaterializesPatchedSpec:
    def test_pipeline_generator_writes_stripped_spec_and_repoints(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        ns = _fake_namespace(tmp_path)
        _write_minimal_fixture(ns)
        _build_spec(["amount", "skipped_feature"]).save(ns.feature_spec_path)

        out_dir = tmp_path / "out"
        gen = PipelineGenerator(
            findings_dir=str(ns.merged_dir), output_dir=str(out_dir),
            pipeline_name="p", namespace=ns,
            parity_ignored_features=["skipped_feature"],
        )
        config = gen._build_config()
        patched_path = out_dir / "findings" / "feature_spec.yaml"
        assert patched_path.exists()
        assert config.feature_spec_path == str(patched_path)
        loaded = FeatureSpec.load(patched_path)
        assert loaded.selected_features == ["amount"]
        ns_loaded = FeatureSpec.load(ns.feature_spec_path)
        assert ns_loaded.selected_features == ["amount", "skipped_feature"]

    def test_databricks_generator_writes_stripped_spec(self, tmp_path):
        from customer_retention.generators.pipeline_generator.databricks_generator import (
            DatabricksPipelineGenerator,
        )

        ns = _fake_namespace(tmp_path)
        _write_minimal_fixture(ns)
        _build_spec(["amount", "skipped_feature"]).save(ns.feature_spec_path)

        out_dir = tmp_path / "out_db"
        gen = DatabricksPipelineGenerator(
            findings_dir=str(ns.merged_dir), output_dir=str(out_dir),
            pipeline_name="p", namespace=ns,
            parity_ignored_features=["skipped_feature"],
        )
        config = gen._build_config()
        patched_path = out_dir / "findings" / "feature_spec.yaml"
        assert patched_path.exists()
        assert config.feature_spec_path == str(patched_path)
        loaded = FeatureSpec.load(patched_path)
        assert loaded.selected_features == ["amount"]

    def test_no_materialize_when_empty(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        ns = _fake_namespace(tmp_path)
        _write_minimal_fixture(ns)
        _build_spec(["amount"]).save(ns.feature_spec_path)

        out_dir = tmp_path / "out"
        gen = PipelineGenerator(
            findings_dir=str(ns.merged_dir), output_dir=str(out_dir),
            pipeline_name="p", namespace=ns,
        )
        config = gen._build_config()
        assert config.feature_spec_path == str(ns.feature_spec_path)
        assert not (out_dir / "findings" / "feature_spec.yaml").exists()


class TestGeneratorForwardsParityIgnoredFeatures:
    def test_pipeline_generator_forwards(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        gen = PipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p", parity_ignored_features=["a", "b"],
        )
        assert gen._parser.parity_ignored_features == frozenset({"a", "b"})

    def test_databricks_generator_forwards(self, tmp_path):
        from customer_retention.generators.pipeline_generator.databricks_generator import (
            DatabricksPipelineGenerator,
        )

        gen = DatabricksPipelineGenerator(
            findings_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
            pipeline_name="p", parity_ignored_features=["a"],
        )
        assert gen._parser.parity_ignored_features == frozenset({"a"})
