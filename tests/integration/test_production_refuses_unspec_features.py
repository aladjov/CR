from pathlib import Path

import pytest
import yaml

from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f)


def _fake_namespace(tmp_path: Path):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    ns = RunNamespace(root=tmp_path, run_id="test-run")
    ns.merged_dir.mkdir(parents=True, exist_ok=True)
    return ns


def _write_minimal_exploration_fixture(tmp_path: Path, dataset_name: str = "src"):
    ns = _fake_namespace(tmp_path)
    findings_doc = {
        "source_path": str(tmp_path / "data.csv"),
        "source_format": "csv",
        "row_count": 1000, "column_count": 3,
        "target_column": "churn",
        "identifier_columns": ["entity_id"],
        "datetime_columns": ["as_of_date"],
        "columns": {},
    }
    findings_path = ns.dataset_findings_dir(dataset_name) / f"{dataset_name}_findings.yaml"
    _write_yaml(findings_path, findings_doc)

    multi_doc = {
        "datasets": {
            dataset_name: {
                "name": dataset_name,
                "findings_path": str(findings_path),
                "source_path": str(tmp_path / "data.csv"),
                "granularity": "entity_level",
                "row_count": 1000, "column_count": 3,
                "entity_column": "entity_id",
                "target_column": "churn",
            },
        },
        "primary_entity_dataset": dataset_name,
        "event_datasets": [],
        "excluded_datasets": [],
    }
    _write_yaml(ns.multi_dataset_findings_path, multi_doc)
    return ns


def _build_spec(selected_features, run_id="test-run"):
    return FeatureSpec(
        exploration_run_id=run_id,
        target_column="churn",
        entity_column="entity_id",
        timestamp_column="as_of_date",
        horizon_days=90,
        selected_features=list(selected_features),
        fitted_transforms=[
            FittedTransform(column=c, action="impute", method="median")
            for c in selected_features
        ],
    )


class TestParserRejectsMissingSpecFeatures:
    def test_parse_raises_when_spec_column_missing_from_pipeline(self, tmp_path):
        ns = _write_minimal_exploration_fixture(tmp_path)
        spec = _build_spec(["feature_not_in_pipeline"])
        spec.save(ns.feature_spec_path)

        parser = FindingsParser(findings_dir=str(ns.merged_dir), namespace=ns)
        with pytest.raises(ValueError, match="parity violation"):
            parser.parse()


class TestRenderedTrainingScriptEnforcesSpec:
    def test_generated_script_has_schema_parity_runtime_check(self, tmp_path):
        spec = _build_spec(["feature_a", "feature_b"])
        spec_path = tmp_path / "feature_spec.yaml"
        spec.save(spec_path)

        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
            TrainingConfig,
        )
        config = PipelineConfig(
            name="test_pipeline", target_column="churn",
            sources=[SourceConfig(name="a", path="a.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(),
            output_dir=str(tmp_path / "out"), composite_name="test_abc",
            training=TrainingConfig(),
            feature_spec_path=str(spec_path),
        )
        rendered = CodeRenderer().render_training(config)
        assert "FeatureSpec.load(_FEATURE_SPEC_PATH)" in rendered
        assert "FeatureSpec parity violation" in rendered
        assert "raise RuntimeError" in rendered
        compile(rendered, "<test-training>", "exec")


class TestDatabricksTrainingScriptEnforcesSpec:
    def test_generated_databricks_notebook_has_spec_gate(self, tmp_path):
        spec = _build_spec(["feature_a"])
        spec_path = tmp_path / "feature_spec.yaml"
        spec.save(spec_path)

        from customer_retention.generators.pipeline_generator.databricks_renderer import (
            DatabricksCodeRenderer,
        )
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
            TrainingConfig,
        )
        config = PipelineConfig(
            name="test_pipeline", target_column="churn",
            sources=[SourceConfig(name="a", path="a.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(),
            output_dir=str(tmp_path / "out"), composite_name="test_abc",
            training=TrainingConfig(),
            feature_spec_path=str(spec_path),
        )
        rendered = DatabricksCodeRenderer(catalog="main", schema="db").render_training(config)
        assert "_apply_feature_spec_gate" in rendered
        assert "FeatureSpec parity violation" in rendered


class TestGeneratorCopiesFeatureSpec:
    def test_copy_feature_spec_from_namespace(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        ns = _fake_namespace(tmp_path)
        spec = _build_spec(["a"])
        spec.save(ns.feature_spec_path)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        gen = PipelineGenerator.__new__(PipelineGenerator)
        gen._findings_dir = tmp_path / "nonexistent"
        gen._output_dir = output_dir
        gen._namespace = ns
        gen._copy_feature_spec()

        copied = output_dir / "findings" / "feature_spec.yaml"
        assert copied.exists()
        loaded = FeatureSpec.load(copied)
        assert loaded.selected_features == ["a"]

    def test_copy_feature_spec_is_noop_when_source_missing(self, tmp_path):
        from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

        ns = _fake_namespace(tmp_path)

        gen = PipelineGenerator.__new__(PipelineGenerator)
        gen._findings_dir = tmp_path / "nonexistent"
        gen._output_dir = tmp_path / "out"
        gen._namespace = ns
        gen._copy_feature_spec()

        dest = tmp_path / "out" / "findings" / "feature_spec.yaml"
        assert not dest.exists()
