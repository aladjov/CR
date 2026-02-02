import ast
from dataclasses import replace

import pytest

from customer_retention.generators.pipeline_generator.models import (
    GoldLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TransformationStep,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


@pytest.fixture
def renderer():
    return CodeRenderer()


@pytest.fixture
def basic_config():
    return PipelineConfig(
        name="test_pipeline",
        target_column="churned",
        sources=[SourceConfig(name="customers", path="customers.parquet", format="parquet", entity_key="customer_id")],
        bronze={},
        silver=SilverLayerConfig(),
        gold=GoldLayerConfig(
            encodings=[TransformationStep(type=PipelineTransformationType.ENCODE, column="region",
                                          parameters={"method": "label"}, rationale="Encode region")],
            scalings=[TransformationStep(type=PipelineTransformationType.SCALE, column="income",
                                         parameters={"method": "standard"}, rationale="Normalize income")],
        ),
        output_dir="./output",
        recommendations_hash="abc12345",
    )


@pytest.fixture
def fit_mode_true_config(basic_config):
    return replace(basic_config, fit_mode=True, artifacts_path="./experiments/artifacts/abc12345")


@pytest.fixture
def fit_mode_false_config(basic_config):
    return replace(basic_config, fit_mode=False, artifacts_path="./experiments/artifacts/abc12345")


class TestPipelineConfigFitMode:
    def test_config_has_fit_mode_default_true(self):
        config = PipelineConfig(
            name="test", target_column="target",
            sources=[SourceConfig(name="src", path="data.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir="./out"
        )
        assert config.fit_mode is True

    def test_config_has_artifacts_path_default_none(self):
        config = PipelineConfig(
            name="test", target_column="target",
            sources=[SourceConfig(name="src", path="data.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir="./out"
        )
        assert config.artifacts_path is None

    def test_config_accepts_fit_mode_false(self):
        config = PipelineConfig(
            name="test", target_column="target",
            sources=[SourceConfig(name="src", path="data.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir="./out",
            fit_mode=False, artifacts_path="./artifacts"
        )
        assert config.fit_mode is False
        assert config.artifacts_path == "./artifacts"


class TestGoldTemplateFitMode:
    def test_gold_template_fit_mode_true_imports_artifact_store(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "ArtifactStore" in code

    def test_gold_template_fit_mode_true_creates_store(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "_store = ArtifactStore(" in code

    def test_gold_template_fit_mode_true_uses_fit_transform(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "fit_transform" in code

    def test_gold_template_fit_mode_true_has_scaling(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "FittedScaler" in code

    def test_gold_template_fit_mode_true_saves_manifest(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "_store.save_manifest()" in code

    def test_gold_template_fit_mode_false_loads_store(self, renderer, fit_mode_false_config):
        code = renderer.render_gold(fit_mode_false_config)
        assert "ArtifactStore.from_manifest" in code

    def test_gold_template_fit_mode_false_uses_transform(self, renderer, fit_mode_false_config):
        code = renderer.render_gold(fit_mode_false_config)
        assert ".transform(" in code

    def test_gold_template_is_valid_python(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        ast.parse(code)

    def test_gold_template_fit_mode_false_is_valid_python(self, renderer, fit_mode_false_config):
        code = renderer.render_gold(fit_mode_false_config)
        ast.parse(code)


class TestConfigTemplateArtifactsPath:
    def test_config_template_includes_artifacts_path(self, renderer, fit_mode_true_config):
        code = renderer.render_config(fit_mode_true_config)
        assert "ARTIFACTS_PATH" in code or "artifacts" in code.lower()

    def test_config_template_is_valid_python(self, renderer, fit_mode_true_config):
        code = renderer.render_config(fit_mode_true_config)
        ast.parse(code)


class TestGoldTemplateNoFitMode:
    def test_gold_template_default_behavior_when_no_fit_mode(self, renderer, basic_config):
        code = renderer.render_gold(basic_config)
        ast.parse(code)
