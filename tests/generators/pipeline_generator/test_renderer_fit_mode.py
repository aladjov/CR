import ast

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
    basic_config.fit_mode = True
    basic_config.artifacts_path = "./experiments/artifacts/abc12345"
    return basic_config


@pytest.fixture
def fit_mode_false_config(basic_config):
    basic_config.fit_mode = False
    basic_config.artifacts_path = "./experiments/artifacts/abc12345"
    return basic_config


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
    def test_gold_template_fit_mode_true_imports_registry(self, fit_mode_true_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_true_config)
        assert "FitArtifactRegistry" in code

    def test_gold_template_fit_mode_true_creates_registry(self, fit_mode_true_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_true_config)
        assert "_registry = FitArtifactRegistry" in code

    def test_gold_template_fit_mode_true_registers_scalers(self, fit_mode_true_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_true_config)
        assert "_registry.register(" in code
        assert 'artifact_type="scaler"' in code

    def test_gold_template_fit_mode_true_registers_encoders(self, fit_mode_true_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_true_config)
        assert 'artifact_type="encoder"' in code

    def test_gold_template_fit_mode_true_saves_manifest(self, fit_mode_true_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_true_config)
        assert "_registry.save_manifest()" in code

    def test_gold_template_fit_mode_false_loads_registry(self, fit_mode_false_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_false_config)
        assert "FitArtifactRegistry.load_manifest" in code

    def test_gold_template_fit_mode_false_loads_transformers(self, fit_mode_false_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_false_config)
        assert "_registry.load(" in code

    def test_gold_template_is_valid_python(self, fit_mode_true_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_true_config)
        ast.parse(code)

    def test_gold_template_fit_mode_false_is_valid_python(self, fit_mode_false_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(fit_mode_false_config)
        ast.parse(code)


class TestScoringTemplateFitMode:
    def test_scoring_template_loads_fit_artifact_registry(self, fit_mode_false_config):
        renderer = CodeRenderer()
        code = renderer.render_scoring(fit_mode_false_config)
        assert "FitArtifactRegistry" in code

    def test_scoring_template_loads_manifest(self, fit_mode_false_config):
        renderer = CodeRenderer()
        code = renderer.render_scoring(fit_mode_false_config)
        assert "load_manifest" in code

    def test_scoring_template_uses_transform_not_fit(self, fit_mode_false_config):
        renderer = CodeRenderer()
        code = renderer.render_scoring(fit_mode_false_config)
        assert ".transform(" in code
        assert ".fit_transform(" not in code or "# Training" not in code

    def test_scoring_template_is_valid_python(self, fit_mode_false_config):
        renderer = CodeRenderer()
        code = renderer.render_scoring(fit_mode_false_config)
        ast.parse(code)


class TestConfigTemplateArtifactsPath:
    def test_config_template_includes_artifacts_path(self, fit_mode_true_config):
        renderer = CodeRenderer()
        code = renderer.render_config(fit_mode_true_config)
        assert "ARTIFACTS_PATH" in code or "artifacts" in code.lower()

    def test_config_template_is_valid_python(self, fit_mode_true_config):
        renderer = CodeRenderer()
        code = renderer.render_config(fit_mode_true_config)
        ast.parse(code)


class TestGoldTemplateNoFitMode:
    def test_gold_template_default_behavior_when_no_fit_mode(self, basic_config):
        renderer = CodeRenderer()
        code = renderer.render_gold(basic_config)
        ast.parse(code)
