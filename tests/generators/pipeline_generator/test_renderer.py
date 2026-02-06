import ast
import json

import pytest

from customer_retention.generators.pipeline_generator.models import (
    BronzeLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    TransformationStep,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


@pytest.fixture
def renderer():
    return CodeRenderer()


@pytest.fixture
def sample_pipeline_config(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
    bronze2 = BronzeLayerConfig(source=event_source, transformations=[
        TransformationStep(type=PipelineTransformationType.CAP_OUTLIER, column="amount", parameters={"lower": 0, "upper": 10000}, rationale="Cap outliers")
    ])
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[entity_source, event_source],
        bronze={"customers": bronze_with_impute, "orders": bronze2},
        silver=silver_with_join,
        gold=gold_with_encode_scale,
        output_dir="/output/test_pipeline"
    )


class TestCodeRendererInit:
    def test_renderer_creates_jinja_environment(self, renderer):
        assert renderer._env is not None


class TestRenderConfig:
    def test_render_config_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_config_includes_pipeline_name(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "test_pipeline" in result

    def test_render_config_includes_target_column(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "churn" in result

    def test_render_config_includes_sources(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "customers" in result
        assert "orders" in result

    def test_render_config_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        ast.parse(result)


class TestRenderBronze:
    def test_render_bronze_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert isinstance(result, str)

    def test_render_bronze_includes_source_name(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "customers" in result

    def test_render_bronze_includes_transformations(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "apply_impute_null" in result

    def test_render_bronze_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        ast.parse(result)


class TestRenderSilver:
    def test_render_silver_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_silver_includes_join(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "merge" in result

    def test_render_silver_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        ast.parse(result)


class TestRenderGold:
    def test_render_gold_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_gold_includes_encoding(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "apply_one_hot_encode" in result or "FittedEncoder" in result

    def test_render_gold_includes_scaling(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "FittedScaler" in result

    def test_render_gold_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)


class TestRenderTraining:
    def test_render_training_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_training_includes_model(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "Classifier" in result or "model" in result.lower()

    def test_render_training_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestRenderRunner:
    def test_render_runner_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_runner_imports_bronze_modules(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "bronze_customers" in result or "customers" in result

    def test_render_runner_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        ast.parse(result)


class TestRenderWorkflow:
    def test_render_workflow_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_workflow(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_workflow_is_valid_json(self, renderer, sample_pipeline_config):
        result = renderer.render_workflow(sample_pipeline_config)
        parsed = json.loads(result)
        assert "name" in parsed or "tasks" in parsed

    def test_render_workflow_has_dependencies(self, renderer, sample_pipeline_config):
        result = renderer.render_workflow(sample_pipeline_config)
        parsed = json.loads(result)
        assert "tasks" in parsed
        silver_task = next((t for t in parsed["tasks"] if "silver" in t["task_key"]), None)
        assert silver_task is not None
        assert "depends_on" in silver_task
