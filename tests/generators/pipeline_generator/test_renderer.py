import ast
import json

import pytest

from customer_retention.generators.pipeline_generator.models import (
    BronzeLayerConfig,
    GoldLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TransformationStep,
)


@pytest.fixture
def sample_pipeline_config():
    source1 = SourceConfig(name="customers", path="/data/customers.csv", format="csv", entity_key="customer_id")
    source2 = SourceConfig(name="orders", path="/data/orders.parquet", format="parquet",
                          entity_key="customer_id", time_column="order_date", is_event_level=True)
    bronze1 = BronzeLayerConfig(source=source1, transformations=[
        TransformationStep(type=PipelineTransformationType.IMPUTE_NULL, column="age", parameters={"value": 0}, rationale="Fill nulls")
    ])
    bronze2 = BronzeLayerConfig(source=source2, transformations=[
        TransformationStep(type=PipelineTransformationType.CAP_OUTLIER, column="amount", parameters={"lower": 0, "upper": 10000}, rationale="Cap outliers")
    ])
    silver = SilverLayerConfig(
        joins=[{"left_key": "customer_id", "right_key": "customer_id", "right_source": "orders", "how": "left"}],
        aggregations=[]
    )
    gold = GoldLayerConfig(
        encodings=[TransformationStep(type=PipelineTransformationType.ENCODE, column="category", parameters={"method": "one_hot"}, rationale="Encode")],
        scalings=[TransformationStep(type=PipelineTransformationType.SCALE, column="amount", parameters={"method": "standard"}, rationale="Scale")]
    )
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[source1, source2],
        bronze={"customers": bronze1, "orders": bronze2},
        silver=silver,
        gold=gold,
        output_dir="/output/test_pipeline"
    )


class TestCodeRendererInit:
    def test_renderer_creates_jinja_environment(self):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        assert renderer._env is not None


class TestRenderConfig:
    def test_render_config_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_config(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_config_includes_pipeline_name(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_config(sample_pipeline_config)
        assert "test_pipeline" in result

    def test_render_config_includes_target_column(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_config(sample_pipeline_config)
        assert "churn" in result

    def test_render_config_includes_sources(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_config(sample_pipeline_config)
        assert "customers" in result
        assert "orders" in result

    def test_render_config_is_valid_python(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_config(sample_pipeline_config)
        ast.parse(result)


class TestRenderBronze:
    def test_render_bronze_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert isinstance(result, str)

    def test_render_bronze_includes_source_name(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "customers" in result

    def test_render_bronze_includes_transformations(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "fillna" in result or "impute" in result.lower()

    def test_render_bronze_is_valid_python(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        ast.parse(result)


class TestRenderSilver:
    def test_render_silver_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_silver(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_silver_includes_join(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_silver(sample_pipeline_config)
        assert "merge" in result

    def test_render_silver_is_valid_python(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_silver(sample_pipeline_config)
        ast.parse(result)


class TestRenderGold:
    def test_render_gold_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_gold(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_gold_includes_encoding(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_gold(sample_pipeline_config)
        assert "get_dummies" in result or "LabelEncoder" in result or "encoding" in result.lower()

    def test_render_gold_includes_scaling(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_gold(sample_pipeline_config)
        assert "Scaler" in result or "scaling" in result.lower()

    def test_render_gold_is_valid_python(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)


class TestRenderTraining:
    def test_render_training_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_training(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_training_includes_model(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_training(sample_pipeline_config)
        assert "Classifier" in result or "model" in result.lower()

    def test_render_training_is_valid_python(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestRenderRunner:
    def test_render_runner_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_runner(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_runner_imports_bronze_modules(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_runner(sample_pipeline_config)
        assert "bronze_customers" in result or "customers" in result

    def test_render_runner_is_valid_python(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_runner(sample_pipeline_config)
        ast.parse(result)


class TestRenderWorkflow:
    def test_render_workflow_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_workflow(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_workflow_is_valid_json(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_workflow(sample_pipeline_config)
        parsed = json.loads(result)
        assert "name" in parsed or "tasks" in parsed

    def test_render_workflow_has_dependencies(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_workflow(sample_pipeline_config)
        parsed = json.loads(result)
        assert "tasks" in parsed
        silver_task = next((t for t in parsed["tasks"] if "silver" in t["task_key"]), None)
        assert silver_task is not None
        assert "depends_on" in silver_task


class TestRenderScoring:
    def test_render_scoring_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_scoring_includes_pipeline_name(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)
        assert "test_pipeline" in result

    def test_render_scoring_includes_target_column_reference(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)
        assert "TARGET_COLUMN" in result

    def test_render_scoring_includes_feast_references(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)
        assert "FEAST" in result or "feast" in result.lower()

    def test_render_scoring_includes_mlflow_references(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)
        assert "mlflow" in result.lower()

    def test_render_scoring_includes_holdout_logic(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)
        assert "holdout" in result.lower() or "original_" in result

    def test_render_scoring_includes_validation_metrics(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)
        assert "metrics" in result.lower()

    def test_render_scoring_is_valid_python(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)
        ast.parse(result)


class TestRenderDashboard:
    def test_render_dashboard_returns_string(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_dashboard_is_valid_json(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)
        parsed = json.loads(result)
        assert "cells" in parsed
        assert "nbformat" in parsed

    def test_render_dashboard_includes_pipeline_name(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)
        assert "test_pipeline" in result

    def test_render_dashboard_includes_shap(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)
        assert "shap" in result.lower()

    def test_render_dashboard_includes_customer_browser(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)
        assert "customer" in result.lower() or "browser" in result.lower()

    def test_render_dashboard_includes_metrics(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)
        assert "metrics" in result.lower()

    def test_render_dashboard_includes_mlflow(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)
        assert "mlflow" in result.lower()

    def test_render_dashboard_has_multiple_cells(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)
        parsed = json.loads(result)
        assert len(parsed["cells"]) > 5
