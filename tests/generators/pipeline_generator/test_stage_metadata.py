import ast

import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import DatabricksCodeRenderer
from customer_retention.generators.pipeline_generator.models import (
    BronzeEventConfig,
    BronzeLayerConfig,
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
def dbx_renderer():
    return DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")


@pytest.fixture
def entity_source():
    return SourceConfig(name="customers", path="/data/customers.csv", format="csv", entity_key="customer_id")


@pytest.fixture
def event_source():
    return SourceConfig(name="orders", path="/data/orders.parquet", format="parquet", entity_key="customer_id", time_column="order_date", is_event_level=True)


@pytest.fixture
def pipeline_config(entity_source, event_source):
    from customer_retention.generators.pipeline_generator.models import AggregationWindowConfig
    bronze = BronzeLayerConfig(source=entity_source, transformations=[
        TransformationStep(type=PipelineTransformationType.IMPUTE_NULL, column="age", parameters={"value": 0}, rationale="Fill"),
    ])
    silver = SilverLayerConfig(joins=[{"left_keys": ["customer_id"], "right_keys": ["customer_id"], "right_source": "orders", "how": "left"}], aggregations=[])
    gold = GoldLayerConfig(encodings=[], scalings=[])
    return PipelineConfig(
        name="test_pipeline", target_column="churn",
        sources=[entity_source, event_source],
        bronze={"customers": bronze},
        bronze_event={"orders": BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="order_date",
            aggregation=AggregationWindowConfig(windows=["30d"], value_columns=["amount"], agg_funcs=["sum"]),
        )},
        silver=silver, gold=gold,
        output_dir="/output", composite_name="cust_orde__abc1234",
    )


class TestLocalSilverMetadata:
    def test_silver_imports_run_namespace(self, renderer, pipeline_config):
        result = renderer.render_silver(pipeline_config)
        assert "from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace" in result

    def test_silver_writes_metadata(self, renderer, pipeline_config):
        result = renderer.render_silver(pipeline_config)
        assert "silver_metadata_path" in result

    def test_silver_metadata_contains_rows(self, renderer, pipeline_config):
        result = renderer.render_silver(pipeline_config)
        fn = result[result.index("def run_silver_merge"):]
        assert '"rows"' in fn
        assert '"columns"' in fn
        assert '"elapsed_seconds"' in fn

    def test_silver_template_valid_python(self, renderer, pipeline_config):
        result = renderer.render_silver(pipeline_config)
        ast.parse(result)


class TestLocalGoldMetadata:
    def test_gold_imports_run_namespace(self, renderer, pipeline_config):
        result = renderer.render_gold(pipeline_config)
        assert "from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace" in result

    def test_gold_writes_metadata(self, renderer, pipeline_config):
        result = renderer.render_gold(pipeline_config)
        assert "gold_metadata_path" in result

    def test_gold_metadata_contains_feature_info(self, renderer, pipeline_config):
        result = renderer.render_gold(pipeline_config)
        fn = result[result.index("def run_gold_features"):]
        assert '"feature_count"' in fn
        assert '"feature_version"' in fn

    def test_gold_template_valid_python(self, renderer, pipeline_config):
        result = renderer.render_gold(pipeline_config)
        ast.parse(result)


class TestLocalTrainingMetadata:
    def test_training_writes_to_namespace(self, renderer, pipeline_config):
        result = renderer.render_training(pipeline_config)
        assert "training_metadata_path" in result

    def test_training_template_valid_python(self, renderer, pipeline_config):
        result = renderer.render_training(pipeline_config)
        ast.parse(result)


class TestLocalRunnerBronzeMetadata:
    def test_run_all_imports_run_namespace(self, renderer, pipeline_config):
        result = renderer.render_run_all(pipeline_config)
        assert "from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace" in result

    def test_run_all_writes_bronze_metadata(self, renderer, pipeline_config):
        result = renderer.render_run_all(pipeline_config)
        assert "bronze_metadata_path" in result

    def test_run_all_captures_bronze_results(self, renderer, pipeline_config):
        result = renderer.render_run_all(pipeline_config)
        assert "_bronze_event_results" in result or "_bronze_results" in result

    def test_run_all_prints_silver_summary(self, renderer, pipeline_config):
        result = renderer.render_run_all(pipeline_config)
        assert "Silver complete:" in result

    def test_run_all_prints_gold_summary(self, renderer, pipeline_config):
        result = renderer.render_run_all(pipeline_config)
        assert "Gold complete:" in result

    def test_run_all_valid_python(self, renderer, pipeline_config):
        result = renderer.render_run_all(pipeline_config)
        ast.parse(result)

    def test_runner_imports_run_namespace(self, renderer, pipeline_config):
        result = renderer.render_runner(pipeline_config)
        assert "from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace" in result

    def test_runner_writes_bronze_metadata(self, renderer, pipeline_config):
        result = renderer.render_runner(pipeline_config)
        assert "bronze_metadata_path" in result

    def test_runner_valid_python(self, renderer, pipeline_config):
        result = renderer.render_runner(pipeline_config)
        ast.parse(result)


class TestDatabricksSilverMetadata:
    def test_silver_writes_metadata(self, dbx_renderer, pipeline_config):
        result = dbx_renderer.render_silver(pipeline_config)
        assert "silver_metadata_path" in result

    def test_silver_valid_python(self, dbx_renderer, pipeline_config):
        result = dbx_renderer.render_silver(pipeline_config)
        ast.parse(result)


class TestDatabricksGoldMetadata:
    def test_gold_writes_metadata(self, dbx_renderer, pipeline_config):
        result = dbx_renderer.render_gold(pipeline_config)
        assert "gold_metadata_path" in result

    def test_gold_valid_python(self, dbx_renderer, pipeline_config):
        result = dbx_renderer.render_gold(pipeline_config)
        ast.parse(result)

    def test_gold_meta_cols_uses_config_variables(self, dbx_renderer, pipeline_config):
        result = dbx_renderer.render_gold(pipeline_config)
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Set) and any(
                isinstance(e, ast.Name) and e.id == "TARGET_COLUMN" for e in node.elts
            ):
                names = [e.id for e in node.elts if isinstance(e, ast.Name)]
                assert "TARGET" not in names, "_meta_cols must use TARGET_COLUMN, not TARGET"
                assert "TARGET_COLUMN" in names
                return
        pytest.fail("_meta_cols set with TARGET_COLUMN not found in gold template")


class TestDatabricksRunnerBronzeMetadata:
    def test_runner_writes_bronze_metadata(self, dbx_renderer, pipeline_config):
        result = dbx_renderer.render_runner(pipeline_config)
        assert "bronze_metadata_path" in result

    def test_runner_captures_bronze_results(self, dbx_renderer, pipeline_config):
        result = dbx_renderer.render_runner(pipeline_config)
        assert "_bronze_results" in result

    def test_runner_valid_python(self, dbx_renderer, pipeline_config):
        result = dbx_renderer.render_runner(pipeline_config)
        ast.parse(result)
