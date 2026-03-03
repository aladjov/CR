import ast

import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import DatabricksCodeRenderer
from customer_retention.generators.pipeline_generator.models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    BronzeLayerConfig,
    GoldLayerConfig,
    LandingLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TransformationStep,
)


@pytest.fixture
def renderer():
    return DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")


@pytest.fixture
def entity_source():
    return SourceConfig(
        name="customers", path="/data/customers.csv", format="csv",
        entity_key="customer_id",
    )


@pytest.fixture
def event_source():
    return SourceConfig(
        name="orders", path="/data/orders.parquet", format="parquet",
        entity_key="customer_id", time_column="order_date", is_event_level=True,
    )


@pytest.fixture
def landing_config():
    source = SourceConfig(
        name="orders", path="/data/orders.parquet", format="parquet",
        entity_key="customer_id", time_column="order_date", is_event_level=True,
    )
    return LandingLayerConfig(
        source=source,
        raw_source_path="/data/orders.parquet",
        raw_source_format="parquet",
        entity_column="customer_id",
        time_column="order_date",
        target_column="churn",
    )


@pytest.fixture
def bronze_event_config(event_source):
    return BronzeEventConfig(
        source=event_source,
        entity_column="customer_id",
        time_column="order_date",
        pre_shaping=[],
        aggregation=AggregationWindowConfig(
            windows=["7d", "30d"],
            value_columns=["amount"],
            agg_funcs=["sum", "mean"],
            reference_date="max",
        ),
    )


@pytest.fixture
def sample_pipeline_config(entity_source, event_source):
    bronze = BronzeLayerConfig(
        source=entity_source,
        transformations=[
            TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column="age", parameters={"value": 0}, rationale="Fill nulls",
            ),
        ],
    )
    silver = SilverLayerConfig(
        joins=[{"left_keys": ["customer_id"], "right_keys": ["customer_id"], "right_source": "orders", "how": "left"}],
        aggregations=[],
    )
    gold = GoldLayerConfig(
        encodings=[],
        scalings=[],
    )
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[entity_source, event_source],
        bronze={"customers": bronze},
        bronze_event={
            "orders": BronzeEventConfig(
                source=event_source,
                entity_column="customer_id",
                time_column="order_date",
                pre_shaping=[],
                aggregation=AggregationWindowConfig(
                    windows=["7d", "30d"],
                    value_columns=["amount"],
                    agg_funcs=["sum", "mean"],
                    reference_date="max",
                ),
            ),
        },
        silver=silver,
        gold=gold,
        output_dir="/output/test_pipeline",
    )


class TestDatabricksLandingOptimize:
    def test_landing_contains_optimize(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "optimize()" in result

    def test_landing_z_orders_entity_and_time(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "executeZOrderBy" in result

    def test_landing_optimize_is_valid_python(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        ast.parse(result)


class TestDatabricksBronzeEventOptimize:
    def test_bronze_event_contains_optimize(self, renderer, bronze_event_config):
        result = renderer.render_bronze_event("orders", bronze_event_config)
        assert "optimize()" in result

    def test_bronze_event_z_orders_entity_and_date(self, renderer, bronze_event_config):
        result = renderer.render_bronze_event("orders", bronze_event_config)
        assert "executeZOrderBy" in result

    def test_bronze_event_optimize_is_valid_python(self, renderer, bronze_event_config):
        result = renderer.render_bronze_event("orders", bronze_event_config)
        ast.parse(result)


class TestDatabricksBronzeEntityOptimize:
    def test_bronze_entity_contains_optimize(self, renderer, bronze_event_config):
        result = renderer.render_bronze_entity(
            "orders_aggregated", bronze_event_config, "orders", "orders",
        )
        assert "optimize()" in result

    def test_bronze_entity_z_orders_entity(self, renderer, bronze_event_config):
        result = renderer.render_bronze_entity(
            "orders_aggregated", bronze_event_config, "orders", "orders",
        )
        assert "executeZOrderBy" in result

    def test_bronze_entity_optimize_is_valid_python(self, renderer, bronze_event_config):
        result = renderer.render_bronze_entity(
            "orders_aggregated", bronze_event_config, "orders", "orders",
        )
        ast.parse(result)


class TestDatabricksSilverOptimize:
    def test_silver_contains_optimize(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "optimize()" in result

    def test_silver_z_orders_entity_id_and_date(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "executeZOrderBy" in result
        assert "entity_id" in result

    def test_silver_optimize_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksGoldOptimize:
    def test_gold_contains_optimize(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "optimize()" in result

    def test_gold_z_orders_entity_id(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "executeZOrderBy" in result
        assert "entity_id" in result

    def test_gold_optimize_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)
