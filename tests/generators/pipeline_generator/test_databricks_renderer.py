import ast

import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
    render_spark_step_call,
    spark_provenance_block,
)
from customer_retention.generators.pipeline_generator.models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    BronzeLayerConfig,
    GoldLayerConfig,
    LifecycleConfig,
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
def sample_pipeline_config():
    source1 = SourceConfig(
        name="customers",
        path="/data/customers.csv",
        format="csv",
        entity_key="customer_id",
    )
    source2 = SourceConfig(
        name="orders",
        path="/data/orders.parquet",
        format="parquet",
        entity_key="customer_id",
        time_column="order_date",
        is_event_level=True,
    )
    bronze1 = BronzeLayerConfig(
        source=source1,
        transformations=[
            TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column="age",
                parameters={"value": 0},
                rationale="Fill nulls",
            ),
        ],
    )
    silver = SilverLayerConfig(
        joins=[
            {
                "left_key": "customer_id",
                "right_key": "customer_id",
                "right_source": "orders",
                "how": "left",
            }
        ],
        aggregations=[],
        derived_columns=[
            TransformationStep(
                type=PipelineTransformationType.DERIVED_COLUMN,
                column="avg_order_value",
                parameters={"action": "ratio", "numerator": "total_amount", "denominator": "order_count"},
                rationale="Average order value",
            ),
        ],
    )
    gold = GoldLayerConfig(
        encodings=[
            TransformationStep(
                type=PipelineTransformationType.ENCODE,
                column="category",
                parameters={"method": "one_hot"},
                rationale="Encode",
            ),
        ],
        scalings=[
            TransformationStep(
                type=PipelineTransformationType.SCALE,
                column="amount",
                parameters={"method": "standard"},
                rationale="Scale",
            ),
        ],
        transformations=[
            TransformationStep(
                type=PipelineTransformationType.LOG_TRANSFORM,
                column="revenue",
                parameters={},
                rationale="Log transform skewed",
            ),
        ],
        feature_selections=["unused_col"],
    )
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[source1, source2],
        bronze={"customers": bronze1},
        bronze_event={
            "orders": BronzeEventConfig(
                source=source2,
                entity_column="customer_id",
                time_column="order_date",
                pre_shaping=[
                    TransformationStep(
                        type=PipelineTransformationType.CAP_OUTLIER,
                        column="amount",
                        parameters={"lower": 0, "upper": 10000},
                        rationale="Cap outliers",
                    ),
                ],
                aggregation=AggregationWindowConfig(
                    windows=["7d", "30d", "90d"],
                    value_columns=["amount"],
                    agg_funcs=["sum", "mean", "count"],
                ),
            ),
        },
        silver=silver,
        gold=gold,
        output_dir="/output/test_pipeline",
        composite_name="cust_orde_aggr__abc1234",
    )


class TestDatabricksCodeRendererInit:
    def test_renderer_creates_jinja_environment(self, renderer):
        assert renderer._env is not None

    def test_renderer_stores_catalog(self, renderer):
        assert renderer._catalog == "ml_catalog"

    def test_renderer_stores_schema(self, renderer):
        assert renderer._schema == "retention"

    def test_renderer_defaults_catalog_and_schema(self):
        r = DatabricksCodeRenderer()
        assert r._catalog == "main"
        assert r._schema == "default"


class TestRenderSparkStepCall:
    def test_impute_null(self):
        step = TransformationStep(
            type=PipelineTransformationType.IMPUTE_NULL,
            column="age",
            parameters={"value": 0},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "fillna" in result
        assert "age" in result

    def test_cap_outlier(self):
        step = TransformationStep(
            type=PipelineTransformationType.CAP_OUTLIER,
            column="amount",
            parameters={"lower": 0, "upper": 10000},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "F.when" in result
        assert "amount" in result
        assert "10000" in result

    def test_drop_column(self):
        step = TransformationStep(
            type=PipelineTransformationType.DROP_COLUMN,
            column="junk",
            parameters={},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "drop" in result
        assert "junk" in result

    def test_winsorize(self):
        step = TransformationStep(
            type=PipelineTransformationType.WINSORIZE,
            column="revenue",
            parameters={"lower_bound": 100, "upper_bound": 50000},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "F.when" in result
        assert "revenue" in result

    def test_log_transform(self):
        step = TransformationStep(
            type=PipelineTransformationType.LOG_TRANSFORM,
            column="revenue",
            parameters={},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "F.log1p" in result
        assert "revenue" in result

    def test_sqrt_transform(self):
        step = TransformationStep(
            type=PipelineTransformationType.SQRT_TRANSFORM,
            column="count",
            parameters={},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "F.sqrt" in result
        assert "count" in result

    def test_encode_one_hot(self):
        step = TransformationStep(
            type=PipelineTransformationType.ENCODE,
            column="category",
            parameters={"method": "one_hot"},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "encode_one_hot" in result
        assert "category" in result

    def test_scale_standard(self):
        step = TransformationStep(
            type=PipelineTransformationType.SCALE,
            column="amount",
            parameters={"method": "standard"},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "scale_standard" in result
        assert "amount" in result

    def test_feature_select(self):
        step = TransformationStep(
            type=PipelineTransformationType.FEATURE_SELECT,
            column="bad_col",
            parameters={},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "drop" in result
        assert "bad_col" in result

    def test_derived_column_ratio(self):
        step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="avg_val",
            parameters={"action": "ratio", "numerator": "total", "denominator": "count"},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "total" in result
        assert "count" in result
        assert "/" in result or "div" in result.lower()

    def test_derived_column_interaction(self):
        step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="combo",
            parameters={"action": "interaction", "features": ["a", "b"]},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "a" in result
        assert "b" in result
        assert "*" in result

    def test_segment_aware_cap(self):
        step = TransformationStep(
            type=PipelineTransformationType.SEGMENT_AWARE_CAP,
            column="revenue",
            parameters={"n_segments": 2},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "revenue" in result

    def test_zero_inflation_handling(self):
        step = TransformationStep(
            type=PipelineTransformationType.ZERO_INFLATION_HANDLING,
            column="transactions",
            parameters={},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "transactions" in result

    def test_cap_then_log(self):
        step = TransformationStep(
            type=PipelineTransformationType.CAP_THEN_LOG,
            column="revenue",
            parameters={},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "revenue" in result
        assert "F.log1p" in result


class TestDatabricksRenderConfig:
    def test_render_config_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_config_includes_catalog(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "ml_catalog" in result

    def test_render_config_includes_schema(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "retention" in result

    def test_render_config_includes_composite_name(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "cust_orde_aggr__abc1234" in result

    def test_render_config_includes_unity_catalog_table_names(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "bronze_entity_customers" in result or "bronze_entity" in result
        assert "silver_featureset" in result or "silver" in result
        assert "gold_features" in result or "gold" in result

    def test_render_config_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksConfigRunPath:
    def test_bronze_entity_uses_parent_config_path(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "%run ../config" in result
        assert "%run ./config" not in result

    def test_bronze_event_uses_parent_config_path(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze_event("orders", sample_pipeline_config.bronze_event["orders"])
        assert "%run ../config" in result
        assert "%run ./config" not in result

    def test_bronze_entity_aggregated_uses_parent_config_path(self, renderer, sample_pipeline_config):
        event_config = sample_pipeline_config.bronze_event["orders"]
        result = renderer.render_bronze_entity("orders_aggregated", event_config, "orders", "orders")
        assert "%run ../config" in result
        assert "%run ./config" not in result

    def test_silver_uses_parent_config_path(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "%run ../config" in result
        assert "%run ./config" not in result

    def test_gold_uses_parent_config_path(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "%run ../config" in result
        assert "%run ./config" not in result

    def test_training_uses_parent_config_path(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "%run ../config" in result
        assert "%run ./config" not in result


class TestDatabricksRenderBronze:
    def test_render_bronze_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert isinstance(result, str)

    def test_render_bronze_uses_spark(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "spark" in result

    def test_render_bronze_uses_delta_format(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "delta" in result.lower() or "saveAsTable" in result or "save_as_table" in result

    def test_render_bronze_no_parquet(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert 'format("parquet")' not in result

    def test_render_bronze_includes_transformations(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "fillna" in result or "impute" in result.lower()
        assert "age" in result

    def test_render_bronze_includes_source_name(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "customers" in result

    def test_render_bronze_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        ast.parse(result)

    def test_render_bronze_has_notebook_header(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "Databricks notebook source" in result


class TestDatabricksRenderBronzeEvent:
    def test_render_bronze_event_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze_event("orders", sample_pipeline_config.bronze_event["orders"])
        assert isinstance(result, str)

    def test_render_bronze_event_includes_pyspark_aggregation(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze_event("orders", sample_pipeline_config.bronze_event["orders"])
        assert "groupBy" in result or "groupby" in result.lower() or "agg(" in result

    def test_render_bronze_event_includes_pre_shaping(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze_event("orders", sample_pipeline_config.bronze_event["orders"])
        assert "amount" in result

    def test_render_bronze_event_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze_event("orders", sample_pipeline_config.bronze_event["orders"])
        ast.parse(result)

    def test_render_bronze_event_uses_delta(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze_event("orders", sample_pipeline_config.bronze_event["orders"])
        assert "delta" in result.lower() or "saveAsTable" in result or "save_as_table" in result

    def test_render_bronze_event_no_parquet(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze_event("orders", sample_pipeline_config.bronze_event["orders"])
        assert 'format("parquet")' not in result


class TestDatabricksRenderBronzeEntity:
    def test_render_bronze_entity_returns_string(self, renderer, sample_pipeline_config):
        event_config = sample_pipeline_config.bronze_event["orders"]
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            event_config,
            "orders_aggregated",
            "orders",
        )
        assert isinstance(result, str)

    def test_render_bronze_entity_is_valid_python(self, renderer, sample_pipeline_config):
        event_config = sample_pipeline_config.bronze_event["orders"]
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            event_config,
            "orders_aggregated",
            "orders",
        )
        ast.parse(result)

    def test_render_bronze_entity_uses_delta(self, renderer, sample_pipeline_config):
        event_config = sample_pipeline_config.bronze_event["orders"]
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            event_config,
            "orders_aggregated",
            "orders",
        )
        assert "delta" in result.lower() or "saveAsTable" in result or "save_as_table" in result

    def test_render_bronze_entity_reads_from_source_events_table(self, renderer, sample_pipeline_config):
        event_config = sample_pipeline_config.bronze_event["orders"]
        result = renderer.render_bronze_entity("orders_aggregated", event_config, "orders", "orders")
        assert 'bronze_table("orders_events")' in result
        assert "orders_aggregated_events" not in result

    def test_render_bronze_entity_with_lifecycle(self, renderer):
        source = SourceConfig(
            name="orders",
            path="/data/orders.csv",
            format="csv",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            lifecycle=LifecycleConfig(
                include_recency_bucket=True,
                include_lifecycle_quadrant=True,
            ),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders_aggregated",
            "orders",
        )
        assert "lifecycle" in result.lower() or "recency" in result.lower()


class TestDatabricksRenderSilver:
    def test_render_silver_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_silver_includes_join(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "join" in result.lower()

    def test_render_silver_includes_derived_columns(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "avg_order_value" in result

    def test_render_silver_includes_composite_name(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "cust_orde_aggr__abc1234" in result or "silver_featureset" in result

    def test_render_silver_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        ast.parse(result)

    def test_render_silver_uses_delta(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "delta" in result.lower() or "saveAsTable" in result or "save_as_table" in result

    def test_render_silver_no_parquet(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert 'format("parquet")' not in result


class TestDatabricksRenderGold:
    def test_render_gold_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_gold_includes_encodings(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "category" in result

    def test_render_gold_includes_scalings(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "amount" in result

    def test_render_gold_includes_transformations(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "log" in result.lower() or "revenue" in result

    def test_render_gold_includes_feature_selection(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "unused_col" in result

    def test_render_gold_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)

    def test_render_gold_uses_delta(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "delta" in result.lower() or "saveAsTable" in result or "save_as_table" in result

    def test_render_gold_no_parquet(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert 'format("parquet")' not in result


class TestDatabricksRenderTraining:
    def test_render_training_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_training_includes_mlflow(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "mlflow" in result.lower()

    def test_render_training_includes_pyspark_ml(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "pyspark" in result.lower() or "spark" in result.lower()

    def test_render_training_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)

    def test_render_training_includes_model(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "Classifier" in result or "model" in result.lower()


class TestDatabricksRenderRunner:
    def test_render_runner_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_runner_uses_dbutils_notebook_run(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "dbutils.notebook.run" in result

    def test_render_runner_references_all_bronze_notebooks(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "bronze_entity_customers" in result
        assert "bronze_event_orders" in result

    def test_render_runner_references_silver(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "silver" in result.lower()

    def test_render_runner_references_gold(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "gold" in result.lower()

    def test_render_runner_references_training(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "training" in result.lower()

    def test_render_runner_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        ast.parse(result)

    def test_render_runner_has_execution_order(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        bronze_pos = result.find("bronze")
        silver_pos = result.find("silver")
        gold_pos = result.find("gold")
        assert bronze_pos < silver_pos < gold_pos


class TestDatabricksConfigSourcePaths:
    def test_config_uses_raw_source_path_when_available(self, renderer):
        source = SourceConfig(
            name="emails",
            path="emails.parquet",
            format="parquet",
            entity_key="customer_id",
            raw_source_path="/dbfs/data/emails.parquet",
        )
        config = PipelineConfig(
            name="test",
            target_column="churn",
            sources=[source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(),
            output_dir=".",
            composite_name="emai__abc1234",
        )
        result = renderer.render_config(config)
        assert "/dbfs/data/emails.parquet" in result
        assert '"path": "emails.parquet"' not in result

    def test_config_falls_back_to_path_when_no_raw(self, renderer):
        source = SourceConfig(
            name="emails",
            path="/data/emails.csv",
            format="csv",
            entity_key="customer_id",
        )
        config = PipelineConfig(
            name="test",
            target_column="churn",
            sources=[source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(),
            output_dir=".",
            composite_name="emai__abc1234",
        )
        result = renderer.render_config(config)
        assert "/data/emails.csv" in result


class TestDatabricksLoadSourceFormat:
    def test_bronze_event_load_source_uses_dynamic_format(self, renderer):
        source = SourceConfig(
            name="orders",
            path="/data/orders.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            deduplicate=False,
            aggregation=AggregationWindowConfig(
                windows=["30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "count"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert 'format("delta")' not in result or "format(fmt)" in result

    def test_bronze_load_source_uses_dynamic_format(self, renderer):
        source = SourceConfig(
            name="customers",
            path="/data/customers.parquet",
            format="parquet",
            entity_key="customer_id",
        )
        config = BronzeLayerConfig(source=source)
        result = renderer.render_bronze("customers", config)
        assert 'format("delta")' not in result or "format(fmt)" in result

    def test_bronze_event_load_source_still_handles_csv(self, renderer):
        source = SourceConfig(
            name="orders",
            path="/data/orders.csv",
            format="csv",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            deduplicate=False,
        )
        result = renderer.render_bronze_event("orders", config)
        assert "inferSchema" in result


class TestSparkProvenanceBlock:
    def test_returns_empty_for_empty_steps(self):
        assert spark_provenance_block([]) == ""

    def test_returns_section_for_impute_null(self):
        steps = [
            TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column="age",
                parameters={"value": 0},
                rationale="Fill nulls",
            ),
        ]
        result = spark_provenance_block(steps)
        assert "Quality Assessment" in result
        assert "Missing Value Analysis" in result
        assert "Source:" in result

    def test_uses_source_notebook_when_set(self):
        steps = [
            TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column="age",
                parameters={"value": 0},
                rationale="Fill nulls",
                source_notebook="05_custom_analysis",
            ),
        ]
        result = spark_provenance_block(steps)
        assert "Custom Analysis" in result
        assert "Missing Value Analysis" in result

    def test_deduplicates_same_provenance(self):
        steps = [
            TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column="age",
                parameters={"value": 0},
                rationale="Fill age",
            ),
            TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column="income",
                parameters={"value": 0},
                rationale="Fill income",
            ),
        ]
        result = spark_provenance_block(steps)
        assert result.count("Source:") == 1

    def test_no_html_links(self):
        steps = [
            TransformationStep(
                type=PipelineTransformationType.CAP_OUTLIER,
                column="revenue",
                parameters={"lower": 0, "upper": 10000},
                rationale="Cap outliers",
            ),
        ]
        result = spark_provenance_block(steps)
        assert "<a " not in result
        assert "href" not in result
        assert ".html" not in result

    def test_format_uses_angle_bracket_separator(self):
        steps = [
            TransformationStep(
                type=PipelineTransformationType.LOG_TRANSFORM,
                column="revenue",
                parameters={},
                rationale="Log transform",
            ),
        ]
        result = spark_provenance_block(steps)
        assert ">" in result
        assert "Relationship Analysis > Feature Distributions" in result


class TestBronzeStepGrouping:
    def test_bronze_groups_transformations_into_functions(self, renderer):
        source = SourceConfig(
            name="customers",
            path="/data/customers.csv",
            format="csv",
            entity_key="customer_id",
        )
        config = BronzeLayerConfig(
            source=source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="age",
                    parameters={"value": 0},
                    rationale="Fill age",
                ),
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="income",
                    parameters={"value": 0},
                    rationale="Fill income",
                ),
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="revenue",
                    parameters={"lower": 0, "upper": 10000},
                    rationale="Cap revenue",
                ),
            ],
        )
        result = renderer.render_bronze("customers", config)
        assert "def impute_remaining_nulls(df):" in result
        assert "def cap_outliers(df):" in result
        assert "df = impute_remaining_nulls(df)" in result
        assert "df = cap_outliers(df)" in result

    def test_bronze_grouped_has_provenance_docstring(self, renderer):
        source = SourceConfig(
            name="customers",
            path="/data/customers.csv",
            format="csv",
            entity_key="customer_id",
        )
        config = BronzeLayerConfig(
            source=source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="age",
                    parameters={"value": 0},
                    rationale="Fill age",
                ),
            ],
        )
        result = renderer.render_bronze("customers", config)
        assert "Source:" in result
        assert "Quality Assessment > Missing Value Analysis" in result

    def test_bronze_grouped_is_valid_python(self, renderer):
        source = SourceConfig(
            name="customers",
            path="/data/customers.csv",
            format="csv",
            entity_key="customer_id",
        )
        config = BronzeLayerConfig(
            source=source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="age",
                    parameters={"value": 0},
                    rationale="Fill age",
                ),
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="revenue",
                    parameters={"lower": 0, "upper": 10000},
                    rationale="Cap revenue",
                ),
            ],
        )
        result = renderer.render_bronze("customers", config)
        ast.parse(result)

    def test_bronze_no_transformations_still_valid(self, renderer):
        source = SourceConfig(
            name="customers",
            path="/data/customers.csv",
            format="csv",
            entity_key="customer_id",
        )
        config = BronzeLayerConfig(source=source, transformations=[])
        result = renderer.render_bronze("customers", config)
        ast.parse(result)
        assert "def apply_transformations(df):" in result


class TestBronzeEventStepGrouping:
    def test_bronze_event_groups_pre_shaping(self, renderer):
        source = SourceConfig(
            name="orders",
            path="/data/orders.csv",
            format="csv",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            pre_shaping=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="amount",
                    parameters={"lower": 0, "upper": 10000},
                    rationale="Cap outliers",
                ),
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="quantity",
                    parameters={"value": 1},
                    rationale="Fill nulls",
                ),
            ],
        )
        result = renderer.render_bronze_event("orders", config)
        assert "def cap_outliers(df):" in result
        assert "def impute_remaining_nulls(df):" in result
        assert "df = cap_outliers(df)" in result

    def test_bronze_event_grouped_is_valid_python(self, renderer):
        source = SourceConfig(
            name="orders",
            path="/data/orders.csv",
            format="csv",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            pre_shaping=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="amount",
                    parameters={"lower": 0, "upper": 10000},
                    rationale="Cap outliers",
                ),
            ],
            aggregation=AggregationWindowConfig(
                windows=["30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "count"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        ast.parse(result)

    def test_bronze_event_pre_shaping_has_provenance(self, renderer):
        source = SourceConfig(
            name="orders",
            path="/data/orders.csv",
            format="csv",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            pre_shaping=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="amount",
                    parameters={"lower": 0, "upper": 10000},
                    rationale="Cap outliers",
                ),
            ],
        )
        result = renderer.render_bronze_event("orders", config)
        assert "Source:" in result
        assert "Global Outlier Detection" in result


class TestBronzeEntityStepGrouping:
    def test_bronze_entity_groups_post_shaping(self, renderer):
        source = SourceConfig(
            name="orders",
            path="/data/orders.csv",
            format="csv",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            post_shaping=[
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="total",
                    parameters={"value": 0},
                    rationale="Fill total",
                ),
                TransformationStep(
                    type=PipelineTransformationType.LOG_TRANSFORM,
                    column="total",
                    parameters={},
                    rationale="Log transform total",
                ),
            ],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        assert "def impute_remaining_nulls(df):" in result
        assert "def apply_log_transforms(df):" in result

    def test_bronze_entity_grouped_is_valid_python(self, renderer):
        source = SourceConfig(
            name="orders",
            path="/data/orders.csv",
            format="csv",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            post_shaping=[
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="total",
                    parameters={"value": 0},
                    rationale="Fill total",
                ),
            ],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        ast.parse(result)


class TestSilverStepGrouping:
    def test_silver_groups_derived_columns(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "def create_ratio_features(df):" in result
        assert "df = create_ratio_features(df)" in result

    def test_silver_grouped_has_provenance(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "Source:" in result
        assert "Feature Engineering Recommendations" in result

    def test_silver_grouped_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        ast.parse(result)


class TestGoldStepGrouping:
    def test_gold_groups_transformations(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "def apply_log_transforms(df):" in result
        assert "df = apply_log_transforms(df)" in result

    def test_gold_encodings_have_provenance(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "Categorical Feature Analysis" in result

    def test_gold_scalings_have_provenance(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "Feature-Target Correlations" in result

    def test_gold_transformations_have_provenance(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "Feature Distributions" in result

    def test_gold_grouped_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)

    def test_gold_no_transformations_still_valid(self, renderer):
        source = SourceConfig(
            name="customers",
            path="/data/customers.csv",
            format="csv",
            entity_key="customer_id",
        )
        config = PipelineConfig(
            name="test",
            target_column="churn",
            sources=[source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(),
            output_dir=".",
            composite_name="cust__abc1234",
        )
        result = renderer.render_gold(config)
        ast.parse(result)

    def test_gold_with_source_notebook_override(self, renderer):
        source = SourceConfig(
            name="customers",
            path="/data/customers.csv",
            format="csv",
            entity_key="customer_id",
        )
        config = PipelineConfig(
            name="test",
            target_column="churn",
            sources=[source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(
                encodings=[
                    TransformationStep(
                        type=PipelineTransformationType.ENCODE,
                        column="category",
                        parameters={"method": "one_hot"},
                        rationale="Encode category",
                        source_notebook="06_custom_report",
                    ),
                ],
            ),
            output_dir=".",
            composite_name="cust__abc1234",
        )
        result = renderer.render_gold(config)
        assert "Custom Report" in result
        assert "Categorical Feature Analysis" in result
