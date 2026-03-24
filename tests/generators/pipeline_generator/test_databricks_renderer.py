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
    DatetimeDerivationConfig,
    GoldLayerConfig,
    HistoryWindowConfig,
    LabelTimestampConfig,
    LandingLayerConfig,
    LifecycleConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TextFeatureConfig,
    TimestampCoalesceConfig,
    TransformationStep,
)


@pytest.fixture
def renderer():
    return DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")


@pytest.fixture
def sample_pipeline_config(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
    silver = SilverLayerConfig(
        joins=silver_with_join.joins,
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
        encodings=gold_with_encode_scale.encodings,
        scalings=gold_with_encode_scale.scalings,
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
        sources=[entity_source, event_source],
        bronze={"customers": bronze_with_impute},
        bronze_event={
            "orders": BronzeEventConfig(
                source=event_source,
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

    def test_derived_ratio_no_runtime_guard(self):
        step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="click_to_open_rate",
            parameters={"action": "ratio", "numerator": "clicked_velocity_pct", "denominator": "opened_velocity_pct"},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "in df.columns" not in result
        assert "clicked_velocity_pct" in result
        assert "opened_velocity_pct" in result

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

    def test_derived_interaction_no_runtime_guard(self):
        step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="combo",
            parameters={"action": "interaction", "features": ["feat_a", "feat_b"]},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "in df.columns" not in result

    def test_derived_composite_no_runtime_guard(self):
        step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="avg_score",
            parameters={"action": "composite", "columns": ["x", "y", "z"]},
            rationale="",
        )
        result = render_spark_step_call(step)
        assert "in df.columns" not in result

    def test_derived_composite_empty_columns_raises(self):
        step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="avg_score",
            parameters={"action": "composite", "columns": []},
            rationale="",
        )
        with pytest.raises(ValueError, match="columns"):
            render_spark_step_call(step)

    def test_derived_composite_missing_columns_key_raises(self):
        step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="avg_score",
            parameters={"action": "composite"},
            rationale="",
        )
        with pytest.raises(ValueError, match="columns"):
            render_spark_step_call(step)

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
        assert result == '_cap_then_log(df, "revenue")'


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

    def test_render_config_includes_fit_mode(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "FIT_MODE" in result

    def test_render_config_fit_mode_true_by_default(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "FIT_MODE = True" in result

    def test_render_config_fit_mode_false(self, renderer, sample_pipeline_config):
        sample_pipeline_config.fit_mode = False
        result = renderer.render_config(sample_pipeline_config)
        assert "FIT_MODE = False" in result

    def test_render_config_includes_recommendations_hash(self, renderer, sample_pipeline_config):
        sample_pipeline_config.recommendations_hash = "abc123"
        result = renderer.render_config(sample_pipeline_config)
        assert 'RECOMMENDATIONS_HASH = "abc123"' in result

    def test_render_config_recommendations_hash_none(self, renderer, sample_pipeline_config):
        sample_pipeline_config.recommendations_hash = None
        result = renderer.render_config(sample_pipeline_config)
        assert "RECOMMENDATIONS_HASH = None" in result

    def test_render_config_includes_entity_key(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "ENTITY_KEY" in result

    def test_render_config_entity_key_defaults_to_entity_id(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert 'ENTITY_KEY = "entity_id"' in result


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

    def test_render_bronze_event_uses_time_column_without_redundant_rename(self, renderer):
        source = SourceConfig(
            name="emails",
            path="/data/emails.csv",
            format="csv",
            entity_key="customer_id",
            time_column="feature_timestamp",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="feature_timestamp",
            deduplicate=True,
            raw_time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["180d", "365d"],
                value_columns=["bounced"],
                agg_funcs=["sum", "count"],
            ),
        )
        result = renderer.render_bronze_event("emails", config)
        assert 'withColumnRenamed("sent_date"' not in result
        assert 'TIME_COLUMN = "feature_timestamp"' in result
        ast.parse(result)

    def test_render_bronze_event_no_rename_when_raw_matches(self, renderer):
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
            deduplicate=True,
        )
        result = renderer.render_bronze_event("orders", config)
        assert "withColumnRenamed" not in result


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


class TestDatabricksSilverHoldout:
    def test_silver_includes_holdout_mask(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "create_holdout_mask" in result

    def test_silver_holdout_creates_original_column(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "original_" in result
        assert "TARGET_COLUMN" in result

    def test_silver_holdout_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        ast.parse(result)

    def test_silver_holdout_samples_entities_not_rows(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert 'df.select("entity_id").distinct().sample(' in result
        assert "holdout_df = df.sample(" not in result


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

    def test_render_silver_loads_native_spark_dataframes(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert ".pandas_api()" not in result

    def test_simplified_merge_renames_entity_key_to_entity_id(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "SparkTemporalMerger" not in result
        assert 'raw_entity_key' in result
        assert 'withColumnRenamed(raw_entity_key, "entity_id")' in result

    def test_simplified_merge_skips_rename_when_already_entity_id(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            SilverLayerConfig,
            SourceConfig,
        )
        source = SourceConfig(name="data", path="/data.csv", format="csv", entity_key="entity_id")
        silver = SilverLayerConfig(joins=[], aggregations=[])
        config = PipelineConfig(
            name="test", target_column="churn", sources=[source],
            bronze={}, silver=silver, gold=None, output_dir="/out",
        )
        result = renderer.render_silver(config)
        assert 'raw_entity_key != "entity_id"' in result


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

    def test_log_model_and_log_metrics_present(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "mlflow.spark.log_model" in result
        assert "mlflow.log_metrics" in result or "mlflow.log_metric" in result

    def test_best_auc_initialized_below_zero(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "best_auc = -1" in result

    def test_training_passes_target_and_timestamp(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "TIMESTAMP_COLUMN" in result
        assert "TARGET" in result

    def test_training_temporal_split_uses_timestamp_column(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "TIMESTAMP_COLUMN" in result
        assert "DataSplitter" not in result


class TestDatabricksTrainingFeatureEngineering:
    def test_training_uses_fe_log_model_for_best(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "fe.log_model" in result

    def test_training_creates_feature_lookup(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model"):]
        assert "FeatureLookup" in fn
        assert "gold_table()" in fn

    def test_training_creates_training_set(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model"):]
        assert "create_training_set" in fn

    def test_training_registers_model_in_uc(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "registered_model_name" in result
        assert "CATALOG" in result.split("registered_model_name")[1][:50]
        assert "SCHEMA" in result.split("registered_model_name")[1][:50]

    def test_training_fe_has_import_fallback(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model"):]
        assert "except ImportError" in fn

    def test_training_fallback_uses_mlflow_spark(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model"):]
        assert "mlflow.spark.log_model" in fn

    def test_training_nested_runs_still_use_mlflow_spark(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("for name, model in models"):]
        assert "mlflow.spark.log_model(fitted" in fn

    def test_training_log_best_model_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksRenderTrainingImbalance:
    def _make_config_with_imbalance(self, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale, strategy):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        silver = SilverLayerConfig(joins=silver_with_join.joins, aggregations=[])
        gold = GoldLayerConfig(
            encodings=gold_with_encode_scale.encodings,
            scalings=gold_with_encode_scale.scalings,
        )
        config = PipelineConfig(
            name="test_pipeline",
            target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute},
            bronze_event={},
            silver=silver,
            gold=gold,
            output_dir="/output",
            composite_name="test__abc1234",
            training=TrainingConfig(imbalance_strategy=strategy, imbalance_ratio=5.0),
        )
        return config

    def test_training_class_weight_adds_weight_col(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        config = self._make_config_with_imbalance(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale, "class_weight")
        result = renderer.render_training(config)
        assert "weightCol" in result or "weight" in result.lower()
        assert "class_weight" in result.lower() or "balanced" in result.lower() or "weight_col" in result

    def test_training_smote_adds_resampling(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        config = self._make_config_with_imbalance(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale, "smote")
        result = renderer.render_training(config)
        assert "SMOTE" in result or "smote" in result.lower()

    def test_training_imbalance_is_valid_python(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        for strategy in ("class_weight", "smote"):
            config = self._make_config_with_imbalance(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale, strategy)
            result = renderer.render_training(config)
            ast.parse(result)

    def test_training_no_imbalance_when_no_config(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "weightCol" not in result
        assert "SMOTE" not in result


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

    def test_render_runner_exits_with_log(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "dbutils.notebook.exit" in result
        assert "_log" in result

    def test_render_runner_accumulates_results(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "_log.append(line)" in result


class TestDatabricksNotebookExitSummary:
    def test_bronze_exits_with_summary(self, renderer):
        source = SourceConfig(name="t", path="t.csv", format="csv", entity_key="id")
        config = BronzeLayerConfig(source=source, transformations=[])
        result = renderer.render_bronze("t", config)
        assert "dbutils.notebook.exit(_summary)" in result
        assert "result.count()" in result

    def test_bronze_event_exits_with_summary(self, renderer):
        source = SourceConfig(name="ev", path="ev.csv", format="csv", entity_key="id", time_column="ts", is_event_level=True)
        config = BronzeEventConfig(source=source, entity_column="id", time_column="ts")
        result = renderer.render_bronze_event("ev", config)
        assert "dbutils.notebook.exit(_summary)" in result

    def test_bronze_entity_exits_with_summary(self, renderer):
        source = SourceConfig(name="ev", path="ev.csv", format="csv", entity_key="id", time_column="ts", is_event_level=True)
        config = BronzeEventConfig(source=source, entity_column="id", time_column="ts")
        result = renderer.render_bronze_entity("ev_aggregated", config, "ev")
        assert "dbutils.notebook.exit(_summary)" in result

    def test_silver_exits_with_summary(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "dbutils.notebook.exit(_summary)" in result

    def test_gold_exits_with_summary(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "dbutils.notebook.exit(_summary)" in result

    def test_training_exits_with_summary(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "dbutils.notebook.exit(json.dumps(_training_results" in result
        assert "TRAINING RESULTS" in result

    def test_landing_exits_with_summary(self, renderer):
        source = SourceConfig(name="t", path="t.csv", format="csv", entity_key="id")
        config = LandingLayerConfig(source=source, raw_source_path="t.csv", raw_source_format="csv", entity_column="id", time_column="ts", target_column="churn")
        result = renderer.render_landing("t", config)
        assert "dbutils.notebook.exit(_summary)" in result


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
    def test_bronze_event_load_source_reads_from_landing(self, renderer):
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
        assert "landing_table(SOURCE_NAME)" in result

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

    def test_bronze_event_load_source_no_raw_file_read(self, renderer):
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
        assert "landing_table(SOURCE_NAME)" in result
        assert "inferSchema" not in result


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
        assert "Source Integrity" in result
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
        assert "Source Integrity > Missing Value Analysis" in result

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


class TestBronzeEventCategoricalAggregationSpark:
    def test_bronze_event_aggregation_has_numeric_type_guard(self, renderer):
        source = SourceConfig(
            name="emails",
            path="/data/emails.csv",
            format="csv",
            entity_key="customer_id",
            time_column="sent_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["send_hour"],
                agg_funcs=["sum", "mean"],
                categorical_columns=["direction", "status"],
                categorical_agg_funcs=["nunique", "mode"],
            ),
        )
        result = renderer.render_bronze_event("emails", config)
        assert "NumericType" in result
        assert "_get_numeric_columns" in result

    def test_bronze_event_aggregation_has_categorical_spark(self, renderer):
        source = SourceConfig(
            name="emails",
            path="/data/emails.csv",
            format="csv",
            entity_key="customer_id",
            time_column="sent_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d"],
                value_columns=["send_hour"],
                agg_funcs=["sum"],
                categorical_columns=["direction"],
                categorical_agg_funcs=["nunique", "mode"],
            ),
        )
        result = renderer.render_bronze_event("emails", config)
        assert "countDistinct" in result
        assert "CATEGORICAL_COLUMNS" in result
        assert "'direction'" in result

    def test_bronze_event_aggregation_empty_categorical(self, renderer):
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
            aggregation=AggregationWindowConfig(
                windows=["30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
                categorical_columns=[],
                categorical_agg_funcs=["nunique", "mode"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert "CATEGORICAL_COLUMNS = []" in result


class TestDatabricksDatetimeDerivation:

    def test_bronze_event_includes_datetime_derivation(self, renderer):
        from customer_retention.generators.pipeline_generator.models import DatetimeDerivationConfig

        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            aggregation=AggregationWindowConfig(
                windows=["30d"], value_columns=["amount"], agg_funcs=["sum"],
            ),
            datetime_derivation=DatetimeDerivationConfig(
                source_columns=["response_at"],
                reference_column="event_date",
                mask_future_columns=[],
            ),
        )
        code = renderer.render_bronze_event("events", config)
        assert "derive_datetime_features" in code
        assert "DATETIME_DERIVATION_SOURCES" in code
        assert "response_at" in code
        assert "_delta_hours" in code

    def test_bronze_event_mask_future_per_column_in_derivation(self, renderer):
        from customer_retention.generators.pipeline_generator.models import DatetimeDerivationConfig

        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            datetime_derivation=DatetimeDerivationConfig(
                source_columns=["next_date", "contract_end"],
                reference_column="feature_timestamp",
                mask_future_columns=["next_date"],
            ),
        )
        code = renderer.render_bronze_event("events", config)
        assert "MASK_FUTURE_COLUMNS" in code
        assert "mask_set" in code
        assert "if col in mask_set" in code

    def test_bronze_event_omits_derivation_when_none(self, renderer):
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
        )
        code = renderer.render_bronze_event("events", config)
        assert "derive_datetime_features" not in code


class TestDatabricksFilterStep:
    def test_filter_non_negative(self, renderer):
        from customer_retention.generators.pipeline_generator.databricks_renderer import render_spark_step_call
        step = TransformationStep(
            type=PipelineTransformationType.FILTER,
            column="amount",
            parameters={"condition": "non_negative"},
            rationale="Filter negative amounts",
        )
        result = render_spark_step_call(step)
        assert "amount" in result
        assert ">= 0" in result

    def test_filter_range(self, renderer):
        from customer_retention.generators.pipeline_generator.databricks_renderer import render_spark_step_call
        step = TransformationStep(
            type=PipelineTransformationType.FILTER,
            column="age",
            parameters={"condition": "range", "min_value": 0, "max_value": 120},
            rationale="Filter out-of-range ages",
        )
        result = render_spark_step_call(step)
        assert "age" in result
        assert "0" in result
        assert "120" in result

    def test_filter_valid_values(self, renderer):
        from customer_retention.generators.pipeline_generator.databricks_renderer import render_spark_step_call
        step = TransformationStep(
            type=PipelineTransformationType.FILTER,
            column="status",
            parameters={"condition": "valid_values", "valid_values": [0, 1]},
            rationale="Validate binary column",
        )
        result = render_spark_step_call(step)
        assert "status" in result
        assert "isin" in result.lower() or "when" in result.lower()

    def test_filter_in_bronze_event_template(self, renderer):
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            pre_shaping=[
                TransformationStep(
                    type=PipelineTransformationType.FILTER,
                    column="amount",
                    parameters={"condition": "non_negative"},
                    rationale="Filter negative amounts",
                ),
            ],
        )
        code = renderer.render_bronze_event("events", config)
        assert "amount" in code
        assert ">= 0" in code
        ast.parse(code)


class TestDatabricksDeduplication:
    def test_basic_dedup_with_bool_true(self, renderer):
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            deduplicate=True,
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" in code
        assert "row_number" in code

    def test_dedup_keep_first(self, renderer):
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            deduplicate=DeduplicationConfig(strategy="keep_first"),
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" in code
        assert "row_number" in code
        ast.parse(code)

    def test_dedup_keep_most_complete(self, renderer):
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            deduplicate=DeduplicationConfig(strategy="keep_most_complete"),
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" in code
        assert "null_count" in code or "isNull" in code.lower() or "isnull" in code.lower()
        ast.parse(code)

    def test_dedup_with_conflict_columns(self, renderer):
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            deduplicate=DeduplicationConfig(
                strategy="keep_first",
                conflict_columns=["customer_id", "event_date", "amount"],
            ),
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" in code
        assert "amount" in code
        ast.parse(code)

    def test_no_dedup_when_false(self, renderer):
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            deduplicate=False,
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" not in code


class TestDatabricksMomentumRatios:
    def test_bronze_entity_includes_momentum_ratios(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(
                include_recency_bucket=True,
                momentum_pairs=[
                    {"short_window": "7d", "long_window": "30d"},
                    {"short_window": "30d", "long_window": "90d"},
                ],
            ),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated", config, "orders", "orders",
        )
        assert "add_momentum_ratios" in result
        assert "momentum_7d_30d" in result
        assert "momentum_30d_90d" in result
        assert "event_count_7d" in result
        assert "event_count_30d" in result

    def test_bronze_entity_momentum_uses_safe_division(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(
                momentum_pairs=[{"short_window": "7d", "long_window": "30d"}],
            ),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated", config, "orders", "orders",
        )
        assert "F.when" in result
        assert "!= 0" in result

    def test_bronze_entity_no_momentum_without_pairs(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(include_recency_bucket=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated", config, "orders", "orders",
        )
        assert "add_momentum_ratios" not in result

    def test_bronze_standalone_entity_momentum(self, renderer):
        source = SourceConfig(
            name="customers", path="/data/customers.csv", format="csv",
            entity_key="customer_id",
        )
        config = BronzeLayerConfig(
            source=source,
            transformations=[],
            lifecycle=LifecycleConfig(
                include_recency_bucket=True,
                momentum_pairs=[
                    {"short_window": "7d", "long_window": "30d"},
                ],
            ),
            entity_column="customer_id",
            time_column="order_date",
        )
        result = renderer.render_bronze("customers", config)
        assert "add_momentum_ratios" in result
        assert "momentum_7d_30d" in result

    def test_bronze_entity_momentum_is_valid_python(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(
                include_recency_bucket=True,
                momentum_pairs=[
                    {"short_window": "7d", "long_window": "30d"},
                    {"short_window": "30d", "long_window": "90d"},
                ],
            ),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated", config, "orders", "orders",
        )
        ast.parse(result)


class TestDatabricksLandingTemplate:
    @pytest.fixture
    def landing_config(self):
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

    def test_landing_generates_valid_notebook(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "# Databricks notebook source" in result
        ast.parse(result)

    def test_landing_includes_derive_feature_timestamp(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "derive_feature_timestamp" in result

    def test_landing_includes_derive_label_timestamp(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "derive_label_timestamp" in result

    def test_landing_includes_derive_label_available_flag(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "derive_label_available_flag" in result

    def test_landing_reads_from_raw_sources_dict(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "RAW_SOURCES[SOURCE_NAME]" in result

    def test_landing_writes_to_landing_table(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "landing_table(SOURCE_NAME)" in result

    def test_landing_with_timestamp_coalesce(self, renderer, landing_config):
        landing_config.timestamp_coalesce = TimestampCoalesceConfig(
            datetime_columns_ordered=["created_at", "updated_at"],
        )
        result = renderer.render_landing("orders", landing_config)
        assert "created_at" in result
        assert "updated_at" in result
        assert "F.coalesce" in result
        ast.parse(result)

    def test_landing_with_datetime_derivation(self, renderer, landing_config):
        landing_config.datetime_derivation = DatetimeDerivationConfig(
            source_columns=["signup_date"],
            reference_column="order_date",
            mask_future_columns=[],
        )
        result = renderer.render_landing("orders", landing_config)
        assert "derive_datetime_features" in result
        ast.parse(result)

    def test_landing_with_history_window(self, renderer, landing_config):
        landing_config.history_window = HistoryWindowConfig(
            time_column="order_date",
            lookback_periods=52,
            cadence_days=7,
        )
        result = renderer.render_landing("orders", landing_config)
        assert "apply_history_window" in result
        ast.parse(result)

    def test_landing_with_label_timestamp(self, renderer, landing_config):
        landing_config.label_timestamp = LabelTimestampConfig(
            label_column="cancel_date",
            fallback_window_days=90,
        )
        result = renderer.render_landing("orders", landing_config)
        assert "cancel_date" in result
        ast.parse(result)


class TestDatabricksConfigLandingTable:
    def test_config_includes_landing_table_function(self, renderer, sample_pipeline_config):
        sample_pipeline_config.landing = {
            "orders": LandingLayerConfig(
                source=sample_pipeline_config.sources[1],
                raw_source_path="/data/orders.parquet",
                raw_source_format="parquet",
                entity_column="customer_id",
                time_column="order_date",
                target_column="churn",
            ),
        }
        result = renderer.render_config(sample_pipeline_config)
        assert "landing_table" in result
        ast.parse(result)


class TestDatabricksBronzeEventReadsFromLanding:
    def test_bronze_event_reads_from_landing_table(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.parquet", format="parquet",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert "landing_table(SOURCE_NAME)" in result
        ast.parse(result)

    def test_bronze_event_aggregation_preserves_feature_timestamp(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.parquet", format="parquet",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert "feature_timestamp" in result
        assert "F.max" in result
        ast.parse(result)

    def test_bronze_event_aggregation_preserves_target_column(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.parquet", format="parquet",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert "F.first(TARGET_COLUMN" in result
        assert "target_agg" in result
        ast.parse(result)


class TestDatabricksGoldFeatureTimestamp:
    def test_gold_renames_feature_timestamp(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "feature_timestamp" in result
        assert "TIMESTAMP_COLUMN" in result
        ast.parse(result)


class TestDatabricksRunnerSchemaCreation:
    def test_runner_creates_schema_before_notebooks(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "CREATE SCHEMA IF NOT EXISTS" in result
        schema_pos = result.index("CREATE SCHEMA IF NOT EXISTS")
        notebook_pos = result.index("run_notebook")
        assert schema_pos < notebook_pos

    def test_runner_schema_uses_catalog_and_schema(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "{CATALOG}.{SCHEMA}" in result

    def test_runner_runs_config_before_schema(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        config_pos = result.index("%run ./config")
        schema_pos = result.index("CREATE SCHEMA IF NOT EXISTS")
        assert config_pos < schema_pos


class TestDatabricksRunnerLandingStep:
    def test_runner_includes_landing_when_present(self, renderer, sample_pipeline_config):
        sample_pipeline_config.landing = {
            "orders": LandingLayerConfig(
                source=sample_pipeline_config.sources[1],
                raw_source_path="/data/orders.parquet",
                raw_source_format="parquet",
                entity_column="customer_id",
                time_column="order_date",
                target_column="churn",
            ),
        }
        result = renderer.render_runner(sample_pipeline_config)
        assert "landing/landing_orders" in result
        assert result.index("landing") < result.index("bronze")
        ast.parse(result)

    def test_runner_omits_landing_when_empty(self, renderer, sample_pipeline_config):
        sample_pipeline_config.landing = {}
        result = renderer.render_runner(sample_pipeline_config)
        assert "landing/landing_" not in result
        ast.parse(result)


class TestDatabricksLandingFeatureTimestampGuard:
    @pytest.fixture
    def landing_config(self):
        source = SourceConfig(
            name="emails", path="/data/emails.csv", format="csv",
            entity_key="customer_id", time_column="sent_date", is_event_level=True,
        )
        return LandingLayerConfig(
            source=source,
            raw_source_path="/data/emails.csv",
            raw_source_format="csv",
            entity_column="customer_id",
            time_column="sent_date",
            target_column="churn",
        )

    def test_derive_feature_timestamp_checks_column_existence(self, renderer, landing_config):
        result = renderer.render_landing("emails", landing_config)
        assert "if TIME_COLUMN in" in result
        ast.parse(result)

    def test_derive_feature_timestamp_preserves_existing(self, renderer, landing_config):
        result = renderer.render_landing("emails", landing_config)
        assert '"feature_timestamp" not in' in result or "feature_timestamp" in result
        ast.parse(result)

    def test_coalesce_path_unaffected(self, renderer, landing_config):
        landing_config.timestamp_coalesce = TimestampCoalesceConfig(
            datetime_columns_ordered=["created_at", "updated_at"],
        )
        result = renderer.render_landing("emails", landing_config)
        assert "F.coalesce" in result
        assert "if TIME_COLUMN in" not in result
        ast.parse(result)


class TestDatabricksBronzeEventNoRedundantRename:
    def test_bronze_event_no_raw_time_column_rename(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.parquet", format="parquet",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            raw_time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert 'withColumnRenamed("sent_date"' not in result
        ast.parse(result)


class TestDatabricksSilverTemporalMerge:
    @pytest.fixture
    def temporal_config(self, entity_source, event_source, bronze_with_impute, gold_with_encode_scale):
        from customer_retention.generators.pipeline_generator.models import TemporalMergeSourceConfig

        silver = SilverLayerConfig(
            joins=[{
                "left_keys": ["customer_id"],
                "right_keys": ["customer_id"],
                "right_source": "orders",
                "how": "left",
            }],
            grid_dates=["2024-01-01", "2024-01-08", "2024-01-15"],
            entity_key="customer_id",
            merge_sources=[
                TemporalMergeSourceConfig(name="customers", granularity="entity_level"),
                TemporalMergeSourceConfig(name="orders", granularity="event_level",
                                          feature_timestamp_column="order_date"),
            ],
        )
        return PipelineConfig(
            name="test_pipeline",
            target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute},
            bronze_event={
                "orders": BronzeEventConfig(
                    source=event_source,
                    entity_column="customer_id",
                    time_column="order_date",
                    aggregation=AggregationWindowConfig(
                        windows=["7d", "30d"],
                        value_columns=["amount"],
                        agg_funcs=["sum", "mean"],
                    ),
                ),
            },
            silver=silver,
            gold=gold_with_encode_scale,
            output_dir="/output/test_pipeline",
            composite_name="cust_orde__abc1234",
        )

    def test_silver_contains_spark_temporal_merger(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "SparkTemporalMerger" in result

    def test_silver_contains_build_spine(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "build_spine" in result

    def test_silver_contains_grid_dates(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "2024-01-01" in result

    def test_silver_contains_merge_all(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "merge_all" in result

    def test_silver_no_parquet(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert 'format("parquet")' not in result

    def test_silver_falls_back_without_grid_dates(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "SparkTemporalMerger" not in result
        assert "merge_sources" in result or "join" in result.lower()

    @pytest.fixture
    def temporal_config_with_key_resolution(self, entity_source, event_source, bronze_with_impute, gold_with_encode_scale):
        from customer_retention.generators.pipeline_generator.models import (
            KeyResolutionStepConfig,
            TemporalMergeSourceConfig,
        )

        silver = SilverLayerConfig(
            joins=[{
                "left_keys": ["customer_id"],
                "right_keys": ["customer_id"],
                "right_source": "orders",
                "how": "left",
            }],
            grid_dates=["2024-01-01", "2024-01-08"],
            entity_key="customer_id",
            merge_sources=[
                TemporalMergeSourceConfig(name="customers", granularity="entity_level"),
                TemporalMergeSourceConfig(
                    name="case_history",
                    granularity="event_level",
                    feature_timestamp_column="created_date",
                    key_resolution_steps=[
                        KeyResolutionStepConfig(
                            bridge_dataset="case",
                            source_key="CASE_ID",
                            bridge_key="CASE_ID",
                            resolve_column="ACCOUNT_ID",
                        ),
                    ],
                ),
            ],
        )
        return PipelineConfig(
            name="test_pipeline",
            target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute},
            bronze_event={
                "orders": BronzeEventConfig(
                    source=event_source,
                    entity_column="customer_id",
                    time_column="order_date",
                    aggregation=AggregationWindowConfig(
                        windows=["7d", "30d"],
                        value_columns=["amount"],
                        agg_funcs=["sum", "mean"],
                    ),
                ),
            },
            silver=silver,
            gold=gold_with_encode_scale,
            output_dir="/output/test_pipeline",
            composite_name="cust_orde__abc1234",
        )

    def test_silver_key_resolution_join(self, renderer, temporal_config_with_key_resolution):
        result = renderer.render_silver(temporal_config_with_key_resolution)
        assert ".join(" in result
        assert "dropDuplicates" in result

    def test_silver_key_resolution_before_entity_rename(self, renderer, temporal_config_with_key_resolution):
        result = renderer.render_silver(temporal_config_with_key_resolution)
        join_pos = result.index(".join(")
        rename_pos = result.index("withColumnRenamed")
        assert join_pos < rename_pos

    def test_silver_no_key_resolution_when_empty(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "dropDuplicates" not in result or "key_resolution" not in result

    def test_silver_with_key_resolution_valid_python(self, renderer, temporal_config_with_key_resolution):
        result = renderer.render_silver(temporal_config_with_key_resolution)
        ast.parse(result)

    def test_silver_temporal_loads_native_spark_dataframes(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert ".pandas_api()" not in result


class TestDatabricksGoldAsOfDate:
    def test_gold_checks_as_of_date(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "as_of_date" in result

    def test_gold_as_of_date_before_feature_timestamp(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        as_of_pos = result.index("as_of_date")
        feature_ts_pos = result.index("feature_timestamp")
        assert as_of_pos < feature_ts_pos


class TestDatabricksTrainingNullImputation:
    def test_training_fills_nan_before_assembly(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        prepare_fn = result[result.index("def prepare_features"):]
        fillna_pos = prepare_fn.index("fillna(0")
        assembler_pos = prepare_fn.index("VectorAssembler")
        assert fillna_pos < assembler_pos

    def test_training_assembler_uses_error_mode(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'handleInvalid="error"' in result
        assert 'handleInvalid="skip"' not in result

    def test_training_fillna_targets_feature_cols(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "fillna(0, subset=feature_cols)" in result


class TestDatabricksTrainingDropAsOfDate:
    def test_training_excludes_timestamp_and_entity(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "TIMESTAMP_COLUMN" in result
        assert '"entity_id"' in result


class TestDatabricksConfigRawSources:
    def test_config_includes_raw_sources(self, renderer, sample_pipeline_config):
        source = sample_pipeline_config.sources[1]
        sample_pipeline_config.landing = {
            "orders": LandingLayerConfig(
                source=source,
                raw_source_path="/data/orders.parquet",
                raw_source_format="parquet",
                entity_column="customer_id",
                time_column="order_date",
                target_column="churn",
            ),
        }
        result = renderer.render_config(sample_pipeline_config)
        assert "RAW_SOURCES" in result
        assert '"/data/orders.parquet"' in result
        assert '"parquet"' in result
        ast.parse(result)

    def test_config_raw_sources_empty_when_no_landing(self, renderer, sample_pipeline_config):
        sample_pipeline_config.landing = {}
        result = renderer.render_config(sample_pipeline_config)
        assert "RAW_SOURCES" in result
        ast.parse(result)


class TestDatabricksBronzeEntityLifecycleReadsFromLanding:
    def test_recency_tenure_reads_from_landing_table(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(include_recency_bucket=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated", config, "orders", "orders",
        )
        assert 'landing_table("orders")' in result
        assert "add_recency_tenure" in result
        recency_fn = result[result.index("def add_recency_tenure"):]
        recency_fn = recency_fn[:recency_fn.index("\ndef ")]
        assert "landing_table" in recency_fn
        assert "bronze_table" not in recency_fn
        ast.parse(result)

    def test_month_cyclical_reads_from_landing_table(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(include_month_cyclical=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated", config, "orders", "orders",
        )
        assert 'landing_table("orders")' in result
        assert "add_month_quarter_cyclical" in result
        cyclical_fn = result[result.index("def add_month_quarter_cyclical"):]
        cyclical_fn = cyclical_fn[:cyclical_fn.index("\ndef ")]
        assert "landing_table" in cyclical_fn
        assert "bronze_table" not in cyclical_fn
        ast.parse(result)

    def test_cyclical_features_reads_from_landing_table(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(include_cyclical_features=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated", config, "orders", "orders",
        )
        assert 'landing_table("orders")' in result
        assert "add_cyclical_features" in result
        assert "dow_sin" in result
        assert "dow_cos" in result
        cyclical_fn = result[result.index("def add_cyclical_features"):]
        cyclical_fn = cyclical_fn[:cyclical_fn.index("\ndef ")]
        assert "landing_table" in cyclical_fn
        assert "bronze_table" not in cyclical_fn
        ast.parse(result)

    def test_cohort_features_reads_from_landing_table(self, renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(include_cohort_features=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated", config, "orders", "orders",
        )
        assert 'landing_table("orders")' in result
        assert "add_cohort_features" in result
        cohort_fn = result[result.index("def add_cohort_features"):]
        cohort_fn = cohort_fn[:cohort_fn.index("\ndef ")]
        assert "landing_table" in cohort_fn
        assert "bronze_table" not in cohort_fn
        ast.parse(result)


class TestDatabricksSilverToSparkConversion:
    @pytest.fixture
    def temporal_config(self, entity_source, event_source, bronze_with_impute, gold_with_encode_scale):
        from customer_retention.generators.pipeline_generator.models import TemporalMergeSourceConfig

        silver = SilverLayerConfig(
            joins=[{
                "left_keys": ["customer_id"], "right_keys": ["customer_id"],
                "right_source": "orders", "how": "left",
            }],
            grid_dates=["2024-01-01", "2024-01-08"],
            entity_key="customer_id",
            merge_sources=[
                TemporalMergeSourceConfig(name="customers", granularity="entity_level"),
                TemporalMergeSourceConfig(name="orders", granularity="event_level",
                                          feature_timestamp_column="order_date"),
            ],
        )
        return PipelineConfig(
            name="test_pipeline", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute},
            bronze_event={"orders": BronzeEventConfig(
                source=event_source, entity_column="customer_id", time_column="order_date",
                aggregation=AggregationWindowConfig(windows=["7d"], value_columns=["amount"], agg_funcs=["sum"]),
            )},
            silver=silver, gold=gold_with_encode_scale,
            output_dir="/output/test_pipeline", composite_name="cust_orde__abc1234",
        )

    def test_silver_temporal_converts_to_spark_after_merge(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        merge_pos = result.index("merge_all")
        to_spark_pos = result.index("to_spark", merge_pos)
        save_pos = result.index("saveAsTable", to_spark_pos)
        assert merge_pos < to_spark_pos < save_pos

    def test_silver_temporal_to_spark_is_guarded(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert 'hasattr(merged, "to_spark")' in result

    def test_silver_temporal_to_spark_valid_python(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        ast.parse(result)


class TestDatabricksGoldBatchedScaling:
    def test_gold_uses_batch_scale_standard(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "_batch_scale_standard" in result

    def test_gold_batch_scale_standard_function_defined(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "def _batch_scale_standard(df, cols):" in result

    def test_gold_batch_scale_minmax_function_defined(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "def _batch_scale_minmax(df, cols):" in result

    def test_gold_no_per_column_scale_calls_in_apply_scalings(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        apply_scalings_fn = result[result.index("def apply_scalings"):]
        apply_scalings_fn = apply_scalings_fn[:apply_scalings_fn.index("\n\n")]
        assert '_scale_standard(df, "' not in apply_scalings_fn
        assert '_scale_minmax(df, "' not in apply_scalings_fn

    def test_gold_mixed_scaling_methods(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join):
        gold = GoldLayerConfig(
            encodings=[],
            scalings=[
                TransformationStep(
                    type=PipelineTransformationType.SCALE, column="amount",
                    parameters={"method": "standard"}, rationale="Standardize amount",
                ),
                TransformationStep(
                    type=PipelineTransformationType.SCALE, column="revenue",
                    parameters={"method": "standard"}, rationale="Standardize revenue",
                ),
                TransformationStep(
                    type=PipelineTransformationType.SCALE, column="score",
                    parameters={"method": "minmax"}, rationale="MinMax score",
                ),
            ],
        )
        config = PipelineConfig(
            name="test_pipeline", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute},
            silver=SilverLayerConfig(joins=silver_with_join.joins, aggregations=[]),
            gold=gold, output_dir="/output", composite_name="cust_orde__abc1234",
        )
        result = renderer.render_gold(config)
        assert '_batch_scale_standard(df, ["amount", "revenue"])' in result
        assert '_batch_scale_minmax(df, ["score"])' in result
        ast.parse(result)

    def test_gold_batch_scale_single_agg_call(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        batch_fn = result[result.index("def _batch_scale_standard"):]
        batch_fn = batch_fn[:batch_fn.index("\ndef ")]
        assert batch_fn.count(".collect()") == 1

    def test_gold_is_valid_python_with_batched_scaling(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksGoldColumnCheckClean:
    def test_gold_no_list_comprehension_column_check(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "[c for c in df.columns]" not in result

    def test_gold_uses_direct_column_check(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert '"as_of_date" in df.columns' in result
        assert '"feature_timestamp" in df.columns' in result


class TestDatabricksGoldCapThenLog:
    def _make_config(self, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        gold = GoldLayerConfig(
            encodings=gold_with_encode_scale.encodings,
            scalings=gold_with_encode_scale.scalings,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_THEN_LOG,
                    column="revenue",
                    parameters={},
                    rationale="cap at p99 then log",
                ),
            ],
        )
        return PipelineConfig(
            name="test_pipeline", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute},
            silver=SilverLayerConfig(joins=silver_with_join.joins, aggregations=[]),
            gold=gold, output_dir="/output", composite_name="cust_orde__abc1234",
        )

    def test_gold_defines_batch_cap_then_log_helper(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        config = self._make_config(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale)
        result = renderer.render_gold(config)
        assert "def _batch_cap_then_log(df, cols):" in result

    def test_gold_cap_then_log_uses_approx_quantile(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        config = self._make_config(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale)
        result = renderer.render_gold(config)
        assert "approxQuantile" in result

    def test_gold_cap_then_log_applies_log1p(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        config = self._make_config(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale)
        result = renderer.render_gold(config)
        assert "F.log1p" in result

    def test_gold_cap_then_log_is_valid_python(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        config = self._make_config(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale)
        result = renderer.render_gold(config)
        ast.parse(result)

    def test_gold_cap_then_log_no_f_lit_column(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        config = self._make_config(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale)
        result = renderer.render_gold(config)
        assert "F.lit(F.col(" not in result


class TestDatabricksGoldColumnFiltering:
    def test_batch_scale_standard_filters_to_existing_columns(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        batch_fn = result[result.index("def _batch_scale_standard"):]
        batch_fn = batch_fn[:batch_fn.index("\ndef ")]
        assert "c in df.columns" in batch_fn or "c for c in cols if c in" in batch_fn

    def test_batch_scale_minmax_filters_to_existing_columns(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        batch_fn = result[result.index("def _batch_scale_minmax"):]
        batch_fn = batch_fn[:batch_fn.index("\ndef ")]
        assert "c in df.columns" in batch_fn or "c for c in cols if c in" in batch_fn

    def test_encode_one_hot_checks_column_exists(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        encode_fn = result[result.index("def _encode_one_hot"):]
        encode_fn = encode_fn[:encode_fn.index("\ndef ")]
        assert "col not in df.columns" in encode_fn

    def test_label_encode_checks_column_exists(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        encode_fn = result[result.index("def _label_encode"):]
        encode_fn = encode_fn[:encode_fn.index("\ndef ")]
        assert "col not in df.columns" in encode_fn

    def test_batch_scale_standard_still_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksGoldFeatureStoreRegistration:
    def test_gold_casts_timestamp_ntz_before_write(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def run_gold"):]
        cast_pos = fn.index("_cast_timestamp_ntz_to_timestamp")
        save_pos = fn.index("saveAsTable")
        assert cast_pos < save_pos

    def test_gold_cast_function_uses_timestamp_type(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _cast_timestamp_ntz_to_timestamp"):]
        assert "TimestampNTZType" in fn
        assert "TimestampType" in fn
        assert ".cast(TimestampType())" in fn

    def test_gold_no_alter_table_for_type_conversion(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "SET DATA TYPE" not in result

    def test_gold_no_feature_engineering_client_import(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "FeatureEngineeringClient" not in result
        assert "FeatureStoreClient" not in result

    def test_gold_registers_via_sql_pk_constraint(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _register_feature_table"):]
        assert "PRIMARY KEY" in fn
        assert "SET NOT NULL" in fn
        assert "TIMESERIES" in fn

    def test_gold_timeseries_in_pk_clause(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _register_feature_table"):]
        assert "TIMESERIES" in fn
        assert "TIMESTAMP_COLUMN" in fn

    def test_gold_timestamp_in_primary_keys(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _register_feature_table"):]
        assert 'pk = ["entity_id", TIMESTAMP_COLUMN] if has_ts' in fn

    def test_gold_registration_after_save(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        run_gold_fn = result[result.index("def run_gold"):]
        save_pos = run_gold_fn.index("saveAsTable")
        reg_pos = run_gold_fn.index("_register_feature_table")
        assert save_pos < reg_pos

    def test_gold_registration_handles_existing_table(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "already exists" in result
        assert "raise" in result
        lines = result.splitlines()
        bare_excepts = [line.strip() for line in lines if line.strip() == "except:"]
        assert len(bare_excepts) == 0

    def test_gold_registration_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)

    def test_gold_not_null_before_pk_constraint(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _register_feature_table"):]
        not_null_pos = fn.index("SET NOT NULL")
        pk_pos = fn.index("PRIMARY KEY")
        assert not_null_pos < pk_pos

    def test_gold_pk_constraint_handles_existing(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _register_feature_table"):]
        assert "already exists" in fn

    def test_gold_no_rdd_access(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert ".rdd" not in result

    def test_gold_reads_back_from_delta_after_save(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def run_gold"):]
        assert "del df" in fn
        assert "saved = spark.table(output_table)" in fn
        assert "return saved" in fn

    def test_gold_no_triple_materialization(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def run_gold"):]
        after_save = fn[fn.index("saveAsTable"):]
        lines_with_df_ref = [
            line.strip() for line in after_save.splitlines()
            if "df" in line and "del df" not in line
            and "reg_df" not in line and "saved" not in line
            and line.strip() and not line.strip().startswith("#")
            and "df.schema" not in line
        ]
        assert not lines_with_df_ref, f"Stale df references after saveAsTable: {lines_with_df_ref}"

    def test_gold_display_is_bounded(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "display(result.limit(" in result

    def test_gold_one_hot_has_cardinality_guard(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _encode_one_hot"):]
        assert "max_categories" in fn
        assert "_label_encode" in fn


class TestDatabricksTextFeatureFitMode:
    def test_bronze_event_text_features_check_fit_mode(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum"],
            ),
            text_features=[TextFeatureConfig(column="notes")],
        )
        result = renderer.render_bronze_event("orders", config)
        assert "FIT_MODE" in result
        assert "fit=True" in result
        assert "fit=False" in result

    def test_bronze_entity_text_features_check_fit_mode(self, renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            text_features=[TextFeatureConfig(column="bio")],
        )
        result = renderer.render_bronze("customers", config)
        assert "FIT_MODE" in result
        assert "fit=True" in result
        assert "fit=False" in result

    def test_bronze_event_text_features_valid_python(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum"],
            ),
            text_features=[TextFeatureConfig(column="notes")],
        )
        result = renderer.render_bronze_event("orders", config)
        ast.parse(result)

    def test_bronze_entity_text_features_valid_python(self, renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            text_features=[TextFeatureConfig(column="bio")],
        )
        result = renderer.render_bronze("customers", config)
        ast.parse(result)


class TestDatabricksTrainingNullLabelFilter:
    def test_training_filters_null_labels_before_prepare(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        train_fn = result[result.index("def train_and_evaluate"):]
        filter_pos = train_fn.index("isNotNull")
        prepare_pos = train_fn.index("prepare_features(df)")
        assert filter_pos < prepare_pos

    def test_training_null_filter_targets_label_column(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "F.col(TARGET).isNotNull()" in result or "col(TARGET).isNotNull()" in result

    def test_training_null_filter_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksTrainingVectorSchema:
    def test_training_imports_vector_udt(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "VectorUDT" in result

    def test_training_smote_uses_explicit_schema(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        silver = SilverLayerConfig(joins=silver_with_join.joins, aggregations=[])
        gold = GoldLayerConfig(encodings=gold_with_encode_scale.encodings, scalings=gold_with_encode_scale.scalings)
        config = PipelineConfig(
            name="test_pipeline", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute}, bronze_event={},
            silver=silver, gold=gold, output_dir="/output",
            composite_name="test__abc1234",
            training=TrainingConfig(imbalance_strategy="smote"),
        )
        result = renderer.render_training(config)
        smote_section = result[result.index("SMOTE"):]
        assert "createDataFrame(resampled_pdf, schema=" in smote_section or "createDataFrame(resampled_pdf, _vector_schema" in smote_section
        ast.parse(result)


class TestDatabricksTrainingInstrumentation:
    def test_training_imports_timing(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "from customer_retention.core.compat.timing import log_timing" in result

    def test_training_has_assert_rows(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "def _assert_rows" in result

    def test_training_asserts_after_load(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        load_pos = fn.index("load_training_data()")
        assert_pos = fn.index("_assert_rows(")
        assert load_pos < assert_pos

    def test_training_asserts_after_null_filter(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        filter_pos = fn.index("isNotNull")
        remaining = fn[filter_pos:]
        assert "_assert_rows(" in remaining

    def test_training_logs_label_distribution(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'groupBy("label")' in result
        assert "Label distribution" in result

    def test_training_no_pandas_roundtrip_for_splitting(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert "createDataFrame(train_pdf" not in fn
        assert "createDataFrame(test_pdf" not in fn

    def test_training_per_model_timing(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "log_timing" in result
        fn = result[result.index("for name, model in"):]
        assert "log_timing" in fn

    def test_training_timing_around_load(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'log_timing("load_gold_table"' in result

    def test_training_timing_around_split(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'log_timing("temporal_split"' in result

    def test_training_timing_around_prepare(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'log_timing("prepare_features"' in result

    def test_training_logs_split_info(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "split_info" in result

    def test_training_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksTrainingFeatureTypeParity:
    def test_training_includes_boolean_in_feature_types(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '"boolean"' in result

    def test_training_includes_byte_in_feature_types(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '"byte"' in result

    def test_training_excludes_temporal_metadata_columns(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '"as_of_date"' in result
        assert '"feature_timestamp"' in result

    def test_training_checks_minimum_two_classes(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "Need at least 2" in result

    def test_training_feature_types_used_in_prepare_features(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        prepare_fn = result[result.index("def prepare_features"):]
        assert "_NUMERIC_TYPES" in prepare_fn
        assert "_EXCLUDE_COLS" in prepare_fn


class TestDatabricksTrainingFeatureProfile:
    def test_training_imports_feature_profile(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "from customer_retention.stages.modeling.feature_profile import" in result
        assert "FeatureProfile" in result
        assert "build_feature_profile" in result
        assert "compare_feature_profiles" in result

    def test_training_computes_production_profile(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert "build_feature_profile(" in fn
        assert '"production"' in fn
        assert "Production profile:" in fn

    def test_training_warns_when_no_exploration_profile(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "_EXPLORATION_PROFILE = None" in result
        assert "No exploration feature profile available for comparison" in result

    def test_training_compares_when_exploration_profile_present(self, renderer, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        config = sample_pipeline_config
        config.training = TrainingConfig(
            exploration_feature_profile={"stage": "exploration", "created_at": "2024-01-01", "row_count": 100, "feature_count": 2, "target_column": "churn", "features": {"col_a": {"dtype": "double", "non_null": 90, "null_count": 10}}, "excluded": {}},
        )
        result = renderer.render_training(config)
        assert "_EXPLORATION_PROFILE = {" in result
        assert "FeatureProfile.from_dict(_EXPLORATION_PROFILE)" in result
        assert "compare_feature_profiles(" in result
        assert "feature discrepancies" in result

    def test_training_profile_uses_batched_null_agg(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert "F.sum(F.when(F.col(c).isNull()" in fn

    def test_training_profile_timing(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'log_timing("feature_profile"' in result

    def test_training_profile_is_valid_python(self, renderer, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        config = sample_pipeline_config
        config.training = TrainingConfig(
            exploration_feature_profile={"stage": "exploration", "created_at": "2024-01-01", "row_count": 100, "feature_count": 2, "target_column": "churn", "features": {"col_a": {"dtype": "double", "non_null": 90, "null_count": 10}}, "excluded": {}},
        )
        result = renderer.render_training(config)
        ast.parse(result)


class TestDatabricksTrainingProfilePersistence:
    def test_training_imports_run_namespace(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace" in result

    def test_training_constructs_namespace_from_widgets(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert "dbutils.widgets.get(" in fn or "_NAMESPACE" in result

    def test_training_saves_production_profile(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert "prod_profile.save(" in fn
        assert "production_feature_profile_path" in fn

    def test_training_profile_save_is_valid_python(self, renderer, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        config = sample_pipeline_config
        config.training = TrainingConfig(
            exploration_feature_profile={"stage": "exploration", "created_at": "2024-01-01", "row_count": 100, "feature_count": 2, "target_column": "churn", "features": {"col_a": {"dtype": "double", "non_null": 90, "null_count": 10}}, "excluded": {}},
        )
        result = renderer.render_training(config)
        ast.parse(result)


class TestDatabricksTrainingDistributedSplit:
    def test_no_topandas_for_splitting(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        split_section = result[result.index("def train_and_evaluate"):]
        if "toPandas" in split_section:
            topandas_pos = split_section.index("toPandas")
            context = split_section[max(0, topandas_pos - 200):topandas_pos + 100]
            assert "mlflow.evaluate" in context or "label" in context.lower()

    def test_no_datasplitter_import(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "from customer_retention.stages.modeling.data_splitter" not in result

    def test_no_pandas_roundtrip_for_train_test(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "createDataFrame(train_pdf" not in result
        assert "createDataFrame(test_pdf" not in result

    def test_uses_percentile_for_cutoff(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "percentile_approx" in result or "approxQuantile" in result

    def test_filter_based_temporal_split(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "TIMESTAMP_COLUMN" in result
        fn = result[result.index("def train_and_evaluate"):]
        assert "filter" in fn.lower() or "where" in fn.lower()

    def test_purge_gap_when_configured(self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        silver = SilverLayerConfig(joins=silver_with_join.joins, aggregations=[])
        gold = GoldLayerConfig(encodings=gold_with_encode_scale.encodings, scalings=gold_with_encode_scale.scalings)
        config = PipelineConfig(
            name="test_pipeline", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute}, bronze_event={},
            silver=silver, gold=gold, output_dir="/output",
            composite_name="test__abc1234",
            training=TrainingConfig(purge_gap_days=14),
        )
        result = renderer.render_training(config)
        assert "14" in result
        assert "purge" in result.lower() or "gap" in result.lower() or "timedelta" in result.lower() or "days" in result.lower()

    def test_no_purge_gap_when_not_configured(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "purge_gap" not in result.lower() or "None" in result

    def test_split_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksTrainingNoCrossValidation:
    def test_no_cv_import(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "CrossValidator" not in result
        assert "CVStrategy" not in result

    def test_no_cv_metrics(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "cv_mean" not in result
        assert "cv_std" not in result


class TestDatabricksTrainingExpandedMetrics:
    def test_evaluates_area_under_pr(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "areaUnderPR" in result

    def test_evaluates_accuracy(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "accuracy" in result
        assert "MulticlassClassificationEvaluator" in result

    def test_evaluates_weighted_precision(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "weightedPrecision" in result

    def test_evaluates_weighted_recall(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "weightedRecall" in result

    def test_logs_all_metrics_to_mlflow(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "mlflow.log_metric" in result or "mlflow.log_metrics" in result
        for metric in ("areaUnderPR", "accuracy", "weightedPrecision", "weightedRecall"):
            assert metric in result


class TestDatabricksTrainingMlflowEvaluate:
    def test_mlflow_evaluate_present(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "mlflow.evaluate(" in result

    def test_collects_only_label_and_probability(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "toPandas" in result
        assert "label" in result.lower()
        assert "probability" in result.lower() or "prediction" in result.lower()

    def test_model_type_classifier(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'model_type="classifier"' in result

    def test_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksTrainingFeatureImportance:
    def test_tree_model_feature_importances(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "featureImportances" in result

    def test_logreg_coefficients(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "coefficients" in result

    def test_logs_feature_importance_artifact(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "log_artifact" in result

    def test_feature_names_in_importance(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "feature_cols" in result
        importance_section = result[result.index("featureImportances"):]
        assert "feature_cols" in importance_section


class TestDatabricksTrainingProgress:
    def test_logreg_objective_history(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "objectiveHistory" in result

    def test_step_metrics_logged(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "mlflow.log_metric" in result
        assert "step=" in result

    def test_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksTrainingMlflowNesting:
    def test_parent_run_wraps_nested_model_runs(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        parent_pos = result.find("with mlflow.start_run(run_name=")
        nested_pos = result.find("nested=True")
        assert parent_pos > 0, "Parent mlflow.start_run not found"
        assert nested_pos > parent_pos, "Nested runs should appear after parent run"

    def test_no_set_tag_outside_start_run(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        set_tag_pos = fn.find('mlflow.set_tag(')
        start_run_pos = fn.find('with mlflow.start_run(')
        assert start_run_pos < set_tag_pos, "set_tag must be inside a start_run context"

    def test_no_standalone_best_model_run(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'start_run(run_name="best_model")' not in result

    def test_best_model_logged_to_parent_run(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert 'mlflow.set_tag("best_model"' in fn
        assert 'mlflow.log_metric("best_roc_auc"' in fn

    def test_parent_run_logs_best_model_full_metrics(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert "best_metrics" in fn
        assert 'mlflow.log_metrics({f"best_{k}"' in fn or "best_metrics" in fn

    def test_parent_run_tags_target_column(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert 'mlflow.set_tag("target_column", TARGET)' in fn

    def test_parent_run_tags_entity_key(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert 'mlflow.set_tag("entity_key", "entity_id")' in fn

    def test_parent_run_tags_timestamp_column(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        assert 'mlflow.set_tag("timestamp_column", TIMESTAMP_COLUMN)' in fn

    def test_parent_run_tags_recommendations_hash(self, renderer, sample_pipeline_config):
        sample_pipeline_config.recommendations_hash = "abc123"
        result = renderer.render_training(sample_pipeline_config)
        assert 'mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)' in result

    def test_disables_autolog_before_explicit_tracking(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        autolog_pos = fn.index("mlflow.autolog(disable=True)")
        start_run_pos = fn.index("mlflow.start_run(")
        assert autolog_pos < start_run_pos

    def test_ends_stale_active_run(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        end_run_pos = fn.index("mlflow.end_run()")
        start_run_pos = fn.index("mlflow.start_run(")
        assert end_run_pos < start_run_pos

    def test_experiment_name_uses_composite_name(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "COMPOSITE_NAME" in result.split("set_experiment")[1].split(")")[0]

    def test_parent_run_name_uses_composite_name(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        parent_line = [line for line in result.splitlines() if "with mlflow.start_run(run_name=" in line and "nested" not in line][0]
        assert "COMPOSITE_NAME" in parent_line

    def test_pipeline_name_tagged(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'mlflow.set_tag("pipeline_name", PIPELINE_NAME)' in result


class TestDatabricksTrainingFeatureListLogging:
    def test_logs_feature_list_artifact(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "features.json" in result

    def test_feature_list_contains_column_names(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "feature_cols" in result
        assert "json.dumps" in result

    def test_feature_list_logged_on_parent_run(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        parent_block = fn[fn.index("with mlflow.start_run("):fn.index("return _results")]
        assert "features.json" in parent_block

    def test_feature_list_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)

    def test_feature_list_in_namespace_metadata(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("_training_meta"):]
        assert '"feature_columns"' in fn


class TestDatabricksTrainingMetadataPersistence:
    def test_writes_training_metadata_to_namespace(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "training_metadata_path" in result
        assert "json.dumps" in result or "json.dump" in result

    def test_metadata_includes_experiment_name(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '"mlflow_experiment_name"' in result

    def test_metadata_includes_run_id(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '"mlflow_run_id"' in result

    def test_metadata_includes_scoring_fields(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate"):]
        for field in ("composite_name", "target_column", "timestamp_column", "best_model_name"):
            assert field in fn

    def test_imports_json(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "import json" in result

    def test_metadata_write_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksBronzeEventColumnBlockedFuncs:
    def _make_bronze_event_config(self, column_blocked_funcs=None):
        source = SourceConfig(
            name="emails",
            path="/data/emails.csv",
            format="csv",
            entity_key="customer_id",
            time_column="sent_date",
            is_event_level=True,
        )
        return BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["send_hour"],
                agg_funcs=["sum", "mean"],
                categorical_columns=["direction", "status"],
                categorical_agg_funcs=["nunique", "mode"],
                binary_columns=["opened"],
                binary_agg_funcs=["rate", "count", "any"],
                column_blocked_funcs=column_blocked_funcs or {},
            ),
        )

    def test_blocked_funcs_dict_rendered(self, renderer):
        config = self._make_bronze_event_config(column_blocked_funcs={"status": ["mode"]})
        result = renderer.render_bronze_event("emails", config)
        assert "COLUMN_BLOCKED_FUNCS" in result
        assert "'status'" in result

    def test_empty_dict_rendered(self, renderer):
        config = self._make_bronze_event_config()
        result = renderer.render_bronze_event("emails", config)
        assert "COLUMN_BLOCKED_FUNCS = {}" in result

    def test_guard_in_categorical_loop(self, renderer):
        config = self._make_bronze_event_config(column_blocked_funcs={"status": ["mode"]})
        result = renderer.render_bronze_event("emails", config)
        assert "_blocked = COLUMN_BLOCKED_FUNCS.get(col, [])" in result
        assert '"nunique" not in _blocked' in result
        assert '"mode" not in _blocked' in result

    def test_rendered_template_valid_python(self, renderer):
        config = self._make_bronze_event_config(column_blocked_funcs={"status": ["mode"]})
        result = renderer.render_bronze_event("emails", config)
        ast.parse(result)
