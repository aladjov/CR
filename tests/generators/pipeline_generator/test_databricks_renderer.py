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

    def test_render_config_defines_read_raw_source_helper(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "def read_raw_source(" in result
        assert "spark.read.table(" in result

    def test_render_config_read_raw_source_detects_uc_table(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "def _is_uc_table_reference(" in result
        assert '"/" in path' in result or "'/' in path" in result

    def test_render_config_read_raw_source_handles_csv_and_file_paths(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        helper = result[result.index("def read_raw_source") : result.index("SOURCES = {")]
        assert 'fmt == "csv"' in helper
        assert 'spark.read.format(fmt).load(path)' in helper
        assert 'spark.read.table(path)' in helper

    def test_rendered_uc_table_detector_distinguishes_tables_from_paths(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        start = result.index("_UC_TABLE_FILE_SUFFIXES")
        end = result.index("def read_raw_source", start)
        ns: dict = {}
        exec(result[start:end], ns)
        detector = ns["_is_uc_table_reference"]
        assert detector("sps.production.case") is True
        assert detector("main.retention.landing_orders") is True
        assert detector("/data/orders.parquet") is False
        assert detector("/Volumes/main/retention/raw/orders.csv") is False
        assert detector("dbfs:/mnt/raw/orders.parquet") is False
        assert detector("orders") is False
        assert detector("data.csv") is False
        assert detector("data.parquet") is False


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

    def test_render_bronze_entity_load_source_reads_landing_table(
        self, renderer, sample_pipeline_config
    ):
        """FW-19: entity-level bronze loads from the landing UC table, not
        from the raw upstream. Pre-FW-19 this asserted ``read_raw_source(``
        was in the rendered output; that helper now lives only in landing
        (and in the bronze_event _load_raw_events helper for lifecycle
        enrichment). See ``docs/sps_nb10_runtime_patches_v3.md`` §FW-19."""
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        # Scope to the load_source body — read_raw_source may appear
        # legitimately elsewhere in the rendered notebook (e.g. the
        # config block for sources that have raw-source helpers).
        load_idx = result.index("def load_source")
        next_def = result.index("\ndef ", load_idx + 1)
        load_body = result[load_idx:next_def]
        assert "spark.table(landing_table(SOURCE_NAME))" in load_body
        assert "read_raw_source(" not in load_body


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

    def test_render_bronze_event_emits_per_column_count_agg(self, renderer):
        """FIX SPS-1 — when 'count' is in agg_funcs, the renderer must
        emit `F.count(col).alias(f"{col}_count_{window}")` alongside the
        row-level `event_count_{window}`. NB01d's time_window_aggregator
        emits `{col}_count_{window}` at exploration; the generator must
        match so NB04/NB08 features / recs remain realizable."""
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            aggregation=AggregationWindowConfig(
                windows=["30d", "all_time"], value_columns=["amount"],
                agg_funcs=["sum", "mean", "count"],
            ),
        )
        result = renderer.render_bronze_event("events", config)
        # Row-level event_count is emitted for every window (unchanged).
        assert 'F.count("*").alias("event_count_30d")' in result
        assert 'F.count("*").alias("event_count_all_time")' in result
        # Per-column count IS emitted now (was skipped pre-FIX-SPS-1).
        assert 'F.count(col).alias(f"{col}_count_30d")' in result
        assert 'F.count(col).alias(f"{col}_count_all_time")' in result
        # Other aggs still rendered as before.
        assert 'F.sum(col).alias(f"{col}_sum_30d")' in result
        assert 'F.mean(col).alias(f"{col}_mean_30d")' in result
        ast.parse(result)

    def test_render_bronze_event_omits_count_when_not_in_agg_funcs(self, renderer):
        """Without 'count' in agg_funcs, per-column count is NOT emitted;
        row-level event_count_{window} remains unconditional."""
        source = SourceConfig(
            name="events", path="/data/events.csv", format="csv",
            entity_key="customer_id", time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
            aggregation=AggregationWindowConfig(
                windows=["30d"], value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = renderer.render_bronze_event("events", config)
        assert 'F.count("*").alias("event_count_30d")' in result
        assert 'F.count(col).alias(f"{col}_count_30d")' not in result

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
        assert "raw_entity_key" in result
        assert 'withColumnRenamed(raw_entity_key, "entity_id")' in result

    def test_simplified_merge_skips_rename_when_already_entity_id(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            SilverLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(name="data", path="/data.csv", format="csv", entity_key="entity_id")
        silver = SilverLayerConfig(joins=[], aggregations=[])
        config = PipelineConfig(
            name="test",
            target_column="churn",
            sources=[source],
            bronze={},
            silver=silver,
            gold=None,
            output_dir="/out",
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
        fn = result[result.index("def _log_best_model") :]
        assert "FeatureLookup" in fn
        assert "gold_table()" in fn

    def test_training_creates_training_set(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model") :]
        assert "create_training_set" in fn

    def test_training_registers_model_in_uc(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "registered_model_name" in result
        assert '_registered_name = f"{CATALOG}.{SCHEMA}.model_{COMPOSITE_NAME}"' in result
        assert "registered_model_name=_registered_name" in result

    def test_training_fe_has_import_fallback(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model") :]
        assert "except ImportError" in fn

    def test_training_fallback_uses_mlflow_spark(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model") :]
        assert "mlflow.spark.log_model" in fn

    def test_training_imports_pipeline_model_for_log_wrapping(self, renderer, sample_pipeline_config):
        """``fe.score_batch`` feeds the logged model RAW feature columns via the
        feature-store lookup. A bare ``RandomForestClassificationModel`` that was
        fit on a pre-assembled ``features: vector`` column cannot accept that
        input — Spark raises ``FIELD_NOT_FOUND: features``. The logged artifact
        must be a ``PipelineModel(stages=[VectorAssembler, fitted_estimator])``
        so the assembler runs inside the model, not driver-side during training."""
        result = renderer.render_training(sample_pipeline_config)
        assert "from pyspark.ml import PipelineModel" in result

    def test_prepare_features_returns_filler_and_assembler(self, renderer, sample_pipeline_config):
        """Both the null-filler ``SQLTransformer`` and the unfitted
        ``VectorAssembler`` must survive out of ``prepare_features`` so they
        can both be baked into the logged ``PipelineModel``. The filler is
        what keeps train-time and FE score-time null handling consistent —
        without it, raw gold NULLs reach the assembler at scoring and trip
        the "Encountered null while assembling a row" crash."""
        result = renderer.render_training(sample_pipeline_config)
        prep_fn = result[result.index("def prepare_features"):result.index("def _temporal_split")]
        assert "return assembled, feature_cols, filler, assembler" in prep_fn
        assembled_unpack = [ln for ln in result.splitlines() if "= prepare_features(df)" in ln]
        assert assembled_unpack, "call site not found"
        assert any(
            ", _filler, _assembler = prepare_features(df)" in ln
            for ln in assembled_unpack
        ), f"call site must unpack filler + assembler: {assembled_unpack}"

    def test_prepare_features_uses_sql_transformer_filler_not_imputer(self, renderer, sample_pipeline_config):
        """Coding_Practices.md bans ``pyspark.ml.Imputer`` past ~100 cols
        (serialized ML model exceeds 1 GB). The null-filler must be a
        ``SQLTransformer`` (pure Transformer, no ``.fit()``, zero fit-time
        cost, scales to any column count)."""
        result = renderer.render_training(sample_pipeline_config)
        assert "SQLTransformer" in result
        assert "from pyspark.ml.feature import SQLTransformer, VectorAssembler" in result
        # No data-prep Imputer
        assert "from pyspark.ml.feature import Imputer" not in result
        assert "Imputer(" not in result
        # COALESCE + nanvl wrap every feature col (NULL→0 AND NaN→0).
        prep_fn = result[result.index("def _build_null_filler"):result.index("def prepare_features")]
        assert "COALESCE(nanvl(CAST(" in prep_fn
        assert "AS DOUBLE), 0.0), 0.0)" in prep_fn

    def test_log_best_model_wraps_three_stage_pipeline(self, renderer, sample_pipeline_config):
        """``_log_best_model`` must construct
        ``PipelineModel(stages=[filler, assembler, model])`` and pass the
        wrapped model to both ``fe.log_model`` and the fallback
        ``mlflow.spark.log_model`` — training- and scoring-time null handling
        live in the same SQLTransformer stage."""
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model"):result.index("def _promote_to_production")]
        assert "PipelineModel(stages=[filler, assembler, model])" in fn
        fe_call_start = fn.index("fe.log_model(")
        fe_call_end = fn.index(")", fe_call_start)
        fe_call = fn[fe_call_start:fe_call_end]
        assert "model=model," not in fe_call, "fe.log_model must receive the wrapped pipeline, not the bare estimator"
        fallback_call_start = fn.index("mlflow.spark.log_model(")
        fallback_call_end = fn.index(")", fallback_call_start)
        fallback_call = fn[fallback_call_start:fallback_call_end]
        assert "\n            model," not in fallback_call, "fallback log_model must receive wrapped pipeline"

    def test_log_best_model_accepts_filler_and_assembler_params(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "def _log_best_model(model, df, feature_cols, test_df, filler, assembler)" in result
        assert "_log_best_model(best_model, df, feature_cols, test_df, _filler, _assembler)" in result

    def test_log_best_model_fallback_signature_uses_raw_columns(self, renderer, sample_pipeline_config):
        """The fallback ``infer_signature`` call must use raw feature columns as
        input — the logged pipeline accepts raw, not pre-assembled vectors.
        Using ``test_df`` (which has ``features: vector``) would produce a
        signature that ``fe.score_batch`` cannot satisfy."""
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model"):result.index("def _promote_to_production")]
        except_block = fn[fn.index("except ImportError"):]
        assert "infer_signature(test_df," not in except_block, (
            "signature must be inferred from raw input frame, not pre-assembled test_df"
        )

    def test_training_imports_shap_attribution_helpers(self, renderer, sample_pipeline_config):
        """Persisted attribution removes the ``fe.score_batch`` + batched-corr
        round-trip from every SHAP consumer; both helpers must be imported so
        the training script can compute and log the artifact."""
        result = renderer.render_training(sample_pipeline_config)
        assert (
            "from customer_retention.stages.modeling.shap_attribution import "
            "compute_shap_attribution, log_attribution"
        ) in result

    def test_training_logs_attribution_for_best_model(self, renderer, sample_pipeline_config):
        """After the best model is registered, the training run must compute
        and log a ``ShapAttribution`` (importances + background means) tied to
        the same MLflow run. Downstream SHAP consumers load this instead of
        rescoring on every call."""
        result = renderer.render_training(sample_pipeline_config)
        main = result[result.index("def train_and_evaluate"):]
        best_block_start = main.index("if best_model is not None:")
        best_block_end = main.index("_results[\"best_model\"]")
        best_block = main[best_block_start:best_block_end]
        assert "compute_shap_attribution(" in best_block
        assert "log_attribution(" in best_block
        assert "PipelineModel(stages=[_filler, _assembler, best_model])" in best_block

    def test_attribution_is_logged_only_for_the_best_model(self, renderer, sample_pipeline_config):
        """Attribution is a function of (model, training cohort). Logging one
        per candidate would waste training time; we only persist the artifact
        tied to the best model — the alias downstream consumers resolve."""
        result = renderer.render_training(sample_pipeline_config)
        main = result[result.index("def train_and_evaluate"):]
        per_candidate_block = main[
            main.index("for name, model in models.items()"):
            main.index("if best_model_name is not None:")
        ]
        assert "compute_shap_attribution" not in per_candidate_block, (
            "attribution must be computed only for the best model, not per candidate"
        )

    def test_training_is_valid_python_after_attribution_emission(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)

    def test_training_nested_runs_still_use_mlflow_spark(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("for name, model in models") :]
        assert "mlflow.spark.log_model(fitted" in fn

    def test_training_fallback_passes_dfs_tmpdir(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model") :]
        assert 'dfs_tmpdir=_DFS_TMPDIR' in fn

    def test_training_nested_run_passes_dfs_tmpdir(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("for name, model in models") :]
        assert 'dfs_tmpdir=_DFS_TMPDIR' in fn

    def test_training_defines_dfs_tmpdir_constant(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '_DFS_TMPDIR = ' in result
        assert 'get_mlflow_dfs_tmpdir' in result

    def test_training_log_best_model_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)

    def test_training_captures_logged_models_list(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "_logged_models = []" in result
        assert "_logged_models.append" in result
        assert '"model_uri": _log_info.model_uri' in result
        assert '"flavor": "spark"' in result

    def test_training_metadata_includes_logged_models(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '"logged_models": _logged_models' in result
        assert '"registered_model_name": _registered_model_name' in result

    def test_training_promotes_to_production_alias(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "def _promote_to_production" in result
        assert "set_registered_model_alias" in result
        assert '"production"' in result
        assert "_promote_to_production(_registered_model_name" in result

    def test_promote_to_production_uses_name_only_filter(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _promote_to_production") : result.index("def train_and_evaluate")]
        assert "search_model_versions(f\"name='{registered_name}'\")" in fn
        assert "and run_id=" not in fn

    def test_promote_to_production_filters_run_id_python_side(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _promote_to_production") : result.index("def train_and_evaluate")]
        assert "v.run_id == parent_run_id" in fn

    def test_promote_to_production_falls_back_to_latest_version(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _promote_to_production") : result.index("def train_and_evaluate")]
        assert "_matching or _versions" in fn
        assert "max(_candidates, key=lambda v: int(v.version))" in fn

    def test_log_best_model_returns_registered_name(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def _log_best_model") : result.index("def _promote_to_production")]
        assert "return _registered_name" in fn


class TestDatabricksRenderTrainingImbalance:
    def _make_config_with_imbalance(
        self, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale, strategy
    ):
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

    def test_training_class_weight_adds_weight_col(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        config = self._make_config_with_imbalance(
            entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale, "class_weight"
        )
        result = renderer.render_training(config)
        assert "weightCol" in result or "weight" in result.lower()
        assert "class_weight" in result.lower() or "balanced" in result.lower() or "weight_col" in result

    def test_training_smote_adds_resampling(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        config = self._make_config_with_imbalance(
            entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale, "smote"
        )
        result = renderer.render_training(config)
        assert "SMOTE" in result or "smote" in result.lower()

    def test_training_imbalance_is_valid_python(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        for strategy in ("class_weight", "smote"):
            config = self._make_config_with_imbalance(
                entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale, strategy
            )
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

    def test_render_runner_sets_batch_execution(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "CR_BATCH_EXECUTION" in result

    def test_render_runner_has_spark_job_counter(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "_spark_job_id" in result
        assert "nextJobId" in result

    def test_render_runner_collects_profile(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "_profile" in result
        assert "spark_jobs" in result

    def test_render_runner_writes_cell_profiles(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "cell_profiles_path" in result


class TestDatabricksNotebookExitSummary:
    def test_bronze_exits_with_summary(self, renderer):
        source = SourceConfig(name="t", path="t.csv", format="csv", entity_key="id")
        config = BronzeLayerConfig(source=source, transformations=[])
        result = renderer.render_bronze("t", config)
        assert "dbutils.notebook.exit(_summary)" in result
        assert "result.count()" in result

    def test_bronze_event_exits_with_summary(self, renderer):
        source = SourceConfig(
            name="ev", path="ev.csv", format="csv", entity_key="id", time_column="ts", is_event_level=True
        )
        config = BronzeEventConfig(source=source, entity_column="id", time_column="ts")
        result = renderer.render_bronze_event("ev", config)
        assert "dbutils.notebook.exit(_summary)" in result

    def test_bronze_entity_exits_with_summary(self, renderer):
        source = SourceConfig(
            name="ev", path="ev.csv", format="csv", entity_key="id", time_column="ts", is_event_level=True
        )
        config = BronzeEventConfig(source=source, entity_column="id", time_column="ts")
        result = renderer.render_bronze_entity("ev_aggregated", config, "ev")
        assert "dbutils.notebook.exit(_summary)" in result

    def test_silver_exits_with_summary(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "dbutils.notebook.exit(json.dumps(_silver_results" in result
        assert "SILVER RESULTS" in result

    def test_gold_exits_with_summary(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "dbutils.notebook.exit(_summary)" in result

    def test_training_exits_with_summary(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "dbutils.notebook.exit(json.dumps(_training_results" in result
        assert "TRAINING RESULTS" in result

    def test_landing_exits_with_summary(self, renderer):
        source = SourceConfig(name="t", path="t.csv", format="csv", entity_key="id")
        config = LandingLayerConfig(
            source=source,
            raw_source_path="t.csv",
            raw_source_format="csv",
            entity_column="id",
            time_column="ts",
            target_column="churn",
        )
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

    def test_bronze_load_source_does_not_dispatch_on_format(self, renderer):
        """FW-19: entity-level bronze reads from the landing UC table — a
        Delta table — regardless of the raw upstream's original format.
        Format-aware dispatch happens in the landing template, not bronze.
        Pre-FW-19 this test asserted ``read_raw_source(`` in the load body
        and that ``format(...)`` didn't appear; now both are absent because
        the load body is a single ``spark.table(landing_table(...))`` call."""
        source = SourceConfig(
            name="customers",
            path="/data/customers.parquet",
            format="parquet",
            entity_key="customer_id",
        )
        config = BronzeLayerConfig(source=source)
        result = renderer.render_bronze("customers", config)
        load_start = result.index("def load_source")
        load_end = result.index("\n\n", load_start)
        load_fn = result[load_start:load_end]
        assert "spark.table(landing_table(SOURCE_NAME))" in load_fn
        assert "read_raw_source(" not in load_fn
        # Format dispatch belongs to landing.py.j2's read_raw_source helper,
        # not the bronze load_source body.
        assert 'format("parquet")' not in load_fn
        assert 'format("delta")' not in load_fn

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
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
            aggregation=AggregationWindowConfig(
                windows=["30d"],
                value_columns=["amount"],
                agg_funcs=["sum"],
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
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
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
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
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
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
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
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
            deduplicate=True,
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" in code
        assert "row_number" in code

    def test_dedup_keep_first(self, renderer):
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig

        source = SourceConfig(
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
            deduplicate=DeduplicationConfig(strategy="keep_first"),
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" in code
        assert "row_number" in code
        ast.parse(code)

    def test_dedup_keep_most_complete(self, renderer):
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig

        source = SourceConfig(
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
            deduplicate=DeduplicationConfig(strategy="keep_most_complete"),
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" in code
        assert "null_count" in code or "isNull" in code.lower() or "isnull" in code.lower()
        ast.parse(code)

    def test_dedup_with_conflict_columns(self, renderer):
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig

        source = SourceConfig(
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
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
            name="events",
            path="/data/events.csv",
            format="csv",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
            deduplicate=False,
        )
        code = renderer.render_bronze_event("events", config)
        assert "deduplicate" not in code


class TestDatabricksMomentumRatios:
    def test_bronze_entity_includes_momentum_ratios(self, renderer):
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
                momentum_pairs=[
                    {"short_window": "7d", "long_window": "30d"},
                    {"short_window": "30d", "long_window": "90d"},
                ],
            ),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        assert "add_momentum_ratios" in result
        assert "momentum_7d_30d" in result
        assert "momentum_30d_90d" in result
        assert "event_count_7d" in result
        assert "event_count_30d" in result

    def test_bronze_entity_momentum_uses_safe_division(self, renderer):
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
                momentum_pairs=[{"short_window": "7d", "long_window": "30d"}],
            ),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        assert "F.when" in result
        assert "!= 0" in result

    def test_bronze_entity_no_momentum_without_pairs(self, renderer):
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
            lifecycle=LifecycleConfig(include_recency_bucket=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        assert "add_momentum_ratios" not in result

    def test_bronze_standalone_entity_momentum(self, renderer):
        source = SourceConfig(
            name="customers",
            path="/data/customers.csv",
            format="csv",
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
                momentum_pairs=[
                    {"short_window": "7d", "long_window": "30d"},
                    {"short_window": "30d", "long_window": "90d"},
                ],
            ),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        ast.parse(result)


class TestDatabricksLandingTemplate:
    @pytest.fixture
    def landing_config(self):
        source = SourceConfig(
            name="orders",
            path="/data/orders.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
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

    def test_landing_uses_read_raw_source_helper(self, renderer, landing_config):
        result = renderer.render_landing("orders", landing_config)
        assert "read_raw_source(" in result
        assert 'spark.read.format(fmt).load(path)' not in result

    def test_landing_with_uc_table_raw_source_uses_read_raw_source(self, renderer):
        source = SourceConfig(
            name="case",
            path="sps.production.case",
            format="delta",
            entity_key="customer_id",
            time_column="created_at",
            is_event_level=True,
        )
        landing_config = LandingLayerConfig(
            source=source,
            raw_source_path="sps.production.case",
            raw_source_format="delta",
            entity_column="customer_id",
            time_column="created_at",
            target_column="churn",
        )
        result = renderer.render_landing("case", landing_config)
        assert "read_raw_source(" in result
        assert "spark.read.format(fmt).load(path)" not in result
        ast.parse(result)

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


class TestFW4LandingBronzeGuards:
    """FW-4: codegen-time landing/bronze guards retiring NB10 §2.1, §2.3, §2.4, §2.6."""

    @pytest.fixture
    def landing_with_self_key_join(self):
        from customer_retention.generators.pipeline_generator.models import (
            KeyResolutionStepConfig,
        )
        source = SourceConfig(
            name="opportunity_product",
            path="/data/oppty_product.parquet",
            format="parquet",
            entity_key="OPPORTUNITY_ID",
            time_column="created_at",
            is_event_level=True,
        )
        return LandingLayerConfig(
            source=source,
            raw_source_path="/data/oppty_product.parquet",
            raw_source_format="parquet",
            entity_column="OPPORTUNITY_ID",
            time_column="created_at",
            target_column="churn",
            key_resolution_steps=[
                KeyResolutionStepConfig(
                    bridge_dataset="opportunity",
                    source_key="OPPORTUNITY_ID",
                    bridge_key="OPPORTUNITY_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        )

    @pytest.fixture
    def landing_with_distinct_key_join(self):
        from customer_retention.generators.pipeline_generator.models import (
            KeyResolutionStepConfig,
        )
        source = SourceConfig(
            name="case",
            path="/data/case.parquet",
            format="parquet",
            entity_key="CASE_ID",
            time_column="created_at",
            is_event_level=True,
        )
        return LandingLayerConfig(
            source=source,
            raw_source_path="/data/case.parquet",
            raw_source_format="parquet",
            entity_column="CASE_ID",
            time_column="created_at",
            target_column="churn",
            key_resolution_steps=[
                KeyResolutionStepConfig(
                    bridge_dataset="account",
                    source_key="CASE_ID",
                    bridge_key="CASE_ID",
                    resolve_column="ACCOUNT_ID",
                ),
            ],
        )

    def test_landing_disables_zorder_advisory_check(self, renderer, landing_with_self_key_join):
        """§2.1 part 1: zorder stats-collection check is silenced via spark.conf.set
        at the top of every landing notebook (advisory only; data correctness
        unaffected)."""
        result = renderer.render_landing("opportunity_product", landing_with_self_key_join)
        assert (
            'spark.conf.set("spark.databricks.delta.optimize.zorder.checkStatsCollection.enabled", "false")'
            in result
        )

    def test_landing_bridge_alias_in_select(self, renderer, landing_with_self_key_join):
        """§2.1 part 2: the bridge frame aliases its key as ``__cr_bridge_<KEY>``
        before the join so source_key == bridge_key never produces UC duplicates."""
        result = renderer.render_landing("opportunity_product", landing_with_self_key_join)
        assert 'F.col("OPPORTUNITY_ID").alias("__cr_bridge_OPPORTUNITY_ID")' in result
        assert 'dropDuplicates(["__cr_bridge_OPPORTUNITY_ID"])' in result

    def test_landing_bridge_alias_in_join_predicate(self, renderer, landing_with_self_key_join):
        """The join predicate references the alias, not the original bridge key."""
        result = renderer.render_landing("opportunity_product", landing_with_self_key_join)
        assert (
            'df.join(_bridge, df["OPPORTUNITY_ID"] == _bridge["__cr_bridge_OPPORTUNITY_ID"], "inner")'
            in result
        )

    def test_landing_drops_alias_unconditionally(self, renderer, landing_with_self_key_join):
        """Drop the aliased column unconditionally — works for both
        source_key == bridge_key (collision) and source_key != bridge_key paths."""
        result = renderer.render_landing("opportunity_product", landing_with_self_key_join)
        assert 'df = df.drop("__cr_bridge_OPPORTUNITY_ID")' in result

    def test_landing_no_legacy_unaliased_join(self, renderer, landing_with_self_key_join):
        """The legacy unaliased ``_bridge.select("KEY", ...)`` form must be gone."""
        result = renderer.render_landing("opportunity_product", landing_with_self_key_join)
        assert '_bridge.select("OPPORTUNITY_ID", "ACCOUNT_ID")' not in result

    def test_landing_distinct_key_also_uses_alias(
        self, renderer, landing_with_distinct_key_join,
    ):
        """Even when source_key != bridge_key, the alias path is used so
        codegen has one shape for both cases."""
        result = renderer.render_landing("case", landing_with_distinct_key_join)
        assert 'F.col("CASE_ID").alias("__cr_bridge_CASE_ID")' in result
        assert 'df = df.drop("__cr_bridge_CASE_ID")' in result

    def test_landing_with_key_resolution_is_valid_python(
        self, renderer, landing_with_self_key_join,
    ):
        result = renderer.render_landing("opportunity_product", landing_with_self_key_join)
        ast.parse(result)


class TestFW4BronzeEventGuards:
    """FW-4: codegen-time bronze_event guards retiring NB10 §2.3, §2.4, §2.6."""

    @pytest.fixture
    def bronze_event_with_temporal_features(self):
        from customer_retention.generators.pipeline_generator.models import (
            TemporalFeatureConfig,
        )
        source = SourceConfig(
            name="case",
            path="/data/case.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="created_date",
            is_event_level=True,
        )
        return BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="created_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
                binary_columns=["is_active"],
            ),
            temporal_features=TemporalFeatureConfig(
                lag_window_days=7,
                num_lags=4,
                lag_columns=["amount"],
                lag_agg_funcs=["sum", "mean"],
                feature_groups=["lagged_windows", "velocity"],
            ),
        )

    @pytest.fixture
    def bronze_event_minimal(self):
        source = SourceConfig(
            name="orders",
            path="/data/orders.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        return BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["30d"],
                value_columns=["amount"],
                agg_funcs=["sum"],
                binary_columns=["is_paid"],
            ),
        )

    def test_bronze_event_filters_temporal_value_cols_by_dtype(
        self, renderer, bronze_event_with_temporal_features,
    ):
        """§2.3: ``compute_temporal_features`` narrows ``value_cols`` to
        columns that exist on raw_df AND carry a numeric Spark dtype, so the
        engineer's ``cast("DOUBLE")`` never sees a STRING flag."""
        result = renderer.render_bronze_event("case", bronze_event_with_temporal_features)
        assert "_raw_dtypes = dict(raw_df.dtypes)" in result
        assert '_raw_dtypes.get(_c, "string") in _numeric_dtypes' in result
        assert '_raw_dtypes.get(_c, "string").startswith("decimal")' in result

    def test_bronze_event_temporal_value_cols_filter_includes_existence(
        self, renderer, bronze_event_with_temporal_features,
    ):
        """The filter must AND the existence check with the dtype check —
        a value column missing from raw_df is also dropped."""
        result = renderer.render_bronze_event("case", bronze_event_with_temporal_features)
        assert "if _c in raw_df.columns" in result

    def test_bronze_event_no_unfiltered_value_cols_call(
        self, renderer, bronze_event_with_temporal_features,
    ):
        """The raw, unfiltered ``engineer.compute(..., value_cols)`` shape must
        not appear unguarded — value_cols is rebound to the filtered list before
        the call site."""
        result = renderer.render_bronze_event("case", bronze_event_with_temporal_features)
        # The engineer call still references value_cols, but value_cols was rebound.
        assert "engineer.compute(raw_df, ENTITY_COLUMN, TIME_COLUMN, value_cols)" in result
        # The numeric filter must precede the engineer call.
        filter_pos = result.index("_numeric_dtypes")
        engineer_pos = result.index("engineer.compute(raw_df")
        assert filter_pos < engineer_pos

    def test_bronze_event_compute_temporal_features_call_wrapped(
        self, renderer, bronze_event_with_temporal_features,
    ):
        """§2.4: the call site of ``compute_temporal_features(agg_df, raw_df)``
        runs inside try/except so any unforeseen failure logs and falls
        through to saveAsTable."""
        result = renderer.render_bronze_event("case", bronze_event_with_temporal_features)
        assert "try:" in result
        assert "agg_df = compute_temporal_features(agg_df, raw_df)" in result
        assert "except Exception as _temporal_err:" in result
        assert "writing without temporal features" in result

    def test_bronze_event_no_temporal_wrap_when_no_temporal_features(
        self, renderer, bronze_event_minimal,
    ):
        """When the source has no temporal_features at all, the try/except
        wrapper must not be emitted (otherwise we'd wrap nothing)."""
        result = renderer.render_bronze_event("orders", bronze_event_minimal)
        assert "compute_temporal_features" not in result
        assert "writing without temporal features" not in result

    def test_bronze_event_binary_columns_dtype_filter(
        self, renderer, bronze_event_with_temporal_features,
    ):
        """§2.6: ``apply_event_aggregation`` narrows BINARY_COLUMNS to columns
        with a numerically coercible Spark dtype before any F.mean/F.sum/F.max
        touches them. Required global declaration so the rebind is visible."""
        result = renderer.render_bronze_event("case", bronze_event_with_temporal_features)
        assert "global BINARY_COLUMNS" in result
        assert "_numeric_dtype_prefixes" in result
        assert "BINARY_COLUMNS = [c for c in BINARY_COLUMNS if _is_numeric_dtype(c)]" in result

    def test_bronze_event_binary_filter_appears_in_apply_event_aggregation(
        self, renderer, bronze_event_with_temporal_features,
    ):
        """The dtype filter must be inside ``apply_event_aggregation`` and
        BEFORE the aggregation builds — so BINARY_COLUMNS is narrowed by the
        time F.mean is wired up."""
        result = renderer.render_bronze_event("case", bronze_event_with_temporal_features)
        func_pos = result.index("def apply_event_aggregation(df):")
        filter_pos = result.index("BINARY_COLUMNS = [c for c in BINARY_COLUMNS if _is_numeric_dtype(c)]")
        for_pos = result.index("for col in BINARY_COLUMNS:")
        assert func_pos < filter_pos < for_pos

    def test_bronze_event_with_fw4_guards_is_valid_python(
        self, renderer, bronze_event_with_temporal_features,
    ):
        result = renderer.render_bronze_event("case", bronze_event_with_temporal_features)
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
            name="emails",
            path="/data/emails.csv",
            format="csv",
            entity_key="customer_id",
            time_column="sent_date",
            is_event_level=True,
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
            joins=[
                {
                    "left_keys": ["customer_id"],
                    "right_keys": ["customer_id"],
                    "right_source": "orders",
                    "how": "left",
                }
            ],
            grid_dates=["2024-01-01", "2024-01-08", "2024-01-15"],
            entity_key="customer_id",
            merge_sources=[
                TemporalMergeSourceConfig(name="customers", granularity="entity_level"),
                TemporalMergeSourceConfig(
                    name="orders", granularity="event_level", feature_timestamp_column="order_date"
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
    def temporal_config_with_key_resolution(
        self, entity_source, event_source, bronze_with_impute, gold_with_encode_scale
    ):
        from customer_retention.generators.pipeline_generator.models import (
            KeyResolutionStepConfig,
            TemporalMergeSourceConfig,
        )

        silver = SilverLayerConfig(
            joins=[
                {
                    "left_keys": ["customer_id"],
                    "right_keys": ["customer_id"],
                    "right_source": "orders",
                    "how": "left",
                }
            ],
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

    def test_silver_kr_step_emits_codegen_guard_for_pre_resolved_source(
        self, renderer, temporal_config_with_key_resolution,
    ):
        """Fix #4: silver kr_steps loop must emit a runtime guard that
        skips the join when ``step["source_key"]`` is no longer a column
        on the bronze output (because the bronze aggregator already
        rolled up to ``resolve_column`` grain). This was previously the
        operator-paste `patch_skip_silver_kr_steps_when_source_key_absent`
        / cycle_024 failure mode; the guard now ships in the framework."""
        result = renderer.render_silver(temporal_config_with_key_resolution)
        # Guard reads `if step["source_key"] not in df.columns`
        assert 'not in df.columns' in result
        # Operator-friendly diagnostic so the skip is visible in run logs
        assert "skipping kr_step" in result
        # Skip must short-circuit before the join — `continue` keyword present
        # AND the guard appears textually before the actual `df.join(`
        assert "continue" in result
        guard_pos = result.find("not in df.columns")
        join_pos = result.find("df = df.join(")
        assert guard_pos != -1 and join_pos != -1
        assert guard_pos < join_pos


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
    def test_training_nulls_handled_inside_pipeline_not_by_fillna(self, renderer, sample_pipeline_config):
        """Regression for ``Encountered null while assembling a row with
        handleInvalid='error'`` at ``fe.score_batch`` time. Null handling
        must be a pipeline *stage* (SQLTransformer), not a driver-side
        ``df.fillna(...)`` — the latter only runs during training and
        leaves raw NULLs in the gold feature-store rows that FE fetches
        at inference, which the logged VectorAssembler then rejects."""
        result = renderer.render_training(sample_pipeline_config)
        prepare_fn = result[result.index("def prepare_features"):result.index("def _temporal_split")]
        # fillna is NOT the null-handling primitive anymore — no upfront
        # driver-side fillna in prepare_features.
        assert "fillna" not in prepare_fn
        # Filler stage is built + transforms the frame before VectorAssembler.
        filler_pos = prepare_fn.index("filler = _build_null_filler(")
        transform_pos = prepare_fn.index("filler.transform(df)")
        assembler_pos = prepare_fn.index("VectorAssembler(")
        assert filler_pos < transform_pos < assembler_pos

    def test_training_assembler_keeps_invalid_rows_as_belt_and_suspenders(
        self, renderer, sample_pipeline_config
    ):
        """With the SQLTransformer filler coercing NULL/NaN to 0 first, the
        VectorAssembler never *sees* a null. ``handleInvalid='keep'`` is the
        defensive default in case a stray NaN slips past the filler — it
        lets the NaN propagate into the vector rather than crashing the
        whole batch scoring run."""
        result = renderer.render_training(sample_pipeline_config)
        assert 'handleInvalid="keep"' in result
        # Must NOT use "error" — that's the mode that crashed at scoring.
        assert 'handleInvalid="error"' not in result

    def test_training_filler_coerces_feature_cols_to_double_and_zero(
        self, renderer, sample_pipeline_config
    ):
        """The SQL wraps every feature in COALESCE+nanvl so both NULL and
        NaN collapse to 0.0 before the assembler sees them."""
        result = renderer.render_training(sample_pipeline_config)
        assert "SQLTransformer" in result
        assert "COALESCE(nanvl(CAST(" in result
        assert "AS DOUBLE), 0.0), 0.0)" in result

    def test_training_filler_uses_star_projection_not_explicit_passthrough(
        self, renderer, sample_pipeline_config
    ):
        """Regression for ``UNRESOLVED_COLUMN`` at fe.score_batch time.

        The SQL must NOT reference pass-through columns (entity_id /
        event_timestamp / label) by name — those exist at training but FE
        feeds the logged pipeline only the feature columns at score time.
        Use ``SELECT *, COALESCE(...) AS col__nafilled`` so:
          (a) training: ``*`` passes label / timestamp / entity_id through,
              and the new ``__nafilled`` copies are added alongside,
          (b) scoring: ``*`` passes the 20 feature columns through and the
              filler never asks for columns that aren't there.
        """
        result = renderer.render_training(sample_pipeline_config)
        prep_fn = result[result.index("def _build_null_filler"):result.index("def prepare_features")]
        assert '"SELECT " + ", ".join(projections) + " FROM __THIS__"' in prep_fn
        assert 'projections = ["*"]' in prep_fn
        # VectorAssembler inputs must be the __nafilled names, not the raw feature cols.
        assert '__nafilled' in prep_fn or '_NAFILLED_SUFFIX' in prep_fn
        main = result[result.index("def prepare_features"):]
        assert "inputCols=filled_cols" in main


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
            lifecycle=LifecycleConfig(include_recency_bucket=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        assert 'landing_table("orders")' in result
        assert "add_recency_tenure" in result
        recency_fn = result[result.index("def add_recency_tenure") :]
        recency_fn = recency_fn[: recency_fn.index("\ndef ")]
        assert "landing_table" in recency_fn
        assert "bronze_table" not in recency_fn
        ast.parse(result)

    def test_month_cyclical_reads_from_landing_table(self, renderer):
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
            lifecycle=LifecycleConfig(include_month_cyclical=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        assert 'landing_table("orders")' in result
        assert "add_month_quarter_cyclical" in result
        cyclical_fn = result[result.index("def add_month_quarter_cyclical") :]
        cyclical_fn = cyclical_fn[: cyclical_fn.index("\ndef ")]
        assert "landing_table" in cyclical_fn
        assert "bronze_table" not in cyclical_fn
        ast.parse(result)

    def test_cyclical_features_reads_from_landing_table(self, renderer):
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
            lifecycle=LifecycleConfig(include_cyclical_features=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        assert 'landing_table("orders")' in result
        assert "add_cyclical_features" in result
        assert "dow_sin" in result
        assert "dow_cos" in result
        cyclical_fn = result[result.index("def add_cyclical_features") :]
        cyclical_fn = cyclical_fn[: cyclical_fn.index("\ndef ")]
        assert "landing_table" in cyclical_fn
        assert "bronze_table" not in cyclical_fn
        ast.parse(result)

    def test_cohort_features_reads_from_landing_table(self, renderer):
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
            lifecycle=LifecycleConfig(include_cohort_features=True),
            post_shaping=[],
        )
        result = renderer.render_bronze_entity(
            "orders_aggregated",
            config,
            "orders",
            "orders",
        )
        assert 'landing_table("orders")' in result
        assert "add_cohort_features" in result
        cohort_fn = result[result.index("def add_cohort_features") :]
        cohort_fn = cohort_fn[: cohort_fn.index("\ndef ")]
        assert "landing_table" in cohort_fn
        assert "bronze_table" not in cohort_fn
        ast.parse(result)


class TestDatabricksSilverToSparkConversion:
    @pytest.fixture
    def temporal_config(self, entity_source, event_source, bronze_with_impute, gold_with_encode_scale):
        from customer_retention.generators.pipeline_generator.models import TemporalMergeSourceConfig

        silver = SilverLayerConfig(
            joins=[
                {
                    "left_keys": ["customer_id"],
                    "right_keys": ["customer_id"],
                    "right_source": "orders",
                    "how": "left",
                }
            ],
            grid_dates=["2024-01-01", "2024-01-08"],
            entity_key="customer_id",
            merge_sources=[
                TemporalMergeSourceConfig(name="customers", granularity="entity_level"),
                TemporalMergeSourceConfig(
                    name="orders", granularity="event_level", feature_timestamp_column="order_date"
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
                    aggregation=AggregationWindowConfig(windows=["7d"], value_columns=["amount"], agg_funcs=["sum"]),
                )
            },
            silver=silver,
            gold=gold_with_encode_scale,
            output_dir="/output/test_pipeline",
            composite_name="cust_orde__abc1234",
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
        apply_scalings_fn = result[result.index("def apply_scalings") :]
        apply_scalings_fn = apply_scalings_fn[: apply_scalings_fn.index("\n\n")]
        assert '_scale_standard(df, "' not in apply_scalings_fn
        assert '_scale_minmax(df, "' not in apply_scalings_fn

    def test_gold_mixed_scaling_methods(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join
    ):
        gold = GoldLayerConfig(
            encodings=[],
            scalings=[
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="amount",
                    parameters={"method": "standard"},
                    rationale="Standardize amount",
                ),
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="revenue",
                    parameters={"method": "standard"},
                    rationale="Standardize revenue",
                ),
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="score",
                    parameters={"method": "minmax"},
                    rationale="MinMax score",
                ),
            ],
        )
        config = PipelineConfig(
            name="test_pipeline",
            target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute},
            silver=SilverLayerConfig(joins=silver_with_join.joins, aggregations=[]),
            gold=gold,
            output_dir="/output",
            composite_name="cust_orde__abc1234",
        )
        result = renderer.render_gold(config)
        assert '_batch_scale_standard(df, ["amount", "revenue"])' in result
        assert '_batch_scale_minmax(df, ["score"])' in result
        ast.parse(result)

    def test_gold_batch_scale_single_agg_call(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        batch_fn = result[result.index("def _batch_scale_standard") :]
        batch_fn = batch_fn[: batch_fn.index("\ndef ")]
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
            name="test_pipeline",
            target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": bronze_with_impute},
            silver=SilverLayerConfig(joins=silver_with_join.joins, aggregations=[]),
            gold=gold,
            output_dir="/output",
            composite_name="cust_orde__abc1234",
        )

    def test_gold_defines_batch_cap_then_log_helper(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        config = self._make_config(
            entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
        )
        result = renderer.render_gold(config)
        assert "def _batch_cap_then_log(df, cols):" in result

    def test_gold_cap_then_log_uses_approx_quantile(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        config = self._make_config(
            entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
        )
        result = renderer.render_gold(config)
        assert "approxQuantile" in result

    def test_gold_cap_then_log_applies_log1p(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        config = self._make_config(
            entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
        )
        result = renderer.render_gold(config)
        assert "F.log1p" in result

    def test_gold_cap_then_log_is_valid_python(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        config = self._make_config(
            entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
        )
        result = renderer.render_gold(config)
        ast.parse(result)

    def test_gold_cap_then_log_no_f_lit_column(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        config = self._make_config(
            entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
        )
        result = renderer.render_gold(config)
        assert "F.lit(F.col(" not in result


class TestDatabricksGoldColumnFiltering:
    def test_batch_scale_standard_filters_to_existing_columns(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        batch_fn = result[result.index("def _batch_scale_standard") :]
        batch_fn = batch_fn[: batch_fn.index("\ndef ")]
        assert "c in df.columns" in batch_fn or "c for c in cols if c in" in batch_fn

    def test_batch_scale_minmax_filters_to_existing_columns(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        batch_fn = result[result.index("def _batch_scale_minmax") :]
        batch_fn = batch_fn[: batch_fn.index("\ndef ")]
        assert "c in df.columns" in batch_fn or "c for c in cols if c in" in batch_fn

    def test_encode_one_hot_checks_column_exists(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        encode_fn = result[result.index("def _encode_one_hot") :]
        encode_fn = encode_fn[: encode_fn.index("\ndef ")]
        assert "col not in df.columns" in encode_fn

    def test_encode_one_hot_failsafe_when_target_missing(self, renderer, sample_pipeline_config):
        """FW-5 §2.8: when a configured encoding target has been dropped
        upstream (feature_selection / leakage exclusion / silver merge prefix
        collision), `_encode_one_hot` warns and returns df unchanged instead
        of raising. Raising blocks gold's saveAsTable 6-7h after the silent
        drop happened — the PARITY_IGNORED_FEATURES escape hatch in training
        handles the missing `<col>_<category>` columns downstream."""
        result = renderer.render_gold(sample_pipeline_config)
        encode_fn = result[result.index("def _encode_one_hot") :]
        encode_fn = encode_fn[: encode_fn.index("\ndef ")]
        assert "raise RuntimeError" not in encode_fn
        assert "_encode_one_hot: skipping" in encode_fn
        assert "return df" in encode_fn

    def test_label_encode_checks_column_exists(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        encode_fn = result[result.index("def _label_encode") :]
        encode_fn = encode_fn[: encode_fn.index("\ndef ")]
        assert "col not in df.columns" in encode_fn

    def test_label_encode_failsafe_when_target_missing(self, renderer, sample_pipeline_config):
        """FW-5 §2.8: same fail-safe path for label encoding."""
        result = renderer.render_gold(sample_pipeline_config)
        encode_fn = result[result.index("def _label_encode") :]
        encode_fn = encode_fn[: encode_fn.index("\ndef ")]
        assert "raise RuntimeError" not in encode_fn
        assert "_label_encode: skipping" in encode_fn

    def test_label_encode_uses_spark_sql_broadcast_join(self, renderer, sample_pipeline_config):
        """FW-5 §2.9: replace pyspark.ml.StringIndexer with a Spark-SQL
        broadcast-join body. Coding_Practices.md:112 forbids ml estimators
        for data prep — StringIndexer.fit/transform overflows the Spark
        Connect ML cache (10 GB cap) on multi-million-row x ~1.2K-col silver."""
        result = renderer.render_gold(sample_pipeline_config)
        encode_fn = result[result.index("def _label_encode") :]
        encode_fn = encode_fn[: encode_fn.index("\ndef ")]
        # ML estimator path must be gone.
        assert "StringIndexer" not in encode_fn
        assert "indexer.fit(df)" not in encode_fn
        # SQL broadcast-join path must be present.
        assert "F.broadcast(_lookup)" in encode_fn
        assert "createDataFrame" in encode_fn
        # Output column name preserved for parity with exploration.
        assert "withColumnRenamed" in encode_fn

    def test_batch_scale_standard_still_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)


class TestFW5TrainingGuards:
    """FW-5: codegen-time training guards retiring NB10 §2.8, §2.9, §2.10."""

    def test_label_encode_no_pyspark_ml_import(self, renderer, sample_pipeline_config):
        # FW-5 §2.9: the rendered gold script no longer pulls anything from
        # pyspark.ml.feature for label encoding. The entire ml estimator
        # path is retired in favor of a Spark-SQL broadcast-join body.
        result = renderer.render_gold(sample_pipeline_config)
        assert "from pyspark.ml.feature import" not in result
        assert "indexer.fit(df).transform" not in result

    def test_label_encode_drops_when_no_non_null_values(self, renderer, sample_pipeline_config):
        """§2.9 edge case: an all-null categorical column would otherwise
        emit an empty lookup DataFrame; the broadcast-join body short-circuits
        and drops the column."""
        result = renderer.render_gold(sample_pipeline_config)
        encode_fn = result[result.index("def _label_encode") :]
        encode_fn = encode_fn[: encode_fn.index("\ndef ")]
        assert "if not _vals:" in encode_fn
        assert "return df.drop(col)" in encode_fn

    def test_training_emits_target_resolver_helper(self, renderer, sample_pipeline_config):
        """§2.10: the training script defines a runtime resolver that
        reconciles the configured TARGET_COLUMN literal against gold's
        actual schema. The literal can drift from gold reality through
        landing original_target rename misses, case mismatch, or silver
        prefix collisions."""
        result = renderer.render_training(sample_pipeline_config)
        assert "def _resolve_target_column(df_columns, requested):" in result

    def test_training_resolver_has_priority_ladder(self, renderer, sample_pipeline_config):
        """The resolver checks: exact > case-insensitive > original_<target>
        prefix > substring > fuzzy."""
        result = renderer.render_training(sample_pipeline_config)
        resolver = result[result.index("def _resolve_target_column") :]
        resolver = resolver[: resolver.index("\ndef ")]
        assert "if requested in df_columns:" in resolver
        assert "case-insensitive" in resolver or "_lower[requested.lower()]" in resolver
        assert '_orig = f"original_{requested}"' in resolver
        assert "difflib.get_close_matches" in resolver

    def test_training_resolver_raises_with_label_shaped_diagnostic(
        self, renderer, sample_pipeline_config,
    ):
        """0 candidates -> hard fail with label-shaped column dump.
        Auto-picking from a label-shaped tie would silently train on the
        wrong column and ship a model scored against the wrong label."""
        result = renderer.render_training(sample_pipeline_config)
        resolver = result[result.index("def _resolve_target_column") :]
        resolver = resolver[: resolver.index("\ndef ")]
        assert "Label-shaped cols" in resolver
        assert "_label_shaped" in resolver
        assert "raise RuntimeError" in resolver

    def test_training_resolver_called_before_target_filter(
        self, renderer, sample_pipeline_config,
    ):
        """The resolver must run BEFORE df.filter(F.col(TARGET).isNotNull())
        — otherwise UNRESOLVED_COLUMN aborts training with a literal
        TARGET that does not exist on gold."""
        result = renderer.render_training(sample_pipeline_config)
        train_fn = result[result.index("def train_and_evaluate"):]
        resolve_pos = train_fn.index("TARGET = _resolve_target_column(df.columns, TARGET_COLUMN)")
        filter_pos = train_fn.index("df.filter(F.col(TARGET).isNotNull())")
        assert resolve_pos < filter_pos

    def test_training_rebinds_exclude_cols_after_resolve(
        self, renderer, sample_pipeline_config,
    ):
        """`_EXCLUDE_COLS` is built at module-load time using the unresolved
        TARGET literal. After resolution, it must be rebuilt with the
        resolved name, otherwise feature_cols would treat the resolved
        target as a feature."""
        result = renderer.render_training(sample_pipeline_config)
        train_fn = result[result.index("def train_and_evaluate"):]
        train_fn = train_fn[: train_fn.index("\ndef ")] if "\ndef " in train_fn else train_fn
        assert "global TARGET, _EXCLUDE_COLS" in train_fn
        assert "_EXCLUDE_COLS = {TARGET, TIMESTAMP_COLUMN, ENTITY_KEY} | GOLD_METADATA_COLUMNS" in train_fn

    def test_training_with_fw5_guards_is_valid_python(
        self, renderer, sample_pipeline_config,
    ):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestDatabricksGoldFeatureStoreRegistration:
    def test_gold_casts_timestamp_ntz_before_write(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def run_gold") :]
        cast_pos = fn.index("_cast_timestamp_ntz_to_timestamp")
        save_pos = fn.index("saveAsTable")
        assert cast_pos < save_pos

    def test_gold_cast_function_uses_timestamp_type(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _cast_timestamp_ntz_to_timestamp") :]
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
        fn = result[result.index("def _register_feature_table") :]
        assert "PRIMARY KEY" in fn
        assert "SET NOT NULL" in fn
        assert "TIMESERIES" in fn

    def test_gold_timeseries_in_pk_clause(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _register_feature_table") :]
        assert "TIMESERIES" in fn
        assert "TIMESTAMP_COLUMN" in fn

    def test_gold_timestamp_in_primary_keys(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _register_feature_table") :]
        assert 'pk = ["entity_id", TIMESTAMP_COLUMN] if has_ts' in fn

    def test_gold_registration_after_save(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        run_gold_fn = result[result.index("def run_gold") :]
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
        fn = result[result.index("def _register_feature_table") :]
        not_null_pos = fn.index("SET NOT NULL")
        pk_pos = fn.index("PRIMARY KEY")
        assert not_null_pos < pk_pos

    def test_gold_pk_constraint_handles_existing(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _register_feature_table") :]
        assert "already exists" in fn

    def test_gold_no_rdd_access(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert ".rdd" not in result

    def test_gold_reads_back_from_delta_after_save(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def run_gold") :]
        assert "del df" in fn
        assert "saved = spark.table(output_table)" in fn
        assert "return saved" in fn

    def test_gold_no_triple_materialization(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def run_gold") :]
        after_save = fn[fn.index("saveAsTable") :]
        lines_with_df_ref = [
            line.strip()
            for line in after_save.splitlines()
            if "df" in line
            and "del df" not in line
            and "reg_df" not in line
            and "saved" not in line
            and line.strip()
            and not line.strip().startswith("#")
            and "df.schema" not in line
        ]
        assert not lines_with_df_ref, f"Stale df references after saveAsTable: {lines_with_df_ref}"

    def test_gold_display_is_bounded(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert ".limit(20)" in result and "display(result" in result

    def test_gold_one_hot_has_cardinality_guard(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        fn = result[result.index("def _encode_one_hot") :]
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
        train_fn = result[result.index("def train_and_evaluate") :]
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

    def test_training_smote_uses_explicit_schema(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig

        silver = SilverLayerConfig(joins=silver_with_join.joins, aggregations=[])
        gold = GoldLayerConfig(encodings=gold_with_encode_scale.encodings, scalings=gold_with_encode_scale.scalings)
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
            training=TrainingConfig(imbalance_strategy="smote"),
        )
        result = renderer.render_training(config)
        smote_section = result[result.index("SMOTE") :]
        assert (
            "createDataFrame(resampled_pdf, schema=" in smote_section
            or "createDataFrame(resampled_pdf, _vector_schema" in smote_section
        )
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
        fn = result[result.index("def train_and_evaluate") :]
        load_pos = fn.index("load_training_data()")
        assert_pos = fn.index("_assert_rows(")
        assert load_pos < assert_pos

    def test_training_asserts_after_null_filter(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        filter_pos = fn.index("isNotNull")
        remaining = fn[filter_pos:]
        assert "_assert_rows(" in remaining

    def test_training_logs_label_distribution(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'groupBy("label")' in result
        assert "Label distribution" in result

    def test_training_no_pandas_roundtrip_for_splitting(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert "createDataFrame(train_pdf" not in fn
        assert "createDataFrame(test_pdf" not in fn

    def test_training_per_model_timing(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "log_timing" in result
        fn = result[result.index("for name, model in") :]
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
        assert "GOLD_METADATA_COLUMNS" in result
        assert "_EXCLUDE_COLS" in result

    def test_training_checks_minimum_two_classes(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "Need at least 2" in result

    def test_training_feature_types_used_in_prepare_features(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        prepare_fn = result[result.index("def prepare_features") :]
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
        fn = result[result.index("def train_and_evaluate") :]
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
            exploration_feature_profile={
                "stage": "exploration",
                "created_at": "2024-01-01",
                "row_count": 100,
                "feature_count": 2,
                "target_column": "churn",
                "features": {"col_a": {"dtype": "double", "non_null": 90, "null_count": 10}},
                "excluded": {},
            },
        )
        result = renderer.render_training(config)
        assert 'exploration_profile.json' in result
        assert 'json.load(_ep_f)' in result
        assert "FeatureProfile.from_dict(_EXPLORATION_PROFILE)" in result
        assert "compare_feature_profiles(" in result
        assert "feature discrepancies" in result

    def test_training_profile_uses_batched_null_agg(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert "F.sum(F.when(F.col(c).isNull()" in fn

    def test_training_profile_timing(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'log_timing("feature_profile"' in result

    def test_training_profile_is_valid_python(self, renderer, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig

        config = sample_pipeline_config
        config.training = TrainingConfig(
            exploration_feature_profile={
                "stage": "exploration",
                "created_at": "2024-01-01",
                "row_count": 100,
                "feature_count": 2,
                "target_column": "churn",
                "features": {"col_a": {"dtype": "double", "non_null": 90, "null_count": 10}},
                "excluded": {},
            },
        )
        result = renderer.render_training(config)
        ast.parse(result)

    # NaN/Inf preservation is now handled by the JSON sidecar (json.dumps
    # allow_nan=True / json.load default-accepts NaN). The renderer no
    # longer embeds an inline _EXPLORATION_PROFILE literal, so the
    # historic "rendered source contains float('nan')" tests are obsolete.
    # Equivalent end-to-end coverage lives in
    # test_databricks_pipeline_generator.py::TestExplorationProfileExternalisation
    # — the sidecar JSON round-trips NaN values through Python's json module.


class TestDatabricksTrainingProfilePersistence:
    def test_training_imports_run_namespace(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace" in result

    def test_training_constructs_namespace_from_widgets(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert "dbutils.widgets.get(" in fn or "_NAMESPACE" in result

    def test_training_saves_production_profile(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert "prod_profile.save(" in fn
        assert "production_feature_profile_path" in fn

    def test_training_profile_save_is_valid_python(self, renderer, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig

        config = sample_pipeline_config
        config.training = TrainingConfig(
            exploration_feature_profile={
                "stage": "exploration",
                "created_at": "2024-01-01",
                "row_count": 100,
                "feature_count": 2,
                "target_column": "churn",
                "features": {"col_a": {"dtype": "double", "non_null": 90, "null_count": 10}},
                "excluded": {},
            },
        )
        result = renderer.render_training(config)
        ast.parse(result)


class TestDatabricksTrainingDistributedSplit:
    def test_no_topandas_for_splitting(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        split_section = result[result.index("def train_and_evaluate") :]
        if "toPandas" in split_section:
            topandas_pos = split_section.index("toPandas")
            context = split_section[max(0, topandas_pos - 200) : topandas_pos + 100]
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
        fn = result[result.index("def train_and_evaluate") :]
        assert "filter" in fn.lower() or "where" in fn.lower()

    def test_purge_gap_when_configured(
        self, renderer, entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
    ):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig

        silver = SilverLayerConfig(joins=silver_with_join.joins, aggregations=[])
        gold = GoldLayerConfig(encodings=gold_with_encode_scale.encodings, scalings=gold_with_encode_scale.scalings)
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
            training=TrainingConfig(purge_gap_days=14),
        )
        result = renderer.render_training(config)
        assert "14" in result
        assert (
            "purge" in result.lower()
            or "gap" in result.lower()
            or "timedelta" in result.lower()
            or "days" in result.lower()
        )

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
        importance_section = result[result.index("featureImportances") :]
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
        fn = result[result.index("def train_and_evaluate") :]
        set_tag_pos = fn.find("mlflow.set_tag(")
        start_run_pos = fn.find("with mlflow.start_run(")
        assert start_run_pos < set_tag_pos, "set_tag must be inside a start_run context"

    def test_no_standalone_best_model_run(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert 'start_run(run_name="best_model")' not in result

    def test_best_model_logged_to_parent_run(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert 'mlflow.set_tag("best_model"' in fn
        assert 'mlflow.log_metric("best_roc_auc"' in fn

    def test_parent_run_logs_best_model_full_metrics(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert "best_metrics" in fn
        assert 'mlflow.log_metrics({f"best_{k}"' in fn or "best_metrics" in fn

    def test_parent_run_tags_target_column(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert 'mlflow.set_tag("target_column", TARGET)' in fn

    def test_parent_run_tags_entity_key(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert 'mlflow.set_tag("entity_key", "entity_id")' in fn

    def test_parent_run_tags_timestamp_column(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        assert 'mlflow.set_tag("timestamp_column", TIMESTAMP_COLUMN)' in fn

    def test_parent_run_tags_recommendations_hash(self, renderer, sample_pipeline_config):
        sample_pipeline_config.recommendations_hash = "abc123"
        result = renderer.render_training(sample_pipeline_config)
        assert 'mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)' in result

    def test_disables_autolog_before_explicit_tracking(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        autolog_pos = fn.index("mlflow.autolog(disable=True)")
        start_run_pos = fn.index("mlflow.start_run(")
        assert autolog_pos < start_run_pos

    def test_ends_stale_active_run(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
        end_run_pos = fn.index("mlflow.end_run()")
        start_run_pos = fn.index("mlflow.start_run(")
        assert end_run_pos < start_run_pos

    def test_experiment_name_uses_composite_name(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "COMPOSITE_NAME" in result.split("set_experiment")[1].split(")")[0]

    def test_parent_run_name_uses_composite_name(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        parent_line = [
            line for line in result.splitlines() if "with mlflow.start_run(run_name=" in line and "nested" not in line
        ][0]
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
        fn = result[result.index("def train_and_evaluate") :]
        parent_block = fn[fn.index("with mlflow.start_run(") : fn.index("return _results")]
        assert "features.json" in parent_block

    def test_feature_list_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)

    def test_feature_list_in_namespace_metadata(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("_training_meta") :]
        assert '"feature_columns"' in fn


class TestDatabricksTrainingMetadataPersistence:
    def test_writes_training_metadata_to_namespace(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "training_metadata_path" in result
        assert "json.dumps" in result or "json.dump" in result

    def test_metadata_includes_experiment_name(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '"mlflow_experiment_name"' in result

    def test_metadata_includes_pipeline_name(self, renderer, sample_pipeline_config):
        # pipeline_name is the NB10 folder name (e.g. "customer_churn"), distinct
        # from mlflow_experiment_name which is a workspace path like
        # "/Shared/training_<cn>". Downstream code (c01 pipeline-runner) needs
        # pipeline_name to locate generated_pipelines/databricks/<pipeline_name>.
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("_training_meta") :]
        assert '"pipeline_name": PIPELINE_NAME' in fn

    def test_metadata_includes_run_id(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert '"mlflow_run_id"' in result

    def test_metadata_includes_scoring_fields(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        fn = result[result.index("def train_and_evaluate") :]
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


class TestFrameworkRepoPathInRenderer:
    def test_config_includes_sys_path_when_repo_path_set(self, sample_pipeline_config):
        renderer = DatabricksCodeRenderer(
            catalog="ml_catalog",
            schema="retention",
            framework_repo_path="/Workspace/Repos/me/churnkit",
        )
        result = renderer.render_config(sample_pipeline_config)
        assert "import sys" in result
        assert 'FRAMEWORK_REPO_ROOT = "/Workspace/Repos/me/churnkit"' in result
        assert "{FRAMEWORK_REPO_ROOT}/src" in result
        assert "sys.path.insert(0, _src)" in result
        ast.parse(result)

    def test_config_no_sys_path_when_repo_path_none(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "FRAMEWORK_REPO_ROOT" not in result

    def test_sys_path_is_first_code_cell(self, sample_pipeline_config):
        renderer = DatabricksCodeRenderer(
            catalog="ml_catalog",
            schema="retention",
            framework_repo_path="/Workspace/Repos/me/churnkit",
        )
        result = renderer.render_config(sample_pipeline_config)
        lines = result.splitlines()
        code_start = next(
            i for i, line in enumerate(lines) if line.strip() and not line.startswith("#") and "MAGIC" not in line
        )
        assert lines[code_start] == "import sys"


class TestDatabricksLifecycleQuadrantParityWithNB01d:
    """Regression for the 3rd-pass parity failure (`gold missing lifecycle_quadrant_occasional_loyal_lifecycle,
    lifecycle_quadrant_one_shot_lifecycle`). NB01d's `classify_lifecycle_quadrants`
    computes tenure from `duration_days = last_event - first_event` (equivalent
    to `days_since_first - days_since_last`) and intensity as `events-per-day`.
    The renderer's `add_lifecycle_quadrant` MUST use the same quantities;
    otherwise the median thresholds land in different places on the same data
    and two of four quadrants end up with zero entities."""

    @staticmethod
    def _render_bronze_entity_with_quadrant(renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(include_lifecycle_quadrant=True),
            post_shaping=[],
        )
        return renderer.render_bronze_entity("orders_aggregated", config, "orders", "orders")

    def test_uses_duration_days_not_raw_days_since_first(self, renderer):
        result = self._render_bronze_entity_with_quadrant(renderer)
        fn = result[result.index("def add_lifecycle_quadrant") :]
        fn = fn[: fn.index("\ndef ")]
        assert "days_since_first\") - F.col(\"days_since_last\")" in fn, (
            "must compute duration_days = days_since_first - days_since_last "
            "to match NB01d's `lc['duration_days']`"
        )

    def test_uses_events_per_day_intensity(self, renderer):
        result = self._render_bronze_entity_with_quadrant(renderer)
        fn = result[result.index("def add_lifecycle_quadrant") :]
        fn = fn[: fn.index("\ndef ")]
        assert "F.greatest(F.col(\"_lifecycle_duration_days\")" in fn
        assert "/ F.greatest" in fn, (
            "must divide event_count by duration to match NB01d's "
            "`intensity = event_count / duration_days.clip(lower=1)`"
        )

    def test_prefers_event_count_all_time(self, renderer):
        result = self._render_bronze_entity_with_quadrant(renderer)
        fn = result[result.index("def add_lifecycle_quadrant") :]
        fn = fn[: fn.index("\ndef ")]
        assert "event_count_all_time" in fn

    def test_batched_approx_quantile_single_job(self, renderer):
        """Both medians computed in one Spark job via list-arg approxQuantile —
        avoids doubling the plan cost on large gold tables."""
        result = self._render_bronze_entity_with_quadrant(renderer)
        fn = result[result.index("def add_lifecycle_quadrant") :]
        fn = fn[: fn.index("\ndef ")]
        assert "approxQuantile(\n        [\"_lifecycle_duration_days\", \"_lifecycle_intensity\"]" in fn

    def test_all_four_quadrant_labels_emitted(self, renderer):
        result = self._render_bronze_entity_with_quadrant(renderer)
        fn = result[result.index("def add_lifecycle_quadrant") :]
        fn = fn[: fn.index("\ndef ")]
        for label in ("steady_loyal_lifecycle", "occasional_loyal_lifecycle",
                       "intense_brief_lifecycle", "one_shot_lifecycle"):
            assert label in fn

    def test_scratch_columns_cleaned_up(self, renderer):
        """The `_lifecycle_*` helper columns must not leak into the gold table
        — they'd pollute the feature space and potentially break downstream
        joins on selected_features."""
        result = self._render_bronze_entity_with_quadrant(renderer)
        fn = result[result.index("def add_lifecycle_quadrant") :]
        fn = fn[: fn.index("\ndef ")]
        assert 'drop("_lifecycle_duration_days", "_lifecycle_intensity")' in fn

    def test_bronze_entity_notebook_is_valid_python(self, renderer):
        result = self._render_bronze_entity_with_quadrant(renderer)
        ast.parse(result)


class TestDatabricksTrainingSpecSnapshot:
    """The production training template must snapshot the FeatureSpec into
    `_NAMESPACE.feature_spec_path` whenever the hardcoded `_FEATURE_SPEC_PATH`
    differs. Without this, NB11's parity report (which reads from the current
    run's namespace) raises `FileNotFoundError: FeatureSpec not found ...` when
    exploration and production ran in different run_ids."""

    @pytest.fixture
    def config_with_feature_spec(self, sample_pipeline_config):
        sample_pipeline_config.feature_spec_path = "/some/exploration/run/merged/feature_spec.yaml"
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        sample_pipeline_config.training = TrainingConfig(feature_spec_path=sample_pipeline_config.feature_spec_path)
        return sample_pipeline_config

    def test_snapshots_spec_into_current_namespace(self, renderer, config_with_feature_spec):
        result = renderer.render_training(config_with_feature_spec)
        fn = result[result.index("def train_and_evaluate") :]
        assert "_runtime_spec_path = _NAMESPACE.feature_spec_path" in fn
        assert "shutil" in fn
        assert "copy2(_FEATURE_SPEC_PATH, _runtime_spec_path)" in fn

    def test_snapshot_skips_when_paths_match(self, renderer, config_with_feature_spec):
        """Idempotent: when the baked path already equals the runtime path, no copy."""
        result = renderer.render_training(config_with_feature_spec)
        fn = result[result.index("def train_and_evaluate") :]
        assert "_runtime_spec_path != _FEATURE_SPEC_PATH" in fn

    def test_snapshot_skips_when_destination_exists(self, renderer, config_with_feature_spec):
        """Avoid overwriting a local spec that may have been edited in the current run."""
        result = renderer.render_training(config_with_feature_spec)
        fn = result[result.index("def train_and_evaluate") :]
        assert "not _runtime_spec_path.exists()" in fn

    def test_prod_diag_records_spec_source_path(self, renderer, config_with_feature_spec):
        """NB11's fallback reads `feature_spec_source_path` from prod_diag to
        locate the original spec when the current run is missing a copy."""
        result = renderer.render_training(config_with_feature_spec)
        assert '"feature_spec_source_path"' in result
        assert "str(_FEATURE_SPEC_PATH)" in result

    def test_rendered_training_is_valid_python(self, renderer, config_with_feature_spec):
        result = renderer.render_training(config_with_feature_spec)
        ast.parse(result)


class TestDatabricksTrainingFullPanelFit:
    @staticmethod
    def _with_full_panel(config):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        config.training = config.training or TrainingConfig()
        config.training.production_full_panel_fit = True
        return config

    def test_full_panel_fit_skips_temporal_split_call(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        assert "train_df, test_df, cutoff_date = _temporal_split(" not in result

    def test_full_panel_fit_binds_train_to_assembled(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        assert "    train_df = assembled\n" in result
        assert "    test_df = assembled\n" in result
        assert "Full-panel fit:" in result

    def test_full_panel_fit_tags_mlflow_mode(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        assert 'mlflow.set_tag("training_mode", "full_panel_fit")' in result

    def test_temporal_split_default_when_not_full_panel(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "train_df, test_df, cutoff_date = _temporal_split(" in result
        assert 'mlflow.set_tag("training_mode", "temporal_split")' in result
        assert "    train_df = assembled\n" not in result

    def test_full_panel_fit_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        ast.parse(result)


class TestDatabricksRecencyBucketLabelsMatchNB01d:
    """The recency_bucket chain writes string labels that later feed one-hot
    encoding. If the labels disagree with NB01d's (which the FeatureSpec
    captures), encoded column names don't match and training fails."""

    @staticmethod
    def _render_bronze_entity_with_recency(renderer):
        source = SourceConfig(
            name="orders", path="/data/orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="order_date",
            lifecycle=LifecycleConfig(include_recency_bucket=True),
            post_shaping=[],
        )
        return renderer.render_bronze_entity("orders_aggregated", config, "orders", "orders")

    def test_default_labels_match_nb01d_canonical(self, renderer):
        result = self._render_bronze_entity_with_recency(renderer)
        fn = result[result.index("def add_recency_buckets") :]
        fn = fn[: fn.index("\ndef ")]
        for label in ('"0-7d"', '"8-30d"', '"31-90d"', '"91-180d"', '">180d"'):
            assert label in fn, f"canonical label {label} missing from renderer output"
        for wrong_label in ('"7-30d"', '"30-90d"', '"90-180d"', '"180-365d"', '"365d+"'):
            assert wrong_label not in fn, (
                f"non-canonical label {wrong_label} leaked into renderer output"
            )
