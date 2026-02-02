

class TestPipelineTransformationType:
    def test_transformation_type_has_impute_null(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        assert PipelineTransformationType.IMPUTE_NULL.value == "impute_null"

    def test_transformation_type_has_cap_outlier(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        assert PipelineTransformationType.CAP_OUTLIER.value == "cap_outlier"

    def test_transformation_type_has_type_cast(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        assert PipelineTransformationType.TYPE_CAST.value == "type_cast"

    def test_transformation_type_has_encode(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        assert PipelineTransformationType.ENCODE.value == "encode"

    def test_transformation_type_has_scale(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        assert PipelineTransformationType.SCALE.value == "scale"

    def test_transformation_type_has_aggregate(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        assert PipelineTransformationType.AGGREGATE.value == "aggregate"

    def test_transformation_type_has_join(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        assert PipelineTransformationType.JOIN.value == "join"


class TestSourceConfig:
    def test_source_config_creation(self):
        from customer_retention.generators.pipeline_generator.models import SourceConfig
        config = SourceConfig(name="customers", path="/data/customers.csv", format="csv", entity_key="customer_id")
        assert config.name == "customers"
        assert config.path == "/data/customers.csv"
        assert config.format == "csv"
        assert config.entity_key == "customer_id"

    def test_source_config_optional_fields(self):
        from customer_retention.generators.pipeline_generator.models import SourceConfig
        config = SourceConfig(name="events", path="/data/events.parquet", format="parquet",
                              entity_key="customer_id", time_column="event_time", is_event_level=True)
        assert config.time_column == "event_time"
        assert config.is_event_level is True

    def test_source_config_default_values(self):
        from customer_retention.generators.pipeline_generator.models import SourceConfig
        config = SourceConfig(name="test", path="/test", format="csv", entity_key="id")
        assert config.time_column is None
        assert config.is_event_level is False


class TestTransformationStep:
    def test_transformation_step_creation(self):
        from customer_retention.generators.pipeline_generator.models import (
            PipelineTransformationType,
            TransformationStep,
        )
        step = TransformationStep(type=PipelineTransformationType.IMPUTE_NULL, column="age",
                                  parameters={"value": 0}, rationale="Fill missing ages")
        assert step.type == PipelineTransformationType.IMPUTE_NULL
        assert step.column == "age"
        assert step.parameters == {"value": 0}
        assert step.rationale == "Fill missing ages"

    def test_transformation_step_with_cap_outlier(self):
        from customer_retention.generators.pipeline_generator.models import (
            PipelineTransformationType,
            TransformationStep,
        )
        step = TransformationStep(type=PipelineTransformationType.CAP_OUTLIER, column="income",
                                  parameters={"lower": 0, "upper": 1000000}, rationale="Cap extreme incomes")
        assert step.parameters["lower"] == 0
        assert step.parameters["upper"] == 1000000


class TestBronzeLayerConfig:
    def test_bronze_layer_config_creation(self):
        from customer_retention.generators.pipeline_generator.models import BronzeLayerConfig, SourceConfig
        source = SourceConfig(name="customers", path="/data/customers.csv", format="csv", entity_key="id")
        config = BronzeLayerConfig(source=source)
        assert config.source.name == "customers"
        assert config.transformations == []

    def test_bronze_layer_config_with_transformations(self):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            PipelineTransformationType,
            SourceConfig,
            TransformationStep,
        )
        source = SourceConfig(name="test", path="/test", format="csv", entity_key="id")
        step = TransformationStep(type=PipelineTransformationType.IMPUTE_NULL, column="col", parameters={"value": 0}, rationale="test")
        config = BronzeLayerConfig(source=source, transformations=[step])
        assert len(config.transformations) == 1


class TestSilverLayerConfig:
    def test_silver_layer_config_creation(self):
        from customer_retention.generators.pipeline_generator.models import SilverLayerConfig
        config = SilverLayerConfig()
        assert config.joins == []
        assert config.aggregations == []

    def test_silver_layer_config_with_joins(self):
        from customer_retention.generators.pipeline_generator.models import SilverLayerConfig
        joins = [{"left_key": "id", "right_key": "customer_id", "right_source": "orders", "how": "left"}]
        config = SilverLayerConfig(joins=joins)
        assert len(config.joins) == 1
        assert config.joins[0]["how"] == "left"

    def test_silver_layer_config_with_aggregations(self):
        from customer_retention.generators.pipeline_generator.models import SilverLayerConfig
        aggs = [{"column": "amount", "function": "sum", "windows": ["7d", "30d"]}]
        config = SilverLayerConfig(aggregations=aggs)
        assert len(config.aggregations) == 1


class TestGoldLayerConfig:
    def test_gold_layer_config_creation(self):
        from customer_retention.generators.pipeline_generator.models import GoldLayerConfig
        config = GoldLayerConfig()
        assert config.encodings == []
        assert config.scalings == []
        assert config.feature_selections == []

    def test_gold_layer_config_with_encodings(self):
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineTransformationType,
            TransformationStep,
        )
        encoding = TransformationStep(type=PipelineTransformationType.ENCODE, column="category",
                                       parameters={"method": "one_hot"}, rationale="One-hot encode")
        config = GoldLayerConfig(encodings=[encoding])
        assert len(config.encodings) == 1

    def test_gold_layer_config_with_scalings(self):
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineTransformationType,
            TransformationStep,
        )
        scaling = TransformationStep(type=PipelineTransformationType.SCALE, column="amount",
                                     parameters={"method": "standard"}, rationale="Standardize")
        config = GoldLayerConfig(scalings=[scaling])
        assert len(config.scalings) == 1


class TestPipelineConfig:
    def test_pipeline_config_creation(self):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )
        source = SourceConfig(name="customers", path="/data/customers.csv", format="csv", entity_key="id")
        bronze = {"customers": BronzeLayerConfig(source=source)}
        silver = SilverLayerConfig()
        gold = GoldLayerConfig()
        config = PipelineConfig(name="test_pipeline", target_column="churn", sources=[source],
                                bronze=bronze, silver=silver, gold=gold, output_dir="/output")
        assert config.name == "test_pipeline"
        assert config.target_column == "churn"
        assert len(config.sources) == 1
        assert config.output_dir == "/output"

    def test_pipeline_config_multiple_sources(self):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )
        source1 = SourceConfig(name="customers", path="/c.csv", format="csv", entity_key="id")
        source2 = SourceConfig(name="orders", path="/o.csv", format="csv", entity_key="customer_id")
        bronze = {"customers": BronzeLayerConfig(source=source1), "orders": BronzeLayerConfig(source=source2)}
        config = PipelineConfig(name="multi", target_column="churn", sources=[source1, source2],
                                bronze=bronze, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir="/out")
        assert len(config.sources) == 2
        assert len(config.bronze) == 2


class TestTimestampCoalesceConfig:

    def test_construction_with_ordered_columns(self):
        from customer_retention.generators.pipeline_generator.models import TimestampCoalesceConfig
        cfg = TimestampCoalesceConfig(datetime_columns_ordered=["sent_date", "unsubscribe_date"])
        assert cfg.datetime_columns_ordered == ["sent_date", "unsubscribe_date"]
        assert cfg.output_column == "feature_timestamp"

    def test_custom_output_column(self):
        from customer_retention.generators.pipeline_generator.models import TimestampCoalesceConfig
        cfg = TimestampCoalesceConfig(datetime_columns_ordered=["a", "b"], output_column="ts")
        assert cfg.output_column == "ts"


class TestLabelTimestampConfig:

    def test_defaults(self):
        from customer_retention.generators.pipeline_generator.models import LabelTimestampConfig
        cfg = LabelTimestampConfig()
        assert cfg.label_column is None
        assert cfg.fallback_window_days == 180
        assert cfg.output_column == "label_timestamp"

    def test_custom_values(self):
        from customer_retention.generators.pipeline_generator.models import LabelTimestampConfig
        cfg = LabelTimestampConfig(label_column="unsub_date", fallback_window_days=90)
        assert cfg.label_column == "unsub_date"
        assert cfg.fallback_window_days == 90


class TestBronzeEventConfig:

    def test_construction_minimal(self):
        from customer_retention.generators.pipeline_generator.models import BronzeEventConfig, SourceConfig
        src = SourceConfig(name="emails", path="emails.csv", format="csv", entity_key="cid")
        cfg = BronzeEventConfig(source=src, entity_column="cid", time_column="sent_date")
        assert cfg.deduplicate is False
        assert cfg.pre_shaping == []
        assert cfg.aggregation is None
        assert cfg.lifecycle is None
        assert cfg.post_shaping == []
        assert cfg.raw_time_column is None

    def test_construction_full(self):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            LifecycleConfig,
            PipelineTransformationType,
            SourceConfig,
            TransformationStep,
        )
        src = SourceConfig(name="emails", path="emails.csv", format="csv", entity_key="cid")
        agg = AggregationWindowConfig(windows=["7d", "30d"], value_columns=["clicks"], agg_funcs=["sum"])
        lc = LifecycleConfig(include_recency_bucket=True)
        step = TransformationStep(
            type=PipelineTransformationType.CAP_OUTLIER, column="clicks",
            parameters={"lower": 0, "upper": 100}, rationale="cap",
        )
        cfg = BronzeEventConfig(
            source=src, entity_column="cid", time_column="sent_date",
            deduplicate=True, pre_shaping=[step], aggregation=agg,
            lifecycle=lc, post_shaping=[], raw_time_column="raw_ts",
        )
        assert cfg.deduplicate is True
        assert len(cfg.pre_shaping) == 1
        assert cfg.aggregation.windows == ["7d", "30d"]
        assert cfg.lifecycle.include_recency_bucket is True
        assert cfg.raw_time_column == "raw_ts"


class TestSlimmedLandingLayerConfig:

    def test_landing_has_no_aggregation_field(self):
        from customer_retention.generators.pipeline_generator.models import LandingLayerConfig, SourceConfig
        src = SourceConfig(name="e", path="e.csv", format="csv", entity_key="cid")
        cfg = LandingLayerConfig(
            source=src, raw_source_path="/data/e.csv", raw_source_format="csv",
            entity_column="cid", time_column="ts", target_column="target",
        )
        assert not hasattr(cfg, "deduplicate")
        assert not hasattr(cfg, "aggregation")
        assert not hasattr(cfg, "lifecycle")
        assert not hasattr(cfg, "pre_agg_transformations")

    def test_landing_has_timestamp_coalesce(self):
        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
            TimestampCoalesceConfig,
        )
        src = SourceConfig(name="e", path="e.csv", format="csv", entity_key="cid")
        coalesce = TimestampCoalesceConfig(datetime_columns_ordered=["a", "b"])
        cfg = LandingLayerConfig(
            source=src, raw_source_path="/data/e.csv", raw_source_format="csv",
            entity_column="cid", time_column="ts", target_column="target",
            timestamp_coalesce=coalesce,
        )
        assert cfg.timestamp_coalesce.datetime_columns_ordered == ["a", "b"]

    def test_landing_has_label_timestamp(self):
        from customer_retention.generators.pipeline_generator.models import (
            LabelTimestampConfig,
            LandingLayerConfig,
            SourceConfig,
        )
        src = SourceConfig(name="e", path="e.csv", format="csv", entity_key="cid")
        label = LabelTimestampConfig(label_column="unsub_date", fallback_window_days=90)
        cfg = LandingLayerConfig(
            source=src, raw_source_path="/data/e.csv", raw_source_format="csv",
            entity_column="cid", time_column="ts", target_column="target",
            label_timestamp=label,
        )
        assert cfg.label_timestamp.label_column == "unsub_date"

    def test_landing_defaults_none_for_optional_fields(self):
        from customer_retention.generators.pipeline_generator.models import LandingLayerConfig, SourceConfig
        src = SourceConfig(name="e", path="e.csv", format="csv", entity_key="cid")
        cfg = LandingLayerConfig(
            source=src, raw_source_path="/data/e.csv", raw_source_format="csv",
            entity_column="cid", time_column="ts", target_column="target",
        )
        assert cfg.timestamp_coalesce is None
        assert cfg.label_timestamp is None
        assert cfg.raw_time_column is None


class TestPipelineConfigBronzeEvent:

    def test_bronze_event_defaults_empty(self):
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )
        src = SourceConfig(name="c", path="c.csv", format="csv", entity_key="id")
        cfg = PipelineConfig(
            name="test", target_column="t", sources=[src],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir=".",
        )
        assert cfg.bronze_event == {}

    def test_bronze_event_populated(self):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )
        src = SourceConfig(name="e", path="e.csv", format="csv", entity_key="cid")
        be = BronzeEventConfig(source=src, entity_column="cid", time_column="ts")
        cfg = PipelineConfig(
            name="test", target_column="t", sources=[src],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(),
            output_dir=".", bronze_event={"events": be},
        )
        assert "events" in cfg.bronze_event
        assert cfg.bronze_event["events"].entity_column == "cid"
