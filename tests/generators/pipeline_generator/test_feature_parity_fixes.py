import ast

import pandas as pd
import pytest
import yaml

from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    GoldLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TrainingConfig,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


@pytest.fixture
def renderer():
    return DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")


@pytest.fixture
def local_renderer():
    return CodeRenderer()


@pytest.fixture
def training_pipeline_config(entity_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[entity_source],
        bronze={"customers": bronze_with_impute},
        bronze_event={},
        silver=SilverLayerConfig(joins=silver_with_join.joins),
        gold=gold_with_encode_scale,
        output_dir="/output/test",
        composite_name="cust__abc1234",
        training=TrainingConfig(split_strategy="temporal", temporal_column="event_timestamp"),
    )


@pytest.fixture
def binary_event_config():
    return BronzeEventConfig(
        source=SourceConfig(
            name="emails", path="/data/emails.csv", format="csv",
            entity_key="customer_id", time_column="sent_date", is_event_level=True,
        ),
        entity_column="customer_id",
        time_column="sent_date",
        aggregation=AggregationWindowConfig(
            windows=["7d", "30d"],
            value_columns=["send_hour"],
            agg_funcs=["sum", "mean", "max", "count"],
            categorical_columns=["opened", "clicked"],
            categorical_agg_funcs=["nunique", "mode"],
            binary_columns=["opened", "clicked"],
            binary_agg_funcs=["rate", "count", "any"],
        ),
    )


@pytest.fixture
def binary_pipeline_config(entity_source, bronze_with_impute, silver_with_join, binary_event_config):
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[entity_source, binary_event_config.source],
        bronze={"customers": bronze_with_impute},
        bronze_event={"emails": binary_event_config},
        silver=SilverLayerConfig(joins=silver_with_join.joins),
        gold=GoldLayerConfig(),
        output_dir="/output/test",
        composite_name="cust_emai__abc1234",
    )


class TestVectorToArrayTemplate:
    def test_training_imports_vector_to_array(self, renderer, training_pipeline_config):
        result = renderer.render_training(training_pipeline_config)
        assert "vector_to_array" in result

    def test_training_uses_vector_to_array_for_probability(self, renderer, training_pipeline_config):
        result = renderer.render_training(training_pipeline_config)
        assert 'vector_to_array(F.col("probability"))' in result
        assert ".getItem(1)" not in result

    def test_training_with_vector_to_array_is_valid_python(self, renderer, training_pipeline_config):
        result = renderer.render_training(training_pipeline_config)
        ast.parse(result)


class TestBinaryColumnAggregation:
    def test_aggregation_config_has_binary_columns_field(self):
        config = AggregationWindowConfig(binary_columns=["opened", "clicked"])
        assert config.binary_columns == ["opened", "clicked"]

    def test_aggregation_config_binary_agg_funcs_defaults(self):
        config = AggregationWindowConfig()
        assert config.binary_agg_funcs == ["rate", "count", "any"]

    def test_build_aggregation_routes_binary_to_both_lists(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        agg_parquet = str(findings_dir / "emails_agg.parquet")

        multi_dataset = {
            "datasets": {
                "emails_agg": {
                    "name": "emails_agg",
                    "findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                    "source_path": agg_parquet,
                    "granularity": "entity_level",
                    "row_count": 500,
                    "column_count": 3,
                    "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "emails_agg",
            "event_datasets": [],
            "excluded_datasets": [],
            "aggregation_windows": ["7d", "30d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        agg_findings = {
            "source_path": agg_parquet,
            "source_format": "parquet",
            "row_count": 500,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "emails_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/emails.csv",
            "source_format": "csv",
            "row_count": 10000,
            "column_count": 7,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "sent_date": {
                    "name": "sent_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "send_hour": {
                    "name": "send_hour",
                    "inferred_type": "numeric_discrete",
                    "confidence": 0.7,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "opened": {
                    "name": "opened",
                    "inferred_type": "binary",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "clicked": {
                    "name": "clicked",
                    "inferred_type": "binary",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["sent_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "feature_timestamp",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                "suggested_aggregations": ["7d", "30d"],
            },
        }
        (findings_dir / "emails_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "emails_agg" in config.bronze_event
        agg_config = config.bronze_event["emails_agg"].aggregation
        assert agg_config is not None
        assert "opened" in agg_config.binary_columns
        assert "clicked" in agg_config.binary_columns
        assert "opened" in agg_config.categorical_columns
        assert "clicked" in agg_config.categorical_columns

    def test_event_aggregated_columns_includes_binary_aggs(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        event_cfg = BronzeEventConfig(
            source=SourceConfig(
                name="emails", path="/data/emails.csv", format="csv",
                entity_key="customer_id", time_column="sent_date", is_event_level=True,
            ),
            entity_column="customer_id",
            time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d"],
                value_columns=["send_hour"],
                agg_funcs=["sum", "mean", "max", "count"],
                categorical_columns=["opened"],
                categorical_agg_funcs=["nunique", "mode"],
                binary_columns=["opened"],
                binary_agg_funcs=["rate", "count", "any"],
            ),
        )
        cols = FindingsParser._event_aggregated_columns(event_cfg)
        assert "opened_rate_7d" in cols
        assert "opened_count_7d" in cols
        assert "opened_any_7d" in cols
        assert "opened_nunique_7d" in cols
        assert "opened_mode_7d" in cols

    def test_databricks_bronze_event_renders_binary_agg_block(self, renderer, binary_event_config):
        result = renderer.render_bronze_event("emails", binary_event_config)
        assert "F.mean" in result or "F.avg" in result
        assert "F.sum" in result
        assert "F.max" in result

    def test_local_bronze_event_renders_binary_agg_block(self, local_renderer, binary_event_config):
        result = local_renderer.render_bronze_event("emails", binary_event_config)
        assert "binary_columns" in result.lower() or "rate" in result or "BINARY" in result or ".mean()" in result

    def test_binary_agg_template_is_valid_python(self, renderer, binary_event_config):
        result = renderer.render_bronze_event("emails", binary_event_config)
        ast.parse(result)


class TestTemporalMetadataRoundTrip:
    @pytest.fixture
    def temporal_event_setup(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "orders_agg": {
                    "name": "orders_agg",
                    "findings_path": str(findings_dir / "orders_agg_findings.yaml"),
                    "source_path": "/data/orders_agg.parquet",
                    "granularity": "entity_level",
                    "row_count": 500,
                    "column_count": 3,
                    "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "orders_agg",
            "event_datasets": [],
            "excluded_datasets": [],
            "aggregation_windows": ["7d", "30d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        agg_findings = {
            "source_path": "/data/orders_agg.parquet",
            "source_format": "parquet",
            "row_count": 500,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "orders_agg_findings.yaml").write_text(yaml.dump(agg_findings))
        return findings_dir

    def _write_preagg_with_temporal(self, findings_dir, temporal_patterns):
        preagg_findings = {
            "source_path": "/data/raw/orders.csv",
            "source_format": "csv",
            "row_count": 10000,
            "column_count": 5,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "order_date": {
                    "name": "order_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "amount": {
                    "name": "amount",
                    "inferred_type": "numeric_continuous",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "quantity": {
                    "name": "quantity",
                    "inferred_type": "numeric_discrete",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["order_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "feature_timestamp",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "orders_agg_findings.yaml"),
                "suggested_aggregations": ["7d", "30d"],
            },
        }
        if temporal_patterns is not None:
            preagg_findings["metadata"] = {"temporal_patterns": temporal_patterns}
        (findings_dir / "orders_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

    def test_temporal_metadata_produces_valid_config(self, temporal_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        self._write_preagg_with_temporal(temporal_event_setup, {
            "lag_features_computed": True,
            "lag_window_days": 30,
            "num_lags": 4,
            "lag_columns": ["amount", "quantity"],
            "lag_agg_funcs": ["sum", "mean", "count", "max"],
        })

        parser = FindingsParser(str(temporal_event_setup))
        config = parser.parse()

        assert "orders_agg" in config.bronze_event
        tf = config.bronze_event["orders_agg"].temporal_features
        assert tf is not None
        assert tf.lag_window_days == 30
        assert tf.num_lags == 4
        assert tf.lag_columns == ["amount", "quantity"]

    def test_temporal_config_reads_lag_agg_funcs(self, temporal_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        self._write_preagg_with_temporal(temporal_event_setup, {
            "lag_features_computed": True,
            "lag_window_days": 30,
            "num_lags": 4,
            "lag_columns": ["amount", "quantity"],
            "lag_agg_funcs": ["sum", "mean", "count", "max"],
        })

        parser = FindingsParser(str(temporal_event_setup))
        config = parser.parse()

        tf = config.bronze_event["orders_agg"].temporal_features
        assert tf is not None
        assert tf.lag_agg_funcs == ["sum", "mean", "count", "max"]
        assert "lagged_windows" in tf.feature_groups
        assert "velocity" in tf.feature_groups

    def test_temporal_metadata_missing_returns_none(self, temporal_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        self._write_preagg_with_temporal(temporal_event_setup, None)

        parser = FindingsParser(str(temporal_event_setup))
        config = parser.parse()

        assert "orders_agg" in config.bronze_event
        assert config.bronze_event["orders_agg"].temporal_features is None


class TestGoldTransformApplicator:
    def test_build_gold_steps_from_registry(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.gold_transform_applicator import (
            build_gold_steps,
        )

        registry = RecommendationRegistry()
        registry.init_gold("churn")
        registry.add_gold_transformation(
            "revenue", "log", {}, "skewed distribution", "04",
        )
        registry.add_gold_transformation(
            "orders", "zero_inflation_handling", {}, "zero inflated", "04",
        )
        steps = build_gold_steps(registry, {"revenue", "orders"})
        assert len(steps) == 2
        types = {s.type for s in steps}
        assert PipelineTransformationType.LOG_TRANSFORM in types
        assert PipelineTransformationType.ZERO_INFLATION_HANDLING in types

    def test_build_silver_derived_steps(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.gold_transform_applicator import (
            build_silver_derived_steps,
        )

        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_ratio(
            "avg_order_value", "total_amount", "order_count",
            "average order metric", "06",
        )
        steps = build_silver_derived_steps(registry, {"total_amount", "order_count"})
        assert len(steps) == 1
        assert steps[0].type == PipelineTransformationType.DERIVED_COLUMN
        assert steps[0].column == "avg_order_value"

    def test_build_gold_steps_skips_missing_columns(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.gold_transform_applicator import (
            build_gold_steps,
        )

        registry = RecommendationRegistry()
        registry.init_gold("churn")
        registry.add_gold_transformation(
            "revenue", "log", {}, "skewed", "04",
        )
        registry.add_gold_transformation(
            "orders", "zero_inflation_handling", {}, "zero inflated", "04",
        )
        steps = build_gold_steps(registry, {"revenue"})
        assert len(steps) == 1
        assert steps[0].column == "revenue"

    def test_executor_applies_gold_steps(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.gold_transform_applicator import (
            build_gold_steps,
        )
        from customer_retention.transforms import TransformExecutor

        registry = RecommendationRegistry()
        registry.init_gold("churn")
        registry.add_gold_transformation(
            "revenue", "log", {}, "skewed", "04",
        )
        steps = build_gold_steps(registry, {"revenue"})
        df = pd.DataFrame({"revenue": [10.0, 100.0, 1000.0], "churn": [0, 1, 0]})
        executor = TransformExecutor()
        result = executor.apply_all(df, steps)
        assert "revenue" in result.columns

    def test_executor_applies_fitted_gold_steps_with_artifact_store(self, tmp_path):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.gold_transform_applicator import (
            build_gold_steps,
        )
        from customer_retention.transforms import TransformExecutor
        from customer_retention.transforms.artifact_store import ArtifactStore

        registry = RecommendationRegistry()
        registry.init_gold("churn")
        registry.add_gold_transformation(
            "revenue", "yeo_johnson", {}, "skewed", "04",
        )
        registry.add_gold_transformation(
            "orders", "log", {}, "skewed", "04",
        )
        steps = build_gold_steps(registry, {"revenue", "orders"})
        df = pd.DataFrame({"revenue": [10.0, 100.0, 1000.0], "orders": [1.0, 5.0, 20.0], "churn": [0, 1, 0]})
        executor = TransformExecutor()
        store = ArtifactStore(tmp_path / "artifacts")
        result = executor.apply_all(df, steps, fit_mode=True, artifact_store=store)
        assert "revenue" in result.columns
        assert "orders" in result.columns
        assert store.has("revenue_power_transformer")

    def test_executor_fitted_gold_without_store_raises(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.gold_transform_applicator import (
            build_gold_steps,
        )
        from customer_retention.transforms import TransformExecutor

        registry = RecommendationRegistry()
        registry.init_gold("churn")
        registry.add_gold_transformation(
            "revenue", "yeo_johnson", {}, "skewed", "04",
        )
        steps = build_gold_steps(registry, {"revenue"})
        df = pd.DataFrame({"revenue": [10.0, 100.0, 1000.0], "churn": [0, 1, 0]})
        executor = TransformExecutor()
        with pytest.raises(AttributeError):
            executor.apply_all(df, steps)


class TestNullFillAfterOuterJoin:
    def test_databricks_event_agg_fills_nulls_after_merge(self, renderer, binary_event_config):
        result = renderer.render_bronze_event("emails", binary_event_config)
        merge_idx = result.find("how=\"outer\"")
        assert merge_idx != -1
        post_merge = result[merge_idx:]
        assert "fillna(0" in post_merge or "fillna({" in post_merge or "fill(0" in post_merge

    def test_local_event_agg_fills_nulls_after_concat(self, local_renderer, binary_event_config):
        result = local_renderer.render_bronze_event("emails", binary_event_config)
        concat_idx = result.find("pd.concat")
        if concat_idx == -1:
            concat_idx = result.find("concat(")
        assert concat_idx != -1
        post_concat = result[concat_idx:]
        assert "fillna(0" in post_concat or "fillna({" in post_concat

    def test_null_fill_targets_count_sum_rate_columns(self, renderer, binary_event_config):
        result = renderer.render_bronze_event("emails", binary_event_config)
        assert "count" in result.lower()
        assert "sum" in result.lower()
        merge_idx = result.find("how=\"outer\"")
        if merge_idx != -1:
            post_merge = result[merge_idx:]
            has_selective_fill = (
                "fillna(0" in post_merge
                or "fill(0" in post_merge
                or "_count_" in post_merge
                or "_sum_" in post_merge
                or "_rate_" in post_merge
            )
            assert has_selective_fill

    def test_null_fill_template_is_valid_python(self, renderer, binary_event_config):
        result = renderer.render_bronze_event("emails", binary_event_config)
        ast.parse(result)
