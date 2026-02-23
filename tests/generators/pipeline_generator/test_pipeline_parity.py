import ast

import pytest
import yaml

from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.findings_parser import (
    FindingsParser,
    _edges_to_labels,
)
from customer_retention.generators.pipeline_generator.models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    BronzeLayerConfig,
    GoldLayerConfig,
    LifecycleConfig,
    PipelineConfig,
    SilverLayerConfig,
    SourceConfig,
    TemporalFeatureConfig,
    TextFeatureConfig,
    TrainingConfig,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


@pytest.fixture
def entity_source():
    return SourceConfig(
        name="customers",
        path="/data/customers.csv",
        format="csv",
        entity_key="customer_id",
    )


@pytest.fixture
def event_source():
    return SourceConfig(
        name="orders",
        path="/data/orders.parquet",
        format="parquet",
        entity_key="customer_id",
        time_column="order_date",
        is_event_level=True,
    )


@pytest.fixture
def renderer():
    return CodeRenderer()


@pytest.fixture
def databricks_renderer():
    return DatabricksCodeRenderer(catalog="test_catalog", schema="test_schema")


@pytest.fixture
def lifecycle_all_features():
    return LifecycleConfig(
        include_lifecycle_quadrant=True,
        include_cyclical_features=True,
        include_recency_bucket=True,
        momentum_pairs=[{"short_window": "7d", "long_window": "30d"}],
        include_trend_features=True,
        include_cohort_features=True,
        include_month_cyclical=True,
        include_quarter_cyclical=True,
    )


@pytest.fixture
def bronze_config_with_lifecycle(entity_source, lifecycle_all_features):
    return BronzeLayerConfig(
        source=entity_source,
        transformations=[],
        lifecycle=lifecycle_all_features,
        entity_column="customer_id",
        time_column="order_date",
    )


@pytest.fixture
def bronze_event_config_with_lifecycle(event_source, lifecycle_all_features):
    return BronzeEventConfig(
        source=event_source,
        entity_column="customer_id",
        time_column="order_date",
        lifecycle=lifecycle_all_features,
        aggregation=AggregationWindowConfig(
            windows=["7d", "30d", "90d", "all_time"],
            value_columns=["amount"],
            agg_funcs=["sum", "mean"],
        ),
    )


@pytest.fixture
def pipeline_config_minimal(entity_source, event_source):
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[entity_source, event_source],
        bronze={"customers": BronzeLayerConfig(source=entity_source)},
        silver=SilverLayerConfig(),
        gold=GoldLayerConfig(),
        output_dir="/output",
        composite_name="test_cn",
    )


# ======================================================================
# Phase 1: Model Tests — Trend, Cohort & Extended Cyclical
# ======================================================================


class TestLifecycleConfigNewFields:
    def test_default_values(self):
        config = LifecycleConfig()
        assert config.include_trend_features is False
        assert config.include_cohort_features is False
        assert config.include_month_cyclical is False
        assert config.include_quarter_cyclical is False

    def test_fields_set_true(self):
        config = LifecycleConfig(
            include_trend_features=True,
            include_cohort_features=True,
            include_month_cyclical=True,
            include_quarter_cyclical=True,
        )
        assert config.include_trend_features is True
        assert config.include_cohort_features is True
        assert config.include_month_cyclical is True
        assert config.include_quarter_cyclical is True


# ======================================================================
# Phase 1: FindingsParser Tests
# ======================================================================


class TestLifecycleFeatureGroupsMapping:
    def _make_findings_dir(self, tmp_path, feature_groups=None, feature_flags=None, notes=None):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir(exist_ok=True)
        multi = {
            "datasets": {
                "orders": {
                    "name": "orders",
                    "findings_path": str(findings_dir / "orders_findings.yaml"),
                    "source_path": "/data/orders.parquet",
                    "granularity": "event_level",
                    "row_count": 100,
                    "column_count": 5,
                    "entity_column": "customer_id",
                    "time_column": "order_date",
                }
            },
            "relationships": [],
            "primary_entity_dataset": "orders",
            "event_datasets": ["orders"],
            "aggregation_windows": ["7d", "30d", "90d", "365d", "all_time"],
            "notes": notes or {},
        }
        if feature_groups:
            multi["notes"]["temporal_config"] = {"feature_groups": feature_groups}
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi))

        metadata = {}
        if feature_flags:
            metadata["temporal_patterns"] = {"feature_flags": feature_flags}

        orders_findings = {
            "source_path": "/data/orders.parquet",
            "source_format": "parquet",
            "row_count": 100,
            "column_count": 5,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "order_date": {
                    "name": "order_date",
                    "inferred_type": "datetime",
                    "confidence": 0.9,
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
                "churn": {
                    "name": "churn",
                    "inferred_type": "binary",
                    "confidence": 0.99,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
            "time_series_metadata": {
                "time_column": "order_date",
                "entity_column": "customer_id",
            },
            "metadata": metadata,
        }
        (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders_findings))
        return findings_dir

    def test_feature_groups_trend_mapped(self, tmp_path):
        findings_dir = self._make_findings_dir(tmp_path, feature_groups=["trend", "lifecycle", "recency"])
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        event_cfg = config.bronze_event["orders"]
        assert event_cfg.lifecycle.include_trend_features is True

    def test_feature_groups_cohort_mapped(self, tmp_path):
        findings_dir = self._make_findings_dir(tmp_path, feature_groups=["cohort", "lifecycle"])
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        event_cfg = config.bronze_event["orders"]
        assert event_cfg.lifecycle.include_cohort_features is True

    def test_feature_groups_regularity_implies_month_quarter(self, tmp_path):
        findings_dir = self._make_findings_dir(tmp_path, feature_groups=["regularity"])
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        event_cfg = config.bronze_event["orders"]
        assert event_cfg.lifecycle.include_month_cyclical is True
        assert event_cfg.lifecycle.include_quarter_cyclical is True
        assert event_cfg.lifecycle.include_cyclical_features is True

    def test_feature_flags_trend_fallback(self, tmp_path):
        findings_dir = self._make_findings_dir(tmp_path, feature_flags={"include_trend_features": True})
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        event_cfg = config.bronze_event["orders"]
        assert event_cfg.lifecycle.include_trend_features is True

    def test_feature_flags_cohort_fallback(self, tmp_path):
        findings_dir = self._make_findings_dir(tmp_path, feature_flags={"include_cohort_features": True})
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        event_cfg = config.bronze_event["orders"]
        assert event_cfg.lifecycle.include_cohort_features is True

    def test_seasonality_implies_month_quarter_cyclical(self, tmp_path):
        findings_dir = self._make_findings_dir(tmp_path, feature_flags={"include_seasonality_features": True})
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        event_cfg = config.bronze_event["orders"]
        assert event_cfg.lifecycle.include_month_cyclical is True
        assert event_cfg.lifecycle.include_quarter_cyclical is True


# ======================================================================
# Phase 1: Renderer Tests — Local
# ======================================================================


class TestLocalRendererTrendCohortCyclical:
    def test_bronze_entity_includes_month_sin(self, renderer, bronze_config_with_lifecycle):
        result = renderer.render_bronze("customers", bronze_config_with_lifecycle)
        assert "month_sin" in result
        assert "month_cos" in result

    def test_bronze_entity_includes_quarter_cyclical(self, renderer, bronze_config_with_lifecycle):
        result = renderer.render_bronze("customers", bronze_config_with_lifecycle)
        assert "quarter_sin" in result
        assert "quarter_cos" in result

    def test_bronze_entity_includes_trend_features(self, renderer, bronze_config_with_lifecycle):
        result = renderer.render_bronze("customers", bronze_config_with_lifecycle)
        assert "recent_vs_overall_ratio" in result
        assert "entity_trend_slope" in result

    def test_bronze_entity_includes_cohort_features(self, renderer, bronze_config_with_lifecycle):
        result = renderer.render_bronze("customers", bronze_config_with_lifecycle)
        assert "cohort_year" in result
        assert "cohort_quarter" in result

    def test_bronze_entity_absent_when_flags_false(self, renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            lifecycle=LifecycleConfig(include_recency_bucket=True),
            entity_column="customer_id",
            time_column="order_date",
        )
        result = renderer.render_bronze("customers", config)
        assert "month_sin" not in result
        assert "cohort_year" not in result
        assert "entity_trend_slope" not in result

    def test_bronze_entity_is_valid_python(self, renderer, bronze_config_with_lifecycle):
        result = renderer.render_bronze("customers", bronze_config_with_lifecycle)
        ast.parse(result)

    def test_bronze_event_entity_includes_new_features(self, renderer, bronze_event_config_with_lifecycle):
        result = renderer.render_bronze_entity("orders", bronze_event_config_with_lifecycle, "orders", "orders")
        assert "month_sin" in result
        assert "cohort_year" in result
        assert "entity_trend_slope" in result

    def test_bronze_event_entity_is_valid_python(self, renderer, bronze_event_config_with_lifecycle):
        result = renderer.render_bronze_entity("orders", bronze_event_config_with_lifecycle, "orders", "orders")
        ast.parse(result)


# ======================================================================
# Phase 1: Renderer Tests — Databricks
# ======================================================================


class TestDatabricksRendererTrendCohortCyclical:
    def test_databricks_bronze_includes_month_cyclical(self, databricks_renderer, bronze_config_with_lifecycle):
        result = databricks_renderer.render_bronze("customers", bronze_config_with_lifecycle)
        assert "month_sin" in result
        assert "month_cos" in result

    def test_databricks_bronze_includes_quarter_cyclical(self, databricks_renderer, bronze_config_with_lifecycle):
        result = databricks_renderer.render_bronze("customers", bronze_config_with_lifecycle)
        assert "quarter_sin" in result
        assert "quarter_cos" in result

    def test_databricks_bronze_includes_trend_features(self, databricks_renderer, bronze_config_with_lifecycle):
        result = databricks_renderer.render_bronze("customers", bronze_config_with_lifecycle)
        assert "recent_vs_overall_ratio" in result
        assert "entity_trend_slope" in result

    def test_databricks_bronze_includes_cohort_features(self, databricks_renderer, bronze_config_with_lifecycle):
        result = databricks_renderer.render_bronze("customers", bronze_config_with_lifecycle)
        assert "cohort_year" in result
        assert "cohort_quarter" in result

    def test_databricks_bronze_entity_includes_new_features(
        self, databricks_renderer, bronze_event_config_with_lifecycle
    ):
        result = databricks_renderer.render_bronze_entity(
            "orders", bronze_event_config_with_lifecycle, "orders", "orders"
        )
        assert "month_sin" in result
        assert "cohort_year" in result
        assert "entity_trend_slope" in result


# ======================================================================
# Phase 2: Model Tests — Configurable Recency Buckets
# ======================================================================


class TestRecencyBucketConfig:
    def test_default_edges(self):
        config = LifecycleConfig()
        assert config.recency_bucket_edges == [0, 7, 30, 90, 180, 365]

    def test_default_labels(self):
        config = LifecycleConfig()
        assert config.recency_bucket_labels == ["0-7d", "7-30d", "30-90d", "90-180d", "180-365d", "365d+"]

    def test_custom_edges_accepted(self):
        config = LifecycleConfig(
            recency_bucket_edges=[0, 14, 60, 120],
            recency_bucket_labels=["0-14d", "14-60d", "60-120d", "120d+"],
        )
        assert config.recency_bucket_edges == [0, 14, 60, 120]
        assert len(config.recency_bucket_labels) == 4


class TestEdgesToLabels:
    def test_standard_edges(self):
        labels = _edges_to_labels([0, 7, 30, 90, 180, 365])
        assert labels == ["0-7d", "7-30d", "30-90d", "90-180d", "180-365d", "365d+"]

    def test_custom_edges(self):
        labels = _edges_to_labels([0, 14, 60])
        assert labels == ["0-14d", "14-60d", "60d+"]


class TestRecencyBucketsFindingsParser:
    def test_custom_edges_from_temporal_config(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        multi = {
            "datasets": {
                "orders": {
                    "name": "orders",
                    "findings_path": str(findings_dir / "orders_findings.yaml"),
                    "source_path": "/data/orders.parquet",
                    "granularity": "event_level",
                    "row_count": 100,
                    "column_count": 5,
                    "entity_column": "customer_id",
                    "time_column": "order_date",
                }
            },
            "relationships": [],
            "primary_entity_dataset": "orders",
            "event_datasets": ["orders"],
            "aggregation_windows": ["7d", "30d", "all_time"],
            "notes": {
                "temporal_config": {
                    "feature_groups": ["recency"],
                    "recency_bucket_edges": [0, 14, 60, 120],
                }
            },
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi))
        orders = {
            "source_path": "/data/orders.parquet",
            "source_format": "parquet",
            "row_count": 100,
            "column_count": 5,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "order_date": {
                    "name": "order_date",
                    "inferred_type": "datetime",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "churn": {
                    "name": "churn",
                    "inferred_type": "binary",
                    "confidence": 0.99,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
            "time_series_metadata": {"time_column": "order_date", "entity_column": "customer_id"},
        }
        (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        lifecycle = config.bronze_event["orders"].lifecycle
        assert lifecycle.recency_bucket_edges == [0, 14, 60, 120]
        assert lifecycle.recency_bucket_labels == ["0-14d", "14-60d", "60-120d", "120d+"]


class TestRecencyBucketsRenderer:
    def test_local_renderer_uses_custom_bins(self, renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            lifecycle=LifecycleConfig(
                include_recency_bucket=True,
                recency_bucket_edges=[0, 14, 60],
                recency_bucket_labels=["0-14d", "14-60d", "60d+"],
            ),
            entity_column="customer_id",
            time_column="order_date",
        )
        result = renderer.render_bronze("customers", config)
        assert "14" in result
        assert "0-14d" in result
        ast.parse(result)

    def test_databricks_renderer_uses_custom_bins(self, databricks_renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            lifecycle=LifecycleConfig(
                include_recency_bucket=True,
                recency_bucket_edges=[0, 14, 60],
                recency_bucket_labels=["0-14d", "14-60d", "60d+"],
            ),
            entity_column="customer_id",
            time_column="order_date",
        )
        result = databricks_renderer.render_bronze("customers", config)
        assert "14" in result
        assert "0-14d" in result


# ======================================================================
# Phase 3: Model Tests — Temporal Feature Config
# ======================================================================


class TestTemporalFeatureConfig:
    def test_default_values(self):
        config = TemporalFeatureConfig()
        assert config.lag_window_days == 30
        assert config.num_lags == 4
        assert config.lag_columns == []
        assert config.lag_agg_funcs == ["sum", "mean", "count", "max"]
        assert config.feature_groups == ["lagged_windows", "velocity"]

    def test_custom_values(self):
        config = TemporalFeatureConfig(
            lag_window_days=14,
            num_lags=3,
            lag_columns=["amount"],
            feature_groups=["velocity"],
        )
        assert config.lag_window_days == 14
        assert config.num_lags == 3
        assert config.lag_columns == ["amount"]

    def test_bronze_event_has_temporal_features_field(self, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            temporal_features=TemporalFeatureConfig(lag_window_days=7),
        )
        assert config.temporal_features.lag_window_days == 7


class TestTemporalFeaturesParser:
    def test_temporal_features_from_notes(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        multi = {
            "datasets": {
                "orders": {
                    "name": "orders",
                    "findings_path": str(findings_dir / "orders_findings.yaml"),
                    "source_path": "/data/orders.parquet",
                    "granularity": "event_level",
                    "row_count": 100,
                    "column_count": 5,
                    "entity_column": "customer_id",
                    "time_column": "order_date",
                }
            },
            "relationships": [],
            "primary_entity_dataset": "orders",
            "event_datasets": ["orders"],
            "aggregation_windows": ["7d", "30d", "all_time"],
            "notes": {
                "temporal_config": {
                    "lag_windows": [7, 14, 30],
                    "lag_window_days": 14,
                    "columns": ["amount"],
                }
            },
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi))
        orders = {
            "source_path": "/data/orders.parquet",
            "source_format": "parquet",
            "row_count": 100,
            "column_count": 5,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "order_date": {
                    "name": "order_date",
                    "inferred_type": "datetime",
                    "confidence": 0.9,
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
                "churn": {
                    "name": "churn",
                    "inferred_type": "binary",
                    "confidence": 0.99,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
            "time_series_metadata": {"time_column": "order_date", "entity_column": "customer_id"},
        }
        (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        tf = config.bronze_event["orders"].temporal_features
        assert tf is not None
        assert tf.lag_window_days == 14
        assert tf.lag_columns == ["amount"]

    def test_temporal_features_none_when_absent(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        multi = {
            "datasets": {
                "orders": {
                    "name": "orders",
                    "findings_path": str(findings_dir / "orders_findings.yaml"),
                    "source_path": "/data/orders.parquet",
                    "granularity": "event_level",
                    "row_count": 100,
                    "column_count": 5,
                    "entity_column": "customer_id",
                    "time_column": "order_date",
                }
            },
            "relationships": [],
            "primary_entity_dataset": "orders",
            "event_datasets": ["orders"],
            "aggregation_windows": ["7d", "30d", "all_time"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi))
        orders = {
            "source_path": "/data/orders.parquet",
            "source_format": "parquet",
            "row_count": 100,
            "column_count": 5,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "order_date": {
                    "name": "order_date",
                    "inferred_type": "datetime",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "churn": {
                    "name": "churn",
                    "inferred_type": "binary",
                    "confidence": 0.99,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
            "time_series_metadata": {"time_column": "order_date", "entity_column": "customer_id"},
        }
        (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert config.bronze_event["orders"].temporal_features is None


# ======================================================================
# Phase 4: Model Tests — Text Feature Config
# ======================================================================


class TestTextFeatureConfig:
    def test_default_values(self):
        config = TextFeatureConfig(column="description")
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.n_components == 5
        assert config.component_columns == []

    def test_custom_values(self):
        config = TextFeatureConfig(
            column="notes",
            embedding_model="custom-model",
            n_components=10,
            component_columns=["pca_0", "pca_1"],
        )
        assert config.column == "notes"
        assert config.n_components == 10

    def test_bronze_event_has_text_features_field(self, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            text_features=[TextFeatureConfig(column="notes")],
        )
        assert len(config.text_features) == 1
        assert config.text_features[0].column == "notes"

    def test_bronze_layer_has_text_features_field(self, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            text_features=[TextFeatureConfig(column="bio")],
        )
        assert len(config.text_features) == 1


# ======================================================================
# Phase 5: Model Tests — Training Config
# ======================================================================


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.split_strategy == "random_stratified"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.temporal_column is None
        assert config.purge_gap_days is None
        assert config.recommended_training_start is None
        assert config.filter_future_dates is False
        assert config.imbalance_strategy == "class_weight"
        assert config.imbalance_ratio is None

    def test_temporal_split_config(self):
        config = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            purge_gap_days=7,
            recommended_training_start="2023-01-01",
        )
        assert config.split_strategy == "temporal"
        assert config.purge_gap_days == 7
        assert config.recommended_training_start == "2023-01-01"

    def test_pipeline_config_has_training_field(self, entity_source):
        config = PipelineConfig(
            name="test",
            target_column="churn",
            sources=[entity_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(),
            output_dir="/output",
            training=TrainingConfig(split_strategy="temporal"),
        )
        assert config.training.split_strategy == "temporal"

    def test_pipeline_config_training_default_none(self, entity_source):
        config = PipelineConfig(
            name="test",
            target_column="churn",
            sources=[entity_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(),
            output_dir="/output",
        )
        assert config.training is None


# ======================================================================
# Phase 5: FindingsParser Tests — Training Config
# ======================================================================


class TestTrainingConfigParser:
    def test_default_random_stratified_returns_none(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        multi = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": str(findings_dir / "customers_findings.yaml"),
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 100,
                    "column_count": 3,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "customers",
            "event_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi))
        customers = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 100,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "churn": {
                    "name": "churn",
                    "inferred_type": "binary",
                    "confidence": 0.99,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
        }
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert config.training is None


# ======================================================================
# Phase 5: Renderer Tests — Training Split
# ======================================================================


class TestTrainingTemplateLocal:
    def test_default_split_is_valid_python(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        ast.parse(result)

    def test_temporal_split_imports_data_splitter(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            purge_gap_days=7,
            test_size=0.2,
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "DataSplitter" in result
        assert "SplitStrategy" in result

    def test_temporal_split_is_valid_python(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            purge_gap_days=7,
            test_size=0.2,
        )
        result = renderer.render_training(pipeline_config_minimal)
        ast.parse(result)

    def test_temporal_split_passes_target_column(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            test_size=0.2,
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "target_column=TARGET_COLUMN" in result
        assert "splitter.split(split_df)" in result

    def test_split_result_uses_attribute_access(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            test_size=0.2,
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "splits.X_train" in result
        assert "splits.X_test" in result
        assert "splits.y_train" in result
        assert "splits.y_test" in result
        assert 'splits["X_train"]' not in result

    def test_recommended_start_filter(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            recommended_training_start="2023-06-01",
            temporal_column="order_date",
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "2023-06-01" in result
        ast.parse(result)

    def test_future_date_filter(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            filter_future_dates=True,
            temporal_column="order_date",
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "Timestamp.now()" in result
        ast.parse(result)

    def test_smote_import(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            imbalance_strategy="smote",
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "ImbalanceHandler" in result
        ast.parse(result)

    def test_class_weight_on_models(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            imbalance_strategy="class_weight",
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert 'class_weight="balanced"' in result
        ast.parse(result)

    def test_temporal_split_uses_feast_timestamp_col(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            test_size=0.2,
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "temporal_column=FEAST_TIMESTAMP_COL" in result
        assert 'temporal_column="order_date"' not in result
        assert 'temporal_column="feature_timestamp"' not in result

    def test_temporal_split_fallback_uses_feast_timestamp_col(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column=None,
            test_size=0.2,
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "temporal_column=FEAST_TIMESTAMP_COL" in result
        assert "feature_timestamp" not in result

    def test_recommended_start_uses_feast_timestamp_col(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            recommended_training_start="2023-06-01",
            temporal_column="order_date",
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "FEAST_TIMESTAMP_COL" in result
        assert '"order_date"' not in result

    def test_future_filter_uses_feast_timestamp_col(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            filter_future_dates=True,
            temporal_column="order_date",
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "FEAST_TIMESTAMP_COL" in result
        assert '"order_date"' not in result

    def test_default_split_no_data_splitter(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "DataSplitter" not in result
        assert "train_test_split" in result


class TestTrainingTemplateDatabricks:
    def test_databricks_training_is_valid_python(self, databricks_renderer, pipeline_config_minimal):
        result = databricks_renderer.render_training(pipeline_config_minimal)
        ast.parse(result)

    def test_databricks_temporal_split_imports_data_splitter(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            purge_gap_days=7,
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "DataSplitter" in result
        assert "SplitStrategy" in result
        ast.parse(result)

    def test_databricks_temporal_split_passes_target_column(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "target_column=TARGET_COLUMN" in result

    def test_databricks_split_result_uses_attribute_access(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "splits.X_train" in result
        assert "splits.X_test" in result
        assert 'splits["X_train"]' not in result

    def test_databricks_temporal_split_has_purge_gap(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            purge_gap_days=14,
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "purge_gap_days=14" in result

    def test_databricks_recommended_training_start_filter(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            recommended_training_start="2023-06-01",
            temporal_column="order_date",
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "2023-06-01" in result
        ast.parse(result)

    def test_databricks_future_date_filter(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            filter_future_dates=True,
            temporal_column="order_date",
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "current_timestamp" in result
        ast.parse(result)

    def test_databricks_temporal_split_uses_timestamp_column(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "temporal_column=TIMESTAMP_COLUMN" in result
        assert 'temporal_column="order_date"' not in result
        assert 'temporal_column="feature_timestamp"' not in result

    def test_databricks_temporal_split_fallback_uses_timestamp_column(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column=None,
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "temporal_column=TIMESTAMP_COLUMN" in result
        assert "feature_timestamp" not in result

    def test_databricks_recommended_start_uses_timestamp_column(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            recommended_training_start="2023-06-01",
            temporal_column="order_date",
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "TIMESTAMP_COLUMN" in result
        assert '"order_date"' not in result

    def test_databricks_future_filter_uses_timestamp_column(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            filter_future_dates=True,
            temporal_column="order_date",
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "TIMESTAMP_COLUMN" in result
        assert '"order_date"' not in result

    def test_databricks_default_random_split(self, databricks_renderer, pipeline_config_minimal):
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "randomSplit" in result
        assert "DataSplitter" not in result

    def test_databricks_training_all_features(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            purge_gap_days=7,
            test_size=0.15,
            recommended_training_start="2023-01-01",
            filter_future_dates=True,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        ast.parse(result)


# ======================================================================
# Phase 3: Renderer Tests — Temporal Features
# ======================================================================


class TestTemporalFeaturesRenderer:
    def test_local_bronze_event_with_temporal_features(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
            temporal_features=TemporalFeatureConfig(
                lag_window_days=14,
                num_lags=3,
                lag_columns=["amount"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert "TemporalFeatureEngineer" in result
        assert "TemporalAggregationConfig" in result
        assert "compute_temporal_features" in result
        ast.parse(result)

    def test_local_bronze_event_no_temporal_when_absent(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert "TemporalFeatureEngineer" not in result

    def test_databricks_bronze_event_with_temporal_features(self, databricks_renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
            temporal_features=TemporalFeatureConfig(
                lag_window_days=14,
                num_lags=3,
                lag_columns=["amount"],
            ),
        )
        result = databricks_renderer.render_bronze_event("orders", config)
        assert "SparkTemporalFeatureEngineer" in result
        assert "compute_temporal_features" in result
        ast.parse(result)

    def test_databricks_bronze_event_no_temporal_when_absent(self, databricks_renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = databricks_renderer.render_bronze_event("orders", config)
        assert "SparkTemporalFeatureEngineer" not in result


# ======================================================================
# Phase 4: Renderer Tests — Text Features
# ======================================================================


class TestTextFeaturesRenderer:
    def test_local_bronze_event_with_text_features(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum"],
            ),
            text_features=[TextFeatureConfig(column="notes", embedding_model="all-MiniLM-L6-v2", n_components=5)],
        )
        result = renderer.render_bronze_event("orders", config)
        assert "TextColumnProcessor" in result
        assert "TextProcessingConfig" in result
        assert "compute_text_features" in result
        assert "FIT_MODE" in result
        ast.parse(result)

    def test_local_bronze_event_no_text_when_absent(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
        )
        result = renderer.render_bronze_event("orders", config)
        assert "TextColumnProcessor" not in result

    def test_local_bronze_entity_with_text_features(self, renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            text_features=[TextFeatureConfig(column="bio")],
        )
        result = renderer.render_bronze("customers", config)
        assert "TextColumnProcessor" in result
        assert "compute_text_features_entity" in result
        ast.parse(result)

    def test_local_bronze_entity_no_text_when_absent(self, renderer, entity_source):
        config = BronzeLayerConfig(source=entity_source)
        result = renderer.render_bronze("customers", config)
        assert "TextColumnProcessor" not in result

    def test_databricks_bronze_event_with_text_features(self, databricks_renderer, event_source):
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
        result = databricks_renderer.render_bronze_event("orders", config)
        assert "TextColumnProcessor" in result
        assert "compute_text_features" in result
        ast.parse(result)

    def test_databricks_bronze_entity_with_text_features(self, databricks_renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            text_features=[TextFeatureConfig(column="bio")],
        )
        result = databricks_renderer.render_bronze("customers", config)
        assert "TextColumnProcessor" in result
        assert "compute_text_features_entity" in result
        ast.parse(result)


# ======================================================================
# Phase 6: Validation Template Tests
# ======================================================================


class TestValidationTemplate:
    def test_validation_template_is_valid_python(self, renderer, pipeline_config_minimal):
        result = renderer.render_validation(pipeline_config_minimal)
        ast.parse(result)

    def test_validation_includes_column_comparison(self, renderer, pipeline_config_minimal):
        result = renderer.render_validation(pipeline_config_minimal)
        assert "_compare_dataframes" in result

    def test_validation_includes_categorize_missing(self, renderer, pipeline_config_minimal):
        result = renderer.render_validation(pipeline_config_minimal)
        assert "_categorize_missing" in result

    def test_validation_includes_training_split_info(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(split_strategy="temporal")
        result = renderer.render_validation(pipeline_config_minimal)
        assert "temporal" in result
        ast.parse(result)

    def test_validation_default_split_info(self, renderer, pipeline_config_minimal):
        result = renderer.render_validation(pipeline_config_minimal)
        assert "random_stratified" in result
        ast.parse(result)


# ======================================================================
# Cross-cutting: ast.parse validation for all new template sections
# ======================================================================


class TestAllTemplatesValidPython:
    def test_bronze_entity_all_lifecycle_features(self, renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            lifecycle=LifecycleConfig(
                include_lifecycle_quadrant=True,
                include_cyclical_features=True,
                include_recency_bucket=True,
                include_trend_features=True,
                include_cohort_features=True,
                include_month_cyclical=True,
                include_quarter_cyclical=True,
                momentum_pairs=[{"short_window": "7d", "long_window": "30d"}],
            ),
            entity_column="customer_id",
            time_column="order_date",
        )
        result = renderer.render_bronze("customers", config)
        ast.parse(result)

    def test_bronze_event_template_valid(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        ast.parse(result)

    def test_training_template_all_features(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            purge_gap_days=7,
            test_size=0.15,
            recommended_training_start="2023-01-01",
            filter_future_dates=True,
            imbalance_strategy="class_weight",
        )
        result = renderer.render_training(pipeline_config_minimal)
        ast.parse(result)

    def test_bronze_event_with_temporal_and_text(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum"],
            ),
            temporal_features=TemporalFeatureConfig(lag_window_days=14, num_lags=3, lag_columns=["amount"]),
            text_features=[TextFeatureConfig(column="description")],
        )
        result = renderer.render_bronze_event("orders", config)
        assert "TemporalFeatureEngineer" in result
        assert "TextColumnProcessor" in result
        ast.parse(result)

    def test_databricks_bronze_event_with_temporal_and_text(self, databricks_renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "all_time"],
                value_columns=["amount"],
                agg_funcs=["sum"],
            ),
            temporal_features=TemporalFeatureConfig(lag_window_days=14, num_lags=3, lag_columns=["amount"]),
            text_features=[TextFeatureConfig(column="description")],
        )
        result = databricks_renderer.render_bronze_event("orders", config)
        assert "SparkTemporalFeatureEngineer" in result
        assert "TextColumnProcessor" in result
        ast.parse(result)

    def test_bronze_entity_with_text_and_lifecycle(self, renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            lifecycle=LifecycleConfig(
                include_recency_bucket=True,
                include_trend_features=True,
            ),
            entity_column="customer_id",
            time_column="order_date",
            text_features=[TextFeatureConfig(column="bio")],
        )
        result = renderer.render_bronze("customers", config)
        assert "TextColumnProcessor" in result
        assert "entity_trend_slope" in result
        ast.parse(result)


class TestFeatureTimestampPreservation:
    def test_bronze_event_aggregation_preserves_feature_timestamp(self, renderer, event_source):
        config = BronzeEventConfig(
            source=event_source,
            entity_column="customer_id",
            time_column="order_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        result = renderer.render_bronze_event("orders", config)
        assert '"feature_timestamp" in df.columns' in result
        assert 'groupby(ENTITY_COLUMN)["feature_timestamp"].max()' in result
        ast.parse(result)

    def test_gold_add_feast_timestamp_renames_feature_timestamp(self, renderer, pipeline_config_minimal):
        result = renderer.render_gold(pipeline_config_minimal)
        assert '"feature_timestamp" in df.columns' in result
        assert 'rename(columns={"feature_timestamp": FEAST_TIMESTAMP_COL})' in result
        ast.parse(result)

    def test_gold_add_feast_timestamp_scalar_fallback(self, renderer, pipeline_config_minimal):
        result = renderer.render_gold(pipeline_config_minimal)
        assert "aggregation_reference_date" in result
        assert "datetime.now()" in result
        ast.parse(result)
