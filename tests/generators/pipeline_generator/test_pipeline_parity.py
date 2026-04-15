import ast
from pathlib import Path

import pytest
import yaml

from customer_retention.generators.pipeline_generator.databricks_generator import (
    DatabricksPipelineGenerator,
)
from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.findings_parser import (
    FindingsParser,
    _edges_to_labels,
)
from customer_retention.generators.pipeline_generator.generator import PipelineGenerator
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
    TemporalFeatureConfig,
    TextFeatureConfig,
    TrainingConfig,
    TransformationStep,
)
from customer_retention.generators.pipeline_generator.protocols import (
    CodeRendererProtocol,
    PipelineGeneratorBase,
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
                "aggregation_windows_used": ["7d", "30d", "90d"],
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

    def test_databricks_bronze_entity_includes_dow_cyclical(
        self, databricks_renderer, bronze_event_config_with_lifecycle
    ):
        result = databricks_renderer.render_bronze_entity(
            "orders", bronze_event_config_with_lifecycle, "orders", "orders"
        )
        assert "dow_sin" in result
        assert "dow_cos" in result
        assert "add_cyclical_features" in result


# ======================================================================
# Phase 2: Model Tests — Configurable Recency Buckets
# ======================================================================


class TestRecencyBucketConfig:
    def test_default_edges_match_nb01d_canonical(self):
        """The renderer's recency_bucket chain is driven by these edges. NB01d
        (the source of truth for spec features) uses boundaries at 7, 30, 90, 180
        with the final bucket being ">180d" (no explicit 365 edge). Defaults that
        add a 365 edge drive the renderer to produce `recency_bucket_180_365d`
        columns that don't match the spec's `recency_bucket_>180d`."""
        config = LifecycleConfig()
        assert config.recency_bucket_edges == [0, 7, 30, 90, 180]

    def test_default_labels_match_nb01d_canonical(self):
        """NB01d writes `"0-7d", "8-30d", "31-90d", "91-180d", ">180d"` into the
        recency_bucket column. After one-hot sanitization those become
        `recency_bucket_0_7d, ..._>180d` in the FeatureSpec; any deviation breaks
        parity at `_apply_feature_spec_gate`."""
        config = LifecycleConfig()
        assert config.recency_bucket_labels == ["0-7d", "8-30d", "31-90d", "91-180d", ">180d"]

    def test_custom_edges_accepted(self):
        config = LifecycleConfig(
            recency_bucket_edges=[0, 14, 60, 120],
            recency_bucket_labels=["0-14d", "14-60d", "60-120d", "120d+"],
        )
        assert config.recency_bucket_edges == [0, 14, 60, 120]
        assert len(config.recency_bucket_labels) == 4


class TestEdgesToLabels:
    """Parity-critical: `_edges_to_labels` must produce the same labels NB01c/NB01d
    write into the recency_bucket column. The canonical convention is `"<start+1>-<end>d"`
    for interior buckets, `"0-<end>d"` for the leading bucket, `">{start}d"` for any
    trailing `inf` edge. See `temporal_pattern_analyzer.generate_bucket_labels`."""

    def test_canonical_edges_match_nb01d(self):
        labels = _edges_to_labels([0, 7, 30, 90, 180])
        assert labels == ["0-7d", "8-30d", "31-90d", "91-180d"]

    def test_custom_edges(self):
        labels = _edges_to_labels([0, 14, 60])
        assert labels == ["0-14d", "15-60d"]

    def test_inf_tail_edge_emits_gt_prefix(self):
        labels = _edges_to_labels([0, 7, 30, float("inf")])
        assert labels == ["0-7d", "8-30d", ">30d"]


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
            "time_series_metadata": {"time_column": "order_date", "entity_column": "customer_id", "aggregation_windows_used": ["7d", "30d", "90d"]},
        }
        (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        lifecycle = config.bronze_event["orders"].lifecycle
        assert lifecycle.recency_bucket_edges == [0, 14, 60, 120]
        assert lifecycle.recency_bucket_labels == ["0-14d", "15-60d", "61-120d"]


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
        assert config.feature_groups == [
            "lagged_windows", "velocity", "acceleration", "lifecycle",
            "recency", "regularity", "cohort_comparison",
        ]

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
            "time_series_metadata": {"time_column": "order_date", "entity_column": "customer_id", "aggregation_windows_used": ["7d", "30d", "90d"]},
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
            "time_series_metadata": {"time_column": "order_date", "entity_column": "customer_id", "aggregation_windows_used": ["7d", "30d", "90d"]},
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
        assert config.split_strategy == "temporal"
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
        assert "datetime.now()" in result
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
        splitter_section = result[result.index("DataSplitter("):]
        assert 'temporal_column="feature_timestamp"' not in splitter_section

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

    def test_default_split_uses_data_splitter(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "DataSplitter" in result
        assert "SplitStrategy.TEMPORAL" in result


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
        assert "percentile_approx" in result or "F.percentile_approx" in result or "percent_rank" in result
        ast.parse(result)

    def test_databricks_temporal_split_passes_target_column(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert '"label"' in result or "label" in result

    def test_databricks_temporal_split_has_purge_gap(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column="order_date",
            purge_gap_days=14,
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "purge" in result or "gap" in result

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
        assert "TIMESTAMP_COLUMN" in result
        assert 'temporal_column="order_date"' not in result
        assert 'temporal_column="feature_timestamp"' not in result

    def test_databricks_temporal_split_fallback_uses_timestamp_column(self, databricks_renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            split_strategy="temporal",
            temporal_column=None,
            test_size=0.2,
        )
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "TIMESTAMP_COLUMN" in result

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

    def test_databricks_default_temporal_split(self, databricks_renderer, pipeline_config_minimal):
        result = databricks_renderer.render_training(pipeline_config_minimal)
        assert "TIMESTAMP_COLUMN" in result
        assert "train" in result.lower()
        assert "test" in result.lower()

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

    def test_databricks_temporal_features_uses_spark_join_not_pandas_merge(self, databricks_renderer, event_source):
        """SparkTemporalFeatureEngineer returns pyspark.pandas — merging with
        native pandas .merge() raises TypeError. The template must stay in
        Spark: convert temporal_df via .to_spark() and use .join()."""
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
        # Extract compute_temporal_features function body
        lines = result.split("\n")
        in_func = False
        func_body = []
        for line in lines:
            if "def compute_temporal_features" in line:
                in_func = True
            elif in_func:
                if line and not line[0].isspace() and not line.startswith("#"):
                    break
                func_body.append(line)
        body_text = "\n".join(func_body)
        assert ".toPandas()" not in body_text, "temporal features must not collect to driver"
        assert ".merge(" not in body_text, "must use Spark .join(), not pandas .merge()"
        assert ".to_spark()" in body_text, "must convert pyspark.pandas to native Spark"
        assert ".join(" in body_text, "must use Spark .join()"

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
        assert "FIT_MODE" in result
        ast.parse(result)

    def test_databricks_bronze_entity_with_text_features(self, databricks_renderer, entity_source):
        config = BronzeLayerConfig(
            source=entity_source,
            text_features=[TextFeatureConfig(column="bio")],
        )
        result = databricks_renderer.render_bronze("customers", config)
        assert "TextColumnProcessor" in result
        assert "compute_text_features_entity" in result
        assert "FIT_MODE" in result
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
        assert "temporal" in result
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

    @staticmethod
    def _extract_and_exec_feast_timestamp(gold_code):
        import warnings
        from datetime import datetime

        import pandas as pd
        tree = ast.parse(gold_code)
        func_src = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "add_feast_timestamp":
                func_src = ast.get_source_segment(gold_code, node)
                break
        assert func_src is not None, "add_feast_timestamp not found in gold template"
        feast_col = "event_timestamp"
        ns = {"pd": pd, "warnings": warnings, "datetime": datetime, "FEAST_TIMESTAMP_COL": feast_col}
        exec(compile(func_src, "<feast_ts>", "exec"), ns)
        return ns["add_feast_timestamp"], feast_col

    def test_gold_add_feast_timestamp_rename_does_not_use_undefined_timestamp(self, renderer, pipeline_config_minimal):
        import pandas as pd
        result = renderer.render_gold(pipeline_config_minimal)
        fn, feast_col = self._extract_and_exec_feast_timestamp(result)
        df = pd.DataFrame({"customer_id": [1], "feature_timestamp": [pd.Timestamp("2024-01-01")]})
        out = fn(df)
        assert feast_col in out.columns
        assert out[feast_col].iloc[0] == pd.Timestamp("2024-01-01")

    def test_gold_add_feast_timestamp_uses_attrs_reference_date(self, renderer, pipeline_config_minimal):
        import pandas as pd
        result = renderer.render_gold(pipeline_config_minimal)
        fn, feast_col = self._extract_and_exec_feast_timestamp(result)
        df = pd.DataFrame({"customer_id": [1], "value": [42]})
        df.attrs["aggregation_reference_date"] = "2024-06-15"
        out = fn(df)
        assert out[feast_col].iloc[0] == pd.Timestamp("2024-06-15")

    def test_gold_add_feast_timestamp_noop_when_already_present(self, renderer, pipeline_config_minimal):
        import pandas as pd
        result = renderer.render_gold(pipeline_config_minimal)
        fn, feast_col = self._extract_and_exec_feast_timestamp(result)
        df = pd.DataFrame({"customer_id": [1], feast_col: [pd.Timestamp("2024-03-01")]})
        out = fn(df)
        assert out[feast_col].iloc[0] == pd.Timestamp("2024-03-01")


# ======================================================================
# Protocol Conformance Tests
# ======================================================================


def _extract_function_names(code: str) -> set:
    tree = ast.parse(code)
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


class TestProtocolConformance:
    def test_local_renderer_satisfies_protocol(self):
        assert isinstance(CodeRenderer(), CodeRendererProtocol)

    def test_databricks_renderer_satisfies_protocol(self):
        assert isinstance(DatabricksCodeRenderer(catalog="c", schema="s"), CodeRendererProtocol)

    def test_local_generator_is_pipeline_generator_base(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        gen = PipelineGenerator(str(findings_dir), str(tmp_path / "out"), "test")
        assert isinstance(gen, PipelineGeneratorBase)

    def test_databricks_generator_is_pipeline_generator_base(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        gen = DatabricksPipelineGenerator(str(findings_dir), str(tmp_path / "out"), "test")
        assert isinstance(gen, PipelineGeneratorBase)


# ======================================================================
# Structural Gold Parity Tests
# ======================================================================


class TestGoldTemplateParity:
    def test_gold_templates_share_core_functions(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_gold(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_gold(pipeline_config_minimal)
        local_funcs = _extract_function_names(local_code)
        db_funcs = _extract_function_names(db_code)
        core = {"apply_encodings", "apply_feature_selection"}
        assert core <= local_funcs, f"Local gold missing: {core - local_funcs}"
        assert core <= db_funcs, f"Databricks gold missing: {core - db_funcs}"


class TestGoldSparkJobBulkOperations:
    """Databricks gold template must batch Spark operations to minimize job count."""

    def _gold_config_with_cap_then_log(self, entity_source, event_source):
        return PipelineConfig(
            name="test",
            target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(transformations=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_THEN_LOG,
                    column="amount_sum", parameters={},
                    rationale="High skewness",
                ),
                TransformationStep(
                    type=PipelineTransformationType.CAP_THEN_LOG,
                    column="amount_mean", parameters={},
                    rationale="High skewness",
                ),
                TransformationStep(
                    type=PipelineTransformationType.CAP_THEN_LOG,
                    column="count_total", parameters={},
                    rationale="High skewness",
                ),
            ]),
            output_dir="/output",
            composite_name="test_cn",
        )

    def _gold_config_with_segment_cap(self, entity_source, event_source):
        return PipelineConfig(
            name="test",
            target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(transformations=[
                TransformationStep(
                    type=PipelineTransformationType.SEGMENT_AWARE_CAP,
                    column="feat_a", parameters={},
                    rationale="Outliers",
                ),
                TransformationStep(
                    type=PipelineTransformationType.SEGMENT_AWARE_CAP,
                    column="feat_b", parameters={},
                    rationale="Outliers",
                ),
            ]),
            output_dir="/output",
            composite_name="test_cn",
        )

    def test_databricks_gold_batches_cap_then_log(self, entity_source, event_source):
        """cap_then_log must use _batch_cap_then_log (1 approxQuantile call)
        instead of per-column _cap_then_log (N calls)."""
        config = self._gold_config_with_cap_then_log(entity_source, event_source)
        result = DatabricksCodeRenderer(catalog="c", schema="s").render_gold(config)
        assert "_batch_cap_then_log" in result
        lines = result.split("\n")
        func_body = []
        in_func = False
        for line in lines:
            if "def apply_cap_then_log_transforms" in line:
                in_func = True
            elif in_func:
                if line and not line[0].isspace() and not line.startswith("#"):
                    break
                func_body.append(line)
        body = "\n".join(func_body)
        assert 'df = _cap_then_log(df,' not in body, "must use batched version"
        assert "_batch_cap_then_log(df," in body
        ast.parse(result)

    def test_databricks_gold_batches_segment_aware_cap(self, entity_source, event_source):
        """segment_aware_cap must use _batch_segment_aware_cap (1 approxQuantile call)."""
        config = self._gold_config_with_segment_cap(entity_source, event_source)
        result = DatabricksCodeRenderer(catalog="c", schema="s").render_gold(config)
        assert "_batch_segment_aware_cap" in result
        lines = result.split("\n")
        func_body = []
        in_func = False
        for line in lines:
            if "def cap_segment_aware_outliers" in line:
                in_func = True
            elif in_func:
                if line and not line[0].isspace() and not line.startswith("#"):
                    break
                func_body.append(line)
        body = "\n".join(func_body)
        assert 'df = _segment_aware_cap(df,' not in body, "must use batched version"
        assert "_batch_segment_aware_cap(df," in body
        ast.parse(result)

    def test_databricks_gold_encode_one_hot_single_collect(self, pipeline_config_minimal):
        """_encode_one_hot must use a single .collect() not count() + collect()."""
        pipeline_config_minimal.gold = GoldLayerConfig(encodings=[
            TransformationStep(
                type=PipelineTransformationType.ENCODE,
                column="category_col", parameters={},
                rationale="Encode categorical",
            ),
        ])
        result = DatabricksCodeRenderer(catalog="c", schema="s").render_gold(
            pipeline_config_minimal,
        )
        lines = result.split("\n")
        func_body = []
        in_func = False
        for line in lines:
            if "def _encode_one_hot" in line:
                in_func = True
            elif in_func:
                if line.startswith("def "):
                    break
                func_body.append(line)
        body = "\n".join(func_body)
        assert ".distinct().count()" not in body, "double scan: use collect then len()"
        ast.parse(result)

    def test_databricks_gold_no_dead_per_column_scalers(self, pipeline_config_minimal):
        """Per-column _scale_standard / _scale_minmax are dead code — only batch versions used."""
        result = DatabricksCodeRenderer(catalog="c", schema="s").render_gold(
            pipeline_config_minimal,
        )
        assert "def _scale_standard(" not in result, "dead per-column scaler"
        assert "def _scale_minmax(" not in result, "dead per-column scaler"
        assert "def _batch_scale_standard(" in result
        assert "def _batch_scale_minmax(" in result


# ======================================================================
# Training Template Parity Tests
# ======================================================================


class TestTrainingTemplateParity:
    def test_both_renderers_produce_valid_python(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        ast.parse(local_code)
        ast.parse(db_code)

    def test_both_renderers_use_nested_runs(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        assert "nested=True" in local_code, "Local should have nested MLflow runs"
        assert "nested=True" in db_code, "Databricks should have nested MLflow runs"
        for code, label in [(local_code, "local"), (db_code, "databricks")]:
            assert "LogisticRegression" in code, f"{label} missing LogisticRegression"
            assert "RandomForest" in code, f"{label} missing RandomForest"

    def test_both_renderers_filter_null_labels(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        assert "notna" in local_code or "isNotNull" in local_code
        assert "isNotNull" in db_code

    def test_both_experiment_names_use_composite_name(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        for code, label in [(local_code, "local"), (db_code, "databricks")]:
            assert 'f"training_{COMPOSITE_NAME}"' in code, f"{label} experiment name missing COMPOSITE_NAME"

    def test_both_use_parent_run_pattern(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        for code, label in [(local_code, "local"), (db_code, "databricks")]:
            parent_lines = [line for line in code.splitlines() if "with mlflow.start_run(run_name=" in line and "nested" not in line]
            assert parent_lines, f"{label} template missing parent mlflow.start_run"
            nested_lines = [line for line in code.splitlines() if "nested=True" in line]
            assert nested_lines, f"{label} template missing nested model runs"

    def test_both_tag_composite_name_inside_parent_run(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        for code, label in [(local_code, "local"), (db_code, "databricks")]:
            assert 'set_tag("composite_name", COMPOSITE_NAME)' in code, f"{label} missing composite_name tag"

    def test_neither_has_standalone_best_model_run(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        assert 'start_run(run_name="best_model")' not in db_code
        assert 'start_run(run_name="best_model")' not in local_code


# ======================================================================
# Generator Base Tests — shared core stages
# ======================================================================


def _make_minimal_findings(findings_dir):
    findings_dir.mkdir(parents=True, exist_ok=True)
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


CORE_STAGE_PATTERNS = ["config.py", "bronze", "silver", "gold", "training", "pipeline_runner.py"]


class TestGeneratorBaseSharedStages:
    def test_local_generator_produces_core_stages(self, tmp_path):
        findings_dir = tmp_path / "findings"
        _make_minimal_findings(findings_dir)
        output_dir = tmp_path / "local_out"
        output_dir.mkdir()
        gen = PipelineGenerator(str(findings_dir), str(output_dir), "test_pipeline")
        files = gen.generate()
        file_strs = [str(f) for f in files]
        for pattern in CORE_STAGE_PATTERNS:
            assert any(pattern in s for s in file_strs), f"Local generator missing core stage: {pattern}"

    def test_databricks_generator_produces_core_stages(self, tmp_path):
        findings_dir = tmp_path / "findings"
        _make_minimal_findings(findings_dir)
        output_dir = tmp_path / "db_out"
        output_dir.mkdir()
        gen = DatabricksPipelineGenerator(str(findings_dir), str(output_dir), "test_pipeline")
        files = gen.generate()
        file_strs = [str(f) for f in files]
        for pattern in CORE_STAGE_PATTERNS:
            assert any(pattern in s for s in file_strs), f"Databricks generator missing core stage: {pattern}"

    def test_both_generators_produce_same_core_file_names(self, tmp_path):
        findings_dir = tmp_path / "findings"
        _make_minimal_findings(findings_dir)
        local_dir = tmp_path / "local_out"
        local_dir.mkdir()
        db_dir = tmp_path / "db_out"
        db_dir.mkdir()
        local_gen = PipelineGenerator(str(findings_dir), str(local_dir), "test_pipeline")
        db_gen = DatabricksPipelineGenerator(str(findings_dir), str(db_dir), "test_pipeline")
        local_files = {f.relative_to(local_dir) for f in local_gen.generate()}
        db_files = {f.relative_to(db_dir) for f in db_gen.generate()}
        core_local = {f for f in local_files if any(p in str(f) for p in CORE_STAGE_PATTERNS)}
        core_db = {f for f in db_files if any(p in str(f) for p in CORE_STAGE_PATTERNS)}
        assert core_local == core_db, f"Core file name mismatch:\nLocal-only: {core_local - core_db}\nDB-only: {core_db - core_local}"


class TestLocalTrainingFeatureProfile:
    def test_local_training_imports_feature_profile(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "from customer_retention.stages.modeling.feature_profile import" in result

    def test_local_training_computes_production_profile(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "build_feature_profile(" in result
        assert '"production"' in result
        assert "Production profile:" in result

    def test_local_training_warns_when_no_exploration_profile(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "_EXPLORATION_PROFILE = None" in result
        assert "No exploration feature profile available for comparison" in result

    def test_local_training_compares_when_exploration_profile_present(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            exploration_feature_profile={"stage": "exploration", "created_at": "2024-01-01", "row_count": 100, "feature_count": 2, "target_column": "churn", "features": {"col_a": {"dtype": "double", "non_null": 90, "null_count": 10}}, "excluded": {}},
        )
        result = renderer.render_training(pipeline_config_minimal)
        assert "_EXPLORATION_PROFILE = {" in result
        assert "FeatureProfile.from_dict(_EXPLORATION_PROFILE)" in result
        assert "compare_feature_profiles(" in result

    def test_local_training_has_assert_rows(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "def _assert_rows" in result

    def test_local_training_timing(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "from customer_retention.core.compat.timing import log_timing" in result
        assert 'log_timing("load_gold_data"' in result
        assert 'log_timing("temporal_split"' in result
        assert 'log_timing("feature_profile"' in result

    def test_local_training_excludes_temporal_metadata(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "GOLD_METADATA_COLUMNS" in result
        assert "_EXCLUDE_COLS" in result

    def test_local_training_is_valid_python(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        ast.parse(result)

    def test_local_training_experiment_uses_composite_name(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert 'f"training_{COMPOSITE_NAME}"' in result

    def test_local_training_parent_run_uses_composite_name(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert 'f"training_{COMPOSITE_NAME}"' in result
        parent_lines = [line for line in result.splitlines() if "with mlflow.start_run(run_name=" in line and "nested" not in line]
        assert parent_lines, "No parent start_run found"

    def test_local_training_with_exploration_profile_is_valid_python(self, renderer, pipeline_config_minimal):
        pipeline_config_minimal.training = TrainingConfig(
            exploration_feature_profile={"stage": "exploration", "created_at": "2024-01-01", "row_count": 100, "feature_count": 2, "target_column": "churn", "features": {"col_a": {"dtype": "double", "non_null": 90, "null_count": 10}}, "excluded": {}},
        )
        result = renderer.render_training(pipeline_config_minimal)
        ast.parse(result)


class TestExplorationFeatureProfileSaved:
    NB08_PATH = Path(__file__).parent.parent.parent.parent / "exploration_notebooks" / "08_baseline_experiments.ipynb"

    def test_nb08_saves_exploration_feature_profile(self):
        import json
        nb = json.loads(self.NB08_PATH.read_text())
        all_source = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
        assert "build_feature_profile(" in all_source
        assert "exploration_feature_profile_path" in all_source

    def test_nb08_profile_uses_namespace_path(self):
        import json
        nb = json.loads(self.NB08_PATH.read_text())
        all_source = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
        assert "_namespace.exploration_feature_profile_path" in all_source

    def test_nb08_profile_guarded_by_skip_and_namespace(self):
        import json
        nb = json.loads(self.NB08_PATH.read_text())
        profile_cell = None
        for cell in nb["cells"]:
            source = "".join(cell["source"])
            if "exploration_feature_profile_path" in source:
                profile_cell = source
                break
        assert profile_cell is not None
        assert "_skip_modeling" in profile_cell
        assert "_namespace" in profile_cell


class TestProductionFeatureProfilePersistence:
    def test_local_training_imports_run_namespace(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace" in result

    def test_local_training_saves_production_profile(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        fn = result[result.index("def run_experiment"):]
        assert "prod_profile.save(" in fn
        assert "production_feature_profile_path" in fn

    def test_local_training_constructs_namespace(self, renderer, pipeline_config_minimal):
        result = renderer.render_training(pipeline_config_minimal)
        assert "RunNamespace.from_env_or_latest" in result or "_NAMESPACE" in result

    def test_databricks_training_saves_production_profile(self, pipeline_config_minimal):
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        fn = db_code[db_code.index("def train_and_evaluate"):]
        assert "prod_profile.save(" in fn
        assert "production_feature_profile_path" in fn

    def test_both_templates_persist_production_profile(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        assert "prod_profile.save(" in local_code
        assert "prod_profile.save(" in db_code
        assert "production_feature_profile_path" in local_code
        assert "production_feature_profile_path" in db_code


class TestTrainingMetricsParity:
    def test_both_templates_log_feature_importance(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        assert "feature_importance" in local_code or "featureImportances" in local_code
        assert "feature_importance" in db_code or "featureImportances" in db_code

    def test_both_templates_log_pr_auc(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        assert "pr_auc" in local_code or "average_precision" in local_code
        assert "areaUnderPR" in db_code or "pr_auc" in db_code

    def test_both_templates_tag_pipeline_name(self, pipeline_config_minimal):
        local_code = CodeRenderer().render_training(pipeline_config_minimal)
        db_code = DatabricksCodeRenderer(catalog="c", schema="s").render_training(pipeline_config_minimal)
        assert "pipeline_name" in local_code
        assert "pipeline_name" in db_code
