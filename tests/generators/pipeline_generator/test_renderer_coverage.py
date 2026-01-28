"""Coverage tests for CodeRenderer to reach 86% threshold.

Note: Most of renderer.py is template strings (TEMPLATES dict ~1500 lines).
Coverage is achieved by testing the rendering methods which execute the templates.
"""

import json

import pytest

from customer_retention.generators.pipeline_generator.models import (
    BronzeLayerConfig,
    FeastConfig,
    GoldLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TransformationStep,
)
from customer_retention.generators.pipeline_generator.renderer import (
    TEMPLATES,
    CodeRenderer,
    InlineLoader,
)


@pytest.fixture
def sample_source_config():
    """Create a sample SourceConfig."""
    return SourceConfig(
        name="customers",
        path="data/customers.csv",
        format="csv",
        entity_key="customer_id",
        time_column="created_at",
        is_event_level=False,
    )


@pytest.fixture
def sample_event_source_config():
    """Create a sample event-level SourceConfig."""
    return SourceConfig(
        name="events",
        path="data/events.parquet",
        format="parquet",
        entity_key="customer_id",
        time_column="event_date",
        is_event_level=True,
    )


@pytest.fixture
def sample_bronze_config(sample_source_config):
    """Create a sample BronzeLayerConfig with transformations."""
    return BronzeLayerConfig(
        source=sample_source_config,
        transformations=[
            TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column="age",
                parameters={"value": 0},
                rationale="Fill missing ages with 0",
            ),
            TransformationStep(
                type=PipelineTransformationType.CAP_OUTLIER,
                column="income",
                parameters={"lower": 0, "upper": 1000000},
                rationale="Cap income outliers",
            ),
            TransformationStep(
                type=PipelineTransformationType.TYPE_CAST,
                column="zip_code",
                parameters={"dtype": "str"},
                rationale="Convert zip to string",
            ),
        ],
    )


@pytest.fixture
def sample_silver_config():
    """Create a sample SilverLayerConfig with joins."""
    return SilverLayerConfig(
        joins=[
            {
                "right_source": "events",
                "left_key": "customer_id",
                "right_key": "customer_id",
                "how": "left",
            }
        ],
        aggregations=[],
    )


@pytest.fixture
def sample_gold_config():
    """Create a sample GoldLayerConfig with encodings and scalings."""
    return GoldLayerConfig(
        encodings=[
            TransformationStep(
                type=PipelineTransformationType.ENCODE,
                column="region",
                parameters={"method": "one_hot"},
                rationale="One-hot encode region",
            ),
            TransformationStep(
                type=PipelineTransformationType.ENCODE,
                column="tier",
                parameters={"method": "label"},
                rationale="Label encode tier",
            ),
        ],
        scalings=[
            TransformationStep(
                type=PipelineTransformationType.SCALE,
                column="income",
                parameters={"method": "standard"},
                rationale="Standardize income",
            ),
            TransformationStep(
                type=PipelineTransformationType.SCALE,
                column="age",
                parameters={"method": "minmax"},
                rationale="MinMax scale age",
            ),
        ],
        feature_selections=["income", "age", "region"],
    )


@pytest.fixture
def sample_feast_config():
    """Create a sample FeastConfig."""
    return FeastConfig(
        repo_path="./feast_repo",
        feature_view_name="customer_features",
        entity_name="customer",
        entity_key="customer_id",
        timestamp_column="event_timestamp",
        ttl_days=365,
    )


@pytest.fixture
def sample_pipeline_config(
    sample_source_config,
    sample_event_source_config,
    sample_bronze_config,
    sample_silver_config,
    sample_gold_config,
    sample_feast_config,
):
    """Create a complete sample PipelineConfig."""
    return PipelineConfig(
        name="test_pipeline",
        target_column="churned",
        sources=[sample_source_config, sample_event_source_config],
        bronze={
            "customers": sample_bronze_config,
            "events": BronzeLayerConfig(source=sample_event_source_config, transformations=[]),
        },
        silver=sample_silver_config,
        gold=sample_gold_config,
        output_dir="./output",
        iteration_id="iter_001",
        parent_iteration_id="iter_000",
        recommendations_hash="abc123",
        feast=sample_feast_config,
        experiments_dir="experiments",
    )


@pytest.fixture
def minimal_pipeline_config(sample_source_config):
    """Create a minimal PipelineConfig without optional fields."""
    return PipelineConfig(
        name="minimal_pipeline",
        target_column="target",
        sources=[sample_source_config],
        bronze={"customers": BronzeLayerConfig(source=sample_source_config, transformations=[])},
        silver=SilverLayerConfig(),
        gold=GoldLayerConfig(),
        output_dir="./output",
    )


class TestInlineLoader:
    """Tests for InlineLoader class."""

    def test_inline_loader_get_source_found(self):
        """Should return template source when template exists."""
        templates = {"test.j2": "Hello {{ name }}"}
        loader = InlineLoader(templates)

        source, name, uptodate = loader.get_source(None, "test.j2")

        assert source == "Hello {{ name }}"
        assert name == "test.j2"
        assert uptodate() is True

    def test_inline_loader_get_source_not_found(self):
        """Should raise exception when template doesn't exist."""
        templates = {"test.j2": "Hello"}
        loader = InlineLoader(templates)

        with pytest.raises(Exception, match="Template .* not found"):
            loader.get_source(None, "nonexistent.j2")


class TestCodeRendererConfig:
    """Tests for render_config method."""

    def test_render_config_with_all_options(self, sample_pipeline_config):
        """Should render config.py with all configuration options."""
        renderer = CodeRenderer()
        result = renderer.render_config(sample_pipeline_config)

        assert 'PIPELINE_NAME = "test_pipeline"' in result
        assert 'TARGET_COLUMN = "churned"' in result
        assert 'ITERATION_ID = "iter_001"' in result
        assert 'PARENT_ITERATION_ID = "iter_000"' in result
        assert 'RECOMMENDATIONS_HASH = "abc123"' in result
        assert "experiments" in result

    def test_render_config_minimal(self, minimal_pipeline_config):
        """Should render config.py with minimal options (None values)."""
        renderer = CodeRenderer()
        result = renderer.render_config(minimal_pipeline_config)

        assert 'PIPELINE_NAME = "minimal_pipeline"' in result
        assert "ITERATION_ID = None" in result
        assert "RECOMMENDATIONS_HASH = None" in result

    def test_render_config_has_path_functions(self, sample_pipeline_config):
        """Config should include path helper functions."""
        renderer = CodeRenderer()
        result = renderer.render_config(sample_pipeline_config)

        assert "def get_bronze_path" in result
        assert "def get_silver_path" in result
        assert "def get_gold_path" in result
        assert "def get_feast_data_path" in result


class TestCodeRendererBronze:
    """Tests for render_bronze method."""

    def test_render_bronze_with_transformations(self, sample_bronze_config):
        """Should render bronze.py with all transformation types."""
        renderer = CodeRenderer()
        result = renderer.render_bronze("customers", sample_bronze_config)

        assert 'SOURCE_NAME = "customers"' in result
        assert "def load_customers" in result
        assert "def apply_transformations" in result
        assert "def run_bronze_customers" in result

        # Check transformation types
        assert "fillna" in result  # impute_null
        assert "clip" in result  # cap_outlier
        assert "astype" in result  # type_cast

    def test_render_bronze_no_transformations(self, sample_source_config):
        """Should render bronze.py without transformations."""
        bronze_config = BronzeLayerConfig(source=sample_source_config, transformations=[])
        renderer = CodeRenderer()
        result = renderer.render_bronze("customers", bronze_config)

        assert "def load_customers" in result
        assert "def apply_transformations" in result
        # transformations should just return df
        assert "return df" in result


class TestCodeRendererSilver:
    """Tests for render_silver method."""

    def test_render_silver_with_joins(self, sample_pipeline_config):
        """Should render silver.py with join operations."""
        renderer = CodeRenderer()
        result = renderer.render_silver(sample_pipeline_config)

        assert "def merge_sources" in result
        assert "merged.merge" in result
        assert 'right_on="customer_id"' in result
        assert 'how="left"' in result

    def test_render_silver_has_holdout_creation(self, sample_pipeline_config):
        """Silver template should include holdout creation."""
        renderer = CodeRenderer()
        result = renderer.render_silver(sample_pipeline_config)

        assert "create_holdout_mask" in result
        assert "holdout_fraction" in result


class TestCodeRendererGold:
    """Tests for render_gold method."""

    def test_render_gold_with_encodings_and_scalings(self, sample_pipeline_config):
        """Should render gold.py with encoding and scaling operations."""
        renderer = CodeRenderer()
        result = renderer.render_gold(sample_pipeline_config)

        assert "def apply_encodings" in result
        assert "def apply_scaling" in result

        # Check encoding types
        assert "get_dummies" in result  # one_hot
        assert "LabelEncoder" in result  # label

        # Check scaling types
        assert "StandardScaler" in result
        assert "MinMaxScaler" in result

    def test_render_gold_has_feast_integration(self, sample_pipeline_config):
        """Gold template should include Feast integration."""
        renderer = CodeRenderer()
        result = renderer.render_gold(sample_pipeline_config)

        assert "add_feast_timestamp" in result
        assert "materialize_to_feast" in result
        assert "aggregation_reference_date" in result

    def test_render_gold_excludes_original_columns(self, sample_pipeline_config):
        """Gold template should exclude original_* columns from Feast."""
        renderer = CodeRenderer()
        result = renderer.render_gold(sample_pipeline_config)

        assert "original_" in result
        assert "Excluding holdout columns" in result or "exclude" in result.lower()


class TestCodeRendererTraining:
    """Tests for render_training method."""

    def test_render_training_with_feast(self, sample_pipeline_config):
        """Should render training.py with Feast integration."""
        renderer = CodeRenderer()
        result = renderer.render_training(sample_pipeline_config)

        assert "get_training_data_from_feast" in result
        assert "FeatureStore" in result
        assert "get_historical_features" in result

    def test_render_training_excludes_original_columns(self, sample_pipeline_config):
        """Training template should exclude original_* columns."""
        renderer = CodeRenderer()
        result = renderer.render_training(sample_pipeline_config)

        assert "original_" in result
        assert "startswith" in result

    def test_render_training_has_model_functions(self, sample_pipeline_config):
        """Training template should include model training functions."""
        renderer = CodeRenderer()
        result = renderer.render_training(sample_pipeline_config)

        assert "train_xgboost" in result
        assert "LogisticRegression" in result
        assert "RandomForestClassifier" in result
        assert "compute_metrics" in result


class TestCodeRendererRunner:
    """Tests for render_runner method."""

    def test_render_runner(self, sample_pipeline_config):
        """Should render runner.py with pipeline orchestration."""
        renderer = CodeRenderer()
        result = renderer.render_runner(sample_pipeline_config)

        assert "run_pipeline" in result
        assert "run_bronze_customers" in result
        assert "run_bronze_events" in result
        assert "run_silver_merge" in result
        assert "run_gold_features" in result


class TestCodeRendererRunAll:
    """Tests for render_run_all method."""

    def test_render_run_all(self, sample_pipeline_config):
        """Should render run_all.py with full pipeline execution."""
        renderer = CodeRenderer()
        result = renderer.render_run_all(sample_pipeline_config)

        assert "run_pipeline" in result
        assert "run_bronze_parallel" in result
        assert "start_mlflow_ui" in result
        assert "setup_experiments_dir" in result


class TestCodeRendererWorkflow:
    """Tests for render_workflow method."""

    def test_render_workflow(self, sample_pipeline_config):
        """Should render valid workflow.json."""
        renderer = CodeRenderer()
        result = renderer.render_workflow(sample_pipeline_config)

        # Should be valid JSON
        workflow = json.loads(result)

        assert "name" in workflow
        assert "tasks" in workflow
        assert len(workflow["tasks"]) >= 4  # bronze tasks + silver + gold + training


class TestCodeRendererFeast:
    """Tests for Feast-related rendering methods."""

    def test_render_feast_config(self, sample_pipeline_config):
        """Should render feature_store.yaml."""
        renderer = CodeRenderer()
        result = renderer.render_feast_config(sample_pipeline_config)

        assert "project: test_pipeline" in result
        assert "registry:" in result
        assert "provider: local" in result

    def test_render_feast_features(self, sample_pipeline_config):
        """Should render features.py with Feast definitions."""
        renderer = CodeRenderer()
        result = renderer.render_feast_features(sample_pipeline_config)

        assert "Entity" in result
        assert "FeatureView" in result
        assert "FileSource" in result
        assert "customer_features" in result


class TestCodeRendererScoring:
    """Tests for render_scoring method."""

    def test_render_scoring(self, sample_pipeline_config):
        """Should render run_scoring.py."""
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)

        assert "run_scoring" in result
        assert "get_scoring_data" in result
        assert "load_best_model" in result
        assert "compute_validation_metrics" in result

    def test_render_scoring_uses_existing_holdout(self, sample_pipeline_config):
        """Scoring should use existing holdout, not create new one."""
        renderer = CodeRenderer()
        result = renderer.render_scoring(sample_pipeline_config)

        # Should check for existing holdout
        assert "ORIGINAL_COLUMN" in result
        assert "Holdout must be created in silver layer" in result or "No holdout found" in result


class TestCodeRendererDashboard:
    """Tests for render_dashboard method."""

    def test_render_dashboard(self, sample_pipeline_config):
        """Should render scoring_dashboard.ipynb."""
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)

        # Should be valid JSON (notebook format)
        notebook = json.loads(result)

        assert "cells" in notebook
        assert "metadata" in notebook
        assert notebook["nbformat"] >= 4

    def test_render_dashboard_has_shap(self, sample_pipeline_config):
        """Dashboard should include SHAP explanations."""
        renderer = CodeRenderer()
        result = renderer.render_dashboard(sample_pipeline_config)

        assert "shap" in result.lower()
        assert "Explainer" in result or "explainer" in result


class TestTemplatesExist:
    """Tests to verify all expected templates exist."""

    def test_all_required_templates_exist(self):
        """All required templates should be present in TEMPLATES."""
        required_templates = [
            "config.py.j2",
            "bronze.py.j2",
            "silver.py.j2",
            "gold.py.j2",
            "training.py.j2",
            "runner.py.j2",
            "run_all.py.j2",
            "workflow.json.j2",
            "feature_store.yaml.j2",
            "features.py.j2",
            "run_scoring.py.j2",
            "scoring_dashboard.ipynb.j2",
        ]

        for template_name in required_templates:
            assert template_name in TEMPLATES, f"Template {template_name} not found"

    def test_templates_are_non_empty(self):
        """All templates should have content."""
        for name, content in TEMPLATES.items():
            assert len(content) > 0, f"Template {name} is empty"


class TestRendererEdgeCases:
    """Edge case tests for renderer."""

    def test_render_config_without_feast(self, minimal_pipeline_config):
        """Should render config when feast config is None."""
        renderer = CodeRenderer()
        result = renderer.render_config(minimal_pipeline_config)

        # Should use defaults
        assert "FEAST_FEATURE_VIEW" in result
        assert "minimal_pipeline_features" in result  # Default naming

    def test_render_bronze_with_parquet_format(self, sample_event_source_config):
        """Should handle parquet format correctly."""
        bronze_config = BronzeLayerConfig(source=sample_event_source_config, transformations=[])
        renderer = CodeRenderer()
        result = renderer.render_bronze("events", bronze_config)

        assert "read_parquet" in result
