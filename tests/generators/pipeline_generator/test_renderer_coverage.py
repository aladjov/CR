import ast
import json

import pytest

import customer_retention.generators.pipeline_generator.renderer as renderer_mod
from customer_retention.generators.pipeline_generator.models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    BronzeLayerConfig,
    FeastConfig,
    GoldLayerConfig,
    LifecycleConfig,
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
    action_description,
    collect_imports,
    group_steps,
    provenance_docstring,
    provenance_docstring_block,
    provenance_key,
    render_step_call,
)


@pytest.fixture
def renderer():
    return CodeRenderer()


@pytest.fixture
def sample_source_config():
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
    return SilverLayerConfig(
        joins=[
            {
                "right_source": "events",
                "left_keys": ["customer_id"],
                "right_keys": ["customer_id"],
                "how": "left",
            }
        ],
        aggregations=[],
    )


@pytest.fixture
def sample_gold_config():
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
    return PipelineConfig(
        name="test_pipeline",
        target_column="churned",
        sources=[sample_source_config, sample_event_source_config],
        bronze={
            "customers": sample_bronze_config,
        },
        bronze_event={
            "events": BronzeEventConfig(
                source=sample_event_source_config,
                entity_column="customer_id",
                time_column="event_date",
                aggregation=AggregationWindowConfig(
                    windows=["30d"],
                    value_columns=["amount"],
                    agg_funcs=["sum"],
                ),
            ),
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
    def test_inline_loader_get_source_found(self):
        templates = {"test.j2": "Hello {{ name }}"}
        loader = InlineLoader(templates)

        source, name, uptodate = loader.get_source(None, "test.j2")

        assert source == "Hello {{ name }}"
        assert name == "test.j2"
        assert uptodate() is True

    def test_inline_loader_get_source_not_found(self):
        templates = {"test.j2": "Hello"}
        loader = InlineLoader(templates)

        with pytest.raises(Exception, match="Template .* not found"):
            loader.get_source(None, "nonexistent.j2")


class TestCodeRendererConfig:
    def test_render_config_with_all_options(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)

        assert 'PIPELINE_NAME = "test_pipeline"' in result
        assert 'TARGET_COLUMN = "churned"' in result
        assert 'ITERATION_ID = "iter_001"' in result
        assert 'PARENT_ITERATION_ID = "iter_000"' in result
        assert 'RECOMMENDATIONS_HASH = "abc123"' in result
        assert "experiments" in result

    def test_render_config_minimal(self, renderer, minimal_pipeline_config):
        result = renderer.render_config(minimal_pipeline_config)

        assert 'PIPELINE_NAME = "minimal_pipeline"' in result
        assert "ITERATION_ID = None" in result
        assert "RECOMMENDATIONS_HASH = None" in result

    def test_render_config_has_path_functions(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)

        assert "def get_bronze_path" in result
        assert "def get_silver_path" in result
        assert "def get_gold_path" in result
        assert "get_feast_data_path" in result

    def test_feast_data_path_aliases_gold_path(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)

        assert "get_feast_data_path = get_gold_path" in result


class TestCodeRendererBronze:
    def test_render_bronze_with_transformations(self, renderer, sample_bronze_config):
        result = renderer.render_bronze("customers", sample_bronze_config)

        assert 'SOURCE_NAME = "customers"' in result
        assert "def load_customers" in result
        assert "def apply_transformations" in result
        assert "def run_bronze_entity_customers" in result

        assert "apply_impute_null" in result
        assert "apply_cap_outlier" in result
        assert "apply_type_cast" in result
        assert "TransformExecutor" not in result

    def test_render_bronze_no_transformations(self, renderer, sample_source_config):
        bronze_config = BronzeLayerConfig(source=sample_source_config, transformations=[])
        result = renderer.render_bronze("customers", bronze_config)

        assert "def load_customers" in result
        assert "def apply_transformations" in result
        assert "return df" in result

    def test_render_bronze_parquet_format_uses_read_parquet(self, renderer):
        parquet_source = SourceConfig(
            name="events",
            path="data/events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="created_at",
            is_event_level=False,
        )
        bronze_config = BronzeLayerConfig(source=parquet_source, transformations=[])
        result = renderer.render_bronze("events", bronze_config)

        assert "pd.read_parquet(str(path))" in result


class TestCodeRendererSilver:
    def test_render_silver_with_joins(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)

        assert "def merge_sources" in result
        assert "merged.merge" in result
        assert "left_on=['customer_id']" in result or 'left_on=["customer_id"]' in result
        assert 'how="left"' in result

    def test_render_silver_has_holdout_creation(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)

        assert "create_holdout_mask" in result
        assert "holdout_fraction" in result


class TestCodeRendererGold:
    def test_render_gold_with_encodings_and_scalings(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)

        assert "def apply_encodings" in result
        assert "def apply_scaling" in result

        assert "apply_one_hot_encode" in result or "FittedEncoder" in result
        assert "FittedScaler" in result
        assert "ArtifactStore" in result
        assert "TransformExecutor" not in result

    def test_render_gold_standardizes_timestamp(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)

        assert "add_feast_timestamp" in result
        assert "materialize_to_feast" not in result
        assert "aggregation_reference_date" in result

    def test_render_gold_no_dual_write(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)

        assert "materialize_to_feast" not in result
        assert result.count("get_delta") >= 1

    def test_render_gold_has_load_gold(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)

        assert "def load_gold" in result
        assert "get_gold_path" in result
        assert "get_delta" in result

    def test_render_gold_train_subset_fitting(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)

        assert "def apply_fitted_transforms" in result
        assert "_train_mask" in result
        assert "temporal_quantile" in result
        assert "_cutoff" in result

    def test_render_gold_no_module_level_store_when_fitted(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)

        fitted_func_pos = result.index("def apply_fitted_transforms")
        before_func = result[:fitted_func_pos]
        assert "_store = ArtifactStore(" not in before_func
        assert "ArtifactStore.from_manifest" not in before_func


class TestCodeRendererTraining:
    def test_render_training_with_feast(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)

        assert "get_training_data_from_feast" in result or "get_training_data" in result
        assert "FeatureStore" in result
        assert "get_historical_features" in result

    def test_render_training_loads_from_gold(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)

        assert "get_gold_path" in result or "get_feast_data_path" in result

    def test_render_training_excludes_original_columns(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)

        assert "original_" in result
        assert "startswith" in result

    def test_render_training_has_model_functions(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)

        assert "train_xgboost" in result
        assert "LogisticRegression" in result
        assert "RandomForestClassifier" in result
        assert "compute_metrics" in result

    def test_compute_metrics_guards_single_class(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "np.unique" in result or "unique" in result
        assert "zero_division" in result

    def test_log_model_before_log_metrics_sklearn(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        sklearn_log_model_pos = result.find("mlflow.sklearn.log_model")
        log_metrics_pos = result.find("mlflow.log_metrics")
        assert sklearn_log_model_pos > 0
        assert log_metrics_pos > 0
        assert sklearn_log_model_pos < log_metrics_pos

    def test_log_model_before_log_metrics_xgboost(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        xgb_section_start = result.find('run_name="xgboost"')
        xgb_section = result[xgb_section_start:]
        xgb_log_model_pos = xgb_section.find("mlflow.xgboost.log_model")
        xgb_log_metrics_pos = xgb_section.find("mlflow.log_metrics")
        assert xgb_log_model_pos > 0
        assert xgb_log_metrics_pos > 0
        assert xgb_log_model_pos < xgb_log_metrics_pos

    def test_training_imports_numpy(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "import numpy as np" in result

    def test_xgboost_autolog_disabled(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "autolog(disable=True)" in result

    def test_best_auc_initialized_below_zero(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "best_auc = None, -1" in result or "best_auc = -1" in result


class TestSmoteTrainingTemplate:
    @pytest.fixture
    def smote_pipeline_config(self, sample_pipeline_config):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        config = sample_pipeline_config
        config.training = TrainingConfig(imbalance_strategy="smote")
        return config

    def test_smote_uses_fit_transform(self, renderer, smote_pipeline_config):
        result = renderer.render_training(smote_pipeline_config)
        assert "fit_transform(" in result
        assert "handler.fit(" not in result

    def test_smote_unpacks_imbalance_result(self, renderer, smote_pipeline_config):
        result = renderer.render_training(smote_pipeline_config)
        assert ".X_resampled" in result
        assert ".y_resampled" in result

    def test_smote_specifies_strategy(self, renderer, smote_pipeline_config):
        result = renderer.render_training(smote_pipeline_config)
        assert "ImbalanceStrategy.SMOTE" in result

    def test_smote_imports_strategy_enum(self, renderer, smote_pipeline_config):
        result = renderer.render_training(smote_pipeline_config)
        assert "ImbalanceStrategy" in result

    def test_smote_template_is_valid_python(self, renderer, smote_pipeline_config):
        result = renderer.render_training(smote_pipeline_config)
        ast.parse(result)


class TestCodeRendererRunner:
    def test_render_runner(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)

        assert "run_pipeline" in result
        assert "run_bronze_entity_customers" in result
        assert "run_bronze_event_events" in result
        assert "run_silver_merge" in result
        assert "run_gold_features" in result

    def test_runner_training_step_number(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "[6/6] Training" in result or "4/4] Training" in result
        assert "[5/6] Training" not in result


class TestCodeRendererRunAll:
    def test_render_run_all(self, renderer, sample_pipeline_config):
        result = renderer.render_run_all(sample_pipeline_config)

        assert "run_pipeline" in result
        assert "run_bronze_entity_parallel" in result
        assert "start_mlflow_ui" in result
        assert "setup_experiments_dir" in result


class TestCodeRendererWorkflow:
    def test_render_workflow(self, renderer, sample_pipeline_config):
        result = renderer.render_workflow(sample_pipeline_config)

        workflow = json.loads(result)

        assert "name" in workflow
        assert "tasks" in workflow
        assert len(workflow["tasks"]) >= 4


class TestCodeRendererFeast:
    def test_render_feast_config(self, renderer, sample_pipeline_config):
        result = renderer.render_feast_config(sample_pipeline_config)

        assert "project: test_pipeline" in result
        assert "registry:" in result
        assert "provider: local" in result

    def test_render_feast_features(self, renderer, sample_pipeline_config):
        result = renderer.render_feast_features(sample_pipeline_config)

        assert "Entity" in result
        assert "FeatureView" in result
        assert "FileSource" in result
        assert "customer_features" in result

    def test_render_feast_features_source_points_at_gold(self, renderer, sample_pipeline_config):
        result = renderer.render_feast_features(sample_pipeline_config)

        assert "gold_features_" in result
        assert "../data/gold/" in result


class TestTemplatesExist:
    def test_all_required_templates_exist(self):
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
        ]

        for template_name in required_templates:
            assert template_name in TEMPLATES, f"Template {template_name} not found"

    def test_templates_are_non_empty(self):
        for name, content in TEMPLATES.items():
            assert len(content) > 0, f"Template {name} is empty"


class TestExplicitOpsAndComments:
    def test_bronze_template_has_rationale_comments(self, renderer, sample_bronze_config):
        result = renderer.render_bronze("customers", sample_bronze_config)
        assert "# Fill missing ages with 0" in result
        assert "# Cap income outliers" in result

    def test_bronze_template_has_action_comments(self, renderer, sample_bronze_config):
        result = renderer.render_bronze("customers", sample_bronze_config)
        assert "# impute nulls in age" in result
        assert "# cap outliers in income" in result

    def test_silver_template_has_rationale_comments(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            SilverLayerConfig,
        )

        source = SourceConfig(name="customers", path="data.csv", format="csv", entity_key="id")
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[source],
            bronze={},
            output_dir="./out",
            silver=SilverLayerConfig(
                derived_columns=[
                    TransformationStep(
                        type=PipelineTransformationType.DERIVED_COLUMN,
                        column="ratio_col",
                        parameters={"action": "ratio", "numerator": "a", "denominator": "b"},
                        rationale="Revenue efficiency",
                    ),
                ]
            ),
            gold=GoldLayerConfig(),
        )
        result = renderer.render_silver(config)
        assert "# Revenue efficiency" in result
        assert "apply_derived_ratio" in result

    def test_gold_template_has_rationale_comments(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "# One-hot encode region" in result
        assert "# Standardize income" in result

    def test_gold_template_has_action_comments(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "# one-hot encode region" in result
        assert "# standard-scale income" in result

    def test_gold_template_has_explicit_ops_calls(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "apply_one_hot_encode" in result or "FittedEncoder" in result
        assert "FittedScaler" in result

    def test_no_executor_dispatch(self, renderer, sample_pipeline_config, sample_bronze_config):
        bronze = renderer.render_bronze("customers", sample_bronze_config)
        silver = renderer.render_silver(sample_pipeline_config)
        gold = renderer.render_gold(sample_pipeline_config)
        for code in [bronze, silver, gold]:
            assert "_executor.apply_all" not in code
            assert "TransformExecutor" not in code

    def test_no_transformation_step_arrays_in_bronze_silver(
        self, renderer, sample_pipeline_config, sample_bronze_config
    ):
        bronze = renderer.render_bronze("customers", sample_bronze_config)
        silver = renderer.render_silver(sample_pipeline_config)
        for code in [bronze, silver]:
            assert "TRANSFORMATIONS = [" not in code
            assert "ENCODINGS = [" not in code
            assert "DERIVED_COLUMNS = [" not in code


class TestLifecycleOrdering:
    def test_landing_has_no_lifecycle_features(self, renderer):
        from customer_retention.generators.pipeline_generator.models import LandingLayerConfig

        source = SourceConfig(
            name="events",
            path="events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        landing_config = LandingLayerConfig(
            source=source,
            raw_source_path="events.parquet",
            raw_source_format="parquet",
            entity_column="customer_id",
            time_column="event_date",
            target_column="churn",
        )
        result = renderer.render_landing("events", landing_config)
        assert "add_recency_tenure" not in result
        assert "add_lifecycle_quadrant" not in result
        assert "add_cyclical_features" not in result
        assert "add_momentum_ratios" not in result
        assert "enrich_lifecycle" not in result

    def test_bronze_event_source_has_lifecycle_enrichment(self, renderer):
        from customer_retention.generators.pipeline_generator.models import LifecycleConfig

        source = SourceConfig(
            name="events",
            path="events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        bronze_config = BronzeLayerConfig(
            source=source,
            transformations=[],
            lifecycle=LifecycleConfig(
                include_lifecycle_quadrant=True,
                include_cyclical_features=True,
                include_recency_bucket=True,
                momentum_pairs=[],
            ),
            entity_column="customer_id",
            time_column="event_date",
        )
        result = renderer.render_bronze("events", bronze_config)
        assert "add_recency_tenure" in result
        assert "enrich_lifecycle" in result
        assert "add_lifecycle_quadrant" in result

    def test_bronze_entity_source_has_no_lifecycle(self, renderer, sample_bronze_config):
        result = renderer.render_bronze("customers", sample_bronze_config)
        assert "enrich_lifecycle" not in result
        assert "add_recency_tenure" not in result

    def test_bronze_lifecycle_runs_after_cleaning(self, renderer):
        from customer_retention.generators.pipeline_generator.models import LifecycleConfig

        source = SourceConfig(
            name="events",
            path="events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        bronze_config = BronzeLayerConfig(
            source=source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="amount",
                    parameters={"value": 0},
                    rationale="Fill nulls",
                ),
            ],
            lifecycle=LifecycleConfig(
                include_lifecycle_quadrant=True,
                include_cyclical_features=False,
                include_recency_bucket=True,
                momentum_pairs=[],
            ),
            entity_column="customer_id",
            time_column="event_date",
        )
        result = renderer.render_bronze("events", bronze_config)
        transform_pos = result.index("apply_transformations(df)")
        lifecycle_pos = result.index("enrich_lifecycle(df)")
        assert transform_pos < lifecycle_pos

    def test_bronze_lifecycle_loads_raw_events(self, renderer):
        from customer_retention.generators.pipeline_generator.models import LifecycleConfig

        source = SourceConfig(
            name="events",
            path="events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        bronze_config = BronzeLayerConfig(
            source=source,
            transformations=[],
            lifecycle=LifecycleConfig(
                include_lifecycle_quadrant=True,
                include_cyclical_features=False,
                include_recency_bucket=True,
                momentum_pairs=[],
            ),
            entity_column="customer_id",
            time_column="event_date",
        )
        result = renderer.render_bronze("events", bronze_config)
        assert "RAW_SOURCES" in result
        assert "_load_raw_events" in result


class TestBronzeEventPreShapingTransformations:
    def test_bronze_event_renders_pre_shaping_transformations(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
        )

        source = SourceConfig(
            name="events",
            path="events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        bronze_event_config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
            deduplicate=True,
            pre_shaping=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="amount",
                    parameters={"lower": 0, "upper": 10000},
                    rationale="Cap outliers in amount",
                ),
            ],
            aggregation=AggregationWindowConfig(windows=["7d", "30d"], value_columns=["amount"], agg_funcs=["sum"]),
        )
        result = renderer.render_bronze_event("events", bronze_event_config)
        assert "apply_cap_outlier" in result
        assert "def apply_pre_shaping" in result
        assert "def cap_outliers" in result
        assert "def apply_event_aggregation" in result
        assert "from customer_retention.transforms import" in result

    def test_bronze_event_no_transforms_when_empty(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
        )

        source = SourceConfig(
            name="events",
            path="events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        bronze_event_config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
            deduplicate=True,
            pre_shaping=[],
            aggregation=AggregationWindowConfig(windows=["7d", "30d"], value_columns=["amount"], agg_funcs=["sum"]),
        )
        result = renderer.render_bronze_event("events", bronze_event_config)
        assert "def apply_pre_shaping" in result
        assert "customer_retention.transforms" not in result

    def test_bronze_event_pre_shaping_is_valid_python(self, renderer):
        import ast

        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
        )

        source = SourceConfig(
            name="events",
            path="events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )
        bronze_event_config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="event_date",
            deduplicate=True,
            pre_shaping=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="amount",
                    parameters={"lower": 0, "upper": 10000},
                    rationale="Cap outliers in amount",
                ),
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="quantity",
                    parameters={"value": 0},
                    rationale="Fill missing quantities",
                ),
            ],
            aggregation=AggregationWindowConfig(windows=["7d", "30d"], value_columns=["amount"], agg_funcs=["sum"]),
        )
        result = renderer.render_bronze_event("events", bronze_event_config)
        ast.parse(result)


class TestRendererEdgeCases:
    def test_render_config_without_feast(self, renderer, minimal_pipeline_config):
        result = renderer.render_config(minimal_pipeline_config)

        assert "FEAST_FEATURE_VIEW" in result
        assert "featureset_minimal_pipeline" in result

    def test_render_bronze_uses_delta(self, renderer, sample_event_source_config):
        bronze_config = BronzeLayerConfig(source=sample_event_source_config, transformations=[])
        result = renderer.render_bronze("events", bronze_config)

        assert "get_delta" in result


def _step(t, col="x", params=None, rationale="test"):
    return TransformationStep(type=t, column=col, parameters=params or {}, rationale=rationale)


class TestActionDescription:
    def test_impute_null(self):
        assert (
            action_description(_step(PipelineTransformationType.IMPUTE_NULL, "age", {"value": 0}))
            == "impute nulls in age with 0"
        )

    def test_cap_outlier(self):
        result = action_description(_step(PipelineTransformationType.CAP_OUTLIER, "income", {"lower": 0, "upper": 500}))
        assert result == "cap outliers in income to [0, 500]"

    def test_type_cast(self):
        assert (
            action_description(_step(PipelineTransformationType.TYPE_CAST, "zip", {"dtype": "str"}))
            == "cast zip to str"
        )

    def test_drop_column(self):
        assert action_description(_step(PipelineTransformationType.DROP_COLUMN, "tmp")) == "drop column tmp"

    def test_winsorize(self):
        result = action_description(
            _step(PipelineTransformationType.WINSORIZE, "v", {"lower_bound": 1, "upper_bound": 99})
        )
        assert result == "winsorize v to [1, 99]"

    def test_segment_aware_cap(self):
        result = action_description(_step(PipelineTransformationType.SEGMENT_AWARE_CAP, "v", {"n_segments": 3}))
        assert result == "segment-aware outlier cap on v (3 segments)"

    def test_log_transform(self):
        assert action_description(_step(PipelineTransformationType.LOG_TRANSFORM, "rev")) == "log-transform rev"

    def test_sqrt_transform(self):
        assert action_description(_step(PipelineTransformationType.SQRT_TRANSFORM, "rev")) == "sqrt-transform rev"

    def test_zero_inflation(self):
        assert (
            action_description(_step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "cnt"))
            == "handle zero-inflation in cnt"
        )

    def test_cap_then_log(self):
        assert (
            action_description(_step(PipelineTransformationType.CAP_THEN_LOG, "rev"))
            == "cap at p99 then log-transform rev"
        )

    def test_feature_select(self):
        assert (
            action_description(_step(PipelineTransformationType.FEATURE_SELECT, "junk"))
            == "drop junk (feature selection)"
        )

    def test_yeo_johnson(self):
        assert action_description(_step(PipelineTransformationType.YEO_JOHNSON, "v")) == "yeo-johnson transform v"

    def test_encode_one_hot(self):
        assert (
            action_description(_step(PipelineTransformationType.ENCODE, "region", {"method": "one_hot"}))
            == "one-hot encode region"
        )

    def test_encode_label(self):
        assert (
            action_description(_step(PipelineTransformationType.ENCODE, "tier", {"method": "label"}))
            == "label-encode tier"
        )

    def test_scale_standard(self):
        assert (
            action_description(_step(PipelineTransformationType.SCALE, "v", {"method": "standard"}))
            == "standard-scale v"
        )

    def test_scale_minmax(self):
        assert (
            action_description(_step(PipelineTransformationType.SCALE, "v", {"method": "minmax"})) == "min-max scale v"
        )

    def test_derived_ratio(self):
        result = action_description(
            _step(
                PipelineTransformationType.DERIVED_COLUMN,
                "r",
                {"action": "ratio", "numerator": "a", "denominator": "b"},
            )
        )
        assert result == "create r = a / b"

    def test_derived_interaction(self):
        result = action_description(
            _step(PipelineTransformationType.DERIVED_COLUMN, "ab", {"action": "interaction", "features": ["a", "b"]})
        )
        assert result == "create ab = a * b"

    def test_derived_composite_uses_join(self):
        result = action_description(
            _step(PipelineTransformationType.DERIVED_COLUMN, "avg", {"action": "composite", "columns": ["a", "b", "c"]})
        )
        assert result == "create avg = mean(a, b, c)"
        assert "mean(['a'" not in result


class TestRenderStepCall:
    def test_impute_null(self):
        result = render_step_call(_step(PipelineTransformationType.IMPUTE_NULL, "age", {"value": 0}))
        assert result == "apply_impute_null(df, 'age', value='0')"

    def test_cap_outlier(self):
        result = render_step_call(_step(PipelineTransformationType.CAP_OUTLIER, "income", {"lower": 0, "upper": 500}))
        assert result == "apply_cap_outlier(df, 'income', lower=0, upper=500)"

    def test_type_cast(self):
        result = render_step_call(_step(PipelineTransformationType.TYPE_CAST, "zip", {"dtype": "str"}))
        assert result == "apply_type_cast(df, 'zip', dtype='str')"

    def test_drop_column(self):
        assert render_step_call(_step(PipelineTransformationType.DROP_COLUMN, "tmp")) == "apply_drop_column(df, 'tmp')"

    def test_winsorize(self):
        result = render_step_call(
            _step(PipelineTransformationType.WINSORIZE, "v", {"lower_bound": 1, "upper_bound": 99})
        )
        assert result == "apply_winsorize(df, 'v', lower_bound=1, upper_bound=99)"

    def test_segment_aware_cap(self):
        result = render_step_call(_step(PipelineTransformationType.SEGMENT_AWARE_CAP, "v", {"n_segments": 3}))
        assert result == "apply_segment_aware_cap(df, 'v', n_segments=3)"

    def test_log_transform(self):
        assert render_step_call(_step(PipelineTransformationType.LOG_TRANSFORM, "v")) == "apply_log_transform(df, 'v')"

    def test_sqrt_transform(self):
        assert (
            render_step_call(_step(PipelineTransformationType.SQRT_TRANSFORM, "v")) == "apply_sqrt_transform(df, 'v')"
        )

    def test_zero_inflation(self):
        assert (
            render_step_call(_step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "v"))
            == "apply_zero_inflation_handling(df, 'v')"
        )

    def test_cap_then_log(self):
        assert render_step_call(_step(PipelineTransformationType.CAP_THEN_LOG, "v")) == "apply_cap_then_log(df, 'v')"

    def test_feature_select(self):
        assert (
            render_step_call(_step(PipelineTransformationType.FEATURE_SELECT, "v")) == "apply_feature_select(df, 'v')"
        )

    def test_yeo_johnson_fit(self):
        result = render_step_call(_step(PipelineTransformationType.YEO_JOHNSON, "v"), fit_mode=True)
        assert result == "FittedPowerTransform().fit_transform(df, 'v', _store)"

    def test_yeo_johnson_transform(self):
        result = render_step_call(_step(PipelineTransformationType.YEO_JOHNSON, "v"), fit_mode=False)
        assert result == "FittedPowerTransform().transform(df, 'v', _store)"

    def test_encode_one_hot(self):
        result = render_step_call(_step(PipelineTransformationType.ENCODE, "region", {"method": "one_hot"}))
        assert result == "apply_one_hot_encode(df, 'region')"

    def test_encode_label_fit(self):
        result = render_step_call(_step(PipelineTransformationType.ENCODE, "tier", {"method": "label"}), fit_mode=True)
        assert result == "FittedEncoder().fit_transform(df, 'tier', _store)"

    def test_encode_label_transform(self):
        result = render_step_call(_step(PipelineTransformationType.ENCODE, "tier", {"method": "label"}), fit_mode=False)
        assert result == "FittedEncoder().transform(df, 'tier', _store)"

    def test_scale_standard_fit(self):
        result = render_step_call(_step(PipelineTransformationType.SCALE, "v", {"method": "standard"}), fit_mode=True)
        assert result == "FittedScaler('standard').fit_transform(df, 'v', _store)"

    def test_scale_minmax_transform(self):
        result = render_step_call(_step(PipelineTransformationType.SCALE, "v", {"method": "minmax"}), fit_mode=False)
        assert result == "FittedScaler('minmax').transform(df, 'v', _store)"

    def test_derived_ratio(self):
        result = render_step_call(
            _step(
                PipelineTransformationType.DERIVED_COLUMN,
                "r",
                {"action": "ratio", "numerator": "a", "denominator": "b"},
            )
        )
        assert "apply_derived_ratio" in result
        assert "numerator='a'" in result

    def test_derived_interaction(self):
        result = render_step_call(
            _step(PipelineTransformationType.DERIVED_COLUMN, "ab", {"action": "interaction", "features": ["a", "b"]})
        )
        assert "apply_derived_interaction" in result
        assert "col_a='a'" in result

    def test_derived_composite(self):
        result = render_step_call(
            _step(PipelineTransformationType.DERIVED_COLUMN, "avg", {"action": "composite", "columns": ["a", "b"]})
        )
        assert "apply_derived_composite" in result

    def test_derived_composite_empty_columns_raises(self):
        with pytest.raises(ValueError, match="columns"):
            render_step_call(
                _step(PipelineTransformationType.DERIVED_COLUMN, "avg", {"action": "composite", "columns": []})
            )

    def test_derived_composite_missing_columns_key_raises(self):
        with pytest.raises(ValueError, match="columns"):
            render_step_call(
                _step(PipelineTransformationType.DERIVED_COLUMN, "avg", {"action": "composite"})
            )

    def test_unknown_type_raises_value_error(self):
        step = _step(PipelineTransformationType.AGGREGATE, "x")
        with pytest.raises(ValueError, match="Unknown transformation type"):
            render_step_call(step)


class TestCollectImports:
    def test_stateless_ops(self):
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
            _step(PipelineTransformationType.DROP_COLUMN, "c"),
        ]
        ops, fitted = collect_imports(steps, include_fitted=True)
        assert ops == {"apply_impute_null", "apply_log_transform", "apply_drop_column"}
        assert fitted == set()

    def test_fitted_types_included(self):
        steps = [
            _step(PipelineTransformationType.YEO_JOHNSON, "a"),
            _step(PipelineTransformationType.SCALE, "b", {"method": "standard"}),
            _step(PipelineTransformationType.ENCODE, "c", {"method": "label"}),
        ]
        ops, fitted = collect_imports(steps, include_fitted=True)
        assert "FittedPowerTransform" in fitted
        assert "FittedScaler" in fitted
        assert "FittedEncoder" in fitted

    def test_fitted_types_excluded(self):
        steps = [
            _step(PipelineTransformationType.YEO_JOHNSON, "a"),
            _step(PipelineTransformationType.SCALE, "b", {"method": "standard"}),
        ]
        ops, fitted = collect_imports(steps, include_fitted=False)
        assert fitted == set()

    def test_one_hot_encode_is_ops(self):
        steps = [_step(PipelineTransformationType.ENCODE, "region", {"method": "one_hot"})]
        ops, fitted = collect_imports(steps, include_fitted=True)
        assert "apply_one_hot_encode" in ops
        assert fitted == set()

    def test_derived_column_variants(self):
        steps = [
            _step(PipelineTransformationType.DERIVED_COLUMN, "r", {"action": "ratio"}),
            _step(PipelineTransformationType.DERIVED_COLUMN, "i", {"action": "interaction"}),
            _step(PipelineTransformationType.DERIVED_COLUMN, "c", {"action": "composite"}),
        ]
        ops, fitted = collect_imports(steps, include_fitted=True)
        assert ops == {"apply_derived_ratio", "apply_derived_interaction", "apply_derived_composite"}


class TestGroupSteps:
    def test_groups_by_type(self):
        steps = [
            _step(PipelineTransformationType.DROP_COLUMN, "a"),
            _step(PipelineTransformationType.IMPUTE_NULL, "b", {"value": 0}),
            _step(PipelineTransformationType.DROP_COLUMN, "c"),
        ]
        groups = group_steps(steps)
        assert len(groups) == 2
        names = [name for name, _ in groups]
        assert "drop_unusable_columns" in names
        assert "impute_remaining_nulls" in names

    def test_preserves_order_of_first_occurrence(self):
        steps = [
            _step(PipelineTransformationType.DROP_COLUMN, "a"),
            _step(PipelineTransformationType.IMPUTE_NULL, "b", {"value": 0}),
            _step(PipelineTransformationType.DROP_COLUMN, "c"),
        ]
        groups = group_steps(steps)
        assert groups[0][0] == "drop_unusable_columns"
        assert groups[1][0] == "impute_remaining_nulls"

    def test_derived_column_subgroups_by_action(self):
        steps = [
            _step(
                PipelineTransformationType.DERIVED_COLUMN,
                "r1",
                {"action": "ratio", "numerator": "a", "denominator": "b"},
            ),
            _step(PipelineTransformationType.DERIVED_COLUMN, "i1", {"action": "interaction", "features": ["a", "b"]}),
            _step(
                PipelineTransformationType.DERIVED_COLUMN,
                "r2",
                {"action": "ratio", "numerator": "c", "denominator": "d"},
            ),
        ]
        groups = group_steps(steps)
        names = [name for name, _ in groups]
        assert "create_ratio_features" in names
        assert "create_interaction_features" in names
        # ratios grouped together
        ratio_group = next(s for n, s in groups if n == "create_ratio_features")
        assert len(ratio_group) == 2

    def test_single_type_returns_one_group(self):
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "a", {"value": 0}),
            _step(PipelineTransformationType.IMPUTE_NULL, "b", {"value": 0}),
        ]
        groups = group_steps(steps)
        assert len(groups) == 1
        assert groups[0][0] == "impute_remaining_nulls"

    def test_empty_returns_empty(self):
        assert group_steps([]) == []


class TestGroupedTemplateOutput:
    @pytest.fixture
    def renderer(self):
        return CodeRenderer()

    @pytest.fixture
    def _source(self):
        return SourceConfig(
            name="customers",
            path="data.csv",
            format="csv",
            entity_key="id",
        )

    def test_bronze_groups_by_operation_type(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                _step(PipelineTransformationType.DROP_COLUMN, "junk"),
                _step(PipelineTransformationType.IMPUTE_NULL, "age", {"value": 0}),
                _step(PipelineTransformationType.WINSORIZE, "income", {"lower_bound": 0, "upper_bound": 1e6}),
                _step(PipelineTransformationType.DROP_COLUMN, "tmp"),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        assert "def drop_unusable_columns(df: pd.DataFrame)" in result
        assert "def impute_remaining_nulls(df: pd.DataFrame)" in result
        assert "def winsorize_outliers(df: pd.DataFrame)" in result
        # orchestrator calls all sub-functions
        assert "def apply_transformations" in result
        lines = result.split("\n")
        in_orchestrator = False
        called = []
        for line in lines:
            if "def apply_transformations" in line:
                in_orchestrator = True
            elif in_orchestrator and line.strip().startswith("def "):
                break
            elif in_orchestrator and "df = " in line:
                called.append(line.strip())
        assert any("drop_unusable_columns" in c for c in called)
        assert any("impute_remaining_nulls" in c for c in called)
        assert any("winsorize_outliers" in c for c in called)

    def test_gold_groups_transformations(self, renderer, _source):
        gold = GoldLayerConfig(
            transformations=[
                _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "cnt"),
                _step(PipelineTransformationType.LOG_TRANSFORM, "rev"),
                _step(PipelineTransformationType.SQRT_TRANSFORM, "qty"),
                _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "amt"),
            ]
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        assert "def handle_zero_inflation(df: pd.DataFrame)" in result
        assert "def apply_log_transforms(df: pd.DataFrame)" in result
        assert "def apply_sqrt_transforms(df: pd.DataFrame)" in result
        assert "def apply_gold_transformations" in result

    def test_silver_groups_derived_columns(self, renderer, _source):
        silver = SilverLayerConfig(
            derived_columns=[
                _step(
                    PipelineTransformationType.DERIVED_COLUMN,
                    "r1",
                    {"action": "ratio", "numerator": "a", "denominator": "b"},
                ),
                _step(
                    PipelineTransformationType.DERIVED_COLUMN, "i1", {"action": "interaction", "features": ["x", "y"]}
                ),
            ]
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=silver,
            gold=GoldLayerConfig(),
            output_dir="./out",
        )
        result = renderer.render_silver(config)
        assert "def create_ratio_features(df: pd.DataFrame)" in result
        assert "def create_interaction_features(df: pd.DataFrame)" in result
        assert "def create_derived_columns" in result

    def test_single_type_still_creates_named_function(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                _step(PipelineTransformationType.IMPUTE_NULL, "a", {"value": 0}),
                _step(PipelineTransformationType.IMPUTE_NULL, "b", {"value": 0}),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        assert "def impute_remaining_nulls(df: pd.DataFrame)" in result

    def test_empty_transformations_no_sub_functions(self, renderer, _source):
        bronze = BronzeLayerConfig(source=_source, transformations=[])
        result = renderer.render_bronze("customers", bronze)
        assert "def apply_transformations" in result
        # no sub-function definitions before apply_transformations
        assert "def drop_unusable_columns" not in result
        assert "def impute_remaining_nulls" not in result

    def test_generated_bronze_is_valid_python(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                _step(PipelineTransformationType.DROP_COLUMN, "junk"),
                _step(PipelineTransformationType.IMPUTE_NULL, "age", {"value": 0}),
                _step(PipelineTransformationType.WINSORIZE, "income", {"lower_bound": 0, "upper_bound": 1e6}),
                _step(PipelineTransformationType.SEGMENT_AWARE_CAP, "rev", {"n_segments": 3}),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        ast.parse(result)

    def test_generated_gold_is_valid_python(self, renderer, _source):
        gold = GoldLayerConfig(
            transformations=[
                _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "cnt"),
                _step(PipelineTransformationType.LOG_TRANSFORM, "rev"),
                _step(PipelineTransformationType.YEO_JOHNSON, "val"),
            ]
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        ast.parse(result)

    def test_generated_bronze_with_provenance_is_valid_python(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="income",
                    parameters={"lower": 0, "upper": 1000000},
                    rationale="Cap outliers",
                    source_notebook="02_source_integrity",
                ),
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="age",
                    parameters={"value": 0},
                    rationale="Fill nulls",
                    source_notebook="02_source_integrity",
                ),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        ast.parse(result)

    def test_generated_gold_with_provenance_is_valid_python(self, renderer, _source):
        gold = GoldLayerConfig(
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.LOG_TRANSFORM,
                    column="rev",
                    parameters={},
                    rationale="Log transform revenue",
                    source_notebook="05_relationship_analysis",
                ),
            ],
            encodings=[
                TransformationStep(
                    type=PipelineTransformationType.ENCODE,
                    column="region",
                    parameters={"method": "one_hot"},
                    rationale="Encode region",
                    source_notebook="05_relationship_analysis",
                ),
            ],
            scalings=[
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="income",
                    parameters={"method": "standard"},
                    rationale="Scale income",
                    source_notebook="05_relationship_analysis",
                ),
            ],
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        ast.parse(result)


class TestProvenanceDocstring:
    def test_with_notebook_and_section(self):
        step = TransformationStep(
            type=PipelineTransformationType.CAP_OUTLIER,
            column="amount",
            parameters={},
            rationale="Cap outliers",
            source_notebook="02_source_integrity",
        )
        result = provenance_docstring(step)
        assert "Source Integrity" in result
        assert "Global Outlier Detection" in result
        assert "docs/02_source_integrity.html#2.8-Global-Outlier-Detection" in result

    def test_with_notebook_no_mapped_section(self):
        step = TransformationStep(
            type=PipelineTransformationType.AGGREGATE,
            column="x",
            parameters={},
            rationale="test",
            source_notebook="05_custom",
        )
        result = provenance_docstring(step)
        assert "Custom" in result
        assert "docs/05_custom.html" in result
        assert "#" not in result.split(".html")[1] if ".html" in result else True

    def test_without_notebook_uses_default(self):
        step = TransformationStep(
            type=PipelineTransformationType.CAP_OUTLIER,
            column="amount",
            parameters={},
            rationale="Cap outliers",
        )
        result = provenance_docstring(step)
        assert "Source Integrity" in result
        assert "Global Outlier Detection" in result
        assert "docs/02_source_integrity.html#2.8-Global-Outlier-Detection" in result

    def test_no_provenance_for_unmapped_type(self):
        step = TransformationStep(
            type=PipelineTransformationType.AGGREGATE,
            column="x",
            parameters={},
            rationale="test",
        )
        result = provenance_docstring(step)
        assert result == ""

    def test_docstring_block_deduplicates(self):
        steps = [
            TransformationStep(
                type=PipelineTransformationType.DROP_COLUMN,
                column="a",
                parameters={},
                rationale="drop a",
            ),
            TransformationStep(
                type=PipelineTransformationType.DROP_COLUMN,
                column="b",
                parameters={},
                rationale="drop b",
            ),
        ]
        result = provenance_docstring_block(steps)
        assert '"""' in result
        assert result.count("Source Integrity") == 1

    def test_docstring_block_empty_for_unmapped(self):
        steps = [
            TransformationStep(
                type=PipelineTransformationType.AGGREGATE,
                column="x",
                parameters={},
                rationale="test",
            ),
        ]
        result = provenance_docstring_block(steps)
        assert result == ""

    def test_set_docs_base_uses_absolute_file_uri(self):
        renderer = CodeRenderer()
        renderer.set_docs_base("/tmp/my_project/experiments")
        step = TransformationStep(
            type=PipelineTransformationType.CAP_OUTLIER,
            column="x",
            parameters={},
            rationale="test",
        )
        result = provenance_docstring(step)
        assert result.startswith("Source Integrity Global Outlier Detection")
        assert "file://" in result
        assert "/docs/02_source_integrity.html#2.8-Global-Outlier-Detection" in result
        renderer_mod._docs_base = "docs"

    def test_set_docs_base_none_uses_relative(self):
        renderer = CodeRenderer()
        renderer.set_docs_base(None)
        step = TransformationStep(
            type=PipelineTransformationType.CAP_OUTLIER,
            column="x",
            parameters={},
            rationale="test",
        )
        result = provenance_docstring(step)
        assert "docs/02_source_integrity.html" in result


class TestProvenanceInTemplates:
    @pytest.fixture
    def renderer(self):
        return CodeRenderer()

    @pytest.fixture
    def _source(self):
        return SourceConfig(
            name="customers",
            path="data.csv",
            format="csv",
            entity_key="id",
        )

    def test_bronze_template_has_provenance_docstring(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.CAP_OUTLIER,
                    column="income",
                    parameters={"lower": 0, "upper": 1000000},
                    rationale="Cap outliers",
                    source_notebook="02_source_integrity",
                ),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        assert '"""' in result
        assert "Source Integrity" in result
        assert "Global Outlier Detection" in result
        assert "docs/02_source_integrity.html#2.8-Global-Outlier-Detection" in result

    def test_bronze_template_omits_provenance_when_no_info(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.AGGREGATE,
                    column="x",
                    parameters={},
                    rationale="test",
                ),
            ],
        )
        step = bronze.transformations[0]
        assert provenance_docstring(step) == ""

    def test_gold_encodings_have_provenance(self, renderer, _source):
        gold = GoldLayerConfig(
            encodings=[
                TransformationStep(
                    type=PipelineTransformationType.ENCODE,
                    column="region",
                    parameters={"method": "one_hot"},
                    rationale="Encode region",
                    source_notebook="05_relationship_analysis",
                ),
            ]
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        assert '"""' in result
        assert "docs/05_relationship_analysis.html" in result

    def test_default_notebook_provenance_in_bronze(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.IMPUTE_NULL,
                    column="age",
                    parameters={"value": 0},
                    rationale="Fill nulls",
                ),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        assert '"""' in result
        assert "docs/02_source_integrity.html" in result


class TestProvenanceDedup:
    @pytest.fixture
    def renderer(self):
        return CodeRenderer()

    @pytest.fixture
    def _source(self):
        return SourceConfig(
            name="customers",
            path="data.csv",
            format="csv",
            entity_key="id",
        )

    def test_provenance_key_returns_notebook_section(self):
        step = TransformationStep(
            type=PipelineTransformationType.DROP_COLUMN,
            column="x",
            parameters={},
            rationale="test",
        )
        assert provenance_key(step) == "02_source_integrity:Missing Value Analysis"

    def test_provenance_key_empty_for_unmapped(self):
        step = TransformationStep(
            type=PipelineTransformationType.AGGREGATE,
            column="x",
            parameters={},
            rationale="test",
        )
        assert provenance_key(step) == ""

    def test_provenance_key_uses_override(self):
        step = TransformationStep(
            type=PipelineTransformationType.DROP_COLUMN,
            column="x",
            parameters={},
            rationale="test",
            source_notebook="99_custom",
        )
        result = provenance_key(step)
        assert result == "99_custom:Missing Value Analysis"

    def test_bronze_same_source_appears_once(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="a",
                    parameters={},
                    rationale="57.7% missing",
                ),
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="b",
                    parameters={},
                    rationale="58.0% missing",
                ),
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="c",
                    parameters={},
                    rationale="60.0% missing",
                ),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        assert result.count('"""') == 2  # one opening, one closing

    def test_bronze_different_sources_each_get_header(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="a",
                    parameters={},
                    rationale="drop junk",
                    source_notebook="02_source_integrity",
                ),
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="b",
                    parameters={},
                    rationale="drop other",
                    source_notebook="99_custom",
                ),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        assert "Source Integrity" in result
        assert "Custom" in result
        assert "docs/02_source_integrity.html" in result
        assert "docs/99_custom.html" in result

    def test_gold_encodings_deduped(self, renderer, _source):
        gold = GoldLayerConfig(
            encodings=[
                TransformationStep(
                    type=PipelineTransformationType.ENCODE,
                    column="region",
                    parameters={"method": "one_hot"},
                    rationale="Encode region",
                ),
                TransformationStep(
                    type=PipelineTransformationType.ENCODE,
                    column="tier",
                    parameters={"method": "one_hot"},
                    rationale="Encode tier",
                ),
                TransformationStep(
                    type=PipelineTransformationType.ENCODE,
                    column="status",
                    parameters={"method": "one_hot"},
                    rationale="Encode status",
                ),
            ]
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        encodings_start = result.index("def apply_encodings")
        encodings_end = result.index("def apply_scaling")
        encodings_block = result[encodings_start:encodings_end]
        # One docstring = one opening """ and one closing """
        assert encodings_block.count('"""') == 2

    def test_gold_scalings_deduped(self, renderer, _source):
        gold = GoldLayerConfig(
            scalings=[
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="income",
                    parameters={"method": "standard"},
                    rationale="Scale income",
                ),
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="age",
                    parameters={"method": "standard"},
                    rationale="Scale age",
                ),
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="score",
                    parameters={"method": "standard"},
                    rationale="Scale score",
                ),
            ]
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        fitted_start = result.index("def apply_fitted_transforms")
        fitted_end = result.index("def get_feature_version_tag")
        fitted_block = result[fitted_start:fitted_end]
        assert fitted_block.count('"""') == 2

    def test_provenance_header_precedes_rationale(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="a",
                    parameters={},
                    rationale="57.7% missing",
                ),
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="b",
                    parameters={},
                    rationale="58.0% missing",
                ),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        lines = result.split("\n")
        docstring_idx = next(i for i, line in enumerate(lines) if '"""' in line)
        rationale_idx = next(i for i, line in enumerate(lines) if "57.7% missing" in line)
        assert docstring_idx < rationale_idx

    def test_deduped_output_is_valid_python(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="a",
                    parameters={},
                    rationale="drop a",
                ),
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="b",
                    parameters={},
                    rationale="drop b",
                ),
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="c",
                    parameters={},
                    rationale="drop c",
                ),
            ],
        )
        bronze_result = renderer.render_bronze("customers", bronze)
        ast.parse(bronze_result)

        gold = GoldLayerConfig(
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.LOG_TRANSFORM,
                    column="rev",
                    parameters={},
                    rationale="Log transform",
                ),
                TransformationStep(
                    type=PipelineTransformationType.LOG_TRANSFORM,
                    column="qty",
                    parameters={},
                    rationale="Log transform",
                ),
            ],
            encodings=[
                TransformationStep(
                    type=PipelineTransformationType.ENCODE,
                    column="region",
                    parameters={"method": "one_hot"},
                    rationale="Encode",
                ),
            ],
            scalings=[
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="income",
                    parameters={"method": "standard"},
                    rationale="Scale",
                ),
            ],
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        gold_result = renderer.render_gold(config)
        ast.parse(gold_result)

    def test_rendered_provenance_has_docstring_link(self, renderer, _source):
        bronze = BronzeLayerConfig(
            source=_source,
            transformations=[
                TransformationStep(
                    type=PipelineTransformationType.DROP_COLUMN,
                    column="a",
                    parameters={},
                    rationale="drop a",
                ),
            ],
        )
        result = renderer.render_bronze("customers", bronze)
        assert '"""' in result
        assert "docs/02_source_integrity.html#2.5-Missing-Value-Analysis" in result
        ast.parse(result)


class TestConfigRawSourcesFormat:
    """RAW_SOURCES should use raw_source_format, not the aggregated source format."""

    @pytest.fixture
    def renderer(self):
        return CodeRenderer()

    @pytest.fixture
    def _source(self):
        return SourceConfig(
            name="events",
            path="events.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="event_date",
            is_event_level=True,
        )

    def test_config_raw_sources_uses_raw_source_format(self, renderer, _source):
        from customer_retention.generators.pipeline_generator.models import LandingLayerConfig

        landing_config = LandingLayerConfig(
            source=_source,
            raw_source_path="/data/raw_events.csv",
            raw_source_format="csv",
            entity_column="customer_id",
            time_column="event_date",
            target_column="churn",
        )
        config = PipelineConfig(
            name="test",
            target_column="churn",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(),
            output_dir="./out",
            landing={"events": landing_config},
        )
        result = renderer.render_config(config)
        # The RAW_SOURCES format should be "csv" (raw), not "parquet" (aggregated)
        assert '"format": "csv"' in result


class TestConfigSourcesPath:
    """SOURCES path should use FINDINGS_DIR / filename."""

    @pytest.fixture
    def renderer(self):
        return CodeRenderer()

    def test_config_sources_path_uses_raw_source_path(self, renderer):
        source = SourceConfig(
            name="customers",
            path="customers.parquet",
            format="parquet",
            entity_key="customer_id",
            raw_source_path="/data/findings/customers.parquet",
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(),
            output_dir="./out",
        )
        result = renderer.render_config(config)
        assert '"/data/findings/customers.parquet"' in result


class TestGoldEncodingsScalingsExports:
    """Gold template should export ENCODINGS and SCALINGS as TransformationStep lists."""

    @pytest.fixture
    def renderer(self):
        return CodeRenderer()

    @pytest.fixture
    def _source(self):
        return SourceConfig(
            name="customers",
            path="data.csv",
            format="csv",
            entity_key="id",
        )

    def test_gold_exports_encodings_and_scalings_lists(self, renderer, _source):
        gold = GoldLayerConfig(
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
            ],
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        assert "ENCODINGS = [" in result
        assert "SCALINGS = [" in result
        assert "TransformationStep(" in result
        assert "PipelineTransformationType.ENCODE" in result
        assert "PipelineTransformationType.SCALE" in result
        ast.parse(result)

    def test_gold_empty_encodings_scalings_still_valid(self, renderer, _source):
        gold = GoldLayerConfig(encodings=[], scalings=[])
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        assert "ENCODINGS = [" in result
        assert "SCALINGS = [" in result
        ast.parse(result)

    def test_gold_encodings_contain_correct_columns(self, renderer, _source):
        gold = GoldLayerConfig(
            encodings=[
                TransformationStep(
                    type=PipelineTransformationType.ENCODE,
                    column="status",
                    parameters={"method": "one_hot"},
                    rationale="Encode status",
                ),
            ],
            scalings=[
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column="age",
                    parameters={"method": "minmax"},
                    rationale="Scale age",
                ),
            ],
        )
        config = PipelineConfig(
            name="test",
            target_column="target",
            sources=[_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=gold,
            output_dir="./out",
        )
        result = renderer.render_gold(config)
        # Check ENCODINGS contains the right column
        enc_start = result.index("ENCODINGS = [")
        enc_end = result.index("]", enc_start) + 1
        enc_block = result[enc_start:enc_end]
        assert 'column="status"' in enc_block
        # Check SCALINGS contains the right column
        scl_start = result.index("SCALINGS = [")
        scl_end = result.index("]", scl_start) + 1
        scl_block = result[scl_start:scl_end]
        assert 'column="age"' in scl_block


class TestBronzeEntityRawSourceLookup:
    def test_load_raw_events_uses_base_source_name(self):
        import ast

        source = SourceConfig(
            name="orders",
            path="orders.parquet",
            format="parquet",
            entity_key="customer_id",
            time_column="order_date",
            is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="order_date",
            lifecycle=LifecycleConfig(include_recency_bucket=True),
        )
        result = CodeRenderer().render_bronze_entity("orders_aggregated", config, "orders_aggregated", "orders")
        ast.parse(result)
        assert 'RAW_SOURCES["orders"]' in result
        assert 'RAW_SOURCES["orders_aggregated"]' not in result


class TestSilverLoadsAggregatedBronze:
    def test_silver_loads_aggregated_path_for_event_sources(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert 'get_bronze_path(f"{name}_aggregated")' in result or "_aggregated" in result
        assert "is_event_level" in result


class TestBronzeEventCategoricalAggregation:
    def _make_bronze_event_config(self, categorical_columns=None, categorical_agg_funcs=None):
        source = SourceConfig(
            name="emails",
            path="data/emails.csv",
            format="csv",
            entity_key="customer_id",
            time_column="sent_date",
            is_event_level=True,
        )
        return BronzeEventConfig(
            source=source,
            entity_column="customer_id",
            time_column="sent_date",
            deduplicate=True,
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["send_hour"],
                agg_funcs=["sum", "mean"],
                categorical_columns=categorical_columns or ["direction", "status"],
                categorical_agg_funcs=categorical_agg_funcs or ["nunique", "mode"],
            ),
        )

    def test_bronze_event_aggregation_has_categorical_columns(self, renderer):
        config = self._make_bronze_event_config()
        result = renderer.render_bronze_event("emails", config)
        assert "CATEGORICAL_COLUMNS" in result
        assert "'direction'" in result
        assert "'status'" in result

    def test_bronze_event_aggregation_has_safe_mode(self, renderer):
        config = self._make_bronze_event_config()
        result = renderer.render_bronze_event("emails", config)
        assert "_safe_mode" in result
        assert "value_counts().idxmax()" in result

    def test_bronze_event_aggregation_has_nunique(self, renderer):
        config = self._make_bronze_event_config()
        result = renderer.render_bronze_event("emails", config)
        assert ".nunique()" in result
        assert "nunique" in result

    def test_bronze_event_aggregation_has_timezone_normalization(self, renderer):
        config = self._make_bronze_event_config()
        result = renderer.render_bronze_event("emails", config)
        assert "as_tz_naive" in result

    def test_bronze_event_aggregation_filters_non_numeric_columns(self, renderer):
        config = self._make_bronze_event_config()
        result = renderer.render_bronze_event("emails", config)
        assert "is_numeric_dtype" in result
        assert "numeric_value_columns" in result

    def test_bronze_event_aggregation_is_valid_python(self, renderer):
        config = self._make_bronze_event_config()
        result = renderer.render_bronze_event("emails", config)
        ast.parse(result)

    def test_bronze_event_aggregation_runtime_mixed_types(self, renderer):
        import pandas as pd

        config = self._make_bronze_event_config(
            categorical_columns=["direction"],
            categorical_agg_funcs=["nunique", "mode"],
        )
        result = renderer.render_bronze_event("emails", config)

        local_ns = {}
        mock_code = (
            "import pandas as pd\nimport numpy as np\n"
            "from pandas.api.types import is_numeric_dtype\n"
            "from customer_retention.core.compat import ensure_timestamp, as_tz_naive\n"
            "TARGET_COLUMN = 'churn'\n"
            "PRODUCTION_DIR = __import__('pathlib').Path('/tmp/test_prod')\n"
        )
        exec(mock_code, local_ns)

        df = pd.DataFrame(
            {
                "customer_id": ["A", "A", "B", "B", "B"],
                "sent_date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "send_hour": [10, 14, 9, 11, 16],
                "direction": ["outbound", "inbound", "outbound", "outbound", "inbound"],
            }
        )

        lines = result.split("\n")
        capture = False
        extracted_lines = []
        for line in lines:
            if line.startswith("ENTITY_COLUMN") or line.startswith("TIME_COLUMN"):
                extracted_lines.append(line)
            if line.startswith("def _parse_window(") or line.startswith("def _safe_mode("):
                capture = True
            if capture:
                extracted_lines.append(line)
            if line.startswith("AGGREGATION_WINDOWS"):
                extracted_lines.append(line)
                capture = False
            if line.startswith("VALUE_COLUMNS") or line.startswith("AGG_FUNCS"):
                extracted_lines.append(line)
            if line.startswith("CATEGORICAL_COLUMNS") or line.startswith("CATEGORICAL_AGG_FUNCS"):
                extracted_lines.append(line)
            if line.startswith("def apply_event_aggregation("):
                capture = True
                extracted_lines.append(line)
            if capture and line.strip() == "return df":
                extracted_lines.append(line)
                capture = False

        extracted = "\n".join(extracted_lines)
        exec(extracted, local_ns)
        result_df = local_ns["apply_event_aggregation"](df.copy())

        assert "send_hour_sum_7d" in result_df.columns
        assert "send_hour_mean_7d" in result_df.columns
        assert "direction_nunique_7d" in result_df.columns
        assert "direction_mode_7d" in result_df.columns
        assert "event_count_7d" in result_df.columns
        assert len(result_df) == 2


class TestDatetimeDerivationRendering:

    def test_landing_template_includes_datetime_derivation(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            DatetimeDerivationConfig,
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
            datetime_derivation=DatetimeDerivationConfig(
                source_columns=["signup_date", "last_login"],
                reference_column="feature_timestamp",
                mask_future_columns=["last_login"],
            ),
        )
        code = renderer.render_landing("orders", config)
        assert "derive_datetime_features" in code
        assert "derive_extra_datetime_features" in code
        assert "mask_future_columns" in code
        assert "'last_login'" in code
        assert "signup_date" in code

    def test_landing_template_omits_derivation_when_none(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
        )
        code = renderer.render_landing("orders", config)
        assert "derive_datetime_features" not in code

    def test_bronze_event_template_includes_datetime_derivation(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            DatetimeDerivationConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="events", path="events.csv", format="csv",
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
        assert "derive_extra_datetime_features" in code
        assert "response_at" in code

    def test_bronze_event_template_omits_derivation_when_none(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", time_column="event_date", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="event_date",
        )
        code = renderer.render_bronze_event("events", config)
        assert "derive_datetime_features" not in code


class TestLandingTemplateSafeToDatetime:

    def test_landing_uses_safe_to_datetime_for_feature_timestamp(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
        )
        code = renderer.render_landing("orders", config)
        assert "safe_to_datetime" in code
        assert 'pd.to_datetime(df[TIME_COLUMN]' not in code

    def test_landing_uses_safe_to_datetime_with_coalesce(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
            TimestampCoalesceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
            timestamp_coalesce=TimestampCoalesceConfig(
                datetime_columns_ordered=["updated_at", "created_at"],
            ),
        )
        code = renderer.render_landing("orders", config)
        assert "safe_to_datetime" in code
        assert 'pd.to_datetime(df["updated_at"]' not in code
        assert 'pd.to_datetime(df["created_at"]' not in code

    def test_landing_uses_safe_to_datetime_for_label_timestamp(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            LabelTimestampConfig,
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
            label_timestamp=LabelTimestampConfig(
                label_column="churn_date", fallback_window_days=180,
            ),
        )
        code = renderer.render_landing("orders", config)
        assert 'safe_to_datetime(df["churn_date"]' in code
        assert 'pd.to_datetime(df["churn_date"]' not in code

    def test_landing_imports_safe_to_datetime(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
        )
        code = renderer.render_landing("orders", config)
        assert "from customer_retention.core.compat import safe_to_datetime" in code


class TestLandingHistoryWindow:
    def test_landing_renders_history_window_with_both(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            HistoryWindowConfig,
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
            history_window=HistoryWindowConfig(
                time_column="order_date", upper_limit="2024-06-30",
                lookback_periods=26, cadence_days=7,
            ),
        )
        code = renderer.render_landing("orders", config)
        assert "apply_history_window" in code
        assert "2024-06-30" in code
        assert "26 * 7" in code
        compile(code, "landing_orders.py", "exec")

    def test_landing_renders_history_window_upper_only(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            HistoryWindowConfig,
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
            history_window=HistoryWindowConfig(
                time_column="order_date", upper_limit="2024-12-31",
            ),
        )
        code = renderer.render_landing("orders", config)
        assert "apply_history_window" in code
        assert "2024-12-31" in code
        assert "lookback_days" not in code
        compile(code, "landing_orders.py", "exec")

    def test_landing_renders_history_window_lookback_only(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            HistoryWindowConfig,
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
            history_window=HistoryWindowConfig(
                time_column="order_date", lookback_periods=52, cadence_days=7,
            ),
        )
        code = renderer.render_landing("orders", config)
        assert "apply_history_window" in code
        assert "52 * 7" in code
        assert "upper = df[time_col].max()" in code
        compile(code, "landing_orders.py", "exec")

    def test_landing_omits_history_window_when_none(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
        )

        source = SourceConfig(
            name="orders", path="orders.csv", format="csv",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.csv",
            raw_source_format="csv", entity_column="customer_id",
            time_column="order_date", target_column="churn",
        )
        code = renderer.render_landing("orders", config)
        assert "apply_history_window" not in code
        compile(code, "landing_orders.py", "exec")
