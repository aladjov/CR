import ast
import json

import pytest

from customer_retention.generators.pipeline_generator.models import (
    BronzeLayerConfig,
    LandingLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    SourceConfig,
    TimestampCoalesceConfig,
    TransformationStep,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


@pytest.fixture
def renderer():
    return CodeRenderer()


@pytest.fixture
def sample_pipeline_config(entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale):
    bronze2 = BronzeLayerConfig(source=event_source, transformations=[
        TransformationStep(type=PipelineTransformationType.CAP_OUTLIER, column="amount", parameters={"lower": 0, "upper": 10000}, rationale="Cap outliers")
    ])
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[entity_source, event_source],
        bronze={"customers": bronze_with_impute, "orders": bronze2},
        silver=silver_with_join,
        gold=gold_with_encode_scale,
        output_dir="/output/test_pipeline"
    )


class TestCodeRendererInit:
    def test_renderer_creates_jinja_environment(self, renderer):
        assert renderer._env is not None


class TestRenderConfig:
    def test_render_config_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_config_includes_pipeline_name(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "test_pipeline" in result

    def test_render_config_includes_target_column(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "churn" in result

    def test_render_config_includes_sources(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        assert "customers" in result
        assert "orders" in result

    def test_render_config_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_config(sample_pipeline_config)
        ast.parse(result)


class TestRenderBronze:
    def test_render_bronze_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert isinstance(result, str)

    def test_render_bronze_includes_source_name(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "customers" in result

    def test_render_bronze_includes_transformations(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        assert "apply_impute_null" in result

    def test_render_bronze_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_bronze("customers", sample_pipeline_config.bronze["customers"])
        ast.parse(result)


class TestRenderSilver:
    def test_render_silver_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_silver_includes_join(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "merge" in result

    def test_render_silver_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        ast.parse(result)

    def test_simplified_merge_renames_entity_key_to_entity_id(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "TemporalMerger" not in result
        assert 'raw_entity_key' in result
        assert 'rename(columns={raw_entity_key: "entity_id"})' in result

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


class TestRenderGold:
    def test_render_gold_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_gold_includes_encoding(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "apply_one_hot_encode" in result or "FittedEncoder" in result

    def test_render_gold_includes_scaling(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "FittedScaler" in result

    def test_render_gold_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)


class TestRenderTraining:
    def test_render_training_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_training_includes_model(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "Classifier" in result or "model" in result.lower()

    def test_render_training_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestRenderRunner:
    def test_render_runner_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_runner_imports_bronze_modules(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        assert "bronze_customers" in result or "customers" in result

    def test_render_runner_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_runner(sample_pipeline_config)
        ast.parse(result)


class TestRenderWorkflow:
    def test_render_workflow_returns_string(self, renderer, sample_pipeline_config):
        result = renderer.render_workflow(sample_pipeline_config)
        assert isinstance(result, str)

    def test_render_workflow_is_valid_json(self, renderer, sample_pipeline_config):
        result = renderer.render_workflow(sample_pipeline_config)
        parsed = json.loads(result)
        assert "name" in parsed or "tasks" in parsed

    def test_render_workflow_has_dependencies(self, renderer, sample_pipeline_config):
        result = renderer.render_workflow(sample_pipeline_config)
        parsed = json.loads(result)
        assert "tasks" in parsed
        silver_task = next((t for t in parsed["tasks"] if "silver" in t["task_key"]), None)
        assert silver_task is not None
        assert "depends_on" in silver_task


class TestLocalLandingFeatureTimestampGuard:
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
        assert "fillna" in result
        assert "if TIME_COLUMN in" not in result
        ast.parse(result)


class TestRenderSilverTemporalMerge:
    @pytest.fixture
    def temporal_config(self, entity_source, event_source, bronze_with_impute, gold_with_encode_scale):
        from customer_retention.generators.pipeline_generator.models import (
            SilverLayerConfig,
            TemporalMergeSourceConfig,
        )

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
            silver=silver,
            gold=gold_with_encode_scale,
            output_dir="/output/test_pipeline",
        )

    def test_silver_contains_temporal_merger(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "TemporalMerger" in result

    def test_silver_contains_build_spine(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "build_spine" in result

    def test_silver_contains_grid_dates(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "2024-01-01" in result

    def test_silver_contains_merge_all(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "merge_all" in result

    def test_silver_is_valid_python(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        ast.parse(result)

    def test_silver_falls_back_without_grid_dates(self, renderer, sample_pipeline_config):
        result = renderer.render_silver(sample_pipeline_config)
        assert "TemporalMerger" not in result
        assert "merge" in result

    @pytest.fixture
    def temporal_config_with_key_resolution(self, entity_source, event_source, bronze_with_impute, gold_with_encode_scale):
        from customer_retention.generators.pipeline_generator.models import (
            KeyResolutionStepConfig,
            SilverLayerConfig,
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
            silver=silver,
            gold=gold_with_encode_scale,
            output_dir="/output/test_pipeline",
        )

    def test_silver_key_resolution_import(self, renderer, temporal_config_with_key_resolution):
        result = renderer.render_silver(temporal_config_with_key_resolution)
        assert "resolve_entity_keys" in result
        assert "KeyResolutionStep" in result

    def test_silver_key_resolution_in_meta(self, renderer, temporal_config_with_key_resolution):
        result = renderer.render_silver(temporal_config_with_key_resolution)
        assert "key_resolution_steps" in result

    def test_silver_key_resolution_before_rename(self, renderer, temporal_config_with_key_resolution):
        result = renderer.render_silver(temporal_config_with_key_resolution)
        resolve_pos = result.index("resolve_entity_keys")
        rename_pos = result.index("rename(columns=")
        assert resolve_pos < rename_pos

    def test_silver_no_key_resolution_import_when_empty(self, renderer, temporal_config):
        result = renderer.render_silver(temporal_config)
        assert "resolve_entity_keys" not in result

    def test_silver_with_key_resolution_valid_python(self, renderer, temporal_config_with_key_resolution):
        result = renderer.render_silver(temporal_config_with_key_resolution)
        ast.parse(result)


class TestRenderGoldAsOfDate:
    def test_gold_checks_as_of_date(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        assert "as_of_date" in result

    def test_gold_as_of_date_before_feature_timestamp(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        as_of_pos = result.index("as_of_date")
        feature_ts_pos = result.index("feature_timestamp")
        assert as_of_pos < feature_ts_pos

    def test_gold_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_gold(sample_pipeline_config)
        ast.parse(result)


class TestRenderTrainingDropAsOfDate:
    def test_training_drops_as_of_date(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "GOLD_METADATA_COLUMNS" in result

    def test_training_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        ast.parse(result)


class TestRenderTrainingCVProgress:
    @staticmethod
    def _with_cv(config, folds=5):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        config.training = config.training or TrainingConfig()
        config.training.production_cv_folds = folds
        return config

    def test_training_imports_cross_validator(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_cv(sample_pipeline_config))
        assert "CrossValidator" in result
        assert "CVStrategy" in result

    def test_training_has_on_fold_callback(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_cv(sample_pipeline_config))
        assert "on_fold_complete" in result
        assert "_on_fold" in result

    def test_training_no_cross_val_score_import(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "from sklearn.model_selection import cross_val_score" not in result

    def test_training_shows_per_fold_progress(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_cv(sample_pipeline_config))
        assert "CV fold" in result

    def test_training_shows_cv_summary(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_cv(sample_pipeline_config))
        assert "cv_mean" in result
        assert "cv_std" in result

    def test_training_without_cv_folds_skips_cv(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "CrossValidator" not in result
        assert "_on_fold" not in result


class TestRenderTrainingFullPanelFit:
    @staticmethod
    def _with_full_panel(config):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        config.training = config.training or TrainingConfig()
        config.training.production_full_panel_fit = True
        return config

    def test_full_panel_fit_skips_temporal_split(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        assert "splitter.split(" not in result
        assert "DataSplitter(" not in result

    def test_full_panel_fit_uses_full_panel_for_x_train(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        assert "X_train, y_train = X, y" in result
        assert "Full-panel fit:" in result

    def test_full_panel_fit_evaluates_on_train_set(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        assert "model.predict_proba(X_train)" in result
        assert "compute_metrics(y_train, y_proba, y_pred)" in result

    def test_full_panel_fit_tags_mlflow_mode(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        assert 'mlflow.set_tag("training_mode", "full_panel_fit")' in result

    def test_full_panel_fit_xgboost_uses_train_as_eval(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        assert "train_xgboost(X_train, y_train, X_train, y_train, feature_names)" in result

    def test_temporal_split_default_when_not_full_panel(self, renderer, sample_pipeline_config):
        result = renderer.render_training(sample_pipeline_config)
        assert "DataSplitter(" in result
        assert 'mlflow.set_tag("training_mode", "temporal_split")' in result
        assert "X_train, y_train = X, y" not in result

    def test_full_panel_fit_is_valid_python(self, renderer, sample_pipeline_config):
        result = renderer.render_training(self._with_full_panel(sample_pipeline_config))
        ast.parse(result)
