import ast
from dataclasses import replace

import pytest

from customer_retention.generators.pipeline_generator.models import (
    GoldLayerConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TransformationStep,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer, partition_gold_steps


@pytest.fixture
def renderer():
    return CodeRenderer()


@pytest.fixture
def basic_config():
    return PipelineConfig(
        name="test_pipeline",
        target_column="churned",
        sources=[SourceConfig(name="customers", path="customers.parquet", format="parquet", entity_key="customer_id")],
        bronze={},
        silver=SilverLayerConfig(),
        gold=GoldLayerConfig(
            encodings=[TransformationStep(type=PipelineTransformationType.ENCODE, column="region",
                                          parameters={"method": "label"}, rationale="Encode region")],
            scalings=[TransformationStep(type=PipelineTransformationType.SCALE, column="income",
                                         parameters={"method": "standard"}, rationale="Normalize income")],
        ),
        output_dir="./output",
        recommendations_hash="abc12345",
    )


@pytest.fixture
def fit_mode_true_config(basic_config):
    return replace(basic_config, fit_mode=True, artifacts_path="./experiments/artifacts/abc12345")


@pytest.fixture
def fit_mode_false_config(basic_config):
    return replace(basic_config, fit_mode=False, artifacts_path="./experiments/artifacts/abc12345")


class TestPipelineConfigFitMode:
    def test_config_has_fit_mode_default_true(self):
        config = PipelineConfig(
            name="test", target_column="target",
            sources=[SourceConfig(name="src", path="data.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir="./out"
        )
        assert config.fit_mode is True

    def test_config_has_artifacts_path_default_none(self):
        config = PipelineConfig(
            name="test", target_column="target",
            sources=[SourceConfig(name="src", path="data.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir="./out"
        )
        assert config.artifacts_path is None

    def test_config_accepts_fit_mode_false(self):
        config = PipelineConfig(
            name="test", target_column="target",
            sources=[SourceConfig(name="src", path="data.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir="./out",
            fit_mode=False, artifacts_path="./artifacts"
        )
        assert config.fit_mode is False
        assert config.artifacts_path == "./artifacts"


class TestGoldTemplateFitMode:
    def test_gold_template_fit_mode_true_imports_artifact_store(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "ArtifactStore" in code

    def test_gold_template_fit_mode_true_creates_store(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "_store = ArtifactStore(" in code

    def test_gold_template_fit_mode_true_uses_fit_transform(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "fit_transform" in code

    def test_gold_template_fit_mode_true_has_scaling(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "FittedScaler" in code

    def test_gold_template_fit_mode_true_saves_manifest(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "_store.save_manifest()" in code

    def test_gold_template_fit_mode_false_loads_store(self, renderer, fit_mode_false_config):
        code = renderer.render_gold(fit_mode_false_config)
        assert "ArtifactStore.from_manifest" in code

    def test_gold_template_fit_mode_false_uses_transform(self, renderer, fit_mode_false_config):
        code = renderer.render_gold(fit_mode_false_config)
        assert ".transform(" in code

    def test_gold_template_is_valid_python(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        ast.parse(code)

    def test_gold_template_fit_mode_false_is_valid_python(self, renderer, fit_mode_false_config):
        code = renderer.render_gold(fit_mode_false_config)
        ast.parse(code)


class TestConfigTemplateArtifactsPath:
    def test_config_template_includes_artifacts_path(self, renderer, fit_mode_true_config):
        code = renderer.render_config(fit_mode_true_config)
        assert "ARTIFACTS_PATH" in code or "artifacts" in code.lower()

    def test_config_template_is_valid_python(self, renderer, fit_mode_true_config):
        code = renderer.render_config(fit_mode_true_config)
        ast.parse(code)


class TestGoldTrainSubsetFitting:
    def test_gold_fit_mode_true_has_temporal_quantile(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "temporal_quantile" in code

    def test_gold_fit_mode_true_has_train_mask(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "_train_mask" in code

    def test_gold_fit_mode_true_has_fit_subset(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "_fit_subset" in code

    def test_gold_fit_mode_true_has_cutoff(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "_cutoff" in code

    def test_gold_fit_mode_true_transforms_all_after_fit(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        fit_pos = code.find("fit_transform")
        transform_pos = code.find(".transform(", fit_pos + 1)
        assert fit_pos > 0
        assert transform_pos > fit_pos

    def test_gold_fit_mode_false_has_no_train_mask(self, renderer, fit_mode_false_config):
        code = renderer.render_gold(fit_mode_false_config)
        assert "_train_mask" not in code
        assert "_fit_subset" not in code

    def test_gold_fit_mode_false_loads_from_manifest(self, renderer, fit_mode_false_config):
        code = renderer.render_gold(fit_mode_false_config)
        assert "ArtifactStore.from_manifest" in code

    def test_gold_has_apply_fitted_transforms_function(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "def apply_fitted_transforms" in code

    def test_gold_calls_apply_fitted_transforms(self, renderer, fit_mode_true_config):
        code = renderer.render_gold(fit_mode_true_config)
        assert "gold = apply_fitted_transforms(gold)" in code

    def test_gold_no_fitted_steps_no_train_mask(self, renderer):
        config = PipelineConfig(
            name="test", target_column="target",
            sources=[SourceConfig(name="src", path="data.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(),
            gold=GoldLayerConfig(
                encodings=[TransformationStep(type=PipelineTransformationType.ENCODE, column="region",
                                              parameters={"method": "one_hot"}, rationale="One-hot encode region")],
            ),
            output_dir="./out", fit_mode=True,
        )
        code = renderer.render_gold(config)
        assert "_train_mask" not in code
        assert "apply_fitted_transforms" not in code
        ast.parse(code)

    def test_gold_with_purge_gap_days(self, renderer):
        from customer_retention.generators.pipeline_generator.models import TrainingConfig
        config = PipelineConfig(
            name="test", target_column="target",
            sources=[SourceConfig(name="src", path="data.csv", format="csv", entity_key="id")],
            bronze={}, silver=SilverLayerConfig(),
            gold=GoldLayerConfig(
                scalings=[TransformationStep(type=PipelineTransformationType.SCALE, column="income",
                                              parameters={"method": "standard"}, rationale="Scale income")],
            ),
            output_dir="./out", fit_mode=True,
            training=TrainingConfig(test_size=0.25, purge_gap_days=30),
        )
        code = renderer.render_gold(config)
        assert "timedelta(days=30)" in code
        assert "1 - 0.25" in code
        ast.parse(code)


class TestPartitionGoldSteps:
    def test_stateless_transforms_exclude_yeo_johnson(self):
        gold = GoldLayerConfig(
            transformations=[
                TransformationStep(type=PipelineTransformationType.CAP_OUTLIER, column="a", parameters={"q": 0.99}, rationale="Cap"),
                TransformationStep(type=PipelineTransformationType.YEO_JOHNSON, column="b", parameters={}, rationale="Power transform"),
            ],
        )
        parts = partition_gold_steps(gold)
        assert len(parts["stateless_transforms"]) == 1
        assert parts["stateless_transforms"][0].column == "a"
        assert len(parts["fitted_transforms"]) == 1
        assert parts["fitted_transforms"][0].column == "b"

    def test_one_hot_goes_to_stateless(self):
        gold = GoldLayerConfig(
            encodings=[
                TransformationStep(type=PipelineTransformationType.ENCODE, column="region", parameters={"method": "one_hot"}, rationale="One-hot"),
                TransformationStep(type=PipelineTransformationType.ENCODE, column="city", parameters={"method": "label"}, rationale="Label"),
            ],
        )
        parts = partition_gold_steps(gold)
        assert len(parts["stateless_encodings"]) == 1
        assert parts["stateless_encodings"][0].column == "region"
        assert len(parts["fitted_encodings"]) == 1
        assert parts["fitted_encodings"][0].column == "city"

    def test_all_scalings_are_fitted(self):
        gold = GoldLayerConfig(
            scalings=[
                TransformationStep(type=PipelineTransformationType.SCALE, column="income", parameters={"method": "standard"}, rationale="Scale"),
                TransformationStep(type=PipelineTransformationType.SCALE, column="age", parameters={"method": "minmax"}, rationale="Scale"),
            ],
        )
        parts = partition_gold_steps(gold)
        assert len(parts["fitted_scalings"]) == 2
        assert len(parts["stateless_transforms"]) == 0

    def test_empty_gold_returns_all_empty(self):
        gold = GoldLayerConfig()
        parts = partition_gold_steps(gold)
        for key in ("stateless_transforms", "stateless_encodings", "fitted_transforms", "fitted_encodings", "fitted_scalings"):
            assert parts[key] == []


class TestGoldTemplateNoFitMode:
    def test_gold_template_default_behavior_when_no_fit_mode(self, renderer, basic_config):
        code = renderer.render_gold(basic_config)
        ast.parse(code)
