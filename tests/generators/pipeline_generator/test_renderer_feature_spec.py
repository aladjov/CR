import ast

import pytest

from customer_retention.generators.pipeline_generator.models import (
    GoldLayerConfig,
    PipelineConfig,
    SilverLayerConfig,
    SourceConfig,
    TrainingConfig,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


@pytest.fixture
def renderer():
    return CodeRenderer()


@pytest.fixture
def minimal_config():
    return PipelineConfig(
        name="test_pipeline",
        target_column="churn",
        sources=[SourceConfig(name="a", path="a.csv", format="csv", entity_key="id")],
        bronze={},
        silver=SilverLayerConfig(),
        gold=GoldLayerConfig(),
        output_dir="/tmp/out",
        composite_name="test_abc",
        training=TrainingConfig(),
    )


def _render_training(renderer: CodeRenderer, config: PipelineConfig) -> str:
    return renderer.render_training(config)


class TestTrainingTemplateWithFeatureSpec:
    def test_spec_path_triggers_feature_spec_import(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        assert "from customer_retention.stages.modeling.feature_spec import FeatureSpec" in out

    def test_spec_path_triggers_verdict_gate(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        assert "is_hard_block()" in out
        assert "CR_OVERRIDE_UNSTABLE_SPEC" in out
        assert "verdict=unstable" in out

    def test_spec_path_triggers_schema_parity_check(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        assert "FeatureSpec parity violation" in out

    def test_spec_path_filters_to_selected_features_in_order(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        assert "X[list(_SPEC.selected_features)].copy()" in out
        assert "feature_names = list(_SPEC.selected_features)" in out

    def test_spec_path_uses_filter_before_drop_no_errors_kwarg(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        assert 'errors="ignore"' not in out.split("_leakage_drops")[1].split("\n")[0:3][1] if "_leakage_drops" in out else True
        assert "X.drop(columns=_leakage_drops)" in out

    def test_spec_path_writes_production_diagnostics(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        assert "production_diagnostics_path" in out
        assert '"run_type": "production"' in out

    def test_spec_path_sets_production_split_constants(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        assert "PRODUCTION_TEST_SIZE" in out
        assert "PRODUCTION_RANDOM_STATE" in out

    def test_production_internal_split_test_size_configurable(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        minimal_config.training = TrainingConfig(production_internal_split_test_size=0.25)
        out = _render_training(renderer, minimal_config)
        assert "PRODUCTION_TEST_SIZE = 0.25" in out

    def test_splitter_uses_production_test_size_when_spec_present(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        assert "test_size=PRODUCTION_TEST_SIZE" in out

    def test_splitter_uses_config_test_size_when_no_spec(self, renderer, minimal_config):
        minimal_config.feature_spec_path = None
        minimal_config.training = TrainingConfig(test_size=0.2)
        out = _render_training(renderer, minimal_config)
        assert "test_size=0.2" in out
        assert "test_size=PRODUCTION_TEST_SIZE" not in out

    def test_renders_valid_python_with_spec(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/some/path/feature_spec.yaml"
        out = _render_training(renderer, minimal_config)
        ast.parse(out)

    def test_renders_valid_python_without_spec(self, renderer, minimal_config):
        minimal_config.feature_spec_path = None
        out = _render_training(renderer, minimal_config)
        ast.parse(out)

    def test_no_spec_omits_spec_code(self, renderer, minimal_config):
        minimal_config.feature_spec_path = None
        out = _render_training(renderer, minimal_config)
        assert "FeatureSpec.load" not in out
        assert "is_hard_block" not in out

    def test_no_spec_keeps_runtime_drop_block(self, renderer, minimal_config):
        minimal_config.feature_spec_path = None
        out = _render_training(renderer, minimal_config)
        assert "_runtime_drops" in out
