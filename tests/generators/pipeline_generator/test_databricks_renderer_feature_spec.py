import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.models import (
    GoldLayerConfig,
    PipelineConfig,
    SilverLayerConfig,
    SourceConfig,
    TrainingConfig,
)


@pytest.fixture
def renderer():
    return DatabricksCodeRenderer(catalog="main", schema="db")


@pytest.fixture
def minimal_config():
    return PipelineConfig(
        name="test_pipeline", target_column="churn",
        sources=[SourceConfig(name="a", path="a.csv", format="csv", entity_key="id")],
        bronze={}, silver=SilverLayerConfig(), gold=GoldLayerConfig(),
        output_dir="/tmp/out", composite_name="test_abc",
        training=TrainingConfig(),
    )


def _strip_magic(text: str) -> str:
    lines = text.split("\n")
    py_lines = [
        line for line in lines
        if not line.startswith("# MAGIC")
        and not line.startswith("# COMMAND")
        and not line.startswith("# Databricks")
    ]
    return "\n".join(py_lines)


class TestDatabricksTrainingFeatureSpec:
    def test_spec_triggers_import(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/path/feature_spec.yaml"
        out = renderer.render_training(minimal_config)
        assert "from customer_retention.stages.modeling.feature_spec import FeatureSpec" in out

    def test_spec_triggers_verdict_gate(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/path/feature_spec.yaml"
        out = renderer.render_training(minimal_config)
        assert "is_hard_block()" in out
        assert "CR_OVERRIDE_UNSTABLE_SPEC" in out

    def test_spec_triggers_schema_parity_check(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/path/feature_spec.yaml"
        out = renderer.render_training(minimal_config)
        assert "FeatureSpec parity violation" in out

    def test_spec_uses_native_spark_select(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/path/feature_spec.yaml"
        out = renderer.render_training(minimal_config)
        assert "df.select(*keep)" in out

    def test_spec_does_not_use_errors_ignore(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/path/feature_spec.yaml"
        out = renderer.render_training(minimal_config)
        assert 'errors="ignore"' not in out

    def test_spec_writes_production_diagnostics(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/path/feature_spec.yaml"
        out = renderer.render_training(minimal_config)
        assert "production_diagnostics_path" in out
        assert '"run_type": "production"' in out

    def test_spec_template_compiles(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/path/feature_spec.yaml"
        out = renderer.render_training(minimal_config)
        compile(_strip_magic(out), "<dbx-training>", "exec")

    def test_no_spec_template_compiles(self, renderer, minimal_config):
        minimal_config.feature_spec_path = None
        out = renderer.render_training(minimal_config)
        compile(_strip_magic(out), "<dbx-training-nospec>", "exec")

    def test_no_spec_omits_spec_code(self, renderer, minimal_config):
        minimal_config.feature_spec_path = None
        out = renderer.render_training(minimal_config)
        assert "_apply_feature_spec_gate" not in out
        assert "FeatureSpec.load" not in out

    def test_no_spec_keeps_runtime_drop_block(self, renderer, minimal_config):
        minimal_config.feature_spec_path = None
        out = renderer.render_training(minimal_config)
        assert "_runtime_drops" in out

    def test_splitter_uses_production_test_size_when_spec_present(self, renderer, minimal_config):
        minimal_config.feature_spec_path = "/path/feature_spec.yaml"
        out = renderer.render_training(minimal_config)
        assert "_temporal_split(assembled, PRODUCTION_TEST_SIZE)" in out
