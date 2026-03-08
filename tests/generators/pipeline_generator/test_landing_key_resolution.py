from __future__ import annotations

import ast

import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import DatabricksCodeRenderer
from customer_retention.generators.pipeline_generator.models import (
    KeyResolutionStepConfig,
    LandingLayerConfig,
    SourceConfig,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


def _landing_config(key_resolution_steps=None):
    source = SourceConfig(
        name="subscription",
        path="/data/subscription.csv",
        format="csv",
        entity_key="ACCOUNT_ID",
        time_column="event_date",
        is_event_level=True,
    )
    return LandingLayerConfig(
        source=source,
        raw_source_path="/raw/subscription.csv",
        raw_source_format="csv",
        entity_column="ACCOUNT_ID",
        time_column="event_date",
        target_column="churn",
        key_resolution_steps=key_resolution_steps or [],
    )


class TestLandingLayerConfigKeyResolution:
    def test_field_exists_and_defaults_to_empty(self):
        config = _landing_config()
        assert config.key_resolution_steps == []

    def test_field_accepts_steps(self):
        steps = [
            KeyResolutionStepConfig(
                bridge_dataset="contract",
                source_key="CONTRACT_ID",
                bridge_key="CONTRACT_ID",
                resolve_column="ACCOUNT_ID",
            )
        ]
        config = _landing_config(key_resolution_steps=steps)
        assert len(config.key_resolution_steps) == 1
        assert config.key_resolution_steps[0].bridge_dataset == "contract"


class TestLocalRendererLandingKeyResolution:
    @pytest.fixture
    def renderer(self):
        return CodeRenderer()

    def test_generates_resolve_entity_key_function(self, renderer):
        steps = [
            KeyResolutionStepConfig(
                bridge_dataset="contract",
                source_key="CONTRACT_ID",
                bridge_key="CONTRACT_ID",
                resolve_column="ACCOUNT_ID",
            )
        ]
        config = _landing_config(key_resolution_steps=steps)
        result = renderer.render_landing("subscription", config)
        assert "def resolve_entity_key(" in result
        assert "contract" in result
        assert "CONTRACT_ID" in result
        assert "ACCOUNT_ID" in result
        ast.parse(result)

    def test_calls_resolve_in_run_landing(self, renderer):
        steps = [
            KeyResolutionStepConfig(
                bridge_dataset="contract",
                source_key="CONTRACT_ID",
                bridge_key="CONTRACT_ID",
                resolve_column="ACCOUNT_ID",
            )
        ]
        config = _landing_config(key_resolution_steps=steps)
        result = renderer.render_landing("subscription", config)
        run_fn_start = result.index("def run_landing_")
        run_fn = result[run_fn_start:]
        assert "resolve_entity_key" in run_fn
        ast.parse(result)

    def test_omits_resolution_when_steps_empty(self, renderer):
        config = _landing_config()
        result = renderer.render_landing("subscription", config)
        assert "resolve_entity_key" not in result
        ast.parse(result)

    def test_drops_bridge_key_when_different_from_source_key(self, renderer):
        steps = [
            KeyResolutionStepConfig(
                bridge_dataset="contract",
                source_key="LOCAL_CONTRACT_ID",
                bridge_key="CONTRACT_ID",
                resolve_column="ACCOUNT_ID",
            )
        ]
        config = _landing_config(key_resolution_steps=steps)
        result = renderer.render_landing("subscription", config)
        assert 'drop(columns=["CONTRACT_ID"])' in result
        ast.parse(result)

    def test_multi_step_resolution(self, renderer):
        steps = [
            KeyResolutionStepConfig(
                bridge_dataset="contract",
                source_key="CONTRACT_ID",
                bridge_key="CONTRACT_ID",
                resolve_column="OPPORTUNITY_ID",
            ),
            KeyResolutionStepConfig(
                bridge_dataset="opportunity",
                source_key="OPPORTUNITY_ID",
                bridge_key="OPPORTUNITY_ID",
                resolve_column="ACCOUNT_ID",
            ),
        ]
        config = _landing_config(key_resolution_steps=steps)
        result = renderer.render_landing("subscription", config)
        assert "contract" in result
        assert "opportunity" in result
        ast.parse(result)


class TestDatabricksRendererLandingKeyResolution:
    @pytest.fixture
    def renderer(self):
        return DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")

    def test_generates_resolve_entity_key_function(self, renderer):
        steps = [
            KeyResolutionStepConfig(
                bridge_dataset="contract",
                source_key="CONTRACT_ID",
                bridge_key="CONTRACT_ID",
                resolve_column="ACCOUNT_ID",
            )
        ]
        config = _landing_config(key_resolution_steps=steps)
        result = renderer.render_landing("subscription", config)
        assert "def resolve_entity_key(" in result
        assert "contract" in result
        assert "CONTRACT_ID" in result
        assert "ACCOUNT_ID" in result
        ast.parse(result)

    def test_calls_resolve_in_run_landing(self, renderer):
        steps = [
            KeyResolutionStepConfig(
                bridge_dataset="contract",
                source_key="CONTRACT_ID",
                bridge_key="CONTRACT_ID",
                resolve_column="ACCOUNT_ID",
            )
        ]
        config = _landing_config(key_resolution_steps=steps)
        result = renderer.render_landing("subscription", config)
        run_fn_start = result.index("def run_landing()")
        run_fn = result[run_fn_start:]
        assert "resolve_entity_key" in run_fn
        ast.parse(result)

    def test_omits_resolution_when_steps_empty(self, renderer):
        config = _landing_config()
        result = renderer.render_landing("subscription", config)
        assert "resolve_entity_key" not in result
        ast.parse(result)

    def test_uses_spark_read_delta_table(self, renderer):
        steps = [
            KeyResolutionStepConfig(
                bridge_dataset="contract",
                source_key="CONTRACT_ID",
                bridge_key="CONTRACT_ID",
                resolve_column="ACCOUNT_ID",
            )
        ]
        config = _landing_config(key_resolution_steps=steps)
        result = renderer.render_landing("subscription", config)
        assert "landing_table(" in result
        ast.parse(result)


class TestDatabricksRunnerLandingOrder:
    @pytest.fixture
    def renderer(self):
        return DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")

    def test_runner_orders_bridge_before_dependent(self, renderer):
        from customer_retention.generators.pipeline_generator.models import (
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
        )

        entity_source = SourceConfig(
            name="account", path="/data/account.csv", format="csv",
            entity_key="ACCOUNT_ID",
        )
        contract_source = SourceConfig(
            name="contract", path="/data/contract.csv", format="csv",
            entity_key="ACCOUNT_ID", time_column="start_date", is_event_level=True,
        )
        subscription_source = SourceConfig(
            name="subscription", path="/data/subscription.csv", format="csv",
            entity_key="ACCOUNT_ID", time_column="event_date", is_event_level=True,
        )

        contract_landing = LandingLayerConfig(
            source=contract_source,
            raw_source_path="/raw/contract.csv",
            raw_source_format="csv",
            entity_column="ACCOUNT_ID",
            time_column="start_date",
            target_column="churn",
        )
        subscription_landing = LandingLayerConfig(
            source=subscription_source,
            raw_source_path="/raw/subscription.csv",
            raw_source_format="csv",
            entity_column="ACCOUNT_ID",
            time_column="event_date",
            target_column="churn",
            key_resolution_steps=[
                KeyResolutionStepConfig(
                    bridge_dataset="contract",
                    source_key="CONTRACT_ID",
                    bridge_key="CONTRACT_ID",
                    resolve_column="ACCOUNT_ID",
                )
            ],
        )

        config = PipelineConfig(
            name="test_pipeline",
            target_column="churn",
            sources=[entity_source, contract_source, subscription_source],
            bronze={},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(),
            output_dir="/output",
            landing={
                "subscription": subscription_landing,
                "contract": contract_landing,
            },
        )
        result = renderer.render_runner(config)
        contract_pos = result.index("landing_contract")
        subscription_pos = result.index("landing_subscription")
        assert contract_pos < subscription_pos
        ast.parse(result)
