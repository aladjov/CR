from __future__ import annotations

import ast

import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import DatabricksCodeRenderer
from customer_retention.generators.pipeline_generator.models import (
    LandingLayerConfig,
    PipelineTransformationType,
    SourceConfig,
    TransformationStep,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


def _landing_config(*, filters=None, lifecycle_enrichments=None):
    source = SourceConfig(
        name="request",
        path="/data/request.csv",
        format="csv",
        entity_key="ACCOUNT_ID",
        time_column="created_date",
        is_event_level=True,
    )
    return LandingLayerConfig(
        source=source,
        raw_source_path="/raw/request.csv",
        raw_source_format="csv",
        entity_column="ACCOUNT_ID",
        time_column="created_date",
        target_column="churn",
        filters=filters or [],
        lifecycle_enrichments=lifecycle_enrichments or [],
    )


def _filter_step(predicate: str) -> TransformationStep:
    return TransformationStep(
        type=PipelineTransformationType.LANDING_FILTER,
        column="request",
        parameters={"predicate": predicate},
        rationale="user filter",
        source_notebook="NB00 § 0.4.6",
    )


def _enrichment_step(config_dict: dict) -> TransformationStep:
    return TransformationStep(
        type=PipelineTransformationType.LANDING_LIFECYCLE_ENRICHMENT,
        column="subscription",
        parameters={"config": config_dict},
        rationale="double stream",
        source_notebook="NB00",
    )


class TestLocalRendererLandingUserExtensions:
    @pytest.fixture
    def renderer(self):
        return CodeRenderer()

    def test_empty_landing_emits_no_filter_or_enrichment_lines(self, renderer):
        code = renderer.render_landing("request", _landing_config())
        assert "df.query(" not in code
        assert "enrich_lifecycle_dataset" not in code
        assert "LifecycleEnrichmentConfig" not in code
        ast.parse(code)

    def test_filter_emits_query_call_with_python_literal_predicate(self, renderer):
        code = renderer.render_landing(
            "request",
            _landing_config(filters=[_filter_step("ACCOUNT_ID.notna() and amount > 0")]),
        )
        assert "df = df.query('ACCOUNT_ID.notna() and amount > 0')" in code
        ast.parse(code)

    def test_filter_runs_after_renames_before_temporal_derivation(self, renderer):
        code = renderer.render_landing(
            "request",
            _landing_config(filters=[_filter_step("amount > 0")]),
        )
        query_idx = code.index("df = df.query(")
        derive_call_idx = code.index("df = derive_temporal_columns(df)")
        assert query_idx < derive_call_idx

    def test_lifecycle_enrichment_emits_helper_call(self, renderer):
        enrichment_cfg = {
            "enriched_view_name": "subscription_events",
            "parent_entity_key": "ACCOUNT_ID",
            "sub_entity_key": "SUBSCRIPTION_ID",
            "valid_from_column": "start_date",
            "valid_to_columns": ["end_date"],
        }
        code = renderer.render_landing(
            "subscription",
            _landing_config(lifecycle_enrichments=[_enrichment_step(enrichment_cfg)]),
        )
        assert "from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset" in code
        assert "LifecycleEnrichmentConfig.from_dict(" in code
        assert "'enriched_view_name': 'subscription_events'" in code
        ast.parse(code)

    def test_predicate_with_single_quote_still_python_safe(self, renderer):
        code = renderer.render_landing(
            "request",
            _landing_config(filters=[_filter_step("STATUS = 'active'")]),
        )
        ast.parse(code)
        assert "df = df.query" in code


class TestDatabricksRendererLandingUserExtensions:
    @pytest.fixture
    def renderer(self):
        return DatabricksCodeRenderer()

    def test_empty_landing_emits_no_filter_or_enrichment_lines(self, renderer):
        code = renderer.render_landing("request", _landing_config())
        assert "df.filter(" not in code
        assert "enrich_lifecycle_dataset" not in code

    def test_filter_emits_spark_filter_with_sql_predicate(self, renderer):
        code = renderer.render_landing(
            "request",
            _landing_config(filters=[_filter_step("ACCOUNT_ID IS NOT NULL AND amount > 0")]),
        )
        assert "df = df.filter('ACCOUNT_ID IS NOT NULL AND amount > 0')" in code

    def test_filter_runs_after_renames_before_timestamp_derivation(self, renderer):
        code = renderer.render_landing(
            "request",
            _landing_config(filters=[_filter_step("amount > 0")]),
        )
        filter_idx = code.index("df = df.filter(")
        derive_call_idx = code.index("df = derive_feature_timestamp(df)")
        assert filter_idx < derive_call_idx

    def test_lifecycle_enrichment_emits_helper_call(self, renderer):
        enrichment_cfg = {
            "enriched_view_name": "subscription_events",
            "parent_entity_key": "ACCOUNT_ID",
            "sub_entity_key": "SUBSCRIPTION_ID",
            "valid_from_column": "start_date",
            "valid_to_columns": ["end_date"],
        }
        code = renderer.render_landing(
            "subscription",
            _landing_config(lifecycle_enrichments=[_enrichment_step(enrichment_cfg)]),
        )
        assert "enrich_lifecycle_dataset(df, LifecycleEnrichmentConfig.from_dict(" in code
        assert "'enriched_view_name': 'subscription_events'" in code


class TestLandingEmissionOrderParityBetweenRenderers:
    """Local + Databricks renderers must keep user-extension steps in the
    same logical slot: after renames, before temporal derivation."""

    def test_both_renderers_place_filter_before_temporal_step(self):
        cfg = _landing_config(filters=[_filter_step("amount > 0")])
        local_code = CodeRenderer().render_landing("request", cfg)
        db_code = DatabricksCodeRenderer().render_landing("request", cfg)
        assert local_code.index("df = df.query(") < local_code.index("df = derive_temporal_columns(df)")
        assert db_code.index("df = df.filter(") < db_code.index("df = derive_feature_timestamp(df)")
