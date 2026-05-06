"""FW-19 — entity-level bronze must read from the landing UC table /
landing Delta path, not from the raw upstream.

Pre-FW-19, ``databricks_bronze.py.j2``'s ``load_source`` called
``read_raw_source(...)`` against the original Snowflake/external upstream.
NB05 / NB06 / NB08 saw findings whose ``columns`` map enumerated the
post-landing schema (carrying ``CREATED_DATE_delta_hours``,
``days_since_CREATED_DATE``, etc.), so ``config.bronze[<ds>]
.transformations`` referenced derived columns. The generated bronze
script crashed at runtime with ``[UNRESOLVED_COLUMN.WITH_SUGGESTION]``
on the first reference. See ``docs/sps_nb10_runtime_patches_v3.md``
§FW-19 for the full failure trace and architecture context.

Path A (these tests): entity-level bronze reads from
``landing_table(SOURCE_NAME)`` (Databricks) / ``get_landing_path(...)``
(local-Feast) — same shape ``bronze_event`` already uses.

Event-bronze is untouched; it has been reading from landing since the
landing layer was first added to the framework. Tests below assert that
contract is preserved.
"""
from __future__ import annotations

import ast

import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


class TestDatabricksEntityBronzeReadsLanding:
    """Databricks template (cluster path)."""

    def test_load_source_reads_landing_table(self, bronze_with_impute):
        rendered = DatabricksCodeRenderer().render_bronze(
            "customers", bronze_with_impute
        )
        assert "spark.table(landing_table(SOURCE_NAME))" in rendered, (
            "FW-19: entity-level bronze must read from the landing UC table; "
            "rendered output is missing the spark.table(landing_table(...)) call."
        )

    def test_load_source_no_read_raw_source_call(self, bronze_with_impute):
        """``read_raw_source`` is the landing-stage helper (legitimately used
        in landing.py.j2 to ingest the upstream). It must NOT appear in
        entity-level bronze output, where its presence is the FW-19
        regression signature."""
        rendered = DatabricksCodeRenderer().render_bronze(
            "customers", bronze_with_impute
        )
        assert "read_raw_source(" not in rendered, (
            "FW-19 regression: entity-level bronze still emits a "
            "read_raw_source(...) call in load_source(). It must read the "
            "landing UC table instead."
        )

    def test_rendered_bronze_is_valid_python(self, bronze_with_impute):
        rendered = DatabricksCodeRenderer().render_bronze(
            "customers", bronze_with_impute
        )
        # Strip Databricks magic / cell separators that aren't valid Python
        # at module level. Same approach the existing renderer tests use.
        sanitized = "\n".join(
            ln for ln in rendered.splitlines()
            if not ln.lstrip().startswith(("# MAGIC", "# COMMAND", "# Databricks"))
        )
        ast.parse(sanitized)


class TestLocalFeastEntityBronzeReadsLanding:
    """Local-Feast template (papermill / single-machine path)."""

    def test_load_source_reads_landing_path(self, bronze_with_impute):
        rendered = CodeRenderer().render_bronze("customers", bronze_with_impute)
        assert "get_landing_path(SOURCE_NAME)" in rendered, (
            "FW-19: local-Feast entity-level bronze must read the Delta "
            "written by landing.py via get_landing_path(...)."
        )

    def test_load_source_does_not_open_raw_csv_or_parquet(self, bronze_with_impute):
        """Pre-FW-19 the local template called pd.read_csv / pd.read_parquet
        against ``SOURCES[SOURCE_NAME]["path"]`` (the raw upstream). Post-
        FW-19 the source is always a Delta produced by landing — the raw
        readers belong only to landing.py.j2."""
        rendered = CodeRenderer().render_bronze("customers", bronze_with_impute)
        # Locate just the load_<src>() function body to scope the assertion
        # — the template still imports SOURCES (used by other helpers).
        load_fn_idx = rendered.find("def load_customers():")
        assert load_fn_idx != -1, "load_<src>() function not found in rendered output"
        body = rendered[load_fn_idx:rendered.find("\n\n", load_fn_idx + 1)]
        assert "pd.read_csv" not in body
        assert "pd.read_parquet" not in body
        assert 'SOURCES[SOURCE_NAME]["path"]' not in body

    def test_rendered_bronze_is_valid_python(self, bronze_with_impute):
        rendered = CodeRenderer().render_bronze("customers", bronze_with_impute)
        ast.parse(rendered)

    def test_get_landing_path_helper_is_emitted_by_config(
        self, sample_pipeline_config_for_config_render
    ):
        """The bronze template imports ``get_landing_path`` from the config
        module; FW-19 added the helper alongside ``get_bronze_path`` /
        ``get_silver_path`` / ``get_gold_path``. Without this emission, the
        bronze script would fail at import time."""
        rendered = CodeRenderer().render_config(sample_pipeline_config_for_config_render)
        assert "def get_landing_path(" in rendered


class TestEventBronzeUnchangedByFw19:
    """Event-bronze ALWAYS read from landing. FW-19 is an entity-level
    fix; if these tests go red, FW-19 has bled into a layer it shouldn't."""

    def test_databricks_event_bronze_load_source_reads_landing(
        self, sample_event_bronze_event_config
    ):
        # render_bronze_event takes (source_name, BronzeEventConfig) and
        # returns the full event-bronze notebook source.
        rendered = DatabricksCodeRenderer().render_bronze_event(
            "orders", sample_event_bronze_event_config
        )
        assert "spark.table(landing_table(SOURCE_NAME))" in rendered

    def test_databricks_event_bronze_does_not_use_read_raw_source_for_load(
        self, sample_event_bronze_event_config
    ):
        rendered = DatabricksCodeRenderer().render_bronze_event(
            "orders", sample_event_bronze_event_config
        )
        # `read_raw_source` may legitimately appear in unrelated helpers;
        # what we're asserting is it isn't the load_source body.
        load_idx = rendered.find("def load_source():")
        assert load_idx != -1, "event bronze missing load_source() function"
        next_def = rendered.find("\ndef ", load_idx + 1)
        load_body = rendered[load_idx:next_def] if next_def != -1 else rendered[load_idx:]
        assert "read_raw_source(" not in load_body


# Shared fixture used by the local-config emission test. The conftest
# fixture set doesn't expose a simple PipelineConfig with the event_source
# pre-attached, so we build a minimal one inline.
@pytest.fixture
def sample_pipeline_config_for_config_render(
    entity_source, event_source, bronze_with_impute, silver_with_join, gold_with_encode_scale
):
    from customer_retention.generators.pipeline_generator.models import (
        BronzeLayerConfig,
        PipelineConfig,
        PipelineTransformationType,
        TransformationStep,
    )
    bronze2 = BronzeLayerConfig(
        source=event_source,
        transformations=[
            TransformationStep(
                type=PipelineTransformationType.CAP_OUTLIER,
                column="amount",
                parameters={"lower": 0, "upper": 10000},
                rationale="Cap outliers",
            )
        ],
    )
    return PipelineConfig(
        name="fw19_test",
        target_column="churn",
        sources=[entity_source, event_source],
        bronze={"customers": bronze_with_impute, "orders": bronze2},
        silver=silver_with_join,
        gold=gold_with_encode_scale,
        output_dir="/output/fw19_test",
    )


@pytest.fixture
def sample_event_bronze_event_config(event_source):
    """Minimal BronzeEventConfig sufficient to render the event-bronze
    template. The event template path takes its own config shape — keep
    the test surface narrow to the load_source body only."""
    from customer_retention.generators.pipeline_generator.models import (
        BronzeEventConfig,
    )
    return BronzeEventConfig(
        source=event_source,
        entity_column="customer_id",
        time_column="order_date",
    )
