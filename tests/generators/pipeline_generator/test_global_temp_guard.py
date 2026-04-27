"""Codegen guard against `global_temp.*` raw_source paths.

Symptom in the field: NB00 stamps a session-scoped Spark global temp view
name (created by `register_temp_view` inside an SPS `@cr:user_code` cell)
into `multi_dataset_findings.yaml.raw_source_path`. The view vanishes when
the exploration kernel exits, so the generated Databricks landing notebook
fails on `spark.read.table("global_temp.sps_filtered_case")` inside a
fresh `dbutils.notebook.run` session.

The root-cause fix is in NB00 cell `915bcef8` — it now reads paths from
`_namespace.original_datasets` (pre-mutation lineage). This file gates the
defense-in-depth layer: even if a `global_temp.*` value somehow reaches the
generator (e.g. from a stale findings file produced before the NB00 fix),
the rendered Databricks `read_raw_source` must refuse it with an actionable
message instead of letting the failure surface mid-pipeline.
"""
from __future__ import annotations

import ast

import pytest

from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.models import (
    GoldLayerConfig,
    LandingLayerConfig,
    PipelineConfig,
    SilverLayerConfig,
    SourceConfig,
)


def _minimal_config(raw_source_path: str = "ml_catalog.retention.case_raw") -> PipelineConfig:
    src = SourceConfig(
        name="case", path=raw_source_path, format="delta",
        entity_key="entity_id", time_column="event_timestamp",
        is_event_level=False, raw_source_path=raw_source_path,
    )
    landing = LandingLayerConfig(
        source=src,
        raw_source_path=raw_source_path,
        raw_source_format="delta",
        entity_column="entity_id",
        time_column="event_timestamp",
        target_column="churn",
    )
    return PipelineConfig(
        name="test_pipeline", target_column="churn",
        sources=[src],
        bronze={}, bronze_event={},
        landing={"case": landing},
        silver=SilverLayerConfig(), gold=GoldLayerConfig(),
        output_dir="/tmp/out", composite_name="case__abc1234",
    )


class TestGlobalTempGuardInDatabricksConfig:
    def test_rendered_config_contains_global_temp_guard(self):
        renderer = DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")
        result = renderer.render_config(_minimal_config())
        assert 'path.startswith("global_temp.")' in result
        assert "session-scoped temp view" in result
        assert "register_temp_view" in result
        assert "add_landing_filter" in result

    def test_rendered_config_is_valid_python(self):
        renderer = DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")
        result = renderer.render_config(_minimal_config())
        ast.parse(result)

    def test_rendered_read_raw_source_function_definition(self):
        """The guard must appear inside `read_raw_source`, not a sibling helper.
        Compile the rendered config, locate the function's AST, walk the body —
        first statement must be the global_temp `if` raising `RuntimeError`.
        """
        renderer = DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")
        rendered = renderer.render_config(_minimal_config())
        tree = ast.parse(rendered)
        fn = next(
            (n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "read_raw_source"),
            None,
        )
        assert fn is not None, "read_raw_source not found in rendered config"
        first = fn.body[0]
        assert isinstance(first, ast.If), "first statement must be the global_temp guard"
        assert isinstance(first.body[0], ast.Raise)
        raise_call = first.body[0].exc
        assert isinstance(raise_call, ast.Call)
        assert getattr(raise_call.func, "id", None) == "RuntimeError"


class TestGuardRuntimeBehavior:
    def test_guard_raises_on_global_temp_path(self):
        """Compile the rendered config, exec it, then call read_raw_source
        with a `global_temp.*` path — must raise RuntimeError with the
        actionable message. We avoid spinning up Spark by giving the exec'd
        module a stub spark name; the guard runs before any spark.read call.
        """
        renderer = DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")
        rendered = renderer.render_config(_minimal_config())
        ns = {"spark": None, "__name__": "_rendered_config"}
        exec(compile(rendered, "<rendered_config>", "exec"), ns)
        read_raw_source = ns["read_raw_source"]
        with pytest.raises(RuntimeError, match="session-scoped temp view"):
            read_raw_source("global_temp.sps_filtered_case", "delta")

    def test_guard_does_not_raise_on_uc_table(self):
        renderer = DatabricksCodeRenderer(catalog="ml_catalog", schema="retention")
        rendered = renderer.render_config(_minimal_config())
        ns = {"__name__": "_rendered_config"}

        class _StubReader:
            def table(self, name):
                return f"<read.table {name}>"

        class _StubSpark:
            read = _StubReader()

        ns["spark"] = _StubSpark()
        exec(compile(rendered, "<rendered_config>", "exec"), ns)
        read_raw_source = ns["read_raw_source"]
        result = read_raw_source("ml_catalog.retention.case_raw", "delta")
        assert "ml_catalog.retention.case_raw" in result


class TestNB00BuildRegistryPrefersOriginalDatasets:
    """Verify NB00 cell `915bcef8` reads from `_namespace.original_datasets`.

    This is a static check on the notebook source (the cell is exploration-
    notebook code, not framework Python). Lifting the regression into a unit
    test catches accidental reverts of the lineage source.
    """

    def test_cell_reads_from_original_datasets(self):
        import json
        from pathlib import Path
        nb_path = Path(__file__).resolve().parents[3] / "exploration_notebooks" / "00_start_here.ipynb"
        nb = json.loads(nb_path.read_text())
        cell = next((c for c in nb["cells"] if c.get("id") == "915bcef8"), None)
        assert cell is not None, "build_dataset_registry cell not found"
        src = "".join(cell["source"])
        assert "_namespace.original_datasets" in src
        assert "_originals" in src
        assert "_originals.get(name, datasets[name])" in src

    def test_cell_fails_fast_on_unrecoverable_global_temp_leak(self):
        import json
        from pathlib import Path
        nb_path = Path(__file__).resolve().parents[3] / "exploration_notebooks" / "00_start_here.ipynb"
        nb = json.loads(nb_path.read_text())
        cell = next((c for c in nb["cells"] if c.get("id") == "915bcef8"), None)
        src = "".join(cell["source"])
        assert 'startswith("global_temp.")' in src
        assert "raise RuntimeError" in src
        assert "add_landing_filter" in src
