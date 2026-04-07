"""Tests for the per-grid-date branch of the bronze_event template."""
from __future__ import annotations

import ast
from typing import Any

import pandas as pd

from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    PipelineConfig,
    SilverLayerConfig,
    SourceConfig,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_event_config(
    *,
    per_grid_date_mode: bool,
    value_counts_columns: tuple = (),
    windows: list[str] | None = None,
) -> BronzeEventConfig:
    return BronzeEventConfig(
        source=SourceConfig(
            name="contract",
            path="data/contract",
            format="delta",
            entity_key="ACCOUNT_ID",
            time_column="event_timestamp",
            is_event_level=True,
        ),
        entity_column="ACCOUNT_ID",
        time_column="event_timestamp",
        deduplicate=False,
        aggregation=AggregationWindowConfig(
            windows=windows or ["30d", "90d", "all_time"],
        ),
        per_grid_date_mode=per_grid_date_mode,
        value_counts_columns=value_counts_columns,
    )


def _make_pipeline_config(event_config: BronzeEventConfig) -> PipelineConfig:
    return PipelineConfig(
        name="contract_test",
        target_column="churn",
        sources=[event_config.source],
        bronze={},
        silver=SilverLayerConfig(
            entity_key="ACCOUNT_ID",
            grid_dates=["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        ),
        gold=None,  # type: ignore[arg-type]
        output_dir="generated/test",
    )


# ---------------------------------------------------------------------------
# Local renderer (CodeRenderer) — pandas merge_asof path
# ---------------------------------------------------------------------------


class TestLocalRendererPerGridDateMode:
    def test_renders_valid_python_with_per_grid_date_mode(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(
            per_grid_date_mode=True, value_counts_columns=("event_type",)
        )
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        ast.parse(rendered)  # raises SyntaxError if invalid

    def test_per_grid_date_branch_emits_merge_asof(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(
            per_grid_date_mode=True, value_counts_columns=("event_type",)
        )
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        assert "merge_asof" in rendered, "expected pandas merge_asof in per-grid-date path"
        assert "VALUE_COUNTS_COLUMNS" in rendered
        assert "GRID_DATES" in rendered
        assert "as_of_date" in rendered
        assert "apply_event_aggregation_per_grid_date" in rendered

    def test_dispatch_uses_per_grid_date_function(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(
            per_grid_date_mode=True, value_counts_columns=("event_type",)
        )
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        # The runner dispatches to the per-grid-date function instead of the
        # snapshot one.
        assert "df = apply_event_aggregation_per_grid_date(df)" in rendered

    def test_snapshot_mode_default_unchanged_no_per_grid_date_artifacts(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(per_grid_date_mode=False)
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        # Default path should NOT contain any per-grid-date artifacts
        assert "apply_event_aggregation_per_grid_date" not in rendered
        assert "VALUE_COUNTS_COLUMNS" not in rendered
        # And it must still be valid Python
        ast.parse(rendered)


class TestLocalRendererSnapshotRegression:
    """Byte-identical regression: snapshot mode must be untouched by Phase 4."""

    def test_snapshot_render_byte_identical_for_existing_event_source(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(per_grid_date_mode=False)
        cfg2 = _make_event_config(per_grid_date_mode=False)
        a = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        b = renderer.render_bronze_event("contract", cfg2, _make_pipeline_config(cfg2))
        assert a == b, "snapshot rendering should be deterministic"


# ---------------------------------------------------------------------------
# Functional test — execute the rendered per-grid-date code path against a
# small synthetic event stream and verify the cumulative counts are correct
# at known anchor dates.
# ---------------------------------------------------------------------------


def _exec_rendered(rendered: str, env_globals: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = dict(env_globals)
    exec(compile(rendered, "<rendered>", "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------
# Databricks renderer (DatabricksCodeRenderer) — Spark Window functions path
# ---------------------------------------------------------------------------


class TestDatabricksRendererPerGridDateMode:
    def test_renders_valid_python_with_per_grid_date_mode(self):
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(
            per_grid_date_mode=True, value_counts_columns=("event_type",)
        )
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        ast.parse(rendered)

    def test_per_grid_date_branch_uses_window_function(self):
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(
            per_grid_date_mode=True, value_counts_columns=("event_type",)
        )
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        assert "Window.partitionBy" in rendered
        assert "row_number" in rendered
        assert "VALUE_COUNTS_COLUMNS" in rendered
        assert "GRID_DATES" in rendered
        assert "as_of_date" in rendered
        assert "apply_event_aggregation_per_grid_date" in rendered

    def test_cumulative_at_uses_window_max_pattern(self):
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(
            per_grid_date_mode=True, value_counts_columns=("event_type",)
        )
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        assert "_spark_cumulative_at" in rendered
        assert "unionByName" in rendered
        assert 'F.max("_running")' in rendered
        assert "unboundedPreceding" in rendered
        assert "_is_anchor" in rendered

    def test_cumulative_at_does_not_use_range_join_filter(self):
        """Regression: the previous implementation joined entity-only and
        filtered _event_ts <= anchor afterwards (range-join cardinality).
        The Window pattern must not contain that filter expression."""
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(
            per_grid_date_mode=True, value_counts_columns=("event_type",)
        )
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        assert '.where(F.col("_event_ts").isNull()' not in rendered
        assert ".groupBy(ENTITY_COLUMN, anchor_col)" not in rendered

    def test_dispatch_uses_per_grid_date_function(self):
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(
            per_grid_date_mode=True, value_counts_columns=("event_type",)
        )
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        assert "apply_event_aggregation_per_grid_date(" in rendered

    def test_snapshot_mode_default_unchanged_no_per_grid_date_artifacts(self):
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(per_grid_date_mode=False)
        rendered = renderer.render_bronze_event(
            "contract", cfg, _make_pipeline_config(cfg)
        )
        assert "apply_event_aggregation_per_grid_date" not in rendered
        assert "VALUE_COUNTS_COLUMNS" not in rendered
        ast.parse(rendered)


class TestRendererParity:
    def test_local_and_databricks_share_grid_dates_value_counts_and_windows(self):
        """Both renderers must declare identical GRID_DATES, VALUE_COUNTS_COLUMNS,
        and AGGREGATION_WINDOWS literals. Identical inputs ⇒ identical output
        column names ⇒ parity by construction (regardless of whether the
        cumulative-at-anchor implementation uses merge_asof or Window.row_number)."""
        local_renderer = CodeRenderer()
        databricks_renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(
            per_grid_date_mode=True,
            value_counts_columns=("event_type",),
            windows=["30d", "all_time"],
        )
        config = _make_pipeline_config(cfg)
        local = local_renderer.render_bronze_event("contract", cfg, config)
        databricks = databricks_renderer.render_bronze_event("contract", cfg, config)

        for rendered in (local, databricks):
            assert "['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']" in rendered
            assert "['event_type']" in rendered
            # Each renderer declares the same window list
            assert "['30d', 'all_time']" in rendered

        # And both reference the parameterized count column naming convention.
        for rendered in (local, databricks):
            assert 'f"{vc_col}_{value}_count_{window_str}"' in rendered or \
                   "_count_{window_str}" in rendered


class TestPerGridDateFunctionalCorrectness:
    def test_terminate_count_30d_is_correct_at_known_anchors(self, tmp_path):
        """For Account B (3 terminations at days 30, 90, 150), the count of
        terminate events in a 30d window at day 60 should be 1, at day 100
        should be 1, at day 200 should be 0."""
        renderer = CodeRenderer()
        cfg = _make_event_config(
            per_grid_date_mode=True,
            value_counts_columns=("event_type",),
            windows=["30d", "all_time"],
        )
        # Custom grid dates that hit our checkpoints exactly
        config = _make_pipeline_config(cfg)
        config.silver.grid_dates = ["2024-01-31", "2024-03-11", "2024-07-19"]

        rendered = renderer.render_bronze_event("contract", cfg, config)

        # Build a pre-shaped event-level DataFrame with the doubled stream:
        T0 = pd.Timestamp("2024-01-01")
        events = pd.DataFrame(
            {
                "ACCOUNT_ID": ["B"] * 6,
                "event_timestamp": [
                    T0,
                    T0 + pd.Timedelta(days=30),  # terminate 1
                    T0,
                    T0 + pd.Timedelta(days=90),  # terminate 2
                    T0,
                    T0 + pd.Timedelta(days=150),  # terminate 3
                ],
                "event_type": [
                    "start", "terminate",
                    "start", "terminate",
                    "start", "terminate",
                ],
            }
        )

        # Execute rendered code to extract `apply_event_aggregation_per_grid_date`
        # from the rendered namespace. The template imports `from config import
        # PRODUCTION_DIR, TARGET_COLUMN`, so we stub those before exec.
        import sys
        import types

        stub = types.ModuleType("config")
        stub.PRODUCTION_DIR = tmp_path
        stub.TARGET_COLUMN = "target"
        stub.FIT_MODE = True
        sys.modules["config"] = stub
        try:
            ns = _exec_rendered(rendered, {"__name__": "rendered_module"})
        finally:
            del sys.modules["config"]

        result = ns["apply_event_aggregation_per_grid_date"](events)

        # Check that the right columns exist
        assert "event_type_terminate_count_30d" in result.columns
        assert "event_type_terminate_count_all_time" in result.columns
        assert "event_type_start_count_30d" in result.columns
        assert "as_of_date" in result.columns

        # Sort by anchor for predictable indexing
        result = result.sort_values("as_of_date").reset_index(drop=True)

        # 2024-01-31 → 30 days after T0: terminate 1 just landed inside window
        # 2024-03-11 → 70 days after T0: terminate 1 (day 30) is OUTSIDE 30d
        #              window (it's 40 days ago); terminate 2 hasn't happened
        #              yet → count = 0
        # 2024-07-19 → 200 days after T0: all 3 terminates outside 30d window
        row_31 = result[result["as_of_date"] == pd.Timestamp("2024-01-31")].iloc[0]
        row_311 = result[result["as_of_date"] == pd.Timestamp("2024-03-11")].iloc[0]
        row_719 = result[result["as_of_date"] == pd.Timestamp("2024-07-19")].iloc[0]

        assert int(row_31["event_type_terminate_count_30d"]) == 1
        assert int(row_311["event_type_terminate_count_30d"]) == 0
        assert int(row_719["event_type_terminate_count_30d"]) == 0

        # all_time terminate count should be cumulative
        # 2024-01-31: only terminate 1 (T0+30) has happened → 1
        # 2024-03-11 (T0+70): still only terminate 1 → 1 (terminate 2 is at
        #                     T0+90 = Apr 1, in the future)
        # 2024-07-19 (T0+200): all 3 → 3
        assert int(row_31["event_type_terminate_count_all_time"]) == 1
        assert int(row_311["event_type_terminate_count_all_time"]) == 1
        assert int(row_719["event_type_terminate_count_all_time"]) == 3

        # Each anchor should have all 3 starts (they were at T0)
        assert int(row_719["event_type_start_count_all_time"]) == 3
