"""Cycle 004 — Event-type windowed counts invariant.

For every event_level dataset whose aggregation config has
`per_grid_date_mode=True` and a non-empty `value_counts_columns`, the
generated bronze_event_<source>.py script (local + Databricks) must
produce — at runtime — one column per (vc_col, distinct_value, window)
triple named `f"{vc_col}_{sanitize_column_token(value)}_count_{window}"`.

The oracle in this module encodes that invariant in ten lines of pure
Python. Every test rendered either:

* greps the rendered source for the template signatures that emit the
  columns (catches quoting / loop-structure template regressions), OR
* executes the rendered per-grid-date function and asserts the oracle's
  column-name set is a subset of the runtime column set.

The regression test `test_seven_window_two_value_shape_emits_14_columns`
reproduces the 7-window x 2-value shape documented in the diagnosis
(generic fixture; no engagement-specific literals) using synthetic
event data. Pre-fix the baseline run produced zero
`event_type_*_count_*` columns; this test is the offline expression of
the cycle's runtime gate.
"""
from __future__ import annotations

import ast
import sys
import types
from typing import Any

import pandas as pd

from customer_retention.core.naming import sanitize_column_token
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


def expected_value_counts_columns(
    vc_col: str, distinct_values: list[object], windows: list[str]
) -> set[str]:
    """Oracle — computes expected output column names from first principles."""
    return {
        f"{vc_col}_{sanitize_column_token(value)}_count_{window}"
        for value in distinct_values
        for window in windows
    }


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
        aggregation=AggregationWindowConfig(windows=windows or ["30d", "all_time"]),
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
        gold=None,
        output_dir="generated/test",
    )


def _exec_rendered(rendered: str, tmp_path) -> dict[str, Any]:
    stub = types.ModuleType("config")
    stub.PRODUCTION_DIR = tmp_path
    stub.TARGET_COLUMN = "target"
    stub.FIT_MODE = True
    sys.modules["config"] = stub
    namespace: dict[str, Any] = {"__name__": "rendered_module"}
    try:
        exec(compile(rendered, "<rendered>", "exec"), namespace)
    finally:
        del sys.modules["config"]
    return namespace


class TestOracle:
    def test_oracle_cross_product_shape(self):
        result = expected_value_counts_columns(
            "event_type", ["start", "terminate"], ["7d", "30d", "all_time"]
        )
        assert result == {
            "event_type_start_count_7d",
            "event_type_start_count_30d",
            "event_type_start_count_all_time",
            "event_type_terminate_count_7d",
            "event_type_terminate_count_30d",
            "event_type_terminate_count_all_time",
        }

    def test_oracle_sanitizes_unsafe_values(self):
        result = expected_value_counts_columns("label", ["a b", "c/d"], ["30d"])
        assert result == {"label_a_b_count_30d", "label_c_d_count_30d"}

    def test_oracle_empty_values_yields_empty(self):
        assert expected_value_counts_columns("vc", [], ["30d"]) == set()

    def test_oracle_empty_windows_yields_empty(self):
        assert expected_value_counts_columns("vc", ["a"], []) == set()


class TestLocalRendererValueCountsInvariant:
    def test_renders_value_counts_literal_with_windows(self):
        rendered = CodeRenderer().render_bronze_event(
            "contract",
            _make_event_config(
                per_grid_date_mode=True,
                value_counts_columns=("event_type",),
                windows=["7d", "30d", "90d", "180d", "365d", "all_time"],
            ),
            _make_pipeline_config(
                _make_event_config(
                    per_grid_date_mode=True, value_counts_columns=("event_type",)
                )
            ),
        )
        ast.parse(rendered)
        assert "VALUE_COUNTS_COLUMNS = ['event_type']" in rendered
        assert "AGGREGATION_WINDOWS = ['7d', '30d', '90d', '180d', '365d', 'all_time']" in rendered

    def test_rendered_source_contains_column_formation_signature(self):
        """Catches template regressions that drop the loop or alter naming."""
        rendered = CodeRenderer().render_bronze_event(
            "contract",
            _make_event_config(
                per_grid_date_mode=True, value_counts_columns=("event_type",)
            ),
            _make_pipeline_config(
                _make_event_config(
                    per_grid_date_mode=True, value_counts_columns=("event_type",)
                )
            ),
        )
        assert 'f"{vc_col}_{sanitize_column_token(value)}_count_{window_str}"' in rendered
        assert "for vc_col in VALUE_COUNTS_COLUMNS" in rendered
        assert "for window_str in AGGREGATION_WINDOWS" in rendered

    def test_runtime_emits_full_cross_product_2_values_7_windows(self, tmp_path):
        windows = ["7d", "30d", "90d", "180d", "365d", "all_time"]
        cfg = _make_event_config(
            per_grid_date_mode=True,
            value_counts_columns=("event_type",),
            windows=windows,
        )
        config = _make_pipeline_config(cfg)
        config.silver.grid_dates = ["2024-03-01", "2024-06-01"]
        rendered = CodeRenderer().render_bronze_event("contract", cfg, config)
        ns = _exec_rendered(rendered, tmp_path)

        T0 = pd.Timestamp("2024-01-01")
        events = pd.DataFrame(
            {
                "ACCOUNT_ID": ["A", "A", "B", "B"],
                "event_timestamp": [T0, T0 + pd.Timedelta(days=30),
                                    T0, T0 + pd.Timedelta(days=60)],
                "event_type": ["start", "terminate", "start", "terminate"],
            }
        )
        result = ns["apply_event_aggregation_per_grid_date"](events)

        expected = expected_value_counts_columns(
            "event_type", ["start", "terminate"], windows
        )
        missing = expected - set(result.columns)
        assert not missing, f"rendered runtime missing columns: {sorted(missing)}"

    def test_runtime_sanitizes_unsafe_value_labels(self, tmp_path):
        cfg = _make_event_config(
            per_grid_date_mode=True,
            value_counts_columns=("kind",),
            windows=["30d", "all_time"],
        )
        config = _make_pipeline_config(cfg)
        config.silver.grid_dates = ["2024-02-01"]
        rendered = CodeRenderer().render_bronze_event("contract", cfg, config)
        ns = _exec_rendered(rendered, tmp_path)

        T0 = pd.Timestamp("2024-01-01")
        events = pd.DataFrame(
            {
                "ACCOUNT_ID": ["A", "A"],
                "event_timestamp": [T0, T0 + pd.Timedelta(days=5)],
                "kind": ["a b", "c/d"],
            }
        )
        result = ns["apply_event_aggregation_per_grid_date"](events)
        expected = expected_value_counts_columns("kind", ["a b", "c/d"], ["30d", "all_time"])
        missing = expected - set(result.columns)
        assert not missing, f"sanitized columns missing: {sorted(missing)}"


class TestDatabricksRendererValueCountsInvariant:
    def test_renders_value_counts_literal_with_windows(self):
        rendered = DatabricksCodeRenderer(catalog="main", schema="default").render_bronze_event(
            "contract",
            _make_event_config(
                per_grid_date_mode=True,
                value_counts_columns=("event_type",),
                windows=["7d", "30d", "90d", "180d", "365d", "all_time"],
            ),
            _make_pipeline_config(
                _make_event_config(
                    per_grid_date_mode=True, value_counts_columns=("event_type",)
                )
            ),
        )
        ast.parse(rendered)
        assert "VALUE_COUNTS_COLUMNS = ['event_type']" in rendered
        assert "AGGREGATION_WINDOWS = ['7d', '30d', '90d', '180d', '365d', 'all_time']" in rendered

    def test_rendered_source_contains_column_formation_signature(self):
        rendered = DatabricksCodeRenderer(catalog="main", schema="default").render_bronze_event(
            "contract",
            _make_event_config(
                per_grid_date_mode=True, value_counts_columns=("event_type",)
            ),
            _make_pipeline_config(
                _make_event_config(
                    per_grid_date_mode=True, value_counts_columns=("event_type",)
                )
            ),
        )
        assert 'f"{vc_col}_{sanitize_column_token(value)}_count_{window_str}"' in rendered
        assert "for vc_col in VALUE_COUNTS_COLUMNS" in rendered
        assert "for window_str in AGGREGATION_WINDOWS" in rendered


class TestRendererParityAcrossEnvironments:
    def test_identical_windows_and_value_counts_on_both_renderers(self):
        windows = ["7d", "30d", "90d", "180d", "365d", "all_time"]
        cfg = _make_event_config(
            per_grid_date_mode=True,
            value_counts_columns=("event_type",),
            windows=windows,
        )
        config = _make_pipeline_config(cfg)
        local = CodeRenderer().render_bronze_event("contract", cfg, config)
        databricks = DatabricksCodeRenderer(catalog="main", schema="default").render_bronze_event(
            "contract", cfg, config
        )
        windows_literal = (
            "AGGREGATION_WINDOWS = ['7d', '30d', '90d', '180d', '365d', 'all_time']"
        )
        for rendered in (local, databricks):
            assert windows_literal in rendered
            assert "VALUE_COUNTS_COLUMNS = ['event_type']" in rendered


class TestSevenWindowTwoValueRegressionShape:
    """Regresses the 2-windows / 0 event_type_*_count_* shape observed in
    the diagnosis on run-under-analysis. Reproduced with synthetic data
    using the same structural properties (7 windows declared, 2 distinct
    event_type values, per_grid_date_mode=True). Post-fix the rendered
    runtime MUST produce all 14 expected columns; pre-C2/C3 it did not."""

    def test_seven_window_two_value_shape_emits_14_columns(self, tmp_path):
        windows = ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"]
        cfg = _make_event_config(
            per_grid_date_mode=True,
            value_counts_columns=("event_type",),
            windows=windows,
        )
        config = _make_pipeline_config(cfg)
        config.silver.grid_dates = ["2024-04-01", "2024-05-01", "2024-06-01"]
        rendered = CodeRenderer().render_bronze_event("contract", cfg, config)
        ns = _exec_rendered(rendered, tmp_path)

        T0 = pd.Timestamp("2024-01-01")
        events = pd.DataFrame(
            {
                "ACCOUNT_ID": ["A"] * 6 + ["B"] * 4,
                "event_timestamp": [
                    T0, T0 + pd.Timedelta(days=5),
                    T0 + pd.Timedelta(days=30), T0 + pd.Timedelta(days=60),
                    T0 + pd.Timedelta(days=90), T0 + pd.Timedelta(days=120),
                    T0 + pd.Timedelta(days=10), T0 + pd.Timedelta(days=20),
                    T0 + pd.Timedelta(days=100), T0 + pd.Timedelta(days=150),
                ],
                "event_type": [
                    "start", "terminate", "start", "terminate",
                    "start", "terminate", "start", "terminate",
                    "start", "terminate",
                ],
            }
        )
        result = ns["apply_event_aggregation_per_grid_date"](events)

        expected = expected_value_counts_columns(
            "event_type", ["start", "terminate"], windows
        )
        assert len(expected) == 14
        missing = expected - set(result.columns)
        assert not missing, (
            f"7-window x 2-value shape regressed — missing "
            f"{len(missing)}/14 expected columns: {sorted(missing)}"
        )
