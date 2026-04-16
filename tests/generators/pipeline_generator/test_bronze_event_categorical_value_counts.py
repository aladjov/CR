"""Tests for categorical_value_counts aggregation in bronze event templates.

Emits per-value count columns like `event_type_start_count_365d` that the
SPS `derive_contract_ratio_features` cell reads. Works in snapshot (default)
aggregation mode — complements the existing `per_grid_date_mode` pattern.
"""
from __future__ import annotations

import ast
from typing import Any

import pandas as pd
import pytest

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


def _make_event_config(
    *,
    categorical_value_counts: dict | None = None,
    windows: list[str] | None = None,
) -> BronzeEventConfig:
    return BronzeEventConfig(
        source=SourceConfig(
            name="contract", path="data/contract", format="delta",
            entity_key="ACCOUNT_ID", time_column="event_timestamp",
            is_event_level=True,
        ),
        entity_column="ACCOUNT_ID",
        time_column="event_timestamp",
        deduplicate=False,
        aggregation=AggregationWindowConfig(
            windows=windows or ["30d", "90d", "all_time"],
            categorical_value_counts=categorical_value_counts or {},
        ),
    )


def _make_pipeline_config(event_config: BronzeEventConfig) -> PipelineConfig:
    return PipelineConfig(
        name="contract_test", target_column="churn",
        sources=[event_config.source], bronze={},
        silver=SilverLayerConfig(
            entity_key="ACCOUNT_ID",
            grid_dates=["2024-01-01", "2024-02-01"],
        ),
        gold=None,  # type: ignore[arg-type]
        output_dir="generated/test",
    )


def _exec_rendered(rendered: str, env_globals: dict[str, Any]) -> dict[str, Any]:
    """Exec a rendered bronze_event module, stubbing the `config` import."""
    import sys
    import types

    config_stub = types.ModuleType("config")
    config_stub.TARGET_COLUMN = "churn"  # type: ignore
    config_stub.SOURCES = {}  # type: ignore
    config_stub.get_bronze_path = lambda name: f"/tmp/{name}"  # type: ignore
    config_stub.get_silver_path = lambda name: f"/tmp/{name}"  # type: ignore
    config_stub.EXPERIMENTS_DIR = "/tmp/experiments"  # type: ignore
    config_stub.FINDINGS_DIR = "/tmp/findings"  # type: ignore
    config_stub.PRODUCTION_DIR = "/tmp/production"  # type: ignore
    config_stub.get_landing_path = lambda name: f"/tmp/landing/{name}"  # type: ignore
    config_stub.ENTITY_COLUMN = "ACCOUNT_ID"  # type: ignore
    config_stub.TIME_COLUMN = "event_timestamp"  # type: ignore

    saved = sys.modules.get("config")
    sys.modules["config"] = config_stub
    try:
        namespace: dict[str, Any] = dict(env_globals)
        exec(compile(rendered, "<rendered>", "exec"), namespace)
        return namespace
    finally:
        if saved is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = saved


class TestLocalRenderer:
    def test_default_empty_no_value_counts_code(self):
        renderer = CodeRenderer()
        cfg = _make_event_config()
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        assert "CATEGORICAL_VALUE_COUNTS" not in rendered or "CATEGORICAL_VALUE_COUNTS = {}" in rendered
        ast.parse(rendered)

    def test_declared_values_produce_constant(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(
            categorical_value_counts={"event_type": ["start", "terminate"]},
        )
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        assert "CATEGORICAL_VALUE_COUNTS" in rendered
        assert "'event_type'" in rendered
        assert "'start'" in rendered
        assert "'terminate'" in rendered
        ast.parse(rendered)

    def test_emits_count_columns_per_window(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(
            categorical_value_counts={"event_type": ["start", "terminate"]},
            windows=["30d", "all_time"],
        )
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        # The template should reference the column-naming pattern:
        # f"{col}_{value}_count_{window}"
        assert "_count_" in rendered
        # Template should loop over CATEGORICAL_VALUE_COUNTS items
        assert "CATEGORICAL_VALUE_COUNTS" in rendered

    def test_functional_end_to_end(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(
            categorical_value_counts={"event_type": ["start", "terminate"]},
            windows=["30d", "all_time"],
        )
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        ns = _exec_rendered(rendered, {})

        # Build a synthetic event stream
        df = pd.DataFrame({
            "ACCOUNT_ID": ["A", "A", "A", "B", "B", "C"],
            "event_timestamp": pd.to_datetime([
                "2024-02-15", "2024-02-28", "2024-03-05",  # A: 2 starts, 1 terminate
                "2024-02-10", "2024-03-01",                # B: 1 start, 1 terminate
                "2024-03-02",                              # C: 1 start
            ]),
            "event_type": ["start", "start", "terminate", "start", "terminate", "start"],
        })

        result = ns["apply_event_aggregation"](df.copy())

        # Column presence (both windows × both values)
        for w in ("30d", "all_time"):
            for v in ("start", "terminate"):
                col = f"event_type_{v}_count_{w}"
                assert col in result.columns, f"expected {col}"

        # Semantic checks on all_time counts (most permissive)
        result_idx = result.set_index("ACCOUNT_ID")
        assert result_idx.loc["A", "event_type_start_count_all_time"] == 2
        assert result_idx.loc["A", "event_type_terminate_count_all_time"] == 1
        assert result_idx.loc["B", "event_type_start_count_all_time"] == 1
        assert result_idx.loc["B", "event_type_terminate_count_all_time"] == 1
        assert result_idx.loc["C", "event_type_start_count_all_time"] == 1
        assert result_idx.loc["C", "event_type_terminate_count_all_time"] == 0

    def test_fail_fast_when_declared_column_missing(self):
        renderer = CodeRenderer()
        cfg = _make_event_config(
            categorical_value_counts={"event_type": ["start", "terminate"]},
            windows=["all_time"],
        )
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        ns = _exec_rendered(rendered, {})

        # DataFrame is MISSING 'event_type' column — should raise
        df = pd.DataFrame({
            "ACCOUNT_ID": ["A", "B"],
            "event_timestamp": pd.to_datetime(["2024-02-15", "2024-03-01"]),
        })
        with pytest.raises(ValueError, match=r"event_type"):
            ns["apply_event_aggregation"](df.copy())

    def test_unseen_value_produces_zero_column(self):
        """Declared value 'cancelled' is never in data; column still emitted as 0."""
        renderer = CodeRenderer()
        cfg = _make_event_config(
            categorical_value_counts={"event_type": ["start", "cancelled"]},
            windows=["all_time"],
        )
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        ns = _exec_rendered(rendered, {})

        df = pd.DataFrame({
            "ACCOUNT_ID": ["A"],
            "event_timestamp": pd.to_datetime(["2024-02-15"]),
            "event_type": ["start"],
        })
        result = ns["apply_event_aggregation"](df.copy())
        assert "event_type_cancelled_count_all_time" in result.columns
        assert result["event_type_cancelled_count_all_time"].iloc[0] == 0


class TestDatabricksRenderer:
    def test_default_empty_no_value_counts_code(self):
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config()
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        ast.parse(rendered)

    def test_declared_values_produce_constant(self):
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(
            categorical_value_counts={"event_type": ["start", "terminate"]},
        )
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        assert "CATEGORICAL_VALUE_COUNTS" in rendered
        assert "'event_type'" in rendered
        assert "'start'" in rendered
        assert "'terminate'" in rendered
        ast.parse(rendered)

    def test_emits_count_aggregation_per_value(self):
        renderer = DatabricksCodeRenderer(catalog="main", schema="default")
        cfg = _make_event_config(
            categorical_value_counts={"event_type": ["start", "terminate"]},
        )
        rendered = renderer.render_bronze_event("contract", cfg, _make_pipeline_config(cfg))
        # Spark uses F.sum(F.when(col==value, 1).otherwise(0))
        assert "F.when" in rendered or "when(" in rendered
        assert "CATEGORICAL_VALUE_COUNTS" in rendered


class TestParity:
    def test_both_renderers_declare_same_constant(self):
        cfg_local = _make_event_config(
            categorical_value_counts={"event_type": ["start", "terminate"]},
            windows=["30d", "all_time"],
        )
        cfg_dbx = _make_event_config(
            categorical_value_counts={"event_type": ["start", "terminate"]},
            windows=["30d", "all_time"],
        )
        local = CodeRenderer().render_bronze_event("contract", cfg_local, _make_pipeline_config(cfg_local))
        dbx = DatabricksCodeRenderer(catalog="main", schema="default").render_bronze_event(
            "contract", cfg_dbx, _make_pipeline_config(cfg_dbx),
        )
        # Both templates must declare the same CATEGORICAL_VALUE_COUNTS literal
        assert "'event_type'" in local and "'event_type'" in dbx
        assert "'start'" in local and "'start'" in dbx
        assert "'terminate'" in local and "'terminate'" in dbx
