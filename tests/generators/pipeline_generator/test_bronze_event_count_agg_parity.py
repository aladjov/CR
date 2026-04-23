"""FIX SPS-1 — per-column count aggregation parity across exploration and production.

NB01d's `time_window_aggregator` emits `{value_col}_count_{window}` at
exploration time for every numeric value column. NB04/NB06 then build
recommendations against those names, and NB08 records them in
`feature_spec.yaml`. The generated bronze_event scripts (local pandas +
Databricks pyspark) and `_event_aggregated_columns` must emit/predict
the same column names, or the parity gate in `_collect_allowlist_drops`
rejects the whole FeatureSpec.

Pre-FIX-SPS-1: both renderers and the predictor hardcoded a `count`
skip for value columns (only row-level `event_count_{window}` was
emitted). Post-fix: `count` is a first-class agg func on value columns,
emitting `{col}_count_{window}` alongside `event_count_{window}`.
"""
from __future__ import annotations

import ast

from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.findings_parser import (
    FindingsParser,
)
from customer_retention.generators.pipeline_generator.models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    SourceConfig,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer


def _src(name: str = "events") -> SourceConfig:
    return SourceConfig(
        name=name, path=f"/data/{name}.csv", format="csv",
        entity_key="customer_id", time_column="event_date",
        is_event_level=True, raw_source_path=f"/data/{name}.csv",
    )


def _bec(agg: AggregationWindowConfig) -> BronzeEventConfig:
    return BronzeEventConfig(
        source=_src(), entity_column="customer_id", time_column="event_date",
        aggregation=agg,
    )


class TestLocalRendererEmitsCountCol:
    def test_count_included_in_agg_funcs_constant(self):
        """Local template bakes `AGG_FUNCS` as a Python constant at the
        top of the generated file. Pre-fix it filtered out 'count'; the
        runtime `for func in AGG_FUNCS` loop therefore never emitted
        `{col}_count_{window}`. Lock in that count is preserved."""
        agg = AggregationWindowConfig(
            windows=["30d", "all_time"], value_columns=["amount"],
            agg_funcs=["sum", "mean", "count"],
        )
        rendered = CodeRenderer().render_bronze_event("events", _bec(agg))
        # The baked constant should contain 'count' (order-insensitive).
        agg_funcs_line = next(
            ln for ln in rendered.splitlines() if ln.strip().startswith("AGG_FUNCS")
        )
        assert "'count'" in agg_funcs_line or '"count"' in agg_funcs_line
        ast.parse(rendered)

    def test_count_omitted_when_not_in_agg_funcs(self):
        """Without count in agg_funcs, the constant must not add it
        back — we only preserve user intent, we don't force it."""
        agg = AggregationWindowConfig(
            windows=["30d"], value_columns=["amount"],
            agg_funcs=["sum", "mean"],
        )
        rendered = CodeRenderer().render_bronze_event("events", _bec(agg))
        agg_funcs_line = next(
            ln for ln in rendered.splitlines() if ln.strip().startswith("AGG_FUNCS")
        )
        assert "'count'" not in agg_funcs_line and '"count"' not in agg_funcs_line


class TestDatabricksRendererEmitsCountExpr:
    def test_count_emits_pyspark_aggregation(self):
        agg = AggregationWindowConfig(
            windows=["30d", "all_time"], value_columns=["amount"],
            agg_funcs=["sum", "count"],
        )
        rendered = DatabricksCodeRenderer().render_bronze_event("events", _bec(agg))
        # Per-column count expression — new under FIX SPS-1.
        assert 'F.count(col).alias(f"{col}_count_30d")' in rendered
        assert 'F.count(col).alias(f"{col}_count_all_time")' in rendered
        # Row-level event_count — preserved, separate from per-column count.
        assert 'F.count("*").alias("event_count_30d")' in rendered
        assert 'F.count("*").alias("event_count_all_time")' in rendered
        ast.parse(rendered)

    def test_count_not_emitted_when_not_in_agg_funcs(self):
        agg = AggregationWindowConfig(
            windows=["30d"], value_columns=["amount"],
            agg_funcs=["sum", "mean"],
        )
        rendered = DatabricksCodeRenderer().render_bronze_event("events", _bec(agg))
        assert 'F.count(col).alias(f"{col}_count_30d")' not in rendered
        # But row-level event_count is unconditional.
        assert 'F.count("*").alias("event_count_30d")' in rendered


class TestPredictorParityWithRenderers:
    """`_event_aggregated_columns` must predict the EXACT set of
    columns the renderers emit, otherwise silver/gold filter passes
    shed recs the generator will then try to execute at runtime."""

    def test_predictor_matches_databricks_renderer_count_emission(self):
        agg = AggregationWindowConfig(
            windows=["30d", "all_time"], value_columns=["amount", "latency"],
            agg_funcs=["sum", "count"],
        )
        cols = FindingsParser._event_aggregated_columns(_bec(agg))
        # Per-column count predicted for both value columns.
        assert {
            "amount_count_30d", "amount_count_all_time",
            "latency_count_30d", "latency_count_all_time",
        } <= cols
        # Other aggs predicted too.
        assert {"amount_sum_30d", "latency_sum_all_time"} <= cols
        # Row-level event_count_{window} also predicted.
        assert {"event_count_30d", "event_count_all_time"} <= cols

    def test_predictor_respects_column_blocked_count(self):
        """Per-column blocking still overrides the default inclusion."""
        agg = AggregationWindowConfig(
            windows=["30d"], value_columns=["amount", "latency"],
            agg_funcs=["sum", "count"],
            column_blocked_funcs={"latency": ["count"]},
        )
        cols = FindingsParser._event_aggregated_columns(_bec(agg))
        assert "amount_count_30d" in cols
        assert "latency_count_30d" not in cols


class TestCountParityIsMirroredAcrossAllThreeSurfaces:
    """Regression oracle — a single scenario, three parallel assertions.
    If any surface (local renderer, Databricks renderer, predictor)
    ever re-introduces the count skip, this test fails."""

    def test_three_way_parity(self):
        agg = AggregationWindowConfig(
            windows=["90d"], value_columns=["amount"],
            agg_funcs=["sum", "mean", "count"],
        )
        bec = _bec(agg)
        local = CodeRenderer().render_bronze_event("events", bec)
        spark = DatabricksCodeRenderer().render_bronze_event("events", bec)
        predicted = FindingsParser._event_aggregated_columns(bec)

        # Local: baked constant retains 'count'.
        assert any(
            ("'count'" in ln or '"count"' in ln)
            for ln in local.splitlines()
            if ln.strip().startswith("AGG_FUNCS")
        )
        # Spark: emits F.count(col) with the expected alias.
        assert 'F.count(col).alias(f"{col}_count_90d")' in spark
        # Predictor: knows about it.
        assert "amount_count_90d" in predicted
