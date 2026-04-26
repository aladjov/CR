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


class TestPerValueCountExpansion:
    """`_event_aggregated_columns` must expand the per-value count shape
    `{col}_{val}_count_{window}` whenever (a) `value_counts` is in
    `categorical_agg_funcs`, (b) `col` is in `event_cfg.value_counts_columns`,
    and (c) `unique_values_by_col` provides the observed values. This
    matches what `time_window_aggregator._compute_value_counts` actually
    emits at runtime when `per_grid_date_mode=True`. Without this, the
    parity gate sees only the literal `event_type_value_counts_365d` shape
    and fails for every `<col>_<val>_count_<w>` feature in `feature_spec`."""

    def _bec_with_value_counts(self, value_counts_columns):
        agg = AggregationWindowConfig(
            windows=["365d"], value_columns=[],
            agg_funcs=[], categorical_columns=["event_type"],
            categorical_agg_funcs=["value_counts"],
        )
        return BronzeEventConfig(
            source=_src(), entity_column="customer_id", time_column="event_date",
            aggregation=agg, per_grid_date_mode=True,
            value_counts_columns=tuple(value_counts_columns),
        )

    def test_per_value_expansion_when_unique_values_provided(self):
        bec = self._bec_with_value_counts(["event_type"])
        cols = FindingsParser._event_aggregated_columns(
            bec, unique_values_by_col={"event_type": ["start", "terminate"]},
        )
        assert "event_type_start_count_365d" in cols
        assert "event_type_terminate_count_365d" in cols
        # The literal `value_counts` shape is suppressed when expansion fires.
        assert "event_type_value_counts_365d" not in cols

    def test_no_expansion_without_unique_values_fallback_to_literal(self):
        """Default behaviour (no `unique_values_by_col` arg) preserves the
        legacy literal `value_counts` column name so existing parity-mode
        tests stay green."""
        bec = self._bec_with_value_counts(["event_type"])
        cols = FindingsParser._event_aggregated_columns(bec)
        assert "event_type_value_counts_365d" in cols
        assert "event_type_start_count_365d" not in cols

    def test_no_expansion_when_col_not_in_value_counts_columns(self):
        bec = self._bec_with_value_counts([])  # value_counts_columns empty
        cols = FindingsParser._event_aggregated_columns(
            bec, unique_values_by_col={"event_type": ["start", "terminate"]},
        )
        assert "event_type_value_counts_365d" in cols
        assert "event_type_start_count_365d" not in cols

    def test_unaffected_when_func_is_not_value_counts(self):
        agg = AggregationWindowConfig(
            windows=["365d"], value_columns=[],
            agg_funcs=[], categorical_columns=["event_type"],
            categorical_agg_funcs=["mode", "nunique"],
        )
        bec = BronzeEventConfig(
            source=_src(), entity_column="customer_id", time_column="event_date",
            aggregation=agg, per_grid_date_mode=True,
            value_counts_columns=("event_type",),
        )
        cols = FindingsParser._event_aggregated_columns(
            bec, unique_values_by_col={"event_type": ["start", "terminate"]},
        )
        assert "event_type_mode_365d" in cols
        assert "event_type_nunique_365d" in cols
        assert "event_type_start_count_365d" not in cols


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
