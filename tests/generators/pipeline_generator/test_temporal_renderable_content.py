"""TemporalFeatureConfig.has_renderable_content + template guard + Spark early raise (Cycle 002).

Layered defenses against the `AssertionError: exprs should not be empty`
Spark crash in `bronze_event_customer_emails`:

- `has_renderable_content()` — pure Python predicate on the config shape;
  the generator emits `compute_temporal_features` only when it returns
  True. Keeps the generated notebook free of dead code.
- Jinja guard — both `databricks_renderer.py` and `renderer.py` wrap the
  `compute_temporal_features` function AND its call site in
  `{% if config.temporal_features and config.temporal_features.has_renderable_content() %}`.
- `_lagged_windows_spark` — framework-level `ValueError` fires before
  Catalyst's `AssertionError` so the operator sees a diagnostic pointing
  at the config, not at a pyspark internal.
"""

from __future__ import annotations

import pytest


class TestHasRenderableContent:
    def test_returns_false_for_empty_config(self):
        from customer_retention.generators.pipeline_generator.models import TemporalFeatureConfig
        cfg = TemporalFeatureConfig(lag_columns=[], feature_groups=[], time_column_known=False)
        assert cfg.has_renderable_content() is False

    def test_returns_true_when_lag_columns_plus_lag_family(self):
        from customer_retention.generators.pipeline_generator.models import TemporalFeatureConfig
        cfg = TemporalFeatureConfig(
            lag_columns=["send_hour"],
            feature_groups=["lagged_windows"],
        )
        assert cfg.has_renderable_content() is True

    def test_returns_true_when_recency_and_time_known_even_without_lag_columns(self):
        from customer_retention.generators.pipeline_generator.models import TemporalFeatureConfig
        cfg = TemporalFeatureConfig(
            lag_columns=[],
            feature_groups=["recency", "regularity"],
            time_column_known=True,
        )
        assert cfg.has_renderable_content() is True

    def test_returns_false_when_only_lag_family_but_no_lag_columns(self):
        from customer_retention.generators.pipeline_generator.models import TemporalFeatureConfig
        cfg = TemporalFeatureConfig(
            lag_columns=[],
            feature_groups=["lagged_windows", "velocity"],
            time_column_known=True,
        )
        assert cfg.has_renderable_content() is False

    def test_returns_false_when_recency_requested_but_no_time_column(self):
        from customer_retention.generators.pipeline_generator.models import TemporalFeatureConfig
        cfg = TemporalFeatureConfig(
            lag_columns=[],
            feature_groups=["recency"],
            time_column_known=False,
        )
        assert cfg.has_renderable_content() is False


def _make_event_cfg(lag_columns, feature_groups):
    from customer_retention.generators.pipeline_generator.models import (
        AggregationWindowConfig,
        BronzeEventConfig,
        SourceConfig,
        TemporalFeatureConfig,
    )
    source = SourceConfig(name="customer_emails", path="/tmp/x.csv", format="csv",
                          entity_key="customer_id", time_column="sent_date")
    return BronzeEventConfig(
        source=source, entity_column="customer_id", time_column="sent_date",
        aggregation=AggregationWindowConfig(
            windows=["180d"],
            value_columns=list(lag_columns),
        ),
        temporal_features=TemporalFeatureConfig(
            lag_columns=list(lag_columns), feature_groups=list(feature_groups),
        ),
    )


class TestRendererGuard:
    def test_databricks_renderer_drops_compute_temporal_features_when_unrenderable(self):
        from customer_retention.generators.pipeline_generator.databricks_renderer import (
            DatabricksCodeRenderer,
        )
        event = _make_event_cfg(lag_columns=[], feature_groups=["lagged_windows"])
        rendered = DatabricksCodeRenderer(catalog="c", schema="s").render_bronze_event(
            "customer_emails", event,
        )
        assert "def compute_temporal_features" not in rendered
        assert "agg_df = compute_temporal_features" not in rendered

    def test_databricks_renderer_keeps_compute_when_renderable(self):
        from customer_retention.generators.pipeline_generator.databricks_renderer import (
            DatabricksCodeRenderer,
        )
        event = _make_event_cfg(
            lag_columns=["send_hour"],
            feature_groups=["lagged_windows", "recency"],
        )
        rendered = DatabricksCodeRenderer(catalog="c", schema="s").render_bronze_event(
            "customer_emails", event,
        )
        assert "def compute_temporal_features" in rendered
        assert "agg_df = compute_temporal_features" in rendered
        assert "compute_recency=True" in rendered
        assert "compute_velocity=False" in rendered
        assert "compute_regularity=False" in rendered


class TestSparkEarlyRaise:
    def test_lagged_windows_spark_raises_valueerror_on_empty_value_cols(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.profiling import spark_temporal_feature_engineer as mod

        class _Cfg:
            num_lags = 4
            lag_window_days = 30
            lag_aggregations = ["sum", "mean", "count", "max"]

        with pytest.raises(ValueError, match="value_cols is empty"):
            mod._lagged_windows_spark(
                spark_df=None, entity_col="entity_id", time_col="as_of_date",
                value_cols=[], ref_spark=None, config=_Cfg(),
            )
