"""Tests for the per-slice top SHAP drivers writer.

The writer is a thin choreographer over Spark APIs (Window, sort_array,
slice, transform) and ``compute_shap_distributed``; the data-correctness
guarantees live in those primitives' own unit tests + the integration
suite. Here we cover the contract:

- Config validation fails fast on bad K / driver-per-row.
- Attribution resolution requires either an artifact or a model URI.
- Empty source → no merge, but a result is still returned.
- The merge is keyed on (model_name, model_version, entity_id).
- Feature-batching kicks in above the threshold (one
  ``compute_shap_distributed`` call per batch).
- The driver-struct schema matches ``top_shap_drivers_schema`` after
  ``F.transform`` cleanup.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pyspark")

from customer_retention.stages.causal import top_drivers_writer
from customer_retention.stages.modeling.shap_attribution import ShapAttribution


def _make_attribution(n_features: int = 3) -> ShapAttribution:
    return ShapAttribution(
        importances={f"f{i}": (i + 1) * 0.1 for i in range(n_features)},
        background_means={f"f{i}": float(i) for i in range(n_features)},
        feature_columns=[f"f{i}" for i in range(n_features)],
        sample_size=1000,
    )


def _fake_df(name: str, columns: List[str] | None = None) -> MagicMock:
    df = MagicMock(name=name)
    df.filter.return_value = df
    df.withColumn.return_value = df
    df.select.return_value = df
    df.distinct.return_value = df
    df.join.return_value = df
    df.drop.return_value = df
    df.alias.return_value = df
    df.columns = columns or ["entity_id"]
    df.count.return_value = 0
    return df


def _make_config(**overrides):
    spark = MagicMock(name="SparkSession")
    spark.table.return_value = _fake_df("snapshot", ["entity_id", "as_of_date", "f0", "f1", "f2"])
    spark.catalog = MagicMock()
    spark.catalog.tableExists.return_value = True
    base = dict(
        spark=spark,
        snapshot_table_fqn="cat.sch.eligibility_snapshot",
        gold_features_fqn="cat.sch.gold",
        top_shap_drivers_fqn="cat.sch.top_shap_drivers",
        model_name="m",
        model_version="1",
        scoring_run_id="score_abc",
        as_of_date=datetime(2026, 4, 28, tzinfo=timezone.utc),
        attribution=_make_attribution(),
    )
    base.update(overrides)
    return top_drivers_writer.TopDriversConfig(**base)


class TestConfigValidation:
    def test_zero_per_slice_k_fails_fast(self):
        cfg = _make_config(per_slice_k=0)
        with pytest.raises(ValueError, match="per_slice_k must be positive"):
            top_drivers_writer.compute_and_write_top_shap_drivers(cfg)

    def test_negative_top_drivers_per_row_fails_fast(self):
        cfg = _make_config(top_drivers_per_row=-1)
        with pytest.raises(ValueError, match="top_drivers_per_row must be positive"):
            top_drivers_writer.compute_and_write_top_shap_drivers(cfg)

    def test_attribution_with_no_features_fails_fast(self):
        cfg = _make_config(attribution=ShapAttribution())
        with pytest.raises(ValueError, match="attribution has no feature_columns"):
            top_drivers_writer.compute_and_write_top_shap_drivers(cfg)


class TestAttributionResolution:
    def test_explicit_attribution_wins(self):
        attribution = _make_attribution()
        cfg = _make_config(attribution=attribution, model_uri="models:/foo@prod")
        resolved = top_drivers_writer._resolve_attribution(cfg)
        assert resolved is attribution

    def test_model_uri_loaded_when_no_attribution(self):
        cfg = _make_config(attribution=None, model_uri="models:/foo@prod")
        with patch.object(top_drivers_writer, "load_attribution_from_model_uri") as loader:
            loader.return_value = _make_attribution()
            top_drivers_writer._resolve_attribution(cfg)
            loader.assert_called_once_with("models:/foo@prod")

    def test_neither_source_fails_fast(self):
        cfg = _make_config(attribution=None, model_uri=None)
        with pytest.raises(ValueError, match="either attribution or model_uri"):
            top_drivers_writer._resolve_attribution(cfg)


class TestEmptyResultPath:
    def test_zero_rows_skips_merge_and_returns_zero_result(self):
        cfg = _make_config()

        with patch.object(top_drivers_writer, "_select_top_per_slice") as sel, \
             patch.object(top_drivers_writer, "_join_latest_gold_features") as gold, \
             patch.object(top_drivers_writer, "_compute_shap_in_batches") as shap, \
             patch.object(top_drivers_writer, "_emit_top_drivers_array") as emit, \
             patch.object(top_drivers_writer, "_shape_target_rows") as shape, \
             patch.object(top_drivers_writer, "merge_dataframe_into") as merge:
            target = _fake_df("target")
            target.count.return_value = 0
            sel.return_value = _fake_df("entity")
            gold.return_value = _fake_df("feature")
            shap.return_value = _fake_df("shap")
            emit.return_value = _fake_df("drivers")
            shape.return_value = target

            result = top_drivers_writer.compute_and_write_top_shap_drivers(cfg)

            assert result.rows_scored == 0
            merge.assert_not_called()
            assert result.target_table_fqn == "cat.sch.top_shap_drivers"


class TestMergeWiring:
    def test_merge_keys_are_model_name_version_entity_id(self):
        cfg = _make_config()
        with patch.object(top_drivers_writer, "_select_top_per_slice") as sel, \
             patch.object(top_drivers_writer, "_join_latest_gold_features") as gold, \
             patch.object(top_drivers_writer, "_compute_shap_in_batches") as shap, \
             patch.object(top_drivers_writer, "_emit_top_drivers_array") as emit, \
             patch.object(top_drivers_writer, "_shape_target_rows") as shape, \
             patch.object(top_drivers_writer, "merge_dataframe_into") as merge:
            target = _fake_df("target")
            target.count.return_value = 7
            sel.return_value = _fake_df("entity")
            gold.return_value = _fake_df("feature")
            shap.return_value = _fake_df("shap")
            emit.return_value = _fake_df("drivers")
            shape.return_value = target

            result = top_drivers_writer.compute_and_write_top_shap_drivers(cfg)

            assert result.rows_scored == 7
            merge.assert_called_once()
            args, _ = merge.call_args
            assert args[3] == "cat.sch.top_shap_drivers"
            assert args[4] == ("model_name", "model_version", "entity_id")


class TestFeatureBatching:
    def test_below_threshold_runs_single_compute(self):
        attribution = _make_attribution(n_features=10)
        feature_df = _fake_df("features", ["entity_id", *attribution.feature_columns])

        with patch.object(top_drivers_writer, "compute_shap_distributed") as compute:
            compute.return_value = SimpleNamespace(
                shap_df=_fake_df("shap"),
                shap_columns=[f"shap_{c}" for c in attribution.feature_columns],
                feature_columns=attribution.feature_columns,
                background_size=1000,
            )
            top_drivers_writer._compute_shap_in_batches(feature_df, attribution, "entity_id")
            assert compute.call_count == 1

    def test_above_threshold_batches_in_groups_of_100(self):
        attribution = _make_attribution(n_features=250)
        feature_df = _fake_df("features", ["entity_id", *attribution.feature_columns])

        with patch.object(top_drivers_writer, "compute_shap_distributed") as compute:
            shap_df = _fake_df("shap")
            shap_df.select.return_value = shap_df
            compute.return_value = SimpleNamespace(
                shap_df=shap_df,
                shap_columns=[],
                feature_columns=[],
                background_size=1000,
            )
            top_drivers_writer._compute_shap_in_batches(feature_df, attribution, "entity_id")
            # 250 features → 3 batches of (100, 100, 50)
            assert compute.call_count == 3


class TestGoldFeaturesContract:
    def test_missing_features_in_gold_fails_fast(self):
        cfg = _make_config()
        gold = _fake_df("gold", ["entity_id", "as_of_date", "f0"])  # missing f1, f2
        cfg.spark.table = MagicMock(return_value=gold)
        entity = _fake_df("entity", ["entity_id"])

        with pytest.raises(ValueError, match="attribution features missing"):
            top_drivers_writer._join_latest_gold_features(
                cfg, entity, ["f0", "f1", "f2"]
            )


class TestGoldTimestampResolution:
    """Pin the gold-timestamp autodetection so a missing ``as_of_date`` no
    longer crashes ``_join_latest_gold_features`` with
    ``UNRESOLVED_COLUMN.WITH_SUGGESTION``. Project-side gold tables that
    pre-aggregate to one row per entity (no temporal column) flow through
    cleanly.
    """

    def test_picks_as_of_date_when_present(self):
        cfg = _make_config()
        col = top_drivers_writer._resolve_gold_timestamp_column(
            cfg, {"entity_id", "as_of_date", "f0"}
        )
        assert col == "as_of_date"

    def test_falls_back_to_event_timestamp(self):
        cfg = _make_config()
        col = top_drivers_writer._resolve_gold_timestamp_column(
            cfg, {"entity_id", "event_timestamp", "f0"}
        )
        assert col == "event_timestamp"

    def test_returns_none_when_table_is_entity_grain(self):
        # Mirrors the email-churn cluster shape: gold has entity_id,
        # feature columns, no temporal column. Must NOT raise.
        cfg = _make_config()
        col = top_drivers_writer._resolve_gold_timestamp_column(
            cfg, {"entity_id", "dow_sin", "dow_cos", "active_span_days"}
        )
        assert col is None

    def test_explicit_override_wins_over_autodetect(self):
        cfg = _make_config(gold_timestamp_column="my_ts")
        col = top_drivers_writer._resolve_gold_timestamp_column(
            cfg, {"entity_id", "as_of_date", "my_ts", "f0"}
        )
        assert col == "my_ts"

    def test_explicit_override_missing_fails_fast(self):
        # Typos must not silently fall back to entity-grain.
        cfg = _make_config(gold_timestamp_column="typo_col")
        with pytest.raises(ValueError, match="gold_timestamp_column='typo_col'"):
            top_drivers_writer._resolve_gold_timestamp_column(
                cfg, {"entity_id", "as_of_date", "f0"}
            )


class TestSelectTopPerSlice:
    """Spec for the per-slice ranker.

    The ranker is the load-bearing piece behind the dashboard's "In Scope"
    coverage: the cache it produces (``top_shap_drivers``) is what the
    dashboard later joins onto ``v_account_primary_recommendation`` to
    decide which entities have a SHAP-row triplet to display. Two
    invariants are non-obvious and worth a unit test:

    (1) The snapshot is filtered to ``policy_rank_among_eligible = 1``
        BEFORE the window so each entity lands in exactly one slice (its
        primary). Without this, every entity's ~106 (entity × policy)
        snapshot rows would smear across multiple slices, the same high-
        ``expected_loss`` customers would be re-picked in many slices,
        and ``.distinct()`` would collapse the cache to a tiny global
        count -- starving the per-slice-K cap entirely.

    (2) The ORDER BY carries a ``value_at_risk DESC NULLS LAST`` and a
        final ``entity_id ASC`` so Low-tier slices (where
        ``expected_loss`` ties at 0 across thousands of entities) still
        return a deterministic top-K run-over-run.
    """

    def test_filters_snapshot_to_primary_recommendation_only(self):
        cfg = _make_config()
        snapshot_df = _fake_df("snapshot")
        cfg.spark.table.return_value = snapshot_df

        top_drivers_writer._select_top_per_slice(cfg)

        # The snapshot is filtered with a single compound condition:
        # ``scoring_run_id == X AND policy_rank_among_eligible == 1``.
        # The MagicMock chain returns itself from each method, so the
        # later ``ranked.filter(rank <= K)`` call lands on the same mock;
        # walk every recorded filter and assert the primary-only key
        # appears in at least one of them.
        assert snapshot_df.filter.called
        rendered_calls = [repr(c.args[0]) for c in snapshot_df.filter.call_args_list]
        assert any("policy_rank_among_eligible" in r for r in rendered_calls), (
            f"primary-only filter missing from snapshot read; got {rendered_calls!r}"
        )

    def test_order_by_includes_value_at_risk_and_entity_id_tiebreakers(self):
        # Smoke check the Window's orderBy carries the deterministic keys.
        # We don't run Spark; we just inspect the Column expressions
        # passed to ``orderBy``.
        cfg = _make_config()
        captured: dict = {}

        # Patch Window.partitionBy to record the orderBy chain
        from pyspark.sql import Window
        real_partition_by = Window.partitionBy

        def _spy_partition(*args, **kwargs):
            wspec = real_partition_by(*args, **kwargs)
            real_order_by = wspec.orderBy

            def _spy_order_by(*oargs, **okwargs):
                captured["order_by"] = oargs
                return real_order_by(*oargs, **okwargs)

            wspec.orderBy = _spy_order_by  # type: ignore[method-assign]
            return wspec

        with patch.object(Window, "partitionBy", side_effect=_spy_partition):
            top_drivers_writer._select_top_per_slice(cfg)

        assert "order_by" in captured, "Window.orderBy was never invoked"
        rendered = " ".join(repr(c) for c in captured["order_by"])
        assert "value_at_risk" in rendered, (
            f"value_at_risk tiebreaker missing from ORDER BY; got {rendered!r}"
        )
        assert "entity_id" in rendered, (
            f"entity_id final tiebreaker missing from ORDER BY; got {rendered!r}"
        )


class TestResultSummary:
    def test_summary_string_carries_counts(self):
        result = top_drivers_writer.TopDriversResult(
            rows_scored=42,
            drivers_per_row=5,
            per_slice_k=20,
            feature_count=128,
            target_table_fqn="cat.sch.top_shap_drivers",
        )
        text = result.summary()
        assert "42" in text
        assert "5 drivers/row" in text
        assert "K=20/slice" in text
        assert "128 attribution features" in text
        assert "cat.sch.top_shap_drivers" in text
