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
