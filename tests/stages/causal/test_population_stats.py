"""Tests for ``population_stats`` — compute helpers, row projection, MERGE wiring."""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal.population_stats import (
    PopulationStatsConfig,
    PopulationStatsRow,
    TopCategory,
    _field_type_ddl,
    _row_to_record,
    _schema_to_ddl,
    _to_optional_float,
    _to_optional_int,
    compute_feature_population_stats,
    write_population_stats,
)


class TestTypeCoercion:
    def test_to_optional_float_passes_none(self):
        assert _to_optional_float(None) is None

    def test_to_optional_float_casts(self):
        assert _to_optional_float(3) == 3.0

    def test_to_optional_int_passes_none(self):
        assert _to_optional_int(None) is None

    def test_to_optional_int_casts(self):
        assert _to_optional_int(3.9) == 3


class TestRowToRecord:
    def test_numeric_row_projection(self):
        ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
        row = PopulationStatsRow(
            run_id="r1",
            feature_name="nps",
            dtype="numeric",
            count_nonnull=1000,
            mean=7.2,
            stddev=2.1,
            q01=1.0, q05=3.0, q25=5.0, q50=7.0, q75=9.0, q95=10.0, q99=10.0,
        )
        record = _row_to_record(row, ts)
        assert record["dtype"] == "numeric"
        assert record["q50"] == 7.0
        assert record["top_categories"] is None
        assert record["computed_at"] == ts

    def test_categorical_row_projection(self):
        ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
        row = PopulationStatsRow(
            run_id="r1",
            feature_name="segment",
            dtype="categorical",
            count_nonnull=500,
            top_categories=[
                TopCategory(value="SMB", count=300, share=0.6),
                TopCategory(value="Mid", count=200, share=0.4),
            ],
        )
        record = _row_to_record(row, ts)
        assert record["top_categories"] == [
            {"value": "SMB", "count": 300, "share": 0.6},
            {"value": "Mid", "count": 200, "share": 0.4},
        ]
        assert record["mean"] is None


class TestSchemaToDdl:
    def test_contains_quantile_and_category_columns(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.causal.schemas import feature_population_stats_schema

        ddl = _schema_to_ddl(feature_population_stats_schema())
        for required in (
            "run_id STRING", "feature_name STRING", "dtype STRING",
            "count_nonnull BIGINT", "mean DOUBLE", "stddev DOUBLE",
            "q01 DOUBLE", "q05 DOUBLE", "q25 DOUBLE", "q50 DOUBLE",
            "q75 DOUBLE", "q95 DOUBLE", "q99 DOUBLE",
            "computed_at TIMESTAMP",
        ):
            assert required in ddl, f"missing {required!r}"
        assert "top_categories" in ddl

    def test_field_type_ddl_handles_array_of_struct(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.causal.schemas import feature_population_stats_schema

        struct = feature_population_stats_schema()
        top_cats = next(f for f in struct.fields if f.name == "top_categories")
        rendered = _field_type_ddl(top_cats.dataType)
        assert rendered.startswith("ARRAY<")


class TestComputeFeaturePopulationStats:
    def test_returns_empty_when_no_features(self):
        result = compute_feature_population_stats(MagicMock(), run_id="r1")
        assert result == []

    def test_numeric_batched_agg_invocation(self):
        train = MagicMock()
        head_row = {
            "cnt_0": 10, "avg_0": 5.0, "std_0": 1.5, "pct_0": [0.1, 0.3, 1, 5, 9, 9.9, 9.99],
            "cnt_1": 8, "avg_1": 3.0, "std_1": 1.0, "pct_1": [None, None, None, None, None, None, None],
        }
        train.agg.return_value.head.return_value = head_row

        rows = compute_feature_population_stats(
            train, run_id="r1", numeric_features=["a", "b"],
        )
        assert len(rows) == 2
        assert rows[0].feature_name == "a"
        assert rows[0].mean == 5.0
        assert rows[0].q05 == 0.3
        assert rows[0].q95 == 9.9
        assert rows[1].q50 is None
        assert rows[1].mean == 3.0
        train.agg.assert_called_once()

    def test_numeric_batches_when_exceeding_batch_size(self, monkeypatch):
        from customer_retention.stages.causal import population_stats as ps

        monkeypatch.setattr(ps, "_NUMERIC_BATCH_SIZE", 2)
        train = MagicMock()

        def head_for_batch():
            head_row = {}
            for i in range(2):
                head_row[f"cnt_{i}"] = i + 1
                head_row[f"avg_{i}"] = float(i)
                head_row[f"std_{i}"] = 1.0
                head_row[f"pct_{i}"] = [0, 0, 0, 0, 0, 0, 0]
            return head_row

        train.agg.return_value.head.side_effect = lambda: head_for_batch()
        rows = ps.compute_feature_population_stats(
            train, run_id="r1", numeric_features=["a", "b", "c"],
        )
        assert [r.feature_name for r in rows] == ["a", "b", "c"]
        assert train.agg.call_count == 2  # two batches

    def test_categorical_top_k(self):
        train = MagicMock()
        non_null = train.filter.return_value
        non_null.count.return_value = 1000

        counts_df = non_null.groupBy.return_value.agg.return_value.orderBy.return_value.limit.return_value
        counts_df.collect.return_value = [
            {"value": "A", "count": 600},
            {"value": "B", "count": 400},
        ]
        rows = compute_feature_population_stats(
            train, run_id="r1", categorical_features=["segment"],
        )
        assert len(rows) == 1
        assert rows[0].dtype == "categorical"
        assert rows[0].feature_name == "segment"
        assert rows[0].count_nonnull == 1000
        assert rows[0].top_categories[0].value == "A"
        assert rows[0].top_categories[0].share == pytest.approx(0.6)
        assert rows[0].top_categories[1].share == pytest.approx(0.4)

    def test_categorical_share_uses_nonnull_denominator(self):
        train = MagicMock()
        non_null = train.filter.return_value
        non_null.count.return_value = 500  # only 500 non-null even if train has more
        counts_df = non_null.groupBy.return_value.agg.return_value.orderBy.return_value.limit.return_value
        counts_df.collect.return_value = [{"value": "A", "count": 500}]
        rows = compute_feature_population_stats(
            train, run_id="r1", categorical_features=["segment"],
        )
        assert rows[0].count_nonnull == 500
        assert rows[0].top_categories[0].share == pytest.approx(1.0)

    def test_categorical_with_zero_nonnull_returns_empty_top(self):
        train = MagicMock()
        non_null = train.filter.return_value
        non_null.count.return_value = 0
        non_null.groupBy.return_value.agg.return_value.orderBy.return_value.limit.return_value.collect.return_value = []
        rows = compute_feature_population_stats(
            train, run_id="r1", categorical_features=["segment"],
        )
        assert len(rows) == 1
        assert rows[0].top_categories is None
        assert rows[0].count_nonnull == 0


class TestWritePopulationStats:
    def test_empty_rows_is_noop(self):
        spark = MagicMock()
        n = write_population_stats(
            PopulationStatsConfig(spark=spark, table_fqn="c.s.population_stats", run_id="r", rows=[])
        )
        assert n == 0
        spark.sql.assert_not_called()

    def test_merge_wires_correctly(self, monkeypatch):
        pytest.importorskip("pyspark")
        delta_tables_module = types.ModuleType("delta.tables")
        delta_module = types.ModuleType("delta")
        merge_builder = MagicMock()
        merge_builder.merge.return_value = merge_builder
        merge_builder.whenMatchedUpdateAll.return_value = merge_builder
        merge_builder.whenNotMatchedInsertAll.return_value = merge_builder
        target_table = MagicMock()
        target_table.alias.return_value = merge_builder
        delta_table_cls = MagicMock()
        delta_table_cls.forName.return_value = target_table
        delta_tables_module.DeltaTable = delta_table_cls
        delta_module.tables = delta_tables_module
        monkeypatch.setitem(sys.modules, "delta", delta_module)
        monkeypatch.setitem(sys.modules, "delta.tables", delta_tables_module)

        spark = MagicMock()
        spark.createDataFrame.return_value.alias.return_value = "aliased"

        rows = [PopulationStatsRow(run_id="r1", feature_name="nps", dtype="numeric")]
        n = write_population_stats(
            PopulationStatsConfig(spark=spark, table_fqn="c.s.population_stats", run_id="r1", rows=rows)
        )
        assert n == 1

        create_sql = spark.sql.call_args.args[0]
        assert "CREATE TABLE IF NOT EXISTS c.s.population_stats" in create_sql
        assert "USING DELTA" in create_sql

        merge_condition = merge_builder.merge.call_args.args[1]
        assert "t.run_id = s.run_id" in merge_condition
        assert "t.feature_name = s.feature_name" in merge_condition
