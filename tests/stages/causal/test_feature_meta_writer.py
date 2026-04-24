"""Tests for ``feature_meta_writer`` — row projection, rendering, MERGE wiring."""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal.feature_meta_writer import (
    FeatureMetaConfig,
    FeatureMetaRow,
    _field_type_ddl,
    _row_to_record,
    _schema_to_ddl,
    write_feature_meta,
)


class TestFeatureMetaRowWithRenderedPhrases:
    def test_fills_window_and_business_phrase_from_lineage(self):
        row = FeatureMetaRow(
            composite_name="cn1",
            feature_name="missed_payments_count_90d",
            aggregation_kind="count",
            window_days=90,
        )
        rendered = row.with_rendered_phrases(source_business_name="missed payments")
        assert rendered.window_phrase == "last 90 days"
        assert rendered.business_phrase == "count of missed payments over last 90 days"

    def test_preserves_existing_window_phrase(self):
        row = FeatureMetaRow(
            composite_name="cn1",
            feature_name="f",
            aggregation_kind="count",
            window_days=90,
            window_phrase="this quarter",
        )
        assert row.with_rendered_phrases("payments").window_phrase == "this quarter"

    def test_preserves_existing_business_phrase(self):
        row = FeatureMetaRow(
            composite_name="cn1",
            feature_name="f",
            aggregation_kind="count",
            window_days=90,
            business_phrase="hand-authored phrase",
        )
        assert row.with_rendered_phrases("payments").business_phrase == "hand-authored phrase"

    def test_falls_back_to_feature_name_when_no_business_name(self):
        row = FeatureMetaRow(
            composite_name="cn1",
            feature_name="missed_payments_count_90d",
            aggregation_kind="count",
            window_days=90,
        )
        assert "missed_payments_count_90d" in row.with_rendered_phrases().business_phrase

    def test_null_window_days_renders_lifetime(self):
        row = FeatureMetaRow(
            composite_name="cn1",
            feature_name="tenure_days",
            aggregation_kind="passthrough",
            window_days=None,
        )
        rendered = row.with_rendered_phrases("tenure in days")
        assert rendered.window_phrase == "lifetime"
        assert rendered.business_phrase == "tenure in days"

    def test_idempotent(self):
        row = FeatureMetaRow(
            composite_name="cn1",
            feature_name="f",
            aggregation_kind="sum",
            window_days=30,
        )
        once = row.with_rendered_phrases("revenue")
        twice = once.with_rendered_phrases("revenue")
        assert once == twice


class TestRowToRecord:
    def test_projects_every_schema_field(self):
        row = FeatureMetaRow(
            composite_name="cn1",
            feature_name="f",
            source_columns=["a", "b"],
            source_table="silver_x",
            aggregation_kind="sum",
            window_days=30,
            window_phrase="last 30 days",
            target_dependency=False,
            mask_future=True,
            polarity="high_is_bad",
            business_phrase="sum of revenue over last 30 days",
        )
        ts = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
        record = _row_to_record("run-1", row, ts)
        assert record == {
            "run_id": "run-1",
            "composite_name": "cn1",
            "feature_name": "f",
            "source_columns": ["a", "b"],
            "source_table": "silver_x",
            "aggregation_kind": "sum",
            "window_days": 30,
            "window_phrase": "last 30 days",
            "target_dependency": False,
            "mask_future": True,
            "polarity": "high_is_bad",
            "business_phrase": "sum of revenue over last 30 days",
            "written_at": ts,
        }

    def test_source_columns_none_stays_none_not_empty_list(self):
        record = _row_to_record(
            "run-1",
            FeatureMetaRow(composite_name="cn1", feature_name="f"),
            datetime.now(timezone.utc),
        )
        assert record["source_columns"] is None

    def test_source_columns_preserved_as_list_copy(self):
        original = ["a", "b"]
        row = FeatureMetaRow(composite_name="cn1", feature_name="f", source_columns=original)
        record = _row_to_record("run-1", row, datetime.now(timezone.utc))
        assert record["source_columns"] == ["a", "b"]
        assert record["source_columns"] is not original


class TestSchemaToDdl:
    def test_contains_every_feature_meta_column(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.causal.schemas import feature_meta_schema

        ddl = _schema_to_ddl(feature_meta_schema())
        for required in (
            "run_id STRING",
            "composite_name STRING",
            "feature_name STRING",
            "source_columns ARRAY<STRING>",
            "source_table STRING",
            "aggregation_kind STRING",
            "window_days INT",
            "window_phrase STRING",
            "target_dependency BOOLEAN",
            "mask_future BOOLEAN",
            "polarity STRING",
            "business_phrase STRING",
            "written_at TIMESTAMP",
        ):
            assert required in ddl, f"missing {required!r} in DDL: {ddl}"

    def test_field_type_ddl_handles_array_of_string(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import ArrayType, StringType

        assert _field_type_ddl(ArrayType(StringType())) == "ARRAY<STRING>"

    def test_field_type_ddl_handles_boolean_and_int(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import BooleanType, IntegerType

        assert _field_type_ddl(BooleanType()) == "BOOLEAN"
        assert _field_type_ddl(IntegerType()) == "INT"


class TestWriteFeatureMetaEarlyReturn:
    def test_empty_rows_is_noop(self):
        spark = MagicMock()
        n = write_feature_meta(
            FeatureMetaConfig(spark=spark, table_fqn="c.s.feature_meta", run_id="r", rows=[])
        )
        assert n == 0
        spark.sql.assert_not_called()
        spark.createDataFrame.assert_not_called()


class TestWriteFeatureMetaMerge:
    """End-to-end write path with pyspark+delta stubs — verifies MERGE wiring
    without needing a real Spark session."""

    def test_create_table_and_merge_executes(self, monkeypatch):
        pytest.importorskip("pyspark")
        delta_tables_module = types.ModuleType("delta.tables")
        delta_module = types.ModuleType("delta")
        merge_builder = MagicMock()
        merge_builder.merge.return_value = merge_builder
        merge_builder.whenMatchedUpdateAll.return_value = merge_builder
        merge_builder.whenNotMatchedInsertAll.return_value = merge_builder
        merge_builder.execute.return_value = None

        target_table = MagicMock()
        target_table.alias.return_value = merge_builder

        delta_table_cls = MagicMock()
        delta_table_cls.forName.return_value = target_table
        delta_tables_module.DeltaTable = delta_table_cls
        delta_module.tables = delta_tables_module
        monkeypatch.setitem(sys.modules, "delta", delta_module)
        monkeypatch.setitem(sys.modules, "delta.tables", delta_tables_module)

        spark = MagicMock()
        source_df = MagicMock()
        source_df.alias.return_value = "aliased-source"
        spark.createDataFrame.return_value = source_df

        rows = [
            FeatureMetaRow(composite_name="cn1", feature_name="f_a", aggregation_kind="count", window_days=30),
            FeatureMetaRow(composite_name="cn1", feature_name="f_b", aggregation_kind="sum", window_days=90),
        ]
        n = write_feature_meta(
            FeatureMetaConfig(spark=spark, table_fqn="c.s.feature_meta", run_id="run-42", rows=rows)
        )

        assert n == 2
        spark.sql.assert_called_once()
        create_sql = spark.sql.call_args.args[0]
        assert "CREATE TABLE IF NOT EXISTS c.s.feature_meta" in create_sql
        assert "USING DELTA" in create_sql
        assert "run_id STRING" in create_sql

        createdf_args, createdf_kwargs = spark.createDataFrame.call_args
        records = createdf_args[0]
        assert [r["feature_name"] for r in records] == ["f_a", "f_b"]
        assert all(r["run_id"] == "run-42" for r in records)
        assert all(isinstance(r["written_at"], datetime) for r in records)
        assert "schema" in createdf_kwargs

        delta_table_cls.forName.assert_called_once_with(spark, "c.s.feature_meta")
        merge_builder.merge.assert_called_once()
        merge_condition = merge_builder.merge.call_args.args[1]
        assert "t.run_id = s.run_id" in merge_condition
        assert "t.composite_name = s.composite_name" in merge_condition
        assert "t.feature_name = s.feature_name" in merge_condition
        merge_builder.whenMatchedUpdateAll.assert_called_once()
        merge_builder.whenNotMatchedInsertAll.assert_called_once()
        merge_builder.execute.assert_called_once()
