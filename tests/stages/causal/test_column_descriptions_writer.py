"""Tests for ``column_descriptions_writer`` — row projection and MERGE wiring."""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from customer_retention.stages.causal.column_descriptions_writer import (
    ColumnDescriptionRow,
    ColumnDescriptionsConfig,
    _field_type_ddl,
    _row_to_record,
    _schema_to_ddl,
    bootstrap_column_descriptions,
    write_column_descriptions,
)


class TestRowToRecord:
    def test_projects_every_field(self):
        ts = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
        verified = datetime(2026, 1, 1, tzinfo=timezone.utc)
        row = ColumnDescriptionRow(
            table="account",
            column_name="NPS_SCORE",
            catalog="prod",
            schema="salesforce",
            business_name="Net Promoter Score",
            business_definition="0-10 loyalty score.",
            unit="score_0_10",
            polarity="high_is_good",
            pii_class="none",
            value_examples="0,5,10",
            last_verified_at=verified,
            source="manual",
        )
        record = _row_to_record(row, ts)
        assert record == {
            "catalog": "prod",
            "schema": "salesforce",
            "table": "account",
            "column_name": "NPS_SCORE",
            "business_name": "Net Promoter Score",
            "business_definition": "0-10 loyalty score.",
            "unit": "score_0_10",
            "polarity": "high_is_good",
            "pii_class": "none",
            "value_examples": "0,5,10",
            "last_verified_at": verified,
            "source": "manual",
            "written_at": ts,
        }

    def test_missing_catalog_and_schema_stay_none(self):
        record = _row_to_record(
            ColumnDescriptionRow(table="t", column_name="c"),
            datetime.now(timezone.utc),
        )
        assert record["catalog"] is None
        assert record["schema"] is None


class TestSchemaToDdl:
    def test_contains_every_column(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.causal.schemas import column_descriptions_schema

        ddl = _schema_to_ddl(column_descriptions_schema())
        for required in (
            "catalog STRING",
            "schema STRING",
            "table STRING",
            "column_name STRING",
            "business_name STRING",
            "business_definition STRING",
            "unit STRING",
            "polarity STRING",
            "pii_class STRING",
            "value_examples STRING",
            "last_verified_at TIMESTAMP",
            "source STRING",
            "written_at TIMESTAMP",
        ):
            assert required in ddl, f"missing {required!r} in DDL: {ddl}"

    def test_field_type_ddl_handles_timestamp(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import TimestampType

        assert _field_type_ddl(TimestampType()) == "TIMESTAMP"


class TestWriteColumnDescriptionsEarlyReturn:
    def test_empty_rows_is_noop(self):
        spark = MagicMock()
        n = write_column_descriptions(
            ColumnDescriptionsConfig(spark=spark, table_fqn="c.s.column_descriptions", rows=[])
        )
        assert n == 0
        spark.sql.assert_not_called()
        spark.createDataFrame.assert_not_called()


class TestWriteColumnDescriptionsMerge:
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
            ColumnDescriptionRow(
                table="account",
                column_name="NPS_SCORE",
                catalog="prod",
                schema="salesforce",
                business_definition="score",
                source="imported_from_md",
            ),
            ColumnDescriptionRow(table="account", column_name="ACCOUNT_ID"),
        ]
        n = write_column_descriptions(
            ColumnDescriptionsConfig(spark=spark, table_fqn="c.s.column_descriptions", rows=rows)
        )
        assert n == 2

        create_sql = spark.sql.call_args.args[0]
        assert "CREATE TABLE IF NOT EXISTS c.s.column_descriptions" in create_sql
        assert "USING DELTA" in create_sql

        records = spark.createDataFrame.call_args.args[0]
        assert {r["column_name"] for r in records} == {"NPS_SCORE", "ACCOUNT_ID"}
        assert all(isinstance(r["written_at"], datetime) for r in records)

        delta_table_cls.forName.assert_called_once_with(spark, "c.s.column_descriptions")
        merge_condition = merge_builder.merge.call_args.args[1]
        assert "t.table = s.table" in merge_condition
        assert "t.column_name = s.column_name" in merge_condition
        assert "coalesce(t.catalog, '')" in merge_condition
        assert "coalesce(t.schema, '')" in merge_condition
        merge_builder.whenMatchedUpdateAll.assert_called_once()
        merge_builder.whenNotMatchedInsertAll.assert_called_once()
        merge_builder.execute.assert_called_once()


class TestBootstrapColumnDescriptions:
    def test_chains_parse_and_write(self, tmp_path):
        md = tmp_path / "seed.md"
        md.write_text(
            "prod.s.account:\n"
            "\n"
            "ACCOUNT_ID (string): Unique identifier for the account.\n"
        )
        spark = MagicMock()
        with patch(
            "customer_retention.stages.causal.column_descriptions_writer.write_column_descriptions",
            return_value=1,
        ) as mocked_write:
            n = bootstrap_column_descriptions(spark, "c.s.column_descriptions", md)
        assert n == 1
        assert mocked_write.call_count == 1
        cfg = mocked_write.call_args.args[0]
        assert cfg.spark is spark
        assert cfg.table_fqn == "c.s.column_descriptions"
        assert len(cfg.rows) == 1
        assert cfg.rows[0].column_name == "ACCOUNT_ID"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bootstrap_column_descriptions(MagicMock(), "c.s.t", tmp_path / "nope.md")
