"""Tests for the Delta MERGE / overwrite helpers.

The pyspark in this environment is the Databricks Connect flavor which
refuses local SparkSession.builder, so these tests use mocks against the
``spark`` interface rather than a live session. The mocks verify the right
Spark API surface is invoked with the right arguments — coverage of the
data semantics happens in integration tests against a real Databricks
workspace.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("pyspark")

from customer_retention.stages.causal import delta_writer
from customer_retention.stages.causal.schemas import (
    playbook_catalog_schema,
    playbook_steps_schema,
)

# ---------------------------------------------------------------------------
# Spark session mock
# ---------------------------------------------------------------------------


def _make_fake_spark(rows_in_table=0, table_exists=True):
    """Build a SimpleNamespace mimicking the SparkSession surface delta_writer touches."""
    fake_writer = MagicMock(name="DataFrameWriter")
    fake_writer.format.return_value = fake_writer
    fake_writer.mode.return_value = fake_writer
    fake_writer.option.return_value = fake_writer
    fake_writer.saveAsTable = MagicMock()

    fake_df = MagicMock(name="DataFrame")
    fake_df.write = fake_writer
    fake_df.count.return_value = rows_in_table
    fake_df.columns = []
    # limit().count() chain for _count_matching
    fake_limit = MagicMock(name="LimitedDF")
    fake_limit.count.return_value = rows_in_table
    fake_df.limit.return_value = fake_limit
    # select().distinct() chain
    fake_df.select.return_value = fake_df
    fake_df.distinct.return_value = fake_df
    fake_df.join.return_value = fake_df
    fake_df.createOrReplaceTempView = MagicMock()

    fake_catalog = MagicMock()
    fake_catalog.tableExists.return_value = table_exists
    fake_catalog.dropTempView = MagicMock()

    spark = MagicMock(name="SparkSession")
    spark.createDataFrame.return_value = fake_df
    spark.table.return_value = fake_df
    spark.catalog = fake_catalog
    spark.sql = MagicMock()
    return spark, fake_df, fake_writer


# ---------------------------------------------------------------------------
# _build_dataframe
# ---------------------------------------------------------------------------


class TestBuildDataframe:
    def test_passes_normalized_rows_with_all_schema_fields(self):
        spark, fake_df, _ = _make_fake_spark()
        rows = [{"playbook_id": "alpha", "version": "1.0.0", "name": "Alpha"}]
        delta_writer._build_dataframe(spark, rows, playbook_catalog_schema())

        # createDataFrame was called once with normalized rows + the schema
        assert spark.createDataFrame.call_count == 1
        call_args = spark.createDataFrame.call_args
        normalized_rows = call_args[0][0]
        assert len(normalized_rows) == 1
        normalized = normalized_rows[0]
        # Every schema field present in the normalized row (with None for missing)
        for field in playbook_catalog_schema().fields:
            assert field.name in normalized
        assert normalized["playbook_id"] == "alpha"
        assert normalized["version"] == "1.0.0"
        assert normalized["name"] == "Alpha"
        # Field absent from input → None in normalized output
        assert normalized["description"] is None
        assert normalized["cost_per_customer_default"] is None

    def test_extra_keys_are_dropped(self):
        spark, _, _ = _make_fake_spark()
        rows = [
            {
                "playbook_id": "alpha",
                "version": "1.0.0",
                "name": "Alpha",
                "not_a_real_field": "should be dropped",
            }
        ]
        delta_writer._build_dataframe(spark, rows, playbook_catalog_schema())
        normalized = spark.createDataFrame.call_args[0][0][0]
        assert "not_a_real_field" not in normalized

    def test_empty_rows_calls_create_with_empty_list(self):
        spark, _, _ = _make_fake_spark()
        delta_writer._build_dataframe(spark, [], playbook_catalog_schema())
        call_args = spark.createDataFrame.call_args
        assert call_args[0][0] == []
        # The schema is passed via the schema kwarg
        assert call_args.kwargs.get("schema") is playbook_catalog_schema() or \
               call_args.kwargs["schema"].fieldNames() == playbook_catalog_schema().fieldNames()


# ---------------------------------------------------------------------------
# overwrite_table
# ---------------------------------------------------------------------------


class TestOverwriteTable:
    def test_invokes_full_write_chain(self):
        spark, fake_df, fake_writer = _make_fake_spark()
        rows = [
            {"playbook_id": "alpha", "version": "1.0.0", "name": "Alpha"},
            {"playbook_id": "beta", "version": "1.0.0", "name": "Beta"},
        ]
        count = delta_writer.overwrite_table(
            spark, rows, playbook_catalog_schema(), "test.cat.playbook_catalog"
        )
        assert count == 2
        fake_writer.format.assert_called_with("delta")
        fake_writer.mode.assert_called_with("overwrite")
        fake_writer.option.assert_called_with("overwriteSchema", "true")
        fake_writer.saveAsTable.assert_called_with("test.cat.playbook_catalog")

    def test_zero_rows_still_calls_write(self):
        spark, fake_df, fake_writer = _make_fake_spark()
        count = delta_writer.overwrite_table(
            spark, [], playbook_steps_schema(), "test.cat.playbook_steps"
        )
        assert count == 0
        fake_writer.saveAsTable.assert_called_with("test.cat.playbook_steps")


# ---------------------------------------------------------------------------
# merge_into
# ---------------------------------------------------------------------------


class TestMergeInto:
    def test_requires_merge_keys(self):
        spark, _, _ = _make_fake_spark()
        with pytest.raises(ValueError, match="merge_key"):
            delta_writer.merge_into(
                spark,
                rows=[{"playbook_id": "alpha", "version": "1.0.0", "name": "A"}],
                schema=playbook_catalog_schema(),
                table_fqn="test.cat.tbl",
                merge_keys=[],
            )

    def test_empty_rows_short_circuits(self):
        spark, _, _ = _make_fake_spark()
        result = delta_writer.merge_into(
            spark,
            rows=[],
            schema=playbook_catalog_schema(),
            table_fqn="test.cat.tbl",
            merge_keys=["playbook_id", "version"],
        )
        # Returns the no-op result without calling spark.sql
        assert result == {"inserted": 0, "updated": 0}
        spark.sql.assert_not_called()

    def test_merge_emits_sql_with_correct_join_and_set_clauses(self):
        spark, fake_df, _ = _make_fake_spark(rows_in_table=2, table_exists=True)
        rows = [
            {"playbook_id": "alpha", "version": "1.0.0", "name": "Alpha"},
            {"playbook_id": "beta", "version": "1.0.0", "name": "Beta"},
        ]
        delta_writer.merge_into(
            spark,
            rows=rows,
            schema=playbook_catalog_schema(),
            table_fqn="test.cat.playbook_catalog",
            merge_keys=["playbook_id", "version"],
        )
        # spark.sql was called with the MERGE statement
        assert spark.sql.called
        sql = spark.sql.call_args[0][0]
        assert "MERGE INTO test.cat.playbook_catalog" in sql
        assert "USING _causal_merge_source" in sql
        # ON clause uses both merge keys
        assert "target.playbook_id = source.playbook_id" in sql
        assert "target.version = source.version" in sql
        # UPDATE SET excludes the merge keys
        assert "target.name = source.name" in sql
        assert "target.playbook_id = source.playbook_id" in sql.split("UPDATE SET")[0]
        # WHEN NOT MATCHED inserts every column
        assert "WHEN NOT MATCHED THEN INSERT" in sql

    def test_merge_creates_target_table_if_missing(self):
        spark, _, fake_writer = _make_fake_spark(table_exists=False)
        delta_writer.merge_into(
            spark,
            rows=[{"playbook_id": "x", "version": "1.0.0", "name": "X"}],
            schema=playbook_catalog_schema(),
            table_fqn="test.cat.new_table",
            merge_keys=["playbook_id"],
        )
        # tableExists returned False, so saveAsTable should have been invoked
        # at least once to create the empty target before MERGE
        save_calls = [c.args[0] for c in fake_writer.saveAsTable.call_args_list]
        assert "test.cat.new_table" in save_calls


# ---------------------------------------------------------------------------
# _ensure_table_exists
# ---------------------------------------------------------------------------


class TestEnsureTableExists:
    def test_no_op_when_table_already_exists(self):
        spark, _, _ = _make_fake_spark(table_exists=True)
        delta_writer._ensure_table_exists(spark, "test.cat.existing", playbook_catalog_schema())
        # createDataFrame must NOT be called when the table already exists
        spark.createDataFrame.assert_not_called()

    def test_creates_empty_delta_table_when_missing(self):
        spark, _, fake_writer = _make_fake_spark(table_exists=False)
        delta_writer._ensure_table_exists(spark, "test.cat.new", playbook_catalog_schema())
        spark.createDataFrame.assert_called_once()
        fake_writer.format.assert_called_with("delta")
        fake_writer.mode.assert_called_with("overwrite")
        fake_writer.saveAsTable.assert_called_with("test.cat.new")


# ---------------------------------------------------------------------------
# _count_matching
# ---------------------------------------------------------------------------


class TestCountMatching:
    def test_empty_source_returns_zero_without_querying_target(self):
        spark, fake_df, _ = _make_fake_spark(rows_in_table=0)
        # Source DataFrame has 0 rows (limit(1).count() == 0)
        empty_source = MagicMock(name="EmptySourceDF")
        empty_source.limit.return_value.count.return_value = 0

        result = delta_writer._count_matching(
            spark, empty_source, "test.cat.tbl", ["playbook_id", "version"]
        )
        assert result == 0
        # Crucially we did NOT call spark.table() — the empty short-circuit
        # avoided unnecessary I/O against the target
        spark.table.assert_not_called()

    def test_non_empty_source_joins_against_target(self):
        spark, fake_df, _ = _make_fake_spark(rows_in_table=5)
        source = MagicMock(name="SourceDF")
        source.limit.return_value.count.return_value = 1  # non-empty
        # Mock the source.select(...).join(target, keys, "inner").count() chain
        join_result = MagicMock()
        join_result.count.return_value = 3
        source.select.return_value.join.return_value = join_result

        result = delta_writer._count_matching(
            spark, source, "test.cat.tbl", ["playbook_id", "version"]
        )
        assert result == 3
        spark.table.assert_called_with("test.cat.tbl")
