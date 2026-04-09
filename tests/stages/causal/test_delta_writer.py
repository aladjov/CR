"""Tests for the Delta MERGE / overwrite helpers.

The pyspark in this environment is the Databricks Connect flavor which
refuses local SparkSession.builder, so these tests use mocks against the
``spark`` interface rather than a live session. The mocks verify the right
Spark API surface is invoked with the right arguments — coverage of the
data semantics happens in integration tests against a real Databricks
workspace.

After the bug-fix rewrite, ``merge_into`` calls
``delta.tables.DeltaTable.forName(spark, fqn).merge(...).execute()``
directly. The previous version assembled raw MERGE SQL and used
``createOrReplaceTempView`` plus three full table scans to fabricate
inserted/updated counts; both have been removed. The DeltaTable Python
API gives us the same upsert semantics for free.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
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


def _make_fake_spark(table_exists=True):
    """Build a fake SparkSession with the surface ``delta_writer`` touches."""
    fake_writer = MagicMock(name="DataFrameWriter")
    fake_writer.format.return_value = fake_writer
    fake_writer.mode.return_value = fake_writer
    fake_writer.option.return_value = fake_writer
    fake_writer.saveAsTable = MagicMock()

    fake_df = MagicMock(name="DataFrame")
    fake_df.write = fake_writer
    fake_df.alias = MagicMock(return_value=fake_df)

    fake_catalog = MagicMock()
    fake_catalog.tableExists.return_value = table_exists

    spark = MagicMock(name="SparkSession")
    spark.createDataFrame.return_value = fake_df
    spark.catalog = fake_catalog
    return spark, fake_df, fake_writer


@pytest.fixture
def patched_delta_table(monkeypatch):
    """Patch ``delta.tables.DeltaTable`` so the merge chain can be inspected."""
    builder = MagicMock(name="MergeBuilder")
    builder.whenMatchedUpdate.return_value = builder
    builder.whenNotMatchedInsert.return_value = builder
    builder.execute = MagicMock()

    delta_target = MagicMock(name="DeltaTarget")
    delta_target_aliased = MagicMock(name="DeltaTargetAliased")
    delta_target.alias.return_value = delta_target_aliased
    delta_target_aliased.merge.return_value = builder

    fake_delta_tables = MagicMock(name="delta.tables")
    fake_delta_tables.DeltaTable = MagicMock(return_value=delta_target)
    fake_delta_tables.DeltaTable.forName = MagicMock(return_value=delta_target)

    fake_delta_pkg = MagicMock(name="delta")
    fake_delta_pkg.tables = fake_delta_tables

    monkeypatch.setitem(sys.modules, "delta", fake_delta_pkg)
    monkeypatch.setitem(sys.modules, "delta.tables", fake_delta_tables)
    return SimpleNamespace(
        delta_table_class=fake_delta_tables.DeltaTable,
        target=delta_target,
        target_aliased=delta_target_aliased,
        builder=builder,
    )


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
    def test_requires_merge_keys(self, patched_delta_table):
        spark, _, _ = _make_fake_spark()
        with pytest.raises(ValueError, match="merge_key"):
            delta_writer.merge_into(
                spark,
                rows=[{"playbook_id": "alpha", "version": "1.0.0", "name": "A"}],
                schema=playbook_catalog_schema(),
                table_fqn="test.cat.tbl",
                merge_keys=[],
            )

    def test_empty_rows_short_circuits(self, patched_delta_table):
        spark, _, _ = _make_fake_spark()
        result = delta_writer.merge_into(
            spark,
            rows=[],
            schema=playbook_catalog_schema(),
            table_fqn="test.cat.tbl",
            merge_keys=["playbook_id", "version"],
        )
        # Empty source returns the source-row-count metric without invoking
        # the Delta merge builder
        assert result == {"source_rows": 0}
        patched_delta_table.target_aliased.merge.assert_not_called()

    def test_merge_invokes_delta_merge_builder_with_correct_join_and_set(
        self, patched_delta_table
    ):
        spark, _, _ = _make_fake_spark(table_exists=True)
        rows = [
            {"playbook_id": "alpha", "version": "1.0.0", "name": "Alpha"},
            {"playbook_id": "beta", "version": "1.0.0", "name": "Beta"},
        ]
        result = delta_writer.merge_into(
            spark,
            rows=rows,
            schema=playbook_catalog_schema(),
            table_fqn="test.cat.playbook_catalog",
            merge_keys=["playbook_id", "version"],
        )
        # DeltaTable.forName was called with the table FQN
        patched_delta_table.delta_table_class.forName.assert_called_once_with(
            spark, "test.cat.playbook_catalog"
        )
        # The merge() join condition uses both merge keys
        merge_call = patched_delta_table.target_aliased.merge.call_args
        join_condition = merge_call.args[1]
        assert "target.playbook_id = source.playbook_id" in join_condition
        assert "target.version = source.version" in join_condition
        # whenMatchedUpdate set excludes the merge keys
        update_set = patched_delta_table.builder.whenMatchedUpdate.call_args.kwargs["set"]
        assert "name" in update_set
        assert "playbook_id" not in update_set
        assert "version" not in update_set
        # whenNotMatchedInsert covers every schema field
        insert_set = patched_delta_table.builder.whenNotMatchedInsert.call_args.kwargs["values"]
        for field in playbook_catalog_schema().fields:
            assert field.name in insert_set
        patched_delta_table.builder.execute.assert_called_once()
        assert result == {"source_rows": 2}

    def test_merge_creates_target_table_if_missing(self, patched_delta_table):
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
        patched_delta_table.builder.execute.assert_called_once()


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
# Note: ``_count_matching`` was removed alongside the bug-fix rewrite.
# It existed only to fabricate inserted/updated counts via three full
# table scans, and the result was structurally wrong (post - pre rows
# does not equal updated rows). Operators who need accurate counts can
# query ``DeltaTable.forName(spark, fqn).history()``.
# ---------------------------------------------------------------------------
