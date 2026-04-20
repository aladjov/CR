"""Delta write helpers for the causal track.

Two write modes are supported:

- ``overwrite_table``: full replacement of the target table (used for the
  small definition tables that are authoritative from YAML — re-publish
  always wins). Schema is replaced with the explicit ``StructType`` from
  ``schemas.py`` so type drift is impossible.

- ``merge_into``: idempotent upsert on a natural key (used for the
  per-derivation-run tables like ``archetype_catalog`` and per-scoring-run
  tables like ``eligibility_snapshot``). Re-running with the same inputs is
  a no-op.

Implementation notes:

- ``merge_into`` uses ``delta.tables.DeltaTable.merge()`` directly. No
  ``createOrReplaceTempView`` (which conflicts with the
  ``register_temp_view`` audit contract introduced in
  ``core/compat/__init__.py``) and no manual MERGE SQL string assembly.
- ``merge_into`` returns the source row count, not a fabricated
  inserted/updated estimate. The previous implementation triggered three
  full table scans (``pre_count``, ``_count_matching`` join+count,
  ``post_count``) just to compute a row-count delta — and the result was
  wrong because ``post - pre`` does not equal ``updated``. The accurate
  metrics live in Delta's commit history, accessible via
  ``DeltaTable.history()`` when an operator wants them.

Callers pass a fully-qualified table name (``catalog.schema.table_name``).
This module does not own catalog/schema resolution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.types import StructType


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def overwrite_table(
    spark: "SparkSession",
    rows: Sequence[Dict[str, Any]],
    schema: "StructType",
    table_fqn: str,
) -> int:
    """Replace the contents of ``table_fqn`` with ``rows``.

    Used for the definition tables (``playbook_catalog``, ``playbook_steps``,
    ``response_schemas``, ``vocabularies``, ``decision_policy``) which are
    authoritative from YAML — every publish is a full replacement, with the
    explicit ``StructType`` enforcing column shape.

    Empty ``rows`` produces an empty table with the correct schema (used by
    fresh-init paths so downstream joins always have a valid table to bind
    to).

    Returns the number of rows written.
    """
    df = _build_dataframe(spark, rows, schema)
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_fqn)
    )
    count = len(rows)
    logger.info("overwrote %s with %d rows", table_fqn, count)
    return count


def merge_into(
    spark: "SparkSession",
    rows: Sequence[Dict[str, Any]],
    schema: "StructType",
    table_fqn: str,
    merge_keys: Sequence[str],
) -> Dict[str, int]:
    """Idempotent upsert via ``DeltaTable.merge()`` from driver-side row dicts.

    Insert new rows, update changed rows on the natural key, and leave
    everything else alone. The target table is created automatically with
    the supplied schema if it does not exist.

    Used for the small definition tables (≤ a few hundred rows). For
    distributed merges (snapshot writer, derivation outputs that ride a
    Spark DataFrame instead of being collected to the driver) call
    :func:`merge_dataframe_into` directly.

    Returns ``{"source_rows": int}`` so the caller has at least one number
    to log without paying for full pre/post table scans. Operators who
    need accurate ``inserted``/``updated`` counts should consult
    ``DeltaTable.forName(spark, fqn).history()``.
    """
    if not merge_keys:
        raise ValueError("merge_into requires at least one merge_key")

    if not rows:
        _ensure_table_exists(spark, table_fqn, schema)
        logger.info("merge_into %s skipped (empty source rows)", table_fqn)
        return {"source_rows": 0}

    source_df = _build_dataframe(spark, rows, schema)
    merge_dataframe_into(spark, source_df, schema, table_fqn, merge_keys)

    logger.info("merged %d rows into %s via DeltaTable.merge", len(rows), table_fqn)
    return {"source_rows": len(rows)}


def merge_dataframe_into(
    spark: "SparkSession",
    source_df: "DataFrame",
    schema: "StructType",
    table_fqn: str,
    merge_keys: Sequence[str],
) -> None:
    """Idempotent upsert via ``DeltaTable.merge()`` from a Spark DataFrame.

    The merge runs entirely on the cluster — ``source_df`` is **never**
    collected to the driver. Used by the snapshot writer where the source
    can be tens of millions of rows wide (population × |playbooks|).

    Auto-creates the target table from ``schema`` when it does not exist,
    and additively evolves the schema (``ALTER TABLE ADD COLUMN``) when
    the desired schema has fields the existing table is missing. Both
    safety nets keep deployments single-shot — a schema addition never
    requires a separate manual migration step before the next merge.
    """
    if not merge_keys:
        raise ValueError("merge_dataframe_into requires at least one merge_key")
    _ensure_table_exists(spark, table_fqn, schema)
    _ensure_table_schema(spark, table_fqn, schema)
    _execute_delta_merge(spark, source_df, table_fqn, merge_keys, schema)
    logger.info("merged Spark DataFrame into %s via DeltaTable.merge", table_fqn)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_dataframe(
    spark: "SparkSession",
    rows: Sequence[Dict[str, Any]],
    schema: "StructType",
) -> "DataFrame":
    """Build a typed Spark DataFrame from row dicts.

    Missing keys in a row become None (matching the field nullability), and
    extra keys are dropped. The explicit schema makes type coercion strict —
    a string in an IntegerType field will fail loudly here rather than
    silently corrupt downstream.
    """
    field_order = [f.name for f in schema.fields]
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized_rows.append({name: row.get(name) for name in field_order})
    if not normalized_rows:
        return spark.createDataFrame([], schema=schema)
    return spark.createDataFrame(normalized_rows, schema=schema)


def _ensure_table_exists(spark: "SparkSession", table_fqn: str, schema: "StructType") -> None:
    """Create an empty Delta table at ``table_fqn`` if it doesn't exist."""
    if spark.catalog.tableExists(table_fqn):
        return
    empty_df = spark.createDataFrame([], schema=schema)
    (
        empty_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_fqn)
    )
    logger.info("created empty Delta table %s", table_fqn)


def _ensure_table_schema(
    spark: "SparkSession", table_fqn: str, schema: "StructType"
) -> None:
    """Additively evolve an existing Delta table's schema to match ``schema``.

    The Delta MERGE writer assembles its UPDATE/INSERT clauses from the
    desired ``schema``. When the target table on disk is older than the
    schema (e.g. a new field was added in code), the merge fails with
    ``DELTA_MERGE_UNRESOLVED_EXPRESSION`` because the UPDATE clause names
    a column the target does not have. This helper closes that gap by
    issuing ``ALTER TABLE ADD COLUMN`` for every field present in
    ``schema`` but absent from the current table — additive only, never
    drop, never type-change. Type changes still surface explicitly at
    merge time so they are not silently coerced.

    Cheap to run on every merge: one ``DESCRIBE``-equivalent + a small
    set diff on the driver, plus one ``ALTER`` per missing column. The
    alters themselves are metadata-only Delta operations (no data
    rewrite). Idempotent: a table already in sync is a no-op.

    No-op when the table does not yet exist — the caller is expected to
    have run ``_ensure_table_exists`` first, which creates the table
    with the full ``schema`` so there are no missing fields by
    construction.
    """
    if not spark.catalog.tableExists(table_fqn):
        return
    actual = {f.name for f in spark.table(table_fqn).schema.fields}
    missing = [f for f in schema.fields if f.name not in actual]
    if not missing:
        return
    for field in missing:
        type_str = field.dataType.simpleString()
        spark.sql(f"ALTER TABLE {table_fqn} ADD COLUMN `{field.name}` {type_str}")
        logger.info(
            "added column %s %s to %s (additive schema evolution)",
            field.name,
            type_str,
            table_fqn,
        )


def _execute_delta_merge(
    spark: "SparkSession",
    source_df: "DataFrame",
    table_fqn: str,
    merge_keys: Sequence[str],
    schema: "StructType",
) -> None:
    """Run ``DeltaTable.forName(spark, fqn).merge(...).execute()``.

    Splitting this out lets unit tests patch ``delta.tables.DeltaTable``
    in isolation while keeping the public API surface clean.
    """
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, table_fqn)
    join_condition = " AND ".join(f"target.{k} = source.{k}" for k in merge_keys)
    update_set = {
        f.name: f"source.{f.name}" for f in schema.fields if f.name not in merge_keys
    }
    insert_set = {f.name: f"source.{f.name}" for f in schema.fields}
    builder = target.alias("target").merge(source_df.alias("source"), join_condition)
    if update_set:
        builder = builder.whenMatchedUpdate(set=update_set)
    builder.whenNotMatchedInsert(values=insert_set).execute()
