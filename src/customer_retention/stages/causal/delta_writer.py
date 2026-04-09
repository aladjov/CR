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

The module is intentionally thin — Delta-specific bits are isolated here so
the loaders and the higher-level orchestrator can stay free of Spark imports.

The functions accept a fully-qualified table name like
``catalog.schema.table_name``. Callers are expected to format that string
themselves (typically using ``CATALOG`` and ``SCHEMA`` from
``customer_retention.core.config``); this module does not own catalog/schema
resolution.
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
    """Idempotent upsert: insert new rows, update changed rows, leave the rest.

    Uses Delta MERGE INTO with a natural key. The target table is created
    automatically with the supplied schema if it does not exist.

    ``merge_keys`` must reference columns present in both ``rows`` and the
    schema; this is the join condition for the MERGE.

    Returns ``{"inserted": int, "updated": int}`` based on the source
    versus existing-target row counts. (Spark does not surface MERGE
    affected-row counts portably, so the return is a best-effort estimate
    derived from a pre-merge count diff.)
    """
    if not merge_keys:
        raise ValueError("merge_into requires at least one merge_key")

    _ensure_table_exists(spark, table_fqn, schema)

    df = _build_dataframe(spark, rows, schema)
    if not rows:
        logger.info("merge_into %s skipped (empty source rows)", table_fqn)
        return {"inserted": 0, "updated": 0}

    pre_count = spark.table(table_fqn).count()
    matching_pre_count = _count_matching(spark, df, table_fqn, merge_keys)

    df.createOrReplaceTempView("_causal_merge_source")
    on_clause = " AND ".join(f"target.{k} = source.{k}" for k in merge_keys)
    update_set = ", ".join(
        f"target.{f.name} = source.{f.name}" for f in schema.fields if f.name not in merge_keys
    )
    insert_cols = ", ".join(f.name for f in schema.fields)
    insert_vals = ", ".join(f"source.{f.name}" for f in schema.fields)

    sql = f"""
    MERGE INTO {table_fqn} AS target
    USING _causal_merge_source AS source
    ON {on_clause}
    WHEN MATCHED THEN UPDATE SET {update_set}
    WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """
    spark.sql(sql)
    spark.catalog.dropTempView("_causal_merge_source")

    post_count = spark.table(table_fqn).count()
    inserted = max(post_count - pre_count, 0)
    updated = max(matching_pre_count, 0)
    logger.info("merged %d rows into %s (inserted=%d, updated=%d)", len(rows), table_fqn, inserted, updated)
    return {"inserted": inserted, "updated": updated}


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
        # createDataFrame with empty list still respects the schema
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


def _count_matching(
    spark: "SparkSession",
    source_df: "DataFrame",
    table_fqn: str,
    merge_keys: Sequence[str],
) -> int:
    """Count source rows that already exist in the target on the merge keys."""
    # Skip the join when the source is empty. Using limit(1).count() avoids
    # .rdd access (banned on Databricks shared clusters / Unity Catalog).
    if source_df.limit(1).count() == 0:
        return 0
    target_df = spark.table(table_fqn).select(*merge_keys).distinct()
    return source_df.select(*merge_keys).join(target_df, list(merge_keys), "inner").count()
