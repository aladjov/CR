"""Per-source-column ``column_descriptions`` writer.

Writes the slowly-changing business-definition layer so the interpretation
pass can lift raw column names (``missed_payment_count_90d``) into
business phrases (``count of missed payments over last 90 days``) without
hitting the LLM. Bootstrapped from ``docs/sps_table_descriptions.md`` and
maintained manually afterwards.

Grain: one row per ``(catalog, schema, table, column_name)``. Re-running
the bootstrap upserts via MERGE — identical pattern to
``run_context_writer`` / ``feature_meta_writer``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession


_DEFAULT_TARGET_TABLE = "column_descriptions"


@dataclass
class ColumnDescriptionRow:
    """One source-column business-definition row.

    ``table`` and ``column_name`` are the only required fields. ``catalog``
    and ``schema`` are nullable because some fixtures / small demos don't
    use three-part naming.
    """

    table: str
    column_name: str
    catalog: Optional[str] = None
    schema: Optional[str] = None
    business_name: Optional[str] = None
    business_definition: Optional[str] = None
    unit: Optional[str] = None
    polarity: Optional[str] = None
    pii_class: Optional[str] = None
    value_examples: Optional[str] = None
    last_verified_at: Optional[datetime] = None
    source: Optional[str] = None


@dataclass
class ColumnDescriptionsConfig:
    """Batch write handle: N rows into one table."""

    spark: "SparkSession"
    table_fqn: str
    rows: List[ColumnDescriptionRow] = field(default_factory=list)


def write_column_descriptions(config: ColumnDescriptionsConfig) -> int:
    """Upsert ``config.rows`` into ``{catalog}.{schema}.column_descriptions``.

    Composite key ``(catalog, schema, table, column_name)`` — NULL-safe on
    ``catalog`` / ``schema`` so rows without three-part naming still merge.
    Empty ``rows`` is a no-op returning 0.
    """
    if not config.rows:
        logger.info("write_column_descriptions: empty rows, skipping %s", config.table_fqn)
        return 0

    from delta.tables import DeltaTable

    from customer_retention.stages.causal.schemas import get_schema

    schema = get_schema(_DEFAULT_TARGET_TABLE)
    config.spark.sql(
        f"CREATE TABLE IF NOT EXISTS {config.table_fqn} "
        f"({_schema_to_ddl(schema)}) USING DELTA"
    )

    written_at = datetime.now(timezone.utc)
    records = [_row_to_record(row, written_at) for row in config.rows]
    source = config.spark.createDataFrame(records, schema=schema)

    (
        DeltaTable.forName(config.spark, config.table_fqn)
        .alias("t")
        .merge(
            source.alias("s"),
            (
                "t.table = s.table "
                "AND t.column_name = s.column_name "
                "AND coalesce(t.catalog, '') = coalesce(s.catalog, '') "
                "AND coalesce(t.schema, '') = coalesce(s.schema, '')"
            ),
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info(
        "wrote %d column_descriptions rows table=%s",
        len(records), config.table_fqn,
    )
    return len(records)


def bootstrap_column_descriptions(
    spark: "SparkSession",
    table_fqn: str,
    md_path: Path | str,
) -> int:
    """Parse ``docs/sps_table_descriptions.md`` and upsert every column.

    Single call NB00 0.15 makes for the one-shot bootstrap. Returns the
    number of rows upserted. Missing file raises ``FileNotFoundError`` —
    fail fast per ``Coding_Practices.md``.
    """
    from customer_retention.stages.causal.interpretation.markdown_bootstrap import (
        parse_table_descriptions_md,
    )

    rows = parse_table_descriptions_md(md_path)
    return write_column_descriptions(
        ColumnDescriptionsConfig(spark=spark, table_fqn=table_fqn, rows=rows)
    )


def _row_to_record(row: ColumnDescriptionRow, written_at: datetime) -> dict:
    return {
        "catalog": row.catalog,
        "schema": row.schema,
        "table": row.table,
        "column_name": row.column_name,
        "business_name": row.business_name,
        "business_definition": row.business_definition,
        "unit": row.unit,
        "polarity": row.polarity,
        "pii_class": row.pii_class,
        "value_examples": row.value_examples,
        "last_verified_at": row.last_verified_at,
        "source": row.source,
        "written_at": written_at,
    }


_TYPE_DDL_MAP = {
    "StringType": "STRING",
    "IntegerType": "INT",
    "LongType": "BIGINT",
    "DoubleType": "DOUBLE",
    "BooleanType": "BOOLEAN",
    "TimestampType": "TIMESTAMP",
}


def _schema_to_ddl(schema: Any) -> str:
    return ", ".join(
        f"{f.name} {_field_type_ddl(f.dataType)}" for f in schema.fields
    )


def _field_type_ddl(dtype: Any) -> str:
    name = type(dtype).__name__
    if name in _TYPE_DDL_MAP:
        return _TYPE_DDL_MAP[name]
    if name == "ArrayType":
        return f"ARRAY<{_field_type_ddl(dtype.elementType)}>"
    return dtype.simpleString().upper()
