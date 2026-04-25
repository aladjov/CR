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
    md_path: Path | str | None = None,
    *,
    llm_endpoint: Optional[str] = None,
    fallback_table_fqns: Optional[List[str]] = None,
) -> int:
    """Bootstrap ``column_descriptions`` from markdown, LLM, or both.

    Resolution order:

    1. **Markdown** — when ``md_path`` is given and exists, parse
       ``docs/sps_table_descriptions.md`` and upsert every column. This is
       the original single-call NB00 0.15 path.
    2. **LLM fallback** — when the markdown is absent (or ``md_path`` is
       None) and ``llm_endpoint`` resolves on this workspace, call
       ``column_describer.generate_column_descriptions`` over each table in
       ``fallback_table_fqns`` (defaults to ``[table_fqn]``) and upsert the
       generated descriptions. Marks each row ``source="llm_proposed"``.
    3. **Empty** — when neither path produces rows, the sidecar still
       exists as a Delta table (zero rows) so downstream consumers can
       operate on a definitive empty bag rather than failing on missing.

    Returns the number of rows upserted.
    """
    rows: List[ColumnDescriptionRow] = []

    if md_path is not None:
        path = Path(md_path)
        if path.exists():
            from customer_retention.stages.causal.interpretation.markdown_bootstrap import (
                parse_table_descriptions_md,
            )
            rows.extend(parse_table_descriptions_md(path))
            logger.info(
                "bootstrap_column_descriptions: parsed %d rows from %s",
                len(rows), path,
            )

    if not rows and llm_endpoint is not None:
        rows.extend(
            _llm_describe_columns(
                spark,
                llm_endpoint=llm_endpoint,
                table_fqns=fallback_table_fqns or [table_fqn],
            )
        )
        if rows:
            logger.info(
                "bootstrap_column_descriptions: LLM fallback produced %d rows via endpoint=%s",
                len(rows), llm_endpoint,
            )

    return write_column_descriptions(
        ColumnDescriptionsConfig(spark=spark, table_fqn=table_fqn, rows=rows)
    )


def _llm_describe_columns(
    spark: "SparkSession",
    *,
    llm_endpoint: str,
    table_fqns: List[str],
) -> List[ColumnDescriptionRow]:
    """Call ``column_describer.generate_column_descriptions`` per table.

    Pulls the column list off Spark's catalog for each FQN and routes the
    LLM output into ``ColumnDescriptionRow`` instances. Failures per table
    are logged and skipped so a single bad table doesn't kill the bootstrap.
    """
    from customer_retention.analysis.auto_explorer.column_describer import (
        fetch_uc_column_comments,
        generate_column_descriptions,
    )

    rows: List[ColumnDescriptionRow] = []
    for fqn in table_fqns:
        try:
            df = spark.table(fqn)
            cols = list(df.columns)
            existing = fetch_uc_column_comments(fqn)
            descs = generate_column_descriptions(
                table_fqn=fqn,
                columns=cols,
                endpoint=llm_endpoint,
                existing_comments=existing,
            )
        except Exception:  # noqa: BLE001 — per-table best-effort
            logger.warning("LLM bootstrap failed for %s", fqn, exc_info=True)
            continue
        catalog, schema, table = _split_fqn(fqn)
        for col, definition in descs.items():
            if not definition:
                continue
            rows.append(
                ColumnDescriptionRow(
                    catalog=catalog,
                    schema=schema,
                    table=table,
                    column_name=col,
                    business_definition=definition,
                    source="llm_proposed",
                )
            )
    return rows


def _split_fqn(fqn: str) -> tuple[Optional[str], Optional[str], str]:
    parts = fqn.split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    return None, None, fqn


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
