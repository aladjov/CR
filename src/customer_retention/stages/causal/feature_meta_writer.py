"""Per-feature ``feature_meta`` writer.

Writes one row per gold feature so downstream interpretation layers
(LLM namer, predicate prose renderer, dashboard) can resolve a feature
to its business phrase instead of guessing from the raw column token.

Grain: one row per ``(run_id, composite_name, feature_name)``. Re-running
the gold stage upserts via MERGE — matches ``run_context_writer`` so the
operator mental model is uniform across metadata tables. Any descriptive
field may be NULL when upstream lineage is incomplete; consumers fall
back to the bare feature name in that case.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional

from customer_retention.stages.causal.interpretation.business_phrase import (
    render_business_phrase,
    render_window_phrase,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession


_DEFAULT_TARGET_TABLE = "feature_meta"


@dataclass
class FeatureMetaRow:
    """One gold-feature lineage row.

    Only ``composite_name`` and ``feature_name`` are required; remaining
    fields are populated from pipeline-generator lineage when available.
    """

    composite_name: str
    feature_name: str
    source_columns: Optional[List[str]] = None
    source_table: Optional[str] = None
    aggregation_kind: Optional[str] = None
    window_days: Optional[int] = None
    window_phrase: Optional[str] = None
    target_dependency: Optional[bool] = None
    mask_future: Optional[bool] = None
    polarity: Optional[str] = None
    business_phrase: Optional[str] = None

    def with_rendered_phrases(self, source_business_name: Optional[str] = None) -> "FeatureMetaRow":
        """Fill ``window_phrase`` and ``business_phrase`` deterministically.

        Idempotent: existing non-empty values are preserved. Falls back to
        the feature's own name when no source business name is given.
        """
        window_phrase = self.window_phrase or render_window_phrase(self.window_days)
        label = (source_business_name or self.feature_name).strip()
        business_phrase = self.business_phrase or render_business_phrase(
            self.aggregation_kind, label, window_phrase
        )
        return replace(self, window_phrase=window_phrase, business_phrase=business_phrase)


@dataclass
class FeatureMetaConfig:
    """Batch write handle: one ``run_id`` + N feature rows."""

    spark: "SparkSession"
    table_fqn: str
    run_id: str
    rows: List[FeatureMetaRow] = field(default_factory=list)


def write_feature_meta(config: FeatureMetaConfig) -> int:
    """Upsert ``config.rows`` into ``{catalog}.{schema}.feature_meta``.

    Composite key ``(run_id, composite_name, feature_name)``. Empty
    ``rows`` is a no-op returning 0.
    """
    if not config.rows:
        logger.info("write_feature_meta: empty rows, skipping %s", config.table_fqn)
        return 0

    from delta.tables import DeltaTable

    from customer_retention.stages.causal.schemas import get_schema

    schema = get_schema(_DEFAULT_TARGET_TABLE)
    config.spark.sql(
        f"CREATE TABLE IF NOT EXISTS {config.table_fqn} "
        f"({_schema_to_ddl(schema)}) USING DELTA"
    )

    written_at = datetime.now(timezone.utc)
    records = [_row_to_record(config.run_id, row, written_at) for row in config.rows]
    source = config.spark.createDataFrame(records, schema=schema)

    (
        DeltaTable.forName(config.spark, config.table_fqn)
        .alias("t")
        .merge(
            source.alias("s"),
            (
                "t.run_id = s.run_id "
                "AND t.composite_name = s.composite_name "
                "AND t.feature_name = s.feature_name"
            ),
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info(
        "wrote %d feature_meta rows run_id=%s table=%s",
        len(records), config.run_id, config.table_fqn,
    )
    return len(records)


def _row_to_record(run_id: str, row: FeatureMetaRow, written_at: datetime) -> dict:
    return {
        "run_id": run_id,
        "composite_name": row.composite_name,
        "feature_name": row.feature_name,
        "source_columns": list(row.source_columns) if row.source_columns is not None else None,
        "source_table": row.source_table,
        "aggregation_kind": row.aggregation_kind,
        "window_days": row.window_days,
        "window_phrase": row.window_phrase,
        "target_dependency": row.target_dependency,
        "mask_future": row.mask_future,
        "polarity": row.polarity,
        "business_phrase": row.business_phrase,
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
