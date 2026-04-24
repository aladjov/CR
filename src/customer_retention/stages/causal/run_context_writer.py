"""Per-scoring-run ``run_context`` writer.

A tiny sidecar table that projects the pieces of ``ProjectContext`` that an
operator-facing UI needs to render a masthead — horizon, model type, primary
objective, temporal posture — and pairs them with the anchor tuple already
frozen on ``eligibility_snapshot`` (``scoring_run_id``, ``as_of_date``,
``model_name``, ``model_version``).

Grain: one row per ``scoring_run_id``. The view ``v_run_context`` selects the
row with the most recent ``as_of_date``, so re-publishing a run is an idempotent
MERGE and the app always sees the latest context without needing to know about
history.

The writer tolerates a missing ``ProjectContext`` — in that case the context
columns are written NULL and the app degrades gracefully (title drops the
horizon segment).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession

    from customer_retention.analysis.auto_explorer.project_context import ProjectContext


_DEFAULT_TARGET_TABLE = "run_context"


@dataclass
class RunContextConfig:
    """All inputs needed to materialize one ``run_context`` row."""

    spark: "SparkSession"
    table_fqn: str
    scoring_run_id: str
    as_of_date: datetime
    model_name: str
    model_version: str
    model_type: Optional[str] = None
    horizon_days: Optional[int] = None
    prediction_horizons: Optional[List[int]] = None
    primary_objective: Optional[str] = None
    temporal_posture: Optional[str] = None
    project_name: Optional[str] = None
    project_run_id: Optional[str] = None
    target_dataset: Optional[str] = None
    entity_column: Optional[str] = None
    dataset_names: Optional[List[str]] = None


def from_project_context(
    *,
    project_context: Optional["ProjectContext"],
    spark: "SparkSession",
    table_fqn: str,
    scoring_run_id: str,
    as_of_date: datetime,
    model_name: str,
    model_version: str,
    model_type: Optional[str] = None,
) -> RunContextConfig:
    """Build a :class:`RunContextConfig` from an optional ``ProjectContext``.

    When ``project_context`` is None (Databricks run without a shipped YAML) we
    still emit a row — only ``model_name`` / ``model_version`` / ``as_of_date``
    are required. The remaining fields stay None and the view returns NULLs.
    """
    horizon_days: Optional[int] = None
    prediction_horizons: Optional[List[int]] = None
    primary_objective: Optional[str] = None
    temporal_posture: Optional[str] = None
    project_name: Optional[str] = None
    project_run_id: Optional[str] = None
    target_dataset: Optional[str] = None
    entity_column: Optional[str] = None
    dataset_names: Optional[List[str]] = None

    if project_context is not None:
        if project_context.intent and project_context.intent.prediction_horizons:
            prediction_horizons = list(project_context.intent.prediction_horizons)
            horizon_days = prediction_horizons[0]
        primary_objective = _enum_value(project_context.primary_objective)
        temporal_posture = _enum_value(project_context.temporal_posture)
        project_name = project_context.project_name
        project_run_id = project_context.run_id
        target_dataset = project_context.target_dataset
        entity_column = project_context.entity_column
        if project_context.datasets:
            dataset_names = sorted(project_context.datasets.keys())

    return RunContextConfig(
        spark=spark,
        table_fqn=table_fqn,
        scoring_run_id=scoring_run_id,
        as_of_date=as_of_date,
        model_name=model_name,
        model_version=model_version,
        model_type=model_type,
        horizon_days=horizon_days,
        prediction_horizons=prediction_horizons,
        primary_objective=primary_objective,
        temporal_posture=temporal_posture,
        project_name=project_name,
        project_run_id=project_run_id,
        target_dataset=target_dataset,
        entity_column=entity_column,
        dataset_names=dataset_names,
    )


def ensure_run_context_table(spark: "SparkSession", table_fqn: str) -> None:
    """Idempotent ``CREATE TABLE IF NOT EXISTS`` for the ``run_context`` Delta table.

    Safe to call from both the writer (before a MERGE) and the view publisher
    (before ``CREATE OR REPLACE VIEW v_run_context``, which Spark validates at
    creation time against the referenced table).
    """
    from .schemas import get_schema

    schema = get_schema(_DEFAULT_TARGET_TABLE)
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table_fqn} "
        f"({_schema_to_ddl(schema)}) USING DELTA"
    )


def write_run_context(config: RunContextConfig) -> None:
    """Upsert one row into ``{catalog}.{schema}.run_context`` keyed on ``scoring_run_id``.

    Creates the table on first call using the schema in :mod:`.schemas`. On
    subsequent calls with the same ``scoring_run_id`` the row is overwritten —
    re-publishing a scoring run refreshes (rather than appends to) the context
    projection.
    """
    from delta.tables import DeltaTable  # lazy — keep Spark-optional imports out of module load

    from .schemas import get_schema

    spark = config.spark
    schema = get_schema(_DEFAULT_TARGET_TABLE)

    ensure_run_context_table(spark, config.table_fqn)

    row = {
        "scoring_run_id": config.scoring_run_id,
        "as_of_date": config.as_of_date,
        "model_name": config.model_name,
        "model_version": config.model_version,
        "model_type": config.model_type,
        "horizon_days": config.horizon_days,
        "prediction_horizons": config.prediction_horizons,
        "primary_objective": config.primary_objective,
        "temporal_posture": config.temporal_posture,
        "project_name": config.project_name,
        "project_run_id": config.project_run_id,
        "target_dataset": config.target_dataset,
        "entity_column": config.entity_column,
        "dataset_names": config.dataset_names,
        "written_at": datetime.now(timezone.utc),
    }
    source = spark.createDataFrame([row], schema=schema)

    target = DeltaTable.forName(spark, config.table_fqn)
    (
        target.alias("t")
        .merge(
            source.alias("s"),
            "t.scoring_run_id = s.scoring_run_id",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info(
        "wrote run_context row scoring_run_id=%s model=%s/%s horizon=%s",
        config.scoring_run_id,
        config.model_name,
        config.model_version,
        config.horizon_days,
    )


def _enum_value(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    value = getattr(obj, "value", None)
    return value if value is not None else str(obj)


_TYPE_DDL_MAP = {
    "StringType": "STRING",
    "IntegerType": "INT",
    "LongType": "BIGINT",
    "DoubleType": "DOUBLE",
    "BooleanType": "BOOLEAN",
    "TimestampType": "TIMESTAMP",
}


def _schema_to_ddl(schema: Any) -> str:
    """Render a StructType as an inline DDL list for CREATE TABLE IF NOT EXISTS."""
    parts: List[str] = []
    for field in schema.fields:
        parts.append(f"{field.name} {_field_type_ddl(field.dataType)}")
    return ", ".join(parts)


def _field_type_ddl(dtype: Any) -> str:
    name = type(dtype).__name__
    if name in _TYPE_DDL_MAP:
        return _TYPE_DDL_MAP[name]
    if name == "ArrayType":
        return f"ARRAY<{_field_type_ddl(dtype.elementType)}>"
    # Fall back to Spark's simpleString — handles MapType / StructType if added later.
    return dtype.simpleString().upper()
