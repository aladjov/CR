"""Per-run, per-feature population statistics for the training split.

Computed once at training time (never on test — leakage). Feeds the
quantile-phrasing layer so the dashboard / LLM prompt can lift raw values
into "elevated" / "typical" / "very low" instead of quoting cutoffs.

Two public entry points:

- ``compute_feature_population_stats(spark_df, numeric_features, categorical_features)``
  returns a list of ``PopulationStatsRow``. Uses batched ``.agg()`` per
  ``docs/Coding_Practices.md`` — 100 numeric columns per Spark job to
  keep Catalyst plans O(10K), far below the O(N²) cliff at 1K+ cols.
- ``write_population_stats(config)`` upserts the rows into Delta, MERGE
  keyed on ``(run_id, feature_name)``. Empty rows is a no-op.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame as SparkDF
    from pyspark.sql import SparkSession


_DEFAULT_TARGET_TABLE = "feature_population_stats"
_NUMERIC_BATCH_SIZE = 100
_TOP_CATEGORIES = 10
_QUANTILE_POINTS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
_QUANTILE_COLUMNS = ("q01", "q05", "q25", "q50", "q75", "q95", "q99")


@dataclass
class TopCategory:
    """One entry inside ``PopulationStatsRow.top_categories``."""
    value: Optional[str]
    count: int
    share: float


@dataclass
class PopulationStatsRow:
    """One feature's population summary."""

    run_id: str
    feature_name: str
    dtype: str
    count_nonnull: Optional[int] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    q01: Optional[float] = None
    q05: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    q95: Optional[float] = None
    q99: Optional[float] = None
    top_categories: Optional[List[TopCategory]] = None


@dataclass
class PopulationStatsConfig:
    """Batch write handle for ``write_population_stats``."""

    spark: "SparkSession"
    table_fqn: str
    run_id: str
    rows: List[PopulationStatsRow] = field(default_factory=list)


def compute_feature_population_stats(
    train_df: Any,
    *,
    run_id: str,
    numeric_features: Sequence[str] = (),
    categorical_features: Sequence[str] = (),
    approx_relative_error: float = 0.01,
) -> List[PopulationStatsRow]:
    """Compute population summaries for numeric and categorical features.

    Accepts a native Spark DataFrame or a ``pyspark.pandas`` DataFrame —
    pyspark.pandas is unwrapped via ``as_spark_df`` so batched ``.agg()``
    operates on the native Spark engine (Catalyst plan O(100²) per batch).
    Numeric: 7 quantiles via ``percentile_approx`` in 100-col batches.
    Categorical: ``groupBy(col).count()`` per column — one Spark job per
    categorical, acceptable since categorical features are typically few.
    """
    spark_df = _as_native_spark(train_df)
    rows: List[PopulationStatsRow] = []
    if numeric_features:
        rows.extend(_numeric_stats(spark_df, run_id, numeric_features, approx_relative_error))
    if categorical_features:
        rows.extend(_categorical_stats(spark_df, run_id, categorical_features))
    return rows


def _as_native_spark(df: Any) -> Any:
    """Return ``df`` as a native Spark DataFrame, unwrapping pyspark.pandas.

    Only ``pyspark.pandas.DataFrame`` exposes ``.to_spark()``; a native
    ``pyspark.sql.DataFrame`` is already Spark and anything else is
    passed through (unit-test mocks, injected stubs).
    """
    if type(df).__module__.startswith("pyspark.pandas"):
        return df.to_spark()
    return df


def _numeric_stats(
    train_df: "SparkDF",
    run_id: str,
    features: Sequence[str],
    approx_relative_error: float,
) -> List[PopulationStatsRow]:
    from pyspark.sql import functions as F  # noqa: N812

    out: List[PopulationStatsRow] = []
    for start in range(0, len(features), _NUMERIC_BATCH_SIZE):
        batch = features[start : start + _NUMERIC_BATCH_SIZE]
        exprs: List[Any] = []
        for i, col in enumerate(batch):
            cast = F.col(col).cast("double")
            exprs.extend([
                F.count(cast).alias(f"cnt_{i}"),
                F.avg(cast).alias(f"avg_{i}"),
                F.stddev_samp(cast).alias(f"std_{i}"),
                F.percentile_approx(cast, list(_QUANTILE_POINTS), 10000).alias(f"pct_{i}"),
            ])
        row = train_df.agg(*exprs).head()
        for i, col in enumerate(batch):
            out.append(_row_from_numeric_agg(row, i, col, run_id, approx_relative_error))
    return out


def _row_from_numeric_agg(
    row: Any, idx: int, feature_name: str, run_id: str, _eps: float,
) -> PopulationStatsRow:
    quantiles = row[f"pct_{idx}"] or [None] * len(_QUANTILE_COLUMNS)
    quantile_kwargs = {
        name: _to_optional_float(q) for name, q in zip(_QUANTILE_COLUMNS, quantiles)
    }
    return PopulationStatsRow(
        run_id=run_id,
        feature_name=feature_name,
        dtype="numeric",
        count_nonnull=_to_optional_int(row[f"cnt_{idx}"]),
        mean=_to_optional_float(row[f"avg_{idx}"]),
        stddev=_to_optional_float(row[f"std_{idx}"]),
        **quantile_kwargs,
    )


def _categorical_stats(
    train_df: "SparkDF",
    run_id: str,
    features: Sequence[str],
) -> List[PopulationStatsRow]:
    from pyspark.sql import functions as F  # noqa: N812

    out: List[PopulationStatsRow] = []
    for col in features:
        non_null = train_df.filter(F.col(col).isNotNull())
        non_null_count = non_null.count()
        counts_df = (
            non_null.groupBy(F.col(col).cast("string").alias("value"))
            .agg(F.count(F.lit(1)).alias("count"))
            .orderBy(F.col("count").desc())
            .limit(_TOP_CATEGORIES)
        )
        rows = counts_df.collect()
        denominator = float(non_null_count) if non_null_count else 0.0
        top = [
            TopCategory(
                value=r["value"],
                count=int(r["count"]),
                share=(float(r["count"]) / denominator) if denominator else 0.0,
            )
            for r in rows
        ]
        out.append(
            PopulationStatsRow(
                run_id=run_id,
                feature_name=col,
                dtype="categorical",
                count_nonnull=non_null_count,
                top_categories=top or None,
            )
        )
    return out


def write_population_stats(config: PopulationStatsConfig) -> int:
    """Upsert ``config.rows`` into the population-stats Delta table."""
    if not config.rows:
        logger.info("write_population_stats: empty rows, skipping %s", config.table_fqn)
        return 0

    from delta.tables import DeltaTable

    from customer_retention.stages.causal.schemas import get_schema

    schema = get_schema(_DEFAULT_TARGET_TABLE)
    config.spark.sql(
        f"CREATE TABLE IF NOT EXISTS {config.table_fqn} "
        f"({_schema_to_ddl(schema)}) USING DELTA"
    )

    computed_at = datetime.now(timezone.utc)
    records = [_row_to_record(row, computed_at) for row in config.rows]
    source = config.spark.createDataFrame(records, schema=schema)

    (
        DeltaTable.forName(config.spark, config.table_fqn)
        .alias("t")
        .merge(
            source.alias("s"),
            "t.run_id = s.run_id AND t.feature_name = s.feature_name",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info(
        "wrote %d feature_population_stats rows run_id=%s table=%s",
        len(records), config.run_id, config.table_fqn,
    )
    return len(records)


def materialize_population_stats_from_sidecar(
    spark: "SparkSession",
    sidecar_path: Any,
    table_fqn: str,
) -> int:
    """Read a population-stats JSON sidecar and upsert its rows into a UC
    Delta table.

    The framework writes population stats to a JSON sidecar under the run
    namespace (``<run_dir>/feature_population_stats/feature_population_stats.json``).
    The dashboard's deviation views read from a UC Delta table, so this
    helper bridges the two when the operator wants the per-account
    deviation panel.

    Idempotent — uses the same MERGE-by-``(run_id, feature_name)`` key as
    :func:`write_population_stats`, so re-running with the same sidecar
    is a no-op. Returns the number of rows materialized; ``0`` when the
    sidecar is empty or missing.
    """
    import json as _json
    from pathlib import Path as _Path

    from delta.tables import DeltaTable

    from customer_retention.stages.causal.schemas import get_schema

    path = _Path(str(sidecar_path))
    if not path.exists():
        logger.info("sidecar %s does not exist; skipping materialization", path)
        return 0
    payload = _json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows") or []
    if not raw_rows:
        logger.info("sidecar %s contained no rows; skipping materialization", path)
        return 0

    schema = get_schema(_DEFAULT_TARGET_TABLE)
    field_names = [f.name for f in schema.fields]
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table_fqn} "
        f"({_schema_to_ddl(schema)}) USING DELTA"
    )

    fallback_now = datetime.now(timezone.utc)
    records: List[dict] = []
    for raw in raw_rows:
        rec = {name: raw.get(name) for name in field_names}
        # ``json.dumps(..., default=str)`` serialises datetime as
        # ``"2026-04-28 10:30:00+00:00"`` -- coerce back so Spark's
        # TimestampType column accepts it.
        if isinstance(rec.get("computed_at"), str):
            rec["computed_at"] = _parse_iso_datetime(rec["computed_at"])
        if rec.get("computed_at") is None:
            rec["computed_at"] = fallback_now
        # ``top_categories`` rides through unchanged; createDataFrame
        # accepts a list of dicts and matches them positionally to the
        # nested struct schema.
        records.append(rec)

    source = spark.createDataFrame(records, schema=schema)
    (
        DeltaTable.forName(spark, table_fqn)
        .alias("t")
        .merge(
            source.alias("s"),
            "t.run_id = s.run_id AND t.feature_name = s.feature_name",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info(
        "materialized %d feature_population_stats rows from %s into %s",
        len(records), path, table_fqn,
    )
    return len(records)


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    """Parse a Python ``str(datetime)`` or ISO-8601 timestamp."""
    try:
        # ``str(datetime)`` uses a space separator; ``fromisoformat``
        # accepts ``T`` and (since 3.11) space.
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(value.replace(" ", "T"))
        except (TypeError, ValueError, AttributeError):
            return None


def _row_to_record(row: PopulationStatsRow, computed_at: datetime) -> dict:
    return {
        "run_id": row.run_id,
        "feature_name": row.feature_name,
        "dtype": row.dtype,
        "count_nonnull": row.count_nonnull,
        "mean": row.mean,
        "stddev": row.stddev,
        "q01": row.q01,
        "q05": row.q05,
        "q25": row.q25,
        "q50": row.q50,
        "q75": row.q75,
        "q95": row.q95,
        "q99": row.q99,
        "top_categories": (
            [{"value": c.value, "count": c.count, "share": c.share} for c in row.top_categories]
            if row.top_categories is not None else None
        ),
        "computed_at": computed_at,
    }


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


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
