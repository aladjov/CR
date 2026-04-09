"""Production batch inference as callable framework code.

Lifts the inline business logic out of ``s10_batch_inference.py``'s generated
cells and into a typed library function that NB12 cell 2, the existing s10
notebook, and any future scheduled job can all invoke without going through
notebook cell strings.

The canonical pattern is the same as ``stages/scd_history/reconstruct.py``:
the framework owns the algorithm and backend dispatch; notebooks just call
``run_batch_inference(config)`` and print the result.

Two execution paths share the same public API:

- **Local** (``is_databricks() = False``): Feast-backed feature retrieval,
  joblib model on disk, parquet/Delta outputs under ``experiments/data/``.
- **Databricks**: Feature Engineering ``fe.score_batch`` against Unity
  Catalog, MLflow ``@production`` alias, ``{catalog}.{schema}.predictions``
  Delta table + ``inference_audit_log`` audit row.

Both paths produce a ``BatchInferenceResult`` with the same shape so the
caller can render a uniform summary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BatchInferenceConfig:
    """Inputs for one batch scoring run.

    Both execution paths read the same fields. Path-specific fields
    (``catalog``/``schema``/``model_name`` for Databricks; ``model_path``/
    ``feature_store_repo`` for local) are populated as appropriate; the
    dispatch in ``run_batch_inference`` ignores the ones that don't apply.
    """

    # Common fields
    threshold: float = 0.5
    risk_tier_high: float = 0.6
    risk_tier_medium: float = 0.3
    inference_timestamp: Optional[datetime] = None  # defaults to now() at run time

    # Databricks path
    catalog: Optional[str] = None
    schema: Optional[str] = None
    model_name: Optional[str] = None
    feature_table: Optional[str] = None  # default: f"{catalog}.{schema}.customer_features"
    target_table: str = "predictions"
    audit_table: str = "inference_audit_log"

    # Local path
    model_path: Optional[Path] = None
    feature_store_repo: Optional[Path] = None
    output_dir: Optional[Path] = None
    dataset_name: Optional[str] = None
    entity_id_columns: List[str] = field(default_factory=list)


@dataclass
class BatchInferenceResult:
    """Output of one batch scoring run."""

    inference_id: str
    inference_timestamp: datetime
    total_scored: int
    predicted_churners: int
    avg_probability: float
    risk_distribution: Dict[str, int]
    model_uri: str
    target_table_fqn: Optional[str] = None  # set on Databricks path
    output_files: List[str] = field(default_factory=list)  # set on local path

    def summary(self) -> str:
        """One-line printable summary suitable for cell output."""
        return (
            f"Scored {self.total_scored:,} customers at "
            f"{self.inference_timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
            f"{self.predicted_churners:,} predicted churners "
            f"({self.avg_probability:.3f} avg prob), risk={self.risk_distribution}"
        )

    def long_summary(self) -> str:
        """Multi-line dashboard-style summary."""
        lines = [
            "=" * 70,
            "BATCH INFERENCE RESULTS",
            "=" * 70,
            "",
            f"Inference Point-in-Time: {self.inference_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Inference ID:            {self.inference_id}",
            f"Model URI:               {self.model_uri}",
            "",
            "Summary:",
            f"  Total customers scored:    {self.total_scored:,}",
            f"  Predicted churners:        {self.predicted_churners:,}",
            f"  Average churn probability: {self.avg_probability:.3f}",
            "",
            "Risk distribution:",
            f"  High:   {self.risk_distribution.get('High', 0):,}",
            f"  Medium: {self.risk_distribution.get('Medium', 0):,}",
            f"  Low:    {self.risk_distribution.get('Low', 0):,}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def run_batch_inference(config: BatchInferenceConfig) -> BatchInferenceResult:
    """Run batch inference against the production churn model.

    Dispatches to the local or Databricks path based on
    ``is_databricks()``. The result shape is identical across both paths.
    """
    from customer_retention.core.compat.detection import is_databricks

    if config.inference_timestamp is None:
        config.inference_timestamp = datetime.now()

    if is_databricks():
        return _run_databricks(config)
    return _run_local(config)


# ---------------------------------------------------------------------------
# Public reusable helpers
# ---------------------------------------------------------------------------


def apply_risk_tiers_pandas(df, prob_col: str, high: float, medium: float):
    """Apply High/Medium/Low risk-tier labels to a pandas DataFrame.

    Tiers come from the configured thresholds rather than hardcoded values
    so historical assignments can be reconstructed from the policy version
    in force at scoring time.
    """
    import pandas as pd

    out = df.copy()
    out["risk_tier"] = pd.cut(
        out[prob_col],
        bins=[0.0, medium, high, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return out


def apply_risk_tiers_spark(df: "DataFrame", prob_col: str, high: float, medium: float) -> "DataFrame":
    """Apply High/Medium/Low risk-tier labels to a Spark DataFrame."""
    from pyspark.sql.functions import col, when

    return df.withColumn(
        "risk_tier",
        when(col(prob_col) >= high, "High")
        .when(col(prob_col) >= medium, "Medium")
        .otherwise("Low"),
    )


# ---------------------------------------------------------------------------
# Databricks execution path
# ---------------------------------------------------------------------------


def _run_databricks(config: BatchInferenceConfig) -> BatchInferenceResult:
    """Score the entire customer table using Databricks Feature Engineering.

    Mirrors the logic from the original ``s10_batch_inference.py`` databricks
    cells: PIT-correct ``fe.score_batch`` against the customer feature table,
    risk-tier bucketing, write to ``{catalog}.{schema}.predictions``, append
    audit row.
    """
    from datetime import datetime as _dt

    from pyspark.sql.functions import col, count, lit, mean
    from pyspark.sql.functions import sum as spark_sum
    from pyspark.sql.types import TimestampType

    spark = _get_spark()

    if not (config.catalog and config.schema and config.model_name):
        raise ValueError(
            "BatchInferenceConfig.catalog, schema, and model_name are required "
            "on the Databricks execution path."
        )

    feature_table = config.feature_table or f"{config.catalog}.{config.schema}.customer_features"
    target_fqn = f"{config.catalog}.{config.schema}.{config.target_table}"
    audit_fqn = f"{config.catalog}.{config.schema}.{config.audit_table}"
    model_uri = f"models:/{config.catalog}.{config.schema}.{config.model_name}@production"
    inference_ts = config.inference_timestamp or _dt.now()

    logger.info("Databricks batch inference: model=%s feature_table=%s", model_uri, feature_table)

    # Build entity_df with PIT timestamp
    df_customers = spark.table(f"{config.catalog}.{config.schema}.gold_customers")
    entity_df = df_customers.select("entity_id").withColumn(
        "inference_timestamp",
        lit(inference_ts).cast(TimestampType()),
    )

    # Score with the feature store
    predictions = _score_with_feature_store(
        spark=spark,
        entity_df=entity_df,
        feature_table=feature_table,
        model_uri=model_uri,
    )

    # Apply threshold + risk tiers + metadata
    df_scored = (
        predictions.withColumn("churn_probability", col("prediction"))
        .withColumn(
            "churn_prediction",
            (col("prediction") >= lit(config.threshold)).cast("integer"),
        )
        .withColumn("inference_point_in_time", lit(inference_ts).cast(TimestampType()))
        .withColumn("model_uri", lit(model_uri))
    )
    df_scored = apply_risk_tiers_spark(
        df_scored, "churn_probability", config.risk_tier_high, config.risk_tier_medium
    )

    # Aggregate summary stats (single Spark job)
    summary_row = df_scored.agg(
        count("*").alias("total"),
        spark_sum("churn_prediction").alias("churners"),
        mean("churn_probability").alias("avg_prob"),
    ).collect()[0]
    total = int(summary_row["total"])
    churners = int(summary_row["churners"] or 0)
    avg_prob = float(summary_row["avg_prob"] or 0.0)

    risk_dist = {row["risk_tier"]: int(row["count"]) for row in df_scored.groupBy("risk_tier").count().collect()}

    # Write predictions
    output_cols = [
        "entity_id",
        "churn_probability",
        "churn_prediction",
        "risk_tier",
        "inference_point_in_time",
        "model_uri",
    ]
    (
        df_scored.select(output_cols)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target_fqn)
    )

    inference_id = f"batch_{inference_ts.strftime('%Y%m%d_%H%M%S')}"

    # Append audit row
    audit_row = spark.createDataFrame(
        [
            {
                "inference_id": inference_id,
                "inference_timestamp": inference_ts,
                "total_customers": total,
                "predicted_churners": churners,
                "avg_probability": avg_prob,
                "model_uri": model_uri,
                "threshold": config.threshold,
                "created_at": _dt.now(),
            }
        ]
    )
    audit_row.write.format("delta").mode("append").saveAsTable(audit_fqn)

    return BatchInferenceResult(
        inference_id=inference_id,
        inference_timestamp=inference_ts,
        total_scored=total,
        predicted_churners=churners,
        avg_probability=avg_prob,
        risk_distribution=risk_dist,
        model_uri=model_uri,
        target_table_fqn=target_fqn,
    )


def _score_with_feature_store(
    spark: "SparkSession",
    entity_df: "DataFrame",
    feature_table: str,
    model_uri: str,
) -> "DataFrame":
    """Try ``fe.score_batch``; fall back to manual feature lookup + sklearn predict.

    The fallback path is preserved verbatim from the original s10 cells so
    the parity diff stays clean: same DataFrame contract in/out.
    """
    from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

    fe = FeatureEngineeringClient()
    feature_lookups = [
        FeatureLookup(
            table_name=feature_table,
            lookup_key=["entity_id"],
            timestamp_lookup_key="inference_timestamp",
        )
    ]
    try:
        return fe.score_batch(df=entity_df, model_uri=model_uri, result_type="double")
    except (AttributeError, NotImplementedError, TypeError) as exc:
        logger.warning("fe.score_batch unavailable (%s); falling back to manual feature retrieval", exc)
    import mlflow

    training_set = fe.create_training_set(df=entity_df, feature_lookups=feature_lookups, label=None)
    inference_df = training_set.load_df()
    model = mlflow.pyfunc.load_model(model_uri)
    pdf = inference_df.toPandas()
    feature_cols = [c for c in pdf.columns if c not in ["entity_id", "inference_timestamp"]]
    pdf["prediction"] = model.predict(pdf[feature_cols])
    return spark.createDataFrame(pdf)


# ---------------------------------------------------------------------------
# Local execution path
# ---------------------------------------------------------------------------


def _run_local(config: BatchInferenceConfig) -> BatchInferenceResult:
    """Score the local pipeline's customer table using Feast + joblib model.

    Mirrors the original local s10 cells. Falls back across multiple input
    locations (gold ``_to_score`` Delta, ``_to_score.parquet``, ``_features``
    Delta, ``_features.parquet``, latest snapshot) to match the existing
    behavior of the cells.
    """
    import joblib

    from customer_retention.integrations.adapters.factory import get_delta
    from customer_retention.integrations.feature_store import (
        FeatureRegistry,
        get_feature_store_manager,
    )

    inference_ts = config.inference_timestamp or datetime.now()
    storage = get_delta()

    # Resolve paths with sensible local defaults if the caller didn't supply them
    base = Path("./experiments")
    model_path = config.model_path or (base / "data" / "models" / "best_model.joblib")
    feature_store_repo = config.feature_store_repo or (base / "feature_store" / "feature_repo")
    output_dir = config.output_dir or (base / "data" / "predictions")
    dataset_name = config.dataset_name or "customer"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    model = joblib.load(model_path)
    logger.info("Loaded local model: %s", type(model).__name__)

    registry_path = base / "feature_store" / "feature_registry.json"
    registry = FeatureRegistry.load(registry_path) if registry_path.exists() else None

    manager = get_feature_store_manager(
        repo_path=str(feature_store_repo),
        output_path=str(base / "data"),
    )

    # Customer fallback chain (order preserved from the original cell)
    df_customers = _load_local_customers(storage, base, dataset_name)

    entity_col_candidates = config.entity_id_columns or ["customer_id"]
    entity_col = entity_col_candidates[0]
    if entity_col not in df_customers.columns and "entity_id" in df_customers.columns:
        entity_col = "entity_id"

    entity_df = df_customers[[entity_col]].copy()
    entity_df = entity_df.rename(columns={entity_col: "entity_id"})
    entity_df["event_timestamp"] = inference_ts

    if registry is not None:
        feature_names = registry.list_features()
    else:
        feat_attr = getattr(model, "feature_names_in_", None)
        if feat_attr is None:
            raise ValueError("Cannot determine feature names. Provide a feature registry.")
        feature_names = list(feat_attr)

    inference_df = manager.get_inference_features(
        entity_df=entity_df,
        registry=registry,
        feature_names=feature_names,
        table_name="customer_features",
        timestamp_column="event_timestamp",
    )

    meta_cols = ["entity_id", "event_timestamp"]
    feature_cols = [c for c in inference_df.columns if c not in meta_cols]
    X = inference_df[feature_cols]
    if X.isnull().any().any():
        X = X.fillna(X.median())

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= config.threshold).astype(int)

    results_df = inference_df[["entity_id"]].copy()
    results_df["churn_probability"] = y_prob
    results_df["churn_prediction"] = y_pred
    results_df = apply_risk_tiers_pandas(
        results_df, "churn_probability", config.risk_tier_high, config.risk_tier_medium
    )
    results_df["inference_timestamp"] = inference_ts
    results_df["model_version"] = str(model_path)

    total_customers = int(len(results_df))
    predicted_churners = int(results_df["churn_prediction"].sum())
    avg_probability = float(results_df["churn_probability"].mean())
    risk_distribution = {
        str(k): int(v) for k, v in results_df["risk_tier"].value_counts().items()
    }

    # Persist predictions: timestamped + latest + JSON metadata
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = inference_ts.strftime("%Y%m%d_%H%M%S")
    timestamped_path = output_dir / f"batch_predictions_{timestamp_str}"
    latest_path = output_dir / "batch_predictions_latest"
    storage.write(results_df, str(timestamped_path))
    storage.write(results_df, str(latest_path))

    metadata_file = output_dir / f"inference_metadata_{timestamp_str}.json"
    metadata_file.write_text(
        json.dumps(
            {
                "inference_timestamp": inference_ts.isoformat(),
                "total_customers": total_customers,
                "predicted_churners": predicted_churners,
                "churn_rate_pct": predicted_churners / max(total_customers, 1) * 100,
                "avg_probability": avg_probability,
                "model_path": str(model_path),
                "threshold": config.threshold,
                "risk_distribution": risk_distribution,
            },
            indent=2,
        )
    )

    return BatchInferenceResult(
        inference_id=f"batch_{timestamp_str}",
        inference_timestamp=inference_ts,
        total_scored=total_customers,
        predicted_churners=predicted_churners,
        avg_probability=avg_probability,
        risk_distribution=risk_distribution,
        model_uri=str(model_path),
        output_files=[str(timestamped_path), str(latest_path), str(metadata_file)],
    )


def _load_local_customers(storage, base: Path, dataset_name: str):
    """Reproduce the original s10 cell's customer-data fallback chain."""
    import pandas as pd

    from customer_retention.stages.temporal import SnapshotManager

    candidates = [
        ("delta", base / "data" / "gold" / f"{dataset_name}_to_score"),
        ("parquet", base / "data" / "gold" / f"{dataset_name}_to_score.parquet"),
        ("delta", base / "data" / "gold" / f"{dataset_name}_features"),
        ("parquet", base / "data" / "gold" / f"{dataset_name}_features.parquet"),
    ]
    for kind, path in candidates:
        if kind == "delta" and path.is_dir() and (path / "_delta_log").is_dir():
            return storage.read(str(path))
        if kind == "parquet" and path.exists():
            return pd.read_parquet(path)

    snapshot_manager = SnapshotManager(base / "data")
    latest = snapshot_manager.get_latest_snapshot()
    if latest is None:
        raise FileNotFoundError("No customer data found")
    df, _ = snapshot_manager.load_snapshot(latest)
    return df


def _get_spark() -> "SparkSession":
    from customer_retention.core.compat.detection import get_spark_session

    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("No active Spark session — Databricks scoring path requires Spark")
    return spark
