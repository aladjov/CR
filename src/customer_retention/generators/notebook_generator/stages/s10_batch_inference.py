"""Batch inference stage for notebook generation.

This stage generates notebooks that perform batch scoring using the feature store
with point-in-time correctness for feature retrieval.
"""

from typing import List
import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class BatchInferenceStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.BATCH_INFERENCE

    @property
    def title(self) -> str:
        return "10 - Batch Inference with Point-in-Time Features"

    @property
    def description(self) -> str:
        return """Score customers in batch using the production model with point-in-time correct feature retrieval.

**Key Concepts:**
- **Point-in-Time (PIT) Correctness**: Features are retrieved as they existed at inference time
- **Inference Timestamp**: The moment when predictions are made, ensuring no future data leakage
- **Feature Store Integration**: Uses Feast (local) or Databricks Feature Store for consistent feature retrieval

This notebook:
1. Sets the inference timestamp (point-in-time for prediction)
2. Retrieves features from the feature store with PIT correctness
3. Scores customers using the production model
4. Generates a dashboard showing predictions with the inference timestamp
"""

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        """Generate cells for local Feast-based batch inference."""
        threshold = self.config.threshold
        return self.header_cells() + [
            self.cb.section("1. Setup and Imports"),
            self.cb.code('''import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from customer_retention.integrations.feature_store import FeatureStoreManager, FeatureRegistry
from customer_retention.stages.temporal import SnapshotManager

print("Batch inference imports loaded")'''),

            self.cb.section("2. Set Inference Point-in-Time"),
            self.cb.markdown('''**Critical**: The inference timestamp determines what point in time we use to retrieve features.
This ensures we only use data that was available at the time of prediction (no future leakage).

- For **real-time inference**: Use `datetime.now()`
- For **historical backtesting**: Use a specific past timestamp
- For **scheduled batch jobs**: Use the job execution timestamp'''),
            self.cb.code('''# INFERENCE_TIMESTAMP: The point-in-time for this batch prediction
# This is the "as-of" time for feature retrieval
INFERENCE_TIMESTAMP = datetime.now()

# Alternative: Use a specific historical timestamp for backtesting
# INFERENCE_TIMESTAMP = datetime(2024, 1, 15, 0, 0, 0)

print(f"=" * 70)
print(f"INFERENCE POINT-IN-TIME: {INFERENCE_TIMESTAMP}")
print(f"=" * 70)
print(f"All features will be retrieved as they existed at this timestamp.")
print(f"This ensures no future data leakage in predictions.")'''),

            self.cb.section("3. Load Production Model"),
            self.cb.code('''# Load the production model
model_path = Path("./experiments/data/models/best_model.joblib")
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

model = joblib.load(model_path)
print(f"Model loaded: {type(model).__name__}")

# Load feature registry to know which features to retrieve
registry_path = Path("./experiments/feature_store/feature_registry.json")
if registry_path.exists():
    registry = FeatureRegistry.load(registry_path)
    print(f"Feature registry loaded: {len(registry)} features")
else:
    print("Warning: No feature registry found. Using model feature names.")
    registry = None'''),

            self.cb.section("4. Initialize Feature Store Manager"),
            self.cb.code('''# Create feature store manager (Feast backend for local)
manager = FeatureStoreManager.create(
    backend="feast",
    repo_path="./experiments/feature_store/feature_repo",
    output_path="./experiments/data",
)

print(f"Feature store initialized")
print(f"Available tables: {manager.list_tables()}")'''),

            self.cb.section("5. Load Customers to Score"),
            self.cb.code(f'''# Load the entities (customers) to score
# These are customers we want to make predictions for

# Option 1: Load from a specific file
customers_path = Path("./experiments/data/gold/customers_to_score.parquet")
if customers_path.exists():
    df_customers = pd.read_parquet(customers_path)
else:
    # Option 2: Use the gold layer customers
    gold_path = Path("./experiments/data/gold/customers_features.parquet")
    if gold_path.exists():
        df_customers = pd.read_parquet(gold_path)
    else:
        # Option 3: Fall back to latest snapshot
        snapshot_manager = SnapshotManager(Path("./experiments/data"))
        latest = snapshot_manager.get_latest_snapshot()
        if latest:
            df_customers, _ = snapshot_manager.load_snapshot(latest)
        else:
            raise FileNotFoundError("No customer data found")

# Ensure entity_id column exists
id_cols = {self.get_identifier_columns()}
entity_col = id_cols[0] if id_cols else "customer_id"
if entity_col not in df_customers.columns and "entity_id" in df_customers.columns:
    entity_col = "entity_id"

print(f"Loaded {{len(df_customers):,}} customers to score")
print(f"Entity column: {{entity_col}}")'''),

            self.cb.section("6. Retrieve Features with Point-in-Time Correctness"),
            self.cb.markdown('''The feature store retrieves features as they existed at the **inference timestamp**.
This is crucial for:
- **Training-serving consistency**: Same features used in training and inference
- **No future leakage**: Only data available at prediction time is used
- **Reproducibility**: Same timestamp always gives same features'''),
            self.cb.code('''# Create entity DataFrame with inference timestamp
# All customers get the same inference timestamp for this batch
entity_df = df_customers[[entity_col]].copy()
entity_df = entity_df.rename(columns={entity_col: "entity_id"})
entity_df["event_timestamp"] = INFERENCE_TIMESTAMP

print(f"Retrieving features for {len(entity_df):,} entities")
print(f"Point-in-Time: {INFERENCE_TIMESTAMP}")

# Get features from feature store with PIT correctness
if registry:
    feature_names = registry.list_features()
else:
    # Fall back to model feature names if available
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None:
        raise ValueError("Cannot determine feature names. Please provide a feature registry.")
    feature_names = list(feature_names)

# Retrieve point-in-time correct features
inference_df = manager.get_inference_features(
    entity_df=entity_df,
    registry=registry,
    feature_names=feature_names,
    table_name="customer_features",
    timestamp_column="event_timestamp",
)

print(f"Retrieved {len(inference_df.columns)} features for {len(inference_df):,} customers")
print(f"Feature retrieval timestamp: {INFERENCE_TIMESTAMP}")'''),

            self.cb.section("7. Generate Predictions"),
            self.cb.code(f'''# Prepare features for prediction
# Remove non-feature columns
meta_cols = ["entity_id", "event_timestamp"]
feature_cols = [c for c in inference_df.columns if c not in meta_cols]

X = inference_df[feature_cols]

# Handle any missing values from feature retrieval
missing_pct = X.isnull().sum().sum() / (len(X) * len(X.columns)) * 100
if missing_pct > 0:
    print(f"Warning: {{missing_pct:.2f}}% missing values in features")
    X = X.fillna(X.median())

# Generate predictions
threshold = {threshold}
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# Add predictions to results
results_df = inference_df[["entity_id"]].copy()
results_df["churn_probability"] = y_prob
results_df["churn_prediction"] = y_pred
results_df["risk_tier"] = pd.cut(
    y_prob,
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"]
)

# Add inference metadata
results_df["inference_timestamp"] = INFERENCE_TIMESTAMP
results_df["model_version"] = str(model_path)

print(f"Predictions generated for {{len(results_df):,}} customers")
print(f"Threshold: {{threshold}}")'''),

            self.cb.section("8. Prediction Summary Dashboard"),
            self.cb.markdown('''This dashboard shows the batch scoring results with the **point-in-time** used for inference prominently displayed.'''),
            self.cb.code('''# Create summary statistics
total_customers = len(results_df)
predicted_churners = results_df["churn_prediction"].sum()
churn_rate = predicted_churners / total_customers * 100
avg_probability = results_df["churn_probability"].mean()

risk_distribution = results_df["risk_tier"].value_counts()

print("=" * 70)
print("BATCH INFERENCE RESULTS DASHBOARD")
print("=" * 70)
print(f"")
print(f"📅 INFERENCE POINT-IN-TIME: {INFERENCE_TIMESTAMP.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"")
print(f"📊 SUMMARY STATISTICS:")
print(f"   Total Customers Scored: {total_customers:,}")
print(f"   Predicted Churners: {predicted_churners:,} ({churn_rate:.1f}%)")
print(f"   Average Churn Probability: {avg_probability:.3f}")
print(f"")
print(f"🎯 RISK DISTRIBUTION:")
for tier in ["High", "Medium", "Low"]:
    count = risk_distribution.get(tier, 0)
    pct = count / total_customers * 100
    print(f"   {tier}: {count:,} ({pct:.1f}%)")
print(f"")
print("=" * 70)'''),

            self.cb.section("9. Interactive Results Dashboard"),
            self.cb.code('''# Create interactive dashboard with Plotly
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f"Risk Distribution (PIT: {INFERENCE_TIMESTAMP.strftime('%Y-%m-%d %H:%M')})",
        "Churn Probability Distribution",
        "Risk by Probability Range",
        "Inference Metadata"
    ],
    specs=[
        [{"type": "pie"}, {"type": "histogram"}],
        [{"type": "bar"}, {"type": "table"}]
    ]
)

# Risk tier pie chart
colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
fig.add_trace(
    go.Pie(
        labels=risk_distribution.index.tolist(),
        values=risk_distribution.values.tolist(),
        marker_colors=[colors.get(tier, "#95a5a6") for tier in risk_distribution.index],
        textinfo="label+percent",
        hole=0.4
    ),
    row=1, col=1
)

# Probability histogram
fig.add_trace(
    go.Histogram(
        x=results_df["churn_probability"],
        nbinsx=50,
        marker_color="#3498db",
        name="Probability"
    ),
    row=1, col=2
)

# Risk by probability range bar chart
prob_bins = pd.cut(results_df["churn_probability"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
prob_counts = prob_bins.value_counts().sort_index()
fig.add_trace(
    go.Bar(
        x=[str(b) for b in prob_counts.index],
        y=prob_counts.values,
        marker_color="#9b59b6",
        name="Count"
    ),
    row=2, col=1
)

# Metadata table
metadata = [
    ["Metric", "Value"],
    ["Inference Point-in-Time", INFERENCE_TIMESTAMP.strftime('%Y-%m-%d %H:%M:%S')],
    ["Total Customers", f"{total_customers:,}"],
    ["Predicted Churners", f"{predicted_churners:,}"],
    ["Churn Rate", f"{churn_rate:.1f}%"],
    ["Model", type(model).__name__],
    ["Threshold", f"{threshold}"],
]
fig.add_trace(
    go.Table(
        header=dict(values=["Metric", "Value"], fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=[[row[0] for row in metadata[1:]], [row[1] for row in metadata[1:]]],
                   fill_color="#ecf0f1")
    ),
    row=2, col=2
)

fig.update_layout(
    title=dict(
        text=f"<b>Batch Inference Dashboard</b><br><sub>Point-in-Time: {INFERENCE_TIMESTAMP.strftime('%Y-%m-%d %H:%M:%S UTC')}</sub>",
        font=dict(size=20)
    ),
    height=700,
    showlegend=False,
    template="plotly_white"
)

fig.show()'''),

            self.cb.section("10. Save Predictions with Metadata"),
            self.cb.code('''# Save predictions with full metadata
output_dir = Path("./experiments/data/predictions")
output_dir.mkdir(parents=True, exist_ok=True)

# Create timestamped filename for audit trail
timestamp_str = INFERENCE_TIMESTAMP.strftime('%Y%m%d_%H%M%S')
output_file = output_dir / f"batch_predictions_{timestamp_str}.parquet"

# Save with all metadata
results_df.to_parquet(output_file, index=False)

# Also save a "latest" version for downstream consumption
latest_file = output_dir / "batch_predictions_latest.parquet"
results_df.to_parquet(latest_file, index=False)

print(f"✅ Predictions saved:")
print(f"   Timestamped: {output_file}")
print(f"   Latest: {latest_file}")
print(f"")
print(f"📅 Inference Point-in-Time: {INFERENCE_TIMESTAMP}")
print(f"📊 Records: {len(results_df):,}")

# Save inference metadata as JSON for audit
import json
metadata_file = output_dir / f"inference_metadata_{timestamp_str}.json"
metadata = {
    "inference_timestamp": INFERENCE_TIMESTAMP.isoformat(),
    "total_customers": int(total_customers),
    "predicted_churners": int(predicted_churners),
    "churn_rate_pct": float(churn_rate),
    "avg_probability": float(avg_probability),
    "model_path": str(model_path),
    "threshold": threshold,
    "risk_distribution": {str(k): int(v) for k, v in risk_distribution.items()},
}
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"   Metadata: {metadata_file}")'''),

            self.cb.section("11. Summary"),
            self.cb.code('''print("=" * 70)
print("BATCH INFERENCE COMPLETE")
print("=" * 70)
print(f"")
print(f"🕐 Point-in-Time Used: {INFERENCE_TIMESTAMP}")
print(f"📊 Customers Scored: {total_customers:,}")
print(f"⚠️  High Risk: {risk_distribution.get('High', 0):,}")
print(f"🟡 Medium Risk: {risk_distribution.get('Medium', 0):,}")
print(f"✅ Low Risk: {risk_distribution.get('Low', 0):,}")
print(f"")
print("Next steps:")
print("1. Review high-risk customers for intervention")
print("2. Schedule next batch inference run")
print("3. Monitor model performance over time")'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        """Generate cells for Databricks Feature Store batch inference."""
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        model_name = self.config.mlflow.model_name
        target = self.get_target_column()
        threshold = self.config.threshold

        return self.header_cells() + [
            self.cb.section("1. Setup and Imports"),
            self.cb.code(f'''from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql.functions import col, lit, current_timestamp, when, count, sum as spark_sum, mean
from pyspark.sql.types import TimestampType
from datetime import datetime
import mlflow

fe = FeatureEngineeringClient()
CATALOG = "{catalog}"
SCHEMA = "{schema}"
FEATURE_TABLE = f"{{CATALOG}}.{{SCHEMA}}.customer_features"
MODEL_URI = f"models:/{{CATALOG}}.{{SCHEMA}}.{model_name}@production"

print(f"Feature Store: {{FEATURE_TABLE}}")
print(f"Model: {{MODEL_URI}}")'''),

            self.cb.section("2. Set Inference Point-in-Time"),
            self.cb.markdown('''**Critical**: The inference timestamp is the point-in-time for feature retrieval.
The Databricks Feature Store uses `timestamp_lookup_key` to ensure PIT correctness.'''),
            self.cb.code('''# INFERENCE_TIMESTAMP: The point-in-time for this batch prediction
# For production batch jobs, use the job execution timestamp
INFERENCE_TIMESTAMP = datetime.now()

# Alternative: Use a specific historical timestamp for backtesting
# INFERENCE_TIMESTAMP = datetime(2024, 1, 15, 0, 0, 0)

print("=" * 70)
print(f"INFERENCE POINT-IN-TIME: {INFERENCE_TIMESTAMP}")
print("=" * 70)
print("Features will be retrieved as they existed at this timestamp.")'''),

            self.cb.section("3. Load Customers to Score"),
            self.cb.code(f'''# Load customers to score from the gold layer
df_customers = spark.table("{catalog}.{schema}.gold_customers")

# Select only entity IDs - features will come from the feature store
entity_df = df_customers.select("entity_id")

# Add the inference timestamp for point-in-time lookup
entity_df = entity_df.withColumn(
    "inference_timestamp",
    lit(INFERENCE_TIMESTAMP).cast(TimestampType())
)

print(f"Customers to score: {{entity_df.count():,}}")
print(f"Inference Point-in-Time: {{INFERENCE_TIMESTAMP}}")
entity_df.show(5)'''),

            self.cb.section("4. Define Feature Lookups with Point-in-Time"),
            self.cb.markdown('''The `timestamp_lookup_key` parameter ensures that features are retrieved
as they existed at the specified inference timestamp - no future data leakage.'''),
            self.cb.code(f'''# Define feature lookups with PIT correctness
# The timestamp_lookup_key ensures features are retrieved as of inference_timestamp
feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key=["entity_id"],
        timestamp_lookup_key="inference_timestamp",  # PIT lookup
    )
]

print("Feature lookups configured with Point-in-Time correctness")
print(f"  Feature Table: {{FEATURE_TABLE}}")
print(f"  Lookup Key: entity_id")
print(f"  Timestamp Key: inference_timestamp")'''),

            self.cb.section("5. Score with Feature Store (PIT-Correct)"),
            self.cb.markdown('''Use `fe.score_batch()` to automatically retrieve features with PIT correctness
and apply the model. This ensures training-serving consistency.'''),
            self.cb.code(f'''# Score using the feature store with automatic PIT feature retrieval
# This is the recommended approach for production inference
try:
    # Method 1: Use fe.score_batch for automatic feature lookup
    predictions = fe.score_batch(
        df=entity_df,
        model_uri=MODEL_URI,
        result_type="double",
    )
    print("Scored using fe.score_batch with automatic feature lookup")
except Exception as e:
    print(f"fe.score_batch not available: {{e}}")
    print("Falling back to manual feature retrieval...")

    # Method 2: Manual feature retrieval with PIT join
    training_set = fe.create_training_set(
        df=entity_df,
        feature_lookups=feature_lookups,
        label=None,
    )
    inference_df = training_set.load_df()

    # Load model and score
    model = mlflow.pyfunc.load_model(MODEL_URI)

    # Convert to pandas for scoring
    pdf = inference_df.toPandas()
    feature_cols = [c for c in pdf.columns if c not in ["entity_id", "inference_timestamp"]]

    predictions_array = model.predict(pdf[feature_cols])
    pdf["prediction"] = predictions_array

    predictions = spark.createDataFrame(pdf)
    print("Scored using manual feature retrieval with PIT join")'''),

            self.cb.section("6. Apply Threshold and Risk Tiers"),
            self.cb.code(f'''threshold = {threshold}

# Add prediction columns and risk tiers
df_scored = (predictions
    .withColumn("churn_probability", col("prediction"))
    .withColumn("churn_prediction", when(col("prediction") >= threshold, 1).otherwise(0))
    .withColumn("risk_tier",
        when(col("prediction") >= 0.6, "High")
        .when(col("prediction") >= 0.3, "Medium")
        .otherwise("Low")
    )
    .withColumn("inference_point_in_time", lit(INFERENCE_TIMESTAMP).cast(TimestampType()))
    .withColumn("model_uri", lit(MODEL_URI))
)

print(f"Applied threshold: {{threshold}}")
print(f"Added risk tiers: High (>=0.6), Medium (>=0.3), Low (<0.3)")'''),

            self.cb.section("7. Batch Inference Results Dashboard"),
            self.cb.markdown('''Display the batch scoring results with the **point-in-time** prominently shown.'''),
            self.cb.code('''# Calculate summary statistics
summary = df_scored.agg(
    count("*").alias("total_customers"),
    spark_sum("churn_prediction").alias("predicted_churners"),
    mean("churn_probability").alias("avg_probability")
).collect()[0]

total = summary["total_customers"]
churners = summary["predicted_churners"]
avg_prob = summary["avg_probability"]

# Risk distribution
risk_dist = df_scored.groupBy("risk_tier").count().collect()
risk_dict = {row["risk_tier"]: row["count"] for row in risk_dist}

print("=" * 70)
print("BATCH INFERENCE RESULTS DASHBOARD")
print("=" * 70)
print(f"")
print(f"📅 INFERENCE POINT-IN-TIME: {INFERENCE_TIMESTAMP.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"")
print(f"📊 SUMMARY STATISTICS:")
print(f"   Total Customers Scored: {total:,}")
print(f"   Predicted Churners: {churners:,} ({churners/total*100:.1f}%)")
print(f"   Average Churn Probability: {avg_prob:.3f}")
print(f"")
print(f"🎯 RISK DISTRIBUTION:")
print(f"   High Risk:   {risk_dict.get('High', 0):,}")
print(f"   Medium Risk: {risk_dict.get('Medium', 0):,}")
print(f"   Low Risk:    {risk_dict.get('Low', 0):,}")
print(f"")
print("=" * 70)'''),

            self.cb.section("8. Interactive Dashboard Display"),
            self.cb.code('''# Display risk distribution
print(f"\\n📊 Risk Distribution (PIT: {INFERENCE_TIMESTAMP.strftime('%Y-%m-%d %H:%M')}):")
display(df_scored.groupBy("risk_tier").count().orderBy("risk_tier"))

# Display sample predictions with inference metadata
print(f"\\n📋 Sample Predictions (showing inference_point_in_time):")
display(
    df_scored.select(
        "entity_id",
        "churn_probability",
        "risk_tier",
        "inference_point_in_time"
    ).limit(10)
)

# Display probability distribution
print(f"\\n📈 Probability Distribution:")
display(df_scored.select("churn_probability").summary())'''),

            self.cb.section("9. Save Predictions with Metadata"),
            self.cb.code(f'''# Save predictions with full audit trail
# Include inference_point_in_time for reproducibility

output_cols = [
    "entity_id",
    "churn_probability",
    "churn_prediction",
    "risk_tier",
    "inference_point_in_time",  # Critical for audit
    "model_uri"
]

# Save to Delta table with timestamp partition
df_scored.select(output_cols).write \\
    .format("delta") \\
    .mode("overwrite") \\
    .option("overwriteSchema", "true") \\
    .saveAsTable("{catalog}.{schema}.predictions")

print(f"✅ Predictions saved to {catalog}.{schema}.predictions")
print(f"📅 Inference Point-in-Time: {{INFERENCE_TIMESTAMP}}")
print(f"📊 Records: {{df_scored.count():,}}")'''),

            self.cb.section("10. Create Predictions Audit Log"),
            self.cb.code(f'''from pyspark.sql.functions import current_timestamp as spark_current_timestamp

# Create or append to audit log
audit_record = spark.createDataFrame([{{
    "inference_id": f"batch_{{INFERENCE_TIMESTAMP.strftime('%Y%m%d_%H%M%S')}}",
    "inference_timestamp": INFERENCE_TIMESTAMP,
    "total_customers": total,
    "predicted_churners": int(churners),
    "avg_probability": float(avg_prob),
    "model_uri": MODEL_URI,
    "threshold": {threshold},
    "created_at": datetime.now(),
}}])

# Append to audit log
audit_record.write \\
    .format("delta") \\
    .mode("append") \\
    .saveAsTable("{catalog}.{schema}.inference_audit_log")

print(f"✅ Audit log updated: {catalog}.{schema}.inference_audit_log")'''),

            self.cb.section("11. Summary"),
            self.cb.code('''print("=" * 70)
print("BATCH INFERENCE COMPLETE")
print("=" * 70)
print(f"")
print(f"🕐 Point-in-Time Used: {INFERENCE_TIMESTAMP}")
print(f"📊 Customers Scored: {total:,}")
print(f"⚠️  High Risk: {risk_dict.get('High', 0):,}")
print(f"🟡 Medium Risk: {risk_dict.get('Medium', 0):,}")
print(f"✅ Low Risk: {risk_dict.get('Low', 0):,}")
print(f"")
print("The inference_point_in_time column in the predictions table")
print("records exactly when features were retrieved, ensuring")
print("full auditability and reproducibility.")
print(f"")
print("Next steps:")
print("1. Review high-risk customers for intervention")
print("2. Set up scheduled inference jobs")
print("3. Monitor prediction drift over time")'''),
        ]
