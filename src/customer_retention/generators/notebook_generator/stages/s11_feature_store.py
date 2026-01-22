"""Feature store stage for notebook generation.

This stage generates notebooks that publish features to the feature store
and retrieve them for training with point-in-time correctness.
"""

from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class FeatureStoreStage(StageGenerator):
    """Generate feature store integration notebooks."""

    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.FEATURE_STORE

    @property
    def title(self) -> str:
        return "11 - Feature Store Integration"

    @property
    def description(self) -> str:
        return """Publish features to the feature store and create training sets with point-in-time correctness.

This notebook:
1. Loads features from the gold layer
2. Registers feature definitions
3. Publishes features to the feature store (Feast or Databricks)
4. Creates point-in-time correct training sets
"""

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        """Generate cells for local Feast-based workflow."""
        return self.header_cells() + [
            self.cb.section("1. Setup and Imports"),
            self.cb.code('''import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from customer_retention.integrations.feature_store import (
    FeatureStoreManager,
    FeatureRegistry,
    TemporalFeatureDefinition,
    FeatureComputationType,
    TemporalAggregation,
)
from customer_retention.stages.temporal import SnapshotManager

print("Feature store imports loaded")'''),

            self.cb.section("2. Load Gold Layer Data"),
            self.cb.markdown('''Load the gold layer features. These should already have `feature_timestamp` for point-in-time correctness.'''),
            self.cb.code('''# Load gold layer data
gold_path = Path("./experiments/data/gold/customers_features.parquet")
if gold_path.exists():
    df = pd.read_parquet(gold_path)
    print(f"Loaded gold layer: {df.shape}")
else:
    # Fall back to snapshot
    snapshot_manager = SnapshotManager(Path("./experiments/data"))
    latest = snapshot_manager.get_latest_snapshot()
    if latest:
        df, _ = snapshot_manager.load_snapshot(latest)
        print(f"Loaded snapshot {latest}: {df.shape}")
    else:
        raise FileNotFoundError("No gold layer or snapshot found")

# Verify temporal columns exist
required_cols = ["entity_id", "feature_timestamp"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"Warning: Missing temporal columns: {missing}")
else:
    print("Temporal columns present")
'''),

            self.cb.section("3. Define Feature Registry"),
            self.cb.markdown('''Create feature definitions with temporal metadata. This ensures consistent feature computation across training and inference.'''),
            self.cb.code('''# Create feature registry
registry = FeatureRegistry()

# Get numeric columns (excluding metadata)
exclude_cols = {"entity_id", "target", "feature_timestamp", "label_timestamp", "label_available_flag"}
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

# Register each numeric feature
for col in numeric_cols:
    registry.register(TemporalFeatureDefinition(
        name=col,
        description=f"Feature: {col}",
        entity_key="entity_id",
        timestamp_column="feature_timestamp",
        source_columns=[col],
        computation_type=FeatureComputationType.PASSTHROUGH,
        data_type=str(df[col].dtype),
        leakage_risk="low",
    ))

print(f"Registered {len(registry)} features")
print(f"Features: {registry.list_features()[:10]}...")  # Show first 10
'''),

            self.cb.section("4. Initialize Feature Store Manager"),
            self.cb.code('''# Create feature store manager (uses Feast locally)
manager = FeatureStoreManager.create(
    backend="feast",
    repo_path="./experiments/feature_store/feature_repo",
    output_path="./experiments/data",
)

print("Feature store manager initialized")
print(f"Existing tables: {manager.list_tables()}")
'''),

            self.cb.section("5. Publish Features to Feature Store"),
            self.cb.code('''# Publish features
table_name = manager.publish_features(
    df=df,
    registry=registry,
    table_name="customer_features",
    entity_key="entity_id",
    timestamp_column="feature_timestamp",
    mode="overwrite",  # Use "merge" for incremental updates
)

print(f"Published features to: {table_name}")
print(f"Tables after publish: {manager.list_tables()}")
'''),

            self.cb.section("6. Create Point-in-Time Training Set"),
            self.cb.markdown('''Create a training set with point-in-time correct feature retrieval. The entity DataFrame specifies when we want to "observe" each customer.'''),
            self.cb.code('''# Create entity DataFrame with observation timestamps
# This simulates "when would we have made a prediction?"
entity_df = df[["entity_id", "feature_timestamp"]].copy()
entity_df = entity_df.rename(columns={"feature_timestamp": "event_timestamp"})

# Get point-in-time correct features
training_df = manager.get_training_features(
    entity_df=entity_df,
    registry=registry,
    feature_names=registry.list_features()[:20],  # First 20 features
    table_name="customer_features",
    timestamp_column="event_timestamp",
)

print(f"Training set shape: {training_df.shape}")
print(f"Columns: {list(training_df.columns)}")
'''),

            self.cb.section("7. Save Feature Registry"),
            self.cb.code('''# Save registry for later use
registry_path = Path("./experiments/feature_store/feature_registry.json")
registry_path.parent.mkdir(parents=True, exist_ok=True)
registry.save(registry_path)
print(f"Saved feature registry to {registry_path}")

# Verify we can reload it
loaded_registry = FeatureRegistry.load(registry_path)
print(f"Reloaded registry: {len(loaded_registry)} features")
'''),

            self.cb.section("8. Validate Feature Store Integration"),
            self.cb.code('''# Validate that features match between direct load and feature store
direct_features = df[["entity_id"] + registry.list_features()[:5]].head(10)
store_features = training_df[["entity_id"] + [f for f in registry.list_features()[:5] if f in training_df.columns]].head(10)

print("Direct load sample:")
print(direct_features)
print("\\nFeature store sample:")
print(store_features)

# Check for mismatches
if set(direct_features.columns) == set(store_features.columns):
    merged = direct_features.merge(store_features, on="entity_id", suffixes=("_direct", "_store"))
    for col in registry.list_features()[:5]:
        if f"{col}_direct" in merged.columns and f"{col}_store" in merged.columns:
            match = np.allclose(
                merged[f"{col}_direct"].fillna(0),
                merged[f"{col}_store"].fillna(0),
                rtol=1e-5
            )
            print(f"  {col}: {'MATCH' if match else 'MISMATCH'}")
'''),

            self.cb.section("9. Summary"),
            self.cb.code('''print("=" * 60)
print("Feature Store Integration Complete")
print("=" * 60)
print(f"Features registered: {len(registry)}")
print(f"Feature table: customer_features")
print(f"Registry saved: {registry_path}")
print(f"Training set shape: {training_df.shape}")
print()
print("Next steps:")
print("1. Use the feature store for model training")
print("2. Use get_inference_features() for online serving")
print("3. Schedule feature refresh jobs")
'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        """Generate cells for Databricks Feature Engineering workflow."""
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema

        return self.header_cells() + [
            self.cb.section("1. Setup"),
            self.cb.code(f'''from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql.functions import col, current_timestamp
import pandas as pd

fe = FeatureEngineeringClient()
CATALOG = "{catalog}"
SCHEMA = "{schema}"

print(f"Using catalog: {{CATALOG}}.{{SCHEMA}}")
'''),

            self.cb.section("2. Load Gold Table"),
            self.cb.code(f'''# Load gold layer
df = spark.table("{catalog}.{schema}.gold_customers")
print(f"Loaded gold table: {{df.count()}} rows")

# Display schema
df.printSchema()
'''),

            self.cb.section("3. Create Feature Table"),
            self.cb.markdown('''Create a Unity Catalog feature table with primary keys and timestamp column for point-in-time lookups.'''),
            self.cb.code(f'''# Define feature table
FEATURE_TABLE = "{catalog}.{schema}.customer_features"

# Select feature columns (exclude metadata)
exclude_cols = {{"entity_id", "target", "feature_timestamp", "label_timestamp", "label_available_flag"}}
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Create or replace feature table
feature_df = df.select(
    "entity_id",
    "feature_timestamp",
    *feature_cols
)

fe.create_table(
    name=FEATURE_TABLE,
    primary_keys=["entity_id"],
    timestamp_keys=["feature_timestamp"],
    df=feature_df,
    description="Customer features for churn prediction",
)

print(f"Created feature table: {{FEATURE_TABLE}}")
'''),

            self.cb.section("4. Create Training Set with Point-in-Time Lookups"),
            self.cb.code('''# Create entity DataFrame with observation timestamps
entity_df = df.select("entity_id", col("feature_timestamp").alias("event_timestamp"))

# Define feature lookups with timestamp_lookup_key for PIT correctness
feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key=["entity_id"],
        timestamp_lookup_key="event_timestamp",
    )
]

# Create training set
training_set = fe.create_training_set(
    df=entity_df,
    feature_lookups=feature_lookups,
    label=None,  # Add label column name if joining labels
)

training_df = training_set.load_df()
print(f"Training set: {training_df.count()} rows, {len(training_df.columns)} columns")
training_df.show(5)
'''),

            self.cb.section("5. Log Model with Feature Store Lineage"),
            self.cb.markdown('''When training models, use `fe.log_model()` to capture feature lineage. This enables automatic feature lookup during inference.'''),
            self.cb.code('''# Example: Train and log model with feature lineage
# (Uncomment and modify for your model)

# from sklearn.ensemble import RandomForestClassifier
# import mlflow
#
# # Prepare features
# pdf = training_df.toPandas()
# feature_cols = [c for c in pdf.columns if c not in ["entity_id", "event_timestamp", "target"]]
# X = pdf[feature_cols]
# y = pdf["target"]
#
# # Train model
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X, y)
#
# # Log with feature lineage
# with mlflow.start_run():
#     fe.log_model(
#         model=model,
#         artifact_path="model",
#         flavor=mlflow.sklearn,
#         training_set=training_set,
#     )
#     print("Model logged with feature lineage")
'''),

            self.cb.section("6. Online Feature Serving"),
            self.cb.markdown('''For real-time inference, use Model Serving with automatic feature lookup.'''),
            self.cb.code('''# Score batch with automatic feature lookup
# The model automatically retrieves latest features from the feature table

# Example: Score new customers
# new_customers = spark.createDataFrame([
#     {"entity_id": "new_customer_1"},
#     {"entity_id": "new_customer_2"},
# ])
#
# # Feature lookups happen automatically during scoring
# predictions = fe.score_batch(
#     df=new_customers,
#     model_uri="models:/churn_model/production",
# )
# predictions.show()
'''),

            self.cb.section("7. Summary"),
            self.cb.code('''print("=" * 60)
print("Databricks Feature Store Integration Complete")
print("=" * 60)
print(f"Feature table: {FEATURE_TABLE}")
print(f"Primary key: entity_id")
print(f"Timestamp key: feature_timestamp")
print(f"Training set rows: {training_df.count()}")
print()
print("Next steps:")
print("1. Train model using training_set")
print("2. Log model with fe.log_model() for lineage")
print("3. Deploy to Model Serving for auto feature lookup")
'''),
        ]
