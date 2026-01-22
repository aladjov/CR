# Feature Store Integration

The framework includes a **unified feature store** module that provides point-in-time correct feature management across both local development (Feast) and production (Databricks Feature Engineering).

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE STORE ARCHITECTURE                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Unified Data (with temporal columns)                                       │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────────────────────┐                                    │
│   │        FeatureRegistry              │                                    │
│   │  • TemporalFeatureDefinition        │  Define features with              │
│   │  • Computation type (passthrough,   │  temporal metadata                 │
│   │    aggregation, derived)            │                                    │
│   │  • Leakage risk annotation          │                                    │
│   └──────────────────┬──────────────────┘                                    │
│                      │                                                       │
│                      ▼                                                       │
│   ┌─────────────────────────────────────┐                                    │
│   │      FeatureStoreManager            │                                    │
│   │  • Unified API for both backends    │                                    │
│   │  • publish_features()               │                                    │
│   │  • get_training_features()          │                                    │
│   │  • get_inference_features()         │                                    │
│   └──────────────────┬──────────────────┘                                    │
│                      │                                                       │
│         ┌────────────┴────────────┐                                          │
│         ▼                         ▼                                          │
│   ┌─────────────┐          ┌─────────────┐                                   │
│   │ FeastBackend│          │ Databricks  │                                   │
│   │   (Local)   │          │  Backend    │                                   │
│   │             │          │             │                                   │
│   │ • Parquet   │          │ • Unity     │                                   │
│   │ • SQLite    │          │   Catalog   │                                   │
│   │ • PIT joins │          │ • Delta Lake│                                   │
│   └─────────────┘          │ • PIT joins │                                   │
│                            └─────────────┘                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

| Component | Purpose |
|-----------|---------|
| `TemporalFeatureDefinition` | Define features with temporal metadata (timestamp column, computation type, leakage risk) |
| `FeatureRegistry` | Collection of feature definitions, serializable to JSON |
| `FeatureStoreManager` | Unified API for publishing and retrieving features |
| `FeastBackend` | Local development using Feast (Parquet + SQLite) |
| `DatabricksBackend` | Production using Databricks Feature Engineering (Unity Catalog) |

## Usage Example

```python
from customer_retention.feature_store import (
    FeatureStoreManager,
    FeatureRegistry,
    TemporalFeatureDefinition,
    FeatureComputationType,
)

# Define features with temporal metadata
registry = FeatureRegistry()
registry.register(TemporalFeatureDefinition(
    name="monthly_charges",
    description="Customer's monthly bill amount",
    entity_key="entity_id",
    timestamp_column="feature_timestamp",
    source_columns=["monthly_charges"],
    computation_type=FeatureComputationType.PASSTHROUGH,
    data_type="float64",
    leakage_risk="low",
))

# Create manager (auto-selects backend based on environment)
manager = FeatureStoreManager.create(
    backend="feast",  # or "databricks" in production
    repo_path="./experiments/feature_store/feature_repo",
)

# Publish features
manager.publish_features(df, registry, "customer_features")

# Get point-in-time correct training features
training_df = manager.get_training_features(
    entity_df,      # entity_id + event_timestamp
    registry,
    feature_names=["monthly_charges", "tenure"],
)
```

## Local Setup with Feast

```bash
# Install Feast
pip install feast

# Initialize feature repository (already created by the framework)
ls experiments/feature_store/feature_repo/
# → feature_store.yaml  features.py
```

**experiments/feature_store/feature_repo/feature_store.yaml:**
```yaml
project: customer_retention
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
```

## Using the Feature Store Adapter

```python
from customer_retention.integrations.adapters.feature_store import (
    FeastAdapter, FeatureViewConfig, get_feature_store
)

# Auto-detect environment (Feast locally, Databricks Feature Engineering on Databricks)
feature_store = get_feature_store()

# Register a feature view
config = FeatureViewConfig(
    name="customer_features",
    entity_key="customer_id",
    features=["age", "tenure_months", "total_spend"],
    ttl_days=30
)

# Register features from your gold layer
import pandas as pd
gold_df = pd.read_parquet("orchestration/churn_prediction/data/gold/features.parquet")
feature_store.register_feature_view(config, gold_df)

# Get historical features for training (point-in-time correct)
entity_df = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"])
})
training_df = feature_store.get_historical_features(
    entity_df,
    feature_refs=["customer_features:age", "customer_features:total_spend"]
)

# Materialize features for online serving
feature_store.materialize(
    feature_views=["customer_features"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Get online features for real-time inference
online_features = feature_store.get_online_features(
    entity_keys={"customer_id": [1, 2, 3]},
    feature_refs=["customer_features:age", "customer_features:total_spend"]
)
```

## Notebook Integration

The generated notebook stage `11_feature_store.ipynb` demonstrates the complete workflow:

1. Load gold layer data with temporal columns
2. Define feature registry from numeric columns
3. Publish features to the feature store
4. Create point-in-time correct training sets
5. Validate feature consistency

## Databricks Feature Engineering

On Databricks, feature store integration uses Databricks Feature Engineering:

```python
from customer_retention.integrations.adapters.feature_store import get_feature_store

# Automatically uses DatabricksFeatureStoreAdapter when on Databricks
feature_store = get_feature_store(catalog="main", schema="features")

# Same API works - but uses Unity Catalog under the hood
feature_store.register_feature_view(config, spark_df)
# Creates table: main.features.customer_features
```

## Next Steps

- [[Local Track]] - Complete local execution with Feast
- [[Databricks Track]] - Production execution with Unity Catalog
- [[Temporal Framework]] - How temporal columns enable PIT joins
