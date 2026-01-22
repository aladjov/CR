# Temporal Framework: Leakage-Safe Architecture

The framework includes a **temporal framework** that prevents data leakage by enforcing point-in-time (PIT) correctness throughout the ML pipeline. This is critical for production models where features must only use data available at prediction time.

## The Problem: Data Leakage

Data leakage occurs when your model inadvertently uses future information during training:

- **Label leakage**: Using the target value (or proxies for it) as features
- **Temporal leakage**: Using features computed from data that wouldn't be available at prediction time
- **Target encoding leakage**: Encoding categories using statistics that include the target

These issues cause models to perform well in development but fail in production.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        TEMPORAL FRAMEWORK ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   RAW DATA                                                                   │
│   ┌─────────────────────┐                                                    │
│   │  customers.csv      │  No timestamps? Multiple date columns?             │
│   │  • customer_id      │  Kaggle-style static snapshot?                     │
│   │  • tenure           │                                                    │
│   │  • churn            │                                                    │
│   └──────────┬──────────┘                                                    │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                   ScenarioDetector                          │            │
│   │  • Analyzes columns for datetime patterns                   │            │
│   │  • Identifies feature vs label timestamps                   │            │
│   │  • Detects derivable timestamps (e.g., tenure → signup)     │            │
│   │  • Returns: scenario + TimestampConfig                      │            │
│   └──────────┬──────────────────────────────────────────────────┘            │
│              │                                                               │
│              │  Scenarios:                                                   │
│              │  • production - explicit timestamps found                     │
│              │  • derived - computable from tenure/contract                  │
│              │  • synthetic - no temporal info, generate synthetic           │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                UnifiedDataPreparer                          │            │
│   │  • Applies TimestampConfig via TimestampManager             │            │
│   │  • Adds: feature_timestamp, label_timestamp                 │            │
│   │  • Adds: label_available_flag                               │            │
│   │  • Validates point-in-time correctness                      │            │
│   └──────────┬──────────────────────────────────────────────────┘            │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────┐                                                    │
│   │  UNIFIED DATA       │  Now has temporal columns                          │
│   │  • entity_id        │                                                    │
│   │  • feature_*        │                                                    │
│   │  • target           │                                                    │
│   │  • feature_timestamp│  ← When features were observed                     │
│   │  • label_timestamp  │  ← When label became known                         │
│   │  • label_available  │  ← Can use for training?                           │
│   └──────────┬──────────┘                                                    │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                   SnapshotManager                           │            │
│   │  • Filters: label_available_flag == True                    │            │
│   │  • Filters: label_timestamp <= cutoff_date                  │            │
│   │  • Computes SHA256 hash for integrity                       │            │
│   │  • Saves versioned parquet + metadata JSON                  │            │
│   └──────────┬──────────────────────────────────────────────────┘            │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────┐    ┌─────────────────────┐                         │
│   │  training_v1.parquet│    │  training_v1_       │                         │
│   │  (filtered data)    │    │  metadata.json      │                         │
│   │                     │    │  • snapshot_id      │                         │
│   │  Only rows where    │    │  • data_hash        │                         │
│   │  label was known    │    │  • cutoff_date      │                         │
│   │  before cutoff      │    │  • row_count        │                         │
│   └─────────────────────┘    └─────────────────────┘                         │
│              │                                                               │
│              │  On load: hash verified                                       │
│              │  → ValueError if data modified                                │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                   MODEL TRAINING                            │            │
│   │  • Uses only label_available records                        │            │
│   │  • Features from feature_timestamp                          │            │
│   │  • Labels from label_timestamp                              │            │
│   │  • No future data leakage possible                          │            │
│   └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

The temporal framework introduces three key columns to enforce PIT correctness:

| Column | Purpose |
|--------|---------|
| `feature_timestamp` | When features were observed (point-in-time for feature values) |
| `label_timestamp` | When the label became known (when ground truth was revealed) |
| `label_available_flag` | Whether the label is available for training (respects observation window) |

**Key Rule**: Features from time T can only be used to predict labels from time T+observation_window.

## Timestamp Scenarios

The framework auto-detects your data's timestamp scenario:

| Scenario | Description | Example |
|----------|-------------|---------|
| **Production** | Explicit feature and label timestamps in data | CRM with `last_activity_date` and `churn_date` |
| **Production (Derived)** | Timestamps derivable from existing columns | `signup_date` + 90 days = label availability |
| **Derived** | Feature timestamps can be computed from date columns | Parse dates from `last_login`, `account_created` |
| **Synthetic** | No timestamps available (Kaggle-style data) | Static customer snapshot with no dates |

```python
from customer_retention.stages.temporal import ScenarioDetector

# Auto-detect scenario
detector = ScenarioDetector()
scenario, config, discovery = detector.detect(df, target_column="churn")

print(f"Scenario: {scenario}")
# → "production" | "partial" | "derived" | "synthetic"

print(f"Strategy: {config.strategy.value}")
# → "production" | "derived" | "synthetic_fixed"
```

## UnifiedDataPreparer: Single Entry Point

The `UnifiedDataPreparer` handles all timestamp scenarios with a unified API:

```python
from customer_retention.stages.temporal import UnifiedDataPreparer, ScenarioDetector

# Detect scenario
detector = ScenarioDetector()
scenario, config, _ = detector.detect(df, "churn")

# Prepare data with PIT correctness
preparer = UnifiedDataPreparer(output_path, config)
prepared_df = preparer.prepare_from_raw(
    df,
    target_column="churn",
    entity_column="customer_id"
)

# Result includes:
# - feature_timestamp (observed or synthetic)
# - label_timestamp (when label became known)
# - label_available_flag (ready for training?)
```

## Versioned Training Snapshots

The `SnapshotManager` creates versioned, integrity-checked training snapshots:

```python
from customer_retention.stages.temporal import SnapshotManager
from datetime import datetime

# Initialize manager
manager = SnapshotManager(base_path="./output")

# Create snapshot at a specific cutoff date
cutoff = datetime(2024, 6, 1)
metadata = manager.create_snapshot(
    df=prepared_df,
    cutoff_date=cutoff,
    target_column="churn",
    snapshot_name="training"
)

print(f"Snapshot: {metadata.snapshot_id}")      # → training_v1
print(f"Hash: {metadata.data_hash[:16]}...")    # → SHA256 integrity hash
print(f"Rows: {metadata.row_count}")            # → Records with label_available=True

# Load snapshot with integrity verification
df, metadata = manager.load_snapshot("training_v1")
# → Raises ValueError if data has been modified

# Compare snapshots
diff = manager.compare_snapshots("training_v1", "training_v2")
print(f"Row diff: {diff['row_diff']}")
print(f"New features: {diff['new_features']}")
```

## Leakage Detection

The `LeakageDetector` validates your prepared data using multiple probes:

```python
from customer_retention.analysis.diagnostics import LeakageDetector

detector = LeakageDetector()
report = detector.validate(
    df=prepared_df,
    target_column="churn",
    feature_columns=feature_cols,
    timestamp_column="feature_timestamp",
    label_timestamp_column="label_timestamp"
)

print(f"Leakage detected: {report.has_leakage}")
for issue in report.issues:
    print(f"  - {issue.probe}: {issue.description}")
    print(f"    Severity: {issue.severity}, Columns: {issue.affected_columns}")

# Probe types:
# - correlation: Suspiciously high correlation with target
# - separation: Near-perfect class separation
# - temporal: Features using future information
# - single_feature_auc: Single feature achieving unrealistic AUC
```

## Migration Script

Migrate existing datasets to the temporal format:

```bash
# Migrate a Kaggle-style dataset
python scripts/data/migrate_to_temporal.py \
    --input data/customers.csv \
    --output output/ \
    --target churn

# Output:
# Loading data from data/customers.csv...
# Detecting timestamp scenario...
#   Scenario: synthetic
#   Strategy: synthetic_fixed
# Preparing data with timestamps...
# Creating training snapshot...
#   Snapshot ID: training_v1
#   Data hash: a1b2c3d4e5f6...
```

## Comparison: With vs Without

| Without Temporal Framework | With Temporal Framework |
|---------------------------|------------------------|
| Models trained on all data | Only label_available=True records used |
| No timestamp validation | feature_timestamp < label_timestamp enforced |
| Data can be modified silently | SHA256 integrity verification on load |
| Single training dataset | Versioned snapshots with comparison |
| Leakage discovered in production | Leakage detected during validation |

## Next Steps

- [[Feature Store]] - Temporal-aware feature management
- [[Local Track]] - Run pipelines with leakage protection
- [[Tutorial: Bank Customer Churn]] - See temporal framework in action
