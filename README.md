# Customer Retention ML Framework

A hands-on framework for learning and implementing customer churn prediction pipelines. Built for data scientists who want to understand the full ML lifecycle—from raw data exploration to production deployment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![CI](https://github.com/aladjov/CR/actions/workflows/ci.yaml/badge.svg)](https://github.com/aladjov/CR/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/aladjov/CR/branch/master/graph/badge.svg)](https://codecov.io/gh/aladjov/CR)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Typed](https://img.shields.io/badge/typed-Pydantic-blue)](https://docs.pydantic.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-integrated-0194E2?logo=mlflow)](https://mlflow.org/)
[![Databricks](https://img.shields.io/badge/Databricks-compatible-FF3621?logo=databricks)](https://databricks.com/)

---

## What This Project Is

This is an **educational framework** that walks you through building a customer retention ML pipeline step-by-step. Instead of a black-box solution, you get:

- **Guided notebooks** that teach the *why* behind each step, not just the *how*
- **Modular components** you can understand, modify, and extend
- **A foundation to build on** as you add more sophisticated techniques

### Current Capabilities

The framework currently implements fundamental techniques:

| Area | What's Implemented |
|------|-------------------|
| **Data Exploration** | Automatic type detection, quality profiling, basic statistics |
| **Text Processing** | Embeddings + PCA dimensionality reduction for unstructured text columns |
| **Time Series Support** | Entity lifecycle analysis, temporal patterns, trend/seasonality detection |
| **Multi-Dataset Analysis** | Auto-detect relationships, join suggestions, time-window aggregations |
| **Cleaning** | Missing value imputation, outlier handling (IQR, Z-score, Winsorization) |
| **Feature Engineering** | Temporal features, categorical encoding, basic interactions |
| **Modeling** | Logistic Regression, Random Forest, XGBoost, LightGBM baselines |
| **Evaluation** | Standard metrics (AUC, precision, recall), threshold tuning |
| **Interpretability** | SHAP values, feature importance |
| **Monitoring** | Basic drift detection (KS test, PSI), performance tracking |
| **Iteration Support** | Version tracking, recommendation status, feedback loops, retraining triggers |
| **Scoring Pipeline** | Production simulation with holdout validation, SHAP explanations, customer-level analysis |

### What We're Building Toward

This is a growing project. Planned additions include:

- Advanced feature engineering (RFM analysis, text embeddings with LLM labeling)
- Deep learning approaches (TabNet, neural networks for sequential data)
- Causal inference for intervention optimization
- More sophisticated drift detection and automated retraining
- Better handling of class imbalance

**Contributions welcome!** See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## The Educational Approach

### Learn by Doing: Exploration Notebooks

The heart of this framework is a series of **interactive notebooks** that guide you through the entire ML workflow. Each notebook:

- Explains the concepts before diving into code
- Shows you what to look for in your data
- Demonstrates common pitfalls and how to avoid them
- Produces artifacts you'll use in later stages

```
Exploration Notebooks (exploration_notebooks/)
────────────────────────────────────────────────────────
00. Start Here            → Quick orientation and setup
01. Data Discovery        → Get to know your dataset (auto-detects granularity)
02. Column Deep Dive      → Understand each feature's distribution
03. Quality Assessment    → Find and fix data issues
04. Relationship Analysis → Discover feature-target patterns

05. Multi-Dataset         → Combine datasets, select which to include
06. Feature Opportunities → Identify engineering possibilities
07. Modeling Readiness    → Capture recommendations, validate before training
08. Baseline Experiments  → Establish performance benchmarks
09. Business Alignment    → Connect ML to business goals
10. Spec Generation       → Generate production pipeline specs
```

**Why this matters**: Most ML tutorials jump straight to `model.fit()`. Real-world projects fail because of data issues, leakage, or misaligned metrics—problems you catch in exploration.

### Bronze Per Shape: Event vs Entity Tracks

**Smart Routing**: Notebook 01 automatically detects if your data is entity-level (one row per customer) or event-level (multiple rows per customer over time). Each "shape" gets its own Bronze treatment:

```
Bronze Per Shape Architecture
─────────────────────────────────────────────────────────

                    Event Sources                 Entity Sources
                         │                              │
                         ▼                              │
                   01 Discovery                         │
                         │                              │
          ┌──────────────┴────────────┐                 │
          ▼                           │                 │
   ┌─────────────────────┐            │                 │
   │ BRONZE: Event Shape │            │                 │
   │  01a Temporal Dive  │            │                 │
   │  01a_a Text Dive *  │ ←── If TEXT columns detected │
   │  01b Temporal Qual  │            │                 │
   │  01c Temporal Pat   │            │                 │
   │  01d Aggregation ───┼──┐         │                 │
   └─────────────────────┘  │         │                 │
                            ▼         ▼                 ▼
                   ┌─────────────────────────────────────┐
                   │      BRONZE: Entity Shape           │
                   │  02  Column Deep Dive               │
                   │  02a Text Deep Dive *  ←── If TEXT  │
                   │  03  Quality Assessment             │
                   │  04  Relationship Analysis          │
                   └─────────────────────────────────────┘
                                     │
                                     ▼
                          05 Multi-Dataset → Silver

* Text notebooks are conditional - only run when TEXT columns are detected
```

**Event Bronze Track** (when EVENT_LEVEL granularity detected):

| Notebook | Purpose |
|----------|---------|
| **01a** Temporal Deep Dive | Entity lifecycle analysis, events per entity, activity segmentation |
| **01a_a** Temporal Text Deep Dive | TEXT column embeddings + PCA, then time-window aggregation (conditional) |
| **01b** Temporal Quality | Duplicate events, temporal gaps, future date validation |
| **01c** Temporal Patterns | Trends, seasonality, cohort analysis, recency vs target |
| **01d** Event Aggregation | Execute aggregations → produces entity-level dataset |

**Key insight**: Notebook 01d produces a new entity-level dataset that then flows into the Entity Bronze Track (02-04). This is not "going backwards" in the medallion architecture—each shape has its own Bronze layer.

**Time Window Aggregations**: The Event Bronze track helps you plan and execute aggregations to convert event-level data to entity-level features:

| Window | Use Case |
|--------|----------|
| 24h | Very recent activity, real-time signals |
| 7d | Weekly patterns, short-term engagement |
| 30d | Monthly patterns, subscription cycles |
| 90d | Quarterly trends, seasonal behavior |
| 180d | Semi-annual patterns, medium-term trends |
| 365d | Annual patterns, year-over-year comparison |
| all_time | Historical totals, lifetime value |

### Text Column Processing (Bronze Layer)

TEXT columns (tickets, emails, messages, unstructured text) require special handling to convert them into numeric features for ML models. The framework provides **conditional notebooks** that only run when TEXT columns are detected.

**Current Approach: Embeddings + PCA**

```
TEXT Column → Sentence Embeddings (384-dim) → PCA → pc1, pc2, ..., pcN
```

| Step | What Happens | Why |
|------|--------------|-----|
| **Embeddings** | `sentence-transformers` generates 384-dimensional vectors | Captures semantic meaning (similar texts → similar vectors) |
| **PCA** | Reduces dimensions to N components (95% variance by default) | Makes features usable with standard ML models |
| **Output** | Numeric features: `column_pc1`, `column_pc2`, ... | Ready for aggregation and modeling |

**Text Notebooks:**

| Notebook | Track | Purpose |
|----------|-------|---------|
| **02a** Text Columns Deep Dive | Entity Bronze | Static text in entity-level data |
| **01a_a** Temporal Text Deep Dive | Event Bronze | Text in events → embed → PCA → time-window aggregation |

**Event-Level Text Processing Flow:**

For event-level data (e.g., support tickets per customer), text is embedded at the event level, then aggregated:

```
Per Event:   ticket_text → embedding → [pc1, pc2, pc3]
Aggregate:   customer_id → ticket_text_pc1_mean_30d, ticket_text_pc2_std_7d, ...
```

This captures how text semantics change over time windows—useful for detecting shifts in customer communication patterns.

**Configuration:**

```python
from customer_retention.profiling import TextColumnProcessor, TextProcessingConfig

config = TextProcessingConfig(
    embedding_model="all-MiniLM-L6-v2",  # Fast, good quality, 384 dimensions
    variance_threshold=0.95,              # Keep components explaining 95% of variance
    min_components=2,                     # At least 2 features per text column
    max_components=10                     # Cap for manageability (optional)
)

processor = TextColumnProcessor(config)
df_with_features, result = processor.process_column(df, "ticket_text")
# Adds: ticket_text_pc1, ticket_text_pc2, ... to DataFrame
```

**Future Enhancement: LLM Labeling**

The architecture is designed to support a second approach for when specific categorical labels are needed:

```
TEXT Column → Sample → LLM Labels → Train Classifier → Apply to All
```

This will be useful when you need interpretable categories (sentiment, topic, intent) rather than latent semantic dimensions. The `processing_approach` field in recommendations supports both `"pca"` (current) and `"llm_labels"` (future).

**Integration with Bronze Layer:**

TEXT processing is a Bronze-layer transformation. The `RecommendationRegistry` captures text processing decisions:

```python
registry.add_bronze_text_processing(
    column="ticket_text",
    embedding_model="all-MiniLM-L6-v2",
    variance_threshold=0.95,
    n_components=5,
    rationale="Convert support ticket text to semantic features",
    source_notebook="02a_text_columns_deep_dive"
)
```

**Important:** TEXT columns are automatically excluded from categorical processing in notebooks 02 and 01a to prevent them from being treated as high-cardinality categoricals.

### From Exploration to Production

The exploration notebooks generate artifacts that drive production pipelines:

```
Exploration Outputs                    Production Usage
───────────────────                    ────────────────
{dataset}_findings.yaml       →        Column types, target, data profile
multi_dataset_findings.yaml   →        Selected datasets, relationships
recommendations (RecommendationRegistry) → Bronze/Silver/Gold transformations
```

These artifacts can be used in two ways:

1. **Local Execution** - Use `DataMaterializer` to apply transformations with pandas
2. **Databricks Production** - Export to standalone PySpark notebooks or use with Databricks Assistant

---

## Architecture Overview

The framework implements a **medallion architecture** with clear separation between the **iterative exploration loop** and **production execution**. Model development is iterative—you explore, train, evaluate, and refine based on feedback.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                     EXPLORATION LOOP (Iterative)                            │   │
│   │                                                                             │   │
│   │    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │   │
│   │    │  Explore │────▶│Recommend │────▶│  Train   │────▶│ Evaluate │          │   │
│   │    │   Data   │     │ Features │     │  Model   │     │ Results  │          │   │
│   │    └──────────┘     └──────────┘     └──────────┘     └────┬─────┘          │   │
│   │         ▲                                                   │               │   │
│   │         │                                                   │               │   │
│   │         │           ┌──────────────────────┐                │               │   │
│   │         └───────────│  Iteration Context   │◀───────────────┘               │   │
│   │                     │  • Version tracking  │                                │   │
│   │                     │  • Feature feedback  │  Triggers:                     │   │
│   │                     │  • Drift signals     │  • Manual refinement           │   │
│   │                     └──────────────────────┘  • Performance drop            │   │
│   │                                               • Data drift detected         │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                         │
│                                           │ When satisfied                          │
│                                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                       PRODUCTION EXECUTION                                  │   │
│   │                                                                             │   │
│   │    Choose ONE track based on your environment:                              │   │
│   │                                                                             │   │
│   │    ┌────────────────────────┐    ┌────────────────────────┐                 │   │
│   │    │   LOCAL TRACK          │    │   DATABRICKS TRACK     │                 │   │
│   │    │   Feast + MLFlow       │    │   Unity Catalog        │                 │   │
│   │    │                        │    │                        │                 │   │
│   │    │   • Feature store      │    │   • Delta Lake tables  │                 │   │
│   │    │   • Experiment tracking│    │   • Spark execution    │                 │   │
│   │    │   • Local serving      │    │   • Workflow jobs      │                 │   │
│   │    └────────────────────────┘    └────────────────────────┘                 │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Medallion Layers

| Layer | Scope | Notebooks | Description |
|-------|-------|-----------|-------------|
| **Landing** | Raw ingestion | 00, 01 | Prerequisites, data discovery |
| **Bronze-Event** | Clean event-shape data | 01a, 01a_a*, 01b, 01c, 01d | Temporal analysis, text processing → aggregation |
| **Bronze-Entity** | Clean entity-shape data | 02, 02a*, 03, 04 | Column/quality/relationship analysis, text processing |
| **Silver** | Join entity sources | 05 | Multi-dataset joins |
| **Gold** | ML-ready features | 06+ | Feature engineering, modeling |

\* Conditional notebooks that only run when TEXT columns are detected

### Temporal Framework: Leakage-Safe Architecture

The framework includes a **temporal framework** that prevents data leakage by enforcing point-in-time (PIT) correctness throughout the ML pipeline. This is critical for production models where features must only use data available at prediction time.

#### The Problem: Data Leakage

Data leakage occurs when your model inadvertently uses future information during training:

- **Label leakage**: Using the target value (or proxies for it) as features
- **Temporal leakage**: Using features computed from data that wouldn't be available at prediction time
- **Target encoding leakage**: Encoding categories using statistics that include the target

These issues cause models to perform well in development but fail in production.

#### Architecture Overview

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

#### Core Concepts

The temporal framework introduces three key columns to enforce PIT correctness:

| Column | Purpose |
|--------|---------|
| `feature_timestamp` | When features were observed (point-in-time for feature values) |
| `label_timestamp` | When the label became known (when ground truth was revealed) |
| `label_available_flag` | Whether the label is available for training (respects observation window) |

**Key Rule**: Features from time T can only be used to predict labels from time T+observation_window.

#### Timestamp Scenarios

The framework auto-detects your data's timestamp scenario:

| Scenario | Description | Example |
|----------|-------------|---------|
| **Production** | Explicit feature and label timestamps in data | CRM with `last_activity_date` and `churn_date` |
| **Production (Derived)** | Timestamps derivable from existing columns | `signup_date` + 90 days = label availability |
| **Derived** | Feature timestamps can be computed from date columns | Parse dates from `last_login`, `account_created` |
| **Synthetic** | No timestamps available (Kaggle-style data) | Static customer snapshot with no dates |

```python
from customer_retention.temporal import ScenarioDetector

# Auto-detect scenario
detector = ScenarioDetector()
scenario, config, discovery = detector.detect(df, target_column="churn")

print(f"Scenario: {scenario}")
# → "production" | "partial" | "derived" | "synthetic"

print(f"Strategy: {config.strategy.value}")
# → "production" | "derived" | "synthetic_fixed"
```

#### UnifiedDataPreparer: Single Entry Point

The `UnifiedDataPreparer` handles all timestamp scenarios with a unified API:

```python
from customer_retention.temporal import UnifiedDataPreparer, ScenarioDetector

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

#### Versioned Training Snapshots

The `SnapshotManager` creates versioned, integrity-checked training snapshots:

```python
from customer_retention.temporal import SnapshotManager
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

#### Leakage Detection

The `LeakageDetector` validates your prepared data using multiple probes:

```python
from customer_retention.temporal import LeakageDetector

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

#### Migration Script

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

#### Why This Matters

| Without Temporal Framework | With Temporal Framework |
|---------------------------|------------------------|
| Models trained on all data | Only label_available=True records used |
| No timestamp validation | feature_timestamp < label_timestamp enforced |
| Data can be modified silently | SHA256 integrity verification on load |
| Single training dataset | Versioned snapshots with comparison |
| Leakage discovered in production | Leakage detected during validation |

> **📐 Architecture Diagram**: For a comprehensive visual overview of the temporal framework, including data flow diagrams, module architecture, and API reference, see [`docs/architecture_temporal.md`](docs/architecture_temporal.md).

### Feature Store Integration

The framework includes a **unified feature store** module that provides point-in-time correct feature management across both local development (Feast) and production (Databricks Feature Engineering).

#### Feature Store Architecture

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

#### Core Components

| Component | Purpose |
|-----------|---------|
| `TemporalFeatureDefinition` | Define features with temporal metadata (timestamp column, computation type, leakage risk) |
| `FeatureRegistry` | Collection of feature definitions, serializable to JSON |
| `FeatureStoreManager` | Unified API for publishing and retrieving features |
| `FeastBackend` | Local development using Feast (Parquet + SQLite) |
| `DatabricksBackend` | Production using Databricks Feature Engineering (Unity Catalog) |

#### Usage Example

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

#### Notebook Integration

The generated notebook stage `11_feature_store.ipynb` demonstrates the complete workflow:

1. Load gold layer data with temporal columns
2. Define feature registry from numeric columns
3. Publish features to the feature store
4. Create point-in-time correct training sets
5. Validate feature consistency

---

## Part 1: The Exploration Loop

The exploration loop is **iterative by design**. Each iteration is versioned, and feedback from model training informs the next round of feature engineering.

### Exploration Workflow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EXPLORATION LOOP                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ITERATION 1                                                                 │
│  ──────────                                                                  │
│  Notebooks 01-04: Explore each dataset                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                        │
│  │ customers.csv│  │  events.csv  │  │ products.csv │                        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                        │
│         │                 │                 │                                │
│         ▼                 ▼                 ▼                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                        │
│  │  findings    │  │  findings    │  │  findings    │  ← Versioned YAML      │
│  │  (YAML)      │  │  (YAML)      │  │  (YAML)      │    with iteration_id   │
│  └──────────────┘  └──────────────┘  └──────────────┘                        │
│                                                                              │
│  Notebook 05: Multi-Dataset Discovery & Selection                            │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  • Discover all explored datasets                  │                      │
│  │  • SELECT which datasets to include               │  ← USER CHOICE        │
│  │  • Define relationships (join keys)               │                       │
│  │  • Save multi_dataset_findings.yaml               │                       │
│  └───────────────────────────────────────────────────┘                       │
│                                                                              │
│  Notebook 06: Capture Recommendations                                        │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  • Bronze: null handling, outlier capping         │                       │
│  │  • Silver: joins, aggregations                    │                       │
│  │  • Gold: encoding, scaling, transformations       │                       │
│  │  • Track: applied vs skipped recommendations      │  ← TRACKED            │
│  └───────────────────────────────────────────────────┘                       │
│                                                                              │
│  Notebooks 07-08: Train & Evaluate                                           │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  • Train baseline models                          │                       │
│  │  • Evaluate metrics (AUC, precision, recall)      │                       │
│  │  • Collect feature importances                    │  ← FEEDBACK           │
│  │  • Analyze prediction errors                      │                       │
│  └───────────────────────────────────────────────────┘                       │
│         │                                                                    │
│         ▼                                                                    │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  Satisfied with results?                          │                       │
│  │                                                   │                       │
│  │  NO → Start ITERATION 2                           │                       │
│  │       • Low-importance features identified        │                       │
│  │       • New feature ideas from error analysis     │                       │
│  │       • Recommendations refined                   │                       │
│  │                                                   │                       │
│  │  YES → Proceed to Production Execution            │                       │
│  └───────────────────────────────────────────────────┘                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Source Pipeline Structure

When you have multiple data sources, the framework generates **parallel bronze notebooks** that merge in silver:

```
landing/
├── customers.csv      ─┐
├── events.csv         ─┼─ Parallel Bronze processing
└── products.csv       ─┘
         │
         ▼
bronze/
├── bronze_customers   ─┐
├── bronze_events      ─┼─ Independent notebooks (can run in parallel)
└── bronze_products    ─┘
         │
         ▼
silver/
└── silver_merged      ─── Joins all bronze outputs (runs after all bronze complete)
         │
         ▼
gold/
└── gold_features      ─── ML-ready dataset (final step)
```

### Dataset Selection (Notebook 05)

Select which datasets to include in your pipeline. Choices are **persisted** and respected by all exports:

```python
from customer_retention.auto_explorer import ExplorationManager, MultiDatasetFindings

# Discover all explored datasets
manager = ExplorationManager(explorations_dir="./experiments/findings")
datasets = manager.list_datasets()
# → [customers, events, products, support_tickets]

# Create multi-dataset view
multi = manager.create_multi_dataset_findings()

# SELECT which datasets to include (persisted!)
multi.exclude_dataset("support_tickets")  # Remove from pipeline
multi.select_dataset("support_tickets")   # Re-include if needed

# Define how datasets join
multi.add_relationship(
    left_dataset="customers",
    right_dataset="events",
    left_column="customer_id",
    right_column="customer_id",
    relationship_type="one_to_many"
)

# Save - selection is persisted!
multi.save("./experiments/findings/multi_dataset_findings.yaml")

# Later: only selected datasets are included
print(multi.selected_datasets)  # customers, events, products
```

### Capturing Recommendations (Notebook 06)

As you explore, capture cleaning and transformation recommendations using the layered builder:

```python
from customer_retention.auto_explorer import RecommendationBuilder

# Create builder from your exploration findings
builder = RecommendationBuilder(findings, notebook="06_modeling_readiness")

# Bronze layer: Data cleaning
builder.bronze() \
    .impute_nulls("age", strategy="median", reason="5% missing values") \
    .cap_outliers("revenue", method="iqr", reason="12% outliers detected") \
    .convert_type("signup_date", target_type="datetime", reason="String to date")

# Silver layer: Joins and aggregations
builder.silver() \
    .aggregate("revenue", aggregation="sum", windows=["7d", "30d"], reason="Revenue trends")

# Gold layer: Feature engineering
builder.gold() \
    .encode("contract_type", method="one_hot", reason="3 categories") \
    .scale("revenue", method="standard", reason="Normalize for model")

# Get the registry with all recommendations
registry = builder.build()
```

### Iteration Tracking

Model development is iterative. The `iteration` module tracks your progress across multiple cycles:

```python
from customer_retention.iteration import (
    IterationOrchestrator,
    IterationTrigger,
    IterationStatus,
    TrackedRecommendation,
    RecommendationType,
    ModelFeedback
)

# Initialize the orchestrator
orchestrator = IterationOrchestrator("./experiments/findings")

# Start a new iteration
ctx = orchestrator.start_new_iteration(IterationTrigger.INITIAL)
print(f"Iteration {ctx.iteration_number}: {ctx.iteration_id}")

# Track which recommendations you apply
rec = TrackedRecommendation(
    recommendation_id="clean_age_impute",
    recommendation_type=RecommendationType.CLEANING,
    source_column="age",
    action="impute_median",
    description="Impute missing age with median"
)
orchestrator.track_recommendation(rec)
orchestrator.apply_recommendation("clean_age_impute")

# After training, collect feedback
feedback = ModelFeedback(
    iteration_id=ctx.iteration_id,
    model_type="RandomForestClassifier",
    metrics={"roc_auc": 0.82, "pr_auc": 0.68},
    feature_importances={
        "age": 0.25,
        "income": 0.35,
        "tenure": 0.39,
        "unused_feature": 0.01  # Low importance!
    }
)
orchestrator.collect_feedback(feedback)

# Get refined recommendations for next iteration
refined = orchestrator.get_refined_recommendations(findings, feedback)
print(f"Features to drop: {refined['features_to_drop']}")
# → ['unused_feature']

# Start next iteration informed by feedback
ctx2 = orchestrator.start_child_iteration(IterationTrigger.MANUAL)
print(f"Iteration {ctx2.iteration_number} (parent: {ctx2.parent_iteration_id[:8]}...)")
```

**Iteration triggers:**
- `INITIAL` - First exploration
- `MANUAL` - User-initiated refinement
- `DRIFT_DETECTED` - Data drift signals
- `PERFORMANCE_DROP` - Model performance degradation
- `SCHEDULED` - Regular retraining schedule

**What gets tracked:**
- Applied vs skipped recommendations with outcomes
- Feature importance feedback from models
- Iteration lineage (parent → child relationships)
- Model metrics for comparison across versions

---

## Part 2: Production Execution

Once satisfied with your exploration results, generate pipelines for production execution. **Choose the track that matches your environment**—the exploration artifacts (findings, recommendations) are the same, but the execution path differs.

```
                        Exploration Loop Complete
                                  │
                                  │ Artifacts:
                                  │ • *_findings.yaml
                                  │ • recommendations
                                  │ • iteration history
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │    LOCAL TRACK        │   │   DATABRICKS TRACK    │
        │                       │   │                       │
        │  PipelineGenerator    │   │  DatabricksExporter   │
        │         +             │   │         +             │
        │  Feast Feature Store  │   │  Unity Catalog        │
        │         +             │   │         +             │
        │  MLFlow Tracking      │   │  Delta Lake           │
        │                       │   │         +             │
        │  → pandas execution   │   │  Workflows            │
        │  → local serving      │   │                       │
        └───────────────────────┘   │  → Spark execution    │
                                    │  → cluster serving    │
                                    └───────────────────────┘
```

---

### Track A: Local Execution (Feast + MLFlow)

The framework includes a **Pipeline Generator** that produces complete, runnable medallion architecture pipelines from your exploration findings. Generated pipelines integrate with **Feast** for feature store management and **MLFlow** for experiment tracking.

### End-to-End Workflow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE ML PIPELINE WORKFLOW                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STEP 1: EXPLORATION (Notebooks 01-07)                                           │
│  ───────────────────────────────────────                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                           │
│  │ customers   │    │   events    │    │  products   │   Raw data sources        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                           │
│         │                  │                  │                                  │
│         ▼                  ▼                  ▼                                  │
│  ┌─────────────────────────────────────────────────────┐                         │
│  │              DataExplorer (per source)              │                         │
│  │  • Auto-detect column types                         │                         │
│  │  • Quality assessment                               │                         │
│  │  • Cleaning recommendations                         │                         │
│  └─────────────────────────────────────────────────────┘                         │
│         │                  │                  │                                  │
│         ▼                  ▼                  ▼                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                          │
│  │ customers_   │   │  events_     │   │ products_    │   Exploration findings   │
│  │ findings.yaml│   │ findings.yaml│   │ findings.yaml│   (per source)           │
│  └──────────────┘   └──────────────┘   └──────────────┘                          │
│         │                  │                  │                                  │
│         └──────────────────┼──────────────────┘                                  │
│                            ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐                         │
│  │           ExplorationManager (Notebook 05)          │                         │
│  │  • Discover all explored datasets                   │                         │
│  │  • Define relationships (join keys)                 │                         │
│  │  • Select/exclude datasets                          │                         │
│  └─────────────────────────────────────────────────────┘                         │
│                            │                                                     │
│                            ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐                         │
│  │          multi_dataset_findings.yaml                │   Combined findings     │
│  └─────────────────────────────────────────────────────┘                         │
│                                                                                  │
│  STEP 2: PIPELINE GENERATION                                                     │
│  ───────────────────────────                                                     │
│                            │                                                     │
│                            ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐                         │
│  │              PipelineGenerator                      │                         │
│  │  • Parse exploration findings                       │                         │
│  │  • Build pipeline configuration                     │                         │
│  │  • Render code from templates                       │                         │
│  └─────────────────────────────────────────────────────┘                         │
│                            │                                                     │
│                            ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐                         │
│  │           GENERATED PIPELINE                        │                         │
│  │  orchestration/{pipeline_name}/                     │                         │
│  │  ├── config.py           ← Pipeline configuration   │                         │
│  │  ├── bronze/                                        │                         │
│  │  │   ├── bronze_customers.py  ← Parallel execution  │                         │
│  │  │   └── bronze_events.py                           │                         │
│  │  ├── silver/                                        │                         │
│  │  │   └── silver_merge.py      ← Join bronze outputs │                         │
│  │  ├── gold/                                          │                         │
│  │  │   └── gold_features.py     ← Feature engineering │                         │
│  │  ├── training/                                      │                         │
│  │  │   └── ml_experiment.py     ← MLFlow integration  │                         │
│  │  └── pipeline_runner.py       ← Local orchestrator  │                         │
│  └─────────────────────────────────────────────────────┘                         │
│                                                                                  │
│  STEP 3: LOCAL EXECUTION WITH FEAST + MLFLOW                                     │
│  ───────────────────────────────────────────                                     │
│                            │                                                     │
│         ┌──────────────────┼──────────────────┐                                  │
│         ▼                  ▼                  ▼                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                           │
│  │   FEAST     │    │   MLFlow    │    │  Pipeline   │                           │
│  │ Feature     │    │ Experiment  │    │  Execution  │                           │
│  │   Store     │    │  Tracking   │    │  (pandas)   │                           │
│  └─────────────┘    └─────────────┘    └─────────────┘                           │
│         │                  │                  │                                  │
│         ▼                  ▼                  ▼                                  │
│  • Feature views    • Metrics logged   • Bronze → Silver → Gold                  │
│  • Online serving   • Model artifacts  • Parallel bronze execution               │
│  • Point-in-time    • Experiment runs  • Local parquet output                    │
│    joins            • Model registry                                             │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Quick Start: Generate a Pipeline

```python
from customer_retention.pipeline_generator import PipelineGenerator

# After running exploration notebooks and saving findings...
generator = PipelineGenerator(
    findings_dir="./experiments/findings",           # Where your findings are saved
    output_dir="./orchestration",            # Where to generate pipeline
    pipeline_name="churn_prediction"         # Name of your pipeline
)

# Generate all pipeline files
generated_files = generator.generate()

print(f"Generated {len(generated_files)} files:")
for f in generated_files:
    print(f"  - {f}")
```

**Generated output:**
```
orchestration/churn_prediction/
├── config.py                    # Pipeline configuration
├── bronze/
│   ├── bronze_customers.py      # Customer data cleaning
│   └── bronze_orders.py         # Orders data cleaning
├── silver/
│   └── silver_merge.py          # Join customers + orders
├── gold/
│   └── gold_features.py         # Feature engineering
├── training/
│   └── ml_experiment.py         # Model training with MLFlow
├── pipeline_runner.py           # Run the complete pipeline locally
└── workflow.json                # (optional) Databricks workflow - see Track B
```

### Feature Store Integration with Feast

The generated pipelines integrate with Feast for feature management. Feast provides:

- **Feature versioning** - Track changes to features over time
- **Point-in-time joins** - Get historical feature values for training
- **Online serving** - Low-latency feature retrieval for inference
- **Feature reuse** - Share features across multiple models

#### Local Setup with Feast

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

#### Using the Feature Store Adapter

```python
from customer_retention.adapters.feature_store import (
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

### MLFlow Integration for Experiment Tracking

Generated pipelines include MLFlow integration for tracking experiments, logging metrics, and managing models.

#### Setting Up MLFlow

```bash
# Install MLFlow
pip install mlflow

# Start local tracking server (optional)
mlflow ui --port 5000
```

#### Using the MLFlow Adapter

```python
from customer_retention.adapters.mlflow import get_mlflow, ExperimentTracker

# Auto-detect environment
mlflow_client = get_mlflow(tracking_uri="./experiments/mlruns")

# High-level experiment tracker
tracker = ExperimentTracker(mlflow_client)

# Log exploration findings
from customer_retention.auto_explorer import ExplorationFindings
findings = ExplorationFindings.load("./experiments/findings/customers_findings.yaml")
tracker.log_exploration(findings, experiment_name="churn_exploration")

# Log pipeline execution
tracker.log_pipeline_execution(
    pipeline_name="churn_prediction",
    stage="gold",
    metrics={"rows_processed": 10000, "features_created": 25}
)

# Search for best runs
best_runs = tracker.get_best_run(
    experiment_name="churn_prediction",
    metric="auc",
    ascending=False
)
```

#### Generated Training Code with MLFlow

The `training/ml_experiment.py` generated by the pipeline includes MLFlow integration:

```python
# Generated code in training/ml_experiment.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from config import get_gold_path, TARGET_COLUMN

def load_gold() -> pd.DataFrame:
    return pd.read_parquet(get_gold_path())

def run_experiment():
    gold = load_gold()
    X = gold.drop(columns=[TARGET_COLUMN])
    y = gold[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics logged automatically when run through pipeline_runner.py
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")

    return model

if __name__ == "__main__":
    run_experiment()
```

### Running the Pipeline Locally

```bash
# Navigate to generated pipeline
cd orchestration/churn_prediction

# Run the complete pipeline
python pipeline_runner.py
```

Or programmatically:

```python
# Import and run from the generated pipeline
import sys
sys.path.insert(0, "orchestration/churn_prediction")

from pipeline_runner import run_pipeline
run_pipeline()
```

**Output:**
```
Starting pipeline: churn_prediction
Bronze complete
Silver complete
Gold complete
Training complete
AUC: 0.8234
Precision: 0.7456
Recall: 0.6892
```

### Complete Local Track Example

Here's the complete workflow from raw data to production pipeline:

```python
# ============================================
# STEP 1: Explore your data (Notebooks 01-04)
# ============================================
from customer_retention.auto_explorer import DataExplorer

# Explore each dataset
explorer = DataExplorer(save_findings=True, output_dir="./experiments/findings")

# Explore customers (entity-level)
customers_findings = explorer.explore("data/customers.csv")
print(f"Customers: {customers_findings.row_count} rows, target={customers_findings.target_column}")

# Explore orders (event-level)
orders_findings = explorer.explore("data/orders.parquet")
print(f"Orders: {orders_findings.row_count} rows, is_event_level={orders_findings.is_time_series}")

# ============================================
# STEP 2: Combine datasets (Notebook 05)
# ============================================
from customer_retention.auto_explorer import ExplorationManager

manager = ExplorationManager(explorations_dir="./experiments/findings")

# Discover all explored datasets
datasets = manager.list_datasets()
print(f"Found datasets: {datasets}")

# Create multi-dataset findings
multi = manager.create_multi_dataset_findings()

# Define relationships
multi.add_relationship(
    left_dataset="customers",
    right_dataset="orders",
    left_column="customer_id",
    right_column="customer_id",
    relationship_type="one_to_many"
)

# Save multi-dataset findings
multi.save("./experiments/findings/multi_dataset_findings.yaml")

# ============================================
# STEP 3: Generate pipeline
# ============================================
from customer_retention.pipeline_generator import PipelineGenerator

generator = PipelineGenerator(
    findings_dir="./experiments/findings",
    output_dir="./orchestration",
    pipeline_name="churn_prediction"
)

files = generator.generate()
print(f"Generated {len(files)} pipeline files")

# ============================================
# STEP 4: Run pipeline locally
# ============================================
import sys
sys.path.insert(0, "./orchestration/churn_prediction")
from pipeline_runner import run_pipeline

run_pipeline()

# ============================================
# STEP 5: Register features with Feast
# ============================================
from customer_retention.adapters.feature_store import get_feature_store, FeatureViewConfig
import pandas as pd

feature_store = get_feature_store(repo_path="./experiments/feature_store/feature_repo")

gold_df = pd.read_parquet("./orchestration/churn_prediction/data/gold/features.parquet")

config = FeatureViewConfig(
    name="churn_features",
    entity_key="customer_id",
    features=[c for c in gold_df.columns if c != "churn"],
    ttl_days=30
)

feature_store.register_feature_view(config, gold_df)
print("Features registered with Feast!")

# ============================================
# STEP 6: Track experiment with MLFlow
# ============================================
from customer_retention.adapters.mlflow import ExperimentTracker, get_mlflow

mlflow_client = get_mlflow(tracking_uri="./experiments/mlruns")
tracker = ExperimentTracker(mlflow_client)

# Log the exploration
tracker.log_exploration(customers_findings, experiment_name="churn_analysis")

print("Experiment tracked with MLFlow!")
print("View at: http://localhost:5000 (run 'mlflow ui' first)")
```

### Best Practices (Local Track)

| Practice | Description |
|----------|-------------|
| **Version findings** | Commit `*_findings.yaml` files to git for reproducibility |
| **Review generated code** | Check generated transformations before running |
| **Test locally first** | Run `pipeline_runner.py` locally before deployment |
| **Use feature store** | Register gold features in Feast for reuse across models |
| **Track experiments** | Use MLFlow to compare model versions and hyperparameters |
| **Parameterize paths** | Update `config.py` paths for your environment |

---

### Scoring Pipeline: Production Simulation

The framework provides two approaches for validating model performance before deployment:

#### Recommended: Temporal Snapshots (Leakage-Safe)

The **temporal framework** (see [Temporal Framework: Leakage-Safe Architecture](#temporal-framework-leakage-safe-architecture)) is the recommended approach for production validation. It enforces point-in-time correctness by:

1. **Automatic timestamp handling** - Detects or synthesizes feature/label timestamps
2. **Versioned snapshots** - Creates reproducible training data with integrity hashing
3. **Leakage detection** - Multi-probe validation catches temporal leakage
4. **Cutoff-based splits** - Train on data before cutoff, validate on data after

```python
from customer_retention.temporal import (
    ScenarioDetector, UnifiedDataPreparer, SnapshotManager
)
from datetime import datetime

# Detect and prepare
detector = ScenarioDetector()
scenario, config, _ = detector.detect(df, "churn")
preparer = UnifiedDataPreparer(output_path, config)
prepared_df = preparer.prepare_from_raw(df, "churn", "customer_id")

# Create training snapshot at cutoff
manager = SnapshotManager(output_path)
train_meta = manager.create_snapshot(
    prepared_df,
    cutoff_date=datetime(2024, 1, 1),
    target_column="churn"
)

# Records after cutoff have label_available_flag=False
# Use these for validation once labels become available
```

#### Legacy: Holdout Masking (Deprecated)

> **Note**: The holdout masking approach described below is deprecated in favor of temporal snapshots. It remains available for backwards compatibility but is not recommended for new projects.

The holdout approach simulates production by masking target values for a subset of records. This was useful for Kaggle-style datasets but has limitations:

- No temporal ordering guarantee
- Relies on manual column exclusion
- No integrity verification
- Cannot detect temporal leakage

For reference, the holdout approach worked by:
1. Holding out a stratified subset of labeled records
2. Masking their target values (simulating "unknown" production data)
3. Running inference as if these were new production records
4. Validating predictions against preserved ground truth

---

### Track B: Databricks Execution (Unity Catalog + Delta Lake)

For production Spark environments, export standalone PySpark notebooks that run on Databricks with **no framework dependency**.

#### Generate Standalone Notebooks

```python
from customer_retention.orchestration import DatabricksExporter
from customer_retention.auto_explorer import MultiDatasetFindings

# Load your exploration artifacts
multi = MultiDatasetFindings.load("./experiments/findings/multi_dataset_findings.yaml")
registry = multi.to_recommendation_registry()

# Configure exporter for Unity Catalog
exporter = DatabricksExporter(
    registry,
    catalog="your_catalog",    # Unity Catalog
    schema="churn_pipeline"
)

# Get complete notebook structure
structure = exporter.export_notebook_structure()

# Save notebooks to files
for source_name, code in structure["bronze"].items():
    with open(f"notebooks/bronze_{source_name}.py", "w") as f:
        f.write(code)

with open("notebooks/silver_merge.py", "w") as f:
    f.write(structure["silver"])

with open("notebooks/gold_features.py", "w") as f:
    f.write(structure["gold"])
```

**Generated code is standalone** - uses only:
- `pyspark.sql.functions` (F.col, F.when, F.mean, etc.)
- `pyspark.sql.window` (Window.partitionBy)
- `pyspark.ml.feature` (StringIndexer, OneHotEncoder, StandardScaler)
- Delta Lake writes (`df.write.format("delta").saveAsTable()`)

**No framework dependency** - the notebooks run standalone on any Databricks cluster.

#### Alternative: Generate Pipeline Specification

For use with Databricks Assistant or other AI tools, generate a markdown specification:

```python
from customer_retention.orchestration import PipelineDocGenerator

doc_gen = PipelineDocGenerator(registry, findings)
spec = doc_gen.generate()

# Copy to Databricks Assistant or save as reference
with open("pipeline_spec.md", "w") as f:
    f.write(spec)
```

The spec includes implementation hints, column statistics, and execution order.

#### Databricks Workflow Definition

Import the generated `workflow.json` into Databricks Workflows:

```python
import json
with open("orchestration/churn_prediction/workflow.json") as f:
    workflow = json.load(f)

# Use Databricks CLI or API to create the workflow
# databricks jobs create --json @workflow.json
```

**Generated workflow structure:**
```json
{
  "name": "churn_prediction_pipeline",
  "tasks": [
    {"task_key": "bronze_customers", "notebook_task": {...}},
    {"task_key": "bronze_orders", "notebook_task": {...}},
    {"task_key": "silver_merge", "depends_on": ["bronze_customers", "bronze_orders"], ...},
    {"task_key": "gold_features", "depends_on": ["silver_merge"], ...},
    {"task_key": "ml_experiment", "depends_on": ["gold_features"], ...}
  ]
}
```

#### Unity Catalog Feature Engineering

On Databricks, feature store integration uses Databricks Feature Engineering:

```python
from customer_retention.adapters.feature_store import get_feature_store

# Automatically uses DatabricksFeatureStoreAdapter when on Databricks
feature_store = get_feature_store(catalog="main", schema="features")

# Same API works - but uses Unity Catalog under the hood
feature_store.register_feature_view(config, spark_df)
# Creates table: main.features.customer_features
```

#### Deployment Checklist

```bash
# 1. Build wheel
uv build

# 2. Upload to Unity Catalog Volume
databricks fs cp dist/customer_retention-*.whl \
    dbfs:/Volumes/catalog/schema/wheels/

# 3. Install on cluster (in notebook)
%pip install /Volumes/catalog/schema/wheels/customer_retention-*.whl

# 4. Or use generated standalone notebooks (no install needed)
```

### Best Practices (Databricks Track)

| Practice | Description |
|----------|-------------|
| **Use standalone notebooks** | Generated PySpark code has no framework dependency |
| **Unity Catalog tables** | Write to Delta tables for data governance |
| **Workflow orchestration** | Use the generated `workflow.json` for job scheduling |
| **Environment detection** | The framework auto-detects Databricks vs local |
| **Test locally first** | Validate with local track before Databricks deployment |

---

## Get Started

### Installation

```bash
# Clone the repository
git clone https://github.com/aladjov/CR.git
cd customer-retention

# Install (uv recommended for speed)
pip install uv
uv pip install -e ".[dev,ml]"

# Or with pip
pip install -e ".[dev,ml]"
```

### Quick Example

```python
from customer_retention.auto_explorer import DataExplorer

# Point to your data and start exploring
explorer = DataExplorer(visualize=True, save_findings=True)
findings = explorer.explore("your_customer_data.csv")

# What you get:
# - Automatic column type detection
# - Data quality report
# - Target variable identification
# - Initial feature recommendations
```

### Working with Multiple Datasets

Real churn analysis often involves multiple data sources. The framework auto-detects relationships:

```python
from customer_retention.auto_explorer import ExplorationManager

# After exploring each dataset individually...
manager = ExplorationManager(explorations_dir="./experiments/findings")

# Discover all explored datasets
datasets = manager.list_datasets()
# Shows: customers (entity-level), transactions (event-level), support_tickets (event-level)

# Create multi-dataset analysis
multi = manager.create_multi_dataset_findings()

# Auto-detected:
# - Primary entity: customers (has target column)
# - Event datasets: transactions, support_tickets
# - Join keys: customer_id
```

### Time-Window Aggregations

For event-level data, aggregate to entity level with configurable windows:

```python
from customer_retention.profiling import TimeWindowAggregator

aggregator = TimeWindowAggregator(
    entity_column="customer_id",
    time_column="transaction_date"
)

# Create features across time windows
features = aggregator.aggregate(
    transactions_df,
    windows=["7d", "30d", "90d"],
    value_columns=["amount"],
    agg_funcs=["sum", "mean", "count"],
    include_recency=True  # days_since_last_event
)

# Result: One row per customer with features like:
# - amount_sum_7d, amount_sum_30d, amount_sum_90d
# - amount_mean_7d, amount_mean_30d, amount_mean_90d
# - event_count_7d, event_count_30d, event_count_90d
# - days_since_last_event
```

### Start with the Notebooks

The best way to learn is to work through the exploration notebooks with your own data:

```bash
# Bootstrap a new project with all notebooks
python scripts/notebooks/init_project.py --output ./my_churn_project --name "My Churn Analysis"

# Or open the exploration notebooks directly
jupyter lab exploration_notebooks/01_data_discovery.ipynb
```

---

## Example: What You'll Build

Here's a realistic example of what you can achieve with the current framework:

### Scenario

You have customer data with usage metrics, support interactions, and billing history. You want to predict which customers might churn.

### What the Framework Helps You Do

**1. Explore and understand** (Notebooks 01-04, or 01 + 01a-01d + 02-04 for event data)
```python
# Automatic profiling finds:
# - 12% missing values in 'last_login_date'
# - 'customer_id' detected as identifier (excluded from features)
# - 'contract_end_date' flagged as potential leakage risk
# - Strong correlation between support_tickets and churn
```

**2. Clean and prepare** (Notebooks 05-06)
```python
from customer_retention.cleaning import MissingValueHandler, OutlierHandler

# Handle missing values with appropriate strategies
handler = MissingValueHandler()
df_clean = handler.impute(df, strategy='median', columns=['tenure_months'])

# Cap outliers using IQR method
outlier_handler = OutlierHandler(method='iqr')
df_clean = outlier_handler.transform(df_clean, columns=['monthly_charges'])
```

**3. Train baseline models** (Notebook 08)
```python
from customer_retention.modeling import BaselineTrainer

trainer = BaselineTrainer()
results = trainer.train_all(X_train, y_train)

# Compare: Logistic Regression, Random Forest, XGBoost, LightGBM
# Get baseline metrics to beat
```

**4. Understand predictions** (Notebook 09)
```python
from customer_retention.interpretability import ShapExplainer

explainer = ShapExplainer(model)
shap_values = explainer.explain(X_test)

# See which features drive predictions
# Identify customer segments with different risk factors
```

---

## Project Structure

```
customer-retention/
├── src/customer_retention/      # Core library
│   ├── analysis/                # Data analysis components
│   │   ├── auto_explorer/       # Automatic data exploration
│   │   ├── business/            # Business logic (ROI, risk scoring)
│   │   ├── diagnostics/         # Model diagnostics
│   │   ├── discovery/           # Data discovery utilities
│   │   ├── interpretability/    # Model explanations (SHAP)
│   │   ├── recommendations/     # Recommendation pipeline
│   │   └── visualization/       # Chart building and display
│   ├── stages/                  # Pipeline stages
│   │   ├── temporal/            # Leakage-safe temporal framework
│   │   ├── profiling/           # Data profiling & quality checks
│   │   ├── transformation/      # Feature transformation
│   │   └── validation/          # Quality gates
│   ├── generators/              # Code generation
│   │   └── notebook_generator/  # Notebook generation for pipelines
│   ├── core/                    # Core abstractions
│   │   ├── config/              # Pipeline configuration
│   │   ├── components/          # Component registry
│   │   └── compat/              # Environment detection
│   └── feature_store/           # Temporal-aware feature store
│
├── exploration_notebooks/       # Interactive exploration notebooks (version controlled)
│   ├── 00_start_here.ipynb
│   ├── 01_data_discovery.ipynb
│   ├── 01a_temporal_deep_dive.ipynb       # Event Bronze Track
│   ├── 01a_a_temporal_text_deep_dive.ipynb # Event Bronze: TEXT columns
│   ├── 01b_temporal_quality.ipynb         # Event Bronze Track
│   ├── 01c_temporal_patterns.ipynb        # Event Bronze Track
│   ├── 01d_event_aggregation.ipynb        # Event → Entity transformation
│   ├── 02_column_deep_dive.ipynb          # Entity Bronze Track
│   ├── 02a_text_columns_deep_dive.ipynb   # Entity Bronze: TEXT columns
│   ├── 03_quality_assessment.ipynb
│   ├── 04_relationship_analysis.ipynb
│   ├── 05_multi_dataset.ipynb
│   ├── 06_feature_opportunities.ipynb
│   ├── 07_modeling_readiness.ipynb
│   ├── 08_baseline_experiments.ipynb
│   ├── 09_business_alignment.ipynb
│   └── 10_spec_generation.ipynb
│
├── generated_pipelines/         # Auto-generated pipeline notebooks (version controlled)
│   ├── local/                   # Local platform notebooks
│   └── databricks/              # Databricks platform notebooks
│
├── experiments/                 # All experiment outputs (gitignored)
│   ├── findings/                # Exploration findings (YAML)
│   ├── data/                    # Pipeline outputs (bronze/silver/gold)
│   ├── mlruns/                  # MLflow experiment tracking
│   └── feature_store/           # Feast feature store
│
├── scripts/                     # Command-line utilities
│   ├── databricks/              # Databricks deployment scripts
│   │   ├── build_wheel.sh       # Build wheel for Databricks
│   │   ├── capture_runtime.py   # Capture DBR baseline packages
│   │   ├── dbr_init.sh          # Cluster init script
│   │   ├── generate_constraints.py # Generate pip constraints
│   │   └── notebook_setup.py    # Notebook-scoped dependencies
│   ├── data/                    # Data generation and migration
│   │   ├── create_snapshot.py   # Create versioned training snapshots
│   │   ├── migrate_to_temporal.py # Migrate to temporal format
│   │   ├── generate_retail_dataset.py # Generate test data
│   │   └── generate_test_data.py # Generate transaction/email data
│   └── notebooks/               # Notebook utilities
│       ├── init_project.py      # Bootstrap new projects
│       ├── test_notebooks.py    # Validate notebooks with papermill
│       └── export_tutorial_html.py # Export to HTML tutorials
│
├── tests/                       # Test suite (3947 tests, 83% coverage)
│   ├── .coverage                # Coverage data
│   └── htmlcov/                 # Coverage HTML reports
│
├── docs/                        # Architecture documentation
│   └── architecture_temporal.md # Temporal framework diagrams
│
└── pyproject.toml
```

---

## Installation Options

```bash
# Basic (exploration & profiling only)
pip install customer-retention

# With ML models (recommended)
pip install "customer-retention[ml]"
# Includes: scikit-learn, xgboost, lightgbm, shap, mlflow, imbalanced-learn

# With text processing (for TEXT column embeddings)
pip install "customer-retention[text]"
# Includes: sentence-transformers

# Full installation (ML + text processing)
pip install "customer-retention[ml,text]"

# Development (includes testing tools)
pip install "customer-retention[dev,ml]"
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/customer_retention --cov-report=term-missing

# Quick smoke test
pytest tests/auto_explorer/ -v
```

---

## Databricks Support

The framework works locally and on Databricks. See [Track B: Databricks Execution](#track-b-databricks-execution-unity-catalog--delta-lake) for full details.

| Environment | Notes |
|-------------|-------|
| **Local (Track A)** | Feast feature store + MLFlow experiment tracking |
| **Databricks (Track B)** | Unity Catalog + Delta Lake + standalone PySpark notebooks |

Quick deployment:
```bash
# Build and upload wheel
uv build && databricks fs cp dist/customer_retention-*.whl dbfs:/Volumes/catalog/schema/wheels/

# Or use generated standalone notebooks (no install needed)
```

---

## Contributing

This project grows through community contributions. Whether you're:

- Adding new techniques
- Improving documentation
- Fixing bugs
- Sharing your use cases

We'd love your help. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas Where Help is Needed

- [ ] Advanced feature engineering recipes
- [ ] Causal inference integration
- [ ] Deep learning models for tabular data
- [ ] More comprehensive drift detection
- [ ] Documentation and tutorials

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on excellent open-source foundations:
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/) / [LightGBM](https://lightgbm.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [MLflow](https://mlflow.org/)
- [Pandas](https://pandas.pydata.org/)
