# Architecture Overview

The framework implements a **medallion architecture** with clear separation between the **iterative exploration loop** and **production execution**. Model development is iterative—you explore, train, evaluate, and refine based on feedback.

## High-Level Architecture

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

## Medallion Layers

| Layer | Scope | Notebooks | Description |
|-------|-------|-----------|-------------|
| **Landing** | Raw ingestion | 00, 01 | Prerequisites, data discovery |
| **Bronze-Event** | Clean event-shape data | 01a, 01a_a*, 01b, 01c, 01d | Temporal analysis, text processing → aggregation |
| **Bronze-Entity** | Clean entity-shape data | 02, 02a*, 03, 04 | Column/quality/relationship analysis, text processing |
| **Silver** | Join entity sources | 05 | Multi-dataset joins |
| **Gold** | ML-ready features | 06+ | Feature engineering, modeling |

\* Conditional notebooks that only run when TEXT columns are detected

## Bronze Per Shape: Event vs Entity Tracks

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

## Time Window Aggregations

The Event Bronze track helps you plan and execute aggregations to convert event-level data to entity-level features:

| Window | Use Case |
|--------|----------|
| 24h | Very recent activity, real-time signals |
| 7d | Weekly patterns, short-term engagement |
| 30d | Monthly patterns, subscription cycles |
| 90d | Quarterly trends, seasonal behavior |
| 180d | Semi-annual patterns, medium-term trends |
| 365d | Annual patterns, year-over-year comparison |
| all_time | Historical totals, lifetime value |

## Delta Lake: Universal Storage Layer

Every medallion layer writes **Delta Lake tables** — both during exploration and in production. ACID transactions, time travel, and version tracking come for free at every tier.

### Storage Abstraction

The `DeltaStorage` abstract base class (`integrations/adapters/storage/base.py`) defines the interface. Two implementations exist:

| Implementation | Backend | Use Case |
|----------------|---------|----------|
| `LocalDelta` | delta-rs (Rust) | Local development, CI, notebooks |
| `DatabricksDelta` | PySpark | Databricks production clusters |

```python
from customer_retention.integrations.adapters.storage.base import DeltaStorage

class DeltaStorage(ABC):
    def read(self, path: str, version: Optional[int] = None) -> pd.DataFrame: ...
    def write(self, df, path, mode="overwrite", partition_by=None, metadata=None): ...
    def merge(self, df, path, condition, update_cols=None): ...
    def history(self, path) -> List[Dict]: ...
    def vacuum(self, path, retention_hours=168): ...
    def exists(self, path) -> bool: ...
```

### Key Design Points

- **`PRODUCTION_DIR` defaults to `EXPERIMENTS_DIR`** — experiments and production share the same Delta tables until you explicitly separate them.
- **`PipelineContext.build_commit_metadata()`** attaches `run_id`, `pipeline_stage`, `run_type`, and `timestamp` to every Delta commit.
- **Version-pinned reads**: `storage.read(path, version=3)` gives you an exact historical snapshot for reproducibility or auditing.

## Transforms Framework

Stateful transformations (scaling, encoding, power transforms) are fitted once during training and replayed identically during scoring — preventing train/serve skew.

### Component Layout

| File | Role | Examples |
|------|------|---------|
| `transforms/ops.py` | Stateless functions | `apply_impute_null`, `apply_cap_outlier`, `apply_log_transform` |
| `transforms/fitted.py` | Stateful wrappers | `FittedScaler`, `FittedEncoder`, `FittedPowerTransform` |
| `transforms/executor.py` | Dispatch table | `TransformExecutor.apply()` routes `TransformationStep` → handler |
| `transforms/artifact_store.py` | Persistence | `ArtifactStore` saves/loads fitted objects + `manifest.yaml` |

### Fit Mode

```python
from customer_retention.transforms.executor import TransformExecutor
from customer_retention.transforms.artifact_store import ArtifactStore

executor = TransformExecutor()
store = ArtifactStore("./artifacts")

# Training: fit_mode=True — fits transformers and persists them
df_train = executor.apply_all(df, steps, fit_mode=True, artifact_store=store)
store.save_manifest()

# Scoring: fit_mode=False — loads persisted transformers and replays
store = ArtifactStore.from_manifest("./artifacts/manifest.yaml")
df_score = executor.apply_all(df, steps, fit_mode=False, artifact_store=store)
```

## Validation Gates

Three validators catch transformation inconsistencies before they reach production:

| Component | File | Purpose |
|-----------|------|---------|
| `ScoringPipelineValidator` | `scoring_pipeline_validator.py` | Compare training vs scoring feature distributions |
| `AdversarialScoringValidator` | `adversarial_scoring_validator.py` | Validate holdout entities get identical features |
| `PipelineValidationRunner` | `pipeline_validation_runner.py` | Orchestrate end-to-end validation |

### Severity Levels

| Level | Threshold | Meaning |
|-------|-----------|---------|
| LOW | < 1% relative diff | Minor numerical noise |
| MEDIUM | 1–5% | Worth investigating |
| HIGH | 5–10% | Likely a real discrepancy |
| CRITICAL | > 25% | Transformation is broken |

`compare_pipeline_outputs()` can compare two Delta table versions directly:

```python
from customer_retention.stages.validation.pipeline_validation_runner import compare_pipeline_outputs

report = compare_pipeline_outputs(
    training_output_path="./data/gold",
    version_a=3,   # training run
    version_b=5,   # scoring run
    entity_column="customer_id",
)
print(report.to_text())
```

## From Exploration to Production

The exploration notebooks generate artifacts that drive production pipelines:

```
Exploration Outputs                    Production Usage
───────────────────                    ────────────────
{dataset}_findings.yaml       →        Column types, target, data profile
multi_dataset_findings.yaml   →        Selected datasets, relationships
recommendations (Registry)    →        Bronze/Silver/Gold transformations
manifest.yaml                 →        Fitted transformer artifacts (scalers, encoders)
validation_report.yaml        →        Scoring validation results
delta_versions (Context)      →        Delta table version pins for reproducibility
```

These artifacts can be used in three ways:

1. **Local Execution** — `TransformExecutor` applies transformations with pandas
2. **Pipeline Generation** — `PipelineGenerator` creates runnable scripts with provenance comments
3. **Databricks Production** — `DatabricksExporter` exports standalone PySpark notebooks

## Project Structure

```
customer-retention/
├── src/customer_retention/      # Core library
│   ├── analysis/                # Data analysis components
│   │   ├── auto_explorer/       # Automatic data exploration
│   │   ├── business/            # Business logic (ROI, risk scoring)
│   │   ├── diagnostics/         # Model diagnostics
│   │   ├── interpretability/    # Model explanations (SHAP)
│   │   └── visualization/       # Chart building and display
│   ├── transforms/              # Fit/transform separation
│   │   ├── ops.py               # Stateless transform functions
│   │   ├── fitted.py            # Stateful wrappers (scaler, encoder, power)
│   │   ├── executor.py          # TransformExecutor dispatch table
│   │   └── artifact_store.py    # ArtifactStore + manifest.yaml persistence
│   ├── stages/                  # Pipeline stages
│   │   ├── temporal/            # Leakage-safe temporal framework
│   │   ├── profiling/           # Data profiling & quality checks
│   │   ├── transformation/      # Feature transformation
│   │   └── validation/          # Scoring pipeline & adversarial validators
│   ├── generators/              # Code generation
│   │   ├── orchestration/       # PipelineContext, DatabricksExporter
│   │   └── pipeline_generator/  # Pipeline code generation with provenance
│   ├── integrations/            # External system adapters
│   │   └── adapters/
│   │       ├── storage/         # DeltaStorage (base, local, databricks)
│   │       ├── feature_store/   # Feast & Databricks feature store
│   │       └── mlflow/          # MLFlow experiment tracking
│   ├── core/                    # Core abstractions and compat layer
│   └── feature_store/           # Temporal-aware feature store
│
├── exploration_notebooks/       # Interactive exploration notebooks
├── experiments/                 # All experiment outputs (gitignored)
│   └── artifacts/               # Fitted transformer artifacts
├── scripts/                     # Command-line utilities
└── tests/                       # Test suite
```

## Next Steps

- [[Exploration Loop]] - Deep dive into the notebook workflow
- [[Temporal Framework]] - Leakage-safe data preparation
- [[Local Track]] - Feast + MLFlow execution
- [[Databricks Track]] - Unity Catalog + Delta Lake execution
