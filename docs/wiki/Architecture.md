# Architecture Overview

The framework implements an **intent-driven medallion architecture**. Before any data is explored, you declare what you are trying to predict and how. That declaration -- the **model intent** -- propagates through every downstream notebook, controlling what temporal evidence is gathered, how snapshots are generated, how datasets merge, and how models are trained and validated.

Development happens in an **exploration loop** of interactive notebooks. When the model is satisfactory, a production pipeline is generated that faithfully replicates the exploration logic.

## High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EXPLORATION LOOP                                     │
│                                                                              │
│   00 ── Intent Contract                                                      │
│         (objective, posture, prediction horizon, control variables)           │
│                          │                                                   │
│            ┌─────────────┴─────────────┐                                     │
│            ▼                           ▼                                     │
│   01 ── Data Discovery          01 ── Data Discovery                         │
│   (per dataset)                 (per dataset)                                │
│            │                           │                                     │
│     ┌──────┴──────┐                    │                                     │
│     ▼             │                    │                                     │
│  BRONZE EVENT     │              BRONZE ENTITY                               │
│  01a Temporal     │              (entity-shape data                          │
│  01b Quality      │               passes through)                            │
│  01c Patterns     │                    │                                     │
│  01d Aggregation  │                    │                                     │
│     └──────┬──────┘                    │                                     │
│            ▼                           ▼                                     │
│   02 ── Source Integrity (per dataset, post-aggregation)                      │
│            │                                                                 │
│            ▼                                                                 │
│   03 ── Dataset Merge (entity_id + as_of_date join)                          │
│            │                                                                 │
│            ▼                                                                 │
│   04 ── Column Deep Dive (merged feature matrix)                             │
│   05 ── Relationship Analysis (correlations, redundancy, interactions)        │
│            │                                                                 │
│            ▼                                                                 │
│   06 ── Feature Opportunities (transformations, derived features)             │
│   07 ── Modeling Readiness (training grid, split policy, label horizon)       │
│            │                                                                 │
│            ▼                                                                 │
│   08 ── Baseline Experiments (entity-grouped temporal CV)                     │
│   09 ── Business Alignment (intervention timing, thresholds, ROI)             │
│            │                                                                 │
│            ▼                                                                 │
│   10 ── Pipeline Generation (local + Databricks)                             │
│   11 ── Scoring Validation (train/serve skew check)                          │
│   12 ── View Documentation (HTML export)                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Three Phases

The pipeline progresses in three major phases. All steps after Bronze operate per objective.

### Phase I -- Bronze: Evidence, Grid, Aggregation

Each event dataset is explored independently (01a-01c) and produces temporal evidence: window feasibility, density and coverage signals, recency/velocity/momentum dynamics, seasonality, and cadence feasibility per objective. Each dataset casts a **vote** on optimal cadence and anchor preference.

After all datasets complete exploration, votes are aggregated into a **consensus snapshot grid** per objective. This grid becomes the deterministic backbone of aggregation. Notebook 01d then aggregates each event dataset against that grid, producing `(entity_id, as_of_date)` snapshots.

Notebook 02 runs source-level integrity checks (duplicates, missingness, date logic, leakage smells) on each dataset *before* merge, so broken columns are eliminated early.

### Phase II -- Silver: Merge and Analytical Exploration

Notebook 03 merges all Bronze outputs on `(entity_id, as_of_date)` into a unified feature matrix using the `TemporalMerger`. Event-level datasets join on both keys; entity-level datasets either broadcast or use as-of joins depending on whether they have a feature timestamp.

Notebooks 04-05 run on the merged dataset: column deep dive (type validation, skewness, encoding hints) and relationship analysis (correlations, redundancy, feature-target associations, interaction opportunities).

### Phase III -- Gold: Modeling and Production

Notebooks 06-07 consolidate feature opportunities and formalize the training setup (grid selection, split policy, label horizon). Notebook 08 trains baseline models with **entity-grouped temporal cross-validation** (`TemporalEntitySplit`) to prevent entity information leakage.

Notebook 09 aligns model output with business objectives. Notebook 10 generates production pipeline code for both local and Databricks execution. Notebook 11 validates that scoring reproduces training features identically.

## Medallion Layers

| Layer | Scope | Notebooks | What Happens |
|-------|-------|-----------|--------------|
| **Intent** | Contract | 00 | Objective, posture, horizon, control variables, snapshot grid |
| **Landing** | Raw → Delta | 01 | Dataset fingerprinting, data discovery, Delta table creation |
| **Bronze-Event** | Temporal evidence | 01a-01d | Temporal deep dive, quality, patterns, aggregation |
| **Bronze-Entity** | Source cleanup | 02 | Integrity checks per dataset before merge |
| **Silver** | Unified features | 03, 04, 05 | Temporal merge, column analysis, relationship analysis |
| **Gold** | ML-ready | 06, 07 | Feature engineering, modeling readiness |
| **Training** | Experiments | 08 | Baseline models with entity-aware temporal CV |
| **Production** | Deployment | 09, 10, 11, 12 | Business alignment, pipeline generation, scoring validation |

## Notebook Catalog

| # | Name | Purpose |
|---|------|---------|
| -1 | Sample Datasets | Optional: generate sample data for tutorials |
| 00 | Start Here | Intent contract, dataset registration, objective detection |
| 01 | Data Discovery | Per-dataset profiling, entity/event routing |
| 01a | Temporal Deep Dive | Window feasibility, density, velocity, cadence votes |
| 01a_a | Temporal Text Deep Dive | Text column temporal analysis (conditional) |
| 01b | Temporal Quality | Temporal data quality assessment |
| 01c | Temporal Patterns | Seasonality, trends, regime detection |
| 01d | Event Aggregation | Aggregate events against consensus grid |
| 02 | Source Integrity | Per-dataset cleanup before merge |
| 03 | Dataset Merge | Temporal merge into unified feature matrix |
| 04 | Column Deep Dive | Column-level analysis on merged data |
| 04a | Text Columns Deep Dive | NLP analysis (conditional) |
| 05 | Relationship Analysis | Correlations, redundancy, interactions |
| 06 | Feature Opportunities | Transformation and encoding candidates |
| 07 | Modeling Readiness | Training grid, split policy, label horizon |
| 08 | Baseline Experiments | Model training with entity-grouped temporal CV |
| 09 | Business Alignment | Intervention strategies, risk thresholds, ROI |
| 10 | Spec Generation | Production pipeline code generation |
| 11 | Scoring Validation | Train/serve skew validation |
| 12 | View Documentation | Export HTML documentation |

## Core Concepts

### Intent Contract (Notebook 00)

Every run begins with a single modeling intent. See [[Model Intent and Objective Support]] for full details.

The intent includes:
- **Prediction objective**: immediate risk, renewal risk, or disengagement
- **Temporal posture**: reactive (short memory) or stable (long memory)
- **Prediction horizon**: how far ahead to predict (e.g., 30, 60, 90 days)
- **Anchor preference**: calendar, contract-end, or inactivity
- **Control variables**: observation window, purge gap, label window, cadence

These settings propagate to every downstream notebook. See [[Snapshot Grid and Control Variables]] for how the intent drives snapshot generation.

### Snapshot Grid

A deterministic set of `as_of_date` values that defines when point-in-time snapshots are taken. Derived from the intent contract, refined by dataset votes from temporal exploration. See [[Snapshot Grid and Control Variables]] for the full derivation logic.

### Objective Support Voting

Notebook 01 records objective support evidence (strength 0-3 per objective) during data discovery via `derive_objective_support()`. Notebooks 01a-01c contribute snapshot grid votes (cadence and date range evidence) during temporal exploration. These are aggregated into a synthesis that shows which objectives the data can support and where gaps exist. See [[Model Intent and Objective Support]].

### Entity-Aware Temporal Cross-Validation

Notebook 08 uses `TemporalEntitySplit` -- a custom sklearn-compatible CV splitter that wraps `GroupKFold` with optional temporal purging. All rows of an entity go entirely to train OR test in each fold, preventing entity information leakage. An optional purge gap removes training rows temporally close to the test period.

### Temporal Merger

Notebook 03 uses `TemporalMerger` to combine multiple datasets into a single training matrix. Three merge strategies apply automatically based on dataset shape:

| Dataset Shape | Strategy | Join Keys |
|---------------|----------|-----------|
| Event-level (aggregated) | Snapshot join | `entity_id` + `as_of_date` |
| Entity-level (no timestamp) | Broadcast | `entity_id` only (features repeat across dates) |
| Entity-level (with timestamp) | As-of join | `entity_id` + backward-looking temporal match |

## Delta Lake: Universal Storage Layer

Every medallion layer writes **Delta Lake tables** -- both during exploration and in production. ACID transactions, time travel, and version tracking are available at every tier.

### Storage Abstraction

The `DeltaStorage` abstract base class (`integrations/adapters/storage/base.py`) defines the interface:

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

- **`PRODUCTION_DIR` defaults to `EXPERIMENTS_DIR`** -- experiments and production share the same Delta tables until you explicitly separate them.
- **`PipelineContext.build_commit_metadata()`** attaches `run_id`, `pipeline_stage`, `run_type`, and `timestamp` to every Delta commit.
- **Version-pinned reads**: `storage.read(path, version=3)` gives you an exact historical snapshot for reproducibility.

## Run Namespace

Each exploration run gets an isolated directory layout under `RunNamespace`:

```
runs/
  {run_id}/
    project_context.yaml        # Intent + dataset registry
    snapshot_grid.yaml           # Locked grid dates
    landing/
      {dataset_name}/            # Delta: NB01 output (raw → Delta, column drops)
    bronze/
      {dataset_name}_aggregated/ # Delta: NB01d output (event aggregation)
    silver/
      silver_merged/             # Delta: NB03 output (temporal merge)
    datasets/
      {dataset_name}/
        findings/                # {name}_findings.yaml
        objective_support.yaml   # Per-dataset voting evidence
        docs/                    # Generated documentation
    merged/
      multi_dataset_findings.yaml
      recommendations.yaml
    session/
      users/
        {username}.json          # Per-user session state
```

`SessionState` tracks the active dataset and last notebook per user. Resolution priority: `CR_DATASET_ID` env var > session state > first available dataset.

## Transforms Framework

Stateful transformations (scaling, encoding, power transforms) are fitted once during training and replayed identically during scoring -- preventing train/serve skew.

| File | Role | Examples |
|------|------|---------|
| `transforms/ops.py` | Stateless functions | `apply_impute_null`, `apply_cap_outlier`, `apply_log_transform` |
| `transforms/fitted.py` | Stateful wrappers | `FittedScaler`, `FittedEncoder`, `FittedPowerTransform` |
| `transforms/executor.py` | Dispatch table | `TransformExecutor.apply()` routes `TransformationStep` to handler |
| `transforms/artifact_store.py` | Persistence | `ArtifactStore` saves/loads fitted objects + `manifest.yaml` |

```python
# Training: fit_mode=True -- fits transformers and persists them
df_train = executor.apply_all(df, steps, fit_mode=True, artifact_store=store)

# Scoring: fit_mode=False -- loads persisted transformers and replays
df_score = executor.apply_all(df, steps, fit_mode=False, artifact_store=store)
```

## Validation Gates

Three validators catch transformation inconsistencies before they reach production:

| Component | Purpose |
|-----------|---------|
| `ScoringPipelineValidator` | Compare training vs scoring feature distributions |
| `AdversarialScoringValidator` | Validate holdout entities get identical features |
| `PipelineValidationRunner` | Orchestrate end-to-end validation |

| Severity | Threshold | Meaning |
|----------|-----------|---------|
| LOW | < 1% relative diff | Minor numerical noise |
| MEDIUM | 1-5% | Worth investigating |
| HIGH | 5-10% | Likely a real discrepancy |
| CRITICAL | > 25% | Transformation is broken |

## Production Pipeline Generation

Notebook 10 generates deterministic production pipelines in two tracks:

| Track | Generator | Output |
|-------|-----------|--------|
| **Local** | `PipelineGenerator` | Python scripts with Delta Lake + pandas |
| **Databricks** | `DatabricksPipelineGenerator` | PySpark notebooks with Unity Catalog |

Both generators read from `FindingsParser` (with optional `RunNamespace` integration) and produce parallel Bronze notebooks that merge in Silver, then flow through Gold. The Databricks track uses `dbutils.notebook.run()` for orchestration and `format("delta").saveAsTable()` for storage.

### Naming Convention

- **Composite Name (CN)**: `{readable_prefix}__{7char_hash}` derived from sorted source names
- **Readable prefix**: first 4 characters of each word, lowercase, joined with `_`
- **Stage naming**: `bronze_entity_{source}`, `silver_featureset_{CN}`, `gold_features_{CN}`

## From Exploration to Production

```
Exploration Outputs                    Production Usage
-------------------                    ----------------
project_context.yaml          ->        Intent, datasets, objectives
snapshot_grid.yaml            ->        Deterministic as-of dates
{dataset}_findings.yaml       ->        Column types, target, data profile
multi_dataset_findings.yaml   ->        Selected datasets, relationships
recommendations.yaml          ->        Bronze/Silver/Gold transformations
manifest.yaml                 ->        Fitted transformer artifacts
validation_report.yaml        ->        Scoring validation results
```

## Project Structure

```
src/customer_retention/
├── analysis/
│   ├── auto_explorer/              # Exploration orchestration
│   │   ├── project_context.py      # Intent contract + dataset registry
│   │   ├── prediction_objective_detector.py  # Objective feasibility
│   │   ├── objective_support_communicator.py  # Evidence voting
│   │   ├── intent_defaults.py      # Rule-based control variables
│   │   ├── snapshot_grid.py        # Point-in-time grid derivation
│   │   ├── dataset_fingerprinter.py # Structural analysis
│   │   ├── entity_timestamp_deriver.py # Timestamp column discovery
│   │   ├── run_namespace.py        # Directory layout
│   │   ├── session.py              # Per-user session state
│   │   ├── active_dataset_store.py # Delta-backed dataset persistence
│   │   └── exploration_manager.py  # Multi-dataset discovery
│   ├── business/                   # Business logic (ROI, risk scoring)
│   ├── diagnostics/                # Model diagnostics
│   ├── interpretability/           # Model explanations (SHAP)
│   └── visualization/              # Chart building and display
├── stages/
│   ├── temporal/                   # Leakage-safe temporal framework
│   │   └── temporal_merger.py      # Multi-dataset as-of join
│   ├── modeling/                   # Model training and evaluation
│   │   ├── cross_validator.py      # TemporalEntitySplit + CrossValidator
│   │   └── data_splitter.py        # Temporal/stratified splitting
│   ├── profiling/                  # Data profiling and quality
│   ├── transformation/             # Feature transformation
│   └── validation/                 # Scoring pipeline validators
├── generators/
│   ├── pipeline_generator/         # Local pipeline code generation
│   │   ├── generator.py            # PipelineGenerator
│   │   ├── databricks_generator.py # DatabricksPipelineGenerator
│   │   ├── renderer.py             # Jinja2 template rendering
│   │   ├── databricks_renderer.py  # PySpark template rendering
│   │   └── findings_parser.py      # Namespace-aware findings loading
│   └── orchestration/              # PipelineContext, DatabricksExporter
├── integrations/adapters/
│   ├── storage/                    # DeltaStorage (base, local, databricks)
│   ├── feature_store/              # Feast and Databricks feature store
│   └── mlflow/                     # MLflow experiment tracking
├── transforms/                     # Fit/transform separation
├── core/                           # Core abstractions and compat layer
└── feature_store/                  # Temporal-aware feature store

exploration_notebooks/              # Interactive exploration notebooks
_experiments/                       # All experiment outputs (gitignored)
scripts/                            # CLI utilities
tests/                              # Test suite (7900+ tests, 91%+ coverage)
```

## Key Architectural Decisions

1. **Grid is global; training is objective-specific.** Aggregation uses the consensus grid. Training uses an objective-specific subset of that grid.

2. **Exploration uses a default snapshot; training uses multiple snapshots.** Notebooks 04-05 run on one representative snapshot for analysis speed. Training uses the full objective slice for robustness.

3. **Source cleanup happens before merge.** Notebook 02 runs per dataset post-aggregation to avoid provenance complexity and eliminate broken columns early.

4. **Relationship analysis happens only after merge.** Interactions and redundancy only exist in the merged feature space.

5. **Reactive vs Stable is a sampling policy, not a separate pipeline.** Same grid, different selection: dense recent vs broad historical.

6. **Train/test split is entity-aware and temporally purged.** Never row-random. `TemporalEntitySplit` ensures all rows of an entity stay together, with optional purge gap.

7. **Each notebook produces something that is not re-evaluated later.** Temporal evidence drives grid design. Aggregation happens once. Cleanup happens before merge. Production faithfully replicates exploration.

## Next Steps

- [[Model Intent and Objective Support]] - How prediction objectives are declared and validated
- [[Snapshot Grid and Control Variables]] - How the as-of grid is derived from intent
- [[Exploration Loop]] - Deep dive into the notebook workflow
- [[Local Track]] - Feast + MLFlow execution
- [[Databricks Track]] - Unity Catalog + Delta Lake execution
