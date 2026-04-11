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

Notebooks 04-05 run on the merged dataset: column deep dive (type validation, skewness, encoding hints) and relationship analysis (correlations, redundancy, feature-target associations, interaction opportunities). Notebook 05 concludes with **statistical feature selection** -- variance and pairwise-correlation filters that produce `drop_weak` and `drop_multicollinear` recommendations persisted to `merged/recommendations.yaml`.

### Phase III -- Gold: Modeling and Production

Notebooks 06-07 consolidate feature opportunities and formalize the training setup (grid selection, split policy, label horizon). Notebook 08 prepares data for modelling in five stages -- prerequisite validation, feature availability filtering, gold-layer transforms (fitted on train only), entity-aware temporal split with purge gap, and **L1-regularised feature selection** that complements the NB05 statistical drops. It then trains baseline models with **entity-grouped temporal cross-validation** (`TemporalEntitySplit`) to prevent entity information leakage.

Notebook 09 aligns model output with business objectives. Notebook 10 generates production pipeline code for both local and Databricks execution. Notebook 11 validates that scoring reproduces training features identically.

## Medallion Layers

| Layer | Scope | Notebooks | What Happens |
|-------|-------|-----------|--------------|
| **Intent** | Contract | 00 | Objective, posture, horizon, control variables, snapshot grid |
| **Landing** | Raw → Delta | 01 | Dataset fingerprinting, data discovery, entity key resolution, Delta table creation |
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
| 05 | Relationship Analysis | Correlations, redundancy, interactions, statistical feature selection |
| 06 | Feature Opportunities | Transformation and encoding candidates |
| 07 | Modeling Readiness | Training grid, split policy, label horizon |
| 08 | Baseline Experiments | L1 feature selection, gold transforms, model training with temporal CV |
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

### Entity Key Resolution (Landing Only)

Not every source dataset contains the entity column directly. Notebook 00 detects these datasets and creates `KeyResolutionStep` entries that describe how to join through a bridge dataset to obtain the entity column (e.g., join `case_history` → `case` on `CASE_ID` to obtain `ACCOUNT_ID`).

Notebook 01 applies these steps once at landing time via `resolve_single_dataset_keys()`. After resolution, the entity column is present in the Delta table written to `landing/{dataset_name}/`. All downstream stages (Bronze, Silver, Gold) operate on data that already contains the entity column — no further key resolution is attempted.

If a resolution step references columns that no longer exist (e.g., schema changed between NB00 and NB01), the step is skipped with a warning rather than failing the notebook.

### Timestamp Sanitization at System Boundaries

All timezone information is removed and extreme timestamps are clamped at the point where data enters the system. Source systems (e.g., Salesforce, Databricks tables) often store timestamps as `timestamp[us, tz=Etc/UTC]` with sentinel values (e.g., year 0017 for nulls). These must be sanitized BEFORE wrapping as pyspark.pandas, because pyspark.pandas operations may internally create Python UDFs that serialize data through Arrow, and Arrow cannot represent timestamps outside the nanosecond range (years 1678-2262).

`sanitize_spark_timestamps(spark_df)` combines `strip_spark_timestamp_tz()` (cast `TimestampType` → `timestamp_ntz`) and `clamp_spark_timestamps()` (null values outside years 1678-2261). It is automatically applied in two places:

1. **`as_pandas_api(spark_df)`** — the gateway from native Spark to pyspark.pandas. Every Spark DataFrame converted to pyspark.pandas is sanitized first, ensuring no downstream pyspark.pandas operation encounters extreme or tz-aware timestamps.
2. **`DatabricksDelta.read()`** — sanitizes before wrapping the loaded Delta table as pyspark.pandas.

On the pandas path, `normalize_timestamps()` delegates to the internal `_normalize_timestamp_columns()` which calls `tz_localize(None)` on any tz-aware column. The public API is always `normalize_timestamps()` — it dispatches to the correct backend (pandas or pyspark.pandas).

Additional safety nets exist at the write boundary (`clamp_distributed_timestamps` in `_write_delta`, `clamp_spark_timestamps` in `DatabricksDelta.write()`), but these are defense-in-depth — the primary sanitization happens at the input boundary.

All downstream stages (Bronze through Gold) operate exclusively on tz-naive timestamps.

### Two-Step Feature Selection (Notebooks 05 and 08)

Feature selection is split across two notebooks because each step requires different inputs and serves a different purpose.

**Step 1 -- Statistical Filters (NB05):** Runs on the merged silver feature matrix *before* any train/test split. Identifies features to drop based on properties intrinsic to the data:

| Filter | What it catches | Threshold |
|--------|----------------|-----------|
| **Variance** | Near-constant features that carry no signal | Configurable (`VARIANCE_THRESHOLD`, default 0.01) |
| **Pairwise correlation** | Redundant pairs where the weaker predictor is dropped | Configurable (`CORRELATION_THRESHOLD`, default 0.95) |

Results are persisted as `drop_multicollinear` and `drop_weak` recommendations in `merged/recommendations.yaml`. NB05 also persists a `feature_selection_config` containing the thresholds used and the list of features analyzed, so NB08 can identify which features are new.

**Step 2 -- L1-Regularised Selection (NB08):** Runs *after* the temporal train/test split, on training data only. Three sub-stages:

1. **NB05 drops applied** — features flagged as `drop_multicollinear` or `drop_weak` are removed immediately.
2. **Statistical filters on new features only** — features added after NB05 (interactions, ratios, composites, gold transforms) are identified via `feature_selection_config.analyzed_features`. Variance and correlation filters run on these new features only, using NB05's original thresholds. Base features that already passed NB05 are protected via `candidate_features` — they cannot be dropped by variance or correlation, but new features that correlate with them will be dropped. This avoids recomputing the full correlation matrix for features already vetted by NB05.
3. **L1 selection** — fits a logistic regression with L1 penalty on *all* remaining features (base + new). Features whose coefficients shrink to zero have no marginal predictive value and are dropped.

**Chi-Squared Rescue Selection (NB08, default since 2026-04):** the GBDT-importance stage now runs in `chi_squared_rescue` mode by default (`GBDT_SELECTION_MODE` knob). Instead of `chi-squared → L1 → GBDT` chained on the full snapshot grid, it picks a single penultimate-snapshot slice (one row per entity, label-complete, IID), runs chi-squared on that slice as the primary selector, and then trains XGBoost (and optionally L1) on the same slice but restricted to the chi-squared **drop pool**. The union of the three keep sets is the final selected feature set; every dropped feature carries a triple-consensus `drop_rescue_consensus` audit dict (chi rank + L1 coefficient + GBDT total_gain + slice metadata). This addresses chi-squared's documented bias against time-varying features in `N_entities × N_snapshots` panel data without inflating compute cost — slicing collapses the rescue training data by an order of magnitude, and the Spark dispatch reuses the existing batched primitives (`_spark_chi_squared_selection`, `_spark_gbdt_importance_selection`, `_spark_l1_selection`) so the path stays distributed and chunked. Legacy chain mode remains available behind `GBDT_SELECTION_MODE = "chain"` for A/B comparison. See `docs/chi_squared_rescue_selection_plan.md` for the design.

The two steps are deliberately separated:

- NB05 filters are **data-intrinsic** (variance, redundancy) and do not require a target split. They run early so that downstream analysis (NB06-07) and gold-layer transforms operate on a cleaner feature set.
- NB08 L1 selection is **model-aware** and must run after the temporal split to avoid target leakage from the test set into the selection process. The intermediate variance/correlation pass catches new features that NB05 never saw.

Both steps write their results to `merged/recommendations.yaml`. NB08 gates NB05 drops behind `APPLY_NB05_DROPS` and its own L1 pass behind `L1_FEATURE_SELECTION_ENABLED`, so either step can be toggled independently.

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
    def optimize(self, path, z_order_columns=None): ...
    def vacuum(self, path, retention_hours=168): ...
    def exists(self, path) -> bool: ...
```

### Key Design Points

- **`PRODUCTION_DIR` defaults to `EXPERIMENTS_DIR`** -- experiments and production share the same Delta tables until you explicitly separate them.
- **`PipelineContext.build_commit_metadata()`** attaches `run_id`, `pipeline_stage`, `run_type`, and `timestamp` to every Delta commit.
- **Version-pinned reads**: `storage.read(path, version=3)` gives you an exact historical snapshot for reproducibility.

### OPTIMIZE + Z-ORDER After Writes

Every Delta write in the pipeline is followed by an automatic `OPTIMIZE` with `Z-ORDER` on the columns most frequently used in downstream reads. This compacts small files produced by the write and clusters rows by the z-order columns, enabling data skipping on subsequent reads.

The `DeltaStorage` interface exposes a single method:

```python
def optimize(self, path: str, z_order_columns: Optional[List[str]] = None) -> None: ...
```

| Implementation | Compact | Z-ORDER |
|----------------|---------|---------|
| `LocalDelta` (delta-rs) | `dt.optimize.compact()` | `dt.optimize.z_order(columns)` |
| `DatabricksDelta` (delta-spark) | `dt.optimize().executeCompaction()` | `dt.optimize().executeZOrderBy(columns)` |

Z-ORDER columns are chosen per stage based on the join keys used by downstream notebooks:

| Stage | Z-ORDER Columns | Why |
|-------|----------------|-----|
| Landing | `(entity_col, time_col)` | NB01d aggregates by entity + time window |
| Bronze Event (aggregated) | `(entity_col, as_of_date)` | NB03 merges on entity + snapshot date |
| Bronze Entity | `(entity_col)` | NB03 joins on entity key |
| Silver | `(entity_id, as_of_date)` | NB04-08 filter/group by entity and date |
| Gold | `(entity_id, event_timestamp)` | Training and scoring read by entity |

The optimization applies in three contexts:

1. **Exploration notebooks** — `save_active_dataset()` (NB01), `save_aggregated_dataset()` (NB01d), and `delta.write()` (NB03) pass `z_order_columns` directly to the storage adapter's `write()` method, which runs OPTIMIZE + Z-ORDER atomically after the write. Columns are derived from findings (entity column, detected timestamp) or from the post-merge standard (`entity_id`, `as_of_date`). `print_write_report()` displays verified diagnostics (file count, core count, z-order confirmation from Delta history).

2. **Generated local pipelines** — Each template calls `get_delta(force_local=True).optimize(str(output_path), z_cols)` after writing. Column guards (`if c in df.columns`) ensure the call succeeds even if a column is absent.

3. **Generated Databricks pipelines** — Each template calls `DeltaTable.forName(spark, output_table).optimize().executeZOrderBy(z_cols)` after `saveAsTable()`. Same column guards using Spark schema field names.

When no z-order columns are available (e.g., column not present in the DataFrame), the call falls back to compaction-only, which still reduces small file overhead.

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

### Framework Imports in Generated Code

Generated pipeline code should minimize dependencies on the exploration framework. Allowed imports are limited to modules that provide runtime functionality the pipeline genuinely needs (transforms, storage adapters, compat layer, data splitting, feature profiling, run namespace). When a framework import is necessary to ensure parity between exploration and production — for example, reading the canonical recommendations at runtime so drop reasons stay in sync — it is acceptable. The test `test_no_framework_imports` enforces the allowlist; add new entries only when the import is required for exploration/production parity and cannot be replaced by data emitted at generation time without risking drift.

### Naming Convention

- **Composite Name (CN)**: `{readable_prefix}__{7char_hash}` derived from sorted source names
- **Readable prefix**: first 4 characters of each word, lowercase, joined with `_`
- **Stage naming**: `bronze_entity_{source}`, `silver_featureset_{CN}`, `gold_features_{CN}`

## Notebook Data Sources

Each notebook loads data from a specific pipeline stage. Post-merge notebooks (04+) always load from `silver_merged`; they never fall back to per-dataset landing or bronze data. This ensures recommendations target column names that actually exist in the merged feature matrix.

| Notebook | Data Source | Findings Source | Loading Function |
|----------|-------------|-----------------|------------------|
| 01, 01a-01d | Landing (per dataset) | Per-dataset | `load_active_dataset` |
| 02 | Bronze (per dataset, post-aggregation) | Per-dataset | `load_active_dataset` |
| 03 | Bronze (all datasets) | Per-dataset (all) | `RunNamespace.discover_all_findings` |
| 04-05 | **Silver merged** | **Merged findings** (`prefer_merged=True`) | `require_silver_merged` |
| 06-09 | **Silver merged** | **Merged findings** (`prefer_merged=True`) | `require_silver_merged` |
| 10 | N/A (code generation) | Merged findings + merged recommendations | `FindingsParser` |

`require_silver_merged` raises `FileNotFoundError` if `silver_merged/` does not exist. There is no fallback to landing or bronze data — NB03 must have run first.

## Recommendation Data Flow

Recommendations are layered (Bronze, Silver, Gold) and accumulate across notebooks. Per-dataset recommendations merge into a single `merged/recommendations.yaml` at the transition from per-dataset to post-merge notebooks.

```
Per-Dataset Phase                       Post-Merge Phase
─────────────────                       ────────────────
NB01 → {dataset}/recs.yaml ──┐
NB02 → {dataset}/recs.yaml ──┼── NB04 merges into merged/recommendations.yaml
                              │         │
                              │         ├── NB04 appends Gold recs (transforms, encoding)
                              │         ├── NB05 reads merged, appends relationship recs
                              │         ├── NB06-07 read merged, append feature recs
                              │         ▼
                              │   merged/recommendations.yaml (cumulative)
                              │         │
                              │         ▼
                              └── NB10 reads merged recs → FindingsParser → pipeline code
```

**NB04 is the merge point.** It collects all `*_recommendations.yaml` files from each dataset's findings directory and merges them via `RecommendationRegistry.merge()` before adding its own Gold-layer recommendations. All subsequent notebooks read from and write to `merged/recommendations.yaml`.

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

8. **Feature selection is two-step: data-intrinsic (NB05) then model-aware (NB08).** NB05 removes variance-dead and pairwise-redundant features before any split. NB08 applies L1-regularised selection on training data only, catching features that are individually plausible but collectively redundant. Splitting the steps avoids target leakage from the test set into the selection process while still cleaning the feature set early for downstream analysis.

## Next Steps

- [[Model Intent and Objective Support]] - How prediction objectives are declared and validated
- [[Snapshot Grid and Control Variables]] - How the as-of grid is derived from intent
- [[Exploration Loop]] - Deep dive into the notebook workflow
- [[Local Track]] - Feast + MLFlow execution
- [[Databricks Track]] - Unity Catalog + Delta Lake execution
