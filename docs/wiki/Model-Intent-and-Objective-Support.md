# Model Intent and Objective Support

## The Core Philosophy: One Run, One Purpose

Traditional ML workflows start with data exploration and defer modeling decisions until late in the process. This creates a problem: by the time you realize your data supports renewal-risk prediction but not disengagement detection, you have already spent hours exploring features that are irrelevant to the feasible objective.

The intent contract flips this. Before any exploration begins, you declare:

- **What** you are predicting (the objective)
- **How** the model should reason about time (the temporal posture)
- **How far ahead** the prediction should look (the horizon)

Every notebook downstream inherits this intent and adapts its behavior accordingly. Event datasets generate only relevant temporal evidence. Aggregation produces only the snapshots required. Modeling setup uses the correct training grid and split logic.

### Each Run Produces a Purpose-Built Model

The same data can yield different production models depending on the intent. The intent is not a filter on the output — it shapes every intermediate step.

**Example — Run A: Immediate Churn Risk**
- Objective: `immediate_risk`, Posture: `reactive`, Horizon: 30 days
- Result: dense daily scoring cadence, short observation window (120 days), rapid feature refresh
- Use case: operations team needs a daily list of at-risk customers for outreach

**Example — Run B: Disengagement Detection**
- Objective: `disengagement`, Posture: `stable`, Horizon: 90 days
- Result: monthly scoring cadence, broad observation window (270 days), behavioral trend features
- Use case: product team needs to identify slowly fading customers before they drop off

Both runs can use the same underlying data. They produce different pipeline code, different feature sets, and different models — because the intent shapes every step from snapshot grid to training split.

### Intent Is Not a Checkbox — It Is a Contract

The `project_context.yaml` file saved by Notebook 00 becomes the single source of truth. No downstream notebook re-derives the intent. Every notebook calls `ProjectContext.load()` and reads the contract as-is. If you want a different model for a different business question, you start a new run.

## What Intent Captures

Three choices define the intent:

### Prediction Objective

| Objective | Business Question | Anchor | Typical Data Signals |
|-----------|-------------------|--------|----------------------|
| `immediate_risk` | "Who is about to churn right now?" | `NOW` | Churn flags, cancellation dates, account status |
| `renewal_risk` | "Who won't renew their contract?" | `CONTRACT` | Contract end dates, subscription renewal columns |
| `disengagement` | "Who is gradually becoming inactive?" | `INACTIVITY` | Long entity time series (60+ days), event density |

### Temporal Posture

| Posture | Internal Name | Behavior |
|---------|---------------|----------|
| **Reactive** | `short_memory` | Focus on recent data. Shorter observation windows, denser snapshot cadence, frequent retraining. Best for businesses with fast-changing customer behavior. |
| **Stable** | `long_memory` | Use broader history. Longer observation windows, coarser cadence, less frequent retraining. Best for businesses with gradual behavioral shifts. |

The posture is not a separate pipeline — it is a sampling policy. Both postures use the same consensus snapshot grid; they differ in which portion of the grid they select for training.

### Prediction Horizon

How far ahead the model predicts (in days). Combined with objective and posture, it drives all control variables.

### How These Three Choices Cascade

The `IntentDefaultsEngine` derives six control variables from the combination of objective, posture, and horizon (`H`). These variables propagate to every downstream notebook.

**Observation Window** (how much history to use for features):

| Objective × Posture | Formula | Example (H=90) |
|----------------------|---------|-----------------|
| Immediate Risk × Reactive | `4 × H` | 360 days |
| Immediate Risk × Stable | `max(180, 3 × H)` | 270 days |
| Disengagement × Reactive | `2 × H` | 180 days |
| Disengagement × Stable | `clamp(0.2 × data_span, 180, 365)` | ~270 days |
| Renewal Risk × Reactive | `cycle` | 365 days |
| Renewal Risk × Stable | `clamp(2 × cycle, 365, 730)` | 730 days |

**Purge Gap** (buffer between features and labels to prevent leakage):

| Objective | Formula | Example (H=90) |
|-----------|---------|-----------------|
| Immediate Risk | `H + 14` | 104 days |
| Disengagement | `H + 21` | 111 days |
| Renewal Risk | `label_window + 21` | varies |

**Cadence** (how often snapshots are taken):

| Objective × Posture | Rule |
|----------------------|------|
| Immediate Risk × Reactive (H ≤ 30) | Daily |
| Immediate Risk × Otherwise | Weekly |
| Disengagement × Reactive | Weekly |
| Disengagement × Stable | Monthly |
| Renewal Risk | Based on cycle tier (monthly→weekly, quarterly→biweekly, yearly→monthly) |

**Split Strategy**: `temporal` for immediate risk and disengagement, `cohort_based` for renewal risk.

See [[Snapshot Grid and Control Variables]] for the full derivation tables and a worked example.

## The Three Prediction Objectives

### Immediate Risk

*"Which customers are about to churn right now?"*

The most common objective. It looks for near-term signals of departure: declining engagement, support complaints, payment failures. The model is trained to predict a binary outcome (churned/retained) within a short horizon.

**Typical data signals**: churn flags, cancellation dates, account status columns. The `PredictionObjectiveDetector` identifies these by matching column name patterns like `churn`, `cancel`, `attrition`, `retained`.

**Anchor**: `NOW` — predictions are made relative to the current date.

### Renewal Risk

*"Which customers are unlikely to renew their contract?"*

This objective applies to subscription or contract-based businesses where there is a known renewal date. The model predicts whether a customer will renew when their contract comes up.

**Typical data signals**: contract end dates, subscription renewal columns, expiry dates. Detected by patterns like `contract`, `subscription`, `renewal`, `expiry`.

**Anchor**: `CONTRACT` — predictions are anchored to the contract end date. The prediction horizon counts backward from the renewal deadline.

### Disengagement

*"Which customers are gradually becoming inactive?"*

Unlike immediate risk (which looks for sudden departures), disengagement captures slow decline. Customers who once logged in daily now log in weekly; customers who once bought monthly now buy quarterly. The model detects this fading pattern before it becomes a hard churn.

**Typical data signals**: long entity time series (60+ days minimum), event density, behavioral frequency patterns. Requires temporal data with multiple observations per entity.

**Anchor**: `INACTIVITY` — the prediction trigger is prolonged absence rather than a calendar date.

## Declaring Intent in Notebook 00

Notebook 00 is a guided setup that walks through intent declaration step by step. Auto-detection proposes sensible defaults at each stage; the user reviews and overrides where needed.

### The NB00 Cell Flow

| Section | What Happens |
|---------|--------------|
| **0.1 Project Metadata** | Names the project, initializes the `RunNamespace`, sets up experiment directory |
| **0.2 Dataset Registration** | User provides dataset paths or table names |
| **0.3 Auto Fingerprinting** | `DatasetFingerprinter` profiles each dataset: column types, granularity (entity vs event), entity columns, time columns, target candidates |
| **0.4 Confirm Semantics** | User reviews and overrides auto-detected column roles |
| **0.5 Target Selection** | Identifies which dataset holds the prediction target (e.g., `churned`) and the entity column |
| **0.6 Objective Detection** | `PredictionObjectiveDetector` scores feasibility for each objective (confidence 0-100%), proposes anchors, records evidence |
| **0.7 Priority Review** | User assigns PRIMARY / SECONDARY / EXPLORATORY / DISABLED priorities |
| **0.8 Join Scaffold** | `RelationshipDetector` discovers join keys and relationship types between datasets, builds `merge_scaffold` |
| **0.9 Temporal Posture** | User chooses `STABLE` (long_memory) or `REACTIVE` (short_memory) |
| **0.10 Intent Configuration** | `IntentDefaultsEngine.suggest()` derives control variables from objective + posture + horizon; user can override any value |
| **0.11 Save Context** | Assembles everything into `ProjectContext` and saves to `project_context.yaml` in the run namespace |
| **0.12 Snapshot Grid** | `SnapshotGrid.from_intent()` creates the temporal grid; entity datasets auto-vote, event datasets await votes from 01a-01c |

### Objective Priority

Each objective has a priority that controls downstream behavior:

| Priority | Meaning |
|----------|---------|
| `PRIMARY` | The main modeling goal. Exactly one objective must be PRIMARY. |
| `SECONDARY` | Explored in parallel but not the focus of training setup. |
| `EXPLORATORY` | Kept as a future possibility; no active modeling. |
| `DISABLED` | Explicitly excluded from all analysis. |

## How Intent Steers Every Downstream Notebook

```
Notebook 00: User declares intent
    │
    ├── PredictionObjectiveDetector: auto-detects feasible objectives
    ├── User confirms objective + posture + horizon
    ├── IntentDefaultsEngine: derives control variables
    ├── SnapshotGrid.from_intent(): creates initial grid
    └── ProjectContext saved to namespace
         │
         ├── Notebook 01: Loads intent, sets RECENT_DAYS from
         │   intent.recent_window_days, records objective support
         │   via derive_objective_support() from TemporalComparison
         │
         ├── Notebooks 01a-01c: Read snapshot grid and intent,
         │   contribute grid votes (cadence + date range evidence)
         │   from temporal exploration
         │
         ├── Notebook 01d: Reads locked grid + intent purge/label
         │   windows for event aggregation
         │
         ├── Notebook 03: Reads merge_scaffold, runs TemporalMerger
         │
         ├── Notebook 07: Reads intent for training grid + split policy
         │
         ├── Notebook 08: Reads intent.temporal_split and
         │   intent.purge_gap_days for entity-aware temporal CV
         │
         └── Notebook 10: Reads full intent + findings for
             pipeline generation (local + Databricks)
```

Every notebook calls `ProjectContext.load()` from the namespace. The intent is never re-derived; it propagates as a persistent contract.

### Key Distinction: Objective Support vs Grid Votes

Two different evidence mechanisms feed back into the system during exploration:

- **Objective support** (`derive_objective_support` in `prediction_objective_detector.py`): Computed in **Notebook 01** during data discovery. Uses a `TemporalComparison` (recent vs historical window) to assess how well the data supports each objective. Produces `ObjectiveSupport` entries with strength ratings and implications.

- **Snapshot grid votes** (`DatasetGridVote` in `snapshot_grid.py`): Recorded by **Notebooks 01a-01c** during temporal exploration. Each event dataset votes on optimal cadence and date range based on its temporal structure. These votes can adjust the snapshot grid before it locks.

Both mechanisms read the intent contract. Neither re-derives it.

## Objective Support: Evidence Voting

During data discovery, Notebook 01 records **evidence signals** about each prediction objective's feasibility. This is the objective support system.

### How It Works

1. Each analysis section (temporal coverage, feature drift, regime detection, etc.) records a `SectionRecord` containing:
   - **Signal strength** per objective: 0 (no evidence), 1 (weak), 2 (moderate), 3 (strong)
   - **Why**: rationale phrases explaining the signal
   - **Positives, negatives, gaps**: qualitative evidence

2. The `ObjectiveSupportCommunicator` aggregates these records across all sections into an `ObjectiveSynthesis` per objective:
   - **Combined strength**: average signal across sections (0-3 scale)
   - **Drivers**: positive evidence phrases, ranked by frequency
   - **Frictions**: negative evidence phrases, ranked by frequency
   - **Gaps**: missing information that would strengthen or weaken the assessment
   - **Confidence**: reliability score based on standard deviation across sections (consistent signals = high confidence)

3. The synthesis is displayed as a visual bar chart and written to `objective_support.yaml` in the dataset's namespace directory.

### Signal Rules

Analysis sections can apply structured rules to compute their signal rather than guessing:

```
base_signal = 2  (moderate by default)

Rules:
  entity_ratio < 0.3     -> decrement 1  (too few entities active)
  drift_detected = True   -> decrement 1  (unstable patterns)
  regime_shift = True     -> cap at 1     (structural break in data)
  coverage_pct > 0.8      -> increment 1  (strong temporal coverage)
```

The `apply_signal_rules()` function applies these decrements, increments, and caps to produce the final signal.

### Why Voting Matters

Objective support is not just a yes/no gate. It provides *graded evidence* that helps the user make informed decisions:

- A strong signal (3.0) with high confidence means the data clearly supports the objective
- A moderate signal (1.5) with low confidence means the evidence is mixed — worth investigating further
- A weak signal (0.5) across all sections means the data probably cannot support the objective — but the user can override if they have domain knowledge

The key insight: **votes are not decisions**. They inform the user's choice but never override it.

## The ProjectContext: Where Intent Lives

All intent decisions are persisted in `ProjectContext`, a Pydantic model saved as `project_context.yaml` in the run namespace.

### Key Fields

| Field | Type | Purpose |
|-------|------|---------|
| `project_name` | str | Human-readable project name |
| `run_id` | str | Unique run identifier |
| `datasets` | dict[str, DatasetRegistryEntry] | All registered datasets with metadata |
| `target_dataset` | str | Which dataset holds the target column |
| `target_column` | str | The prediction target (e.g., "churned") |
| `entity_column` | str | The entity identifier (e.g., "customer_id") |
| `objectives` | list[ObjectiveSpec] | All declared objectives with priorities |
| `primary_objective` | PredictionObjective | Which objective is PRIMARY |
| `temporal_posture` | TemporalPosture | Reactive or stable |
| `objective_support` | ObjectiveSupport | Aggregated evidence per objective |
| `intent` | IntentConfig | Derived control variables (see [[Snapshot Grid and Control Variables]]) |
| `merge_scaffold` | list[MergeScaffoldEntry] | How datasets join together |

### Dataset Registry

Each dataset is registered with structural metadata discovered by `DatasetFingerprinter`:

| Field | Purpose |
|-------|---------|
| `name` | Dataset identifier |
| `path` | File path |
| `storage_format` | csv, parquet, or delta |
| `entity_column` | Entity ID column |
| `time_column` | Timestamp column (if any) |
| `granularity` | `EVENT_LEVEL` or `ENTITY_LEVEL` |
| `row_count`, `unique_entities` | Size metrics |
| `target_candidates` | Columns that look like targets |
| `join_key`, `join_to`, `relationship` | Multi-dataset join specification |
| `provenance` | Landing/Bronze/Silver/Gold table references |
| `validation` | Temporal quality, leakage gates, join integrity |

### Validation Rules

`ProjectContext` enforces structural consistency:

- Exactly one objective must be `PRIMARY`
- The `primary_objective` field must match the PRIMARY-priority objective
- No duplicate objectives
- Objectives list cannot be empty

### Target Column Mounting

In multi-dataset projects, the target column (e.g., "churned") often lives in only one dataset. Other datasets need it for correlations and effect sizes during exploration. The `mount_target_column()` function solves this by left-joining the target from the target dataset onto the working DataFrame via the entity key.

## Multiple Runs, Multiple Models

Each run gets its own isolated `RunNamespace` directory under `experiments/runs/`. Different runs can coexist, each targeting a different business question against the same data.

### Directory Layout

```
experiments/
  runs/
    retention-reactive/
      project_context.yaml        # immediate_risk, reactive, 30d
      snapshot_grid.yaml
      datasets/
        ...
    retention-stable/
      project_context.yaml        # disengagement, stable, 90d
      snapshot_grid.yaml
      datasets/
        ...
```

### Practical Workflow

1. **Run "retention-reactive"**: Configure NB00 with `immediate_risk`, `reactive`, 30-day horizon. Run through NB01-NB10. The generated pipeline targets short-term churn with daily scoring.

2. **Run "retention-stable"**: Start a new run in NB00 with `disengagement`, `stable`, 90-day horizon. Same data sources, but NB01-NB10 produce different temporal evidence, different aggregation cadence, different features, and different model training.

3. **Compare and deploy**: Each run generates its own production pipeline code in NB10. Deploy whichever model (or both) serves the business need.

Each run is self-contained. The production pipeline generated from one run does not depend on any other run's artifacts.

## Next Steps

- [[Snapshot Grid and Control Variables]] - How the intent drives snapshot generation
- [[Architecture]] - Overall system architecture
- [[Exploration Loop]] - The notebook workflow in detail
