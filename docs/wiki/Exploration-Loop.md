# Exploration Loop

The exploration loop is **iterative by design**. Each iteration is versioned, and feedback from model training informs the next round of feature engineering.

## Workflow Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EXPLORATION LOOP                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ITERATION 1                                                                 │
│  ──────────                                                                  │
│  Notebook 00: Intent Contract                                                │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  • Declare prediction objective + posture + horizon│                      │
│  │  • Register datasets, fingerprint, detect targets  │                      │
│  │  • IntentDefaultsEngine derives control variables  │                      │
│  │  • SnapshotGrid.from_intent() creates temporal grid│                      │
│  │  • Save project_context.yaml to namespace          │                      │
│  └───────────────────────────────────────────────────┘                       │
│                          │                                                   │
│            ┌─────────────┴─────────────┐                                     │
│            ▼                           ▼                                     │
│     EVENT DATASETS              ENTITY DATASETS                              │
│  ┌──────────────────┐      ┌──────────────────┐                              │
│  │ 01  Discovery     │      │ 01  Discovery     │                             │
│  │ 01a Temporal Dive │      │ (entity-shape     │                             │
│  │ 01b Quality       │      │  data passes      │                             │
│  │ 01c Patterns      │      │  through)          │                            │
│  │ 01d Aggregation   │      │                    │                            │
│  │ 02  Integrity     │      │ 02  Integrity      │                            │
│  └────────┬─────────┘      └────────┬───────────┘                            │
│            └─────────────┬───────────┘                                       │
│                          ▼                                                   │
│  Notebook 03: Dataset Merge                                                  │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  • TemporalMerger joins on (entity_id, as_of_date)│                      │
│  │  • Entity datasets broadcast or as-of-join         │                      │
│  │  • Produces unified feature matrix                 │                      │
│  └───────────────────────────────────────────────────┘                       │
│                          │                                                   │
│                          ▼                                                   │
│  Notebooks 04-05: Post-Merge Analysis                                        │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  04  Column Deep Dive (types, skewness, encoding)  │                      │
│  │  04a Text Columns Deep Dive (if text columns)      │                      │
│  │  05  Relationship Analysis (correlations, redund.) │                      │
│  └───────────────────────────────────────────────────┘                       │
│                          │                                                   │
│                          ▼                                                   │
│  Notebooks 06-08: Feature Engineering & Training                             │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  06  Feature Opportunities (transforms, encoding)  │                      │
│  │  07  Modeling Readiness (training grid, split)     │                      │
│  │  08  Baseline Experiments (entity-grouped CV)      │  ← FEEDBACK          │
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
│  │  YES → Continue to notebooks 09-12                │                       │
│  └───────────────────────────────────────────────────┘                       │
│         │                                                                    │
│         ▼                                                                    │
│  Notebooks 09-12: Release & Validate                                         │
│  ┌───────────────────────────────────────────────────┐                       │
│  │  09. Business Alignment  → Connect ML to goals    │                       │
│  │  10. Spec Generation     → Generate pipeline code │  ← PRODUCTION         │
│  │  11. Scoring Validation  → Validate scoring       │  ← VALIDATION         │
│  │      matches training pipeline                    │                       │
│  │  12. View Documentation  → Export HTML docs       │                       │
│  └───────────────────────────────────────────────────┘                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## The Educational Approach

### Learn by Doing: Exploration Notebooks

The heart of this framework is a series of **interactive notebooks** that guide you through the entire ML workflow. Each notebook:

- Explains the concepts before diving into code
- Shows you what to look for in your data
- Demonstrates common pitfalls and how to avoid them
- Produces artifacts you'll use in later stages

**Why this matters**: Most ML tutorials jump straight to `model.fit()`. Real-world projects fail because of data issues, leakage, or misaligned metrics—problems you catch in exploration.

## Early Target Mount

In multi-dataset projects, the target column (e.g., `churned`) typically lives in only one dataset (e.g., customer profiles). Per-dataset notebooks need the target for correlations, effect sizes, and aggregation strategy — but non-target datasets don't have it.

The **Early Target Mount** mechanism solves this:

1. **Notebook 00** scans all datasets using `DatasetFingerprinter`, which detects the target holder, join keys, and relationship types. Results are saved to `project_context.yaml`.
2. **Notebook 01** checks the project context when no target is detected in the current dataset. If found, it calls `mount_target_column()` to left-join the target from the target dataset onto the current DataFrame.
3. Downstream per-dataset notebooks then have full target-aware analysis available.

```python
from customer_retention.analysis.auto_explorer import DatasetContextScanner, mount_target_column

scanner = DatasetContextScanner()
context = scanner.scan(["profiles.csv", "transactions.csv", "tickets.csv"])
# context.target_dataset = "profiles", context.target_column = "churned"

# Mount target onto a non-target dataset
df_transactions, info = mount_target_column(df, context, "transactions")
# df_transactions now has "churned" column joined via customer_id
```

## Dataset Merge (Notebook 03)

Notebook 03 uses the `TemporalMerger` to combine all Bronze outputs into a unified feature matrix. Three merge strategies apply automatically based on dataset shape:

| Dataset Shape | Strategy | Join Keys |
|---------------|----------|-----------|
| Event-level (aggregated) | Snapshot join | `entity_id` + `as_of_date` |
| Entity-level (no timestamp) | Broadcast | `entity_id` only (features repeat across dates) |
| Entity-level (with timestamp) | As-of join | `entity_id` + backward-looking temporal match |

The merge scaffold (join keys and relationship types) is defined in Notebook 00 and stored in `ProjectContext`. Notebook 03 reads it from the run namespace.

## Capturing Recommendations (Notebook 06)

As you explore, capture cleaning and transformation recommendations using the layered builder:

```python
from customer_retention.analysis.auto_explorer import RecommendationBuilder

# Create builder from your exploration findings
builder = RecommendationBuilder(findings, notebook="06_feature_opportunities")

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

## Iteration Tracking

Model development is iterative. The `iteration` module tracks your progress across multiple cycles:

```python
from customer_retention.iteration import (
    IterationOrchestrator,
    IterationTrigger,
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
```

### Iteration Triggers

- `INITIAL` - First exploration
- `MANUAL` - User-initiated refinement
- `DRIFT_DETECTED` - Data drift signals
- `PERFORMANCE_DROP` - Model performance degradation
- `SCHEDULED` - Regular retraining schedule

### What Gets Tracked

- Applied vs skipped recommendations with outcomes
- Feature importance feedback from models
- Iteration lineage (parent → child relationships)
- Model metrics for comparison across versions

## Closing the Loop: Scoring Validation (Notebook 11)

When exploration is complete and a production pipeline has been generated (Notebook 10), Notebook 11 validates that the **scoring pipeline reproduces training features identically** for holdout entities. This catches train/serve skew before deployment.

The validation process:

1. **Load gold features** — including holdout entities whose target was masked during training
2. **Re-run the scoring pipeline** on holdout entities using `TransformExecutor` with `fit_mode=False`
3. **Compare features** — `AdversarialScoringValidator` checks that recomputed features match the training-produced features within tolerance
4. **Compare predictions** — `ScoringPipelineValidator` verifies model outputs are consistent
5. **Generate report** — `ValidationReport` with severity levels (LOW → CRITICAL) and per-feature drift statistics

Notebook 12 then exports all notebook outputs as browsable HTML documentation.

## From Recommendations to Transforms

Recommendations captured in Notebook 06 (e.g., "impute nulls in age", "scale revenue") become `TransformationStep` objects stored in the pipeline specification. These steps are replayed by `TransformExecutor` — stateless operations run directly via `ops.py`, while stateful ones (scaling, encoding, power transforms) go through `fitted.py` wrappers that persist their parameters to `ArtifactStore`. This ensures the exact same transformation is applied during scoring as during training.

See [[Transforms & Scoring Validation|Transforms-and-Scoring-Validation]] for full details.

## Multi-Source Pipeline Structure

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

## Next Steps

- [[Snapshot Grid and Control Variables]] - Leakage-safe temporal grid and control variables
- [[Transforms & Scoring Validation|Transforms-and-Scoring-Validation]] - Fit/transform separation and validation gates
- [[Local Track]] - Generate and run pipelines locally
- [[Tutorial: Retail Customer Retention|Tutorial-Retail-Churn]] - Complete hands-on example
