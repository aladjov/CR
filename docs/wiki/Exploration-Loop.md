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

## The Educational Approach

### Learn by Doing: Exploration Notebooks

The heart of this framework is a series of **interactive notebooks** that guide you through the entire ML workflow. Each notebook:

- Explains the concepts before diving into code
- Shows you what to look for in your data
- Demonstrates common pitfalls and how to avoid them
- Produces artifacts you'll use in later stages

**Why this matters**: Most ML tutorials jump straight to `model.fit()`. Real-world projects fail because of data issues, leakage, or misaligned metrics—problems you catch in exploration.

## Dataset Selection (Notebook 05)

Select which datasets to include in your pipeline. Choices are **persisted** and respected by all exports:

```python
from customer_retention.analysis.auto_explorer import ExplorationManager

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

## Capturing Recommendations (Notebook 06)

As you explore, capture cleaning and transformation recommendations using the layered builder:

```python
from customer_retention.analysis.auto_explorer import RecommendationBuilder

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

- [[Temporal Framework]] - Leakage-safe data preparation
- [[Local Track]] - Generate and run pipelines locally
- [[Tutorial: Bank Customer Churn]] - Hands-on example
