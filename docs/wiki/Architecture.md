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

## From Exploration to Production

The exploration notebooks generate artifacts that drive production pipelines:

```
Exploration Outputs                    Production Usage
───────────────────                    ────────────────
{dataset}_findings.yaml       →        Column types, target, data profile
multi_dataset_findings.yaml   →        Selected datasets, relationships
recommendations (Registry)    →        Bronze/Silver/Gold transformations
```

These artifacts can be used in two ways:

1. **Local Execution** - Use `DataMaterializer` to apply transformations with pandas
2. **Databricks Production** - Export to standalone PySpark notebooks

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
│   ├── stages/                  # Pipeline stages
│   │   ├── temporal/            # Leakage-safe temporal framework
│   │   ├── profiling/           # Data profiling & quality checks
│   │   ├── transformation/      # Feature transformation
│   │   └── validation/          # Quality gates
│   ├── generators/              # Code generation
│   │   └── notebook_generator/  # Notebook generation for pipelines
│   ├── core/                    # Core abstractions
│   └── feature_store/           # Temporal-aware feature store
│
├── exploration_notebooks/       # Interactive exploration notebooks
├── experiments/                 # All experiment outputs (gitignored)
├── scripts/                     # Command-line utilities
└── tests/                       # Test suite (83% coverage)
```

## Next Steps

- [[Exploration Loop]] - Deep dive into the notebook workflow
- [[Temporal Framework]] - Leakage-safe data preparation
- [[Local Track]] - Feast + MLFlow execution
- [[Databricks Track]] - Unity Catalog + Delta Lake execution
