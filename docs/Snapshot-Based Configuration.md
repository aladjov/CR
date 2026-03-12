# Snapshot-Based Configuration for Churn Modeling Framework

This document defines the minimal constants, configuration, and operating assumptions required to support three practical churn model types using a single snapshot data layer and two training tiers (fast/reactive and slow/historical).

The goal is to:

- Maintain one point-in-time–correct snapshot store
- Support multiple churn model types from the same dataset
- Train multiple algorithms per model
- Run daily reactive models and weekly historical models
- Keep complexity low while remaining extensible

---

# 1. Snapshot Invariants

All models operate from the same snapshot dataset.

For each snapshot partition:

- `as_of` defines the decision moment
- `feature_timestamp` is equal to `as_of`
- All features are computed using events where `event_timestamp` is less than or equal to `as_of`
- Labels are computed only using outcomes that occur after `as_of`
- The snapshot table is partitioned by `as_of` date

This ensures:

- point-in-time correctness
- shared feature lineage
- compatibility across all model types

---

# 2. Supported Model Types

The framework supports three base churn model types.

## Model Type 1 — Early Termination Risk

Question:
Which customers will terminate prematurely within the next defined horizon?

- Cohort: active customers at the snapshot time
- Target: early termination within the horizon window

---

## Model Type 2 — Renewal Non-Renewal Risk

Question:
Among customers approaching renewal, which will not renew?

- Cohort: customers whose renewal date falls within a defined lookahead window
- Target: absence of renewal after renewal date plus grace period

---

## Model Type 3 — Inactivity / Behavioral Churn

Question:
Which customers will become inactive within the next prediction horizon?

- Cohort: active customers at snapshot time
- Target: inactivity threshold crossed within prediction horizon

---

# 3. Shared Feature Configuration

Feature engineering uses a fixed set of rolling behavioral windows.

Feature window durations:

- 7 days
- 30 days
- 90 days

Recency control:

- recency values are capped at 365 days to avoid extreme outliers

These windows support:

- activity velocity
- failure rates
- engagement intensity
- trend signals

---

# 4. Label Horizon and Cohort Parameters

## Early Termination

- prediction horizon defined in days
- example: 60-day horizon

---

## Renewal Non-Renewal

- renewal lookahead window defined in days
- grace period defined in days to determine non-renewal observability

---

## Inactivity Churn

- inactivity threshold defined in days (no qualifying activity)
- prediction horizon defined in days

Additionally:

- a clear definition of what constitutes “activity” must be provided
  (e.g., successful transactions, login, shipment activity, payments, EDI usage)

---

# 5. Snapshot Cadence

Snapshots are materialized daily.

Weekly training pipelines reuse the same daily snapshots and sample from them when needed.

---

# 6. Training Tier Configuration

Two model training tiers are supported.

## Fast / Reactive Tier

- retrained daily
- optimized for detecting recent behavioral changes
- uses shorter training history

Typical training history range:

- 90 to 180 days

---

## Slow / Historical Tier

- retrained weekly
- optimized for long-term stability and structural patterns
- uses longer training history

Typical training history range:

- 365 to 730 days

---

# 7. Evaluation Protocol

Evaluation is time-consistent and avoids random splits.

Rules:

- training uses snapshot partitions up to a cutoff point
- testing uses the most recent partitions following the cutoff
- rows are included only when outcomes are fully observable

Typical evaluation holdout window:

- 60 to 90 days

---

# 8. Base Model Families

Each model type is trained using three algorithm families:

- Logistic Regression
- Random Forest
- Gradient Boosted Trees (e.g., XGBoost)

Optional shared practices:

- class imbalance handling via class weighting
- probability calibration when required

---

# 9. Model Matrix Produced

The framework produces the following combinations:

- 3 model types
- 3 algorithms
- 2 training tiers

Total models generated per run:

- 18 models

All models are trained from the same snapshot dataset.

---

# 10. Data Layer Structure

## Snapshot Table

One row per entity per snapshot date.

Contains:

- entity identifiers
- feature timestamp
- behavioral feature windows
- derived engagement metrics

## Label Tables

Separate label definitions per model type:

- early termination labels
- renewal labels
- inactivity labels

Joined to snapshot using entity identifier and snapshot date.

---

# 11. Core Design Principles

1. Single source of truth for features
2. Point-in-time–correct snapshots
3. Model types derived via label configuration
4. Training tiers differ only in training history
5. Algorithms interchangeable
6. No duplication of feature pipelines

---

# 12. Out of Scope (For Now)

The following capabilities are intentionally excluded to keep the framework focused:

- causal and uplift modeling
- treatment optimization
- intervention policy design
- survival modeling
- deep learning architectures
- multi-task learning

The architecture remains compatible with future expansion.
