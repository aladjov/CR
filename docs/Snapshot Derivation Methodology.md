# Snapshot Derivation Methodology for Exploration-Driven Medallion Framework

This document defines the **operational rules and implementation guidance** for producing a single, reusable snapshot layer from heterogeneous datasets within a medallion architecture.

The snapshot must:

- support multiple churn model types
- remain point-in-time correct
- work across event-level and entity-level datasets
- be derived consistently during exploration and production
- preserve lineage and reproducibility
- remain usable for multiple training cadences and downstream use cases

This is not a modeling document.
It defines **how the data layer produces snapshots**.

---

# 1. Conceptual Definition

A snapshot represents:

> The state of each entity at a specific decision moment, constructed only from information that was available at that moment.

Every snapshot is defined by a single parameter:

- **as_of** — the time boundary representing “what was knowable then”

All features in the snapshot must reflect entity state at that time.

---

# 2. Role of Exploration Phase

The exploration phase exists to determine:

- viable aggregation windows
- data quality and cleaning strategies
- feature transformations
- dataset cadence alignment
- entity definitions
- activity definitions
- feasibility of targets

Exploration **does not produce final modeling datasets**.
It produces:

- configuration
- aggregation rules
- transformation logic
- feature definitions

Those rules are then used to generate reproducible snapshots.

---

# 3. Medallion Architecture Responsibilities

## Landing

Purpose:

- ingest raw data
- standardize schema
- infer timestamps
- detect cadence
- perform minimal cleaning

Must produce:

- canonical timestamp columns
- entity identifiers
- source metadata

No aggregation occurs here.

---

## Bronze

Purpose:

- separate event-level and entity-level datasets
- normalize timestamps
- validate data quality
- perform type corrections
- preserve raw lineage

Rules:

- event datasets retain event timestamps
- entity datasets retain validity/update timestamps
- no cross-dataset merges
- no feature windows yet

---

## Silver

Purpose:

- apply aggregation logic
- align dataset cadence
- prepare entity-level feature inputs

This is the **critical layer where snapshot derivation begins.**

---

## Gold

Purpose:

- produce final entity snapshot datasets
- attach labels
- freeze feature set for modeling

Gold contains the actual snapshot tables used by models.

---

# 4. Core Rules for Snapshot Derivation

## Rule 1 — Snapshot Anchor

Every snapshot must be anchored to a single time:

- `as_of`

And must include:

- `feature_timestamp = as_of`

No entity-specific anchors allowed.

---

## Rule 2 — Temporal Filtering

Before any aggregation:

All event data must be filtered:

- include only records where event_timestamp ≤ as_of

This rule applies to:

- transactions
- tickets
- interactions
- usage logs
- payments
- contract changes

No exceptions.

---

## Rule 3 — Entity State Representation

For each entity at `as_of`, compute:

- last known attributes
- aggregated behavior
- recency metrics
- velocity metrics
- lifecycle indicators

Derived only from filtered data.

---

## Rule 4 — Dataset Cadence Alignment

Datasets may have different cadences:

- event datasets (high frequency)
- entry datasets (snapshot-style)
- contract datasets (slow changing)
- support logs (irregular)

Silver must align them:

- transform all into entity-level features valid at `as_of`

No joins allowed using future timestamps.

---

## Rule 5 — Event Aggregation

Event-level datasets must be converted to entity features using:

- rolling windows
- counts
- rates
- time-since metrics

Windows must be anchored at `as_of`.

Examples:

- activity last 7/30/90 days
- failure rate last 30 days
- support tickets last 60 days

---

## Rule 6 — Entity-Level Dataset Handling

Entity-level datasets must represent:

- latest known state prior to or at `as_of`

Use:

- last valid record ≤ as_of
- or effective-dated records

Never use:

- current state
- latest state after as_of

---

## Rule 7 — Multi-Dataset Merge

Merging occurs only after:

- each dataset independently transformed to entity-level features at `as_of`

Join key:

- entity identifier

Never merge:

- event rows directly across datasets

---

## Rule 8 — Snapshot Construction

For each entity at `as_of`, produce a row containing:

- entity identifier
- feature_timestamp
- behavioral features
- lifecycle features
- contract attributes
- activity metrics

This row represents the entity’s state at that moment.

---

## Rule 9 — Label Attachment

Labels are computed separately and attached after snapshot creation.

Label rules:

- outcome occurs strictly after `as_of`
- label timestamp indicates when outcome becomes observable
- snapshot must not include post-outcome data

---

## Rule 10 — Exploration vs Production Consistency

The same logic used during exploration must be used in production:

- same aggregation windows
- same transformations
- same cleaning
- same cadence alignment

Exploration determines configuration.
Production executes configuration.

---

# 5. Autonomous Exploration Guidance

Exploration must probe:

## Dataset structure

- entity keys
- timestamp availability
- cadence
- missingness

## Event distribution

- frequency
- density
- burstiness
- inactivity spans

## Feature feasibility

- recency stability
- window effectiveness
- transformation necessity

## Data quality

- invalid timestamps
- duplicates
- sparse entities
- inconsistent identifiers

Exploration must output:

- recommended window sizes
- feature candidates
- aggregation viability
- data sufficiency signals

---

# 6. Snapshot Generation Workflow

For each snapshot date:

1. Select as_of
2. Filter all event datasets by timestamp ≤ as_of
3. Extract entity state from entity datasets valid at as_of
4. Aggregate event datasets into entity features
5. Align all datasets to entity level
6. Merge into unified entity feature table
7. Attach labels (post-as_of)
8. Persist snapshot partition

Repeat for each as_of date.

---

# 7. Handling Multiple Snapshot Cadences

Snapshots may be generated:

- daily
- weekly

But logic remains identical.

Differences only in:

- as_of values
- training window selection

No change in feature logic.

---

# 8. Lineage and Reproducibility

Every snapshot must record:

- as_of
- aggregation configuration version
- source dataset versions
- transformation rules
- feature set version

Snapshots must be reproducible deterministically.

---

# 9. Data Evolution and Stability

The framework must allow:

- adding datasets
- removing features
- adjusting windows

Without breaking historical snapshots.

Historical snapshots remain immutable.

---

# 10. Key Anti-Patterns to Avoid

- using latest available data instead of as_of data
- aggregating before temporal filtering
- merging event datasets directly
- deriving features after labels
- using entity last-event timestamps as snapshot anchors
- recomputing features differently in production vs exploration

---

# 11. Design Outcome

Following these rules produces a snapshot layer that:

- supports multiple churn definitions
- supports multiple model types
- supports multiple training cadences
- preserves causal correctness
- remains extensible
- avoids leakage
- keeps complexity manageable

The snapshot becomes the universal foundation for:

- early termination models
- renewal models
- inactivity models
- segmentation
- experimentation
- future causal modeling
