# Snapshot Grid and Control Variables

## The Core Problem: Time Leakage

Imagine you are training a model to predict which customers will churn in the next 30 days. You have a year of historical data. The naive approach is to take each customer's full history, compute features, and train on whether they eventually churned. But this lets the model peek into the future -- features computed from data *after* the prediction point leak information about the outcome.

The solution is **point-in-time snapshots**. For each entity, you freeze the data at a specific date (the "as-of date") and compute features using only data available up to that point. The label is then whether the entity churned in the window *after* that date.

The snapshot grid is the set of as-of dates used across all entities. Getting this grid right is the foundation of leakage-free modeling.

## What the Snapshot Grid Controls

The snapshot grid defines:
- **When** snapshots are taken (the `as_of_date` values)
- **How frequently** snapshots occur (the cadence)
- **How far back** features look (the observation window)
- **How far forward** the label looks (the label window)
- **What gap** separates feature computation from label observation (the purge gap)

Together these create the temporal geometry of training data:

```
                          observation_window
                    ◄──────────────────────────►
    ────────────────┬──────────────────────────┬────────┬──────────────┬──────
                    │     Features computed     │ Purge  │ Label window │
                    │     from this period      │  gap   │ (outcome)    │
    ────────────────┴──────────────────────────┴────────┴──────────────┴──────
                                             as_of_date ▲
```

- **Observation window**: how many days of history to use for features at each snapshot
- **Purge gap**: a buffer zone between the last feature date and the start of the label window. This prevents slow-to-materialize data from leaking into features.
- **Label window**: the period after the purge gap during which the outcome is observed

## Control Variables

Notebook 00 sets six control variables that drive all downstream temporal behavior. These are stored in `IntentConfig` within `ProjectContext`.

### The Variables

| Variable | Default | What It Controls |
|----------|---------|------------------|
| `prediction_horizons` | [30, 60, 90] | How far ahead to predict. Multiple horizons let you compare short-term vs long-term models. |
| `recent_window_days` | 270 | How many days of history to include in features. Also called the observation window. |
| `observation_window_days` | 270 | Same as `recent_window_days` (kept in sync). |
| `purge_gap_days` | 104 | Days between the last feature date and the start of the label window. Prevents time leakage from delayed data. |
| `label_window_days` | 90 | How many days after the purge gap to observe the outcome. |
| `cadence_interval` | weekly | How often to take snapshots: daily, weekly, biweekly, or monthly. |
| `temporal_split` | true | Whether to use time-ordered train/test splits (vs random). |
| `split_strategy` | temporal | `temporal` for time-ordered splits, `cohort_based` for renewal-contract cohorts. |

### Why These Defaults Exist

The defaults (270-day window, 104-day purge, 90-day label) are not arbitrary. They come from the `IntentDefaultsEngine`, which derives values from the combination of your objective, posture, and prediction horizon using explicit formulas.

## How Defaults Are Derived

The `IntentDefaultsEngine` uses a dispatch table indexed by `(objective, posture)`. Given a prediction horizon `H`, data span, and optional renewal cycle, it computes each variable.

### Observation Window (recent_window_days)

The observation window determines how much history is available for feature computation at each snapshot.

| Objective | Posture | Formula | Intuition |
|-----------|---------|---------|-----------|
| Immediate Risk | Reactive | `4 * H` | Short-term signals need recent context, but 4x horizon captures trends |
| Immediate Risk | Stable | `max(180, 3 * H)` | Longer baseline, at least 6 months |
| Disengagement | Reactive | `2 * H` | Recent engagement trajectory |
| Disengagement | Stable | `clamp(0.2 * data_span, 180, 365)` | Proportional to data, bounded to 6-12 months |
| Renewal Risk | Reactive | `cycle` | One full renewal cycle |
| Renewal Risk | Stable | `clamp(2 * cycle, 365, 730)` | Two cycles, bounded to 1-2 years |

**Example**: Immediate Risk, Stable posture, 90-day horizon: `max(180, 3*90) = max(180, 270) = 270 days`.

### Label Window (label_window_days)

How long after the purge gap to observe the outcome.

| Objective | Formula | Intuition |
|-----------|---------|-----------|
| Immediate Risk | `H` | Label matches the prediction horizon exactly |
| Disengagement | `H` | Same -- the horizon *is* the observation period |
| Renewal Risk | `cycle` (or 180 fallback) | The full renewal cycle defines the outcome window |

### Purge Gap (purge_gap_days)

The buffer between features and labels. Prevents leakage from data that arrives late.

| Objective | Formula | Intuition |
|-----------|---------|-----------|
| Immediate Risk | `H + 14` | Horizon plus two weeks for data pipeline delays |
| Disengagement | `H + 21` | Horizon plus three weeks (disengagement detection is slower) |
| Renewal Risk | `label_window + 21` | Full label window plus three weeks for contract processing |

**Why the extra buffer?** In practice, data does not arrive instantly. A customer might churn on day 30, but the "churned" flag might not appear in your data warehouse until day 35. The purge gap ensures that these delayed signals cannot leak into features.

### Prediction Horizons

Multiple horizons let you compare model performance at different time scales.

| Horizon `H` | Generated List | Reasoning |
|-------------|----------------|-----------|
| `H >= 60` | `[H/3, 2H/3, H]` | Short, medium, and full horizon |
| `H < 60` | `[H/2, H]` | Just half and full (too short for three splits) |

**Example**: H=90 generates [30, 60, 90]. H=30 generates [15, 30].

### Cadence Interval

How often snapshots are generated.

| Objective | Posture | Rule | Result |
|-----------|---------|------|--------|
| Immediate Risk | Reactive, H <= 30 | Daily | Captures rapid changes |
| Immediate Risk | Otherwise | Weekly | Balance of granularity and size |
| Disengagement | Reactive | Weekly | Weekly engagement patterns |
| Disengagement | Stable | Monthly | Broader trends |
| Renewal Risk | Any | Depends on cycle tier | Monthly cycles -> weekly; quarterly -> biweekly; yearly -> monthly |

### Split Strategy

| Objective | Strategy | Reasoning |
|-----------|----------|-----------|
| Immediate Risk | `temporal` | Time-ordered: train on past, test on future |
| Disengagement | `temporal` | Same rationale |
| Renewal Risk | `cohort_based` | Contracts define natural cohorts; splitting by cohort prevents leakage across renewal cycles |

## The Snapshot Grid

Once control variables are set, the `SnapshotGrid` generates the actual as-of dates.

### Grid Construction

1. `SnapshotGrid.from_intent()` creates the grid from `IntentConfig` and registered datasets
2. Entity-level datasets are auto-voted (they don't need temporal exploration)
3. Event-level datasets start as "unvoted" -- they must complete notebooks 01a-01c before voting

### Dataset Voting

Each event dataset's temporal exploration produces evidence about optimal cadence and date range. This is captured in a `DatasetGridVote`:

| Field | Purpose |
|-------|---------|
| `dataset_name` | Which dataset is voting |
| `granularity` | Event-level or entity-level |
| `voted` | Has this dataset completed temporal exploration? |
| `suggested_cadence` | The cadence this dataset recommends |
| `suggested_start` | Earliest viable snapshot date |
| `data_span_start`, `data_span_end` | The full temporal coverage of this dataset |

Entity-level datasets auto-vote (`voted=True`) because they don't have temporal structure to explore. Event-level datasets must explicitly record their vote after completing temporal analysis.

### Grid Modes

| Mode | Behavior |
|------|----------|
| `NO_ADJUSTMENTS` | Grid is derived entirely from the intent. Dataset votes are informational only. Ready for aggregation immediately. |
| `ALLOW_ADJUSTMENTS` | Grid waits until all event-level datasets have voted. Their suggested cadence and date ranges can refine the grid before it locks. |

### Locking

Once all votes are in (or if mode is `NO_ADJUSTMENTS`), the grid is **locked**. Locking:
- Prevents further votes
- Generates the actual `grid_dates` list (sequence of ISO date strings)
- The grid becomes the deterministic backbone for all downstream aggregation

```python
grid = SnapshotGrid.from_intent(intent, datasets)
# ... datasets vote during 01a-01c ...
grid.lock()
# grid.grid_dates is now a fixed list: ["2023-01-07", "2023-01-14", ...]
```

### Grid Size Estimation

Before locking, you can estimate the training matrix size:

```
estimated_rows = n_entities * len(grid_dates)
```

For 10,000 entities with weekly cadence over 12 months (~52 dates), that is ~520,000 training rows. This helps you decide whether to adjust cadence or observation window before committing.

## How the Grid Flows Through the Pipeline

```
Notebook 00
  └── IntentDefaultsEngine.suggest() → IntentConfig
       └── SnapshotGrid.from_intent() → initial grid
            │
Notebooks 01a-01c (per event dataset)
  └── Temporal analysis → DatasetGridVote
       └── grid.record_vote() (if ALLOW_ADJUSTMENTS mode)
            │
Notebook 01d
  └── grid.lock() → grid_dates finalized
       └── Aggregation uses grid_dates as as_of_date values
            │
Notebook 03
  └── TemporalMerger builds spine from grid_dates
       └── Merged matrix: (entity_id, as_of_date) rows
            │
Notebook 07
  └── Training grid = objective-specific subset of grid_dates
       └── Reactive: dense recent dates
       └── Stable: broader historical dates
            │
Notebook 08
  └── Entity-grouped temporal CV on training grid
       └── Purge gap applied per fold
```

## Putting It All Together: A Worked Example

Suppose you are building a churn model for a SaaS company:

- **Objective**: Immediate Risk (PRIMARY)
- **Posture**: Stable (you want broad historical context)
- **Prediction horizon**: 90 days
- **Data span**: 2 years of data, 15,000 customers

The `IntentDefaultsEngine` derives:

| Variable | Formula | Value |
|----------|---------|-------|
| `recent_window_days` | max(180, 3*90) | **270 days** |
| `label_window_days` | H = 90 | **90 days** |
| `purge_gap_days` | H + 14 = 90 + 14 | **104 days** |
| `prediction_horizons` | [90/3, 180/3, 90] | **[30, 60, 90]** |
| `cadence_interval` | weekly (immediate, H=90) | **weekly** |
| `split_strategy` | temporal | **temporal** |

The snapshot grid generates weekly dates from the start of data availability minus the observation window through the end of the usable date range (end of data minus purge_gap minus label_window). With 2 years of data and weekly cadence, that is roughly 50-60 grid dates.

Training matrix: 15,000 entities * 55 dates = **~825,000 rows**.

At each row, features use 270 days of lookback, there is a 104-day purge gap, and the label is whether the customer churned in the following 90 days.

## Formula Quick Reference

### Observation Window

```
Immediate + Reactive:    4 * H
Immediate + Stable:      max(180, 3 * H)
Disengagement + Reactive: 2 * H
Disengagement + Stable:  clamp(0.2 * data_span, 180, 365)
Renewal + Reactive:      cycle
Renewal + Stable:        clamp(2 * cycle, 365, 730)
```

### Purge Gap

```
Immediate:    H + 14
Disengagement: H + 21
Renewal:      label_window + 21
```

### Cadence

```
Immediate + Reactive + H<=30: daily
Immediate + otherwise:        weekly
Disengagement + Reactive:     weekly
Disengagement + Stable:       monthly
Renewal:                      based on cycle tier (monthly->weekly, quarterly->biweekly, yearly->monthly)
```

## Next Steps

- [[Model Intent and Objective Support]] - How objectives are declared and validated
- [[Architecture]] - Overall system architecture
- [[Exploration Loop]] - The notebook workflow in detail
