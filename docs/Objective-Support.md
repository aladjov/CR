# ObjectiveSupport Lifecycle in the Exploration → Modeling Pipeline

This document shows how the notebook process evolves objective status:

```
evidence → feasibility → readiness → decision
```

…and how we persist this evolution in a small set of **ObjectiveSupport** tracking objects.

---

# Diagram: Two Dataset Tracks + Three Tracking Objects (Mermaid)

Requirements reflected here:

- **Two dataset tracks** in the notebook panel:
  - **Entity dataset track**
  - **Event dataset track**
- **Only three tracking objects** total:
  1) ObjectiveSupport for Entity dataset
  2) ObjectiveSupport for Event dataset
  3) ObjectiveSupport for the **merged feature set**
- Tracking boxes explicitly show the lifecycle evolution (evidence → feasibility → readiness → decision).

```mermaid
flowchart TB

  %% =============================
  %% NOTEBOOKS (LEFT - VERTICAL)
  %% =============================
  subgraph NB["Notebooks (process)"]
    direction TB

    N00["<div style='text-align:left'>
    <b>00 Intent Contract</b><br/>
    • identify datasets<br/>
    • detect entity_id + timestamps<br/>
    • declare objectives + posture + horizon<br/>
    • derive control variables
    </div>"]:::nb

    %% Entity dataset track
    subgraph T1["Entity dataset track"]
      direction TB
      EN["<div style='text-align:left'>
      <b>Entity exploration (entity-shaped)</b><br/><br/>
      01 Data Discovery<br/>
      02 Source Integrity<br/><br/>
      = Evidence stage
      </div>"]:::nb_ds
    end

    %% Event dataset track
    subgraph T2["Event dataset track"]
      direction TB
      EV["<div style='text-align:left'>
      <b>Event exploration (event-shaped → aggregated)</b><br/><br/>
      01 Data Discovery<br/>
      01a Temporal Deep Dive<br/>
      01b Temporal Quality<br/>
      01c Temporal Patterns<br/>
      01d Event Aggregation<br/>
      02 Source Integrity<br/><br/>
      = Evidence stage
      </div>"]:::nb_ds
    end

    N03["<div style='text-align:left'>
    <b>03 Dataset Merge</b><br/>
    • keys + coverage overlap<br/>
    • as-of-date + cadence alignment<br/>
    • window compatibility<br/><br/>
    = Feasibility stage
    </div>"]:::nb

    N04_05["<div style='text-align:left'>
    <b>04–05 Column Deep Dive + Relationship Analysis</b><br/>
    • type validation, skewness<br/>
    • correlations, redundancy<br/><br/>
    = Readiness stage (part 1)
    </div>"]:::nb

    N06["<div style='text-align:left'>
    <b>06 Feature Opportunities</b><br/>
    • feature planning<br/>
    • leakage checks<br/><br/>
    = Readiness stage (part 2)
    </div>"]:::nb

    N07_08["<div style='text-align:left'>
    <b>07–08 Modeling Readiness + Baseline Experiments</b><br/>
    • training grid, split policy<br/>
    • baseline models, stability checks<br/><br/>
    = Decision stage
    </div>"]:::nb

    %% Vertical process
    N00 --> EN
    N00 --> EV
    EN --> N03
    EV --> N03
    N03 --> N04_05
    N04_05 --> N06
    N06 --> N07_08
  end

  %% =============================
  %% TRACKING OBJECTS (RIGHT)
  %% =============================
  subgraph TR["Tracking (ObjectiveSupport objects)"]
    direction TB

    OS_EN["ObjectiveSupport (Entity dataset)<br/><br/>01–02 → evidence"]:::os
    OS_EV["ObjectiveSupport (Event dataset)<br/><br/>01–02 → evidence"]:::os
    OS_M["ObjectiveSupport (Merged feature set)<br/><br/>03 → feasibility<br/>04–06 → readiness<br/>07–08 → decision"]:::os
  end

  %% Cross-panel writes/updates
  EN -. write/update .-> OS_EN
  EV -. write/update .-> OS_EV
  N03 -. derive/update .-> OS_M
  N04_05 -. update .-> OS_M
  N06 -. update .-> OS_M
  N07_08 -. update .-> OS_M

  %% =============================
  %% Styling
  %% =============================
  classDef nb fill:#f6f8fa,stroke:#3b3b3b,stroke-width:1px;
  classDef nb_ds fill:#ffffff,stroke:#3b3b3b,stroke-width:1px;
  classDef os fill:#f2f2f2,stroke:#999999,stroke-width:1px;

```

---

# ObjectiveSupport YAML (single schema, different scopes)

The same schema supports:
- dataset-level (entity/event)
- merged feature-set level

## Dataset-level ObjectiveSupport (Entity or Event)

```yaml
objective_support:
  scope: dataset
  dataset_id: customer_profile            # or: edi_events
  dataset_kind: entry                     # entry | event
  lifecycle_stage: evidence               # produced by notebooks 01–02

  objectives:
    immediate_risk:
      strength: weak | medium | strong
      confidence: low | medium | high
      signals: {}
      blockers: []

    disengagement:
      strength: weak | medium | strong
      confidence: low | medium | high
      signals: {}
      blockers: []

    renewal_risk:
      strength: weak | medium | strong
      confidence: low | medium | high
      signals: {}
      blockers: []
```

## Merged feature-set ObjectiveSupport

```yaml
objective_support:
  scope: merged_featureset
  modeling_table: weekly_customer_snapshot_v1
  lifecycle_stage: feasibility            # 03 → feasibility, 04-06 → readiness, 07-08 → decision

  datasets_used:
    - customer_profile
    - edi_events

  merge_context:
    cadence: weekly
    as_of_date_strategy: global
    entity_coverage_overlap: 0.87

  objectives:
    immediate_risk:
      strength: medium
      blockers: []
      readiness_score: 0.72               # (populated in 06)
      decision: proceed | defer | reject  # (populated in 07-08)

    disengagement:
      strength: strong
      readiness_score: 0.89
      decision: proceed

    renewal_risk:
      strength: medium
      readiness_score: 0.58
      decision: defer
      blockers:
        - insufficient contract history
```

---

# Notes

- **01–02** are the **Evidence** stage (per dataset). NB01 records objective support via `derive_objective_support()`. NB01a-01c contribute snapshot grid votes. NB02 runs integrity checks.
- **03** is the first place we can truthfully evaluate **Feasibility** (because joining + cadence alignment exist).
- **04–06** upgrade feasible objectives to **Readiness** (feature plan exists, leakage checked).
- **07–08** record the **Decision** based on baseline experiments and stability.
