# Phased Plan

## Phase 0 — Baseline Guardrails and Vocabulary

### Goal
Introduce minimal shared vocabulary and validation utilities so later phases fail loudly instead of silently leaking.

### Questions
- Should the canonical “as_of” column be named exactly `feature_timestamp`, or do you want a separate `as_of` column and keep `feature_timestamp` derived?

### Best insertion point analysis
- Identify existing naming/constants module that already defines timestamp column names.
- Identify existing validation/leakage gate logic and enhance it rather than introducing a new validator.

### Tests (write first)
- Snapshot output missing `feature_timestamp` must fail validation.
- If `label_timestamp` exists and any row has `feature_timestamp > label_timestamp`, validation must fail.
- Validation must be a no-op for datasets without labels (feature timestamp required, label timestamp optional).

### Implementation steps
- Introduce canonical constants:
  - `EVENT_TIMESTAMP_COLUMN`
  - `FEATURE_TIMESTAMP_COLUMN`
  - `LABEL_TIMESTAMP_COLUMN` (optional)
- Add `validate_temporal_integrity(df, stage_name)` as a reusable utility.

### Definition of done
- Tests pass.
- Ruff passes.
- No pipeline templates changed yet.

---

## Phase 1 — Ensure Timestamps Are Derived for All Sources and Preserved

### Goal
All sources (event and entity) must flow through Landing and preserve derived timestamp columns through Bronze outputs.

### Questions
- For non-target sources, should `label_timestamp` be absent or present-but-null?
- For entity datasets without a true update timestamp, what is the fallback strategy (error vs ingestion timestamp vs configured timestamp column)?

### Best insertion point analysis
- Find the existing timestamp derivation logic used in exploration (or landing) and reuse it in generated production landing.
- Modify bronze entity loader to consume landing output rather than re-implementing timestamp derivation.

### Tests (write first)
- For every configured source: landing output includes `feature_timestamp`.
- Bronze entity reads from landing output and retains `feature_timestamp`.
- Bronze event aggregation does not drop `feature_timestamp`.
- Anti-pattern: any bronze output missing `feature_timestamp` causes validation failure.

### Implementation steps
- Generate landing for all sources.
- Change bronze entity to read landing output.
- Preserve timestamp columns across bronze event aggregation.

### Definition of done
- All sources produce bronze outputs with `feature_timestamp`.
- Tests cover both event and entity pathways.
- Ruff passes.

---

## Phase 2 — Introduce Explicit `as_of` and Make Snapshot Anchor Uniform

### Goal
Snapshots represent entity state at a single decision moment `as_of`. All rows in a snapshot partition share the same `feature_timestamp = as_of`.

### Questions
- What is the canonical representation of `as_of` in production runs?
  - passed in as a parameter?
  - derived from “latest complete ingestion time”?
  - provided by the orchestrator?

### Best insertion point analysis
- Locate the first stage where aggregation occurs and enforce filtering there.
- Prefer a single function that applies time filtering before aggregation to avoid repeated logic per dataset.

### Tests (write first)
- Given events before and after `as_of`, snapshot features must not change when post-`as_of` events are added.
- All snapshot rows in a partition must have identical `feature_timestamp`.
- Anti-pattern: any implementation that uses per-entity max(event_timestamp) as anchor should fail a targeted test.

### Implementation steps
- Add an `as_of` parameter.
- Filter event datasets by `event_timestamp <= as_of` before any windowing/aggregation.
- Set `feature_timestamp = as_of` for all entity rows.
- Ensure entity datasets select “latest record <= as_of” when historical rows exist.

### Definition of done
- Feature timestamp is uniform and equals `as_of`.
- Tests demonstrate cutoff correctness.
- Ruff passes.

---

## Phase 3 — Silver Merge Alignment Contract

### Goal
Merging becomes safe and simple because every dataset is already transformed to entity-level features at the same `as_of`.

### Questions
- On mismatch, should we hard fail or drop mismatched rows?
  - Recommended: hard fail (break loudly).

### Best insertion point analysis
- Add a single alignment assertion in the merge entry point.
- Avoid per-join checks if there is a shared invariant across all sources.

### Tests (write first)
- If any source has a different `feature_timestamp` than the snapshot `as_of`, merge must fail.
- Merged output retains exactly one `feature_timestamp` column.
- Anti-pattern: merging event-level rows directly must be rejected (shape/keys mismatch test).

### Implementation steps
- Add strict alignment checks in silver merge.
- Ensure merge is purely on entity key(s).
- Validate temporal integrity post-merge.

### Definition of done
- Multi-dataset merge is deterministic and aligned.
- Ruff passes.

---

## Phase 4 — Labels as a Separate Layer for Three Model Types

### Goal
Compute and attach labels deterministically from the snapshot and post-`as_of` outcomes, without contaminating features.

### Questions
- Do you want labels stored in separate label tables per model type, or columns on the snapshot table?
  - Recommended: separate label tables, joined by (entity_id, as_of).

### Best insertion point analysis
- Reuse existing exploration label builders if present, adapted to the production pipeline.
- Centralize label computation to avoid duplication per model type.

### Tests (write first)
- Label correctness for:
  - early termination horizon
  - renewal non-renewal with grace
  - inactivity churn definition
- Anti-pattern: labels must never depend on events before `as_of` for the “outcome event”, and must never use features computed after `as_of`.
- `feature_timestamp <= label_timestamp` always.

### Implementation steps
- Implement three label builders with constants:
  - early termination horizon
  - renewal lookahead + grace
  - inactivity threshold + prediction horizon
- Exclude label columns from feature computation by enforcement.

### Definition of done
- Labels reproducible for a fixed `as_of`.
- Tests cover edge cases (missing renewal date, missing activity events, termination already happened).
- Ruff passes.

---

## Phase 5 — Training Tier Selection From a Single Snapshot Store

### Goal
Train 18 models (3 model types × 3 algorithms × 2 tiers) from the same snapshot partitions.

### Questions
- Should the slow tier train on daily partitions (longer history) or weekly sampled partitions (less compute)?

### Best insertion point analysis
- Implement partition selection as a pure function returning a list of `as_of` dates.
- Avoid entangling partition selection with model training logic.

### Tests (write first)
- Fast tier selects last N days.
- Slow tier selects last M days or weekly-sampled dates, deterministically.
- Holdout split selects the most recent K partitions.
- Anti-pattern: any random row-based split should be rejected by configuration validation.

### Implementation steps
- Implement deterministic partition selectors:
  - fast: last N daily partitions
  - slow: last M partitions, optionally weekly sampled
- Ensure training is time-consistent.

### Definition of done
- Training orchestration can emit 18 model runs deterministically.
- Ruff passes.

---

## Phase 6 — Cleanup, Consolidation, and End-to-End Regression Tests

### Goal
Remove legacy timestamp hacks and hard-coded references, and add end-to-end regression coverage.

### Best insertion point analysis
- Remove only after tests prove new behavior is stable.
- Consolidate constants and shared utilities rather than duplicating across templates.

### Tests (write first)
- End-to-end: build snapshot + merge + labels for a small multi-source fixture.
- Regression: ensure adding post-`as_of` events does not change snapshot.
- Ensure no scalar timestamp usage remains.

### Implementation steps
- Remove deprecated scalar timestamp attributes and old naming constants.
- Align pandas and PySpark template behavior.
- Add integration test fixtures.

### Definition of done
- End-to-end tests pass.
- Ruff passes.
- No deprecated timestamp paths remain.

---

# How to Keep Agent Context Small

For each phase, provide the agent only:

- the phase goal
- the relevant spec excerpt (snapshot invariants + the phase rule)
- the list of files to inspect (max 5)
- the tests to write first
- the “definition of done” checklist

Avoid giving the entire plan or historical context unless needed.

---

# Recommended Starting Point

Start with Phase 1 (timestamps derived/preserved). It is foundational, testable, and low-risk.

Then Phase 2 (introduce `as_of` + uniform snapshot anchor) as the single most important semantic change.

Both phases can be implemented with tight scope and strong tests, preparing the codebase for labels, merge alignment, and multi-model training without ambiguity.
