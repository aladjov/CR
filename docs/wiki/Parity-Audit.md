# Parity Audit — Production Replay of Exploration Intent

This document specifies a framework subsystem that guarantees, **before any data flows**, that the production pipeline a job is about to run is a faithful replay of the exploration intent declared by the same job's notebooks. It exists to make a recurring class of bug — where production silently applies operations exploration would have skipped (or vice versa) — structurally impossible to ship.

## 1. Purpose

A medallion engagement runs as a single Databricks job whose tasks are the numbered notebooks `00 → 10` plus a parity pre-flight at `-1` (named for sort order: `-1_parity_contract.ipynb` always runs before `00_start_here.ipynb`). The first half (`00–09`) is exploration; `10` generates a production pipeline by translating the registry of recommendations exploration produced into deployable scripts. The two halves run on different code paths:

- **Exploration** uses framework functions from `analysis.auto_explorer`, `stages.profiling`, `stages.lifecycle`, etc. Cells in NB00–NB09 invoke these directly.
- **Production** uses Jinja templates inside `generators.pipeline_generator.databricks_renderer` (and a parallel local renderer in `renderer.py`). The templates emit standalone landing/bronze/silver/gold/training scripts that are then executed by NB10's runner.

These two paths must apply the same operations to the same datasets under the same conditions. They often diverge silently. Examples that have shipped to engagements:

| # | Symptom | Root cause | Detected by |
|---|---|---|---|
| 1 | Silver cohort collapsed from 85,745 entities to 2 | Production landing applied `apply_history_window` on a frame whose `feature_timestamp` had a `2055-07-31` sentinel; exploration skipped the lookback entirely for `INTERVAL_START_TIME` datasets | SQL inspection after a 20-hour silver run |
| 2 | `bronze_entity_account` skipped despite stale upstream | Resume gate had no notion of "target_derive mutated my source"; only caught when the silver `churned` distribution was visibly wrong | Post-hoc SQL audit |
| 3 | `landing_account` not found 30s into bronze | Production landing ran datasets alphabetically; exploration ran them in dependency order so sibling temp views existed | Runtime crash after a partial pipeline run |
| 4 | `user_extensions.py` NameError at production landing | Harvester wrote `@cr.register` decorators without importing `cr` | Runtime crash on production execution |

Each of these was detectable *from the source code alone*, with no data, the moment the engagement notebooks were committed. We discovered them by running 20-hour pipelines instead.

The parity audit closes this gap: it runs as the first task of the job, statically validates that production will faithfully replay exploration, and aborts the job before any data moves if the validation fails. A second, dynamic confirmation runs after the pipeline completes, comparing actual exploration and production traces row-count by row-count.

## 2. Goals and non-goals

### Goals

- **Active maintenance**: the audit runs on every job, not as an occasional check. Drift is caught at the moment it's introduced.
- **Structural parity is pre-execution**: any divergence detectable from source code is reported before exploration starts. The 20-hour wait for the symptom to surface is gone by construction.
- **Numerical parity is post-pipeline**: divergences that depend on actual data (row counts, distributional drift) are reported after the pipeline runs.
- **Zero per-engagement maintenance burden**: engagements declare intent through existing mechanisms (project_context, registry, semantics_overrides). The audit derives everything else from those declarations and the framework's typed primitive surface.
- **Framework changes are testable in isolation**: a new framework branch ships with a synthetic fixture and an expected audit outcome. Framework drift is caught in framework CI, not in engagement runs.

### Non-goals

- **Behavioural equivalence at column-value granularity**. Two implementations producing the same row count but different aggregations on a feature column is not caught. Detecting this requires running both paths on the same data and diffing per-cell, which is too expensive for routine use.
- **Validation of registered user code semantics**. If `@cr.register`-decorated user code has a bug (e.g., wrong arithmetic in `derive_churn_target`), the audit verifies the function is called in both paths but doesn't validate the function body's correctness.
- **Replacement of the existing recommendation_registry**. Recommendations are *what should be applied*; the audit verifies the manifest of *what is applied* on both sides. They live in different layers and complement each other.

## 3. Conceptual model

### 3.1 Apply primitives

A function is an **apply primitive** if and only if its primary effect is to produce a transformed DataFrame (or DataFrame-typed dataset map) that downstream code will use as input. Every apply primitive is decorated with `@apply_op` and registered with a `kind` from a closed `ApplyOpKind` enum.

Decoration is a one-time act when the framework is designed. After decoration, the audit derives everything else automatically. Authors of new framework features add a new enum value and a `@apply_op` decoration; no separate registry is maintained.

```python
from customer_retention.parity import apply_op, ApplyOpKind

@apply_op(kind=ApplyOpKind.TEMPORAL_LOOKBACK)
def apply_temporal_lookback(df, time_col, intent):
    ...

@apply_op(kind=ApplyOpKind.LIFECYCLE_ENRICH)
def enrich_lifecycle_dataset(raw_df, config):
    ...

@apply_op(kind=ApplyOpKind.DATETIME_DERIVE)
def derive_extra_datetime_features(df, source_columns, reference_column, mask_future_columns):
    ...
```

### 3.2 The manifest

A **manifest** is an ordered list of apply-op invocations, scoped to a dataset:

```yaml
- dataset: contract
  kind: LIFECYCLE_ENRICH
  kwargs_fingerprint:
    valid_from_column: CONTRACT_START_DATE
    valid_to_columns: [BILLING_TERMINATION_DATE]
    status_column: CONTRACT_STATUS
    terminal_status_values: [Cancelled]
    on_corrupt_row: skip
  source: 00_start_here.ipynb / cell id=00025001 / line 6
- dataset: contract
  kind: FEATURE_TIMESTAMP_DERIVE
  kwargs_fingerprint:
    time_column: event_timestamp
  source: generated/landing/landing_contract.py / run_landing / line 41
- dataset: contract
  kind: LABEL_TIMESTAMP_DERIVE
  ...
```

The audit produces two manifests on every job:

- **Exploration manifest**: derived from a static AST walk of every scheduled exploration notebook (NB00–NB09) and the framework functions they invoke. The walk follows `@apply_op` registrations and captures the gate condition dominating each call site.
- **Production manifest**: derived from running the renderer in-memory against synthesised inputs, then statically walking the rendered output. No data flows; the renderer is a pure function of (project_context, intent, registry, findings stub).

### 3.3 The diff

Parity holds if for every dataset, the multiset of `(kind, kwargs_fingerprint)` pairs in the exploration manifest equals that in the production manifest. A divergence is a **ParityGap** with structured location traces on both sides.

The diff is the audit's primary output. When it's empty, the job proceeds. When it's non-empty, the job aborts with a structured exit listing every gap.

### 3.4 Three knowability checkpoints

The audit fires at three points in time, each with progressively richer inputs:

| Checkpoint | Trigger | Inputs available | Parity surface checkable |
|---|---|---|---|
| **T0** — job start | `-1_parity_contract.ipynb` runs as the first task | All notebook source code; framework code; declared project_context, intent, registry calls | Landing layer (structurally fully determined) |
| **T1** — post-findings | Re-invocation after NB01–NB09 produce findings | + recommendation registry, findings YAMLs | Bronze / silver / gold / training |
| **T3** — post-pipeline | Final cell of the production run | + actual production execution trace, row counts per stage | Runtime confirmation that what was predicted at T0/T1 is what actually ran |

T0 is the gate that prevents 20-hour wastes. T1 is the gate that prevents incorrect aggregation/feature shapes from reaching silver. T3 is the safety net for runtime-data-dependent kwargs that couldn't be predicted statically.

### 3.5 The schedule as part of the contract

A Databricks job is a graph of notebook tasks. The audit treats the schedule itself as audit surface:

- The **outer schedule** is the order of `00_…` through `10_…` notebooks in the job definition. Defaults to the numeric prefix when no explicit schedule is provided.
- The **inner schedule** is NB10's `run_pipeline` cell, which orchestrates landing → target_derive → bronze → silver → gold → training and triggers each generated sub-notebook via `dbutils.notebook.run()`.

The audit reads both. Schedule-level failures include:

- A notebook with `@apply_op` call sites that isn't in the outer schedule (orphan exploration)
- A dataset registered in NB00 but missing from the inner schedule's landing stage (orphan registration)
- A landing notebook in the inner schedule for a dataset NB00 never registered (orphan production)
- Topological order violations (NB10 runs before NB01; landing for `account` runs before landing for `contract` when `account`'s cohort filter depends on `contract`)

## 4. Architecture

### 4.1 Components

```
customer_retention/
├── parity/
│   ├── __init__.py
│   ├── decorator.py              ── @apply_op + APPLY_REGISTRY
│   ├── kinds.py                  ── ApplyOpKind enum (closed set)
│   ├── exploration_scan.py       ── AST walker for NB00-NB09 cells
│   ├── production_scan.py        ── in-memory renderer + AST walker for generated scripts
│   ├── schedule.py               ── outer + inner DAG parsing
│   ├── manifest.py               ── manifest dataclass + diff
│   ├── gaps.py                   ── ParityGap dataclass + structured exit codes
│   ├── trace.py                  ── runtime tracer for T3 dynamic confirmation
│   └── audit.py                  ── top-level audit_landing / audit_bronze / audit_pipeline
└── …

debug/<engagement>/
├── -1_parity_contract.ipynb      ── new; runs at job start (T0)
├── 00_start_here.ipynb           ── existing
├── …
├── 09_…
├── 10_spec_generation.ipynb      ── existing; T1 audit added near end of cell sequence
└── …
```

### 4.2 The audit notebook (`-1_parity_contract.ipynb`)

This notebook is added to the engagement directory and runs as the first task of the job. Its body is approximately:

```python
# Cell 1: setup
import sys; sys.path.insert(0, f"{FRAMEWORK_REPO_ROOT}/src")
from customer_retention.parity import audit_landing, AuditOutcome

# Cell 2: locate the engagement directory and the schedule
ENGAGEMENT_DIR = "/Workspace/Repos/.../debug/<engagement>"
OUTER_SCHEDULE = None  # default: infer from numeric prefix convention
INNER_SCHEDULE = None  # default: parse NB10's run_pipeline cell

# Cell 3: run the audit
outcome = audit_landing(
    engagement_dir=ENGAGEMENT_DIR,
    outer_schedule=OUTER_SCHEDULE,
    inner_schedule=INNER_SCHEDULE,
)

# Cell 4: structured exit
if outcome.has_gaps:
    print(outcome.format_report())
    dbutils.notebook.exit(outcome.to_failed_json())
else:
    print(outcome.format_summary())
```

A failed audit produces a structured exit that the Databricks workflow interprets as a task failure, halting the rest of the job.

### 4.3 The data flow

```
                ┌────────────────────────────────────────────────────────┐
                │  T0 — Job task #1: -1_parity_contract.ipynb            │
                │                                                         │
                │  parses every NB source + framework + renderer         │
                │  ▼                                                      │
                │  scan_exploration_manifest()  ◄── AST walks NB00-NB09  │
                │  scan_production_manifest()   ◄── runs renderer in mem │
                │  diff_manifests()             ◄── computes ParityGaps  │
                │                                                         │
                │  empty diff  → pass, job continues                     │
                │  non-empty  → structured exit, job halts               │
                └────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         exploration runs (NB00 → NB09)
                                       │
                                       ▼
                ┌────────────────────────────────────────────────────────┐
                │  T1 — Inside NB10, after generate_databricks cell      │
                │                                                         │
                │  re-runs audit with findings_stub replaced by actual   │
                │  findings; audits bronze + silver + gold + training    │
                │  surface                                                │
                └────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         pipeline runs (landing → training)
                                       │
                                       ▼
                ┌────────────────────────────────────────────────────────┐
                │  T3 — Final cell of pipeline: confirm_trace_parity()   │
                │                                                         │
                │  reads exploration_trace.yaml (recorded by @apply_op   │
                │  decorator when NB00-NB09 ran with tracing on)         │
                │  reads production_trace.yaml (recorded the same way   │
                │  when the production scripts ran)                      │
                │  diffs row counts per stage, flags drift               │
                └────────────────────────────────────────────────────────┘
```

## 5. The `ApplyOpKind` enum

A closed set of operation categories. Adding a new kind is a deliberate framework act, equivalent to adding a new cell tag.

```python
from enum import Enum

class ApplyOpKind(str, Enum):
    # Landing stage
    LIFECYCLE_ENRICH        = "landing.lifecycle_enrich"
    SAMPLE_FILTER           = "landing.sample_filter"
    LANDING_FILTER          = "landing.filter"
    KEY_RESOLUTION          = "landing.key_resolution"
    FEATURE_TIMESTAMP_DERIVE= "landing.feature_timestamp_derive"
    LABEL_TIMESTAMP_DERIVE  = "landing.label_timestamp_derive"
    LABEL_AVAILABLE_FLAG    = "landing.label_available_flag"
    DATETIME_DERIVE         = "landing.datetime_derive"
    TEMPORAL_LOOKBACK       = "landing.temporal_lookback"
    TIMESTAMP_NORMALIZE     = "landing.timestamp_normalize"

    # Target derivation
    TARGET_DERIVE           = "target_derive.user_code"

    # Bronze
    BRONZE_AGGREGATE        = "bronze.aggregate"
    BRONZE_VALUE_COUNTS     = "bronze.value_counts"

    # Silver
    SILVER_TEMPORAL_MERGE   = "silver.temporal_merge"
    SILVER_DERIVED_FEATURE  = "silver.derived_feature"
    SILVER_HOLDOUT_MASK     = "silver.holdout_mask"
    SILVER_TARGET_LABEL_MAP = "silver.target_label_map"

    # Gold
    GOLD_TRANSFORMATION     = "gold.transformation"
    GOLD_ENCODING           = "gold.encoding"
    GOLD_FEATURE_SPEC_GATE  = "gold.feature_spec_gate"

    # Training
    TRAINING_SPLIT          = "training.split"
    TRAINING_FIT            = "training.fit"
    TRAINING_EVALUATE       = "training.evaluate"
```

Each kind has, by convention:

- A canonical implementation function (where `@apply_op` lives)
- A documented kwargs signature (what's recorded in the manifest fingerprint)
- A gate predicate (when the operation fires — extractable from the dominating `if` in code, or declared as a `gate=` argument to `@apply_op` when the gate isn't structurally co-located)

## 6. Component specifications

### 6.1 `parity.decorator.apply_op`

```python
def apply_op(
    kind: ApplyOpKind,
    *,
    gate: Optional[str] = None,   # symbolic gate expression, default: "True" (always-on)
    capture_kwargs: Optional[Set[str]] = None,  # which kwargs to fingerprint, default: all
) -> Callable[[F], F]:
    """
    Decorate a function as an apply primitive.

    Side effects:
    - Adds (qualified_name, kind, gate, capture_kwargs) to APPLY_REGISTRY.
    - Wraps the function so that, when CR_PARITY_TRACE=1, each invocation
      appends (kind, dataset_hint, kwargs_fingerprint, input_rows, output_rows,
      call_order) to a thread-local trace buffer.

    The runtime tracing is enabled by NB00 and by the production pipeline runner.
    Disabled by default in tests and framework CI to avoid I/O overhead.
    """
```

`APPLY_REGISTRY` is a process-global `Dict[str, ApplyOpDescriptor]` keyed by qualified function name. It is the single source of truth for "what counts as an apply primitive in this codebase".

### 6.2 `parity.exploration_scan.scan_exploration_manifest`

```python
def scan_exploration_manifest(
    notebook_paths: List[Path],
    framework_root: Path,
) -> Manifest:
    """
    Static AST walk of every scheduled exploration notebook and the framework
    functions reachable from those notebooks. For each apply_op call site,
    capture:
      - kind (from APPLY_REGISTRY lookup)
      - dataset (resolved from kwargs or from an enclosing `with apply_context(...)`)
      - kwargs (literal-evaluated where possible; symbolic otherwise)
      - dominating gate (the AST of the nearest enclosing `if`)
      - source location (notebook path, cell id, line)

    Returns an ordered Manifest with one entry per call site.
    """
```

The walker resolves imports cell-by-cell (`from X import Y as Z` maps local name `Z` to qualified `X.Y`), then identifies `Call` nodes whose qualified callee is in `APPLY_REGISTRY`. For each, it walks the AST upward to the nearest dominating `If` and captures the condition.

For complex gates (multi-conditional, function-scoped), the gate is captured as an AST and evaluated symbolically against synthesised inputs during the diff phase.

### 6.3 `parity.production_scan.scan_production_manifest`

```python
def scan_production_manifest(
    project_context: ProjectContext,
    intent: IntentConfig,
    registry: RecommendationRegistry,
    findings_stub: Optional[Dict[str, ExplorationFindings]] = None,
    *,
    scope: AuditScope = AuditScope.LANDING,
) -> Manifest:
    """
    Runs `findings_parser` and the appropriate renderer (databricks or local)
    against the supplied inputs, producing the would-be-generated scripts as
    in-memory strings. Walks the strings via AST to extract apply_op call sites.

    For T0 (scope=LANDING), findings_stub is auto-generated as skeletal
    ExplorationFindings per dataset — enough to satisfy the parser's landing
    code path. For T1 (scope=BRONZE|SILVER|GOLD|TRAINING), actual findings
    are required and loaded from the namespace.

    The renderer must emit apply_op calls at the top level of run_landing(),
    run_bronze() etc. — no helper indirection. This is a lint rule enforced
    by `test_renderer_emits_flat_apply_ops.py`.
    """
```

### 6.4 `parity.manifest.Manifest` and `diff_manifests`

```python
@dataclass(frozen=True)
class ManifestEntry:
    dataset: str
    kind: ApplyOpKind
    kwargs_fingerprint: Mapping[str, Any]
    call_order: int
    source_location: SourceLocation

@dataclass(frozen=True)
class Manifest:
    entries: Tuple[ManifestEntry, ...]

    def by_dataset(self, dataset: str) -> Tuple[ManifestEntry, ...]: ...
    def kinds_for(self, dataset: str) -> FrozenSet[ApplyOpKind]: ...

def diff_manifests(
    exploration: Manifest,
    production: Manifest,
) -> List[ParityGap]:
    """
    Compute the set difference, per dataset, of (kind, kwargs_fingerprint)
    pairs. Returns one ParityGap per divergence with structured location
    pointers into both sides.
    """
```

### 6.5 `parity.gaps.ParityGap`

```python
class GapKind(str, Enum):
    PRODUCTION_ONLY      = "production_only"       # production emits, exploration skips
    EXPLORATION_ONLY     = "exploration_only"      # exploration emits, production skips
    KWARGS_MISMATCH      = "kwargs_mismatch"       # both emit, fingerprints differ
    ORDER_MISMATCH       = "order_mismatch"        # both emit, sequence differs
    ORPHAN_NOTEBOOK      = "orphan_notebook"       # apply_op calls in unscheduled notebook
    ORPHAN_REGISTRATION  = "orphan_registration"   # dataset registered, no production emit
    ORPHAN_PRODUCTION    = "orphan_production"     # production emit, no exploration call
    SCHEDULE_TOPOLOGY    = "schedule_topology"     # inner-DAG dependency violation

@dataclass(frozen=True)
class ParityGap:
    gap_kind: GapKind
    dataset: str
    op_kind: Optional[ApplyOpKind]
    exploration_location: Optional[SourceLocation]
    production_location: Optional[SourceLocation]
    detail: str

    def format(self) -> str: ...
```

### 6.6 Runtime tracing (T3)

The `@apply_op` decorator wraps every primitive with a thin tracer that, when `CR_PARITY_TRACE=1`, appends to a thread-local buffer:

```python
def _wrap(fn, descriptor):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _tracing_active():
            return fn(*args, **kwargs)
        dataset = _resolve_dataset_hint(args, kwargs, descriptor)
        fingerprint = _fingerprint_kwargs(kwargs, descriptor.capture_kwargs)
        input_rows = _safe_row_count(args[0]) if args else None
        result = fn(*args, **kwargs)
        output_rows = _safe_row_count(result)
        _record(descriptor.kind, dataset, fingerprint, input_rows, output_rows)
        return result
    return wrapper
```

NB00 sets `CR_PARITY_TRACE=1` at session start; the trace flushes to `<namespace>/apply_trace_exploration.yaml` when NB09 (the last exploration notebook) exits.

The production pipeline runner sets the same flag; the trace flushes to `<namespace>/apply_trace_production.yaml`.

The final T3 cell diffs both traces. Row-count divergences exceeding a per-kind tolerance (default ±0.5%, configurable per kind) fail the cell.

### 6.7 The schedule parsers

```python
@dataclass(frozen=True)
class JobSchedule:
    outer: Tuple[ScheduledNotebook, ...]
    inner: Tuple[GeneratedStage, ...]

def parse_outer_schedule(
    engagement_dir: Path,
    explicit_schedule_file: Optional[Path] = None,
) -> Tuple[ScheduledNotebook, ...]:
    """
    If explicit_schedule_file is given, parse it (YAML). Otherwise infer from
    numeric prefixes of notebooks in engagement_dir: prefix_*.ipynb scheduled
    in lexicographic-then-numeric order. The `-` prefix on -1_parity_contract
    sorts before any `00_*` notebook, which is why we chose negative numbering
    for pre-exploration tasks (-2, -3, … are reserved for future
    pre-execution checks: schema validation, raw-source reachability, etc.).
    """

def parse_inner_schedule(
    nb10_path: Path,
) -> Tuple[GeneratedStage, ...]:
    """
    Parse NB10's run_pipeline cell. Extract:
      - _stages list (landing, target_derive, bronze, silver, gold, training)
      - per-stage notebook discovery logic (the glob + topo-sort §A in cell
        8b659505 for landing; alphabetical for others)
      - resume gate behaviour (which stages opt out of skip)
    """
```

## 7. Worked example: the history-window bug

The bug that motivated this design, traced through the audit:

**Engagement state at T0**: NB00 cell `enrich_contract_lifecycle` registers a lifecycle config for the `contract` dataset and sets `semantics_overrides["contract"].raw_time_column_role = INTERVAL_START_TIME`. NB00 intent has `lookback_periods = 36`.

**Exploration scan**: AST walk of `01_data_discovery.ipynb` finds a call to `apply_temporal_lookback` inside an `if` block whose condition includes `_lookback_role.should_apply_lookback`. The walker captures:

```yaml
- dataset: contract
  kind: TEMPORAL_LOOKBACK
  kwargs_fingerprint:
    time_column: event_timestamp
    intent.lookback_periods: 36
  gate_evaluated: _lookback_role.should_apply_lookback
  gate_value: False  # because role == INTERVAL_START_TIME
  emitted: False
  source: 01_data_discovery.ipynb / cell id=5a76bd95 / line 97
```

Result: zero entries with `kind == TEMPORAL_LOOKBACK` for `dataset == contract` in the exploration manifest.

**Production scan**: `scan_production_manifest` runs the renderer in-memory with the parsed project_context. Pre-fix renderer's `_build_history_window_config` returns a `HistoryWindowConfig` whenever `intent.lookback_periods is not None`, independent of role. The template emits `apply_history_window(df)`. AST walk captures:

```yaml
- dataset: contract
  kind: TEMPORAL_LOOKBACK
  kwargs_fingerprint:
    time_column: event_timestamp
    intent.lookback_periods: 36
  gate_evaluated: True  # gate is "intent.lookback_periods is not None"
  emitted: True
  source: (generated in-memory) landing_contract.py / run_landing / line 80
```

Result: one entry with `kind == TEMPORAL_LOOKBACK` for `dataset == contract` in the production manifest.

**Diff**:

```
ParityGap(
    gap_kind=PRODUCTION_ONLY,
    dataset="contract",
    op_kind=TEMPORAL_LOOKBACK,
    exploration_location=None,
    production_location=SourceLocation(
        file="generators/pipeline_generator/databricks_renderer.py",
        line=3371,
        component="apply_history_window template",
    ),
    detail=(
        "Production would emit TEMPORAL_LOOKBACK for contract, but "
        "exploration skips this op (role=INTERVAL_START_TIME gates "
        "should_apply_lookback=False at sampling.py:39). Likely fix: "
        "findings_parser._build_history_window_config must consume "
        "raw_time_column_role and return None when "
        "should_apply_lookback is False."
    ),
)
```

**Audit result**: T0 fails. Job exits. Operator sees the gap with both sides referenced before any data has been touched.

**Post-fix verification**: After `findings_parser` is updated, the production scan returns zero entries for `contract / TEMPORAL_LOOKBACK`. Diff is empty. Audit passes. Job proceeds.

## 8. Implementation guide — phased landing

The full system is large enough to warrant incremental delivery. Each phase is independently useful and adds a layer of guarantee.

### Phase 1 — Foundation (1-2 days)

- `parity/decorator.py`: `@apply_op` decorator, `APPLY_REGISTRY`
- `parity/kinds.py`: full `ApplyOpKind` enum
- `parity/manifest.py`: `Manifest`, `ManifestEntry`, `diff_manifests`
- `parity/gaps.py`: `ParityGap`, `GapKind`, structured exit formatting

Validation: `pytest tests/parity/test_decorator.py tests/parity/test_manifest.py`. Decorate 3 framework functions and verify the registry contains them.

### Phase 2 — Exploration scan (1-2 days)

- `parity/exploration_scan.py`: AST walker, import resolution, gate extraction, symbolic kwargs evaluation
- Decorate the 5 most painful apply primitives: `apply_temporal_lookback`, `enrich_lifecycle_dataset`, `derive_extra_datetime_features`, `apply_target_label_map`, `derive_churn_target` template

Validation: run the scan against an existing engagement directory and verify the produced manifest matches a hand-curated expected manifest. The expected manifest is committed as test fixture.

### Phase 3 — Production scan (2-3 days)

- `parity/production_scan.py`: in-memory renderer invocation, AST walk of rendered output
- Refactor templates that emit apply_op calls through helpers into flat top-level calls. Add `test_renderer_emits_flat_apply_ops.py` lint
- Synthetic findings stub generator for landing-scope audits

Validation: run T0 audit on the SPS engagement (post-fix). Manifest diff must be empty. Then revert the history-window fix; the manifest diff must report the gap with both source locations.

### Phase 4 — Schedule parsing + audit notebook (1 day)

- `parity/schedule.py`: outer + inner DAG parsing
- `audit/audit_landing` top-level function
- `-1_parity_contract.ipynb` template

Validation: end-to-end test on the SPS engagement directory. Audit notebook produces empty report. Inject a synthetic gap (e.g., add a `@apply_op` call to a non-scheduled cell) and verify it's caught.

### Phase 5 — T1 bronze/silver/gold audit (2-3 days)

- Decorate apply primitives in `stages/profiling/`, `stages/temporal/`, gold template appliers
- Extend `scan_production_manifest` to handle non-landing scopes with real findings as input
- Add the T1 invocation to NB10 (cell-level integration)

Validation: synthetic engagement fixture exercising every kind. T1 diff is empty.

### Phase 6 — T3 runtime trace (2-3 days)

- `parity/trace.py`: thread-local tracer, YAML serialization
- NB00 entry: enable `CR_PARITY_TRACE`
- NB09 exit: flush exploration trace
- Production runner: enable tracing + flush production trace
- Final T3 cell: compare traces by row count per stage

Validation: a full SPS pipeline run produces matching traces. Inject a regression that drops 50% of rows in a known stage; the T3 audit fires.

### Phase 7 — Framework CI integration (1 day)

- `pytest tests/parity/test_renderer_contract.py`: fixture-driven audit running on synthetic engagements covering the cartesian product of (granularity × role × intent variations)
- CI hook that fails any framework PR that breaks parity against the fixtures

Validation: revert a known-good framework state; the CI test passes. Apply a known-bad change (e.g., the pre-fix history_window logic); CI fails with a structured ParityGap.

## 9. Validation criteria for the complete solution

A new contributor (or new session) building the system end-to-end can validate completion by checking:

### 9.1 Behavioural

1. The SPS engagement's T0 audit passes on the post-fix codebase, with empty manifest diff
2. Reverting the history-window parity fix produces a T0 audit failure that:
   - Names the dataset (`contract`)
   - Names the `ApplyOpKind` (`TEMPORAL_LOOKBACK`)
   - References the production source location (the renderer's template)
   - References the exploration call site (sampling.py)
   - Exits the audit notebook with a structured failure
3. The job halts on T0 audit failure — no downstream notebook executes
4. T1 audit runs after exploration with real findings and catches a synthetic bronze-aggregation parity gap
5. T3 audit runs after pipeline completion and catches a >0.5% row-count drift in one stage

### 9.2 Structural

1. `APPLY_REGISTRY` contains every framework function whose primary effect is DataFrame transformation
2. A pytest fixture walks `stages/` and `analysis/` modules, identifies any DataFrame-returning function not in `APPLY_REGISTRY` (and not explicitly allowlisted as non-apply), and fails
3. Every renderer template emits apply_op calls at the top level of `run_<stage>()`; nested helpers are rejected by the renderer-flat-emission lint
4. The `ApplyOpKind` enum is closed — adding a new kind requires a framework PR with a corresponding decoration

### 9.3 Operational

1. The T0 audit notebook is the first task in the SPS engagement's Databricks job
2. A failed T0 audit halts the job; this is visible in the workflow run page
3. Operators reading a T0 failure can resolve the gap by:
   - Adjusting an NB00 declaration (project_context override, intent setting)
   - Or filing a framework issue with the structured ParityGap report attached
4. The T3 audit appears as the final cell of NB10's pipeline runner and produces a parity report file in the namespace

## 10. Open design questions

These are deliberate non-decisions left for the implementer; the framing above does not depend on which choice is made.

1. **Outer schedule source** — read from a checked-in `workflow.yml` per engagement, or infer from numeric-prefix convention with explicit override? Recommendation: numeric prefix by default, optional file override.
2. **Dataset hint resolution** — DataFrame-fingerprint chasing, or explicit `with apply_context(dataset="X")` blocks in cells? Recommendation: explicit context manager — clearer, debuggable, no schema sniffing.
3. **Kwargs symbolic evaluation depth** — full constant propagation, or only literal kwargs? Recommendation: literal + simple name resolution (assign-then-call pattern); flag deeper symbolic kwargs as `<dynamic>` and skip kwargs comparison for those entries while keeping kind comparison.
4. **T3 tolerance** — fixed ±0.5% global, or per-kind tolerances? Recommendation: per-kind, with `LIFECYCLE_ENRICH` allowed up to 5% (lifecycle doubling rates vary with cancellation rate), other kinds tighter.
5. **Replay-driven codegen** — long-term, should the renderer become a literal trace replayer (emit `replay_apply_op(kind, kwargs)` instead of expanding templates inline)? Recommendation: not in scope for Phase 1-7; revisit after a year of cross-check mode operation surfaces what's worth migrating.

## 11. Glossary

| Term | Definition |
|---|---|
| **Apply primitive** | A function that transforms a DataFrame and is decorated with `@apply_op`. The closed set of these defines the parity surface |
| **Apply manifest** | Ordered list of apply-op invocations scoped to a dataset, with `(kind, kwargs_fingerprint, source_location)` |
| **ApplyOpKind** | Closed enum of operation categories. Adding a value is a framework-level act |
| **Parity gap** | A divergence between exploration and production manifests for the same dataset |
| **T0 / T1 / T3** | The three audit checkpoints: pre-execution, post-findings, post-pipeline |
| **Outer schedule** | The Databricks job's task DAG: pre-flight notebooks at negative-prefix slots (`-1_parity_contract`, future `-2`, `-3`…), then exploration (`00–09`), then codegen + runner (`10`) |
| **Inner schedule** | NB10's run_pipeline cell — the production pipeline's per-stage notebook DAG |
| **Sandbox renderer** | Running `findings_parser` + `databricks_renderer` in-memory with synthesised inputs to predict what production scripts would contain |
| **Trace** | Runtime recording (when `CR_PARITY_TRACE=1`) of every apply_op invocation with row counts. Used at T3 |
| **Schedule-vs-render symmetry** | Property: every dataset registered in NB00 has a corresponding landing notebook in NB10's inner schedule, and vice versa |

## 12. Relationship to existing systems

- **Cell-tag system** (`Notebook-Sync.md`): same architectural pattern applied at function-definition scope. Cell tags classify cells; `@apply_op` classifies functions. Both are enforced by tooling, declared at the definition site, and used to drive downstream automation.
- **Lane-1 / Lane-2 registration**: `@cr.register` becomes a specialisation of `@apply_op(kind=ApplyOpKind.TARGET_DERIVE)` etc. Lane-2 markers (`cr.mark_lane2_executed`) remain as audit traces but their role narrows to "this function was actually invoked during exploration", which is also what the runtime tracer records.
- **Recommendation registry** (`merged_recommendations.yaml`): unchanged in purpose. Recommendations say *what should be applied*; manifests say *what is applied*. The audit verifies these align in both exploration and production.
- **Resume gate** (NB10 cell `8b659505`): operates at the table-existence level. The audit operates at the operation-shape level. They cover different failure modes; both remain.
- **`v5 §A` topo-sort, `v5 §J` bronze diagnostics, `v5 §L` stale-table check, `v5 §M` enrichment replay dry-run**: these are point-in-time probes for specific bugs we hit. As the audit matures, these become subsumed by the structural and runtime checks. They should remain in place during the transition; deprecation per probe is a decision for each release cycle.

---

This specification is intended to be executable: a contributor (or new session) reading it should be able to land Phase 1-7 end-to-end, validate against the criteria in Section 9, and produce a working active-parity-maintenance system that catches the failure modes in Section 1 before they cost a 20-hour run.
