# Resume Pattern — Design Doc

A universal **opt-in, no-op-by-default** pattern for resuming long-running pipelines from the last successful step. Covers three contexts under one configuration:

- **Production stages** — generated landing/bronze/silver/gold/training notebooks invoked by NB10's `run_pipeline`.
- **Exploration notebooks** — NB05 (relationship analysis), NB08 (baseline experiments), and any future heavy notebook.
- **Cell-level checkpoints** — individual heavy cells inside an exploration notebook (e.g., NB08's baseline-training cell).

Status: **proposed**. This document is the design contract; implementation lands in phases.

---

## Why one pattern

Both production and exploration share `RunNamespace` for state, both have an ordered sequence of expensive steps, both today re-run the entire sequence on a single broken step. The unique-per-context concerns (UC Delta tables vs findings YAML vs MLflow runs) reduce to a single sentinel-schema variation. One module, one config dataclass, one operator UX.

## Design decisions (locked)

| Question | Decision | Rationale |
|---|---|---|
| Generation hash granularity | **per-stage** | A renderer tweak inside silver doesn't force bronze re-runs. Matches the operator's mental model when triaging the SPS bridge. |
| Hash mismatch policy | **warn-only** | Resume is a build-phase tool. Operators tolerate occasional false-positive skips (warn) more than mandatory re-runs (strict). When the pipeline is stable, resume is off entirely. |
| Exploration granularity | **cell-level (`with resume.guard()`)** | NB05 / NB08 have a small number of expensive cells; per-cell guards skip the heavy work while letting cheap setup cells run normally. |
| MLflow integration depth | **verify-exist only** | Sentinel records `run_id`; resume gate confirms the run still exists. Re-loading models is the consumer's job (cheap). |
| **No-op-by-default** | **`RESUME_MODE=False` ⇒ zero overhead** | When resume is off, no sentinels are written, no hashes are computed, the gate is a single boolean check that short-circuits. Production-stable phases pay nothing. |

## No-op-by-default contract

When `ResumeConfig.enabled is False` (the default):

- **`is_complete(...)` returns `False` immediately** — no file reads, no hash computation, no Spark calls.
- **`mark_complete(...)` is a no-op** — no JSON written, no namespace dir created.
- **`@checkpoint(...)` decorator pass-through** — wrapped function is called identically to the un-decorated form.
- **`with resume.guard(...)` block** — `ckpt.skip` is always `False`; the heavy compute always runs; `ckpt.save_outputs(...)` is a no-op.
- **Stage templates' inserted sentinel-write block** — guarded by `if RESUME_MODE:` so the JSON write only happens when needed.
- **Manifest's per-stage hash computation** — only computed when at least one cycle has `RESUME_MODE=True`; otherwise the field is omitted from the manifest.

This means: shipping the pattern doesn't slow down operators who don't want it, doesn't fill `runs/<run_id>/checkpoints/` with files when resume is off, and doesn't add Spark calls during normal stable-phase runs.

---

## Public API — `customer_retention/core/checkpoints.py`

Three usage shapes, all backed by the same `Checkpoint` and `ResumeConfig` types.

### Orchestrator gate (NB10 run_pipeline, exploration runners)

```python
from customer_retention.core.checkpoints import resume, ResumeConfig

_resume_cfg = ResumeConfig.from_namespace_or_env(_namespace)

if resume.is_complete(
    step_id="bronze/bronze_event_case",
    config=_resume_cfg,
    namespace=_namespace,
    generation_hash=_stage_hashes.get("bronze"),
):
    print("[SKIP] bronze/bronze_event_case (resumed)")
    continue

_result = dbutils.notebook.run(_path, 86400, _ns_params)
# child notebook writes its own sentinel before exit
```

### Decorator (rendered stage templates, exploration helpers)

```python
@resume.checkpoint(
    step_id="exploration/05_relationship_analysis",
    inputs_provider=lambda: _findings_input_hash(_namespace),
    outputs=[("namespace_path", "merged_recommendations.yaml")],
)
def run_relationship_analysis():
    ...
```

### Context manager (heavy cells inside exploration NBs)

```python
from customer_retention.core.checkpoints import resume

with resume.guard(
    step_id="exploration/08_baseline_experiments/train_baselines",
    inputs_hash=_findings_hash,
    config=_resume_cfg,
    namespace=_namespace,
) as ckpt:
    if ckpt.skip:
        models = ckpt.load_outputs()
    else:
        models = train_all_baselines()
        ckpt.save_outputs(
            models,
            outputs=[
                ("mlflow_run", _run.info.run_id),
                ("pickle_path", "baseline_models.pkl"),
            ],
        )
```

All three reduce to the same primitives: `read_sentinel`, `compare_hashes`, `verify_outputs_exist`, `write_sentinel`.

---

## Sentinel schema

Stored at `{namespace.checkpoints_dir}/{step_id}.json`. `step_id` may contain `/`, which maps to nested directories under `checkpoints_dir`.

```json
{
  "step_id": "bronze/bronze_event_case",
  "kind": "production_stage",
  "outputs": [
    {
      "type": "uc_table",
      "name": "prod.churn.bronze_event_case",
      "rows": 12345678,
      "column_hash": "sha256:..."
    }
  ],
  "inputs_hash": "sha256:...",
  "generation_hash": "abc123...",
  "framework_version": "1.01.9a8",
  "elapsed_seconds": 393.0,
  "completed_at": "2026-04-30T11:34:12Z"
}
```

### Output type taxonomy

| `outputs[*].type` | Required fields | Verifier |
|---|---|---|
| `uc_table` | `name`, `rows`, `column_hash` | `spark.catalog.tableExists(name)` and (optional) `spark.table(name).count() == rows` |
| `namespace_path` | `path` (relative to namespace root) | `(namespace.root / path).exists()` |
| `mlflow_run` | `run_id` | `mlflow.get_run(run_id)` does not raise; run is not in `DELETED` lifecycle stage |
| `pickle_path` | `path` (relative to namespace root) | `(namespace.root / path).exists()` |

A sentinel may list multiple outputs (e.g., NB08 produces both an MLflow run AND a baseline-models pickle). The gate verifies **all** outputs exist before declaring `is_complete`.

---

## Configuration — `ResumeConfig`

Single shape, persistable to `namespace.resume_config_path`, env-overridable, widget-driven.

```python
@dataclass(frozen=True)
class ResumeConfig:
    enabled: bool = False                  # RESUME_MODE — global toggle
    invalidate: tuple[str, ...] = ()       # exact step_ids or glob patterns
    invalidate_from: str | None = None     # cascade-invalidate from this step onward
    require_output_check: bool = True      # also verify UC table / mlflow run exists
    hash_policy: str = "warn"              # "warn" | "strict"
    hash_granularity: str = "stage"        # "stage" | "notebook" | "global"

    @classmethod
    def from_namespace_or_env(cls, namespace) -> "ResumeConfig":
        """Priority: explicit widget values > namespace.resume_config_path
        > CR_RESUME_* env vars > defaults."""
```

### Configuration sources (in priority order)

1. **NB10 / NB00 widget values** (most local, easiest to override per run).
2. **`namespace.resume_config_path` JSON** (persisted between runs in the same namespace).
3. **`CR_RESUME_*` environment variables** (cluster/workflow-level defaults).
4. **Hard-coded defaults** (resume off; no overhead).

### Operator widget panel (NB10)

```python
RESUME_MODE = False                # toggle resume on/off
INVALIDATE = []                    # list[str] of step_ids or globs to drop (no cascade)
INVALIDATE_FROM = None             # str | None: cascade-invalidate from this step onward
RESUME_REQUIRE_OUTPUT_CHECK = True # also verify UC table / mlflow run exists
RESUME_HASH_POLICY = "warn"        # "warn" | "strict"
```

When `RESUME_MODE=False`, the rest of the widgets are inert. Cells that would have written sentinels short-circuit to no-ops.

---

## Step taxonomy

| Kind | step_id format | Output types | Gate location | Sentinel write location |
|---|---|---|---|---|
| `production_stage` | `{stage}/{notebook_name}` (e.g., `bronze/bronze_event_case`) | `uc_table` | NB10 `run_pipeline` cell | inside the rendered child notebook, before `dbutils.notebook.exit(...)` |
| `exploration_notebook` | `exploration/{notebook_basename}` (e.g., `exploration/05_relationship_analysis`) | `namespace_path` + optional `mlflow_run` | top-of-notebook `resume_check` cell (after init, before heavy work) — exits early on hit | bottom-of-notebook `resume_mark` cell |
| `cell_checkpoint` | `exploration/{notebook}/{cell_id}` (e.g., `exploration/08_baseline_experiments/train_baselines`) | any (`pickle_path`, `mlflow_run`, `namespace_path`) | inside the cell via `with resume.guard(...)` | inside the same `with` block via `ckpt.save_outputs(...)` |

The gate logic is identical regardless of kind. Output verification is dispatched on `outputs[*].type`.

---

## Generation hash — per-stage default

`_write_generation_manifest` computes:

```python
def _stage_hash(stage_files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in sorted(stage_files):
        h.update(f.relative_to(output_dir).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()

manifest = {
    ...
    "stage_hashes": {
        "landing": _stage_hash([... landing files ...]),
        "bronze":  _stage_hash([... bronze files ...]),
        "silver":  _stage_hash([... silver files ...]),
        "gold":    _stage_hash([... gold files ...]),
        "training":_stage_hash([... training files ...]),
    },
    "global_hash": _stage_hash([... all rendered files ...]),
}
```

Sentinels record the stage hash that was current when they were written. NB10's gate compares `sentinel.generation_hash` against `manifest.stage_hashes[stage]` — mismatch ⇒ STALE under strict policy, WARN-only under default policy.

For exploration steps, the equivalent input hash is `sha256(active_dataset_id + framework_version + relevant_findings_yaml_hash)` — exploration depends on data + findings + framework, not rendered code.

### Hash granularity options

| `hash_granularity` | Hash inputs | Re-run scope on a one-line renderer change in `silver/silver_featureset_*.py` |
|---|---|---|
| `"stage"` (default) | Per-stage content hash | Only silver + gold + training re-run |
| `"notebook"` | Per-rendered-file hash | Only the changed notebook re-runs |
| `"global"` | Hash of every rendered file | Everything re-runs |

Operators tune the knob to their build phase: aggressive `"notebook"` during stabilization, conservative `"stage"` once the bridge is settled, no-op `enabled=False` for production-stable runs.

---

## Hash mismatch — warn-only by default

Default behavior (`hash_policy="warn"`):

```
[RESUME] bronze/bronze_event_case: HASH-MISMATCH (sentinel=abc123, current=def456)
         policy=warn -> proceeding with cached output anyway
[SKIP]   bronze/bronze_event_case (resumed)
```

The cached output is used; a warning line names the drift. Operator who knows the change is cosmetic (whitespace, comment, log message) gets resume's full speed-up. Operator who knows the change is meaningful flips `INVALIDATE = ("bronze/bronze_event_case",)` for one cycle.

Strict policy (`hash_policy="strict"`):

```
[RESUME] bronze/bronze_event_case: HASH-MISMATCH (sentinel=abc123, current=def456)
         policy=strict -> running notebook
```

Re-runs on any drift. Recommended for late-stage validation cycles where parity matters.

---

## Cascade invalidation

`INVALIDATE_FROM = "silver"` deletes sentinels for `silver`, `gold`, `training` (production) or for all exploration notebooks ordered after the silver-equivalent. The ordered step list is the single source of dependency truth, declared once in the resume module:

```python
PRODUCTION_STEP_ORDER = ("landing", "bronze", "silver", "gold", "training")

EXPLORATION_STEP_ORDER = (
    "00_start_here",
    "01a_event_dataset_initialization",
    "01a_a_lifecycle_filter",
    "01b_temporal_continuity",
    "01c_silver_distribution",
    "01d_event_aggregation",
    "02_source_integrity",
    "03_dataset_merge",
    "04_column_deep_dive",
    "04a_text_columns_deep_dive",
    "05_relationship_analysis",
    "06_feature_opportunities",
    "07_modeling_readiness",
    "08_baseline_experiments",
    "09_business_alignment",
    "10_spec_generation",
)
```

`invalidate_from(step_id, ordered_ids, namespace)` finds the index of `step_id` and unlinks every sentinel for steps ≥ that index.

`INVALIDATE = ("bronze/bronze_event_case",)` removes only that exact sentinel — no cascade. Useful for transient single-notebook failures.

`INVALIDATE = ("bronze/*",)` removes every bronze sentinel via glob — no cascade. Useful when the operator knows bronze must re-run but silver+ shouldn't.

---

## RunNamespace extensions

```python
class RunNamespace:
    @property
    def checkpoints_dir(self) -> Path:
        """`runs/<run_id>/checkpoints/` — root of all sentinels."""
        return self.run_dir / "checkpoints"

    def checkpoint_path(self, step_id: str) -> Path:
        """Resolve a step_id (which may contain '/') to its sentinel JSON path."""
        # bronze/bronze_event_case -> runs/<run_id>/checkpoints/bronze/bronze_event_case.json
        return self.checkpoints_dir / f"{step_id}.json"

    @property
    def resume_config_path(self) -> Path:
        """Single JSON for the persisted ResumeConfig in this run."""
        return self.run_dir / "resume_config.json"
```

Backwards-compatible: `checkpoints_dir` is created lazily on first sentinel write; runs that never enable resume never create the directory.

---

## Where to instrument

### 1. New module `core/checkpoints.py` (~250 lines)

Public API:
- `Checkpoint` dataclass (sentinel schema).
- `ResumeConfig` dataclass.
- `is_complete(step_id, *, config, namespace, generation_hash, current_inputs_hash) -> bool`
- `mark_complete(step_id, *, kind, outputs, namespace, generation_hash, inputs_hash, elapsed_seconds, config) -> Path | None` (returns `None` when `config.enabled is False` — no-op).
- `invalidate(patterns: tuple[str, ...], namespace) -> list[str]`
- `invalidate_from(step_id: str, ordered_ids: tuple[str, ...], namespace) -> list[str]`
- `@checkpoint(step_id, inputs_provider=, outputs=)` decorator.
- `guard(step_id, *, inputs_hash, config, namespace)` context manager.
- Output verifiers for each `outputs[*].type` (lazy-imported per type so unused environments don't pay).

### 2. RunNamespace extension (~10 lines)

`checkpoints_dir`, `checkpoint_path(step_id)`, `resume_config_path`.

### 3. Generated stage templates (~10 lines per template × 5 templates = 50 lines)

Each landing/bronze/silver/gold/training template inserts before `dbutils.notebook.exit(...)`:

```python
if _RESUME_CFG.enabled:
    from customer_retention.core.checkpoints import resume
    resume.mark_complete(
        step_id=f"{STAGE}/{NOTEBOOK_NAME}",
        kind="production_stage",
        outputs=[{
            "type": "uc_table",
            "name": output_table,
            "rows": _row_count,
            "column_hash": _column_hash,
        }],
        namespace=_NAMESPACE,
        generation_hash=GENERATION_HASH_FOR_STAGE,
        inputs_hash=None,
        elapsed_seconds=_total_elapsed,
        config=_RESUME_CFG,
    )
```

The `if _RESUME_CFG.enabled:` guard is the no-op default. Stage templates already compute `_row_count` and `_total_elapsed`; `_column_hash` is a new lightweight `sha256(",".join(sorted(df.columns)))`.

### 4. `_write_generation_manifest` extension (~15 lines)

Compute per-stage content hash; include `stage_hashes` and `global_hash` in the manifest. The manifest write itself runs unconditionally (cheap), but the hashes are only consulted when `RESUME_MODE=True`.

### 5. NB10's `run_pipeline` cell (~25 lines)

```python
from customer_retention.core.checkpoints import resume, ResumeConfig

_resume_cfg = ResumeConfig.from_namespace_or_env(_namespace)
if RESUME_MODE:
    _resume_cfg = replace(_resume_cfg, enabled=True,
                          invalidate=tuple(INVALIDATE),
                          invalidate_from=INVALIDATE_FROM)
    if _resume_cfg.invalidate:
        _dropped = resume.invalidate(_resume_cfg.invalidate, _namespace)
        print(f"[RESUME] invalidated {len(_dropped)} sentinels: {_dropped}")
    if _resume_cfg.invalidate_from:
        _dropped = resume.invalidate_from(
            _resume_cfg.invalidate_from, PRODUCTION_STEP_ORDER, _namespace,
        )
        print(f"[RESUME] cascade-invalidated {len(_dropped)} sentinels from {_resume_cfg.invalidate_from}")

_manifest = _read_generation_manifest(output_dir)
_stage_hashes = _manifest.get("stage_hashes", {})

for _stage in _stages:
    ...
    for _nb in _notebooks:
        _step_id = f"{_stage}/{_nb}"
        if resume.is_complete(
            step_id=_step_id, config=_resume_cfg, namespace=_namespace,
            generation_hash=_stage_hashes.get(_stage),
        ):
            print(f"[SKIP] {_step_id}")
            continue
        _result = dbutils.notebook.run(_path, 86400, _ns_params)
```

When `RESUME_MODE=False`, `is_complete` returns `False` immediately and the loop runs as today.

### 6. Exploration heavy notebooks (NB05, NB08, optionally NB04)

Top-of-notebook `resume_check` cell (after init, before heavy work):

```python
# @cr:code name='resume_check' id=<HEX>
from customer_retention.core.checkpoints import resume, ResumeConfig

_resume_cfg = ResumeConfig.from_namespace_or_env(_namespace)
if resume.is_complete(
    step_id="exploration/05_relationship_analysis",
    config=_resume_cfg,
    namespace=_namespace,
    current_inputs_hash=_findings_input_hash(_namespace),
):
    print("[SKIP] 05_relationship_analysis (resumed)")
    dbutils.notebook.exit("skipped")
```

For cell-level granularity inside NB08, heavy cells use `with resume.guard(...)`:

```python
# @cr:code name='train_baselines' id=<HEX>
with resume.guard(
    step_id="exploration/08_baseline_experiments/train_baselines",
    inputs_hash=_findings_hash,
    config=_resume_cfg,
    namespace=_namespace,
) as ckpt:
    if ckpt.skip:
        baseline_models = ckpt.load_outputs()
    else:
        baseline_models = _train_all_baselines(...)
        ckpt.save_outputs(
            outputs=[
                ("mlflow_run", _run.info.run_id),
                ("pickle_path", "baseline_models.pkl"),
            ],
            payload=baseline_models,
        )
```

When `RESUME_MODE=False`, `ckpt.skip` is always `False` and the cell runs as today.

---

## Operator workflows

### A. Production failure → fix → resume

1. Cycle N: `landing → bronze → bronze_event_opportunity FAILS`. 14 sentinels written for completed work; nothing for `bronze_event_opportunity` and downstream.
2. Operator diagnoses + applies a patch.
3. Sets `RESUME_MODE=True` in NB10. Doesn't touch `INVALIDATE_*`.
4. Re-runs `run_pipeline`. Output:
   ```
   [SKIP] landing/landing_case
   [SKIP] landing/landing_contract
   ...
   [SKIP] bronze/bronze_entity_implementation_project_aggregated
   [BRONZE] bronze_event_opportunity: 453.1s   <- runs
   [BRONZE] bronze_event_opportunity_product: 181.9s
   ...
   [SILVER] silver_featureset_...
   ```
5. ~30 min saved on bronze re-execution per repair attempt; ~6 h saved if silver was the failure point.

### B. Generation regenerated → automatic per-stage invalidation

1. Operator re-runs `29c36168 generate_databricks` (e.g., to pick up a renderer fix).
2. Manifest updates per-stage hashes. Bronze hash unchanged; silver hash changed.
3. Next NB10 run with `RESUME_MODE=True`: bronze `[SKIP]`s automatically (hashes match); silver `[RUNNING]` (hash mismatch under strict, or `WARN + SKIP` under default warn-only). No operator action needed.

### C. Manual cascade invalidation

`INVALIDATE_FROM = "silver"` — operator wants to redo silver onwards regardless of hashes. NB10's `run_pipeline` opening cleans `silver/*`, `gold/*`, `training/*` sentinels before the loop.

### D. Single-notebook reset

`INVALIDATE = ("bronze/bronze_event_case",)` — exact-match invalidation, no cascade. Useful for transient failures (cluster issue, race condition).

### E. Production-stable cycle — resume entirely off

`RESUME_MODE=False`. No sentinels written. No hashes computed in the gate. No `checkpoints_dir` created. Pipeline runs identically to a pre-resume version. Zero overhead.

### F. Exploration repair — NB08 retrains baselines after fixing a feature

1. NB05/NB06/NB07 had completed cleanly in cycle M; sentinels exist.
2. NB08's baseline-training cell crashed mid-train; no NB08 sentinel exists.
3. Operator fixes the bug, sets `RESUME_MODE=True`, opens NB08, runs all cells.
4. The `with resume.guard(...)` around baseline training sees no sentinel → trains. Other heavy cells in NB08 (e.g., feature importance) write their own sentinels.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sentinel exists but UC table dropped manually | `require_output_check=True` (default) re-verifies via `spark.catalog.tableExists`; mismatch ⇒ STALE ⇒ re-run regardless of policy |
| Stale schema (column_hash drift) | Sentinel records `column_hash`; mismatch under any policy ⇒ STALE (column-list change is structural, not cosmetic) |
| Concurrent operator runs | Document "single-operator assumption" for v1; future v2 adds optimistic file lock under `checkpoints_dir/.lock` |
| Whitespace-only renderer change forces re-runs (under strict policy) | `hash_policy="warn"` is the default — drift logged, cached output reused |
| Sentinel orphaned across run_id changes | Sentinels live under `runs/<run_id>/checkpoints/`. New `run_id` starts clean. Cross-run reuse not supported (intentional — different runs may have different findings/datasets) |
| Exploration notebook is skipped but downstream cells expected its in-process state | Cell-level `with resume.guard()` reloads outputs via `ckpt.load_outputs()` so downstream consumers see the data. Notebook-level skip via `dbutils.notebook.exit("skipped")` is only used when no downstream notebook reads the in-process state — only persisted artefacts. |
| MLflow run lifecycle = DELETED | Verifier treats `DELETED` as missing ⇒ STALE ⇒ re-run |
| Resume on, but operator wants forced re-run of one step | `INVALIDATE = ("step_id",)` removes that one sentinel; the next run re-executes only it |

---

## Coding-practice compliance

- **Single source of truth via `RunNamespace`** (Coding_Practices.md line 18-39): all state under `namespace`; one discovery chain; no glob fallbacks.
- **No defensive code hiding errors** (line 7): hash mismatch under strict policy fails loudly; missing dependencies log a clear `[STALE]` reason naming the broken sentinel; `require_output_check` mismatches are explicit.
- **No public exports of single-environment functions** (line 102): the resume module's public API is environment-agnostic. Output verifiers (`spark.catalog.tableExists`, `mlflow.get_run`, etc.) are lazy-imported per `outputs[*].type` so missing dependencies in unused environments don't break import.
- **No threading** (stored memory): sequential gate evaluation; no parallel sentinel writes (banned on UC shared clusters).
- **Stays distributed** (line 5): sentinels are tiny JSON written by the driver; `column_hash` is computed from `sorted(df.columns)` (metadata only). No driver-collect of data, no `toPandas()`, no per-row work.
- **No-op-by-default**: when `enabled is False`, every public function returns immediately without I/O, hash compute, or Spark calls. Verified by tests.
- **TDD with ≥90% coverage** (line 3): test plan below.

---

## Test plan

### Unit (fast, CI-runnable, no Spark required)

- `is_complete` with `enabled=False` returns `False` instantly without reading filesystem.
- `mark_complete` with `enabled=False` is a no-op (no file written, no directory created).
- `is_complete` returns `True` when sentinel exists, hash matches, and outputs verified.
- `is_complete` returns `False` when sentinel missing.
- `is_complete` under `hash_policy="warn"` returns `True` on hash mismatch and logs WARN line.
- `is_complete` under `hash_policy="strict"` returns `False` on hash mismatch.
- `is_complete` returns `False` when output verification fails (mocked verifiers).
- `mark_complete` writes valid sentinel JSON to expected path.
- `invalidate(patterns)` removes matching sentinels (exact + glob).
- `invalidate_from(step_id, order)` cascades correctly.
- `@checkpoint` decorator: no-op pass-through when `enabled=False`; skip + load when complete; call + write when not complete.
- `guard` context manager: `ckpt.skip=False` always when `enabled=False`; round-trip save → load works.
- `ResumeConfig.from_namespace_or_env` priority order respected.

### Integration (per stage template)

- Each rendered stage template includes the `if _RESUME_CFG.enabled: resume.mark_complete(...)` block before exit.
- Generated `_write_generation_manifest` includes `stage_hashes` and `global_hash`.
- `column_hash` in sentinel matches the verifier's recompute on the actual UC table (test with synthetic Spark DF).

### E2E

- Simulate full pipeline run (mocked `dbutils.notebook.run`): 2 stages succeed, 3rd fails ⇒ 2 sentinels exist, 0 for failed stage.
- Operator fixes, re-runs with `RESUME_MODE=True` ⇒ only 3rd stage's notebooks run + downstream.
- `INVALIDATE_FROM` cascade test: set to stage 3, run, verify sentinels for stages 4-5 are deleted, 1-2 are kept.
- Hash-policy test: tweak a comment in a rendered file ⇒ generation hash changes ⇒ under `warn` cached output is reused with WARN log; under `strict` re-runs.
- No-op test: full pipeline run with `RESUME_MODE=False` ⇒ `checkpoints_dir` does not exist after run; identical wall-clock to a pre-resume baseline.

---

## Phased implementation plan

| Phase | Scope | Risk | Effort | Default-off behavior |
|---|---|---|---|---|
| 1 | Core `checkpoints.py` module + `RunNamespace` extension + unit tests | Low (no behavior change) | 1-2 days | API exists but nothing calls it |
| 2 | Production: stage templates write sentinels (guarded by `if _RESUME_CFG.enabled`); NB10 `run_pipeline` gate; `_write_generation_manifest` extended | Low (default `RESUME_MODE=False` keeps existing behavior) | 1-2 days | Pipeline runs identically when off |
| 3 | Exploration: NB05, NB08 (and NB04 if heavy enough) get top-level `resume_check` cells; NB08's heavy cells get `with resume.guard(...)` blocks | Low (default off) | 1 day | Notebooks run as today when off |
| 4 | UX polish: NB10 / NB00 widget panels; `cr-resume` CLI for terminal invalidation; this doc plus per-track examples | Low | 0.5 day | n/a |

**Total:** ~4-5 days. Each phase independently shippable; phase 4 is optional.

---

## Backwards compatibility

- v1 of every stage template: no sentinel write. Existing pipelines keep working.
- v2 templates write sentinels but only when `_RESUME_CFG.enabled is True`. Off-by-default ⇒ identical wall-clock to v1.
- Operators flip `RESUME_MODE=True` per run when they want resume behavior. No forced migration.
- `generate_databricks` regenerates with the latest template version on next run ⇒ new sentinel writes naturally rolled out per cycle.
- Cross-run reuse not supported (each `run_id` has its own `checkpoints_dir`). Resuming requires running in the same `RunNamespace`.

---

## Open follow-ups

These are non-blocking and can be added after phase 4:

1. **`cr-resume` CLI** (`bin/cr-resume {list,invalidate,clear} [step_id ...]`) — for terminal-driven invalidation outside Databricks.
2. **MLflow model re-load helpers** — `ckpt.load_outputs()` for `mlflow_run` types could call `mlflow.<flavor>.load_model(...)` automatically when an `artifact_path` field is recorded. Currently the consumer reloads explicitly.
3. **Concurrent-operator safety** — optimistic file lock under `checkpoints_dir/.lock` to detect two operators running NB10 simultaneously in the same namespace.
4. **AST-based hash mode** — `hash_granularity="ast"` ignores comments and whitespace by hashing token streams. Reduces false-positive invalidation under `hash_policy="strict"`.
5. **Resume from arbitrary previous run** — copy a healthy `runs/<old_run_id>/checkpoints/` into a new `runs/<new_run_id>/checkpoints/` to bootstrap a new run from prior state. Useful for branching experiments off a stable baseline.
