# Cycle NNN — <one-line summary>

**Engagement:** `<code-name>` (keep neutral; no client product names)
**Opened:** YYYY-MM-DD
**Author:** <name>
**Run ID under analysis:** `<run-id>`

## Problem

<2–5 lines drawn from the engagement's diagnosis doc. Link the probe
cell IDs whose FAIL triggered the cycle, e.g. `probe-l2-et` for bronze
per-grid-date event_type coverage.>

## Impacted files

<grep-verified, absolute paths>

- `src/customer_retention/...`
- `exploration_notebooks/NN_<name>.ipynb` cells: `<id1>`, `<id2>`

## Reference contracts

- `docs/Coding_Practices.md` §<section> — <why it applies>
- `docs/Architecture.md` §<section> — <why it applies>

## Downstream contamination risk

List every notebook or pipeline stage consuming the changed artifacts.
Run `python diagnostic_notebooks/check_replay_safety.py --since <base>`
and paste the reach-chain summary here. If any heavy stage (e.g. NB05,
NB08) appears in the chain, the cycle **must** include running those
stages; no replay-from-cache shortcut.

```
<paste scanner output>
```

Decision: replay-safe = YES / NO. If NO, list stages that must run.

## Fix

Tests-first when the logic is non-trivial. Otherwise a bullet list of
edits suffices.

- <file:line> — <change>
- <file:line> — <change>

## Validation

- **Offline**:
  - `ruff check src/ diagnostic_notebooks/ <cycle-paths>`
  - `pytest <relevant tests>`
  - If any exploration notebook cell changed: `python scripts/build_framework_phase_map.py && git add framework/phase_map.yaml`
  - Render-and-grep where applicable: `python -m customer_retention.generators ...
    --output /tmp/fx && grep <literal> /tmp/fx/<file>`
- **Runtime**:
  - Notebook: `debug/<engagement>/cycles/NNN_<slug>.ipynb`
  - Required probe checks (must PASS in `probe/result.json`):
    - `(layer, dataset, check)` — e.g. `("landing", "<ds>", "exists_nonzero")`
    - ...

## Close criteria

- `cycle_<NNN>/result.json` shows `status: "PASS"`.
- All `REQUIRED_CHECKS` populated and PASS.
- No regression in previously-green checks from prior cycles (diff
  against the last tagged phase).

## Doc updates

- `docs/Coding_Practices.md` — <section amended, or "no updates">
- `docs/Architecture.md` — <section amended, or "no updates">

## Privacy check

```
<paste output of: python diagnostic_notebooks/check_privacy.py --diff <base> --denylist debug/<engagement>/.check_privacy.yaml>
```

Must be `clean` before commit.

## Commit

- Branch: `fix-cycle-NNN-<slug>`
- Commit message: `cycle NNN: <one-line summary>`
- Tag on merge: `cycle-NNN-green`
