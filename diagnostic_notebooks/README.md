# diagnostic_notebooks/

Universal, client-agnostic tooling for remediation cycles. Complements
`docs/sps_phased_fix_strategy.md` (rename pending → `phased_fix_strategy.md`).

**Contract:** every file in this directory is client-agnostic. No client
names, catalog prefixes, internal segment labels, or run IDs. Instantiations
with client content live under the gitignored `debug/<engagement>/` tree.

## Contents

| File | Purpose |
|---|---|
| `probe_template.ipynb` | Parametrized landing/bronze/silver sanity probe. Takes `DATASETS`, `BRONZE_WINDOWS`, `FORBIDDEN_SILVER_SUBSTRINGS`, `EXPECTED_MILESTONE_COLUMNS`, `TARGET_HOST_DATASET`, `TARGET_COLUMN` via the `probe-config` cell. Emits `<session_dir>/probe/result.json` + per-layer CSVs. |
| `cycle_template.ipynb` | Per-cycle validation shell. Reads the probe's `result.json`, filters to the cycle's `REQUIRED_CHECKS`, emits `<session_dir>/cycle_NNN/result.json`. |
| `cycle_template.md` | Markdown skeleton every fix cycle copies (into `debug/<engagement>/fix_cycles/NNN_<slug>.md`). |
| `check_privacy.py` | CLI; pre-commit gate. Scans a git diff or explicit paths against an engagement-local denylist plus CRM-record-ID regexes. |
| `check_replay_safety.py` | CLI; cycle-close gate. Walks the static reach-chain from changed files to every file that textually references them. Evidence for "which stages must re-run". |

## How to instantiate for a new engagement

Assumes the engagement code-name is `engagement_a`. All paths below are
inside `debug/engagement_a/` (gitignored).

1. **Create engagement tree**
   ```
   debug/engagement_a/
     .check_privacy.yaml            # denylist: client names, catalogs, etc.
     .engagement.yaml               # framework_repo_root for Databricks sys.path
     probe.ipynb                    # copy of probe_template.ipynb, with DATASETS filled
     fix_cycles/                    # per-cycle docs (managed by new_cycle.py)
     cycles/                        # per-cycle notebooks (managed by new_cycle.py)
     artifacts/                     # HTML exports, downloaded result.json, CSVs
   ```

2. **Fill `debug/engagement_a/.engagement.yaml`**
   ```yaml
   framework_repo_root: /Workspace/Repos/<user>/customer_retention
   ```
   Only needed if the Databricks cluster runs `customer_retention` from a
   Workspace Repo clone (i.e. the package is not pip-installed). The
   `@cr:code_system` cell that `new_cycle.py` injects is a no-op when
   pip-install is present (the `sys.path` guard skips duplicate insertion).

3. **Fill `debug/engagement_a/.check_privacy.yaml`**
   ```yaml
   terms:
     - <client product name>
     - <internal catalog / schema prefix>
     - <segment labels unique to this client>
   id_regexes: []   # optional — defaults include an 18-char CRM-ID regex
   ```

4. **Fill `debug/engagement_a/probe.ipynb`'s `probe-config` cell**
   See cell docstring. `DATASETS` is the main config. Everything else is optional.
   When first copying `probe_template.ipynb` → `debug/engagement_a/probe.ipynb`,
   also prepend a `@cr:code_system` cell if your cluster needs it
   (same pattern `new_cycle.py` uses — one-time setup for the probe).

5. **Open cycle 001**
   ```
   python diagnostic_notebooks/new_cycle.py \
       --engagement engagement_a --cycle 1 --slug <short-name>
   ```

   `new_cycle.py` copies the templates and prepends the
   `@cr:code_system` cell so `customer_retention.*` imports resolve on
   Databricks before the first cell runs. Fill the doc's problem /
   impacted-files / required-checks sections and the notebook's
   cycle-config cell, then run on Databricks.

## Running from Databricks

Both notebooks use `RunNamespace.from_env_or_latest()` — the same discovery
chain as the exploration notebooks. If `RUN_ID` is set in the config cell,
the probe / cycle targets that run explicitly; otherwise it picks up the
active run from the session state.

Outputs land under `namespace.session_dir/probe/` and
`namespace.session_dir/cycle_<NNN>/`. Download the HTML export of the
notebook plus `result.json` to bring the evidence back to Claude Code.

## Running the CLIs locally

```bash
# Before commit: privacy gate
python diagnostic_notebooks/check_privacy.py \
    --diff HEAD \
    --denylist debug/engagement_a/.check_privacy.yaml

# Before closing a cycle: replay-safety scan
python diagnostic_notebooks/check_replay_safety.py --since main
```

Both scripts are pure Python (only `pyyaml`) and run on CI workers.

## Conventions this directory follows

From `docs/Coding_Practices.md`:

- `# @cr:TYPE name='X' id=HEX` on line 1 of every notebook code cell.
- Profiler block in the `init_progress` cell of every notebook.
- `from customer_retention.core.compat import native_pd` — never
  `import pandas as pd`.
- `from customer_retention.analysis.visualization import console,
  display_table` for output.
- Fail fast: no `try/except Exception: pass`. `_try_load` returns an
  error string rather than raising, but the error is always surfaced
  in the probe's `CHECKS` record, never silenced.
- Compact code, descriptive names, short single-responsibility helpers.

## Evolution

When a cycle discovers a universal probe class that's missing here,
promote it — but scrub client content first, then extend
`probe_template.ipynb`. The probe template is the canonical list of
generic checks; engagement probes are allowed to add, never to soften.
