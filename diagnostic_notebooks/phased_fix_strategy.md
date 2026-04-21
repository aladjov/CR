# Phased Fix Strategy (Methodology)

**Scope:** any CustomerRetention remediation where end-to-end
execution is expensive and observation is indirect (e.g. Databricks
output read back via HTML exports). Dataset- and engagement-
agnostic by construction — the methodology applies to every run,
regardless of which datasets the framework is pointed at.

**Version-control and privacy contract:**
- This repo is public. Nothing committed — docs, tests, fixtures,
  notebooks, or code — may include engagement-identifying content
  (client names, internal catalog or schema names, customer IDs,
  segment labels unique to an engagement, product taxonomies).
  Generic B2B-SaaS column names (`ACCOUNT_ID`, `CONTRACT_ID`, etc.)
  are fine; anything that hints *which* engagement is not.
- `debug/` is gitignored (see `.gitignore:86`) and is the only
  acceptable home for anything engagement-specific: per-run
  artifacts, per-engagement sanity probes, fix-cycle docs, HTML
  exports, CSVs.
- `diagnostic_notebooks/` is the home for **universal** tooling:
  probe templates, validators, cycle notebook templates,
  replay-safety scanners. Engagement-agnostic by contract. Any file
  here that mentions an engagement, run ID, or catalog is a bug.

**Companions (must stay in sync):**
- `docs/Coding_Practices.md` — every fix cycle cites the sections
  that apply; violations block cycle close.
- `docs/Architecture.md` — same contract; updated whenever a cycle
  changes a pattern.
- `debug/<engagement>/diagnosis_*.md` — the symptom list the
  current cycle is working against (not committed;
  engagement-specific).

---

## Principles (non-negotiable)

1. **Failures stay loud.** No `try/except` to let things proceed.
   A silenced error is the root of the "week-long invisible
   regression" problem and is never worth the short-term smoothness.
2. **Proof-of-fix lives in the probe, not in prose.** The sanity
   probe (universal template in `diagnostic_notebooks/`,
   engagement-specific instantiation in `debug/`) is the single
   gate. New failure classes extend the probe; the probe is never
   softened to let a cycle close.
3. **No blind caching of slow stages.** Heavy stages are only
   skipped when a diff scan proves the change cannot affect their
   inputs. The scan is mechanical and lives in code (§ Contamination
   safety), not in a human judgement call.
4. **One fix cycle = one doc = one commit.** Cycles are
   sequentially numbered under `debug/<engagement>/fix_cycles/`.
   Incomplete cycles block later ones.
5. **Minimum tests to prove the specific fix.** Not "tests for
   everything." Probe cells are the integration check; unit tests
   only when the logic is non-trivial and hard to prove from the
   probe alone. Test fixtures must be generic — no engagement data.
6. **Living reference docs are part of the cycle.** If a cycle
   changes a pattern in `Architecture.md` or `Coding_Practices.md`,
   the cycle is not closed until those docs are updated.
7. **Privacy is part of cycle close.** Before commit, grep the
   changed set for engagement identifiers (a small script under
   `diagnostic_notebooks/` does this mechanically). Any hit blocks
   close.

---

## Directory layout

```
diagnostic_notebooks/        # versioned, universal, engagement-agnostic
  probe_template.ipynb       # generic sanity-check probe (RUN_ROOT + DATASETS injected)
  cycle_template.ipynb       # generic cycle-validation notebook shape
  check_replay_safety.py     # git-diff → pipeline-reach-chain scanner
  check_privacy.py           # pre-commit grep for engagement identifiers
  README.md                  # how to instantiate for a new engagement

debug/                       # gitignored, per-engagement, per-run
  <engagement>/
    diagnosis_*.md           # symptom lists
    fix_cycles/
      _template.md           # cycle doc template (local copy)
      NNN_<slug>.md          # one per cycle
    cycles/
      NNN_<slug>.ipynb       # cycle-specific validation notebook
    artifacts/
      <run_id>/              # HTML exports, CSVs, result.json

docs/                        # versioned, public
  Architecture.md
  Coding_Practices.md
  phased_fix_strategy.md     # this doc — universal
```

The `debug/<engagement>/` subtree name itself should not identify
the engagement — use a neutral code name (e.g. `engagement_a/`)
rather than the client's product name.

---

## The fix-cycle document

Every cycle is a markdown file at
`debug/<engagement>/fix_cycles/NNN_<slug>.md` with this fixed
structure. Template lives at `diagnostic_notebooks/cycle_template.md`
(generic) and is copied into each engagement's `fix_cycles/`
directory as `_template.md` on first use.

```markdown
# Cycle NNN — <one-line summary>

## Problem
<2-5 lines; link to the probe cell(s) that catch it.>

## Impacted files
<grep-verified list; absolute paths.>

## Reference contracts
- Coding_Practices.md § <section> — <one-line why it applies>
- Architecture.md § <section> — <one-line why it applies>

## Downstream contamination risk
<For each changed file/config, list the notebooks or pipeline
stages that consume it. If any heavy stage consumes it, the cycle
must include running those stages — no replay shortcut.>

## Fix
<Code diff or bullet list of edits. Tests-first when non-trivial.>

## Validation
- Offline: <unit tests, ruff, render-and-grep for expected literals>
- Runtime: <cycle notebook path + probe cell IDs that must PASS>

## Close criteria
<Probe cell IDs that must PASS. No prose criteria.>

## Doc updates
<Which sections of Coding_Practices/Architecture this cycle amends,
if any. "No updates needed" is a valid but explicit answer.>

## Privacy check
<Output of diagnostic_notebooks/check_privacy.py on the diff set.
Must be empty before commit.>
```

The template is short on purpose. A cycle that needs more than two
pages is probably two cycles.

### Cycle instantiation — always via `new_cycle.py`

Cycle notebooks and docs are **never hand-copied**. The helper
`diagnostic_notebooks/new_cycle.py` is the single instantiation path
— it:

1. Reads the engagement's `debug/<engagement>/.engagement.yaml`
   (gitignored) for the `framework_repo_root` used to put `src/` on
   `sys.path` on Databricks Workspace-Repos clusters.
2. Copies `diagnostic_notebooks/cycle_template.ipynb` →
   `debug/<engagement>/cycles/NNN_<slug>.ipynb`.
3. Prepends a `@cr:code_system name='framework_path' id=cr-syspath`
   cell with the engagement's path so `customer_retention.*` imports
   resolve before the `init_progress` cell runs.
4. Copies `diagnostic_notebooks/cycle_template.md` →
   `debug/<engagement>/fix_cycles/NNN_<slug>.md`.

One-time engagement setup:

```bash
cat > debug/<engagement>/.engagement.yaml <<EOF
framework_repo_root: /Workspace/Repos/<user>/customer_retention
EOF
```

Per-cycle use:

```bash
python diagnostic_notebooks/new_cycle.py \
    --engagement <engagement> --cycle NNN --slug <short-name>
```

**Why this is mandatory:** the `code_system` cell must run before
any `customer_retention` import or the cycle notebook errors
immediately on Databricks. Making the injection mechanical
eliminates a class of first-run failures that would otherwise burn
a round-trip per cycle. The path stays in the engagement config
file (gitignored), never in a committed notebook — privacy contract
preserved.

The same cell pattern is also required for the engagement's
`probe.ipynb`; that's a one-time setup when the engagement is first
instantiated, not per-cycle.

---

## Diagnostic notebooks (the automation layer)

Validation is a notebook, not a prose protocol. Universal templates
live in `diagnostic_notebooks/`; per-cycle instances live in
`debug/<engagement>/cycles/`.

**Universal layer (versioned):**
- `probe_template.ipynb` — parametrized sanity probe. Takes
  `RUN_ROOT`, `DATASETS` dict, `WINDOWS` list, `FORBIDDEN_SUBSTRINGS`
  as first-cell inputs. No engagement literals. Each engagement
  copies it into `debug/<engagement>/probe.ipynb` and fills the
  config.
- `cycle_template.ipynb` — skeleton that runs a subset of probe
  cells, writes `result.json`. Copied into
  `debug/<engagement>/cycles/NNN_<slug>.ipynb`, then edited.
- `check_replay_safety.py` and `check_privacy.py` — CLI scripts.

**Cycle instances (not versioned):**
- Live in `debug/<engagement>/cycles/`.
- Emit `debug/<engagement>/artifacts/<run_id>/cycle_NNN/result.json`
  with shape `{"cycle": NNN, "run_id": "<id>", "status":
  "PASS|FAIL", "checks": [{cell_id, status, detail}]}`.

If an engagement's probe grows a reusable cell class, it gets
promoted into `diagnostic_notebooks/probe_template.ipynb` — but
scrubbed of engagement content first.

---

## Notebook API notes (known gotchas)

These are wrong-in-a-plausible-way patterns that cost round-trips
during early execution. Every new cycle notebook and every template
update must steer clear — they're encoded here so Claude Code
sessions bootstrapping from this document inherit the lessons
without re-learning them on a Databricks round-trip.

1. **`console.print(...)` does not exist.**
   `from customer_retention.analysis.visualization import console`
   imports the **module** `customer_retention.analysis.visualization.console`,
   not a `rich.Console` instance. Its public helpers are
   `console.header/info/warning/success/error/metric/kv/bullets` — no
   `.print()`, no `rich` markup (`[bold]...[/]`). Use plain `print()`
   for diagnostic output; use `display_table(df)` for DataFrame
   rendering; use `console.header(title)` for section headers if you
   want Markdown-styled output on Databricks. Never embed
   `[bold]`, `[yellow]`, `[/]` tags anywhere — they survive as
   literal text in plain-`print` output.

2. **Run layout is `<experiments_root>/runs/<run_id>/`, not
   `<root>/<run_id>`.**
   `RunNamespace.root` returns the *experiments directory*, not the
   run directory. The intermediate `/runs/` segment is part of the
   convention. The correct construction in a cycle / probe notebook
   is:
   ```python
   RUN_DIR  = Path(EXPERIMENTS_ROOT) / "runs" / _namespace.run_id
   RUN_ROOT = RUN_DIR / "data"
   ```
   Anything like `_namespace.root / _namespace.run_id` silently
   produces a wrong path that later fails on
   `sample_entity_ids.json` / `landing/*` lookups.

3. **`get_experiments_dir()` is unsafe on Databricks Workspace-Repos
   clusters.** It falls back to `$CWD/experiments`, which on a
   Workspace-Repos cluster resolves to the repo mount
   (`/Workspace/Repos/<user>/<repo>/experiments`) — not the Volumes
   catalog where the run actually lives. Every cycle/probe
   notebook must declare `EXPERIMENTS_ROOT` explicitly in its
   config cell, e.g.
   ```python
   EXPERIMENTS_ROOT = "/Volumes/<catalog>/<schema>/experiments"
   ```
   and pass `Path(EXPERIMENTS_ROOT)` directly into
   `RunNamespace(root=..., run_id=...)`. No reliance on
   `get_experiments_dir()` in diagnostic notebooks.

4. **Landing Deltas are post-sample, post-filter. Don't try to
   re-execute NB00 filter SQL from a diagnostic.** `landing/<name>/`
   under `<run>/data/` is materialized by NB01 after the sampler has
   already narrowed to `sample_ids`. It already reflects
   `SAMPLE_FILTER_COLUMNS`. A diagnostic that tries to re-evaluate a
   filter like `ACCOUNT_ID in (SELECT ACCOUNT_ID FROM contract WHERE
   event_type='start')` needs every referenced dataset registered as
   a Spark temp view AND the same post-enrichment state that existed
   during NB00 — neither guaranteed in a later session. The correct
   diagnostic pattern is to compare sampler output to
   `landing/<primary>.unique_entities` directly; the orphan count
   falls out of the subtraction. Do not re-run filter subqueries.

5. **Dataset inventory + entity columns + bridges live in
   `<run>/project_context.yaml`, not in the diagnostic's code.**
   Every cycle / probe notebook loads
   `yaml.safe_load((RUN_DIR / "project_context.yaml").read_text())`
   and reads `datasets[name].entity_column` and
   `datasets[name].key_resolution` from there. Hardcoded
   `DATASETS_IN_MERGE = [...]` lists silently drift from what NB00
   actually registered; config-driven inventory makes the diagnostic
   self-adapting per engagement. Same pattern for `sample_filter`,
   `intent`, and any NB00 user-code state — project_context is the
   single source of truth.

6. **Two "cell ids" exist — don't conflate them.** Every code cell
   has two independent identifiers:
   - **`cell.id`** in the notebook JSON (nbformat metadata, used by
     nbformat itself — e.g. `"c1-step2"`).
   - **`id=` in the `# @cr:` tag** on line 1 of the cell source
     (permanent sync-engine identity — e.g. `"c1-step2-recomp"`).
   They are deliberately different; `churnkit-sync` matches on the
   tag id, nbformat matches on the JSON id. A patch script that
   filters cells by `c['id'] == 'c1-step2-recomp'` silently no-ops,
   because that string lives in `c['source'][0]` not `c['id']`. When
   mass-editing notebook JSON programmatically, always filter by the
   JSON `cell.id` field (or parse the tag out of the source); never
   mix them. Round-trip cost of getting this wrong: one Databricks
   import to discover the edit didn't land.

7. **`framework/phase_map.yaml` is generated; regenerate before
   committing notebook changes.** The pre-commit hook
   `framework-phase-map` does byte-equality between the committed
   file and a fresh run of `scripts/build_framework_phase_map.py`
   that walks every `.ipynb`. Any cell modified / added / tagged /
   repositioned makes the map stale and blocks the commit. Fix is
   one command: `python scripts/build_framework_phase_map.py &&
   git add framework/phase_map.yaml`. Run it as the last step of
   offline pre-flight, right before `git commit`. Every cycle that
   touches `exploration_notebooks/*.ipynb` needs this; cycles that
   only touch `src/` don't.

The universal templates under `diagnostic_notebooks/` already embed
these patterns; engagement-specific instantiations must not undo
them.

---

## Contamination safety

A change made early in the pipeline can alter the schema consumed
late. "Skipping heavy stages" is only safe when the change
demonstrably doesn't reach them. Rule:

**Before closing any cycle that skipped any stage**, run:

```bash
python diagnostic_notebooks/check_replay_safety.py \
  --cycle NNN --since <base-ref>
```

The script does mechanical checks only:

1. `git diff --name-only <base-ref>` → list of changed files.
2. For each, grep the pipeline for imports / references and walk
   the call graph. If anything downstream of a stage that the
   cycle plans to skip touches it, flag.
3. For changed notebook `.ipynb` files, parse cells tagged
   `@cr:code` / `@cr:user_code` and grep downstream notebooks for
   cell-name references.
4. Exits 0 (replay safe) or 1 (must re-run the flagged stages)
   with the specific reach-chain printed.

If the script flags contamination, the cycle either:
- includes a full run of the flagged stages in its validation, or
- is deferred until grouped with other cycles that already require
  a full run (to amortize the cost).

There is no judgement call. The scan is the gate.

---

## Living reference docs

Each cycle's opening section cites the `Coding_Practices.md` and
`Architecture.md` sections it touches. The closing `## Doc updates`
section is required — even "no updates needed" is an explicit
statement, not an omission.

A cycle that introduces a recurring pattern (e.g. "how to wire a
new override from a user-code cell through to generated code")
must update `Architecture.md` with that pattern as a named section.
The next cycle facing the same class of bug then references that
section directly. Over N cycles this builds a working reference
rather than re-deriving the same knowledge.

If a pattern stabilizes enough to warrant tooling, it becomes a
Claude Code skill under `.claude/skills/` or a generic validator
under `diagnostic_notebooks/`. Skills emerge from repeated cycle
experience, not speculation. Skills committed to the repo are
engagement-agnostic like anything else in version control.

---

## Privacy check

Before any commit from a cycle:

```bash
python diagnostic_notebooks/check_privacy.py --diff <base-ref>
```

The script greps the staged changes for:
- A configurable denylist of client names, product names, segment
  labels (stored in an engagement-local `.check_privacy.yaml`
  under `debug/<engagement>/`, itself gitignored).
- Common catalog/schema prefixes that identify an engagement's
  Databricks environment.
- Common CRM record-ID formats (15/18-char regex, etc.).
- Any file path under `debug/` that got staged by mistake.

Exit 0 = clean; exit 1 = blocks commit with the specific hits.

---

## Starting a new engagement

Under this methodology the entry sequence for any new engagement is:

1. Ensure `diagnostic_notebooks/` contains `probe_template.ipynb`,
   `cycle_template.ipynb`, `cycle_template.md`,
   `check_replay_safety.py`, `check_privacy.py`, `new_cycle.py`.
   Scrub any engagement literals on contribution.
2. Instantiate `debug/<engagement>/` per
   `diagnostic_notebooks/README.md` — `.engagement.yaml`,
   `.check_privacy.yaml`, `probe.ipynb` filled in.
3. Run the probe once against the current run to produce
   `result.json` — that's the symptom list the first cycle
   works against.
4. Open cycle 001 via `new_cycle.py` — highest-value single fix,
   smallest downstream contamination.
5. Execute cycle 001 to close. No other cycles start until 001
   closes.
6. Decide cycle 002 based on what 001 taught us.

The sequence matters less than the discipline: one cycle at a time,
each closes cleanly, the probe proves it, reference docs stay
current, nothing engagement-identifying leaves `debug/`.

---

## What this strategy will not do

- Make runtime observation direct. Round-trips are real; the
  strategy only makes them informative.
- Eliminate heavy-stage runs. It avoids paying for them on
  cascaded bugs; periodic full runs still happen.
- Cover modelling-stage bugs (the heaviest stage's internals).
  Those need their own probe once the upstream is proven clean.
- Substitute for pairing when a bug lives in runtime-internal
  state (cell caches, cluster env, widget state). A cycle that
  fails twice on the same fix is a signal to pair, not to try a
  third time.

The strategy is designed to be usable starting from the next fix
cycle with no speculative infrastructure. If any section here
delays the first real fix, cut that section — the doc serves the
fix work, not vice versa.
