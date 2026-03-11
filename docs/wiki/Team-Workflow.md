# Team Workflow

This guide describes how a team of data scientists can work independently on the same ChurnKit project and merge their configuration decisions into a shared baseline.

## Architecture

```
Shared "main" copy                Individual copies
(Databricks Repo or               (each developer's workspace)
 shared workspace folder)
                                  ┌─ /Workspace/Users/alice@co/exploration_notebooks/
┌─ /Repos/team/churnkit/  ◄──────┤
│  exploration_notebooks/         └─ /Workspace/Users/bob@co/exploration_notebooks/
│  ├── 00_start_here.ipynb              │                            │
│  ├── 01_data_discovery.ipynb          │  independent runs          │
│  ├── ...                              ▼                            ▼
│  └── 10_spec_generation.ipynb    Alice's run namespace        Bob's run namespace
│                                  runs/churn-a1b2c3d4/         runs/churn-b5c6d7e8/
```

- **Shared copy** is the agreed-upon baseline — the "main branch" for notebook configurations
- **Individual copies** are where each developer explores data, modifies `@cr:config` cells, writes `@cr:user_code`, and runs the pipeline
- **Merge** reconciles individual changes back into the shared copy using the three-way merge tool

## Setup

### 1. Create the shared copy

One team member initializes the project. This becomes the shared baseline.

**Option A — Databricks Repos (recommended for version control):**

```python
# Clone the team repo in Databricks
# The exploration_notebooks/ directory inside it is the shared copy
```

**Option B — Shared workspace folder:**

```python
from customer_retention import databricks_init

result = databricks_init(
    catalog="analytics",
    schema="churnkit",
    workspace_path="Shared/team_project/customer_retention",
    model_name="customer_retention",
)
```

### 2. Each developer creates their own copy

Each developer copies the shared notebooks to their own workspace:

```python
import shutil
from pathlib import Path

shared = Path("/Workspace/Shared/team_project/customer_retention/exploration_notebooks")
mine = Path("/Workspace/Users/alice@company.com/exploration_notebooks")
shutil.copytree(shared, mine, dirs_exist_ok=True)
```

Or on Databricks, use the workspace UI: right-click the shared folder, export, then import into your user workspace.

### 3. Initialize your own run

Each developer calls `databricks_init()` in their own workspace with the **same catalog and schema** but their **own workspace path**:

```python
result = databricks_init(
    catalog="analytics",
    schema="churnkit",
    workspace_path="Users/alice@company.com/customer_retention",
    model_name="customer_retention",
)
```

Runs are isolated by `run_id` — Alice and Bob can explore the same dataset simultaneously without conflict.

## Daily Workflow

1. **Work in your own copy.** Modify `@cr:config` cells (drop columns, set prediction horizons, configure type overrides) and `@cr:user_code` cells (custom target logic, business context). Run notebooks as usual.

2. **Test your configuration.** Run through notebooks 00–08 to validate findings and recommendations. Run notebook 10 to generate the pipeline and verify it produces correct results.

3. **Merge into the shared copy** when your configuration is tested and ready. This is where the merge tool comes in.

## Merging

The merge tool performs a three-way comparison:

- **Base** — the installed package notebooks (the framework's original defaults)
- **Theirs** — one developer's copy
- **Ours** — another developer's copy (or the shared copy)

### What gets merged

| Cell type | Merge behavior |
|-----------|---------------|
| `@cr:code` | Always taken from base (framework code) |
| `@cr:doc` (markdown) | Always taken from base |
| `@cr:config` | Three-way merge with structural auto-resolution |
| `@cr:user_code` | Three-way merge (textual — conflicts if both changed) |
| User-added cells | Included from both sides |

### Auto-resolution rules for `@cr:config` cells

Config cells contain Python assignments. When both sides modified the same cell, the merge tool parses assignments with `ast` and merges structurally:

| Assignment type | Example | Auto-resolution |
|----------------|---------|-----------------|
| Dict (by dataset) | `DROP_COLUMNS = {"emails": [...], "transactions": [...]}` | Key-level union — different dataset keys merge cleanly |
| List | `EXCLUDE_DATASETS = ["a", "b"]` | Union + dedup |
| Scalar | `PREDICTION_HORIZON = 90` | Conflict if both changed to different values |
| Complex (enums, calls) | `TEMPORAL_POSTURE = TemporalPosture.STABLE` | Conflict if both changed |

When a conflict cannot be auto-resolved, the output cell contains conflict markers:

```python
# @cr:config name='intent_config' id=ea85abf2
# <<<< alice
PREDICTION_HORIZON = 120
# ====
PREDICTION_HORIZON = 60
# >>>> bob
```

Open the merged notebook and resolve these manually before proceeding.

### Merge rules for `@cr:user_code` cells

User code cells contain arbitrary Python — target derivation logic, business context definitions, custom analysis functions. Unlike config cells, these cannot be structurally parsed and merged. The merge tool uses **textual three-way comparison**:

| Scenario | Result |
|----------|--------|
| Only one side changed the cell | The changed version is accepted |
| Both changed to the same code | Accepted (no conflict) |
| Both changed differently | **Conflict** — both versions shown with markers |

A user_code conflict looks like this:

```python
# @cr:user_code name='business_context' id=064d7a18
# <<<< alice
BUSINESS_CONTEXT = {
    "project_name": "Customer Churn — Enterprise",
    "business_objective": "Reduce churn by 30%",
}
# ====
BUSINESS_CONTEXT = {
    "project_name": "Customer Churn — SMB",
    "business_objective": "Identify at-risk accounts within 14 days",
}
# >>>> bob
```

Because user_code cells are free-form Python, there is no automatic resolution — the team must decide which version to keep (or combine them manually).

**Practical tip:** If two developers need to write different custom logic in the same notebook, consider having each person add their own *new* `@cr:user_code` cell with a unique `id` rather than editing the same one. User-added cells from both sides are included in the merge output without conflict.

## How to Merge

### From the command line

**Merge all notebooks in two directories:**

```bash
churnkit-merge dir \
    /Workspace/Users/alice@co/exploration_notebooks \
    /Workspace/Users/bob@co/exploration_notebooks \
    -o /Workspace/Shared/team_project/merged_notebooks \
    --theirs-label alice --ours-label bob
```

**Merge a single notebook:**

```bash
churnkit-merge pair \
    /Workspace/Users/alice@co/exploration_notebooks/00_start_here.ipynb \
    /Workspace/Users/bob@co/exploration_notebooks/00_start_here.ipynb \
    -o /Workspace/Shared/team_project/merged/00_start_here.ipynb \
    --theirs-label alice --ours-label bob
```

**Preview without writing (dry run):**

```bash
churnkit-merge dir alice_notebooks/ bob_notebooks/ -o merged/ --dry-run
```

**Fail in CI if conflicts remain:**

```bash
churnkit-merge dir alice/ bob/ -o merged/ --fail-on-conflict
```

The `--base-dir` flag is optional. When omitted, base notebooks are resolved from the installed ChurnKit package automatically.

### From a Databricks notebook cell

Create a utility notebook (e.g., `99_merge_notebooks.ipynb`) in your workspace:

```python
# @cr:config name='merge_config' id=<generate-your-id>
from pathlib import Path
from customer_retention.generators.notebook_merge.cli import (
    compute_all_merges, apply_pending_merges,
)

ALICE_DIR = Path("/Workspace/Users/alice@company.com/exploration_notebooks")
BOB_DIR = Path("/Workspace/Users/bob@company.com/exploration_notebooks")
OUTPUT_DIR = Path("/Workspace/Shared/team_project/merged_notebooks")

# Base notebooks from the installed package
from customer_retention.generators.notebook_generator.project_init import ProjectInitializer
BASE_DIR = ProjectInitializer(project_name="")._get_exploration_source_dir()
```

```python
# @cr:code name='run_merge' id=<generate-your-id>
pending = compute_all_merges(
    ALICE_DIR, BOB_DIR, BASE_DIR,
    theirs_label="alice", ours_label="bob",
)

for name, pm in pending.items():
    status = "CONFLICT" if pm.report.has_conflicts else "ok"
    print(f"  {name}: {pm.report.format_summary()} [{status}]")

conflicts = sum(1 for pm in pending.values() if pm.report.has_conflicts)
print(f"\n{len(pending)} notebooks, {conflicts} with conflicts")
```

```python
# @cr:code name='apply_merge' id=<generate-your-id>
if conflicts == 0:
    apply_pending_merges(pending, OUTPUT_DIR)
    print(f"Merged notebooks written to {OUTPUT_DIR}")
else:
    apply_pending_merges(pending, OUTPUT_DIR)
    print(f"Merged notebooks written to {OUTPUT_DIR}")
    print("WARNING: some notebooks have unresolved conflicts — review them before using")
```

### Merging a single notebook pair from a cell

```python
from pathlib import Path
from customer_retention.generators.notebook_merge.cli import merge_notebook_pair

report = merge_notebook_pair(
    theirs_path=Path("/Workspace/Users/alice@co/exploration_notebooks/00_start_here.ipynb"),
    ours_path=Path("/Workspace/Users/bob@co/exploration_notebooks/00_start_here.ipynb"),
    output_path=Path("/Workspace/Shared/merged/00_start_here.ipynb"),
    theirs_label="alice", ours_label="bob",
)
print(report.format_details())
```

### Using the merge engine directly

For programmatic control, use `NotebookMergeEngine`:

```python
import nbformat
from customer_retention.generators.notebook_merge import NotebookMergeEngine

engine = NotebookMergeEngine(theirs_label="alice", ours_label="bob")

base_nb = nbformat.read("/Workspace/.../base/00_start_here.ipynb", as_version=4)
alice_nb = nbformat.read("/Workspace/Users/alice@co/.../00_start_here.ipynb", as_version=4)
bob_nb = nbformat.read("/Workspace/Users/bob@co/.../00_start_here.ipynb", as_version=4)

merged_nb, report = engine.merge(base_nb, alice_nb, bob_nb)

if not report.has_conflicts:
    nbformat.write(merged_nb, "/Workspace/Shared/.../00_start_here.ipynb")
else:
    print("Conflicts found:")
    for entry in report.conflict_entries:
        print(f"  Cell {entry.cell_id}: {entry.message}")
```

## Recommended Team Process

### Converging on a shared configuration

```
 Alice explores ──► tests config ──► merges into shared ──┐
                                                           ├──► shared copy
 Bob explores ────► tests config ──► merges into shared ──┘    is the baseline
```

1. **Alice** finishes configuring `DROP_COLUMNS`, `PREDICTION_HORIZON`, `TYPE_OVERRIDES` for the `emails` dataset. She runs notebooks 00–08 and validates findings.

2. **Alice merges her copy with the current shared copy:**

   ```bash
   churnkit-merge dir \
       /Workspace/Users/alice@co/exploration_notebooks \
       /Workspace/Shared/team_project/exploration_notebooks \
       -o /Workspace/Shared/team_project/exploration_notebooks \
       --theirs-label alice --ours-label shared
   ```

   Because the shared copy hasn't changed since Alice branched, all of Alice's changes are accepted cleanly.

3. **Bob** finishes configuring the `transactions` dataset. His `DROP_COLUMNS` dict has different keys than Alice's (different datasets). He merges:

   ```bash
   churnkit-merge dir \
       /Workspace/Users/bob@co/exploration_notebooks \
       /Workspace/Shared/team_project/exploration_notebooks \
       -o /Workspace/Shared/team_project/exploration_notebooks \
       --theirs-label bob --ours-label shared
   ```

   The dict configs auto-merge at key level: Alice's `"emails"` keys and Bob's `"transactions"` keys combine into a single dict. No conflict.

4. If both Alice and Bob changed the same scalar (e.g., `PREDICTION_HORIZON`), the merge produces a conflict marker. They review the merged notebook, decide on the value, and remove the conflict markers.

### After a framework upgrade

When ChurnKit is updated, use `churnkit-sync` to upgrade the shared copy first:

```bash
churnkit-sync --repo-dir /path/to/new/exploration_notebooks \
              --user-dir /Workspace/Shared/team_project/exploration_notebooks
```

This updates `@cr:code` cells while preserving the team's `@cr:config` and `@cr:user_code` cells. Then each developer syncs their own copy from the shared baseline.

## Conflict Resolution

When a merge produces conflicts:

1. Open the merged notebook in Databricks or Jupyter
2. Search for `# <<<<` to find conflict markers
3. Choose the correct value, remove the markers (`# <<<<`, `# ====`, `# >>>>`)
4. Run the notebook to validate

Conflict markers are valid Python comments, so the notebook will load without errors — but the config variable will not be set correctly until you resolve the markers.

## Merge Report

Every merge produces a report showing what happened to each cell:

```
  00_start_here.ipynb — merged: Merge: 3 unchanged, 2 took_theirs, 1 auto_merged, 0 conflict, ...
  01_data_discovery.ipynb — merged: Merge: 5 unchanged, 0 took_theirs, 0 took_ours, 1 conflict, ...  ** CONFLICTS **
  02_source_integrity.ipynb — no changes

Merged 14 notebook(s), 1 with conflicts
```

| Action | Meaning |
|--------|---------|
| `unchanged` | Cell identical in all three versions |
| `from_base` | Framework cell taken from base |
| `took_theirs` | Only theirs changed — accepted |
| `took_ours` | Only ours changed — accepted |
| `both_same` | Both changed identically — accepted |
| `auto_merged` | Structural merge succeeded (dict union, list dedup) |
| `conflict` | Both changed differently — needs manual resolution |
| `added_theirs` / `added_ours` | User-added cell from one side |
