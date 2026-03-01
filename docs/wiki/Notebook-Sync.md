# Notebook Cell-Level Sync

Exploration notebooks ship with ChurnKit and get cloned into user projects via `churnkit-init`. When the package updates, users face a choice: lose their configuration changes or skip the update. The notebook sync tool solves this by merging repo code updates while preserving user-edited cells.

## How It Works

Every cell in an exploration notebook is classified into one of four types, identified by a tag on line 1:

| Tag | Cell type | Meaning | On sync |
|-----|-----------|---------|---------|
| `# @cr:config` | Code | User configuration (ALL_CAPS variables) | **Preserved** -- your values stay |
| `# @cr:user_code` | Code | User-written logic (target derivation, business context) | **Preserved** -- your code stays |
| `# @cr:code` | Code | Repo framework code | **Overwritten** from the new release |
| `[//]: # (cr:doc ...)` | Markdown | Documentation / section headings | **Overwritten** from the new release |

Markdown cells use an invisible markdown comment (standard link-reference syntax) that renders as nothing in any markdown viewer.

### Tag Format

Code cells use a Python comment tag on line 1:

```
# @cr:TYPE name='descriptive_name' id=HEXUUID
```

Markdown cells use an invisible markdown link-reference tag on line 1:

```
[//]: # (cr:doc name='descriptive_name' id=HEXUUID)
```

| Attribute | Purpose | Example |
|-----------|---------|---------|
| `TYPE` | Cell sync behavior (`config`, `user_code`, `code`, `doc`) | `config` |
| `name` | Human-readable label describing the cell's purpose | `project_settings` |
| `id` | Stable 8-character hex UUID for cell matching | `846f56cb` |

The `id` is the **primary key** for matching cells between repo and user notebooks. It never changes, even when cells are reordered or new cells are inserted. The `name` is for human readability when maintaining notebooks.

### What each type looks like

**Config cell** -- variables you customize per project:

```python
# @cr:config name='project_settings' id=846f56cb
PROJECT_NAME = "my_project"
LIGHT_RUN = False
SAMPLE_FRACTION = None
SAMPLE_ENTITY_COUNT = None
MAX_GRID_DATES = None
```

**User code cell** -- logic you write yourself:

```python
# @cr:user_code name='business_context' id=064d7a18
BUSINESS_CONTEXT = {
    "project_name": "Customer Churn Prediction",
    "business_objective": "Reduce churn by 20%",
    "stakeholders": ["Marketing", "Customer Success"],
}
```

**Repo code cell** -- framework logic maintained by the package:

```python
# @cr:code name='fingerprint_datasets' id=43e89806
from customer_retention.analysis.auto_explorer import DatasetFingerprinter

fingerprinter = DatasetFingerprinter()
results = fingerprinter.profile_all(datasets)
```

**Doc cell** -- markdown documentation maintained by the package:

```markdown
[//]: # (cr:doc name='project_metadata' id=cca55d99)
## 0.1 Project Metadata

Configure your project name and settings below.
```

Every code cell in the repo notebooks has an explicit tag, and every markdown cell has a `cr:doc` tag. An untagged code cell is a signal that the user created it without tagging -- the sync tool will prompt before removing it.

## Cell IDs and Embedded Matching

Every cell has a stable 8-character hex UUID as its ID (e.g., `846f56cb`). This UUID appears in two places:

1. **Jupyter cell metadata** -- the `cell.id` field in the notebook JSON
2. **Cell source** -- embedded in the tag line (`# @cr:code ... id=846f56cb` for code cells, `[//]: # (cr:doc ... id=846f56cb)` for markdown cells)

The embedded ID is the authoritative source. When notebooks are re-saved by Databricks or other platforms, Jupyter cell metadata can be stripped or regenerated. The sync engine reads the `id=` attribute from the tag line and sets `cell.id` accordingly before matching. This makes cell matching robust across any platform for both code and markdown cells.

### Why embedded IDs

Databricks workspaces can strip or regenerate nbformat cell IDs when notebooks are re-saved. Sequential IDs (like `nb00-003`) break when users insert cells between them. Embedding a stable UUID in the source code comment solves both problems -- the ID travels with the cell content itself and never changes regardless of cell position or platform behavior.

## Using the Sync Tool

### Preview changes (recommended first step)

```bash
churnkit-sync --repo-dir /path/to/new/exploration_notebooks --dry-run
```

This shows what would change without modifying any files.

### Sync all notebooks

```bash
churnkit-sync --repo-dir /path/to/new/exploration_notebooks
```

By default, a `.bak` backup is created for every modified notebook. Disable with `--no-backup`.

### Sync a single notebook

```bash
churnkit-sync --repo-dir ./updated --notebook 01_data_discovery.ipynb
```

### Full options

```
churnkit-sync --repo-dir DIR        # Required: directory with updated notebooks
              --user-dir DIR        # Your notebooks (default: exploration_notebooks/)
              --notebook FILE       # Sync only this notebook
              --dry-run             # Preview without writing
              --no-backup           # Skip .bak creation
              --force               # Skip confirmation prompt when cells are removed
```

### Typical upgrade workflow

```bash
# 1. Install the new package version
pip install --upgrade churnkit

# 2. Extract the new notebooks to a temp directory
python -c "
import importlib.resources as res
import shutil
src = res.files('churnkit') / 'exploration_notebooks'
shutil.copytree(src, '/tmp/new_notebooks', dirs_exist_ok=True)
"

# 3. Preview what will change
churnkit-sync --repo-dir /tmp/new_notebooks --dry-run

# 4. Apply the sync
churnkit-sync --repo-dir /tmp/new_notebooks

# 5. Clean up
rm -rf /tmp/new_notebooks
```

### Databricks automatic sync

On Databricks, `databricks_init()` automatically syncs exploration notebooks when they already exist in the workspace. It uses the same `NotebookSyncEngine` -- `@cr:code` cells are updated from the installed package, while `@cr:config` and `@cr:user_code` cells are preserved. New notebooks are copied; existing notebooks are synced. The result includes both `notebooks_copied` and `notebooks_synced` lists.

## Merge Algorithm

The sync engine walks cells in **repo order** (canonical ordering) and applies these rules:

1. **Cell exists in both repo and user, tagged `config` or `user_code`** -- keep user's version (source, outputs, execution count all preserved)
2. **Cell exists in both, tagged `code` or untagged** -- overwrite from repo (outputs stripped)
3. **Cell in repo but not in user** -- insert from repo (new cell added in the update)
4. **Cell in user but not in repo, tagged `config`/`user_code` or markdown** -- keep near its original position
5. **Cell in user but not in repo, untagged code** -- drop after confirmation (orphaned or user-created without tag)

Cell matching is by embedded `id=` attribute, not by position or sequential numbering.

### Removal confirmation

When sync detects cells that will be removed, it pauses and shows a preview of each cell's content before proceeding. You must confirm (`y`) or cancel (`n`). This prevents accidental loss of user-created cells that were not tagged.

- `--dry-run` shows removals in the report without prompting
- `--force` skips the prompt and removes cells immediately
- Removal previews show the first 5 lines of each cell

### Output handling

| Cell type | Outputs | Execution count |
|-----------|---------|-----------------|
| `config` / `user_code` | Preserved | Preserved |
| `code` / untagged | Cleared | Cleared |
| Markdown | N/A | N/A |

Config and user_code cells keep their outputs because the user may have run them and the results are meaningful. Code cells get their outputs stripped since the source was replaced.

## Databricks Initialization

On Databricks, you call `databricks_init()` once to bind a catalog, schema, workspace path, and model name to your project. This cell contains environment-specific values that must never be overwritten by sync.

Put the call in its own notebook -- `00_databricks_setup.ipynb` -- inside your `exploration_notebooks/` directory:

```python
# @cr:config name='databricks_setup' id=<your-uuid>
from customer_retention.integrations.databricks_init import databricks_init

result = databricks_init(
    catalog="my_existing_catalog",
    schema="my_existing_schema",
    workspace_path="Users/user@example.com/customer_retention",
    model_name="customer_retention",
)
```

Run this notebook once. The result is persisted and all downstream notebooks pick it up via `CR_CATALOG` / `CR_SCHEMA` env vars automatically.

This notebook is completely invisible to sync. The sync tool only processes notebooks that exist in the **repo** directory -- since `00_databricks_setup.ipynb` exists only in your local copy, it is never read, never compared, and never modified. You can upgrade ChurnKit as many times as you like; your Databricks binding stays exactly as you wrote it.

## Adding Your Own Cells

You can add cells to notebooks and they will survive sync -- with one rule:

- **Tag your cell** with `# @cr:config name='your_name' id=<8-char-hex>` or `# @cr:user_code name='your_name' id=<8-char-hex>` on line 1. Generate a unique hex ID with `python -c "import uuid; print(uuid.uuid4().hex[:8])"`. Without a tag, user-added code cells will be dropped during sync (they look like orphaned repo cells).
- **Markdown cells** you add are always kept.
- User-added cells are inserted after their nearest preceding repo cell to maintain logical ordering.

## For Notebook Developers

### Tagging guidelines

When adding or modifying exploration notebooks in the repo:

- Pure configuration variables (ALL_CAPS assignments) go in `# @cr:config` cells
- Cells where users write custom logic (target derivation, business context, success metrics) get `# @cr:user_code`
- Everything else is repo code -- **always** tag with `# @cr:code`. All framework cells must be explicitly tagged so that "untagged" means "user forgot to tag"
- Markdown cells must have a `[//]: # (cr:doc name='...' id=...)` tag on line 1. Use `scripts/notebooks/tag_markdown_cells.py` to tag new markdown cells automatically
- Keep config and code in **separate cells**
- Every tag must include `name='descriptive_name'` and `id=<8-char-hex>` -- bare tags like `# @cr:code` are not valid
- The `name` should be a short snake_case description of the cell's purpose (e.g., `load_findings`, `detect_target`, `save_pattern_findings`)
- The `id` must be a unique 8-character hex string. Generate with `python -c "import uuid; print(uuid.uuid4().hex[:8])"`
- IDs are permanent -- once assigned, they never change even if the cell is moved or its content is rewritten

### CI guards

Seven structural tests run on every CI build to prevent accidental de-standardization:

- `test_all_cells_have_standardized_ids` -- every cell ID is an 8-character hex string
- `test_no_duplicate_ids_within_notebook` -- no two cells share an ID
- `test_cell_ids_are_valid_format` -- IDs are present and reasonable length
- `test_all_code_cells_have_explicit_tags` -- every code cell has `# @cr:TYPE name='...' id=...` format
- `test_all_markdown_cells_have_doc_tags` -- every markdown cell has `[//]: # (cr:doc name='...' id=...)` format
- `test_embedded_id_matches_cell_id` -- the `id=` in the tag line matches the Jupyter `cell.id` (both code and markdown cells)
- `test_config_cells_have_magic_comments` -- pure config cells have the `# @cr:config` tag

## Troubleshooting

**"Cell ID not found" during sync** -- The user notebook has cells whose embedded IDs don't match the repo. This can happen if a cell was duplicated or its tag was manually edited. Check the `id=` values in the tag lines.

**Config values unexpectedly reset** -- The cell is missing its `# @cr:config` tag. Add the full tag on line 1 (with `name=` and `id=`) and re-sync.

**User-added cell disappeared** -- The cell was untagged code. Add `# @cr:config name='your_name' id=<hex>` or `# @cr:user_code name='your_name' id=<hex>` on line 1 to protect it from removal.

**`databricks_init()` call lost after sync** -- The call was inside a repo notebook without a `# @cr:config` tag. Move it to a standalone `00_databricks_setup.ipynb` notebook (see [Databricks Initialization](#databricks-initialization)). Standalone notebooks are never touched by sync.

**Backup files everywhere** -- `.bak` files are created by default. Use `--no-backup` to disable, or clean up with `find . -name "*.ipynb.bak" -delete`.

**Databricks stripped cell IDs** -- Not a problem. The sync engine reads embedded `id=` attributes from both code tag lines (`# @cr:`) and markdown doc tag lines (`[//]: # (cr:doc ...)`) and restores `cell.id` before matching. As long as the tag line is intact, sync works correctly.

**Duplicate markdown sections after sync** -- The markdown cell was missing its `[//]: # (cr:doc ...)` tag, so the sync engine couldn't match it. Run `python scripts/notebooks/tag_markdown_cells.py` to add doc tags to all untagged markdown cells.
