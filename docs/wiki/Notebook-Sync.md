# Notebook Cell-Level Sync

Exploration notebooks ship with ChurnKit and get cloned into user projects via `churnkit-init`. When the package updates, users face a choice: lose their configuration changes or skip the update. The notebook sync tool solves this by merging repo code updates while preserving user-edited cells.

## How It Works

Every code cell in an exploration notebook is classified into one of three types, identified by a magic comment on line 1:

| Tag | Meaning | On sync |
|-----|---------|---------|
| `# @cr:config` | User configuration (ALL_CAPS variables) | **Preserved** -- your values stay |
| `# @cr:user_code` | User-written logic (target derivation, business context) | **Preserved** -- your code stays |
| `# @cr:code` | Repo framework code | **Overwritten** from the new release |

Markdown cells are always overwritten from the repo version (documentation improvements come through automatically).

### What each type looks like

**Config cell** -- variables you customize per project:

```python
# @cr:config
PROJECT_NAME = "my_project"
LIGHT_RUN = False
SAMPLE_FRACTION = None
SAMPLE_ENTITY_COUNT = None
MAX_GRID_DATES = None
```

**User code cell** -- logic you write yourself:

```python
# @cr:user_code
BUSINESS_CONTEXT = {
    "project_name": "Customer Churn Prediction",
    "business_objective": "Reduce churn by 20%",
    "stakeholders": ["Marketing", "Customer Success"],
}
```

**Repo code cell** -- framework logic maintained by the package. All framework cells are explicitly tagged with `# @cr:code`:

```python
# @cr:code
from customer_retention.analysis.auto_explorer import DatasetFingerprinter

fingerprinter = DatasetFingerprinter()
results = fingerprinter.profile_all(datasets)
```

Every code cell in the repo notebooks has an explicit tag. An untagged code cell is a signal that the user created it without tagging -- the sync tool will prompt before removing it.

## Cell ID Convention

Every cell has a standardized ID following the pattern `{notebook_prefix}-{sequence:03d}`:

```
nb00-001, nb00-002, nb00-003, ...   (00_start_here.ipynb)
nb01-001, nb01-002, ...             (01_data_discovery.ipynb)
nb01a-001, nb01a-002, ...           (01a_temporal_deep_dive.ipynb)
nb09-001, nb09-002, ...             (09_business_alignment.ipynb)
```

Cell IDs are the anchor for sync -- the tool matches cells between repo and user notebooks by ID. Do not rename cell IDs manually.

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

## Merge Algorithm

The sync engine walks cells in **repo order** (canonical ordering) and applies these rules:

1. **Cell exists in both repo and user, tagged `config` or `user_code`** -- keep user's version (source, outputs, execution count all preserved)
2. **Cell exists in both, tagged `code` or untagged** -- overwrite from repo (outputs stripped)
3. **Cell in repo but not in user** -- insert from repo (new cell added in the update)
4. **Cell in user but not in repo, tagged `config`/`user_code` or markdown** -- keep near its original position
5. **Cell in user but not in repo, untagged code** -- drop after confirmation (orphaned or user-created without tag)

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
# @cr:config
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

- **Tag your cell** with `# @cr:config` or `# @cr:user_code` on line 1. Without a tag, user-added code cells will be dropped during sync (they look like orphaned repo cells).
- **Markdown cells** you add are always kept.
- User-added cells are inserted after their nearest preceding repo cell to maintain logical ordering.

## For Notebook Developers

### Tagging guidelines

When adding or modifying exploration notebooks in the repo:

- Pure configuration variables (ALL_CAPS assignments) go in `# @cr:config` cells
- Cells where users write custom logic (target derivation, business context, success metrics) get `# @cr:user_code`
- Everything else is repo code -- **always** tag with `# @cr:code`. All framework cells must be explicitly tagged so that "untagged" means "user forgot to tag"
- Keep config and code in **separate cells**. The migration script split blended cells for this reason.

### CI guards

Five structural tests run on every CI build to prevent accidental de-standardization:

- `test_all_cells_have_standardized_ids` -- every cell ID matches `nb\w+-\d{3}`
- `test_no_duplicate_ids_within_notebook` -- no two cells share an ID
- `test_cell_ids_are_valid_format` -- IDs are present and reasonable length
- `test_all_code_cells_have_explicit_tags` -- every code cell starts with `# @cr:` (config, user_code, or code)
- `test_config_cells_have_magic_comments` -- pure config cells have the `# @cr:config` tag

### Re-running the migration

If you add new notebooks or restructure existing ones:

```bash
python scripts/notebooks/migrate_notebook_cell_ids.py --dry-run
python scripts/notebooks/migrate_notebook_cell_ids.py
```

The migration is idempotent -- running it twice produces the same result. It standardizes cell IDs and adds magic comment tags based on the split/tag maps defined in the script.

### Tagging new framework cells

After adding new code cells to notebooks, tag them with `# @cr:code`:

```bash
python scripts/notebooks/tag_framework_cells.py --dry-run
python scripts/notebooks/tag_framework_cells.py
```

This is also idempotent -- it only tags cells that don't already have a `# @cr:` magic comment.

## Troubleshooting

**"Cell ID not found" during sync** -- The user notebook has different cell IDs than the repo. This happens if the notebook was created before migration. Run the migration script on the user's notebooks first.

**Config values unexpectedly reset** -- The cell is missing its `# @cr:config` tag. Add the tag on line 1 and re-sync.

**User-added cell disappeared** -- The cell was untagged code. Add `# @cr:config` or `# @cr:user_code` on line 1 to protect it from removal.

**`databricks_init()` call lost after sync** -- The call was inside a repo notebook without a `# @cr:config` tag. Move it to a standalone `00_databricks_setup.ipynb` notebook (see [Databricks Initialization](#databricks-initialization)). Standalone notebooks are never touched by sync.

**Backup files everywhere** -- `.bak` files are created by default. Use `--no-backup` to disable, or clean up with `find . -name "*.ipynb.bak" -delete`.
