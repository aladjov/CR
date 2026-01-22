# Scripts

Utility scripts organized by purpose.

## Directory Structure

```
scripts/
├── databricks/     # Databricks deployment and runtime management
├── data/           # Data generation, migration, and snapshot tools
├── notebooks/      # Notebook testing, export, and project initialization
└── templates/      # HTML templates for tutorial export
```

## databricks/

Scripts for Databricks cluster initialization and dependency management.

| Script | Description |
|--------|-------------|
| `build_wheel.sh` | Build wheel package for Databricks deployment |
| `capture_runtime.py` | Capture baseline packages from Databricks Runtime |
| `dbr_init.sh` | Cluster init script (wheel/repo/huggingface modes) |
| `generate_constraints.py` | Generate filtered pip constraints from DBR baseline |
| `notebook_setup.py` | Notebook-scoped dependency installation helper |

## data/

Scripts for test data generation, data migration, and snapshot management.

| Script | Description |
|--------|-------------|
| `create_snapshot.py` | Create versioned point-in-time training snapshots |
| `generate_retail_dataset.py` | Generate synthetic retail dataset for tests |
| `generate_test_data.py` | Generate transaction and email datasets with patterns |
| `migrate_to_temporal.py` | Migrate datasets to leakage-safe temporal format |

## notebooks/

Scripts for notebook validation, export, and project bootstrapping.

| Script | Description |
|--------|-------------|
| `export_tutorial_html.py` | Export notebooks to self-contained tutorial HTML |
| `init_project.py` | Bootstrap new customer retention projects |
| `test_notebooks.py` | Validate exploration notebooks with papermill |
