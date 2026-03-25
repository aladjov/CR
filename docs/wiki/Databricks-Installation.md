# Databricks Installation

## Prerequisites

- Databricks workspace with **Unity Catalog** enabled
- DBR **17.x ML runtime** (ships with MLflow 3.0+, SHAP, and PySpark pre-installed)
- A cluster with compute access

## Installation Options

### Option A: PyPI / Wheel (recommended for production)

Install ChurnKit as a cluster library or via `%pip install`. The package is importable directly.

**Cell 1** — install ChurnKit (one-time, re-runs are harmless):

```python
%pip install /Volumes/main/default/wheels/churnkit-0.99.6a4-py3-none-any.whl
```

**Cell 2** — initialize:

```python
from customer_retention.integrations.databricks_init import databricks_init

result = databricks_init(
    catalog="my_catalog",                                        # Unity Catalog catalog name
    schema="my_schema",                                          # Unity Catalog schema name
    workspace_path="Users/user@example.com/customer_retention",  # Workspace path for notebooks
    model_name="customer_retention",                             # Registered model name
    # experiment_name="my_experiment",                           # MLflow experiment (auto-detected if omitted)
    # copy_notebooks=True,                                      # Copy exploration notebooks to workspace
)
```

> **Note:** `%pip install` restarts the Python process. Place the install and `databricks_init()` in **separate cells**.

### Option B: Workspace Repo (development / latest code)

Clone the framework repo into Databricks Workspace Repos. This avoids building and uploading wheels — code changes are available immediately.

**Cell 1** — add the repo to Python path:

```python
import sys

FRAMEWORK_REPO_ROOT = "/Workspace/Repos/user@example.com/churnkit"
_src = f"{FRAMEWORK_REPO_ROOT}/src"
if _src not in sys.path:
    sys.path.insert(0, _src)
```

**Cell 2** — initialize with `framework_repo_path`:

```python
from customer_retention.integrations.databricks_init import databricks_init

result = databricks_init(
    catalog="my_catalog",                                                  # Unity Catalog catalog name
    schema="my_schema",                                                    # Unity Catalog schema name
    workspace_path="Users/user@example.com/customer_retention",            # Workspace path for notebooks
    model_name="customer_retention",                                       # Registered model name
    framework_repo_path="/Workspace/Repos/user@example.com/churnkit",      # Local repo path (enables sys.path injection)
    # experiment_name="my_experiment",                                     # MLflow experiment (auto-detected if omitted)
    # copy_notebooks=True,                                                 # Copy exploration notebooks to workspace
)
```

When `framework_repo_path` is set, `databricks_init` automatically:
- Injects a `code_system` cell at the top of each exploration notebook that adds the repo to `sys.path`
- Copies `requirements-databricks.txt` to the workspace for compute-scoped library installation
- Sets `CR_FRAMEWORK_REPO_PATH` environment variable for downstream use

**Dependencies:** attach `requirements-databricks.txt` (in the repo root) to your cluster as compute-scoped init libraries. This file excludes packages already pre-installed in the Databricks ML Runtime.

### Persistent Install (teams)

For shared clusters where every notebook should have ChurnKit available without `%pip install` cells:

1. Upload the ChurnKit wheel to a Unity Catalog Volume
2. Go to **Compute → Libraries → Install New → Upload**
3. Select the `.whl` file from the Volume

The library is installed on every cluster start. Notebooks only need the `databricks_init()` cell.

## SHAP

DBR 17.x ML runtime includes SHAP pre-installed. If you use a non-ML runtime, install the `[ml-shap]` extra:

```python
%pip install "/Volumes/main/default/wheels/churnkit-0.99.6a4-py3-none-any.whl[ml-shap]"
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | `"main"` | Unity Catalog catalog name |
| `schema` | `"default"` | Unity Catalog schema name |
| `experiment_name` | auto-detected | MLflow experiment name (defaults to notebook name) |
| `workspace_path` | `None` | Workspace path for notebooks and experiments |
| `copy_notebooks` | `True` | Whether to copy exploration notebooks to workspace |
| `model_name` | `"customer_retention"` | Registered model name in Unity Catalog |
| `framework_repo_path` | `None` | Path to cloned framework repo in Workspace Repos. When set, injects `sys.path` setup into notebooks |

After `databricks_init()`, these environment variables are available:

| Variable | Example Value | Description |
|----------|---------------|-------------|
| `CR_CATALOG` | `analytics` | Unity Catalog catalog |
| `CR_SCHEMA` | `churn` | Unity Catalog schema |
| `CR_WORKSPACE_PATH` | `Users/me/project` | Workspace base path |
| `CR_EXPERIMENT_NAME` | `churn_analysis` | MLflow experiment name |
| `CR_EXPERIMENTS_DIR` | `/Volumes/analytics/churn/experiments` | Experiments output directory (Unity Catalog Volume) |
| `CR_FRAMEWORK_REPO_PATH` | `/Workspace/Repos/me/churnkit` | Framework repo path (only when using repo clone) |

All subsequent ChurnKit calls (feature store, MLflow, data loading) automatically use these values.

## Exploration

After initialization, exploration notebooks are at `{workspace_path}/exploration_notebooks/`. Open `00_start_here.ipynb` from the Databricks workspace file browser and follow the guided flow.

The notebooks automatically detect the Databricks environment and use Unity Catalog tables, Delta Lake storage, and MLflow tracking.

## Feature Engineering

On Databricks, ChurnKit automatically uses the **Databricks Feature Engineering** client instead of Feast. The same ChurnKit API works in both environments.

```python
from customer_retention.integrations.adapters import get_feature_store

fs = get_feature_store()

result = fs.create_table(
    name="customer_features",
    schema={"customer_id": "int", "tenure": "float", "monthly_charges": "float"},
    primary_keys=["customer_id"],
)

fs.write_table("customer_features", features_df, mode="merge")

df = fs.read_table("customer_features")
```

Feature Engineering only supports `mode='merge'` — attempting `mode='overwrite'` raises a `ValueError`.

## MLflow Experiments

MLflow is automatically configured by `databricks_init()`. All experiments track to the Databricks-managed MLflow.

```python
from customer_retention.integrations.adapters import get_mlflow

mlflow_adapter = get_mlflow()
run_id = mlflow_adapter.start_run(experiment_name="churn_analysis")
mlflow_adapter.log_params({"learning_rate": 0.01})
mlflow_adapter.log_metrics({"accuracy": 0.95})
mlflow_adapter.end_run()
```

### Model Registration to Unity Catalog

Models are registered to Unity Catalog using the three-level namespace:

```python
model_uri = mlflow_adapter.log_model(
    model, artifact_path="model",
    registered_name=f"{catalog}.{schema}.churn_model",
)
```

### Alias-Based Model Promotion

Instead of the deprecated stage-based promotion, use aliases:

```python
mlflow_adapter.set_alias("analytics.churn.churn_model", "champion", "1")
model_version = mlflow_adapter.get_model_by_alias("analytics.churn.churn_model", "champion")
```

## Troubleshooting

### `RuntimeError: DATABRICKS_RUNTIME_VERSION not found`

`databricks_init()` must be called from a Databricks notebook. It validates the environment by checking for the `DATABRICKS_RUNTIME_VERSION` environment variable.

### `ImportError: PySpark required`

The Databricks adapters require PySpark. Ensure you are running on a Databricks cluster with a ML runtime, not a local Python environment.

### Feature table write fails with `ValueError`

`FeatureEngineeringClient.write_table` only supports `mode='merge'`. Change your call from `mode='overwrite'` to `mode='merge'`.

### MLflow experiment not found

Ensure `databricks_init()` was called before any MLflow operations. It calls `mlflow.set_experiment()` to create or select the experiment.

### Notebooks not copied

If `copy_notebooks=True` but no notebooks appear:
- Verify `workspace_path` is set and points to a writable location
- Check that ChurnKit was installed with exploration notebooks included in the package
- Notebooks are only copied if they don't already exist at the destination (idempotent)
