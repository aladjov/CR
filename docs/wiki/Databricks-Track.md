# Databricks Track: Unity Catalog + Delta Lake

For production Spark environments, export standalone PySpark notebooks that run on Databricks with **no framework dependency**.

## Overview

```
Exploration Artifacts
         │
         │ • *_findings.yaml
         │ • recommendations
         │
         ▼
┌───────────────────────┐
│   DatabricksExporter  │
│                       │
│  • Standalone PySpark │
│  • No dependencies    │
│  • Unity Catalog      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Generated Notebooks │
│                       │
│  bronze_customers.py  │
│  bronze_orders.py     │
│  silver_merge.py      │
│  gold_features.py     │
│  workflow.json        │
└───────────────────────┘
```

## Generate Standalone Notebooks

```python
from customer_retention.generators.orchestration import DatabricksExporter
from customer_retention.analysis.auto_explorer import MultiDatasetFindings

# Load your exploration artifacts
multi = MultiDatasetFindings.load("./experiments/findings/multi_dataset_findings.yaml")
registry = multi.to_recommendation_registry()

# Configure exporter for Unity Catalog
exporter = DatabricksExporter(
    registry,
    catalog="your_catalog",    # Unity Catalog
    schema="churn_pipeline"
)

# Get complete notebook structure
structure = exporter.export_notebook_structure()

# Save notebooks to files
for source_name, code in structure["bronze"].items():
    with open(f"notebooks/bronze_{source_name}.py", "w") as f:
        f.write(code)

with open("notebooks/silver_merge.py", "w") as f:
    f.write(structure["silver"])

with open("notebooks/gold_features.py", "w") as f:
    f.write(structure["gold"])
```

## Generated Code Characteristics

**Generated code is standalone** - uses only:
- `pyspark.sql.functions` (F.col, F.when, F.mean, etc.)
- `pyspark.sql.window` (Window.partitionBy)
- `pyspark.ml.feature` (StringIndexer, OneHotEncoder, StandardScaler)
- Delta Lake writes (`df.write.format("delta").saveAsTable()`)

**No framework dependency** - the notebooks run standalone on any Databricks cluster.

### Example Generated Bronze Notebook

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: customers
# MAGIC Generated from exploration findings

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------
# Configuration
CATALOG = "your_catalog"
SCHEMA = "churn_pipeline"
SOURCE_PATH = "/Volumes/landing/customers.csv"

# COMMAND ----------
# Read source data
df = spark.read.format("csv").option("header", True).load(SOURCE_PATH)

# COMMAND ----------
# Null handling: age (median imputation)
median_age = df.approxQuantile("age", [0.5], 0.01)[0]
df = df.withColumn("age", F.coalesce(F.col("age"), F.lit(median_age)))

# COMMAND ----------
# Outlier capping: monthly_charges (IQR method)
q1, q3 = df.approxQuantile("monthly_charges", [0.25, 0.75], 0.01)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df = df.withColumn("monthly_charges",
    F.when(F.col("monthly_charges") < lower, lower)
     .when(F.col("monthly_charges") > upper, upper)
     .otherwise(F.col("monthly_charges")))

# COMMAND ----------
# Write to Delta table
df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_customers")
```

## Generate Pipeline Specification

For use with Databricks Assistant or other AI tools, generate a markdown specification:

```python
from customer_retention.generators.orchestration import PipelineDocGenerator

doc_gen = PipelineDocGenerator(registry, findings)
spec = doc_gen.generate()

# Copy to Databricks Assistant or save as reference
with open("pipeline_spec.md", "w") as f:
    f.write(spec)
```

The spec includes implementation hints, column statistics, and execution order.

## Databricks Workflow Definition

Import the generated `workflow.json` into Databricks Workflows:

```python
import json
with open("orchestration/churn_prediction/workflow.json") as f:
    workflow = json.load(f)

# Use Databricks CLI or API to create the workflow
# databricks jobs create --json @workflow.json
```

**Generated workflow structure:**
```json
{
  "name": "churn_prediction_pipeline",
  "tasks": [
    {"task_key": "bronze_customers", "notebook_task": {...}},
    {"task_key": "bronze_orders", "notebook_task": {...}},
    {"task_key": "silver_merge", "depends_on": ["bronze_customers", "bronze_orders"], ...},
    {"task_key": "gold_features", "depends_on": ["silver_merge"], ...},
    {"task_key": "ml_experiment", "depends_on": ["gold_features"], ...}
  ]
}
```

## Unity Catalog Feature Engineering

On Databricks, feature store integration uses Databricks Feature Engineering:

```python
from customer_retention.integrations.adapters.feature_store import get_feature_store

# Automatically uses DatabricksFeatureStoreAdapter when on Databricks
feature_store = get_feature_store(catalog="main", schema="features")

# Same API works - but uses Unity Catalog under the hood
feature_store.register_feature_view(config, spark_df)
# Creates table: main.features.customer_features
```

## Deployment Checklist

### Option 1: Use Standalone Notebooks (Recommended)

No installation needed - just import the generated notebooks:

```bash
# Upload notebooks to Databricks workspace
databricks workspace import_dir ./notebooks /Workspace/pipelines/churn_prediction
```

### Option 2: Install Framework as Wheel

```bash
# 1. Build wheel
uv build

# 2. Upload to Unity Catalog Volume
databricks fs cp dist/customer_retention-*.whl \
    dbfs:/Volumes/catalog/schema/wheels/

# 3. Install on cluster (in notebook)
%pip install /Volumes/catalog/schema/wheels/customer_retention-*.whl
```

### Cluster Init Script

For clusters that need the framework pre-installed:

```bash
#!/bin/bash
# scripts/databricks/dbr_init.sh

pip install /dbfs/Volumes/catalog/schema/wheels/customer_retention-*.whl
```

## Best Practices

| Practice | Description |
|----------|-------------|
| **Use standalone notebooks** | Generated PySpark code has no framework dependency |
| **Unity Catalog tables** | Write to Delta tables for data governance |
| **Workflow orchestration** | Use the generated `workflow.json` for job scheduling |
| **Environment detection** | The framework auto-detects Databricks vs local |
| **Test locally first** | Validate with local track before Databricks deployment |
| **Volume storage** | Store wheels in Unity Catalog Volumes for versioning |

## Environment Detection

The framework automatically detects the execution environment:

```python
from customer_retention.core.compat import is_databricks, is_spark_available

if is_databricks():
    # Running on Databricks - use Unity Catalog
    feature_store = get_feature_store(catalog="main", schema="features")
else:
    # Running locally - use Feast
    feature_store = get_feature_store(repo_path="./feature_repo")
```

## Next Steps

- [[Local Track]] - Test locally before deploying
- [[Feature Store]] - Feature management on Databricks
- [[Temporal Framework]] - Leakage-safe data preparation
