# Notebook Memory Management

## Problem

Exploration notebooks run in a single process where all cell variables share the global scope. Large DataFrames and models accumulate and cannot be reclaimed by `gc.collect()` while references exist.

On shared Databricks clusters, `session._jvm.System.gc()` fails with `JVM_ATTRIBUTE_NOT_SUPPORTED`.

## Mechanism

### `track_stage_object(*objects)`

Registers heavy objects in a module-level registry. `release_stage_memory()` unpersists and clears them.

- Native Spark DataFrame: `.unpersist()`
- pyspark.pandas DataFrame: `.spark.unpersist()`
- Anything else: cleared from registry, reclaimed by `gc.collect()`

### `release_stage_memory()`

1. Iterate tracked objects → unpersist each (tolerates `AttributeError`, `RuntimeError`, `OSError`)
2. Clear the registry
3. `gc.collect()`
4. `session.catalog.clearCache()`

No JVM access. Works on shared Databricks clusters.

## What to Track

| Track | Examples |
|-------|----------|
| Loaded DataFrames | `df = require_silver_merged(...)` |
| Materialized results | `merged = merger.merge(...)` |
| Models | `rf_model`, `lr_model` |
| Large arrays | `shap_values`, `X_train`, `X_test` |

## What NOT to Track

| Skip | Why |
|------|-----|
| Function-local variables | Cleaned when function returns |
| Config/findings objects | Small |
| Small summary DataFrames | `native_pd.DataFrame` with a few rows |

## Notebook Pattern

```python
df = require_silver_merged(_namespace)
track_stage_object(df)
```

Final cell:

```python
release_stage_memory()
del df
```

## Column Classification

`classify_columns(findings, exclude=set(TEMPORAL_METADATA_COLS))` computes column-type lists in a single pass, replacing repeated list comprehensions in NB04 and NB05:

```python
from customer_retention.analysis.auto_explorer.findings import classify_columns

cc = classify_columns(findings, exclude=set(TEMPORAL_METADATA_COLS))
cc.numeric       # [str]
cc.categorical   # [str] — includes cyclical
cc.datetime      # [str] — includes feature/label timestamps
cc.binary        # [str]
cc.text          # [str]
cc.identifier    # [str]
cc.target        # Optional[str]
```
