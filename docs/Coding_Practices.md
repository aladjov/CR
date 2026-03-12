Before introducing any changes to the codebase look in the existing code and look for the best place to introduce changes / enhancements to avoid duplication and overlapping scope.  Make sure that you ask questions instead of silently making architectural decisions.

Follow test driven development practices with at least 90% coverage with meaningfull tests that cover edge cases and antipatterns. Create tests first using the specification then do the implementation.After completing all tasks do once again check against the specification. Make sure that all tests pass. Any bug we find should be covered by dedicated unit or integration test. What other similar diconnects / edge cases are possible - can we try to come up with more tests to cover them.

When running in Databricks the data must stay distributed as pyspark.pandas throughout. Pyspark operations should be performed in bulk and not inflate unnecessary the number of Spark jobs. Use Spark optimized functions in the compatibility layer if such are available.

Do not write defencive code which hides the reall errors and making maintainance and root cause investigation harder. Fail fast so we can quicly address the issue at its root not downstream when it

When working on changes in exporation steps 00-08 contributing new features or operations over data layers - these need to be coordinated with both local and databricks generated pipelines in such a way that if we have the same data set assesed from models produced based on production data vs exploration it should produce identical feature values and model predictions. Any deviation from full parity exploration local databricks is considered major architectural issue - do not allow that!

When you write a new code make sure that you are aware of the context this code will run - is it a common high level code that should only use public methods from compatibility layer (delegating the environment specific implementations to that layer) or you are already in a guarded Databricks vs Local and should use only relevant functions for this environment


Do run ruff tests and make sure that you do fix errors that are not ignored in our ci.yaml

Do not use comments but instead descriptive class methods fields name. Favor short single responsibility functions with no side effects which operate on the same abstract level. Start from public methods high level functions first then functions that are implementation details. Design the structure of the code in such a way to avoid code repetition and allow inheritance to take care of variations. Favour compact code with no ceremonial assignments (for variables used just once). Try to make everything testable and easy to read a code formatting request do not spread method arguments on many lines unless they are more than 5. In general prefer compact code that does not spread too many lines

## Run Namespace — Single Source of Truth

All run metadata, progress, and paths derive from `RunNamespace(root, run_id)`. No alternative mechanisms, fallbacks, or legacy paths are permitted.

**Discovery chain** (one mechanism, used everywhere):
`RunNamespace.from_env_or_latest()` → env vars → sentinel file → latest run directory

**What RunNamespace provides** — given `(experiments_dir, run_id)`:
- Dataset findings: `namespace.dataset_findings_dir(name)`
- Merged findings: `namespace.multi_dataset_findings_path` / `merged_recommendations_path`
- Session state: `namespace.user_session_path(username)` → `SessionState` (active dataset, last notebook)
- Grid/config: `namespace.grid_path`, `namespace.session_dir`

**Databricks for_each_task contract**: Only `(experiments_dir, run_id, dataset_id)` passed via DAB base_parameters. Everything else derived from `RunNamespace`.

**Prohibited patterns** (CI-enforced):
- No `findings_dir` glob fallback — `load_notebook_findings` requires a RunNamespace
- No `notebook_progress.json` — progress tracked via `SessionState` only
- No `get_notebook_experiments_dir()` — use `get_experiments_dir()` only
- No `except Exception: pass/return None` in core metadata files — narrow to specific types
- No import-time side effects — `accept_workflow_params()` called explicitly in each notebook

## Notebook Cell Tags

Every code cell in the exploration notebooks must have a `# @cr:` tag on line 1 with the format:

```
# @cr:TYPE name='descriptive_name' id=HEXUUID
```

- **TYPE**: `config` for ALL_CAPS user settings, `user_code` for user-written logic, `code` for framework cells
- **name**: short snake_case label describing what the cell does (e.g. `load_findings`, `detect_target`)
- **id**: unique 8-character hex string — generate with `python -c "import uuid; print(uuid.uuid4().hex[:8])"`

IDs are permanent. Once assigned they never change even if the cell moves or its content is rewritten. The sync engine matches cells by embedded `id`, not by position. CI enforces that every code cell has a valid tag, that embedded IDs match `cell.id` in the notebook JSON, and that no duplicates exist. See [Notebook-Sync](wiki/Notebook-Sync.md) for full details.

## pyspark.pandas Incompatible Patterns

When running on Databricks, data stays distributed as `pyspark.pandas` DataFrames. Several pandas APIs silently fail or raise at runtime. The table below lists banned patterns and their fixes. A static guard test (`tests/core/compat/test_spark_pandas_guard.py`) enforces these in CI.

| Banned Pattern | Why It Fails | Fix |
|---|---|---|
| `import pandas as pd` | Bypasses compat layer — code runs on native pandas locally but breaks on Databricks | Import `pd` (and helpers) from `customer_retention.core.compat` |
| `.dt.to_period(freq)` | Not implemented in pyspark.pandas | `period_start_time(series, freq)` from `core.compat` (uses `F.date_trunc` on Spark) |
| `.values` (on DataFrame/Series) | May raise on pyspark.pandas | `.to_numpy()` |
| `.iloc[-N]` (negative indexing) | Fails on pyspark.pandas | `.min()` / `.max()`, or collect to numpy first |
| `.agg([list])` on GroupBy | Not supported in pyspark.pandas | `groupby_multi_agg(df, group_col, agg_col, funcs)` from `core.compat` |
| `.shift()` on timestamp column | `isnan` called internally on `lag()` result — fails on TIMESTAMP_NTZ | `timestamp_diffs_seconds(series)` from `core.compat` |
| Timestamp subtraction `.dt.days` | Returns integer seconds in pyspark.pandas, not timedelta | `timedelta_to_days(series)` / `timedelta_to_seconds(series)` from `core.compat` |
| `pd.DataFrame(rows)` for small results | If `pd` is pyspark.pandas, creates unnecessary distributed overhead | `native_pd.DataFrame(rows)` for small, local-only DataFrames |
| `.describe()` on mixed-type DataFrame | Spark tries `avg()` on TIMESTAMP_NTZ columns — `DATATYPE_MISMATCH` | `safe_describe(df)` from `core.compat` |
| `.sample(n=)` or `.sample(count)` | pyspark.pandas does not support specifying exact number of items | `safe_sample(df, n)` from `core.compat` |
| `np.where(series, ...)` | `np.where` calls `__iter__()` on pyspark.pandas Series | `series.where(cond, other)` or `F.when().otherwise()` |
| `np.select(conditions, ...)` | `np.select` calls `__iter__()` on pyspark.pandas Series via `broadcast_arrays` | `safe_select(conditions, choices, default)` from `core.compat` |
| `df.loc[other.index, col]` | `Index.__iter__()` not implemented in pyspark.pandas | `concat([...], axis=1)` to combine columns, or `df[col]` when indices are aligned |
| `for x in series.unique()` / `set(series.unique())` / `list(series.unique())` / `sorted(series.unique())` | `Series.__iter__()` not implemented in pyspark.pandas; `set()`, `list()`, `sorted()`, `for` all trigger it | `safe_to_list(series.unique())` from `core.compat`, then iterate/sort/convert the list |
| `for v in series[:N]` (iterating over Series slice) | Same `__iter__()` issue on sliced pyspark.pandas Series | `safe_to_list(series.head(N))` from `core.compat` |
| `pyspark_pandas_df.merge(spark_df)` | pyspark.pandas tries `spark_df._internal` which doesn't exist on raw Spark DataFrames | Convert raw Spark DataFrames via `.pandas_api()` before merge, or use `.to_spark()` + Spark `.join()` |
| `spark_df._internal` | Internal pyspark.pandas attribute, not present on raw Spark DataFrames | Ensure consistent DataFrame types — don't mix pyspark.pandas and raw Spark DataFrames in merge/join operations |
| Per-column `df.agg(F.mean(col)).collect()` in generated Databricks code | Triggers one Spark job per column — 90+ jobs for typical pipelines | Batch all stats into ONE `.agg(*exprs).collect()` call, then apply `.withColumn()` transformations |
| `index[:n]` (slicing pyspark.pandas Index) | pyspark.pandas `Index` is not subscriptable | `head_as_list(index, n)` from `core.compat` — detects Index via `.to_series()` fallback |
| `pd.Timestamp(...)` / `pd.Timestamp.now()` / `isinstance(v, pd.Timestamp)` | `ps.Timestamp` scalar is not reimplemented — raises `PandasNotImplementedError` even when used only as a type in `isinstance()` checks. Also applies to Databricks-targeted templates and generated code that runs with pyspark.pandas | `datetime.datetime` (parent class of `pandas.Timestamp`) for type checks, `native_pd.Timestamp` for constructors. In Databricks renderer templates use `datetime.datetime(...)` — `pd.Timestamp` is only safe in local-only generated code |
| `_pandas.api.types.is_datetime64_any_dtype(series)` before `_is_spark_pandas(series)` | `is_datetime64_any_dtype` returns `True` for pyspark.pandas timestamp columns, then `_pandas.Series(series)` wrapping triggers `__iter__()`. The dtype check short-circuits the spark branch | Always check `_is_spark_pandas(series)` first in compat helpers — handle distributed data before falling through to native pandas paths |
| `_pandas.Series(pyspark_pandas_series)` | Wrapping a pyspark.pandas Series in native `pandas.Series()` triggers `list(data)` → `__iter__()` — pyspark.pandas does not implement `__iter__()` | Never wrap pyspark.pandas objects in native pandas constructors. Use `.to_pandas()` for explicit collection, or return the distributed object as-is |
| `df[col].isin(large_collection)` | `.isin()` with >100K values generates a SQL `IN(...)` clause that exceeds Snowflake's 200K expression limit and may cause similar issues on other Spark SQL backends | `safe_isin(df, col, values)` from `core.compat` — uses Spark semi/anti join for large lists, falls back to `.isin()` for small lists. Use `negate=True` for exclusion |
| `native_pd.to_datetime(df[col])` | `native_pd.to_datetime` is native pandas — internally calls `__iter__()` on pyspark.pandas Series via `_maybe_cache` → `should_cache` → `set(islice(arg))` | `safe_to_datetime(df[col])` from `core.compat` — handles both backends. OK for scalars only (e.g. `native_pd.to_datetime(value)` inside a loop) |
| `df.query("col in ['a', 'b']")` | pyspark.pandas `.query()` translates to Spark SQL which doesn't support Python `[...]` list syntax — raises `PARSE_SYNTAX_ERROR` at `[` | `safe_query(df, expr)` from `core.compat` — converts `[...]` to `(...)` for Spark SQL compatibility |
| `int(row[f"__agg__{col}"])` on Spark `.agg().collect()[0]` results | `F.coalesce(F.sum(...), F.lit(0))` can still return SQL NULL on certain Databricks/Spark configurations — bare `int()` raises `TypeError: int() argument must be ... not 'NoneType'` | `_safe_int(row[...])` from `core.compat.bulk_profiling` for all integer aggregate results from Spark. Already handles None → 0 |
| `pd.api.types.is_numeric_dtype(series)` | `pyspark.pandas` has no `api` submodule — raises `AttributeError: module 'pyspark.pandas' has no attribute 'api'` | `is_numeric_dtype(series)` from `core.compat` — delegates to native `pandas.api.types` internally |
| `pd.qcut(series, q=N)` | `pyspark.pandas` does not implement `qcut` — raises `AttributeError` | `qcut(series, q=N, **kwargs)` from `core.compat` — collects to pandas first on Spark, passes through on native pandas |
| Public export of single-environment function from `core.compat` | Callers assume compat exports work on both backends — a pandas-only public function silently breaks on Databricks | Keep single-environment implementations private (prefix `_`). Only export dispatcher functions that handle both pandas and pyspark.pandas (e.g. `normalize_timestamps` dispatches to `_normalize_timestamp_columns` or `_normalize_timestamps_distributed`) |
| Blindly mapping pandas `object` dtype → `StringType()` in Spark schema | Aggregation columns (e.g. `*_hour_mean_90d`) often have `object` dtype containing `numpy.float64` values (NaN mixing forces object). Arrow serialization fails: `ArrowTypeError: Expected bytes, got a 'numpy.float64' object` | `pandas_dtype_to_spark_schema` uses `_infer_object_column_spark_type()` which calls `_pandas.api.types.infer_dtype(series, skipna=True)` to inspect actual values — maps `floating`/`mixed-integer-float` → `DoubleType`, `integer` → `LongType`, `boolean` → `BooleanType`, `datetime` → `TimestampNTZType`, everything else → `StringType` |

### Allowlisted files

`window_recommendation.py` — operates on tiny summary DataFrames only; bare pandas is intentional.
