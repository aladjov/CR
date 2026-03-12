# Codebase Audit Report

**Date:** 2026-03-09
**Scope:** Full codebase audit against `docs/wiki/Architecture.md` and `docs/Coding_Practices.md`
**Files audited:** 316 source modules in `src/customer_retention/`, 20 exploration notebooks, CI configuration

---

## Executive Summary

| Category | Violations | Severity |
|----------|-----------|----------|
| Architecture deviations | 1 remaining (3 FIXED) | 1 LOW |
| Comments in code | ~500 instances across ~30 files | MEDIUM |
| pyspark.pandas compatibility | 22 violations across 10 files | 6 HIGH, 16 MEDIUM |
| Defensive code (error hiding) | 9 problematic + 1 bug | 1 CRITICAL, 5 HIGH, 3 MEDIUM |
| Code style | 12 patterns across 8 files | 4 HIGH, 8 MEDIUM |
| Notebook cell tags | 0 | COMPLIANT |
| Test coverage & CI | 0 critical | COMPLIANT |
| Naming conventions | 0 | COMPLIANT |
| Temporal split enforcement | 0 | COMPLIANT |
| Z-ORDER after writes | 0 | COMPLIANT |

---

## 1. Architecture Deviations

### 1.1 ~~HIGH~~ FIXED — Notebooks 06-07 now use require_silver_merged

**Rule violated:** Architecture.md §Notebook Data Sources — "Post-merge notebooks (04+) always load from `silver_merged`; they never fall back to per-dataset landing or bronze data."

**Fix applied:** Replaced conditional branching (`load_active_dataset` / `get_delta` / `load_silver_merged`) with `require_silver_merged(_namespace)` in both notebooks, matching NB04/NB05 pattern. NB06 also now uses `_namespace.merged_recommendations_path` instead of string replacement on `FINDINGS_PATH`.

**CI enforcement:** Added `TestNotebook06Architecture` and `TestNotebook07Architecture` test classes that assert `require_silver_merged` is present and `load_active_dataset(` is absent.

### 1.2 ~~MEDIUM~~ FIXED — Notebook 10 now uses namespace discovery

**Rule violated:** Coding_Practices.md §Run Namespace — "No `findings_dir` glob fallback — `load_notebook_findings` requires a RunNamespace"

**Fix applied:** Replaced the inline `load_findings_and_recommendations()` function (which used `findings_dir.glob("*_findings.yaml")` and `findings_dir.glob("*_recommendations.yaml")`) with namespace-aware discovery: `_namespace.discover_all_findings()`, `_namespace.merged_recommendations_path`, and `_namespace.multi_dataset_findings_path`. Also fails fast with `FileNotFoundError` if no namespace exists.

**CI enforcement:** Added `TestNotebook10Architecture` with `test_nb10_uses_namespace_discovery` and `test_nb10_no_findings_dir_glob`.

### 1.3 LOW — Parquet fallbacks exist in two modules

**Rule violated:** Architecture.md §Delta Lake — "Every medallion layer writes Delta Lake tables"

| File | Line | Issue |
|------|------|-------|
| `src/customer_retention/stages/profiling/time_window_aggregator.py` | 539 | `df.to_parquet()` fallback when `get_delta()` import fails |
| `src/customer_retention/stages/temporal/data_preparer.py` | 95 | `df.to_parquet()` fallback when `self.storage` is None |

**Mitigation:** Both try Delta first and only fall back on import failure. Acceptable as defensive code for environments without delta-rs, but contradicts the "fail fast" principle.

---

## 2. Comments in Code

**Rule violated:** Coding_Practices.md — "Do not use comments but instead descriptive class methods fields name."

### 2.1 High-priority files (20+ comments each)

| File | Count | Nature |
|------|-------|--------|
| `generators/pipeline_generator/renderer.py` | ~48 | Configuration explanations, implementation step narration |
| `generators/pipeline_generator/databricks_renderer.py` | ~30 (non-MAGIC) | Algorithmic explanations, step-by-step narration |
| `stages/profiling/temporal_features.py` | ~43 | Section headers (`# Tenure features`, `# Recency features`) |
| `analysis/visualization/chart_builder.py` | ~42 | Plotly construction narration (`# Confidence band`, `# Rolling mean`) |
| `generators/pipeline_generator/mlflow_pipeline_generator.py` | ~22 | Transformation explanations (`# Log transform`, `# Standard scaling`) |

### 2.2 Medium-priority files (10-20 comments each)

| File | Count | Nature |
|------|-------|--------|
| `generators/notebook_generator/stages/s10_batch_inference.py` | ~15 | Step-by-step narration |
| `generators/notebook_generator/stages/s11_feature_store.py` | ~15 | Configuration explanations |
| `stages/validation/data_validators.py` | ~19 | Validation logic narration |
| `stages/validation/timeseries_detector.py` | ~44 | Section headers and algorithmic explanations |
| `stages/profiling/feature_engineer.py` | ~13 | Generator initialization narration |

### 2.3 What should be refactored

- **Section headers** (`# Tenure features`) → extract into named methods (`_compute_tenure_features()`)
- **Configuration explanations** (`# MLflow tracking - using SQLite backend`) → use descriptive constant names
- **Step narration** (`# Skip if holdout already exists`) → extract predicate methods (`_holdout_exists()`)
- **Transformation labels** (`# Log transform for skewed columns`) → method names (`apply_log_transform_for_skew()`)

### 2.4 Acceptable exceptions

- `# MAGIC %md` / `# COMMAND ----------` in Databricks templates — platform directives, not comments
- `# type: ignore`, `# noqa`, `# fmt: off/on` — tooling directives
- `# @cr:` notebook cell tags — required by Coding_Practices.md

---

## 3. pyspark.pandas Compatibility Violations

**Rule violated:** Coding_Practices.md §pyspark.pandas Incompatible Patterns

### 3.1 `import pandas as pd` — bypasses compat layer (15 files)

| File | Line |
|------|------|
| `analysis/recommendations/base.py` | 5 |
| `analysis/recommendations/pipeline.py` | 3 |
| `analysis/auto_explorer/analysis_context.py` | 6 |
| `analysis/discovery/discovery_flow.py` | 3 |
| `analysis/discovery/type_inferencer.py` | 5 |
| `core/config/column_config.py` | 4 |
| `integrations/adapters/feature_store/base.py` | 6 |
| `integrations/adapters/feature_store/databricks.py` | 3 |
| `integrations/adapters/feature_store/feast_adapter.py` | 5 |
| `integrations/adapters/feature_store/local.py` | 5 |
| `integrations/adapters/storage/databricks.py` | 4 |
| `integrations/adapters/storage/local.py` | 3 |
| `generators/pipeline_generator/renderer.py` | 974, 2224 |
| `generators/notebook_generator/stages/s11_feature_store.py` | 215 |
| `generators/notebook_generator/stages/s09_monitoring.py` | 83 |

**Note:** `window_recommendation.py` is allowlisted. Files in `core/compat/` are the implementation layer itself.

### 3.2 `.agg([list])` on GroupBy — not supported in pyspark.pandas (2 instances)

| File | Line | Code |
|------|------|------|
| `generators/pipeline_generator/renderer.py` | 365 | `raw_df.groupby(ENTITY_COLUMN)[TIME_COLUMN].agg(["min", "max"])` |
| `generators/pipeline_generator/renderer.py` | 2035 | Same pattern |

**Fix:** Use `groupby_multi_agg()` from `core.compat`.

### 3.3 `pd.Timestamp()` — scalar not reimplemented in pyspark.pandas (4 instances)

| File | Line | Code |
|------|------|------|
| `generators/pipeline_generator/renderer.py` | 923 | `pd.Timestamp(df.attrs["aggregation_reference_date"])` |
| `generators/pipeline_generator/renderer.py` | 1150 | `pd.Timestamp("{{ config... }}")` in template |
| `generators/pipeline_generator/renderer.py` | 1155 | `pd.Timestamp.now()` in template |
| `generators/notebook_generator/stages/s05_feature_engineering.py` | 60 | `pd.Timestamp.now()` |

**Fix:** Use `datetime.datetime(...)` or `native_pd.Timestamp(...)`.

### 3.4 `pd.to_datetime()` on distributed data (6 instances)

| File | Line |
|------|------|
| `analysis/auto_explorer/prediction_objective_detector.py` | 167 |
| `analysis/diagnostics/leakage_detector.py` | 219, 240, 258, 339, 340 |

**Fix:** Use `safe_to_datetime()` from `core.compat`.

### 3.5 `.shift()` on timestamp column (1 instance)

| File | Line | Code |
|------|------|------|
| `stages/profiling/spark_temporal_feature_engineer.py` | 109 | `sorted_events.groupby(entity_col)[time_col].shift(1)` |

**Fix:** Use `timestamp_diffs_seconds()` from `core.compat`.

---

## 4. Defensive Code Patterns

**Rule violated:** Coding_Practices.md — "Do not write defensive code which hides real errors. Fail fast."

### 4.1 CRITICAL — Bug in realtime_scorer.py

| File | Line | Issue |
|------|------|-------|
| `integrations/streaming/realtime_scorer.py` | 130-131 | `except Exception: store_connected = True` — sets `True` on exception; should be `False` |

This is a logical bug, not just a style issue. On connection failure the code reports success.

### 4.2 HIGH — Broad `except Exception: pass` hiding real errors

| File | Line | Context |
|------|------|---------|
| `analysis/auto_explorer/skip_logic.py` | 106-107 | Loading ExplorationFindings — silently fails |
| `analysis/auto_explorer/skip_logic.py` | 127-128 | Detecting event-level granularity — silently fails |
| `analysis/auto_explorer/skip_logic.py` | 154-155 | Checking for text columns — silently fails |
| `analysis/auto_explorer/skip_logic.py` | 172-173 | Text column type detection — silently fails |
| `generators/notebook_generator/project_init.py` | 162-163 | Notebook discovery — silently fails |
| `stages/validation/timeseries_detector.py` | 366-367 | Datetime parsing — silently fails |
| `generators/pipeline_generator/findings_parser.py` | 1371-1372 | `except Exception: continue` — silently skips corrupted findings |
| `generators/pipeline_generator/databricks_renderer.py` | 1856-1857 | `except Exception: global_skip, global_reasons = set(), {}` — silent fallback |

### 4.3 HIGH — Silent fallback hiding Feast/data load failures

| File | Line | Issue |
|------|------|-------|
| `generators/pipeline_generator/renderer.py` | 1054-1056 | `except Exception as e: print(...), return _load_feast_data()` — swallows error |
| `generators/notebook_generator/stages/s10_batch_inference.py` | 477-479 | `except Exception as e: print(...)` + silent fallback |

### 4.4 MEDIUM — Empty defaults hiding upstream failures

| File | Line | Issue |
|------|------|-------|
| `core/compat/bulk_profiling.py` | 431-432 | Skew calculation fails → empty Series (no log) |
| `core/compat/bulk_profiling.py` | 436-437 | Kurtosis fails → empty Series (no log) |
| `core/compat/bulk_profiling.py` | 764-765 | Datetime stats fail → empty `DatetimeColumnStats()` (no log) |

### 4.5 Acceptable patterns (not violations)

- `core/compat/detection.py` — all `except Exception:` patterns are for optional import detection (pyspark, koalas, dbutils). These are expected conditions, not error hiding.
- `core/compat/__init__.py:36-45` — graceful degradation between pyspark.pandas → koalas → pandas.
- `integrations/streaming/realtime_scorer.py:184-194` — production scorer with explicit fallback strategy and error counting.

---

## 5. Code Style Violations

### 5.1 HIGH — Long functions mixing abstraction levels

| File | Line | Function | Lines | Issue |
|------|------|----------|-------|-------|
| `generators/pipeline_generator/findings_parser.py` | 977-1043 | `_build_landing_configs()` | ~70 | Mixes config building, conditional logic, path resolution |
| `analysis/auto_explorer/explorer.py` | 117-200+ | `_explore_all_columns()` | ~80+ | Type detection + bulk stats + typed stats in one method |
| `generators/pipeline_generator/findings_parser.py` | 1089-1130+ | `_build_lifecycle_config()` | ~50+ | Complex branching with mixed abstraction |
| `integrations/adapters/storage/databricks.py` | 67-90 | `write()` | ~25 | I/O intent mixed with Spark config, schema, metadata encoding |

### 5.2 MEDIUM — Code repetition

| Pattern | Files | Description |
|---------|-------|-------------|
| Column-existence guard | `generators/orchestration/data_materializer.py` (6 instances) | `if col not in df.columns: return df` repeated — should be a decorator or helper |
| Bulk stats accessor | `analysis/auto_explorer/explorer.py` (~20 instances) | `bulk.columns.get(col)` / `bulk.numeric.get(col)` pattern repeated |
| Percentage calculation | `stages/validation/data_validators.py` (~10 instances) | `(value / total * 100) if total > 0 else 0.0` repeated |
| Config builder structure | `generators/pipeline_generator/findings_parser.py` (5 methods) | `_build_*_config()` methods all follow identical validate→extract→construct pattern |

### 5.3 MEDIUM — Method arguments formatting

| File | Line | Issue |
|------|------|-------|
| `stages/profiling/text_processor.py` | 33-34, 46-47, 74-75 | 2-3 parameter functions split across multiple lines unnecessarily |
| `stages/profiling/temporal_feature_analyzer.py` | 147 | Same — few parameters spread across lines |

**Coding practice:** "do not spread method arguments on many lines unless they are more than 5"

---

## 6. Compliant Areas

### 6.1 Notebook Cell Tags — FULLY COMPLIANT

All 20 exploration notebooks implement the `# @cr:TYPE name='descriptive_name' id=HEXUUID` format correctly:
- All three types (`config`, `code`, `user_code`) used appropriately
- All IDs are valid 8-character hex strings
- No duplicate IDs detected
- No missing tags on code cells
- Consistent `snake_case` naming

### 6.2 Test Coverage & CI — COMPLIANT

- **CI enforces:** 75% minimum (`--cov-fail-under=75`), targets 90%+
- **Test organization:** 332 test files mirror 265 source modules
- **Edge case coverage:** Boundary values, property testing, state transitions
- **Anti-pattern guard:** `test_spark_pandas_guard.py` enforces 13 pyspark.pandas banned patterns in CI
- **Test naming:** Consistently uses `test_<behavior>_<condition>` pattern
- **60 untested modules:** Intentional — thin wrappers, optional streaming features, CLI orchestration tested via integration tests

### 6.3 Ruff Configuration — WELL-TUNED

- Rules enforced: E, F, I, N, W
- 7 global ignores (all justified: ML naming conventions, conditional imports)
- 7 per-file ignore patterns (all narrow and scoped)
- Zero overly broad ignores

### 6.4 Composite Naming Convention — COMPLIANT

`core/naming.py` correctly implements:
- `{readable_prefix}__{7char_hash}` format
- First 4 characters per word, lowercase, joined with `_`
- SHA256-based 7-character hash from sorted source names

### 6.5 Temporal Split Enforcement — COMPLIANT

- `DataSplitter` defaults to `SplitStrategy.TEMPORAL`
- Notebook 08 uses `strategy=SplitStrategy.TEMPORAL`
- Generated templates use temporal splits exclusively
- No row-random splits in main training paths

### 6.6 Z-ORDER After Writes — COMPLIANT

All Delta writes in generated templates (both local and Databricks) include `optimize()` calls with column guards:
- Landing: `(entity_col, time_col)`
- Bronze: `(entity_col, as_of_date)`
- Silver: `(entity_id, as_of_date)`
- Gold: `(entity_id, event_timestamp)`

### 6.7 Run Namespace — MOSTLY COMPLIANT

- `RunNamespace.from_env_or_latest()` is the standard discovery chain
- Sentinel file mechanism works correctly
- SessionState tracks active dataset and last notebook
- ~~One deviation: Notebook 10's inline glob helper~~ — FIXED (§1.2)

---

## 7. Remediation Priority

### P0 — Must fix (bugs, architectural violations)

1. **`realtime_scorer.py:130`** — Fix `store_connected = True` → `False` on exception
2. ~~**Notebooks 06-07**~~ — FIXED: Now use `require_silver_merged()` with CI enforcement tests

### P1 — Should fix (fail-fast violations, compatibility)

3. **`skip_logic.py`** (4 locations) — Replace `except Exception: pass` with specific exception types or add logging
4. **`findings_parser.py:1371`** — Replace `except Exception: continue` with specific exception + logging
5. **`databricks_renderer.py:1856`** — Replace `except Exception:` with specific exception + logging
6. **`leakage_detector.py`** (5 locations) — Replace `pd.to_datetime()` with `safe_to_datetime()`
7. **`renderer.py:365, 2035`** — Replace `.agg(["min", "max"])` with `groupby_multi_agg()`
8. **`renderer.py:923, 1150, 1155`** — Replace `pd.Timestamp()` with `datetime.datetime()` or `native_pd.Timestamp()`
9. **`renderer.py:1054`** — Remove silent Feast fallback; fail fast if data missing

### P2 — Should improve (code quality)

10. **15 files** — Replace `import pandas as pd` with compat layer imports
11. **`renderer.py`, `temporal_features.py`, `chart_builder.py`** — Remove explanatory comments; refactor into descriptively-named methods
12. **`findings_parser.py`** — Decompose `_build_landing_configs()` and `_build_lifecycle_config()` into smaller functions
13. **`explorer.py`** — Decompose `_explore_all_columns()` by extracting type detection, bulk stats, typed stats into separate methods
14. ~~**Notebook 10**~~ — FIXED: Now uses `_namespace.discover_all_findings()` with CI enforcement tests

### P3 — Nice to have (minor style)

15. **`data_materializer.py`** — Extract column-existence guard into decorator/helper
16. **`data_validators.py`** — Extract percentage calculation into utility
17. **`text_processor.py`** — Compact multi-line parameter declarations for <5 params
18. **`bulk_profiling.py`** (3 locations) — Add logging when skew/kurtosis/datetime stats fail silently
