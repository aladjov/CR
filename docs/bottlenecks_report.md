# Spark Job Bottleneck Analysis: Notebooks 06, 07, 08

**Date:** 2026-02-22
**Context:** 14K+ accumulated Spark jobs observed by section 6.3 on Databricks

---

## Executive Summary

After tracing every Spark-triggering operation in notebooks 06, 07, and 08 and their
underlying profiling modules, the key findings are:

| Notebook | Current Jobs (N=50 numeric, M=20 cat) | Biggest Bottleneck |
|----------|--------------------------------------|--------------------|
| **06** Feature Opportunities | ~30 | `batched_corr_matrix` called twice |
| **07** Modeling Readiness | **~370** | `check_separation`: 4 jobs/column, `check_single_feature_auc`: 2 jobs/column |
| **08** Baseline Experiments | ~55 | Per-column `.median()` for fillna |

**Notebook 07 is the worst offender**, generating ~7N Spark jobs for N numeric features.
With 200 numeric features (common after temporal feature engineering), that's ~1,400 jobs
from NB07 alone.

The 14K total by section 6.3 primarily accumulates from **notebooks 01-05** (DataExplorer,
temporal profiling, relationship analysis). Notebooks 06-08 together add ~455 jobs for a
50-column dataset, scaling to ~2,000+ for 200 columns.

---

## Notebook 06: Feature Opportunities

### Section 6.3 — Feature Capacity Analysis

**Call:** `FeatureCapacityAnalyzer().analyze(df, feature_cols, target_col)`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `target.value_counts()` | feature_capacity.py:155 | 1 | No |
| `batched_corr_matrix()` — stddev batch | compat/__init__.py:446 | 1 | No (all cols) |
| `batched_corr_matrix()` — pair batches | compat/__init__.py:456-460 | ceil(P/500) | No (batched) |
| Eigenvalue decomposition | feature_capacity.py:260 | 0 | NumPy only |

**Formula:** `2 + ceil(N*(N-1)/2 / 500)` where N = numeric features

| N features | Pairs | Corr batch jobs | Total section jobs |
|-----------|-------|-----------------|-------------------|
| 20 | 190 | 1 | 4 |
| 50 | 1,225 | 3 | 6 |
| 100 | 4,950 | 10 | 13 |
| 200 | 19,900 | 40 | 43 |

### Section 6.3.2 — Segment-Specific Capacity

**Call:** `capacity_analyzer.analyze_segment_capacity(df, feature_cols, target_col, segment_col)`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `batched_corr_matrix()` (recomputed!) | feature_capacity.py:286→219 | same as 6.3 | No |
| `df.groupby(segment_col).size()` | feature_capacity.py:320 | 1 | No |
| `df.groupby(segment_col)[target].count()` | feature_capacity.py:321 | 1 | No |
| `df.groupby([segment, target]).size()` | feature_capacity.py:322 | 1 | No |
| `class_counts.groupby(segment)["n"].min()` | feature_capacity.py:323 | 1 | No |
| `safe_to_list(sizes.index)` | feature_capacity.py:326 | 1 | No |

**Inefficiency:** `batched_corr_matrix` is called twice — once in `analyze()` and once in
`analyze_segment_capacity()`. The correlation matrix is independent of segments, so the
second call is redundant.

**Optimization:** Cache the `EffectiveFeaturesResult` from `analyze()` and pass it to
`analyze_segment_capacity()`, or have `analyze_segment_capacity` accept a pre-computed
correlation result. **Saves:** `2 + ceil(P/500)` jobs.

### Section 6.5 — Business-Driven Features

**Calls:** `create_tenure_features()`, `create_recency_features()`, `create_engagement_score()`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `df[created_col].max()` | customer_segmentation.py:412 | 1 | No |
| `df[activity_col].max()` | customer_segmentation.py:466 | 1 | No |
| `open_rate.max()` | customer_segmentation.py:369 | 1 | No |
| `click_rate.max()` | customer_segmentation.py:371 | 1 | No |
| `timedelta_to_days()` (×2) | customer_segmentation.py:419,470 | 0 | Lazy transform |

**Total: 4 jobs** (all single aggregations, no per-column loops)

**Optimization:** Batch all 4 `.max()` calls into a single `df.agg(F.max(c1), F.max(c2), F.max(c3), F.max(c4))`. **Saves:** 3 jobs.

### Section 6.6 — Customer Segmentation

**Calls:** `segment_by_value_frequency()`, `segment_by_recency()`, `segment_by_engagement()`, `groupby().mean()`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `df[value_col].median()` | customer_segmentation.py:115 | 1 | No |
| `df[freq_col].median()` | customer_segmentation.py:117 | 1 | No |
| `value_counts().to_dict()` (×3) | customer_segmentation.py:126,221,291 | 3 | No |
| `groupby(segment)[target].mean()` | notebook cell | 1 | No |

**Total: 6 jobs**

**Optimization:** Batch the 2 `.median()` calls into one `select(F.percentile_approx(...))`. Batch the 3 `value_counts()` calls is harder since they're on different columns created sequentially. **Saves:** 1 job.

### Section 6.7 — Numeric Transformation Opportunities

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `df[numeric_cols].skew()` | notebook cell | 1 | No (single call) |

**Total: 1 job**

### Section 6.8 — Categorical Encoding Opportunities

No Spark jobs (metadata lookup only from pre-computed findings).

### Notebook 06 Total

| Section | Jobs (N=50) | Jobs (N=200) |
|---------|------------|-------------|
| 6.3 Feature Capacity | 6 | 43 |
| 6.3.2 Segment Capacity | 11 | 48 |
| 6.5 Business Features | 4 | 4 |
| 6.6 Segmentation | 6 | 6 |
| 6.7 Transformations | 1 | 1 |
| 6.8 Categorical Encoding | 0 | 0 |
| **Total** | **28** | **102** |

**With optimization (cache corr matrix + batch maxes):** 22 / 59 jobs.

---

## Notebook 07: Modeling Readiness

### Section 7.3 — Class Imbalance Analysis

**Call:** `ImbalanceRecommender().recommend(target_series, n_samples=len(df))`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `y.value_counts().to_dict()` | imbalance_handler.py:198 | 1 | No |

**Total: 1 job**

### Section 7.4 — Data Leakage Risk (Per-Column Correlation Loop)

**Call:** `df[[col_name, target]].corr()` in a loop over all feature columns

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `.corr()` on 2-col DataFrame | notebook cell (loop) | **N** | **YES — 1 per column** |

**Total: N jobs** where N = number of feature columns in findings

This is a **redundant bottleneck** — the subsequent `LeakageDetector.run_all_checks()`
already computes correlations in batched mode.

**Optimization:** Replace per-column `.corr()` loop with `bulk_corr_with_target()`.
**Saves:** `N - ceil(N/500)` jobs. For N=200: saves ~199 jobs.

### Section 7.X — Final Leakage Validation

**Call:** `LeakageDetector().run_all_checks(X, y, include_pit=False)`

This calls 6 sub-checks. Here's the breakdown for each:

#### 7.X.1 `check_correlations(X, y)`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `bulk_corr_with_target()` | leakage_detector.py:79 | ceil(N/500) | No (batched) |

**Total:** 1 job for N < 500

#### 7.X.2 `check_separation(X, y)` — BIGGEST BOTTLENECK

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `feature[target == 0].dropna()` | leakage.py:135 | 0 | Lazy |
| `feature[target == 1].dropna()` | leakage.py:135 | 0 | Lazy |
| `class_0.min()` | leakage.py:138 | **1** | **YES** |
| `class_0.max()` | leakage.py:138 | **1** | **YES** |
| `class_1.min()` | leakage.py:139 | **1** | **YES** |
| `class_1.max()` | leakage.py:139 | **1** | **YES** |

**Per column: 4 Spark jobs** (min/max on each class-filtered series)
**Total: 4N jobs**

| N features | Jobs from check_separation |
|-----------|---------------------------|
| 20 | 80 |
| 50 | 200 |
| 100 | 400 |
| 200 | 800 |

**Optimization — Batched version:**
```python
# Instead of per-column min/max on filtered series:
# Single Spark job for ALL columns:
exprs = []
for col in numeric_cols:
    exprs.extend([
        F.min(F.when(F.col(target) == 0, F.col(col))).alias(f"{col}__min0"),
        F.max(F.when(F.col(target) == 0, F.col(col))).alias(f"{col}__max0"),
        F.min(F.when(F.col(target) == 1, F.col(col))).alias(f"{col}__min1"),
        F.max(F.when(F.col(target) == 1, F.col(col))).alias(f"{col}__max1"),
    ])
row = spark_df.agg(*exprs).collect()[0]
```
**After optimization: 1 Spark job (batch of 500 columns) instead of 4N.**
**Saves for N=200: 799 jobs.**

#### 7.X.3 `check_temporal_logic(X, y)`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `bulk_corr_with_target()` on temporal cols only | leakage_detector.py:137 | ceil(T/500) | No |

**Total:** 1 job (T = temporal columns, usually < 20)

#### 7.X.4 `check_single_feature_auc(X, y)` — SECOND BIGGEST BOTTLENECK

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `feature.values.reshape(-1, 1)` | leakage_detector.py:167 | **1** | **YES** (collects to numpy) |
| `y.values[mask]` | leakage_detector.py:169 | **1** | **YES** (collects to numpy) |
| `cross_val_predict(model, ...)` | leakage_detector.py:174 | 0 | sklearn/CPU only |
| `roc_auc_score(...)` | leakage_detector.py:175 | 0 | numpy only |

**Per column: 2 Spark jobs** (collecting X and y to numpy)
**Total: 2N jobs**

| N features | Jobs from check_single_feature_auc |
|-----------|-----------------------------------|
| 20 | 40 |
| 50 | 100 |
| 100 | 200 |
| 200 | 400 |

**Optimization — Batched version:**
```python
# Collect ALL numeric columns to numpy in ONE operation:
X_numpy = df[numeric_cols].to_numpy()   # 1 Spark job
y_numpy = y.to_numpy()                  # 1 Spark job (or reuse)
# Then loop over columns in pure numpy — zero Spark jobs
for i, col in enumerate(numeric_cols):
    X_single = X_numpy[:, i].reshape(-1, 1)
    # ... sklearn cross_val_predict ...
```
**After optimization: 2 Spark jobs instead of 2N.**
**Saves for N=200: 398 jobs.**

#### 7.X.5 `check_target_in_features(X, y)`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `bulk_corr_with_target()` on candidates | leakage_detector.py:305 | ceil(N/500) | No |

**Total:** 1 job for N < 500

#### 7.X.6 `check_domain_target_patterns(X, y)`

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `bulk_corr_with_target()` on domain cols | leakage_detector.py:356 | ceil(D/500) | No |

**Total:** 1 job (D = domain-pattern matching columns, usually < 10)

### Notebook 07 Total

| Section | Jobs (N=50) | Jobs (N=200) |
|---------|------------|-------------|
| 7.3 Class Imbalance | 1 | 1 |
| 7.4 Per-column corr loop | 50 | 200 |
| 7.X.1 check_correlations | 1 | 1 |
| 7.X.2 check_separation | **200** | **800** |
| 7.X.3 check_temporal_logic | 1 | 1 |
| 7.X.4 check_single_feature_auc | **100** | **400** |
| 7.X.5 check_target_in_features | 1 | 1 |
| 7.X.6 check_domain_target_patterns | 1 | 1 |
| **Total** | **355** | **1,405** |

**With optimization (batch separation + batch collect + remove corr loop):** 8 / 9 jobs.
**Savings: 98-99% reduction.**

---

## Notebook 08: Baseline Experiments

### Section 8.2 — Feature Selection

**Call:** `FeatureSelector.get_availability_recommendations(findings.feature_availability)`

No Spark jobs (pure Python metadata inspection).

### Section 8.3 — Data Preprocessing

| Operation | Location | Jobs | Per-Column? |
|-----------|----------|------|-------------|
| `X[col].astype(str).factorize()[0]` | notebook cell (loop) | **M** | **YES** — 1 per cat column |
| `X[col].median()` | notebook cell (loop) | **N_null** | **YES** — 1 per col with nulls |
| `X[col].fillna(median)` | notebook cell | 0 | Lazy transform |

**Total: M + N_null jobs**

**Optimization:** Batch the `.median()` calls into a single
`df.agg(*[F.percentile_approx(c, 0.5) for c in null_cols])`. Batch the `.factorize()`
is harder since pyspark.pandas doesn't support multi-column factorize, but could convert
categorical columns to Spark StringIndexer in a single pipeline.

### Section 8.4 — Model Training

All sklearn (LogisticRegression, RandomForest, GradientBoosting). Zero Spark jobs.

### Section 8.5 — Model Evaluation

All numpy/sklearn metrics. Zero Spark jobs.

### Notebook 08 Total

| Section | Jobs (M=20, N_null=30) |
|---------|----------------------|
| 8.2 Feature Selection | 0 |
| 8.3 Preprocessing | 50 |
| 8.4 Model Training | 0 |
| 8.5 Evaluation | 0 |
| **Total** | **50** |

**With optimization (batch medians + StringIndexer pipeline):** ~5 jobs.

---

## Summary: All Three Notebooks

### Current State (N=50 numeric, M=20 categorical)

| Notebook | Current Jobs | After Optimization | Savings |
|----------|-------------|-------------------|---------|
| 06 Feature Opportunities | 28 | 22 | 21% |
| 07 Modeling Readiness | **355** | **9** | **97%** |
| 08 Baseline Experiments | 50 | 5 | 90% |
| **Combined** | **433** | **36** | **92%** |

### Current State (N=200 numeric, M=20 categorical)

| Notebook | Current Jobs | After Optimization | Savings |
|----------|-------------|-------------------|---------|
| 06 Feature Opportunities | 102 | 59 | 42% |
| 07 Modeling Readiness | **1,405** | **9** | **99%** |
| 08 Baseline Experiments | 55 | 5 | 91% |
| **Combined** | **1,562** | **73** | **95%** |

---

## Priority-Ranked Optimization Opportunities

### Priority 1: `check_separation` batch (leakage_detector.py + leakage.py)
- **Current:** 4N per-column Spark jobs (min/max on class-filtered data)
- **Fix:** Single batched `agg()` with conditional min/max expressions
- **Savings:** 4N → 1 job (e.g., 800 → 1 for 200 features)
- **Effort:** Small — replace `calculate_class_overlap` loop with batch function

### Priority 2: `check_single_feature_auc` batch (leakage_detector.py)
- **Current:** 2N per-column Spark jobs (`.values` collects per column)
- **Fix:** Single `.to_numpy()` on all numeric columns, then loop in pure numpy
- **Savings:** 2N → 2 jobs (e.g., 400 → 2 for 200 features)
- **Effort:** Small — collect once, index per column

### Priority 3: Remove per-column `.corr()` loop in NB07
- **Current:** N Spark jobs (redundant — `run_all_checks` already computes correlations)
- **Fix:** Replace loop with `bulk_corr_with_target()` or remove entirely (use leakage results)
- **Savings:** N → 1 job (e.g., 200 → 1)
- **Effort:** Minimal — notebook cell change

### Priority 4: Cache correlation matrix in NB06
- **Current:** `batched_corr_matrix` called twice (section 6.3 + 6.3.2)
- **Fix:** Pass `EffectiveFeaturesResult` from `analyze()` to `analyze_segment_capacity()`
- **Savings:** ceil(P/500) + 1 jobs (e.g., 41 → 0 for 200 features)
- **Effort:** Small — add parameter to `analyze_segment_capacity`

### Priority 5: Batch medians in NB08
- **Current:** N_null per-column `.median()` calls
- **Fix:** Single `agg(F.percentile_approx(...))` for all null columns
- **Savings:** N_null → 1 job (e.g., 30 → 1)
- **Effort:** Small

### Priority 6: Batch `.max()` calls in NB06 section 6.5
- **Current:** 4 separate `.max()` calls
- **Fix:** Single `agg(F.max(c1), F.max(c2), F.max(c3), F.max(c4))`
- **Savings:** 3 jobs
- **Effort:** Minimal

---

## Appendix: Where Are the Other 14K Jobs Coming From?

Notebooks 06-08 together contribute ~430 jobs (N=50). The remaining ~13,500+ jobs
come from **notebooks 01-05**, primarily:

| Notebook | Likely Source | Estimated Jobs |
|----------|-------------|---------------|
| 01 DataExplorer.explore() | `compute_bulk_stats` (6) + per-column profilers (fallback paths) | 6-500+ |
| 02 Source Integrity | Per-column validation checks | 50-200 |
| 03 Temporal Profiling | temporal_analyzer, time_series_profiler, temporal_coverage per-date-column | **2,000-5,000** |
| 04 Relationship Analysis | categorical_target_analyzer (per-column), relationship_recommender | **1,000-3,000** |
| 05 Aggregation | Multi-dataset merge + aggregated findings | 100-500 |

**Temporal profiling (NB03) is likely the largest single contributor** with:
- `temporal_analyzer.analyze()` + `analyze_seasonality()` + `calculate_growth_rate()`: ~25 jobs per date column
- `temporal_pattern_analyzer.analyze_cohorts()` + `analyze_recency()`: ~20 jobs
- `temporal_feature_analyzer.calculate_velocity()` + `calculate_momentum()` + `compare_cohorts()`: ~27N jobs per column
- `temporal_feature_engineer._compute_regularity()`: **4 jobs PER ENTITY** (critical scaling issue)
- `temporal_feature_engineer._compute_lifecycle()`: **3 jobs PER ENTITY**
- `time_window_aggregator._compute_value_counts()`: 1 job PER UNIQUE VALUE PER WINDOW

These per-entity and per-unique-value loops in notebooks 03-04 are the primary source of
the 14K accumulated job count. A separate analysis of notebooks 01-05 would be needed to
quantify and optimize those.
