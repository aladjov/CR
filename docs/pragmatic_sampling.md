# Pragmatic Sampling Strategy

## Goal

Establish a single, early sampling step that produces a well-stratified subset of entities. All downstream exploration notebooks operate on this subset. Production pipelines skip sampling entirely and run on all data — the findings/recommendations discovered during exploration are deterministic rules (thresholds, encodings, windows) that transfer directly to full-scale execution.

---

## Design Principle: Sample Entities, Not Rows

Every sampling decision operates at the **entity level**:

1. Pick a representative set of entity IDs (stratified by target, temporal cohort, and user-chosen columns).
2. For each selected entity, keep **all** its events/rows across **all** landing datasets.
3. Write the sampled entity ID list once (`sample_entity_ids.json`).
4. Every downstream notebook filters each dataset to those IDs before doing any work.

This guarantees entity-level integrity (aggregations, lifecycle features, temporal sequences stay correct) while reducing data volume proportionally.

---

## What the Sampling Must Preserve

| Property | Why It Matters | How We Preserve It |
|---|---|---|
| **Target distribution** (churn ratio) | Findings about class imbalance, thresholds, stratified splits | Proportional allocation per target class |
| **Temporal distribution** (event timing) | Trend/seasonality detection, drift, regime discovery | Stratify entities by first-event cohort (year-quarter) |
| **Entity integrity** (all rows per entity) | Aggregation correctness, lifecycle features, lag/velocity | Sample at entity level, never row level |
| **Rare category representation** | Categorical encoding decisions, outlier recommendations | Floor guarantee: keep all entities with rare target values |
| **User-chosen column distributions** | Domain-specific representativeness (e.g., geography, plan type) | Optional extra stratification axes from NB00 config |

---

## Sampling Algorithm (Notebook 00)

```
Input:  target_df with ENTITY_COLUMN, TARGET_COLUMN
Config: SAMPLE_ENTITY_COUNT (e.g., 5000), extra_strat_cols (optional)

1. Deduplicate to one row per entity (entity_id, target, cohort_quarter, extra_cols...)
2. Build stratification key = (target_class, cohort_quarter, extra_cols...)
3. Compute stratum proportions from full population
4. Allocate entity count per stratum (proportional, min floor = 1)
5. Within each stratum, random sample (seed=42)
6. If any target class has < min_rare_count entities total → keep ALL of them
7. Save sampled entity IDs to namespace.sample_entity_ids_path
8. Every dataset filtered to those IDs before downstream exploration
```

Cohort quarter = `year-Q{quarter}` of each entity's earliest event timestamp. This preserves temporal distribution without requiring row-level time stratification.

---

## Layer-by-Layer Analysis

### How to Read the Table

- **Layer**: Pipeline stage (Landing → Bronze → Silver → Gold → Training → Scoring)
- **Operation**: What computation or decision is made
- **Based On (Section)**: Which notebook section produces the finding/recommendation
- **Sampling Technique**: Can it work on the entity-stratified sample, or does it need full data?
- **Verdict**: SAMPLE = works on sample; FULL = needs all data; N/A = no data dependency

---

### Landing Layer

| Operation | Based On (Section) | Sampling Technique | Verdict |
|---|---|---|---|
| Column type inference | NB01 `DataExplorer.explore()` — bulk stats batch 1 (count, null, distinct) + 200-row head for type detection | Entity-stratified sample. Type detection uses `head(200)` already; bulk stats (mean, std, percentiles) converge well on 5K+ entities | **SAMPLE** |
| Null percentage measurement | NB01 bulk stats batch 1 | Proportions stable on stratified sample (CLT: SE < 1% at n=5000) | **SAMPLE** |
| Cardinality / distinct count | NB01 bulk stats batch 1 (`approx_count_distinct`) | Under-estimates on sample for high-cardinality columns; but decision thresholds (low=categorical, high=identifier) remain correct | **SAMPLE** |
| Distribution stats (skew, kurtosis, percentiles) | NB01 bulk stats batch 2 | Robust on 5K+ entity sample. Percentile error < 2% | **SAMPLE** |
| Outlier detection (IQR, Z-score bounds) | NB01 bulk stats batch 3 | IQR bounds stable on sample. May miss extreme rare outliers but bounds transfer to production | **SAMPLE** |
| Histogram bins | NB01 bulk stats batch 4 | Bin edges stable on sample. Shape preserved | **SAMPLE** |
| Datetime range detection (min/max dates) | NB01 bulk stats batch 5 | Stratification by cohort_quarter ensures early and late entities are represented. Min/max dates accurate | **SAMPLE** |
| Categorical value counts (top-20) | NB01 bulk stats batch 6 | Frequencies scale proportionally. Rare categories (<0.1%) may drop below detection — acceptable, they get "other" encoding anyway | **SAMPLE** |
| Text column metrics (length stats) | NB01 bulk stats batch 6 | Length distribution stable on sample | **SAMPLE** |
| Identifier pattern detection | NB01 bulk stats batch 7 | Regex patterns on `head(100)` — already sampled | **SAMPLE** |
| Target column identification | NB01 auto-detection | Binary/multiclass detection needs representative class balance — guaranteed by stratified sample | **SAMPLE** |
| Entity column identification | NB01 auto-detection | Uniqueness ratio (`nunique/len`) stable on sample | **SAMPLE** |
| Timestamp column identification | NB01 pattern matching | Format detection uses head samples already | **SAMPLE** |

**Summary**: All Landing findings work on the entity-stratified sample. No full-data pass needed.

---

### Temporal Analysis (Landing sub-phase)

| Operation | Based On (Section) | Sampling Technique | Verdict |
|---|---|---|---|
| Granularity detection (event vs entity vs snapshot) | NB01a `TemporalAnalyzer.detect_granularity()` | Ratio of rows-to-entities is preserved by entity sampling (same avg events per entity) | **SAMPLE** |
| Entity column confirmation | NB01a | Same uniqueness ratio | **SAMPLE** |
| Events-per-entity distribution | NB01a | Preserved exactly — all events kept per sampled entity | **SAMPLE** |
| Time span (days in data) | NB01a | Cohort stratification ensures min/max dates represented | **SAMPLE** |
| Aggregation window recommendations | NB01a — coverage per window (7d, 30d, 90d...) | Window coverage = fraction of entities with data in window. Proportional sampling preserves this | **SAMPLE** |
| Heterogeneity analysis (eta-squared) | NB01a | Effect size stable on 5K+ entities | **SAMPLE** |
| Temporal quality — missing value patterns by date | NB01b | Per-date null rates scale proportionally | **SAMPLE** |
| Temporal quality — duplicate detection | NB01b `check_duplicates()` | Need exact duplicate counts for dedup recommendations. Sample gives proportional estimate; production applies dedup to all rows regardless | **SAMPLE** (proportional estimate sufficient for deciding strategy) |
| Feature availability (new/retired tracking) | NB01b `FeatureAvailabilityMetadata` | First/last valid date per column needs early+late entities — guaranteed by cohort stratification | **SAMPLE** |
| Seasonality detection | NB01c `autocorr(lag)` | Autocorrelation on aggregated time series (counts per week). Aggregation over fewer entities still captures seasonal pattern if n > 1K | **SAMPLE** (n >= 1K entities) |
| Trend detection | NB01c `linregress(period_counts)` | Trend = slope of count-per-period. Proportional sample preserves trend direction; magnitude scales | **SAMPLE** |
| Regime detection | NB01c | Change-point detection on period aggregates. Works if sample is large enough per period | **SAMPLE** (n >= 2K entities) |
| Drift / PSI (population stability index) | NB01c | PSI compares distributions across time windows. Proportional sample preserves distribution shapes | **SAMPLE** |
| Recency analysis (days since last event) | NB01c | Per-entity metric — exact for sampled entities | **SAMPLE** |
| Event aggregation (time-window agg) | NB01d `spark_time_window_aggregator` | All events per entity preserved → aggregation is exact for sampled entities | **SAMPLE** |
| Aggregation function selection | NB01d | Based on column types (numeric→sum/mean/std, categorical→mode/count). Type-based, not data-volume-based | **SAMPLE** |

**Summary**: Temporal analysis works on entity-stratified sample because (a) all events per entity are kept, (b) cohort stratification ensures temporal coverage, (c) statistical measures converge at n >= 2K entities.

---

### Bronze Layer

| Operation | Based On (Section) | Sampling Technique | Verdict |
|---|---|---|---|
| Null handling strategy (impute median/mode/mean, drop) | NB04 `ColumnFinding.cleaning_recommendations` | Median/mode/mean converge well on sample. Production re-computes from all data | **SAMPLE** (decision transfers; fit values re-computed in production) |
| Outlier handling (IQR cap, percentile cap, winsorize) | NB04 outlier metrics | Cap bounds from sample percentiles. Production re-fits bounds on all data | **SAMPLE** |
| Segment-aware outlier capping | NB04 `segment_aware_cap` | Segment definitions (KMeans clusters) from sample. Acceptable — segments are approximate. Production re-fits | **SAMPLE** |
| Type conversions | NB04 type detection | Schema-level decision, not data-volume dependent | **SAMPLE** |
| Case consistency normalization | NB04 case variation detection | Variant detection (e.g., "yes"/"YES"/"Yes") visible in sample if variant frequency > 1/n_sample | **SAMPLE** |
| Deduplication strategy (keep_first/last/mode) | NB02 + NB04 | Strategy decision (which key, which resolution) based on pattern analysis. Works on sample | **SAMPLE** |
| Text embedding (model selection, PCA dims) | NB04a `TextProcessingMetadata` | PCA variance explained stabilizes on sample. Component count transfers to production | **SAMPLE** |
| Datetime feature derivation | NB01 `datetime_derivation_sources` | Derivation rules are deterministic transforms (year, month, day_of_week) — no fitting | **SAMPLE** |
| Future-column leakage masking | NB01 `datetime_allow_future_columns` | Detection of "future values" compares timestamps — works if temporal range is covered (guaranteed by cohort strat) | **SAMPLE** |

**Summary**: All Bronze decisions are either schema-level (type conversions) or statistical (medians, percentiles, cluster centers) that converge on sample and get re-fitted in production.

---

### Silver Layer

| Operation | Based On (Section) | Sampling Technique | Verdict |
|---|---|---|---|
| Cross-dataset relationship detection | NB02 + NB05 `RelationshipDetector` | Join key overlap ratio (unique_overlap / unique_count). Proportional sample preserves ratio | **SAMPLE** |
| Join strategy (left/inner/outer) | NB03 `MergeConfig` | Based on relationship type (1:1, 1:N, M:N) and orphan row percentage. Both preserved in sample | **SAMPLE** |
| Entity key selection | NB03 | Based on column uniqueness and naming patterns | **SAMPLE** |
| Merge order | NB03 | Based on dataset sizes and relationship directions | **SAMPLE** |
| Derived column recommendations (ratios, interactions) | NB06 `derived_columns` | Correlation-based feature suggestions. Pearson/Spearman stable at n >= 3K | **SAMPLE** |
| Temporal feature recommendations (lags, velocity, momentum) | NB06 + NB01a temporal metadata | Lag/velocity are per-entity transforms — exact for sampled entities | **SAMPLE** |
| Aggregation window selection | NB01d + NB05 | Window coverage analysis. Proportional sample preserves coverage ratios | **SAMPLE** |

**Summary**: Silver operations are merge/join decisions (structural, not volume-dependent) and correlation-based feature suggestions (statistically stable on sample).

---

### Gold Layer

| Operation | Based On (Section) | Sampling Technique | Verdict |
|---|---|---|---|
| Categorical encoding (one_hot, label, target, hash) | NB04 + NB08 `GoldRecommendations.encoding` | Encoding type based on cardinality thresholds. Sample cardinality is proportional | **SAMPLE** |
| Target encoding (mean target per category) | NB08 | Mean target per category. Converges at ~30 samples per category. If rare category has <30, target encoding wouldn't be recommended anyway | **SAMPLE** |
| Numeric scaling (standard, robust, min-max) | NB04 + NB08 `GoldRecommendations.scaling` | Scaling type decision based on distribution shape (skewness, outlier%). Transfers to production; fit values re-computed | **SAMPLE** |
| Power transformations (log, sqrt, yeo-johnson) | NB04 + NB08 `GoldRecommendations.transformations` | Transform selection based on skewness. Skewness stable on sample | **SAMPLE** |
| Feature selection — multicollinearity (drop correlated) | NB05 + NB08 `drop_multicollinear` | Correlation matrix. Pearson correlation stable at n >= 3K entities | **SAMPLE** |
| Feature selection — weak features (drop low effect) | NB08 `drop_weak` | Effect size (eta-squared, R-squared) with target. Effect sizes converge slower; may miss marginal features on small sample. Conservative approach: keep features where sample effect size is ambiguous | **SAMPLE** (conservative threshold) |
| Feature importance ranking | NB08 baseline model | Tree-based feature importance. Ordering is mostly stable on 50%+ of entities; magnitude varies | **SAMPLE** (ranking transfers; magnitude is informational) |
| Zero-inflation handling | NB04 | Zero percentage stable on sample | **SAMPLE** |

**Summary**: Gold operations are encoding/scaling/transform decisions based on distribution properties — all converge on sample. Feature selection needs slightly larger samples (3K+ entities) for effect size stability.

---

### Training Layer

| Operation | Based On (Section) | Sampling Technique | Verdict |
|---|---|---|---|
| Temporal train/test split cutoff date | NB08 `DataSplitter.TEMPORAL` | Cutoff = quantile of temporal column. During exploration, quantile on sample is fine for validation. **Production re-computes cutoff on full data** | **SAMPLE** for exploration; **FULL** for production |
| Cross-validation fold boundaries | NB08 `CVStrategy.TEMPORAL_ENTITY` | Fold boundaries from sample. Purpose is model selection, not final training | **SAMPLE** |
| Baseline model training (hyperparameter search) | NB08 | Model selection (which algorithm, which hyperparams). Decision transfers; final model re-trained in production on all data | **SAMPLE** |
| Class imbalance strategy (SMOTE, class_weight) | NB08 `ImbalanceHandler` | Imbalance ratio preserved by stratified sampling. Strategy decision transfers | **SAMPLE** |
| Model evaluation metrics (AUC, F1, precision, recall) | NB08 | Metrics on sample are estimates. Useful for model selection, not final reporting. Production re-evaluates | **SAMPLE** (directional accuracy) |

**Summary**: Training during exploration is for **model selection and hyperparameter tuning** — sample is sufficient. Production re-trains on full data using the selected configuration.

---

### Scoring Layer

| Operation | Based On (Section) | Sampling Technique | Verdict |
|---|---|---|---|
| Scoring pipeline validation | NB11 | Validate that scoring code runs correctly. Small sample sufficient | **SAMPLE** |
| Score distribution analysis | NB11 | Distribution shape on sample. Directional | **SAMPLE** |
| Production scoring | Generated `scoring.py` | Must score ALL entities | **FULL** (production only) |

---

## Consolidated Decision Table

| Layer | # Operations | All on Sample | Need Full Data | Need Full in Production Only |
|---|---|---|---|---|
| Landing | 13 | 13 | 0 | 0 |
| Temporal | 13 | 13 | 0 | 0 |
| Bronze | 9 | 9 | 0 | 0 |
| Silver | 7 | 7 | 0 | 0 |
| Gold | 8 | 8 | 0 | 0 |
| Training | 5 | 5 | 0 | 5 (production re-trains) |
| Scoring | 3 | 2 | 0 | 1 (production scores all) |
| **Total** | **58** | **57** | **0** | **6** |

**Key insight**: Every exploration operation can work on the entity-stratified sample. No operation requires full data during exploration. Production always operates on full data — it applies the discovered rules, re-fits statistics, and re-trains the model.

---

## Sampling Sizes and Expected Accuracy

| Sample Size (entities) | Statistical Confidence | Use Case |
|---|---|---|
| 500 | ~90% for major patterns, misses rare effects | Quick feasibility check |
| 2,000 | ~95% for distributions, seasonality, trends | Standard exploration |
| 5,000 | ~98% for correlations, feature importance ranking | Recommended default |
| 10,000 | ~99% for effect sizes, weak feature detection | Thorough exploration |
| Full | 100% | Production only |

Rule of thumb: **5,000 entities** is the sweet spot — large enough for all statistical decisions to converge, small enough to run the full notebook chain in minutes instead of hours.

---

## Implementation: Where Sampling Happens

### Current State

Notebook 00 (`00_start_here.ipynb`) already supports:
- `SAMPLE_ENTITY_COUNT` — fixed number of entities
- `SAMPLE_FRACTION` — percentage of entities
- `CR_SAMPLE_ENTITY_COUNT` — environment variable override
- Saves sampled IDs to `namespace.sample_entity_ids_path`

Notebook 01 (`01_data_discovery.ipynb`) already reads `sample_entity_ids.json` and filters `active_df`.

### What Needs to Change

1. **NB00**: Add cohort-quarter stratification (currently stratifies by target only). Add optional extra stratification columns.

2. **All notebooks (01a, 01b, 01c, 01d, 02, 03, 04, 04a, 05, 06, 07, 08, 09)**: After loading any dataset, filter to sampled entity IDs if `sample_entity_ids.json` exists. Currently only NB01 does this.

3. **Production pipeline** (`renderer.py` templates): Never sample. `FindingsParser` outputs the same `PipelineConfig` regardless of whether exploration used sampling — because findings are rules (thresholds, encodings, windows), not fitted values.

### Propagation Pattern

```
NB00  →  Determines sample entity IDs
          ↓ writes sample_entity_ids.json
NB01  →  Loads per-dataset, filters to sample IDs  ← already implemented
NB01a →  Loads per-dataset, filters to sample IDs  ← needs filter
NB01b →  Loads per-dataset, filters to sample IDs  ← needs filter
NB01c →  Loads per-dataset, filters to sample IDs  ← needs filter
NB01d →  Loads per-dataset, filters to sample IDs  ← needs filter
NB02  →  Loads multi-dataset, filters each to IDs  ← needs filter
NB03  →  Loads for merge, filters each to IDs      ← needs filter
NB04  →  Loads merged silver, filter to IDs         ← needs filter
NB05  →  Loads merged silver, filter to IDs         ← needs filter
NB06  →  Loads merged silver, filter to IDs         ← needs filter
NB07  →  Loads merged silver, filter to IDs         ← needs filter
NB08  →  Loads gold features, filter to IDs         ← needs filter
NB09  →  Loads gold features, filter to IDs         ← needs filter
          ↓
NB10  →  Generates pipeline code (no data loaded)   ← no change needed
```

The filter is a one-liner per dataset load:
```python
if _sample_ids_path.exists():
    _saved_ids = json.loads(_sample_ids_path.read_text())
    df = df[df[entity_col].isin(_saved_ids)]
```

### Alternative: Filter at Landing Write

Instead of filtering in each notebook, filter once when writing to the landing Delta table in NB01. Then all downstream stages (bronze, silver, gold) automatically see only sampled data.

Advantage: Zero changes to NB02-NB09.
Disadvantage: Re-running NB01 with different sample size requires re-writing landing tables.

**Recommendation**: Filter at landing write (NB01) for simplicity. If the user changes `SAMPLE_ENTITY_COUNT`, NB01 re-runs and all downstream data reflects the new sample.

---

## Stratified Sampling Algorithm Detail

```python
def stratified_entity_sample(
    entity_df,              # one row per entity: entity_id, target, first_event_date
    n_entities,             # desired sample size
    target_col,             # target column name
    time_col=None,          # optional: first event timestamp for cohort stratification
    extra_strat_cols=None,  # optional: user-chosen columns (e.g., region, plan_type)
    min_rare_count=10,      # floor for rare target classes
    random_state=42,
):
    # 1. Build stratification key
    strat_parts = [entity_df[target_col].astype(str)]
    if time_col and time_col in entity_df.columns:
        cohort = (entity_df[time_col].dt.year.astype(str)
                  + "-Q"
                  + entity_df[time_col].dt.quarter.astype(str))
        strat_parts.append(cohort)
    for col in (extra_strat_cols or []):
        if col in entity_df.columns:
            strat_parts.append(entity_df[col].astype(str))
    entity_df["_strat_key"] = reduce(lambda a, b: a + "|" + b, strat_parts)

    # 2. Proportional allocation per stratum
    strat_counts = entity_df["_strat_key"].value_counts()
    total = len(entity_df)
    allocation = (strat_counts / total * n_entities).round().astype(int).clip(lower=1)

    # 3. Rare target class floor
    for cls_val in entity_df[target_col].unique():
        cls_mask = entity_df[target_col] == cls_val
        cls_count = cls_mask.sum()
        if cls_count <= min_rare_count:
            # Keep all entities of this rare class
            allocation[entity_df.loc[cls_mask, "_strat_key"].unique()] = \
                entity_df.loc[cls_mask].groupby("_strat_key").size()

    # 4. Sample within each stratum
    sampled = []
    for key, group in entity_df.groupby("_strat_key"):
        n = min(allocation.get(key, 1), len(group))
        sampled.append(group.sample(n=n, random_state=random_state))

    return pd.concat(sampled)["entity_id"].tolist()
```

---

## Production Fitting: The Temporal Boundary Problem

This is the most important section of this document. Production fitting must respect the same temporal boundaries that training uses, or the model trains on features whose distribution doesn't match what the scaler/encoder was fitted on.

### Current Production Pipeline Flow

```
Landing (all data)
  → Bronze (clean, aggregate — stateless transforms, hardcoded bounds)
    → Silver (merge — no fitting)
      → Gold (fit scalers/encoders on ALL silver data, save artifacts)
        → Training (temporal split → train on pre-cutoff, test on post-cutoff)
          → Scoring (load fitted artifacts from gold, transform new data)
```

### Where Fitting Actually Happens

Tracing through `renderer.py` templates and `transforms/`:

| Transform | Where Fitted | What Data Sees | Temporal-Safe? |
|---|---|---|---|
| `apply_impute_null(df, col, value='median')` | **At execution time** — `df[col].fillna(df[col].median())` | Whatever `df` is passed | Depends on caller |
| `apply_cap_outlier(df, col, lower=X, upper=Y)` | **Hardcoded bounds** from exploration findings | N/A — bounds are constants | **YES** — no fitting |
| `apply_winsorize(df, col, lower_bound=X, upper_bound=Y)` | **Hardcoded bounds** from exploration findings | N/A — bounds are constants | **YES** — no fitting |
| `apply_cap_then_log(df, col)` | **At execution time** — `q99 = df[col].quantile(0.99)` | Whatever `df` is passed | Depends on caller |
| `apply_segment_aware_cap(df, col, n_segments=2)` | **At execution time** — KMeans + IQR on `df[col]` | Whatever `df` is passed | Depends on caller |
| `FittedScaler.fit_transform(df, col, store)` | **Gold step** — `StandardScaler.fit(ALL silver)` | **ALL silver rows** | **NO — PROBLEM** |
| `FittedEncoder.fit_transform(df, col, store)` | **Gold step** — `LabelEncoder.fit(ALL silver)` | **ALL silver rows** | **MINOR — categories only** |
| `FittedPowerTransform.fit_transform(df, col, store)` | **Gold step** — `PowerTransformer.fit(ALL silver)` | **ALL silver rows** | **NO — PROBLEM** |
| `apply_one_hot_encode(df, col)` | **Stateless** — `get_dummies()` | Whatever categories present | **YES** |
| `apply_log_transform` / `apply_sqrt_transform` | **Stateless** — fixed formula | N/A | **YES** |
| `LabelEncoder().fit_transform()` in `prepare_features()` | **Training step** — fit on `X` (post-split, all rows with notna target) | Pre-split data | **PARTIAL — see below** |

### The Three Problems

**Problem 1: Gold fits scalers/encoders on ALL silver data, including future test rows.**

`run_gold_features()` in the generated gold template:
```python
silver = load_silver()                    # ← ALL silver rows
gold = apply_gold_transformations(silver) # ← cap_then_log fits q99 on ALL
gold = apply_encodings(gold)              # ← FittedEncoder fits on ALL
gold = apply_scaling(gold)                # ← FittedScaler fits on ALL
_store.save_manifest()                    # ← artifacts reflect ALL data
```

Then training does:
```python
training_data = get_training_data_from_feast()  # ← loads ALL gold rows
splitter = DataSplitter(strategy=SplitStrategy.TEMPORAL, ...)
splits = splitter.split(split_df)               # ← splits ALREADY-SCALED data
X_train = splits.X_train                        # ← train scaled with test's distribution
```

The scaler's mean/std, the power transform's lambda, and `cap_then_log`'s q99 were all computed on the full dataset including the test period. This is **distribution leakage** — the train set's scaled values are influenced by test-period data.

**Problem 2: `apply_impute_null(value='median')` computes median on ALL data.**

In the bronze template, `df[col].fillna(df[col].median())` computes the median on whatever data is passed. In production, that's all landing data (all time periods). The median includes future values that the model shouldn't see during training.

**Problem 3: `prepare_features()` in training template fits `LabelEncoder` on pre-split but post-temporal-filter data.**

```python
y = training_data[TARGET_COLUMN]
X = prepare_features(training_data.drop(columns=[TARGET_COLUMN]))
# prepare_features calls LabelEncoder().fit_transform(df[col].astype(str))
# This fits on ALL data (train + test), before temporal split
```

### Severity Assessment

| Problem | Severity | Why |
|---|---|---|
| Scaler fit on all data | **Medium** | StandardScaler mean/std shifts by ~2-5% if test period has different distribution (drift). For stable distributions, negligible. For drifting features, causes train-test distribution mismatch. |
| PowerTransform fit on all data | **Medium-High** | Yeo-Johnson lambda is sensitive to distribution shape. Including test period can shift the transform, especially if the test period has a regime change. |
| Median imputation on all data | **Low** | Median is robust to temporal shifts unless there's a strong trend. Even then, the effect is small. |
| `cap_then_log` q99 on all data | **Low** | 99th percentile is robust. Including test period barely moves it unless test has extreme new outliers. |
| LabelEncoder on all data | **Low** | Only affects category-to-integer mapping. The mapping itself doesn't leak information — it's arbitrary label assignment. |

### The Fix: Fit on Train Data Only

The correct production flow is:

```
Landing (all data)
  → Bronze (stateless transforms only — hardcoded bounds, no median imputation)
    → Silver (merge)
      → Training:
          1. Temporal split on raw silver data
          2. Fit scalers/encoders/transforms on TRAIN split only
          3. Transform BOTH train and test using train-fitted artifacts
          4. Train model on transformed train
          5. Evaluate on transformed test
          6. Save fitted artifacts for scoring
      → Scoring:
          Load train-fitted artifacts → transform new data → predict
```

This means **gold should NOT fit** — it should only apply stateless transforms (log, sqrt, cap with hardcoded bounds, one-hot). Fitted transforms (scalers, power transforms, label encoders) must move into the training step, after the temporal split.

### What Changes in the Generated Code

**Gold template** (`gold.py`):
- Keep: `apply_cap_outlier`, `apply_winsorize`, `apply_log_transform`, `apply_sqrt_transform`, `apply_one_hot_encode`, `apply_cap_then_log` (with hardcoded q99 from exploration), `apply_zero_inflation_handling`, `apply_feature_select`, `apply_drop_column`, derived columns
- Remove: `FittedScaler.fit_transform`, `FittedEncoder.fit_transform`, `FittedPowerTransform.fit_transform`
- Replace `apply_impute_null(value='median')` with hardcoded median from exploration findings, OR move median computation to training

**Training template** (`training.py`):
- After temporal split, before model.fit:
  ```python
  # Fit on train only
  X_train = FittedScaler('standard').fit_transform(X_train, col, _store)
  X_test = FittedScaler('standard').transform(X_test, col, _store)  # use train params
  ```
- Same for `FittedPowerTransform` and `FittedEncoder`

**Scoring template**:
- Load artifacts fitted during training (not gold): `_store = ArtifactStore.from_manifest(...)`
- Apply `transform()` (never `fit_transform`) using train-fitted params

### Impact on `cap_then_log` and `segment_aware_cap`

These currently compute their parameters at execution time from whatever data they receive:

```python
# cap_then_log: q99 computed on ALL data
q99 = df[column].quantile(0.99)

# segment_aware_cap: KMeans + IQR on ALL data
labels = KMeans(n_clusters=n_segments).fit_predict(valid.values.reshape(-1, 1))
```

**Fix options**:
1. **Hardcode from exploration**: During exploration, compute q99 and segment boundaries on the sample. Bake these as constants into the generated code (like `cap_outlier` already does with `lower`/`upper`).
2. **Compute on train split**: Move into training step, compute on train data only.

Option 1 is simpler and consistent with how `cap_outlier` already works. The exploration findings determine the bound; production applies it as a constant.

### Summary of Fitting Scopes

| Transform | Current Scope | Correct Scope | Fix |
|---|---|---|---|
| `cap_outlier(lower, upper)` | Hardcoded constants | Hardcoded constants | **Already correct** |
| `winsorize(lower_bound, upper_bound)` | Hardcoded constants | Hardcoded constants | **Already correct** |
| `log_transform` | Stateless | Stateless | **Already correct** |
| `sqrt_transform` | Stateless | Stateless | **Already correct** |
| `one_hot_encode` | Stateless | Stateless | **Already correct** |
| `feature_select` / `drop_column` | Stateless | Stateless | **Already correct** |
| `derived_ratio/interaction/composite` | Stateless | Stateless | **Already correct** |
| `cap_then_log` | All data (q99) | Hardcode from exploration OR train-only | Bake q99 as constant |
| `segment_aware_cap` | All data (KMeans) | Hardcode from exploration OR train-only | Bake segment bounds |
| `impute_null(median)` | All data | Train-only OR hardcode from exploration | Bake median as constant |
| `FittedScaler` | **All silver (WRONG)** | Train split only | Move to training step |
| `FittedEncoder` (label) | **All silver (WRONG)** | Train split only | Move to training step |
| `FittedPowerTransform` | **All silver (WRONG)** | Train split only | Move to training step |

**9 of 15 transforms are already correct** (stateless or hardcoded). The 3 `Fitted*` classes must move to training. The 3 execution-time fits (`cap_then_log`, `segment_aware_cap`, `impute_null(median)`) should either hardcode from exploration or move to training.

---

## What Production Does Differently

| Aspect | Exploration (sampled) | Production (full data) |
|---|---|---|
| Data volume | 5K entities | All entities |
| Stateless transforms (cap, log, sqrt, one-hot) | Applied on sample | Applied on all data — same rules |
| Fitted transforms (scaler, encoder, power) | Fitted on sample (for model selection) | **Fitted on train split only** (post temporal split) |
| Hardcoded bounds (cap upper/lower, winsorize) | Discovered on sample | Applied as constants — same values |
| `cap_then_log` q99, `impute_null` median | Computed on sample | Baked as constant from exploration, OR computed on train split |
| Model training | On sample (for algorithm selection) | On full train split (final model) |
| Temporal split cutoff | Sample quantile (for validation) | Full-data quantile |
| Recommendations/rules | Discovered on sample | Applied as-is (which transforms, which columns, which windows) |
| Pipeline code | Generated from findings | Same code, different data |

The key contract: **findings are rules, not fitted values**. Exploration discovers *what to do* (impute with median, standard-scale, cap at 99th percentile, one-hot encode, use 30d window). The production pipeline applies stateless transforms directly and defers all fitting to the training step where it can respect the temporal split boundary.

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Rare category missed in sample | Medium (if n < 2K) | Low — falls to "other" bucket | Floor guarantee in sampling; default n=5K |
| Correlation sign flip on small sample | Low (if n >= 3K) | Medium — wrong feature dropped | Use n >= 5K; conservative multicollinearity threshold (0.95 not 0.9) |
| Seasonality missed (too few periods) | Low (if cohort-stratified) | High — wrong window recommendations | Cohort stratification ensures all periods represented |
| Extreme outlier not in sample | Medium | Low — production re-fits bounds | Acceptable; IQR-based bounds are distribution-based, not value-based |
| Weak feature appears strong on sample (or vice versa) | Medium (if n < 5K) | Low — production model re-evaluates | Conservative: keep ambiguous features, let production model decide |
| Target encoding unstable | Low (if n >= 5K) | Low — only used for high-frequency categories | Min 30 samples per category enforced |
| **Fitted transforms leak test into train** | **HIGH (current code)** | **Medium-High** — scaler/power params polluted by test period | **Move Fitted* to training step, fit on train split only** |
| Hardcoded bounds from sample don't match production | Low | Low — bounds are approximate by nature | Acceptable; cap/winsorize bounds work as directional constraints |

---

## Summary

- **Sample once** in NB00: entity-stratified by target + temporal cohort + optional user columns.
- **Filter once** in NB01 at landing write: all downstream stages see sampled data automatically.
- **Default 5,000 entities**: sufficient for all 58 exploration operations to produce accurate findings.
- **Production ignores sampling**: applies the discovered rules on full data, re-fits all statistics.
- **No operation requires full data during exploration**: the entire notebook chain runs on the sample.
- **Critical fix needed**: `FittedScaler`, `FittedEncoder`, and `FittedPowerTransform` must move from gold to training step so they fit on train-split-only data, never on future/purged rows. Stateless and hardcoded transforms stay in gold. This is independent of sampling — it's a correctness issue in the current pipeline.
