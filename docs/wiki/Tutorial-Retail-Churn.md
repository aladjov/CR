# Tutorial: Retail Customer Retention

This tutorial demonstrates a complete customer retention ML pipeline using a synthetic retail dataset. Rather than just showing *what* we do, we focus on *why* each step matters and *what decisions* follow from the analysis.

The framework follows an **intent-driven medallion architecture**: you declare what you're trying to predict, the framework propagates that declaration through every downstream notebook, and a deterministic production pipeline is generated at the end.

**[View Interactive Tutorial (HTML)](https://aladjov.github.io/CR/tutorial/retail-churn/)** - Browse all executed notebooks with visualizations

---

## The Business Problem

A retail company wants to predict which customers will churn so they can intervene proactively. The goal is to identify at-risk customers before they leave, enabling targeted retention campaigns.

**Key Questions We'll Answer:**
1. Is the data suitable for ML modeling?
2. What features drive customer retention?
3. How accurately can we predict churn?
4. What's the right model for production?
5. Does the model hold up on truly unseen future data?

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | `tests/fixtures/customer_retention_retail.csv` |
| **Rows** | 30,801 |
| **Columns** | 15 (raw), expanding to 34 after datetime derivation |
| **Unique Customers** | 30,769 |
| **Target** | `retained` (binary: 0=churned, 1=retained) |
| **Retention Rate** | 79.5% (3.9:1 class imbalance) |
| **Time Span** | 2008-06-17 to 2018-01-17 (3,501 days) |

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `custid` | Identifier | Unique customer ID |
| `retained` | Target | Customer retained (0/1) |
| `created` | Datetime | Account creation date |
| `firstorder` | Datetime | Date of first order |
| `lastorder` | Datetime | Date of last order |
| `esent` | Numeric | Emails sent to customer |
| `eopenrate` | Numeric | Email open rate (%) |
| `eclickrate` | Numeric | Email click rate (%) |
| `avgorder` | Numeric | Average order value |
| `ordfreq` | Numeric | Order frequency |
| `paperless` | Binary | Paperless billing (0/1) |
| `refill` | Binary | Auto-refill enabled (0/1) |
| `doorstep` | Binary | Doorstep delivery (0/1) |
| `favday` | Cyclical | Favorite shopping day (0-6) |
| `city` | Categorical | Customer city (DEL, BOM, MAA, BLR) |

---

## Pipeline Architecture

The pipeline progresses in three major phases, following the **medallion architecture**:

```
Phase I  - Bronze:  Intent → Discovery → Temporal Evidence → Source Integrity
Phase II - Silver:  Merge → Column Deep Dive → Relationship Analysis
Phase III - Gold:   Feature Engineering → Modeling → Production Pipeline → Scoring Validation
```

All steps after Bronze operate per objective. For this retail dataset -- entity-level with one row per customer -- the temporal event track (01a-01d) runs but produces simpler output than it would for event-level data with multiple transactions per customer.

---

## Phase 0: Intent Contract

**Purpose:** Before any exploration begins, declare the modeling intent. Every downstream notebook inherits this contract.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/00_start_here.html)

### Why Intent Matters

Without a declared intent, exploration is aimless. The intent contract ensures that temporal evidence, snapshot grids, aggregation logic, and training setup all serve a single coherent modeling goal. You declare *what* you're predicting and *how*, and the framework propagates those choices everywhere.

### What Gets Declared

| Parameter | Value | How It's Derived |
|-----------|-------|------------------|
| **Primary Objective** | Immediate risk | Auto-detected from target column `retained` matching churn/cancel pattern |
| **Secondary Objective** | Disengagement | Temporal span (3,501 days) suggests possible disengagement modeling |
| **Temporal Posture** | Long memory | Stable customer base with decade-long history |
| **Prediction Horizons** | 30, 60, 90 days | Derived from H=90: [H/3, 2H/3, H] |
| **Observation Window** | 270 days | max(180, 3 x H) = 270 |
| **Purge Gap** | 104 days | H + 14 = 104 (prevents label leakage) |
| **Label Window** | 90 days | Equals prediction horizon |
| **Cadence** | Weekly | Standard for immediate risk with H=90 |
| **Split Strategy** | Temporal | Time-ordered split (never random for production) |

### Dataset Fingerprint

The framework auto-detects dataset structure:

| Property | Detected Value |
|----------|---------------|
| Entity Column | `custid` |
| Time Column | `created` |
| Target Candidates | `retained` |
| Granularity | Entity-level (one row per customer) |
| Structure | Snapshot pattern |

### Snapshot Grid

A deterministic set of **434 weekly `as_of_date` values** spanning 2009-03-14 to 2017-07-01. This grid becomes the backbone of all aggregation and training -- every downstream notebook operates on these exact dates.

For event-level data, each dataset would *vote* on optimal cadence during temporal exploration (01a-01c), and votes would be aggregated into a consensus grid. For our entity-level dataset, the grid is derived directly from the intent parameters.

### Decision Made
- **Primary objective**: Immediate risk (highest confidence, 100%)
- **Posture**: Long memory (stable patterns over 10 years of history)
- **Purge gap**: 104 days to prevent temporal leakage between train and test
- **Cadence**: Weekly snapshots for adequate temporal coverage

### Alternatives Considered
- **Disengagement objective**: Detected at 60% confidence, available as secondary objective
- **Reactive posture**: Would use dense recent snapshots only; not ideal for this stable dataset
- **Shorter label window** (30 days): More aggressive churn definition, fewer positive examples

---

## Phase I: Bronze

### Stage 1: Data Discovery

**Purpose:** Profile each dataset, detect temporal patterns, derive datetime features, and create the landing Delta table.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/01_data_discovery.html)

### Why Point-in-Time Matters

The most common mistake in churn modeling is **data leakage** -- accidentally using information from the future to predict the past. If we use a customer's behavior from December to predict their January churn status, we're cheating.

The framework automatically:
1. Detects temporal columns (`created`, `firstorder`, `lastorder`)
2. Identifies the **feature timestamp** (`lastorder` -- when we last observe the customer)
3. Derives datetime features with **leakage guards** that mask future values
4. Creates a Delta Lake landing table for downstream consumption

### Key Findings

| Metric | Value | Implication |
|--------|-------|-------------|
| Structure | Entity-level | One row per customer (not event-level) |
| Feature Timestamp | `lastorder` (auto-detected) | Point-in-time anchor for each customer |
| Temporal Pattern | Snapshot | Static entity data, not repeated events |
| Stability Score | 0.85 (Stable) | Patterns consistent between historical and recent windows |
| Target Shift | 78.5% → 86.6% | Recent cohort has higher retention |
| Run ID | retail-e7471284 | Tracked for reproducibility |

**Datetime Feature Derivation:**
The framework derives **18 additional datetime features** from `created` and `firstorder`, expanding the dataset from 15 to 34 columns. Each derived feature includes a leakage guard: future dates for `created` (1.1% of rows) and `firstorder` (0.0%) are masked to NaN.

**Structural Stability:**

| Signal | Value | Meaning |
|--------|-------|---------|
| Volume Ratio | 1.56x | Recent window slightly busier |
| Entity Overlap (Jaccard) | 0.00 | Historical and recent cohorts are disjoint |
| Null Drift | 0.26% | Negligible missingness change |
| Distribution Drift | 0/27 columns shifted | No numeric features drifted significantly |
| Cadence Ratio | 1.63x | Recent cadence faster (13.7/day vs 8.4/day) |

### Temporal Track (01a-01d)

For **event-level data** (multiple rows per customer), the framework runs four additional notebooks:
- **01a** Temporal Deep Dive: Window feasibility, density, velocity, cadence votes
- **01b** Temporal Quality: Temporal data quality assessment
- **01c** Temporal Patterns: Seasonality, trends, regime detection
- **01d** Event Aggregation: Aggregate events against the consensus snapshot grid

For our entity-level dataset, these notebooks run but produce simpler output since there's one observation per customer. The key output is still the aggregated `(entity_id, as_of_date)` snapshots that feed into Silver.

### Decision Made
- **Feature timestamp**: `lastorder` (auto-detected as most recent activity indicator)
- **Leakage guards**: `created` and `firstorder` future values masked
- **Snapshot grid**: 434 weekly dates, ready for aggregation

### Alternatives Considered
- **Manual timestamp selection**: Override auto-detection if business has specific event definitions
- **Different recent window**: 270 days is derived from intent; could be overridden for seasonal businesses
- **Event-level modeling**: If data had transaction-level rows, the temporal track would produce richer aggregation features (recency, velocity, momentum, cohort trends)

---

### Stage 2: Source Integrity

**Purpose:** Validate data quality *per dataset* before merge. Drop unusable columns early to avoid provenance complexity downstream.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/02_source_integrity.html)

### Why Integrity Before Merge?

Garbage in, garbage out. By checking each source independently *before* merging, we:
- Eliminate broken columns early (no need to trace issues through merged data)
- Maintain clear data lineage (this column was bad in *this* source)
- Avoid merge artifacts masking quality issues

### Key Findings

**Target Distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| Retained (1) | 24,453 | 79.5% |
| Churned (0) | 6,316 | 20.5% |

**Imbalance ratio: 3.9:1** -- Mild to moderate. We'll handle it with class weights.

**Integrity Checks:**
- **Duplicates**: None detected
- **Missing values**: Minimal (0.06%) -- negligible impact
- **Lag2 columns**: 100% missing (expected -- lag windows extend beyond available history)
- **Outliers**: Cohort comparison features (`*_vs_cohort_mean`, `*_vs_cohort_pct`) have widespread out-of-range values due to entity-level data producing degenerate cohort statistics

**Quality Recommendations:**
- 405 column-level recommendations generated
- Severity levels: LOW (null imputation), MEDIUM (clip outliers), HIGH (drop or create indicator)
- Most issues are LOW severity -- data quality is excellent overall

### Decision Made
- Keep data mostly as-is -- quality is excellent
- Use **balanced class weights** in models to handle 3.9:1 imbalance
- Drop lag2 columns (100% missing -- no data in that time window)
- Flag cohort comparison features for review in Column Deep Dive

### Alternatives Considered
- **SMOTE oversampling**: Not needed -- imbalance is mild and we have 6,316 churned examples (plenty to learn from)
- **Undersampling majority**: Throws away data. Only useful for extreme imbalance (>10:1)
- **Aggressive column removal**: Could drop more aggressively, but prefer to defer decisions to modeling stage

---

## Phase II: Silver

### Stage 3: Dataset Merge

**Purpose:** Merge all Bronze outputs into a unified feature matrix aligned on `(entity_id, as_of_date)`.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/03_dataset_merge.html)

### Why Temporal Merge Matters

The `TemporalMerger` builds a **spine** from the snapshot grid and all known entities, then joins each dataset using the appropriate strategy:

| Dataset Shape | Strategy | Join Keys |
|---------------|----------|-----------|
| Event-level (aggregated) | Snapshot join | `entity_id` + `as_of_date` |
| Entity-level (no timestamp) | Broadcast | `entity_id` only (features repeat across dates) |
| Entity-level (with timestamp) | As-of join | `entity_id` + backward-looking temporal match |

### Key Findings

| Metric | Value |
|--------|-------|
| Unique Entities | 30,769 |
| Grid Dates | 434 (weekly cadence) |
| Grid Range | 2009-03-14 to 2017-07-01 |
| Spine Rows | **13,354,180** (30,769 entities x 434 dates) |
| Output Columns | 35 (entity_id + as_of_date + 33 features) |

For a single entity-level dataset, the merge is straightforward -- broadcast join. For multi-dataset scenarios, this is where temporal alignment becomes critical: event-level aggregations snap to the grid dates, and entity features propagate forward until updated.

### Decision Made
- **Merge strategy**: Broadcast (entity-level, single dataset)
- **Output**: Silver Delta table at `silver_merged`
- All downstream notebooks (04-11) operate on this merged table

---

### Stage 4: Column Deep Dive

**Purpose:** Analyze each column's distribution in the merged feature space. Detect issues and determine transformations.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/04_column_deep_dive.html)

### Why Distribution Analysis Matters

Not all features are created equal. Understanding distributions helps us:
- Identify **skewed features** that need transformation
- Detect **zero-inflation** (many zeros requiring special handling)
- Choose appropriate **encoding strategies** for categoricals
- Spot **data quality issues** before modeling

### Key Findings

**211 numeric columns analyzed** after datetime derivation and aggregation. Selected highlights:

| Feature | Skewness | Zeros % | Issue | Recommended Transform |
|---------|----------|---------|-------|----------------------|
| esent_sum_all_time | -0.05 | 11.0% | Symmetric | None (standard scaling) |
| eopenrate_sum_all_time | 1.17 | 24.8% | Right-skewed | Sqrt transform |
| eclickrate_sum_all_time | 3.89 | **50.2%** | Highly skewed, zero-inflated | Zero-inflation handling |
| avgorder_sum_all_time | **11.70** | 0.0% | Extreme outliers (kurtosis 548) | Cap + log transform |
| ordfreq_sum_all_time | **10.47** | **61.7%** | Highly skewed, zero-inflated | Zero-inflation handling |
| event_count_all_time | **47.72** | 0.0% | Extreme skew (kurtosis 5,125) | Yeo-Johnson |

**Key Insight:** Two features (`eclickrate`, `ordfreq`) have over 50% zeros. This is **zero-inflation** -- we can't just log-transform these. Instead, we create a binary `_is_zero` indicator plus a log-transformed value for non-zeros.

**Categorical Features:**

| Feature | Categories | Encoding Strategy | Why |
|---------|------------|-------------------|-----|
| city | 4 (DEL, BOM, MAA, BLR) | Target encoding | Low cardinality |
| favday | 7 (days of week) | Cyclical (sin/cos) | Preserves that Sunday ~ Saturday |
| cohort_quarter | 41 quarters | Target encoding | High cardinality (imbalance ratio 7,221x) |

### Decision Made
- Mark skewed features for log transformation in production pipeline
- Apply cyclical encoding for day-of-week (not ordinal, because Monday isn't "greater than" Sunday)
- Create zero-inflation indicators for features with >40% zeros
- Flag `avgorder` for cap + log (kurtosis 548 means extreme outliers)

### Alternatives Considered
- **Winsorization** (cap outliers at 99th percentile): Simpler but loses information
- **Binning**: Converts numeric to categorical, loses granularity but adds robustness
- **Keep as-is**: Tree-based models don't require normality, but interpretability suffers

---

### Stage 5: Relationship Analysis

**Purpose:** Identify which features predict retention, detect multicollinearity, and find interaction opportunities.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/05_relationship_analysis.html)

### Why Relationship Analysis Matters

Now that all columns are visible in the merged space, we can understand:
- Which features have **predictive signal**
- Are there **multicollinearity** issues (redundant features)?
- What's the **nature of the relationship** (linear? non-linear?)

This analysis only makes sense *after* merge -- interactions and redundancy only exist in the unified feature space.

### Key Findings

**Feature Importance (Effect Size):**

| Feature | Cohen's d | Correlation | Interpretation |
|---------|-----------|-------------|----------------|
| **esent_sum_all_time** | **+2.551** | **+0.718** | **LARGE** - dominates! |
| esent_vs_cohort_mean | +2.551 | +0.718 | Redundant with esent |
| lag0_esent_sum | +2.551 | +0.718 | Redundant with esent |
| esent_beginning | -- | +0.659 | Temporal split of esent |
| eopenrate_trend_ratio | -- | +0.433 | Moderate signal |
| avgorder_beginning | -- | -0.413 | Inverse moderate signal |

**Critical Insight:** `esent` (emails sent) has a **massive** effect size (d=2.551). This means retained customers receive, on average, 2.5 standard deviations more emails than churned customers.

**Multicollinearity:**
- **936 feature pairs** with |r| > 0.7
- Primary redundancy cluster: `esent_sum_all_time`, `esent_mean_all_time`, `esent_max_all_time`, `lag0_esent_sum`, `esent_vs_cohort_*` are all perfectly correlated (r=1.000) -- these are different views of the same underlying `esent` value
- **591+ HIGH priority recommendations** to drop multicollinear features

**Retention by Cohort Quarter:**

| Cohort | Retention Rate | Observation |
|--------|----------------|-------------|
| 2018 Q1 | **98.9%** (lift 1.24x) | Newest cohort, highest retention |
| 2009 Q4 | **48.4%** (lift 0.61x) | Oldest cohort, lowest retention |

Clear cohort effect: newer cohorts have substantially higher retention rates.

### Decision Made
- **Prioritize `esent`** -- it's the strongest predictor
- **Drop redundant features** -- 936 multicollinear pairs produce 591+ drop recommendations
- **Keep weak features** -- may help in combination after redundancy removal
- **Flag cohort effect** -- newer customers more likely retained, important for temporal splits

### Alternatives Considered
- **Drop all weak features** (avgorder, ordfreq): Risk losing predictive power in combinations
- **Build segment-specific models per city**: Adds complexity, may not be worth it
- **Use only non-linear models**: Would work, but linear interpretability valuable for business

### Caution: Feature Dominance Risk
`esent` accounts for most predictive power. This creates **concentration risk**:
- If email data quality degrades, model fails
- Model may be capturing "customers who receive emails stay" rather than underlying behavior
- Consider: Is this a **leading indicator** (more emails -> retention) or **trailing indicator** (retained customers get more emails because they're active)?

---

## Phase III: Gold

### Stage 6: Feature Opportunities

**Purpose:** Determine feature capacity, consolidate transformation strategies, and create derived features.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/06_feature_opportunities.html)

### Why Feature Capacity Matters

Adding more features isn't always better. With limited data, too many features leads to **overfitting**. The "Events Per Variable" (EPV) ratio tells us our budget:

| EPV Threshold | Risk Level | Recommended For |
|---------------|------------|-----------------|
| EPV < 10 | High overfitting risk | Only with strong regularization |
| EPV 10-20 | Moderate risk | Regularized models |
| EPV > 20 | Safe | Standard modeling |

### Key Findings

**Our Capacity:**

| Model Type | Max Features | Current | Status |
|------------|-------------|---------|--------|
| Linear (no regularization) | 274,374 | 19 | **ABUNDANT** |
| Regularized (L1/L2) | 548,749 | 19 | **ABUNDANT** |
| Tree-based | 445,139 | 19 | **ABUNDANT** |

With 13.3M rows and substantial minority class representation, we have capacity for hundreds of features without overfitting risk.

**47 Features Recommended:**
- 4 high-priority datetime features (days_since_as_of_date, days_since_created, days_since_firstorder, days_since_lastorder)
- 5 high-priority log transforms (eopenrate_log, eclickrate_log, avgorder_log, ordfreq_log, created_delta_hours_log)
- 2 high-priority categoricals (favday_sin_cos, city_encoded)
- 17 lower-priority binned features

**Key Derived Features:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| tenure_days | reference_date - created | Longer tenure -> more likely retained |
| days_since_last_order | reference_date - lastorder | Recency is key churn signal |
| email_engagement_score | 0.6 x openrate + 0.4 x clickrate | Composite engagement metric |
| service_adoption_score | paperless + refill + doorstep | More services -> stickier customer |

### Decision Made
- 47 features selected for modeling (transformations + derived + encoded)
- Apply log transforms to skewed features
- Use cyclical encoding for favday, target encoding for city and cohort_quarter
- Create zero-inflation indicators for eclickrate and ordfreq

### Alternatives Considered
- **Aggressive feature engineering** (100+ features): Capacity allows it, but diminishing returns
- **PCA dimensionality reduction**: Not needed with manageable feature count post-dedup
- **Automated feature generation** (featuretools): Overkill for this dataset

---

### Stage 7: Modeling Readiness

**Purpose:** Formalize the training setup: verify data quality, check for leakage, and confirm the dataset is ready for modeling.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/07_modeling_readiness.html)

### Key Findings

**Pre-Modeling Checklist (all PASS):**

| Check | Status |
|-------|--------|
| Target column identified | Pass |
| Feature columns available | Pass |
| No columns with >50% missing | Pass |
| Quality score >= 70 | Pass |
| Sufficient sample size (>=100) | Pass |

**Leakage Risk Assessment:**
- `is_future_created` and `is_future_firstorder`: flagged as **Medium risk** (name suggests post-prediction information). These are the leakage guard columns from datetime derivation -- they're metadata, not features, and are excluded from modeling.

**Result:** **Modeling Readiness Score: 100/100** -- 32 usable features, proceed to training.

---

### Stage 8: Baseline Experiments

**Purpose:** Establish performance benchmarks with standard models using entity-grouped temporal cross-validation.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/08_baseline_experiments.html)

### Why Baseline First?

Never start with complex models. Baselines tell us:
- Is the problem solvable at all?
- How much signal exists in the data?
- What's the performance floor to beat?

### Key Findings

**Model Performance Comparison:**

| Model | Test AUC | PR-AUC | F1 | Precision | Recall | CV Mean | CV Std |
|-------|----------|--------|-----|-----------|--------|---------|--------|
| Logistic Regression | 0.9685 | 0.9886 | 0.944 | 97.9% | 91.1% | 0.9696 | 0.0027 |
| Random Forest | 0.9818 | 0.9923 | 0.978 | 96.6% | 99.0% | 0.9824 | 0.0027 |
| **Gradient Boosting** | **0.9825** | **0.9937** | **0.981** | **97.0%** | **99.1%** | **0.9849** | **0.0024** |

**Winner: Gradient Boosting** with AUC 0.9825 -- Excellent!

**What These Numbers Mean:**
- **AUC 0.9825:** Model correctly ranks 98.3% of customer pairs by churn risk
- **PR-AUC 0.9937:** Near-perfect precision-recall tradeoff despite class imbalance
- **CV Std 0.0024:** Extremely stable across 5-fold cross-validation

**Feature Importance (Random Forest, top features):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **esent_mean_all_time** | **0.098** |
| 2 | esent_vs_cohort_mean | 0.092 |
| 3 | lag0_esent_sum | 0.086 |

**Critical Insight:** `esent`-derived features dominate the top 3 positions. The email signal is the primary predictor across all model types.

### Interpreting the Model

Feature importance tells us *what* matters, but not *how* it matters. For deeper understanding:

**Business Implication:** Focus email campaigns on customers receiving <10 emails -- that's where marginal impact is highest. Diminishing returns after ~30 emails.

**Caution on Causality:** High `esent` correlation with retention could mean:
- (A) More emails -> Higher retention (emails cause retention)
- (B) Active customers -> More emails (activity causes both)
- (C) Both are caused by a third factor (e.g., customer lifetime value)

Only A/B testing can establish causality. The model predicts, it doesn't explain *why*.

### Decision Made
- **Primary model:** Gradient Boosting (best AUC and stability)
- **Fallback model:** Logistic Regression (most interpretable, nearly as good)
- **Class weights:** Balanced (handles 3.9:1 imbalance well)

### Alternatives Not Explored (Future Work)
- **Neural Networks:** Overkill for tabular data
- **Hyperparameter tuning:** Current results are with defaults -- tuning could improve
- **Ensemble of all three models:** Could squeeze extra AUC

---

### Stage 9: Business Alignment

**Purpose:** Translate model output into operational decision logic with business-approved thresholds.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/09_business_alignment.html)

### Key Findings

**Success Metrics:**

| Metric | Target | Priority |
|--------|--------|----------|
| Model AUC | >= 0.80 | High |
| Precision at 20% | >= 0.60 | High |
| Churn Rate Reduction | 20% | High |
| Model Latency | < 100ms | Medium |

**Intervention Strategy:**

| Risk Level | Intervention | Cost | Expected Effectiveness |
|------------|-------------|------|----------------------|
| High (>0.8) | Personal call from account manager | $50/customer | 40% retention |
| Medium (0.5-0.8) | Personalized email + discount offer | $10/customer | 20% retention |
| Low (<0.5) | Automated engagement email | $0.50/customer | 5% retention |

### Decision Made
- Three-tier intervention strategy aligned with model confidence
- No direct PII in features (compliance-safe)
- Minimum 12 months historical depth required for production deployment

---

## Production

### Stage 10: Pipeline Generation

**Purpose:** Generate a deterministic production pipeline that faithfully replicates the exploration logic.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/10_spec_generation.html)

### Why Code Generation?

Exploration happens interactively in notebooks. Production requires deterministic, testable code. The framework bridges this gap by auto-generating production pipelines in two tracks:

| Track | Generator | Output |
|-------|-----------|--------|
| **Local** | `PipelineGenerator` | Python scripts with Delta Lake + pandas |
| **Databricks** | `DatabricksPipelineGenerator` | PySpark notebooks with Unity Catalog |

Both tracks read the same findings and recommendations, ensuring exploration and production are mathematically identical.

### Key Findings

**Generated Pipeline:**
- **17 files** across landing, bronze, silver, gold, training, and validation stages
- **Composite Name:** `cust_rete_reta_aggr__b6be84a`
- **Recommendations Hash:** `0575ed11` (version tag: `v1.0.0_0575ed11`)
- **Pipeline configuration**: 173 transformations, 954 feature selections, 1 encoding

**Recommendations by Layer:**

| Layer | Count | Examples |
|-------|-------|---------|
| Bronze | 3 | Datetime derivation, column type overrides |
| Silver | 7 | Merge strategy, temporal alignment |
| Gold | 1,128 | Drop multicollinear features, encoding, transformations |

**Generated Structure:**
```
generated_pipelines/local/customer_churn/
  config.py                  # Pipeline configuration (entity key, target, CN)
  pipeline_runner.py         # Orchestrates all stages
  landing/                   # Raw CSV -> Delta
  bronze/                    # Event aggregation, entity processing
  silver/                    # Feature merge
  gold/                      # Transformations, encoding, feature selection
  training/                  # MLflow experiment (LR, RF, XGBoost)
  feature_repo/              # Feast feature store definitions
  validation/                # Pipeline validation gates
```

### Decision Made
- **Local track**: Generated for development and testing
- **Feast integration**: Feature store materialized for online serving
- **MLflow tracking**: All experiments logged with hash-based versioning

---

### Stage 11: Scoring Validation

**Purpose:** Run the generated pipeline end-to-end, train models on production features, and validate on a point-in-time holdout set.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/11_scoring_validation.html)

### Why Validation Isn't Enough

Cross-validation tells us how well our model generalizes to **similar** data. But production data is from the **future** -- it may have different patterns due to seasonality, business changes, customer behavior shifts, or data quality drift.

The scoring validation runs the *full generated pipeline* and tests on a **point-in-time holdout** (data after the training cutoff).

### Key Findings

**Holdout Model Performance:**

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|-------|---------|--------|-----|-----------|--------|----------|
| Logistic Regression | 0.9675 | 0.9891 | 0.952 | 90.9% | 99.9% | 91.9% |
| Random Forest | 0.9653 | 0.9868 | 0.970 | 95.7% | 98.3% | 95.1% |
| **XGBoost** | **0.9687** | **0.9887** | **0.971** | **94.8%** | **99.5%** | **95.2%** |

**Training vs Holdout Comparison:**

| Model | Training AUC | Holdout AUC | **Gap** |
|-------|-------------|-------------|---------|
| Gradient Boosting / XGBoost | 0.9825 | 0.9687 | **-1.4%** |
| Random Forest | 0.9818 | 0.9653 | **-1.7%** |
| **Logistic Regression** | 0.9685 | **0.9675** | **-0.1%** |

**XGBoost remains the top performer on holdout** (0.9687), unlike the previous version of this tutorial where complex models dramatically degraded. All three models generalize well. However, Logistic Regression shows the smallest gap (0.1%) -- confirming that simpler models are more *stable* across distribution shifts, even when they don't outperform.

The [[Tutorial: Retail Customer Retention -- Initial Run|initial tutorial run]] showed a dramatic reversal: XGBoost dropped from 0.9854 to 0.9142 (-7.2%) while LR held at 0.9441 (-2.4%). That reversal is **not reproduced** in this run. The key difference: the initial run used **single-snapshot training** with **stratified random validation** (a single train/test split, stratified by target), while the current run uses **entity-grouped temporal cross-validation** with a 104-day purge gap. Combined with the new pipeline's feature engineering (datetime derivation with leakage guards, zero-inflation handling, multicollinearity removal), this produces a more stable feature set that doesn't degrade under distribution shift.

**Key takeaway:** A well-designed feature pipeline can reduce drift sensitivity for all model types, narrowing the robustness gap between simple and complex models.

### SHAP Analysis: What Drives Predictions

**Global Feature Importance (Top 10):**

| Rank | Feature | SHAP Importance |
|------|---------|-----------------|
| 1 | **esent_sum_all_time** | **2.141** |
| 2 | esent_mean_all_time | 1.227 |
| 3 | city_mode_all_time | 0.327 |
| 4 | city_mode_30d | 0.290 |
| 5 | event_count x esent (interaction) | 0.286 |
| 6 | eclickrate_is_zero | 0.214 |
| 7 | favday_mode_all_time | 0.181 |
| 8 | paperless_mode_all_time | 0.130 |
| 9 | firstorder_delta_hours_is_zero | 0.124 |
| 10 | city_mode_180d | 0.124 |

**Individual Customer Examples:**

*Customer 1 (retained, esent=45):* `esent_sum_all_time` contributes SHAP +2.369. High email engagement is the dominant retention signal.

*False Positive (predicted retained, actually churned, esent=20):* Moderate email engagement (SHAP +0.918) misled the model, but zero click rate (SHAP -0.430) was a warning sign the model partially captured.

*False Negative (predicted churned, actually retained, esent=10):* Low email count (SHAP -1.841) drove the churn prediction, but the customer retained anyway -- likely through a channel not captured in features.

### The Key Lesson

**A well-engineered feature pipeline matters more than model choice.** In the [[Tutorial: Retail Customer Retention -- Initial Run|initial run]] (single-snapshot training with stratified random split, no temporal purge gaps or leakage guards), XGBoost degraded by 7.2% on holdout while LR held steady. In this run, with entity-grouped temporal cross-validation and the full intent-driven pipeline, all models degrade minimally (0.1-1.7%). The framework's temporal safeguards -- purge gap, leakage masking, entity-aware splitting -- stabilized the feature space enough that even complex models generalize well.

That said, LR's 0.1% gap vs XGBoost's 1.4% gap still shows that simpler models have an inherent stability advantage. In environments with more severe drift, this margin could widen.

### Recommendations

1. **XGBoost for production** -- best holdout AUC (0.9687) with acceptable 1.4% gap
2. **LR as drift-resilient fallback** -- only 0.1% degradation; switch if drift monitoring triggers
3. **Monitor `esent` features** -- they dominate SHAP (2.141 + 1.227 = top 2 features)
4. **Watch `city` encoding drift** -- city appears in 3 of the top 10 SHAP features
5. **Track eclickrate_is_zero** -- zero-inflation indicators are 6th most important
6. **Set drift alerts** -- retrain when feature distributions shift >20%

---

## Model Interpretability: Beyond SHAP

While SHAP provides both global and local explanations (as demonstrated in Stage 11), it's not the only interpretability technique. Different methods answer different questions:

### Comparison of Interpretation Methods

| Method | Question It Answers | Scope | Pros | Cons |
|--------|---------------------|-------|------|------|
| **SHAP** | How much did each feature contribute to this prediction? | Local & Global | Theoretically grounded, consistent | Slow for large datasets |
| **LIME** | Which features matter for this specific prediction? | Local | Fast, model-agnostic | Unstable, sensitive to parameters |
| **Permutation Importance** | How much does performance drop if we shuffle this feature? | Global | Simple, reliable | Correlated features problematic |
| **Partial Dependence (PDP)** | What's the average effect of changing this feature? | Global | Easy to understand | Assumes feature independence |
| **Counterfactuals** | What minimal change would flip this prediction? | Local | Actionable insights | Multiple valid answers |

### When to Use Each Method

**For Business Stakeholders (need simple explanations):**
- **Partial Dependence Plots**: "Customers who receive more than 20 emails have 90% retention"
- **Permutation Importance**: "Email count is the most important feature overall"
- **Counterfactuals**: "If this customer had received 5 more emails, they would likely have stayed"

**For Model Debugging (need detailed analysis):**
- **SHAP**: Understand exactly how each feature contributed (see Stage 11 examples)
- **Error analysis**: Compare SHAP profiles of false positives vs true positives

**For Regulatory/Compliance (need audit trail):**
- **SHAP with force plots**: Show contribution breakdown for each decision
- **Counterfactuals**: Demonstrate what would change the outcome

### Interpretation Example for Our Model

**Global View (SHAP from Stage 11):**
```
Feature                                SHAP Importance
────────────────────────────────────────────────────────
esent_sum_all_time                     2.141  ████████████████████████
esent_mean_all_time                    1.227  █████████████
city_mode_all_time                     0.327  ███
eclickrate_is_zero                     0.214  ██
favday_mode_all_time                   0.181  ██
paperless_mode_all_time                0.130  █
```

**Local View (False Negative -- predicted churned but retained):**
```
Current:    esent=10, prediction=CHURN
SHAP:       esent_sum_all_time → -1.841 (strong churn push)
            ordfreq_is_zero=0 → -0.621 (some orders, but not enough to overcome)

Actionable insight: Customers with low email engagement can still retain through
                    other channels -- consider broader feature set.
```

### Caution: Interpretation Pitfalls

1. **Correlation != Causation**: `esent` predicts retention, but does sending more emails *cause* retention, or do retained customers simply receive more emails because they're active?

2. **Feature Interactions**: SHAP captures interactions (ranked #5: `event_count x esent`), but interpreting them requires domain knowledge.

3. **Out-of-Distribution Explanations**: SHAP explanations can be misleading for unusual customers (e.g., someone with 200 emails sent -- far outside training distribution).

4. **Redundant Features**: Multiple `esent`-derived features in the top ranks inflate the apparent importance of email. After deduplication, the picture may be more balanced.

---

## Summary: Key Takeaways

### What We Learned

| Phase | Stage | Key Insight |
|-------|-------|-------------|
| Intent | 00 Start Here | Intent contract ensures all downstream steps serve a coherent goal |
| Bronze | 01 Data Discovery | Auto-detected `lastorder` as feature timestamp; 18 datetime features derived with leakage guards |
| Bronze | 02 Source Integrity | 3.9:1 class imbalance handled with balanced weights; data quality excellent |
| Silver | 03 Dataset Merge | 434 weekly grid dates produce 13.3M row spine for temporal modeling |
| Silver | 04 Column Deep Dive | 50%+ zero-inflation in `eclickrate` and `ordfreq` requires special handling |
| Silver | 05 Relationships | `esent` dominates with effect size d=2.551; 936 multicollinear pairs flagged |
| Gold | 06 Feature Opportunities | Feature capacity ABUNDANT; 47 features selected |
| Gold | 07 Modeling Readiness | Readiness score 100/100; no blocking issues |
| Gold | 08 Baseline Experiments | Gradient Boosting AUC 0.9825; all models excellent |
| Gold | 09 Business Alignment | Three-tier intervention strategy aligned with model confidence |
| Production | 10 Pipeline Generation | 17 files generated; 1,138 recommendations codified |
| Production | 11 Scoring Validation | **All models generalize well (0.1-1.7% gap) -- proper feature engineering reduces drift sensitivity** |

### Decisions Made vs. Alternatives

| Decision | Why | Alternative | When Alternative Better |
|----------|-----|-------------|------------------------|
| Immediate risk objective | 100% confidence, target matches churn pattern | Disengagement objective | If re-engagement timing more important than risk |
| Long memory posture | Stable 10-year dataset | Reactive posture | If recent patterns dominate (e.g., seasonal business) |
| Temporal cross-validation (not random) | Simulates production conditions; [[Tutorial: Retail Customer Retention -- Initial Run|initial run]] showed 7.2% degradation with stratified random split | Random stratified | Only for small datasets without temporal structure |
| Balanced class weights | 3.9:1 is mild imbalance | SMOTE oversampling | If imbalance >10:1 |
| Log transform skewed features | Reduces outlier impact | Winsorization | If outliers are meaningful |
| Keep weak features | May help in combination | Drop features | If interpretability paramount |
| Cyclical encoding for day | Preserves circular structure | One-hot | If days don't wrap (Monday != Sunday) |
| XGBoost for production | Best holdout AUC (0.9687), acceptable 1.4% gap | Logistic Regression | If drift is severe and stability is paramount |

### What Wasn't Explored (Future Work)

**Model Improvements:**
1. **Hyperparameter tuning** -- Could improve AUC by 1-2%
2. **SHAP-based feature selection** -- Remove features with negligible SHAP contribution
3. **Segment-specific models** -- Cohort effects (48% vs 99% retention by quarter) suggest potential

**Drift Robustness:**
4. **Walk-forward cross-validation** -- Train on 2008-2015, validate on 2016, etc.
5. **Adversarial feature selection** -- Remove drift-prone features
6. **Recency-weighted training** -- Give more weight to recent samples

**Interpretability:**
7. **Counterfactual analysis** -- "What would change this prediction?"
8. **Partial Dependence Plots** -- Visualize non-linear feature effects for stakeholders

**Production:**
9. **A/B testing framework** -- Measure actual intervention effectiveness
10. **Online learning pipeline** -- Continuous model updates
11. **Prediction intervals** -- Quantify uncertainty in predictions

---

## Running the Tutorial

```bash
# Clone and install
git clone https://github.com/aladjov/CR.git
cd CR
pip install -e ".[dev,ml]"

# Start Jupyter
jupyter lab exploration_notebooks/00_start_here.ipynb
```

Set `DATA_PATH = "tests/fixtures/customer_retention_retail.csv"` in the first notebook and run notebooks 00-11 sequentially. The framework handles dataset registration, intent detection, and all downstream configuration automatically.

Alternatively, run all notebooks non-interactively:
```bash
python scripts/notebooks/run_exploration.py --notebooks-dir exploration_notebooks/ --timeout 600
```

---

## Next Steps

- [[Architecture]] -- Understand the intent-driven medallion architecture
- [[Snapshot Grid and Control Variables]] -- How the as-of grid is derived from intent
- [[Model Intent and Objective Support]] -- How prediction objectives are declared and validated
- [[Local Track]] -- Generate production pipelines with Feast + MLflow
- [[Databricks Track]] -- Deploy to Databricks with Unity Catalog
