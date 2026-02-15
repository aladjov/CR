# Tutorial: Customer Email Engagement & Retention

This tutorial demonstrates a complete customer retention ML pipeline using an **event-based email engagement** dataset. Unlike the [[Retail Retention|Tutorial-Retail-Churn]] which works with entity-level data (one row per customer), this tutorial shows how the framework handles **event-level data** -- where each row is an email event and multiple rows belong to a single customer. The pipeline follows an **intent-driven medallion architecture** organized into three phases: Bronze (evidence gathering and aggregation), Silver (merge and feature exploration), and Gold (modeling and production). The focus is on *why* each phase's decisions drive everything downstream.

**[View Interactive Tutorial (HTML)](https://aladjov.github.io/CR/tutorial/customer-emails/)** - Browse all executed notebooks with visualizations

---

## The Business Problem

A company wants to predict which customers will disengage based on their email interaction history. The data: 83,198 email events spanning nine years -- opens, clicks, bounces, timestamps -- across 4,998 customers. Somewhere in these interactions lies the pattern that separates customers who stay from those who leave.

The goal is straightforward: build a model that predicts who will churn. But the data is organized by *event*, not by *customer*. Each row is a single email send, and a single customer may have dozens or hundreds of rows. Before we can build customer profiles, we need to reshape the entire dataset -- aggregating thousands of individual events into a coherent feature set for each customer.

**The five questions we'll answer:**
1. How do we convert 83K email events into customer-level features for modeling?
2. What temporal patterns in email engagement predict retention?
3. Which aggregation windows and derived features carry the most signal?
4. How accurately can we predict churn from email behavior alone?
5. Does the best validation model hold up in production conditions?

**The key thing to understand upfront:** This is event-level data. In entity-level datasets, each customer is already a single row with pre-computed features, and the analysis proceeds directly to column profiling. Here, the framework must first *detect* that rows represent events, then *aggregate* them into customer-level features -- choosing the right time windows, summary statistics, and derived metrics. This aggregation step (Event Aggregation, NB01d) is the critical decision point that doesn't exist in entity-level pipelines. Everything before it is preparation; everything after depends on it.

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | `tests/fixtures/customer_emails.csv` |
| **Total Events** | 83,198 email interactions |
| **Customers** | 4,998 unique entities |
| **Features** | 13 columns (event-level) |
| **Target** | `unsubscribed` (binary: 0=retained, 1=churned) |
| **Retention Rate** | 55.4% retained / 44.6% churned (after aggregation) |
| **Time Span** | 2015-01-01 to 2023-12-30 (9 years) |
| **Avg Events/Customer** | 16.6 emails per customer |

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `email_id` | Identifier | Unique email event ID |
| `customer_id` | Entity Key | Customer identifier (links events to customers) |
| `sent_date` | Datetime | When the email was sent |
| `campaign_type` | Categorical | Campaign category (6 types) |
| `opened` | Binary | Whether customer opened the email (0/1) |
| `clicked` | Binary | Whether customer clicked a link (0/1) |
| `send_hour` | Numeric | Hour of day email was sent (6-22) |
| `subject_line_category` | Categorical | Subject line category (6 types) |
| `device_type` | Categorical | Device used to open (3 types) |
| `bounced` | Binary | Whether email bounced (0/1) |
| `time_to_open_hours` | Numeric | Hours until email was opened |
| `unsubscribed` | Binary | Whether customer unsubscribed (target) |
| `unsubscribe_date` | Datetime | Date of unsubscription (if applicable) |

**Critical distinction:** This is **event-level** data. The 83K rows represent individual email sends, with many rows per customer. The framework must aggregate these into one row per customer before modeling.

---

## Phase I -- Bronze: Evidence, Grid & Aggregation

The first phase transforms raw event data into clean, aggregated entity-level features. Event datasets are explored independently (NB01a--01c) to gather temporal evidence, then aggregated onto a deterministic snapshot grid (NB01d). Source-level quality checks (NB02) ensure clean inputs before the Silver merge. All steps produce artifacts consumed downstream but never re-evaluated later.

---

### Intent Contract -- Defining the Prediction Goal (NB00)

Before any data is examined, the framework requires a declaration of *what* you are trying to predict and *how*. This **intent contract** propagates through every downstream notebook, controlling what temporal evidence is gathered, how snapshots are generated, and how models are trained.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/00_start_here.html)

#### What the Framework Configures

| Parameter | Value | Derivation |
|-----------|-------|------------|
| **Project Name** | `email` | User-specified |
| **Primary Objective** | `immediate_risk` (100% confidence) | Auto-detected from `unsubscribed` column |
| **Secondary Objective** | `disengagement` (90% confidence) | Temporal span sufficient (3,285 days) |
| **Temporal Posture** | `STABLE` (long memory) | Use extended historical context |
| **Prediction Horizon** | 90 days | User-specified |
| **Observation Window** | 270 days | `max(180, 3 × H)` = 270 |
| **Purge Gap** | 104 days | `H + 14` = 104 (prevents leakage between feature cutoff and label start) |
| **Label Window** | 90 days | Equals prediction horizon |
| **Cadence** | Weekly (7 days) | Immediate risk with H=90 |
| **Split Strategy** | Temporal | Time-ordered split required |

The intent contract ensures that every downstream decision -- which grid dates to aggregate on, how to split train/test, what label window to use -- traces to a single, explicit configuration rather than ad-hoc choices scattered across notebooks.

#### Snapshot Grid

The intent drives a **snapshot grid** -- the deterministic set of `as_of_date` values that define when features are computed. For this dataset:

| Setting | Value |
|---------|-------|
| **Grid Mode** | `NO_ADJUSTMENTS` (fixed from intent) |
| **Cadence** | Weekly (7 days) |
| **Observation Window** | 270 days |
| **Grid Dates** | 404 weekly snapshots |
| **Grid Range** | 2015-09-28 to 2023-06-19 |

In `NO_ADJUSTMENTS` mode, the grid is fully determined by the intent and doesn't wait for evidence from temporal notebooks (01a--01c). In multi-dataset scenarios with `ALLOW_ADJUSTMENTS`, each event dataset would vote on cadence and window parameters during temporal exploration, and a consensus grid would be derived after all votes are collected.

#### What We Take Forward

- **Intent contract** propagates objective, horizon, and posture to all downstream notebooks
- **Snapshot grid** (404 weekly dates) becomes the temporal backbone for aggregation in NB01d
- **Purge gap** (104 days) prevents temporal leakage between features and labels
- **Cadence** (weekly) determines scoring frequency in production

---

### Data Discovery -- First Look at the Data (NB01)

Before searching for patterns, we need to understand what we're working with. Is the data organized by customer (entity-level) or by event (event-level)? Getting this wrong -- treating events as entities or vice versa -- would either lose temporal information or produce nonsensical features. The first job is classification, not analysis.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01_data_discovery.html)

#### What the Framework Detects

The framework's temporal detection engine examines the data and identifies its structure automatically:

| Metric | Value | Implication |
|--------|-------|-------------|
| **Temporal Scenario** | Production | Real timestamps available, auto-detected |
| **Granularity** | EVENT_LEVEL | Multiple rows per customer detected |
| **Temporal Pattern** | EVENT_LOG | Timestamped sequence of interactions |
| **Entity Column** | `customer_id` | Auto-detected |
| **Unique Entities** | 4,998 | Customers in the dataset |
| **Avg Events/Entity** | 16.6 | Sufficient for temporal aggregation |
| **Timestamp Source** | `last_action_date` (coalesced) | See below |
| **Coverage** | 83,198 / 83,198 (100%) | All rows have a resolved timestamp |
| **Date Range** | 2015-01-01 to 2023-12-30 | ~9 years of email history |
| **Event-Level Target** | 97.3% class 0, 2.7% class 1 | Per-event distribution (misleading -- see Temporal Patterns) |

This last number -- 97.3% vs 2.7% -- is misleading. It looks like almost no one churns. But that's the *event-level* view: most individual emails don't trigger an unsubscription. The *customer-level* picture is dramatically different, as we'll discover in Temporal Patterns after aggregation reveals the true 55:45 split.

#### Timestamp Coalescing

**Timestamp coalescing** is a subtle but important operation. The framework builds a **coalesced timestamp** (`last_action_date`) by analyzing all available datetime columns (`sent_date`, `unsubscribe_date`), ordering them chronologically by median date, and resolving each row's timestamp from latest to earliest. This is essential for event-level data: `unsubscribe_date` is populated for only ~2.7% of events, while `sent_date` covers every row. Coalescing guarantees 100% coverage.

#### What We Take Forward

- **Activated temporal track** (notebooks 01a--01d) for event aggregation -- this is the path unique to event-level data
- **Coalesced timestamps** used because no single datetime column has full coverage
- The misleading event-level target (97.3:2.7) will be corrected in Temporal Patterns after aggregation reveals the true 55.4:44.6 split
- All downstream notebooks load from the versioned snapshot

---

### Temporal Deep Dive -- Understanding the Rhythm of Engagement (NB01a)

With the data classified as event-level, the next question is deceptively simple: *If we're going to summarize each customer's email history into fixed time windows, which windows should we use?*

The wrong answer wastes effort (windows too short produce mostly zeros) or loses signal (windows too long blur recent behavior into historical noise). The framework evaluates each candidate through a multi-gate scoring process that prevents any single metric from dominating the decision.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01a_temporal_deep_dive.html)

#### Time Series Profiling

**Why we do this:** Before evaluating windows, we need to understand the basic rhythm of the data -- how many events per customer, how spread out, how long the history spans. These metrics determine which windows are even plausible.

**What the analysis shows:**

| Metric | Value |
|--------|-------|
| Time Span | 3,285 days (9.0 years) |
| Median Inter-Event Gap | 95 days |
| Mean Inter-Event Gap | 145.5 days (right-skewed) |
| Volume Trend | Declining (-30%) |
| Data Gaps | 0 detected |

The median gap of 95 days is the single most important number from this stage. It means the typical customer interacts roughly once a quarter -- and any window shorter than that will be empty for most people. This observation will recur throughout the analysis: it explains the window selection here, the high null rate in aggregated features, and the zero-inflation in Column Deep Dive.

#### Entity Lifecycle Analysis

**Why we do this:** Customers vary in both *how long* they've been around (tenure) and *how intensely* they engage (event frequency). Crossing these dimensions reveals whether one model fits all.

**What the analysis shows:** Four quadrants emerge with very different behavioral profiles. The framework computes **eta-squared** = 0.335 (high heterogeneity), meaning lifecycle segment explains 33.5% of the variance in engagement patterns.

**Auto-derived recommendation:** Add `lifecycle_quadrant` as a feature. The eta-squared value is well above the 0.14 threshold for "high heterogeneity." This segmentation feature will prove its worth repeatedly: it explains the false outlier problem in Source Integrity (Cramer's V in Relationship Analysis) and drives a 75+ percentage-point churn spread in Event Aggregation.

**Alternative technique:** Build separate models per quadrant. Rejected because the smallest segment has insufficient EPV (far below the minimum of 10), as Feature Opportunities confirms.

#### The Multi-Gate Scoring Process

Each candidate window passes through a sequence of gates:

**Gate 1 -- Dataset Span (Hard Gate):** The dataset's total time span must be at least **2x the window size**. A 365-day window needs at least 730 days of data.

**Gate 2 -- Entity Duration Adequacy:** Even if the dataset covers 9 years, individual customers may have been active for only 3 months. The framework checks each entity's `duration_days` against the window size.

**Gate 3 -- Event Density:** For each customer, the framework projects expected events in the window: `event_count * (window_days / duration_days)`. A threshold of **>=2 expected events** separates meaningful aggregation from noise.

**Gate 4 -- Coverage Threshold:** A window is only useful if at least **10%** of entities pass both Gates 2 and 3.

#### Gate Results for This Dataset

| Window | Coverage | Density (events/entity) | Result |
|--------|----------|-------------------------|--------|
| 7d | <10% | ~0.1 | Excluded -- fails coverage |
| 30d | <10% | ~0.3 | Excluded -- fails coverage |
| 90d | Borderline | ~0.9 | Excluded -- density below 2-event threshold |
| 180d | Passes | ~1.9 | Included -- near threshold but above coverage gate |
| 365d | Passes | ~3.8 | Included -- adequate density |
| All time | 100% | ~15.0 | Always included |

The median inter-event gap of 95 days means the 90-day window is timing-aligned but still fails the 2-event density test for most customers. The 180-day window is the shortest window that clears all gates -- it's the tightest recency signal the data can support.

#### Decision Made
- **Three aggregation windows**: `180d`, `365d`, `all_time` -- the minimum set that passed all gates while preserving recency-vs-history contrast
- **Lifecycle quadrant** added as a categorical feature (high heterogeneity, eta-squared=0.335)
- **Drift risk flagged**: Volume declining (-30%), population stability 0.66

> **Caution:** The declining volume trend (-30%) means recent windows contain less data per customer than historical ones. Features from 180-day windows will have high null values -- customers with no recent activity. This is informative (absence of activity is a signal), but models must handle the missingness.

---

### Temporal Quality -- Validating Event Data (NB01b)

Aggregating events without first validating their quality risks propagating data issues into features. Event-level data can contain duplicates, temporal gaps, future-dated records, and ordering inconsistencies that would silently corrupt aggregated features.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01b_temporal_quality.html)

#### Quality Checks

| Component | Score | Issues |
|-----------|-------|--------|
| Duplicate Events | 23.1/25 | 371 duplicates (0.50%) |
| Temporal Gaps | 25.0/25 | None |
| Future Dates | 25.0/25 | None |
| Event Ordering | 23.1/25 | 371 ambiguous |
| **Total** | **96/100 (Grade A)** | |

#### Missing Value Patterns

- **`time_to_open_hours` at 77.6% missing** -- this is MNAR (Missing Not At Random). The column is only populated when an email is opened. The 77.6% of missing values *are* the data: they represent emails that were never opened. Imputing these with a mean or median would destroy the signal.
- **`unsubscribe_date` at 97.3% missing** -- also MNAR. Only populated for churned customers. The missingness pattern directly encodes the target variable.

**Auto-derived recommendation:** Preserve both patterns. Do not impute. These will flow through aggregation as informative nulls.

#### Decision Made
- **371 duplicate events removed** during aggregation (0.5% of data)
- **`time_to_open_hours`**: Retained despite 77.6% missingness -- the pattern of missingness itself is informative
- **Segment-aware outlier treatment** recommended over global treatment

---

### Temporal Patterns -- Discovering Behavioral Signals (NB01c)

This is where key insights emerge. We've understood the data structure (NB01), its rhythm (NB01a), and verified its integrity (NB01b). Now we look for the behavioral signatures that separate customers who stay from those who leave.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01c_temporal_patterns.html)

#### Target Resolution: Correcting the Class Balance

**Why we do this:** The event-level target distribution (97.3% retained, 2.7% churned) is misleading because one customer contributes many events. We need to see the picture at the granularity where we'll actually model.

**What the analysis shows:** After aggregating via `max` per customer:
- Event-level: 97.3% class 0, 2.7% class 1 (misleading)
- Entity-level: **55.4% retained, 44.6% churned**

This changes our entire understanding of class balance. What looked like an extreme imbalance problem (97:3) is actually a nearly balanced one (55:45). The event-level view was dominated by prolific customers who generated many email events but only one churn label.

#### Recency Analysis: The Strongest Signal

**Why we do this:** How recently a customer engaged is often the single strongest churn predictor.

**What the analysis shows:**

| Metric | Value |
|--------|-------|
| Median recency | 246 days |
| Target correlation | **0.772** (strong) |
| Cohen's d | **+2.23** (large effect) |
| Retained mean recency | 1,399 days |
| Churned mean recency | 165 days |

Churned customers were active **1,234 days more recently** than retained ones. This is counterintuitive at first. Why would *churned* customers have more *recent* activity?

The explanation: customers who unsubscribe do so shortly after receiving emails (triggering recent activity). The "retained" customers haven't interacted in years -- they simply never unsubscribed. The model is detecting "recently engaged then left" rather than "gradually disengaged."

> **Caution on Causality:** The strong recency signal could be a **trailing indicator**, not a leading one. By the time recency flags a customer, it may already be too late for intervention.

#### Effect Sizes (Cohen's d)

| Feature | Cohen's d | Interpretation |
|---------|-----------|----------------|
| `tenure_days` | -2.403 | Churned customers have much longer tenure |
| `opened_std` | -0.988 | Churned have more variable open behavior |
| `opened_sum` | -0.915 | Churned opened more emails total |
| `opened_mean` | -0.834 | Churned have higher open rate |
| `event_count` | -0.759 | Churned received more emails |
| `clicked_sum` | -0.630 | Churned clicked more |

The negative Cohen's d values reveal a counterintuitive pattern: churned customers were *more engaged*, not less. They opened more emails, clicked more links, had higher open rates. These are customers who actively engaged then consciously decided to unsubscribe. The truly disengaged customers never bothered to unsubscribe; they just stopped opening.

#### Velocity and Momentum

Clicked momentum (d=-0.97) and opened momentum (d=1.01) are both strong signals -- the rate of change in engagement discriminates nearly as well as the level of engagement itself.

#### Feature Engineering Summary

Recommendations for aggregation:
- **Recency features** (highest priority): `days_since_last_event`, recency buckets
- **Seasonality encoding**: `dow_sin`, `dow_cos`
- **Momentum**: `clicked_momentum_180_365`
- **Skip**: trend features (R² too low), cohort features (90% onboarded in 2015 -- insufficient variation)

These recommendations directly configure the aggregation step -- every derived feature has a traceable origin in NB01a or NB01c.

---

### Event Aggregation -- Building Customer Profiles (NB01d)

Everything converges here. The windows from NB01a, the quality guarantees from NB01b, the feature recommendations from NB01c -- all feed into this single transformation. We're converting 82,798 individual email events into 4,998 customer profiles, each described by 217 features. This is the irreversible step: poor aggregation loses signal, over-aggregation creates redundancy, and the choices made here cascade through every downstream analysis.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01d_event_aggregation.html)

#### How Findings Inform Aggregation

Every parameter traces to a prior analysis:

| Source | Insight | Application in Aggregation |
|--------|---------|---------------------------|
| NB00 | Snapshot grid (404 weekly dates) | Temporal backbone for as_of_date snapshots |
| NB01a | 95-day median gap → 180d/365d/all_time | Window selection |
| NB01a | Eta-squared=0.335 | Add lifecycle_quadrant feature |
| NB01b | 371 duplicates, 96/100 quality | Deduplicate before aggregation |
| NB01b | time_to_open_hours 77.6% MNAR | Preserve nulls, don't impute |
| NB01c | Recency d=2.23 | Create days_since_last_event |
| NB01c | Weekly autocorrelation | Add dow_sin/cos |
| NB01c | Clicked momentum d=-0.97 | Add clicked_momentum_180_365 |

#### The Shape Transformation

| Metric | Value |
|--------|-------|
| **Input events** | 82,798 (after dedup) |
| **Output entities** | 4,998 |
| **Features created** | 217 |
| **Memory** | 3.5 MB |
| **Target distribution** | 55.4% retained, 44.6% churned |

The 217 features are organized into seven temporal feature groups:

| Group | Count | Purpose |
|-------|-------|---------|
| Lagged Windows | 80 | Sequential non-overlapping time horizons |
| Velocity | 10 | Rate of change between windows |
| Acceleration | 10 | Change in velocity, momentum |
| Lifecycle | 20 | Beginning/middle/end of history |
| Recency | 4 | How recently customer was active |
| Regularity | 5 | Consistency of engagement |
| Cohort Comparison | 15 | Customer vs peer group |

Plus derived features: `dow_sin`, `dow_cos`, `lifecycle_quadrant`, `recency_bucket`, `clicked_momentum_180_365`, and additional aggregation statistics.

#### Target Proxy Detection

A critical safety feature: the framework detects **target-proxy datetime columns** by checking whether a datetime column's null pattern correlates with the target. For this dataset, `unsubscribe_date` has a null-pattern correlation of **1.00** with `unsubscribed` -- it is non-null only when unsubscribed=1. Including it in datetime derivation would create 70+ perfectly leaky features.

The framework automatically excludes `unsubscribe_date` from temporal feature derivation, preventing this leakage silently.

#### Lifecycle Quadrant vs. Churn Rate

| Quadrant | Customers | Churn Rate | Interpretation |
|----------|-----------|------------|----------------|
| Intense & Brief | 1,679 | **82.7%** | High engagement, short tenure -- likely to churn |
| One-shot | 816 | **76.7%** | Minimal engagement, short tenure -- expected churn |
| Occasional & Loyal | 1,683 | 7.6% | Sparse but persistent -- low risk |
| Steady & Loyal | 820 | 10.4% | Consistent engagement -- lowest risk |

The 75+ percentage-point spread between Intense & Brief (82.7%) and Occasional & Loyal (7.6%) confirms that lifecycle segmentation captures meaningful behavioral differences. This single feature nearly predicts churn on its own.

#### Recency Bucket Distribution

| Bucket | Entities | Percentage |
|--------|----------|------------|
| 0-7d | 123 | 2.5% |
| 8-30d | 364 | 7.3% |
| 31-90d | 725 | 14.5% |
| 91-180d | 702 | 14.0% |
| >180d | 3,084 | **61.7%** |

The 61.7% of customers in the >180d bucket connects directly to the quarterly cadence from NB01a -- these customers haven't interacted recently relative to the observation window. This pattern is informative, not missing data.

#### Leakage Validation

The leakage check flagged 2 potential issues after target-proxy exclusion:
- `clicked_velocity_pct` (LD010) -- possible class separation
- `active_span_days` (LD053) -- domain pattern correlation

Both are assessed as false positives (legitimate behavioral features that happen to correlate with churn). The leakage gate is non-fatal: it warns rather than blocks, allowing the modeling notebook to evaluate these features empirically.

#### Decision Made
- **217 features** from 3 windows, 7 feature groups + derived features
- **Nulls preserved** as informative (not imputed to zero)
- **`unsubscribe_date` excluded** from datetime derivation (target-proxy, correlation=1.00)
- **371 duplicate events removed** before aggregation

---

### Source Integrity -- Per-Dataset Quality Gate (NB02)

Before merging datasets in Silver, each Bronze dataset undergoes independent quality validation. Source-level cleanup happens *before* the merge to avoid provenance complexity later -- dropping unusable columns early prevents them from polluting the merged feature space.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/02_source_integrity.html)

#### What the Framework Checks

| Check | Result | Detail |
|-------|--------|--------|
| **Duplicates** | 0 (0.00%) | Clean entity keys (email_id is unique) |
| **Quality Score** | 93.1/100 | Excellent |
| **Target Distribution** | 97.3:2.7 (event-level) | 36.19:1 imbalance at event level |
| **Missing Values** | 2 columns | `time_to_open_hours` (77.6%), `unsubscribe_date` (97.3%) |
| **Segment-Aware Outliers** | 3 segments detected | 84.7% of `time_to_open_hours` outliers are false positives |
| **Date Logic** | Valid | No placeholder dates, range 2015-01-01 to 2023-12-30 |
| **Binary Fields** | 3 valid | `opened`, `clicked`, `bounced` -- all clean 0/1 |
| **Consistency** | No issues | No case variants or spacing problems |

The segment-aware outlier analysis is a key finding: 782 of 923 global outliers in `time_to_open_hours` are normal values within their segment. Global outlier treatment would distort legitimate data from different customer behavioral groups.

#### Recommendations Generated

4 Bronze-layer recommendations saved for pipeline generation:
- Imbalance strategy: SMOTE consideration for severe event-level imbalance (not needed at entity level after aggregation)
- Segment-aware outlier treatment for `time_to_open_hours`
- Missing indicator strategy for high-missingness columns

---

## Phase II -- Silver: Merge & Analytical Exploration

The Silver phase creates a unified feature matrix by merging all Bronze datasets onto a temporal spine, then explores the merged feature space through column-level analysis and relationship detection. Everything operates on the merged data -- cross-dataset interactions and redundancy only exist in the combined space.

---

### Dataset Merge -- Creating the Temporal Spine (NB03)

All Bronze datasets are merged onto a temporal spine defined by the cross product of all entity IDs and all grid dates. This produces a unified `(entity_id, as_of_date)` feature matrix where every entity has a feature row at every grid date -- even if it had no activity in that period. The spine preserves the temporal structure needed for point-in-time correct training and scoring.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/03_dataset_merge.html)

#### Spine Construction

| Metric | Value |
|--------|-------|
| **Unique Entities** | 4,998 |
| **Grid Dates** | 404 (weekly cadence) |
| **Spine Rows** | 2,019,192 |
| **Estimated Size** | 121.3 MB (spine only) |

#### Merge Results

| Metric | Value |
|--------|-------|
| **Datasets Merged** | 1 (customer_emails) |
| **Final Shape** | 2,019,192 rows × 218 columns |
| **Temporal Integrity** | PASS |
| **Spine Preservation** | Verified (row count matches) |

The merge follows the medallion architecture's **PIT column standardization**: event datasets join on `(entity_id, as_of_date)` via equi-join, entity datasets broadcast across all dates, and entity datasets with `feature_timestamp` use as-of joins. For this single-dataset scenario, the customer_emails aggregated data joins directly on the grid dates.

The merged Silver table is saved as a Delta Lake table at `namespace.silver_merged_path`. A column-level exploration (`silver_merged_findings.yaml`) is generated automatically, cataloguing all 218 columns for downstream notebooks.

#### Why This Step Matters

In multi-dataset scenarios (e.g., emails + transactions + support tickets), the merge produces cross-dataset feature interactions that don't exist in any single source. Even for this single-dataset tutorial, the temporal spine creates the 2M-row structure needed for point-in-time feature engineering in the production pipeline -- ensuring that features computed at each `as_of_date` only use data available at that time.

---

### Column Deep Dive -- Post-Merge Feature Distributions (NB04)

With 218 features in the merged space, we examine each one. Aggregated features from event data tend to look unusual -- heavy right skew, zero-inflation, extreme values -- but many of these "problems" are expected consequences of the quarterly engagement cadence identified in NB01a.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/04_column_deep_dive.html)

#### Numeric Distribution Analysis

The 180-day window features are consistently the most problematic because a majority of customers had no activity in that window. This is a direct consequence of the quarterly engagement cadence -- the 95-day median gap means more than half of customers will have no events in any given 180-day window.

**Auto-derived transformation decision tree:** Zeros >40% → binary indicator + log(non-zeros); |skewness| >1 → log; kurtosis >10 → cap first, then transform.

#### Categorical Analysis

| Column | Categories | Encoding |
|--------|-----------|----------|
| `lifecycle_quadrant` | 4 | One-hot |
| `recency_bucket` | 5 | One-hot |

Both features have low cardinality, making one-hot encoding straightforward.

#### Transformation Recommendations

The framework auto-derives transformation recommendations for the Gold layer. These feed directly into the generated pipeline (NB10), ensuring production applies the same transforms as exploration.

---

### Relationship Analysis -- Which Features Matter? (NB05)

With 217+ features, many derived from the same underlying data, we need to determine which carry unique signal, which are redundant, and which dominate. This is the narrowing phase -- from everything we've built to what actually matters. Relationship analysis happens *after* the merge because interactions and redundancy only exist in the merged feature space.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/05_relationship_analysis.html)

#### Feature-Target Relationships

`days_since_last_event` dominates -- the recency signal from NB01c (d=2.23 at the event level) is *stronger* after aggregation because the entity-level view separates the signal more cleanly. Count features across windows show high multicollinearity, as expected.

#### Categorical Features (Cramer's V)

Both `lifecycle_quadrant` and `recency_bucket` show strong association with the target. The lifecycle quadrant created during aggregation (NB01d), which originated from the heterogeneity analysis in NB01a (eta-squared=0.335), has been validated through the entire pipeline: high eta-squared → 75+ percentage-point churn spread → strong Cramer's V.

#### Redundancy Detection

101 redundant features identified in correlated clusters. Tree-based models handle this natively; for linear models, feature selection or PCA would be needed.

> **Caution: Feature Dominance Risk.** `days_since_last_event` dominates at nearly 2x stronger than any other feature. This is both a strength (strong signal) and a risk (model becomes a recency detector). If recency patterns shift over time, the model could degrade rapidly.

---

## Phase III -- Gold: Modeling & Production

The Gold phase transforms the Silver feature matrix into modeling-ready data, trains and validates models, aligns results with business requirements, and generates a production pipeline. Per the medallion architecture, **label columns** (`label_timestamp`, `target`) are introduced *only* at the Gold layer, keeping them out of Bronze and Silver feature engineering to prevent leakage.

---

### Feature Opportunities -- Consolidating the Feature Set (NB06)

This notebook consolidates all transformation, derivation, and encoding recommendations from earlier stages into a candidate feature set.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/06_feature_opportunities.html)

#### Feature Capacity (EPV)

| Metric | Value | Status |
|--------|-------|--------|
| **Total Samples** | 4,998 | -- |
| **Current Features** | 210+ usable | -- |
| **Redundant Features** | 101 | In correlated clusters |

EPV is well above the minimum of 10, meaning we have ample data for all model types.

#### Segment-Specific Capacity

The smallest lifecycle segment has EPV far below the minimum of 10. Separate models per segment would overfit severely.

**Auto-derived recommendation:** Single global model with `lifecycle_quadrant` as a feature, confirming the NB01a advisory.

---

### Modeling Readiness -- Final Safety Gate (NB07)

This is the final safety gate before model training. The notebook validates that the data is ready for machine learning -- checking for leakage, class imbalance, missing values, and sample size.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/07_modeling_readiness.html)

#### Readiness Assessment

| Check | Status | Detail |
|-------|--------|--------|
| Target column identified | Pass | `unsubscribed` (binary) |
| Feature columns available | Pass | 210 usable features |
| No data leakage detected | Pass | No columns with >0.9 target correlation |
| Quality score >= 70 | Pass | |
| Sufficient sample size (>=100) | Pass | 4,998 rows |

#### Class Imbalance Assessment

| Metric | Value |
|--------|-------|
| Retained (0) | 2,771 (55.4%) |
| Churned (1) | 2,227 (44.6%) |
| Imbalance Ratio | 1.24:1 (**LOW**) |
| Recommended Strategy | `class_weight='balanced'` |

The imbalance is very mild -- stratified sampling and class weights are sufficient. No oversampling needed.

---

### Baseline Experiments -- Training and Validation (NB08)

Two connected steps: validate data readiness, then train baseline models to test whether the features actually predict churn. The notebook loads the aggregated entity-level data (4,998 rows × 210 features) rather than the full temporal matrix, because modeling operates at entity granularity.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/08_baseline_experiments.html)

#### Data Preparation

| Setting | Value |
|---------|-------|
| **Data Source** | Aggregated entity-level (4,998 rows) |
| **Features** | 210 (29 binary, 2 categorical, 79 continuous, 100 discrete) |
| **Split** | Stratified random 80/20 |
| **Train** | 3,998 rows |
| **Test** | 1,000 rows |

#### Model Comparison

| Model | Test AUC | PR-AUC | F1-Score | Precision | Recall |
|-------|----------|--------|----------|-----------|--------|
| Logistic Regression | 0.9638 | 0.9661 | 0.9291 | 0.9529 | 0.9096 |
| Random Forest | 0.9697 | 0.9717 | 0.9326 | 0.9704 | 0.8966 |
| **Gradient Boosting** | **0.9708** | **0.9726** | **0.9333** | **0.9799** | **0.8883** |

**All three models achieve AUC > 0.96.** The performance gap is only 0.7% AUC spread (0.9638 to 0.9708). This tells us something important: the signal in the features is strong enough that model choice barely matters. The aggregation decisions in NB01d determined the outcome more than any model architecture could.

**Classification Report (Gradient Boosting):**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Retained (0) | 0.92 | 0.98 | 0.95 |
| Churned (1) | 0.98 | 0.89 | 0.93 |
| **Accuracy** | | | **0.94** |

#### Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `days_since_last_event` | **0.073** |
| 2 | `days_since_first_event` | 0.065 |
| 3 | `send_hour_count_365d` | 0.055 |

Recency features dominate -- confirming NB01c's prediction. The importance is distributed more evenly across the 210 features than in the old pipeline (which had only 72 features), but recency remains the strongest predictor.

#### Decision Made
- **Primary model:** Gradient Boosting (best AUC at 0.9708)
- **Fallback model:** Logistic Regression (nearly identical performance, more interpretable)
- **Class weights:** Balanced (handles 1.24:1 imbalance)
- **Assessment:** Excellent predictive signal -- production-ready with tuning

---

### Business Alignment -- Mapping Predictions to Actions (NB09)

With a working model, we document business context, success criteria, and intervention strategies. A model with AUC 0.97 is useless if it doesn't align with how the business will act on predictions.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/09_business_alignment.html)

#### Success Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Model AUC | >= 0.80 | 0.9708 | Exceeds by 21% |
| Precision at 20% | >= 0.60 | -- | To validate |
| Churn Rate Reduction | 20% | -- | Post-deployment |
| Model Latency | < 100ms | -- | Infrastructure-dependent |

#### Intervention Strategy

| Risk Level | Intervention | Cost | Expected Retention |
|------------|-------------|------|--------------------|
| High (>0.8) | Personal call from account manager | $50/customer | 40% |
| Medium (0.5-0.8) | Personalized email + discount offer | $10/customer | 20% |
| Low (<0.5) | Automated engagement email | $0.50/customer | 5% |

---

### Pipeline Generation -- From Exploration to Production (NB10)

The exploration notebooks made hundreds of decisions -- window sizes, aggregation functions, transforms, encodings, feature selections. Hand-coding a pipeline that reproduces all of these introduces errors. Auto-generation ensures the production pipeline exactly matches the exploration findings.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/10_spec_generation.html)

#### Recommendations Applied

| Layer | Recommendations |
|-------|----------------|
| **Bronze** | 3 recommendations (null handling, outlier treatment, deduplication) |
| **Silver** | 7 recommendations (joins, aggregations, derived columns) |
| **Gold** | 664 recommendations (encoding, scaling, transformations, feature selection) |

#### Generated Pipeline Structure

Following the medallion architecture, the pipeline uses **Composite Names** (CN) derived from the sorted source names. For `customer_emails_aggregated`, the CN is `cust_emai_aggr__26e8271`.

```
generated_pipelines/local/customer_churn/
├── landing/landing_customer_emails.py
├── bronze/bronze_event_customer_emails.py
├── bronze/bronze_entity_customer_emails_aggregated.py
├── silver/silver_featureset_cust_emai_aggr__26e8271.py
├── gold/gold_features_cust_emai_aggr__26e8271.py
├── training/ml_experiment.py
├── scoring/run_scoring.py
├── feature_repo/
│   ├── feature_store.yaml
│   └── features.py
├── validation/validate_pipeline.py
├── config.py
├── pipeline_runner.py
├── run_all.py
├── workflow.json
└── manifest.json
```

#### Pipeline Execution Results

| Stage | Output |
|-------|--------|
| **Landing** | 83,198 raw events loaded |
| **Bronze** | Event aggregation applied |
| **Silver** | Holdout: 499 entities (10%), Training: 4,499 (90%) |
| **Gold** | Fit artifacts saved (version `v1.0.0_742edc35`), features materialized to Feast (4,998 rows) |
| **Training** | 3 models trained via MLflow |

**Pipeline model results:**

| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|-----|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 |

The pipeline-trained models achieve AUC 1.0 -- higher than the exploration-phase models because the pipeline applies the full 664 Gold-layer transformations and trains on the temporal feature matrix with holdout masking. The recommendations hash (`742edc35`) ensures version traceability between the generated pipeline and the exploration findings that produced it.

---

### Scoring Validation -- Production Reality Check (NB11)

Cross-validation tells us how well our model generalizes to *similar* data. But production data comes from the *future*. The scoring pipeline tests our models on a **point-in-time holdout** (the 10% of entities masked in the Silver layer), simulating true deployment conditions.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/11_scoring_validation.html)

#### Holdout Scoring Results

| Metric | Value |
|--------|-------|
| **Holdout Records** | 499 (10% of data) |
| **Overall Correct** | 496 / 499 (**99.4% accuracy**) |
| **Misclassified** | 3 (all false negatives) |
| **ROC-AUC** | 1.0000 |

All three models perform identically on the holdout -- the features carry such strong signal that model choice is irrelevant. The 3 misclassified records are false negatives (churned customers predicted as retained).

#### Adversarial Pipeline Validation

The scoring pipeline produces identical features to training -- scaler re-fitting, encoder inconsistencies, and feature ordering differences are all checked and passed. The pipeline is consistent.

#### Feature Importance (SHAP Analysis)

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | `event_count_365d × event_count_all_time` | 0.004375 |
| 2 | `unsubscribe_date_is_weekend_count_all_time` | 0.003182 |
| 3 | `campaign_type_nunique_all_time` | 0.000694 |

The SHAP analysis reveals that interaction features and diversity metrics contribute alongside recency -- the full 664 Gold-layer transformations create feature combinations that the exploration-phase models didn't access.

---

## Key Lessons

Looking back across the analysis, a clear arc emerges. We started by defining an intent contract (NB00) that set the prediction objective, temporal posture, and snapshot grid. Then we examined 83,000 email events -- raw data that couldn't be modeled directly because each customer had many events and the data was organized by *what happened*, not by *who it happened to*. Phase I (NB01--NB02) was entirely about understanding the data well enough to make the aggregation decisions: which time windows to use (driven by the 95-day median cadence from NB01a), which derived features to create (driven by the recency and momentum signals from NB01c), how to handle the heterogeneity across customer segments (driven by NB01a's eta-squared analysis), and which datetime columns to exclude from derivation (unsubscribe_date as a target proxy, detected in NB01d).

The aggregation step (NB01d) was the pivot point -- the transformation from events to entities that every downstream analysis depended on. Phase II (NB03--NB05) merged datasets onto the temporal spine and progressively validated that the aggregated features carried the signal we expected. Phase III (NB06--NB11) confirmed it through modeling, business alignment, and production pipeline generation. The final models achieved AUC >0.97 regardless of algorithm, which tells us the most important lesson: **aggregation choices mattered more than model choices**.

### Core Principles

**Intent drives everything.** The intent contract in NB00 -- prediction objective, temporal posture, prediction horizon, cadence -- propagates through all 14 notebooks. The snapshot grid (404 weekly dates), purge gap (104 days), and observation window (270 days) are all derived formulaically from the intent, not chosen ad-hoc.

**Aggregation is the feature engineering.** In entity-level data, features come pre-computed. In event-level data, the aggregation step *is* the feature engineering. The choice of windows (180d, 365d, all_time from NB01a), functions (sum, mean, max, count), and derived features (recency from NB01c, lifecycle from NB01a, momentum from NB01c) determines what signal the model can access. NB01d consolidated all of these into 217 features -- and the model simply learned from what that step provided.

**Window selection drives everything.** The 95-day median inter-event gap from NB01a was the single most consequential number in the analysis. It eliminated short windows (7d, 30d, 90d), determined that 180d was the tightest viable recency signal, explained the high null rates in aggregated features, the zero-inflation in NB04, and the feature importance rankings in NB08 (365d features dominated because they captured roughly four engagement cycles).

**Absence is signal.** A majority of customers had no 180-day activity, and that null pattern is highly predictive (NB01d--NB05). `time_to_open_hours` was 77.6% missing because most emails were never opened -- and the missingness encodes engagement quality more directly than any imputed value could (NB01b). Event-based pipelines must preserve missingness as information rather than imputing it away.

**Event-level targets mislead.** The raw 97.3:2.7 split (NB01) was misleading -- most individual emails don't trigger unsubscription. After entity-level aggregation via `max` (NB01c), the true distribution was 55.4:44.6 -- a nearly balanced dataset that only needed `class_weight='balanced'` (NB07), not the extreme imbalance handling the event-level view would have suggested.

**Recency dominates -- and that's both a strength and a risk.** `days_since_last_event` achieved Cohen's d of 2.23 at the event level (NB01c), top feature importance in the model (NB08), and consistent dominance through production scoring (NB11). This consistency across phases builds confidence that the signal is real. But over-reliance on a single feature creates fragility -- if recency patterns shift (new campaign cadences, seasonal changes), the model degrades rapidly. The caution from NB01c about trailing vs. leading indicators remains unresolved.

**Source cleanup before merge.** Per the new process flow, NB02 runs *per dataset* before the Silver merge (NB03). This eliminates broken columns early, avoids provenance complexity in the merged space, and ensures each dataset enters the merge clean.

### What Wasn't Explored (Future Work)

**Model Improvements:**
1. **Hyperparameter tuning** -- defaults achieved AUC 0.9708; tuning could push higher
2. **Feature selection** -- 101 redundant features could be removed; test impact
3. **Model without recency** -- assess performance when removing `days_since_last_event`
4. **Segment-specific thresholds** -- different probability thresholds per lifecycle quadrant

**Temporal Enhancements:**
5. **Shorter observation windows with activity flags** -- binary "any activity in 30d" may capture recency without sparse aggregation
6. **Sequential modeling** -- RNN/LSTM on raw event sequences instead of aggregation
7. **Campaign response features** -- aggregate by campaign type for campaign-specific engagement rates
8. **Consensus grid with ALLOW_ADJUSTMENTS** -- let temporal notebooks vote on grid parameters in multi-dataset scenarios

**Production Readiness:**
9. **Walk-forward validation** -- simulate true temporal deployment
10. **Drift monitoring dashboard** -- track `days_since_last_event` distribution over time
11. **A/B testing framework** -- measure intervention effectiveness
12. **Databricks track** -- deploy with DLT + Unity Catalog + Feature Store via `DatabricksPipelineGenerator`

---

## Running the Tutorial

```bash
# Clone and install
git clone https://github.com/aladjov/CR.git
cd CR
pip install -e ".[dev,ml]"

# Option 1: Run all notebooks sequentially (recommended)
python scripts/notebooks/run_exploration.py

# Option 2: Run with a dry run first to see which notebooks will execute
python scripts/notebooks/run_exploration.py --dry-run

# Option 3: Open notebooks interactively
jupyter lab exploration_notebooks/00_start_here.ipynb
```

Set `datasets = {"customer_emails": "../tests/fixtures/customer_emails.csv"}` in the intent contract notebook (NB00).

**Automated execution:** `run_exploration.py` runs all notebooks in the correct order with smart skip logic -- temporal notebooks (01a--01d) are skipped when no event-level data is detected, and text notebooks (01a_a, 04a) are skipped when no TEXT columns exist. For this dataset, 17 of 19 notebooks execute (the two text-specific notebooks are skipped).

**Running the generated pipeline:**
```bash
# After completing exploration notebooks, run the generated pipeline
python generated_pipelines/local/customer_churn/run_all.py

# Run scoring on holdout data
python generated_pipelines/local/customer_churn/scoring/run_scoring.py
```

**Experiments directory:** After execution, all findings, artifacts, and data are stored in `experiments/`. The run namespace organizes data into `runs/{run_id}/datasets/{name}/findings/`, `data/bronze/`, `data/silver/`, and `merged/` directories. For a full walkthrough of the directory structure, see the [[Architecture]] page.

---

## Next Steps

- [[Tutorial-Retail-Churn]] - Compare with the entity-level retail tutorial
- [[Architecture]] - Understand the intent-driven medallion architecture
- [[Snapshot Grid and Control Variables]] - Leakage-safe temporal grid and control variables
- [[Local Track]] - How generated pipelines work (Feast + MLflow)
- [[Databricks Track]] - Deploy to Databricks with Unity Catalog
