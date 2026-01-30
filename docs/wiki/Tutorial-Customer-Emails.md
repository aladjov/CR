# Tutorial: Customer Email Engagement & Retention

This tutorial demonstrates a complete customer retention ML pipeline using an **event-based email engagement** dataset. Unlike the [[Tutorial-Retail-Churn|retail tutorial]] which works with entity-level data (one row per customer), this tutorial shows how the framework handles **event-level data** -- where each row is an email event and multiple rows belong to a single customer. The focus is on *why* aggregation and temporal analysis drive every downstream decision.

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

**The key thing to understand upfront:** This is event-level data. In entity-level datasets, each customer is already a single row with pre-computed features, and the analysis proceeds directly to column profiling. Here, the framework must first *detect* that rows represent events, then *aggregate* them into customer-level features -- choosing the right time windows, summary statistics, and derived metrics. This aggregation step (Stage 5) is the critical decision point that doesn't exist in entity-level pipelines. Everything before it is preparation; everything after depends on it.

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | `tests/fixtures/customer_emails.csv` |
| **Total Events** | 83,198 email interactions |
| **Customers** | 4,998 unique entities |
| **Features** | 13 columns (event-level) |
| **Target** | `unsubscribed` (binary: 0=retained, 1=churned) |
| **Retention Rate** | 60.7% retained / 39.3% churned (after aggregation) |
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

## Stage 1: Data Discovery -- First Look at the Data

Before searching for patterns, we need to understand what we're working with. Is the data organized by customer (entity-level) or by event (event-level)? Getting this wrong -- treating events as entities or vice versa -- would either lose temporal information or produce nonsensical features. The first job is classification, not analysis.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01_data_discovery.html)

### What the Framework Detects

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
| **Cutoff Date** | 2022-09-26 | 90% train / 10% score split |
| **Training Events** | 74,842 | Events before cutoff |
| **Snapshot ID** | `training_v2` | Versioned for reproducibility |
| **Event-Level Target** | 97.4% class 0, 2.6% class 1 | Per-event distribution (misleading -- see Stage 4) |

This last number -- 97.4% vs 2.6% -- is misleading. It looks like almost no one churns. But that's the *event-level* view: most individual emails don't trigger an unsubscription. The *customer-level* picture is dramatically different, as we'll discover in Stage 4.

### Timestamp Coalescing and the Cutoff

**Timestamp coalescing** is a subtle but important operation. The framework does not simply pick a single datetime column. Instead, it builds a **coalesced timestamp** (`last_action_date`) by analyzing all available datetime columns (`sent_date`, `unsubscribe_date`), ordering them chronologically by median date, and resolving each row's timestamp from latest to earliest. This is essential for event-level data: `unsubscribe_date` is a single event per customer and only populated for the small percentage who actually unsubscribed, while `sent_date` covers every row. Coalescing guarantees 100% coverage -- every row gets a resolved timestamp regardless of which columns happen to be null.

The **cutoff date** is then derived from this coalesced timestamp by finding the date that achieves the requested train/score ratio (here, 90/10). In this dataset, the 90th-percentile date is 2022-09-26 -- all events before this date become training data, all events after become the scoring window.

The `label_window_days` parameter (180 days) defines the **inactivity period** after which a customer is considered churned. When the dataset lacks an explicit outcome column (contract expiration, policy termination, account closure), a customer with no events in the 180-day window following their last activity is labeled as churned. Here, `unsubscribe_date` exists but covers only ~2.6% of events, so the coalesced approach combined with the label window provides the most robust temporal anchoring.

### The 90/10 + Holdout Strategy

The cutoff date creates a **90% train / 10% score** split at the event level. Within the training portion, the pipeline further holds out ~10% of *entities* for scoring validation (with their target values masked in the silver layer). This two-level split approximates a standard 80/20 train/test split but with an important benefit: the scoring holdout records flow through the *same* Bronze-Silver-Gold pipeline as training data. This means the scoring pipeline validates not just model accuracy but also the correctness of the entire data processing chain -- any bug in feature engineering, encoding, or scaling would show up as degraded holdout performance. We'll see this pay off in Stage 11, when the production pipeline reproduces the exploration-phase results almost exactly.

### What We Take Forward

- **Activated temporal track** (notebooks 01a-01d) for event aggregation -- this is the path unique to event-level data
- **Cutoff date 2022-09-26** derived from the 90th percentile of the coalesced `last_action_date` -- this date anchors every windowed feature in Stage 5
- **Holdout: ~10% of entities** masked in silver layer for end-to-end pipeline validation -- our final test in Stage 11
- **Coalesced timestamps** used because no single datetime column has full coverage
- The misleading event-level target (97.4:2.6) will be corrected in Stage 4 after aggregation reveals the true 60.5:39.5 split
- All downstream notebooks load from the versioned snapshot

### Alternative Approaches

- **Entity-level path**: If the framework detected one row per customer, it would skip the temporal track entirely (as in the retail tutorial)
- **Manual cutoff override**: Useful if a business event (campaign launch, policy change) should define the boundary
- **Single timestamp column**: Could use `sent_date` alone (100% coverage) but would ignore `unsubscribe_date` signal; coalescing preserves both
- **Different inactivity windows**: 90 days (aggressive -- flags customers as churned sooner) or 365 days (conservative) -- matters most when churn is inferred from absence rather than an explicit event
- **Simple random 80/20 split**: Would not test the scoring pipeline's data processing -- only the model itself

---

## Stage 2: Temporal Deep Dive -- Understanding the Rhythm of Engagement

With the data classified as event-level, the next question is deceptively simple: *If we're going to summarize each customer's email history into fixed time windows, which windows should we use?*

The wrong answer wastes effort (windows too short produce mostly zeros) or loses signal (windows too long blur recent behavior into historical noise). The framework evaluates each candidate through a multi-gate scoring process that prevents any single metric from dominating the decision.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01a_temporal_deep_dive.html)

### The Central Question

The temporal deep dive asks about each candidate time window (24h, 7d, 14d, 30d, 90d, 180d, 365d, all_time): **"Do we have enough data in this window for meaningful analysis?"**

"Enough data" is not a single test. A window could pass one criterion and fail another -- a 7-day window might cover the full dataset span but contain zero events for most customers. The framework evaluates each window through multiple gates, scores them independently, then aggregates those scores into a unified recommendation.

### Time Series Profiling

**Why we do this:** Before evaluating windows, we need to understand the basic rhythm of the data -- how many events per customer, how spread out, how long the history spans. These metrics determine which windows are even plausible.

**What the analysis shows:**

| Metric | Value |
|--------|-------|
| Time Span | 2,825 days (7.7 years) |
| Median Inter-Event Gap | 93 days |
| Mean Inter-Event Gap | 143 days (right-skewed) |
| Volume Trend | Declining (-28%) |
| Data Gaps | 0 detected |

The median gap of 93 days is the single most important number from this stage. It means the typical customer interacts roughly once a quarter -- and any window shorter than that will be empty for most people. This observation will recur throughout the analysis: it explains the window selection here, the 57.7% null rate in Stage 5, and the zero-inflation in Stage 6.

**Auto-derived recommendation:** Window candidates depend on these metrics -- only windows significantly larger than the median gap will capture enough events for stable aggregations.

**Alternative technique:** Kernel density estimation of inter-event times could provide a smoother picture, but the median is sufficient for window selection decisions.

### Events per Entity Distribution

**Why we do this:** We need to identify power users vs one-timers, because sparse customers need different feature treatment than prolific ones.

**What the analysis shows:** The framework segments customers by event count. One-time customers (single event) cannot contribute to any temporal feature -- no trend, no variance, no momentum is computable from one data point. High-activity customers risk dominating training with their volume.

**Auto-derived recommendation:** Segment-aware window evaluation -- assess window viability separately for customer segments rather than globally.

**Alternative technique:** Quantile-based segmentation (e.g., deciles of event count) instead of the lifecycle quadrant approach.

### Entity Lifecycle Analysis

**Why we do this:** Customers vary in both *how long* they've been around (tenure) and *how intensely* they engage (event frequency). Crossing these dimensions reveals whether one model fits all.

**What the analysis shows:** Four quadrants emerge with very different behavioral profiles. The framework computes **eta-squared** = 0.308 (high heterogeneity), meaning lifecycle segment explains 30.8% of the variance in engagement patterns.

**Auto-derived recommendation:** Add `lifecycle_quadrant` as a feature. The eta-squared value is well above the 0.14 threshold for "high heterogeneity." This segmentation feature will prove its worth repeatedly: it explains the false outlier problem in Stage 7 (Cramer's V=0.665 in Stage 8) and drives a 70+ percentage-point churn spread in Stage 5.

**Alternative technique:** Build separate models per quadrant. Rejected because the smallest segment has EPV=1.0 (far below the minimum of 10), as Stage 9 confirms.

### Temporal Coverage

**Why we do this:** Data gaps, volume trends, and arrival patterns can corrupt aggregations if not detected. A six-month gap in email sends would produce misleading rolling features -- zeros that mean "no data" rather than "no activity."

**What the analysis shows:** No gaps detected. Volume is declining (-28%). Customer arrivals are concentrated in 2015 (90% of onboarding).

**Auto-derived recommendation:** Flag drift risk from the declining volume. Skip cohort features -- insufficient variation with 90% of customers sharing the same arrival year.

**Alternative technique:** Cohort analysis (monthly or quarterly) if arrivals were more distributed.

### Inter-Event Timing

**Why we do this:** The gap between events determines which windows will produce meaningful aggregations. If the typical gap is 93 days, a 30-day window captures at most one event for most customers -- not enough to compute variance, trends, or ratios.

**What the analysis shows:** The median 93-day gap means the 90-day window is *timing-aligned* (same order of magnitude) but still borderline for density. The 180-day window is the minimum that clears the 2-event density threshold.

**Auto-derived recommendation:** 180d, 365d, all_time as the window set.

**Alternative technique:** Fixed-count windows (last N events per customer) instead of fixed-time windows. Useful when time windows produce too many zeros; not needed here since 180d passes.

### The Multi-Gate Scoring Process

Each candidate window passes through a sequence of gates -- increasingly specific questions:

**Gate 1 -- Dataset Span (Hard Gate):** *Is the dataset long enough to even consider this window?*

The dataset's total time span must be at least **2x the window size**. A 365-day window needs at least 730 days of data. This is a hard gate -- failing it excludes the window immediately (unless it's in the always-include list, which defaults to `all_time`). The 2x multiplier exists because a window needs to be observable across multiple periods to produce stable aggregations, not just appear once.

**Gate 2 -- Entity Duration Adequacy:** *Does each customer's individual history span at least the window duration?*

Even if the dataset covers 7 years, individual customers may have been active for only 3 months. The framework checks each entity's `duration_days` against the window size. A customer with 60 days of history cannot contribute meaningfully to a 180-day window.

**Gate 3 -- Event Density:** *Would each customer have at least 2 expected events in the window?*

This is the core density test. For each customer, the framework projects: given their total event count and total active duration, how many events would we expect in a window of this size? The formula is `event_count * (window_days / duration_days)`. A threshold of **>=2 expected events** separates meaningful aggregation from noise -- you cannot compute a meaningful trend, variance, or change rate from a single event.

**Gate 4 -- Coverage Threshold:** *What fraction of entities pass both Gates 2 and 3?*

A window is only useful if a sufficient proportion of the population can contribute to it. The default threshold is **10%** -- at least 10% of entities must have both adequate duration and sufficient event density. Below this, the window produces too many nulls to be a reliable feature.

The framework also tracks `meaningful_pct` -- among entities that pass the duration gate, what fraction also has sufficient density? This separates "the window is too long for most customers" from "the window is fine in length but events are too sparse."

### Annotation Layers

After gating, the framework enriches each window with contextual annotations. These don't change inclusion decisions but inform interpretation:

**Segment Relevance:** Which lifecycle segments benefit most from each window? The framework checks coverage per quadrant (e.g., "Steady & Loyal" vs "One-shot") and flags which segments contribute >=15% of the beneficial population for a given window. This tells you whether a window captures behavior broadly or is dominated by one customer type.

**Seasonality Alignment:** If temporal pattern analysis (Stage 4) detects seasonal periods, the framework checks whether any candidate window aligns with those periods. A window that matches a seasonal cycle captures one full cycle of variation -- useful for trend features.

**Inter-Event Timing:** The framework compares the median inter-event gap to each window size. When the ratio falls between 0.5 and 2.0, the window is "timing-aligned" -- it's in the same order of magnitude as how frequently customers naturally interact. Such windows tend to produce the most informative aggregations.

### Heterogeneity Analysis

**Why we do this:** Do different customer segments behave so differently that a single model might be inadequate?

The framework computes **eta-squared** (the ratio of between-group variance to total variance) for engagement intensity and event count across lifecycle quadrants:

| Eta-squared | Level | Advisory |
|-------------|-------|----------|
| < 0.06 | Low | Single model is fine -- union windows lose minimal signal |
| 0.06 -- 0.14 | Moderate | Add lifecycle quadrant as a feature -- let the model learn segment differences |
| > 0.14 | High | Consider separate models for distinct segments, especially if cold-start population (One-shot / One-time) exceeds 30% |

**What the analysis shows:** Eta-squared = 0.308 (high), with the One-shot quadrant at 17.3% -- below the 30% cold-start threshold. The advisory: `consider_segment_feature` -- add lifecycle quadrant as a categorical feature rather than building separate models.

### Gate Results for This Dataset

| Window | Coverage | Density (events/entity) | Result |
|--------|----------|-------------------------|--------|
| 7d | <10% | ~0.1 | Excluded -- fails coverage |
| 30d | <10% | ~0.3 | Excluded -- fails coverage |
| 90d | Borderline | ~1.0 | Excluded -- density below 2-event threshold |
| 180d | Passes | ~1.9 | Included -- near threshold but above coverage gate |
| 365d | Passes | ~3.9 | Included -- adequate density |
| All time | 100% | ~15.0 | Always included |

The median inter-event gap of 93 days means the 90-day window is timing-aligned but still fails the 2-event density test for most customers. The 180-day window is the shortest window that clears all gates -- it's the tightest recency signal the data can support.

### Decision Made
- **Three aggregation windows**: `180d`, `365d`, `all_time` -- the minimum set that passed all gates while preserving recency-vs-history contrast
- **Lifecycle quadrant** added as a categorical feature (high heterogeneity, eta-squared=0.308)
- **`time_to_open_hours` flagged** for Yeo-Johnson transform (skewness 2.08)

### Alternative Approaches
- **Shorter windows (7d, 30d)**: Failed coverage and density gates -- most customers would have zero events
- **Single window**: Would lose the recency-vs-history contrast that 180d vs all_time provides
- **Fixed-count windows** (last N events): Alternative approach when time windows produce too many zeros; not needed here since 180d passes
- **Separate models per quadrant**: Heterogeneity is high, but cold-start fraction (17.3%) is below the 30% threshold -- a segment feature is more pragmatic

> **Caution:** The declining volume trend (-28%) means recent windows contain less data per customer than historical ones. Features from 180-day windows will have 57.7% null values -- customers with no recent activity. This is informative (absence of activity is a signal), but models must handle the missingness.

---

## Stage 3: Temporal Quality -- Validating Event Data

Aggregating events without first validating their quality risks propagating data issues into features. Event-level data can contain duplicates, temporal gaps, future-dated records, and ordering inconsistencies that would silently corrupt aggregated features. Catching these *before* aggregation is essential -- once events are summarized, the original issues become invisible.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01b_temporal_quality.html)

### Duplicate Events (TQ001)

**Why we check this:** Duplicates inflate counts and skew aggregation statistics. A customer who opened 5 emails but has 6 duplicate records would appear 20% more active than reality.

**What the analysis shows:** 371 duplicates found (0.50% of events).

**Auto-derived recommendation:** Deduplicate before aggregation. At 0.5%, the impact is small, but cleaning is cheap and prevents double-counting.

**Alternative technique:** Ignore at this threshold -- negligible impact on aggregation. But since deduplication costs nothing, there's no reason to leave dirty data in the pipeline.

### Temporal Gaps (TQ002)

**Why we check this:** Gaps in event data produce misleading rolling features -- zeros that mean "no data was collected" rather than "the customer was inactive." A three-month gap in email sends (system outage, campaign pause) would make every customer appear to have gone silent.

**What the analysis shows:** No significant gaps detected. The event stream is continuous over the 9-year span.

**Auto-derived recommendation:** Proceed with confidence -- no gap indicators needed.

**Alternative technique:** If gaps existed, add binary gap indicator features (e.g., `had_gap_in_180d`) so models can distinguish true inactivity from data absence.

### Future Dates (TQ003)

**Why we check this:** Future-dated events cause data leakage -- the model would see tomorrow's data during training, producing artificially inflated performance that collapses in production.

**What the analysis shows:** None found. All events have valid historical timestamps.

**Auto-derived recommendation:** Proceed.

### Event Ordering (TQ004)

**Why we check this:** When multiple events share the same timestamp, sequence features become ambiguous. If two emails were "sent" at the same second, which came first?

**What the analysis shows:** 371 events with ambiguous ordering (same timestamp, different events).

**Auto-derived recommendation:** Use stable sort. This is a minor concern -- aggregation functions (sum, mean, count) are order-invariant, so ordering ambiguity only affects sequence-dependent features we're not computing here.

### Quality Score

**Why we compute this:** A unified metric to decide proceed/investigate/stop, rather than weighing individual checks subjectively.

**What the analysis shows:**

| Component | Score | Issues |
|-----------|-------|--------|
| Duplicate Events | 23.1/25 | 371 duplicates |
| Temporal Gaps | 25.0/25 | None |
| Future Dates | 25.0/25 | None |
| Event Ordering | 23.1/25 | 371 ambiguous |
| **Total** | **96/100 (Grade A)** | |

Grade thresholds: A (90+) proceed with confidence, B (75-89) document issues and proceed, C (60-74) fix issues first, D (<60) investigate data source.

### Missing Value Patterns

**Why we examine this:** We need to understand whether missingness is random (safe to impute) or informative (preserve as signal). This distinction will prove critical throughout the analysis.

**What the analysis shows:**
- **`time_to_open_hours` at 77.7% missing** -- this is MNAR (Missing Not At Random). The column is only populated when an email is opened. The 77.7% of missing values *are* the data: they represent emails that were never opened. Imputing these with a mean or median would destroy the signal.
- **`unsubscribe_date` at 97.4% missing** -- also MNAR. Only populated for churned customers. The missingness pattern directly encodes the target variable.

**Auto-derived recommendation:** Preserve both patterns. Do not impute. These will flow through to Stage 5's aggregation as informative nulls, and we'll see them again in Stage 6 as zero-inflation and in Stage 7 as missing value patterns.

**Alternative technique:** Drop high-missingness columns entirely. This would simplify the pipeline but lose genuine engagement signal from `time_to_open_hours`.

### Segment-Aware Outlier Recommendation

**Why we compute this:** Global outlier detection fails when data has natural subgroups. A value that looks extreme across all customers may be perfectly normal for a specific segment.

**What the analysis shows:** The framework recommends segment-aware outlier treatment, with false outlier rates up to 99.6% for some features under global detection. This is a preview of a critical finding that will surface in Stage 7.

**Auto-derived recommendation:** Carry forward to Stage 7 quality assessment. Do not apply global outlier caps.

### Decision Made
- **371 duplicate events removed** during aggregation (0.5% of data)
- **`time_to_open_hours`**: Retained despite 77.7% missingness -- the pattern of missingness itself is informative (unopened emails)
- **Segment-aware outlier treatment** recommended over global treatment (false outlier rate up to 99.6% for some features)

### Alternative Approaches
- **Drop `time_to_open_hours` entirely**: Its 77.7% missingness could justify removal, but the non-missing values carry signal (time-to-open correlates with engagement quality)
- **Strict deduplication policy**: Could remove duplicates before snapshot creation rather than during aggregation
- **Ignore duplicates**: At 0.5%, the impact is negligible -- but cleaning is cheap and prevents double-counting during aggregation

---

## Stage 4: Temporal Patterns -- Discovering Behavioral Signals

This is where key insights emerge. We've understood the data structure (Stage 1), its rhythm (Stage 2), and verified its integrity (Stage 3). Now we look for the behavioral signatures that separate customers who stay from those who leave. Not all temporal features are created equal -- before we aggregate, we need to know which patterns carry signal.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01c_temporal_patterns.html)

### Target Resolution: Correcting the Class Balance

**Why we do this:** The event-level target distribution (97.4% retained, 2.6% churned) is misleading because one customer contributes many events. We need to see the picture at the granularity where we'll actually model.

**What the analysis shows:** After aggregating via `max` per customer:
- Event-level: 97.4% class 0, 2.6% class 1 (misleading)
- Entity-level: **60.5% retained, 39.5% churned**

This changes our entire understanding of class balance. What looked like an extreme imbalance problem (97:3) is actually a manageable one (60:40). The event-level view was dominated by prolific customers who generated many email events but only one churn label.

**Auto-derived recommendation:** Always verify target distribution at the modeling granularity, not the raw data granularity.

**Alternative technique:** Use `any` instead of `max` for aggregation -- equivalent for binary targets, but `max` generalizes to ordinal targets.

### Trend Detection

**Why we do this:** Strong trends in engagement volume cause data leakage if not handled. A model could learn *time itself* rather than *behavior* -- if email volume steadily increases over years, the model might use recency as a proxy for volume.

**What the analysis shows:** STABLE direction (R-squared=0.465, medium confidence). The trend exists but is not strong enough to warrant special treatment.

**Auto-derived recommendation:** Skip trend features. R-squared below the action threshold.

**Alternative technique:** If the trend were strong (R²>0.7), add trend slope as a feature or detrend the data before aggregation.

### Seasonality Detection

**Why we do this:** Periodic patterns suggest cyclical features. Windows aligned to seasonal cycles capture a full period of variation, producing more stable aggregations.

**What the analysis shows:**

| Pattern | Variation | Autocorrelation |
|---------|-----------|-----------------|
| Day of Week | 1.4% | -- |
| Monthly | 8.7% | -- |
| Quarterly | 5.0% | -- |
| Yearly | **59.9%** | -- |
| Weekly (7d) cycle | -- | 0.484 (Moderate) |

The yearly pattern captures the most variation (59.9%), and the weekly cycle shows moderate autocorrelation (0.484). Day-of-week variation is small (1.4%) but still worth encoding because cyclical features are cheap to compute and can capture send-time preferences.

**Auto-derived recommendation:** Add `dow_sin`, `dow_cos` cyclical encoding -- captures the weekly pattern without creating 7 dummy variables.

**Alternative technique:** Month dummies or seasonal decomposition (STL). Overkill for weak patterns, and dummies increase dimensionality unnecessarily.

### Recency Analysis: The Strongest Signal

**Why we do this:** How recently a customer engaged is often the single strongest churn predictor. A customer who hasn't opened an email in two years tells a very different story than one who clicked yesterday.

**What the analysis shows:**

| Metric | Value |
|--------|-------|
| Median recency | 246 days |
| Target correlation | **0.772** (strong) |
| Cohen's d | **+2.23** (large effect) |
| Retained mean recency | 1,399 days |
| Churned mean recency | 165 days |

Churned customers were active **1,234 days more recently** than retained ones. This is the strongest signal we've found -- and it's counterintuitive at first. Why would *churned* customers have more *recent* activity?

The explanation: customers who unsubscribe do so shortly after receiving emails (triggering recent activity). The "retained" customers haven't interacted in years -- they simply never unsubscribed. The model is detecting "recently engaged then left" rather than "gradually disengaged."

**Auto-derived recommendation:** `days_since_last_event` as the highest-priority derived feature, plus recency buckets for the categorical version.

**Alternative technique:** Survival analysis (time-to-event modeling) would handle the temporal nature of recency more naturally, but the framework's aggregation approach captures the core signal.

> **Caution on Causality:** The strong recency signal could be a **trailing indicator**, not a leading one. Customers who unsubscribe do so shortly after receiving emails (triggering recent activity). The model may be detecting "about to unsubscribe" rather than "at risk of disengaging." This distinction matters for intervention timing -- by the time recency flags a customer, it may already be too late.

### Effect Sizes (Cohen's d)

**Why we do this:** Cohen's d quantifies how well each feature discriminates between retained and churned customers. It's effect-size based (not p-value based), so it doesn't inflate with sample size.

**What the analysis shows:**

| Feature | Cohen's d | Interpretation |
|---------|-----------|----------------|
| `tenure_days` | -2.403 | Churned customers have much longer tenure |
| `opened_std` | -0.988 | Churned have more variable open behavior |
| `opened_sum` | -0.915 | Churned opened more emails total |
| `opened_mean` | -0.834 | Churned have higher open rate |
| `event_count` | -0.759 | Churned received more emails |
| `clicked_sum` | -0.630 | Churned clicked more |

The negative Cohen's d values reveal a counterintuitive pattern: churned customers were *more engaged*, not less. They opened more emails, clicked more links, had higher open rates. This makes sense in context -- these are customers who actively engaged with the company's emails and then consciously decided to unsubscribe. The truly disengaged customers never bothered to unsubscribe; they just stopped opening.

**Auto-derived recommendation:** Prioritize high-|d| features for inclusion in aggregation. The sign of d is as informative as the magnitude.

### Velocity and Acceleration

**Why we do this:** Rate of change captures dynamics that static aggregations miss. A customer whose click rate is *declining* tells a different story than one whose click rate has been *consistently low*.

**What the analysis shows:** Clicked momentum (d=-0.97) and opened momentum (d=1.01) are both strong signals -- the rate of change in engagement discriminates nearly as well as the level of engagement itself.

**Auto-derived recommendation:** Add `clicked_momentum_180_365` as a ratio feature (recent activity / historical activity). Values >1.0 indicate increasing engagement; <1.0 indicate declining.

**Alternative technique:** Exponentially weighted moving averages (EWMA) for a smoother picture of engagement trends.

### Momentum Analysis

**Why we do this:** Momentum compares recent-to-historical activity as a behavioral change signal. It's the derivative of engagement -- are customers speeding up or slowing down?

**What the analysis shows:** Momentum ratios using the 180d/365d window pair from Stage 2 produce strong discrimination. The specific window pair was chosen because both windows passed the multi-gate scoring -- a direct connection from Stage 2's analysis to Stage 4's feature engineering.

**Auto-derived recommendation:** Use the window pairs validated in Stage 2 for momentum computation.

### Information Value and KS Statistics

**Why we do this:** Information Value (IV) and KS statistics measure predictive strength differently from Cohen's d -- they handle non-linearity and distributional differences that d may miss.

**What the analysis shows:** Rankings validate Cohen's d findings. Recency and engagement features dominate by all measures.

**Auto-derived recommendation:** Confirms feature selection -- no hidden features missed by Cohen's d.

### Categorical Analysis (Cramer's V)

**Why we do this:** Cramer's V is the effect size measure for categorical features, analogous to Cohen's d for numeric ones.

**What the analysis shows:** `lifecycle_quadrant` already shows strong association with the target. This validates the segmentation decision from Stage 2.

**Auto-derived recommendation:** Include `lifecycle_quadrant` in the model.

### Feature Engineering Summary

**Why we consolidate this:** All pattern-derived feature recommendations need to be assembled into a coherent configuration for Stage 5's aggregation.

**What the analysis recommends:**
- **Recency features** (highest priority): `days_since_last_event`, recency buckets
- **Seasonality encoding**: `dow_sin`, `dow_cos`
- **Momentum**: `clicked_momentum_180_365`
- **Skip**: trend features (R² too low), cohort features (insufficient variation)

These recommendations directly configure Stage 5's aggregation -- every derived feature has a traceable origin in Stage 2 or Stage 4.

### Decision Made
- **Recency features**: High priority -- add `days_since_last_event`, recency buckets
- **Seasonality features**: Add `dow_sin`, `dow_cos` cyclical encoding
- **Momentum features**: Add `clicked_momentum_180_365` ratio
- **Trend features**: Skipped (R-squared too low)
- **Cohort features**: Skipped (90% of customers onboarded in 2015 -- insufficient variation)

---

## Stage 5: Event Aggregation -- Building Customer Profiles

Everything converges here. The windows from Stage 2, the quality guarantees from Stage 3, the feature recommendations from Stage 4 -- all of them feed into this single transformation. We're converting 74,000+ individual email events into 4,998 customer profiles, each described by 72 features. This is the irreversible step: poor aggregation loses signal, over-aggregation creates redundancy, and the choices made here cascade through every downstream analysis.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/01d_event_aggregation.html)

### How Findings Inform Aggregation

This is not ad-hoc feature engineering. Every parameter traces to a prior analysis:

| Source | Insight | Application in Aggregation |
|--------|---------|---------------------------|
| Stage 1 | Cutoff date 2022-09-26 | Reference date for all windows |
| Stage 2 | 93-day median gap → 180d/365d/all_time | Window selection |
| Stage 2 | Eta-squared=0.308 | Add lifecycle_quadrant feature |
| Stage 3 | 371 duplicates, 96/100 quality | Deduplicate before aggregation |
| Stage 3 | time_to_open_hours 77.7% MNAR | Preserve nulls, don't impute |
| Stage 4 | Recency d=2.23 | Create days_since_last_event |
| Stage 4 | Weekly autocorrelation 0.484 | Add dow_sin/cos |
| Stage 4 | Clicked momentum d=-0.97 | Add clicked_momentum_180_365 |

### The Shape Transformation

**Why we do this:** The fundamental operation -- many-rows-per-entity to one-row-per-entity. This is the step that doesn't exist in entity-level pipelines and determines what signal the model can access.

**What the analysis shows:**

| Metric | Value |
|--------|-------|
| **Input events** | 74,471 (after dedup) |
| **Output entities** | 4,998 |
| **Features created** | 72 |
| **Memory** | 3.5 MB |
| **Completeness** | 84.1% |
| **Target distribution** | 60.7% retained, 39.3% churned |

The math: 5 value columns (`opened`, `clicked`, `send_hour`, `bounced`, `time_to_open_hours`) × 3 windows × 4 aggregation functions (sum, mean, max, count) = 60 features. Plus 12 derived features = 72 total.

### Derived Feature Rationale

Each derived feature exists because a prior stage identified it:

| Derived Feature | Source | Rationale |
|----------------|--------|-----------|
| `days_since_last_event` | Stage 4 recency analysis (d=2.23) | Strongest predictor found |
| `days_since_first_event` | Stage 2 lifecycle analysis | Tenure component of customer profile |
| `lifecycle_quadrant` | Stage 2 heterogeneity (eta-squared=0.308) | Segmentation feature for high-heterogeneity data |
| `dow_sin`, `dow_cos` | Stage 4 seasonality (weekly autocorrelation 0.484) | Cyclical encoding of day-of-week patterns |
| `clicked_momentum_180_365` | Stage 4 momentum analysis (d=-0.97) | Engagement trend signal |
| `recency_bucket` | Stage 4 recency analysis | Categorical version of recency for tree splits |
| `event_count_180d/365d/all_time` | Stage 2 window selection | Activity volume per window |

### Lifecycle Quadrant vs. Churn Rate

**Why we validate this:** The segmentation decision from Stage 2 (eta-squared=0.308) needs to be confirmed with actual churn rates. High heterogeneity in engagement patterns should translate to high heterogeneity in outcomes.

**What the analysis shows:**

| Quadrant | Customers | Churn Rate | Interpretation |
|----------|-----------|------------|----------------|
| Intense & Brief | 1,627 | **77.7%** | High engagement, short tenure -- likely to churn |
| One-shot | 867 | **59.1%** | Minimal engagement, short tenure -- expected churn |
| Occasional & Loyal | 1,632 | 7.8% | Sparse but persistent -- low risk |
| Steady & Loyal | 872 | 6.9% | Consistent engagement -- lowest risk |

The 70+ percentage-point spread between Intense & Brief (77.7%) and Steady & Loyal (6.9%) confirms that lifecycle segmentation captures meaningful behavioral differences. This single feature nearly predicts churn on its own. We'll see it validated again in Stage 8 (Cramer's V=0.665) and it will explain the false outlier problem in Stage 7.

### Missing Values After Aggregation

**Why we examine this:** The informative nulls from Stage 3 have now propagated through aggregation. We need to verify they're preserved as signal, not accidentally imputed.

**What the analysis shows:**

| Window | Null Rate | Reason |
|--------|-----------|--------|
| 180d features | **57.7%** (2,884 customers) | No emails in last 180 days |
| 365d features | 41.4% (2,069 customers) | No emails in last year |
| all_time features | 0% | Every customer has at least one email |
| `time_to_open_hours` (180d) | **86.3%** | Only populated when email opened |

The 57.7% null rate in 180d features connects directly to the 93-day median cadence from Stage 2. If customers typically interact once every 93 days, many will have zero events in any given 180-day window -- especially relative to the 2022-09-26 cutoff. These nulls are *informative*: a customer with no 180-day activity is behaviorally different from one with activity, and models should treat this missingness as signal, not noise.

**Auto-derived recommendation:** Preserve nulls. Do not impute to zero or to the mean.

**Alternative technique:** Impute zeros (would destroy signal -- a customer with zero activity is not the same as a customer whose activity is unknown).

### Leakage Validation

**Why we check this:** This is the safety gate before modeling. Any feature that uses information from after the cutoff date, or that encodes the target variable, would produce artificially inflated performance.

**What the analysis shows:** 83 checks, zero critical issues. The target column was excluded from all aggregations, and no future-looking features were detected.

### Decision Made
- **72 features** from 3 windows, 5 value columns, 4 aggregation functions + derived features
- **Nulls preserved** as informative (not imputed to zero)
- **371 duplicate events removed** before aggregation

### Alternative Approaches
- **More windows (7d, 30d, 90d)**: Rejected -- too few events per customer in short windows (median cadence 93 days)
- **Fewer aggregation functions**: Could drop `max` and `count` to reduce features, but count is highly informative (perfect correlation with event_count per window)
- **Entity-level only (skip aggregation)**: Would lose all temporal signal -- the entire feature engineering comes from this step

---

## Stage 6: Column Analysis -- Post-Aggregation Feature Distributions

With 72 features in hand, we examine each one. Aggregated features from event data tend to look unusual -- heavy right skew, zero-inflation, extreme values -- but many of these "problems" are expected consequences of the quarterly engagement cadence we identified in Stage 2.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/02_column_deep_dive.html)

### Value Range Validation

**Why we check this:** Catch impossible values -- negative counts, rates above 100%, dates in the future. These would indicate bugs in the aggregation pipeline.

**What the analysis shows:** All ranges are valid after aggregation. No impossible values detected.

### Numeric Distribution Analysis

**Why we do this:** Determine which features need transformation for linear models. Highly skewed features can dominate regularization penalties and produce unstable coefficients.

**What the analysis shows:**

| Feature | Skewness | Zeros (%) | Issue | Recommended Transform |
|---------|----------|-----------|-------|----------------------|
| `event_count_180d` | 2.13 | 57.7% | Zero-inflated | Binary indicator + log |
| `event_count_all_time` | 2.90 | 0% | Right-skewed | Cap then log |
| `opened_sum_180d` | 3.31 | 86.3% | Heavily zero-inflated | Binary indicator + log |
| `clicked_sum_180d` | 4.76 | 95.3% | Extremely zero-inflated | Binary indicator + log |
| `bounced_sum_180d` | 7.54 | -- | Heavily right-skewed | Zero-inflation handling |

The 180-day window features are consistently the most problematic because 57.7% of customers had no activity in that window. This is a direct consequence of the quarterly engagement cadence discovered in Stage 2 -- the 93-day median gap means more than half of customers will have no events in any given 180-day window relative to the cutoff.

**Auto-derived transformation decision tree:** Zeros >40% → binary indicator + log(non-zeros); |skewness| >1 → log; kurtosis >10 → cap first, then transform.

**Alternative technique:** No transforms for tree-based models (scale-invariant). But linear models and regularization benefit from normalized features, so the transforms are generated regardless.

### Categorical Analysis

**Why we do this:** Choose encoding strategy based on cardinality and balance.

**What the analysis shows:**

| Column | Categories | Imbalance | Encoding |
|--------|-----------|-----------|----------|
| `lifecycle_quadrant` | 4 | 1.9x (balanced) | One-hot |
| `recency_bucket` | 5 | 21.5x (imbalanced) | One-hot |

Both features have low cardinality, making one-hot encoding straightforward.

**Alternative technique:** Target encoding (higher information density, single column per feature). Rejected due to leakage risk without careful cross-validation -- and one-hot is sufficient for 4-5 categories.

### 54 Transformation Recommendations

The framework auto-derives 54 transformation recommendations for the Gold layer. These feed directly into the generated pipeline (Stage 10's spec generation), ensuring that the production pipeline applies the same transforms as the exploration notebooks.

### Decision Made
- **Zero-inflation handling** for 180d features: Binary indicator for "has activity" + log transform of non-zero values
- **One-hot encoding** for both categorical features
- **Log transforms** for all right-skewed numeric features

### Alternative Approaches
- **Power transforms (Yeo-Johnson)**: More flexible than log, but adds complexity for marginal benefit on already-sparse features
- **Target encoding** for categoricals: Higher information density than one-hot, but risks leakage without careful cross-validation
- **No transformation**: Tree-based models are scale-invariant, but linear models and regularization benefit from normalized features

---

## Stage 7: Quality Assessment -- Pre-Modeling Quality Gate

Before modeling, a final quality check. This stage produces the most surprising finding of the analysis: what looks like an outlier problem is actually the heterogeneity from Stage 2 manifesting in a new form.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/03_quality_assessment.html)

### Quality Score

**Why we compute this:** A unified go/no-go metric for the aggregated data.

**What the analysis shows:**

| Metric | Value | Assessment |
|--------|-------|------------|
| **Quality Score** | **89/100** | Good -- minor issues to address |
| **Duplicates** | 0 (0.00%) | Clean entity keys |
| **Class 0 (Churned)** | 3,034 (60.7%) | Majority class |
| **Class 1 (Retained)** | 1,964 (39.3%) | Minority class |
| **Imbalance Ratio** | 1.54:1 | Mild -- stratified sampling sufficient |
| **Columns with Missing Values** | 22 | All from windowed aggregation |
| **Highest Missingness** | 86.3% | `time_to_open_hours` 180d features |

### Duplicate Check

**Why we check this:** Entity-level duplicates would mean the same customer counted twice, inflating training data and leaking information.

**What the analysis shows:** 0 duplicates. Clean entity keys after aggregation.

### Target Distribution

**Why we verify this:** Class balance drives modeling strategy -- severe imbalance requires oversampling or threshold tuning.

**What the analysis shows:** 60.7:39.3 (1.54:1 = mild imbalance).

**Auto-derived recommendation:** `class_weight='balanced'` in the model -- lets the algorithm upweight the minority class without synthetic oversampling.

**Alternative technique:** SMOTE oversampling. Overkill for 1.54:1 -- typically reserved for ratios above 10:1.

### Missing Value Analysis

**Why we examine patterns:** Understanding whether missingness is MCAR (random), MAR (depends on observed data), or MNAR (depends on the missing value itself) determines the imputation strategy.

**What the analysis shows:** 22 columns with missing values, all from windowed aggregation. The pattern is MNAR -- missingness means "no activity in this window," which is itself a behavioral signal. This is the same informative null pattern we identified in Stage 3 and preserved through Stage 5.

### Segment-Aware Outlier Analysis: Why Global Methods Fail

**Why we do this:** Global outlier detection fails when data has natural subgroups. This is where the Stage 2 heterogeneity finding (eta-squared=0.308) becomes a practical problem.

**What the analysis shows:** Global detection flags up to 24.8% of values as outliers for some features (`opened_sum_365d`). But within lifecycle segments, **93-99% of these are false positives**. Values that look extreme globally are perfectly normal for "Intense & Brief" customers, who by definition have high engagement concentrated in a short period.

This connects back through the analysis: Stage 2 identified high heterogeneity across lifecycle segments (eta-squared=0.308). Stage 5 turned that into a feature (`lifecycle_quadrant`) with a 70+ percentage-point churn spread. Now Stage 7 reveals the *practical consequence*: any global statistical method that doesn't account for segments will mischaracterize the data.

**Auto-derived recommendation:** Segment-aware outlier treatment. Do NOT apply global outlier caps.

**Alternative technique:** Global IQR capping -- the standard approach, but it would distort the "Intense & Brief" segment data, clipping valid high-engagement values.

> **Caution:** Naive outlier removal strips valid data from high-engagement segments. A customer who opened 50 emails in 365 days is an outlier globally but entirely normal for the "Intense & Brief" quadrant.

### Decision Made
- **Stratified sampling** for train/test splits (mild imbalance)
- **Segment-aware outlier treatment** -- do not apply global outlier caps
- **Missing values preserved** as informative (not imputed)

### Alternative Approaches
- **SMOTE oversampling**: Overkill for 1.54:1 ratio; typically reserved for >10:1 imbalance
- **Drop high-missingness columns**: Would remove all 180d features (57.7% null), losing recency signal
- **Global outlier capping**: Would distort the "Intense & Brief" segment where high values are normal behavior, not anomalies

---

## Stage 8: Relationship Analysis -- Which Features Matter?

With 72 features, many derived from the same underlying data, we need to determine which carry unique signal, which are redundant, and which dominate. This is the narrowing phase -- from everything we've built to what actually matters.

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/04_relationship_analysis.html)

### Correlation Matrix

**Why we do this:** Identify multicollinearity -- redundant features that carry the same information. Redundant features don't hurt tree-based models but can destabilize linear models.

**What the analysis shows:** 147 feature pairs with |r| >= 0.7. Count features across windows are perfectly correlated (e.g., `event_count_180d` = `opened_count_180d` = `clicked_count_180d`). This makes sense: the count function applied to any column within a window produces the same result (number of rows in that window).

**Auto-derived recommendation:** Tree-based models handle this natively. For linear models, would need feature selection or PCA.

### Effect Sizes (Cohen's d)

**Why we do this:** Quantify the discriminative power of each aggregated feature. This is the entity-level version of Stage 4's analysis, now applied to the full 72-feature set.

**What the analysis shows:**

| Feature | Cohen's d | Direction | Interpretation |
|---------|-----------|-----------|----------------|
| `days_since_last_event` | **+2.442** | Higher in churned | Churned customers have longer recency |
| `event_count_365d` | -1.405 | Lower in churned | Churned had more recent email activity |
| `send_hour_sum_365d` | -1.367 | Lower in churned | -- |
| `event_count_180d` | -1.087 | Lower in churned | -- |
| `send_hour_sum_180d` | -1.042 | Lower in churned | -- |
| `opened_sum_all_time` | -0.929 | Lower in churned | -- |
| `opened_mean_all_time` | -0.836 | Lower in churned | -- |

`days_since_last_event` at d=2.442 dominates -- nearly 2x stronger than any other feature. This is the recency signal from Stage 4 (d=2.23 at the event level), now *stronger* after aggregation because the entity-level view separates the signal more cleanly.

**Auto-derived recommendation:** Recency is the dominant predictor, but dominance is fragile -- monitor for drift.

### Categorical Features (Cramer's V)

**Why we do this:** Cramer's V is the categorical equivalent of Cohen's d -- it quantifies how strongly each categorical feature associates with the target.

**What the analysis shows:**

| Feature | Cramer's V | Strength |
|---------|------------|----------|
| `lifecycle_quadrant` | **0.665** | Strong |
| `recency_bucket` | **0.598** | Strong |

Both categorical features are highly predictive. The lifecycle quadrant created during aggregation (Stage 5), which originated from the heterogeneity analysis in Stage 2 (eta-squared=0.308), has now been validated three ways: high eta-squared → 70+ percentage-point churn spread → Cramer's V of 0.665.

### Feature-Target Correlations

**Why we do this:** Complement effect sizes with correlation direction to ensure rankings are consistent across measures.

**What the analysis shows:** Confirms Cohen's d rankings. No hidden features with strong correlations but weak effect sizes.

### Actionable Recommendations

The framework auto-derives recommendations from the relationship analysis:
- Feature selection based on effect sizes and correlation clusters
- Stratification by `lifecycle_quadrant` (70.8% retention spread across segments)
- Tree-based models preferred due to 147 correlated pairs

> **Caution: Feature Dominance Risk.** `days_since_last_event` dominates at d=2.44 -- nearly 2x stronger than any other feature. This is both a strength (strong signal) and a risk (model becomes a recency detector). If recency patterns shift over time, the model could degrade rapidly. Consider building a secondary model without recency features as a fallback.

### Decision Made
- **Tree-based models preferred** (handle multicollinearity natively)
- **Stratify by `lifecycle_quadrant`** (70.8% retention spread across segments)
- **Monitor `days_since_last_event` drift** in production

---

## Stage 9: Feature Opportunities -- Do We Have Enough Data?

A practical checkpoint before modeling. We've built 72 features for 4,998 customers. Is there enough data to learn from all of them without overfitting?

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/06_feature_opportunities.html)

### Feature Capacity (EPV)

**Why we compute this:** Events Per Variable (EPV) measures whether we have enough data per feature to avoid overfitting. The rule of thumb: EPV >= 10 for stable models.

**What the analysis shows:**

| Metric | Value | Status |
|--------|-------|--------|
| **Total Samples** | 4,998 | -- |
| **Minority Class** | 1,964 (39.3%) | -- |
| **Current Features** | 59 numeric | -- |
| **EPV (Events Per Variable)** | **33.3** | Adequate |
| **Effective Independent Features** | 18.0 | After correlation clustering |
| **Redundant Features** | 29 | In 12 correlated clusters |
| **Feature Budget Remaining** | 137 more (at EPV=10) | Ample headroom |

EPV of 33.3 is well above the minimum of 10, meaning we have ample data for all model types.

### Model Complexity Guidance

**Why we assess this:** Data size determines which model types are viable. Small datasets restrict you to simple models; large datasets unlock complex ones.

**What the analysis shows:** All model types (linear, regularized, tree-based) have sufficient data. The strong linear signal (d=2.44 for recency) means even simple linear models should perform well.

**Auto-derived recommendation:** Linear models viable given strong linear signal. No need for deep learning or ensemble methods to achieve good performance.

### Segment-Specific Capacity

**Why we check this:** If we considered separate models per segment (as the Stage 2 heterogeneity analysis flagged), would each segment have enough data?

**What the analysis shows:** Smallest segment EPV = 1.0 -- far below the minimum of 10. Separate models per segment would overfit severely.

**Auto-derived recommendation:** Single global model with `lifecycle_quadrant` as a feature, confirming the Stage 2 advisory.

### Decision Made
- **Single global model** (not segment-specific)
- **Feature budget: ample** -- current 59 features well within capacity limits
- **No additional derived features needed** -- existing temporal features provide sufficient signal

---

## Stage 10: Baseline Experiments -- Can These Features Predict Churn?

The analysis has produced 72 features with strong recency signal, lifecycle segmentation, and clean data. Now we test: can these features actually predict churn?

[View Notebook →](https://aladjov.github.io/CR/tutorial/customer-emails/08_baseline_experiments.html)

### Data Preparation

**Why we prepare carefully:** Proper train/test splitting and scaling prevents overfitting estimation. Scaling after the split (not before) ensures the test set remains truly unseen -- if you fit the scaler on all data, the test set leaks information about its distribution into training.

**What the analysis shows:** Stratified split preserving class balance, standard scaling applied after split, median imputation for missing values.

### Baseline Models with Class Weights

**Why we use class weights:** The `class_weight='balanced'` parameter lets the algorithm upweight the minority class during training without synthetic oversampling. For 1.54:1 imbalance, this is sufficient.

**What the analysis shows:** All three models (Logistic Regression, Random Forest, Gradient Boosting) train successfully.

### Model Comparison

**Why we test multiple families:** Different model architectures have different strengths -- linear models excel at interpretability and stability; tree-based models handle nonlinearity and feature interactions.

**What the analysis shows:**

| Model | Test AUC | PR-AUC | F1-Score | Precision | Recall | CV AUC (Mean) | CV AUC (Std) |
|-------|----------|--------|----------|-----------|--------|---------------|--------------|
| Logistic Regression | 0.9598 | 0.9544 | 0.8819 | 0.9106 | 0.8550 | 0.9577 | 0.0051 |
| Random Forest | 0.9549 | 0.9500 | 0.8829 | 0.9371 | 0.8346 | 0.9531 | 0.0047 |
| **Gradient Boosting** | **0.9620** | **0.9577** | **0.8844** | **0.9373** | **0.8372** | **0.9562** | **0.0068** |

**All three models achieve AUC > 0.95.** The performance gap is only 0.7% AUC spread (0.9549 to 0.9620). This tells us something important: the signal in the features is strong enough that model choice barely matters. The aggregation decisions in Stage 5 determined the outcome more than any model architecture could.

**Classification Report (Gradient Boosting):**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Churned (0) | 0.90 | 0.96 | 0.93 |
| Retained (1) | 0.94 | 0.84 | 0.88 |
| **Accuracy** | | | **0.91** |

### Feature Importance

**Why we verify this:** The model should rely on the features our earlier analysis identified as strongest. If it doesn't, either our analysis was wrong or the model is finding spurious patterns.

**What the analysis shows:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `days_since_last_event` | **0.1848** |
| 2 | `event_count_365d` | 0.0789 |
| 3 | `clicked_count_365d` | 0.0589 |
| 4 | `bounced_count_365d` | 0.0554 |
| 5 | `send_hour_count_365d` | 0.0547 |
| 6 | `send_hour_sum_365d` | 0.0493 |
| 7 | `opened_count_365d` | 0.0451 |
| 8 | `clicked_count_all_time` | 0.0304 |
| 9 | `send_hour_count_all_time` | 0.0263 |
| 10 | `bounced_count_all_time` | 0.0230 |

`days_since_last_event` accounts for **18.5% of total importance** -- confirming Stage 4's prediction that recency would dominate. The 365-day window features fill positions 2-7, validating Stage 2's window analysis (365d captures the most predictive signal). 180-day features are notably absent from the top 10 -- their 57.7% null rate limits their utility, though the nulls themselves are captured through the binary indicator transforms from Stage 6.

**Alternative technique:** Permutation importance or SHAP values provide model-agnostic importance rankings. SHAP is explored in the scoring dashboard (Stage 11).

### PR Curves

**Why we check this:** ROC-AUC can look optimistic for imbalanced data because it credits correct rejection of the majority class. Precision-Recall curves focus on the minority class -- how well does the model identify the customers who actually churn?

**What the analysis shows:** PR-AUC follows the same ranking as ROC-AUC. No model looks artificially good on one metric but poor on the other.

### Decision Made
- **Primary model:** Gradient Boosting (best AUC at 0.9620)
- **Fallback model:** Logistic Regression (nearly identical performance, more interpretable, likely more robust to drift)
- **Class weights:** Balanced (handles 1.54:1 imbalance)
- **Assessment:** Excellent predictive signal -- production-ready with tuning

### Alternatives Not Explored (Future Work)
- **Hyperparameter tuning**: Could improve AUC by 1-2%
- **Feature selection**: The 29 redundant features could be removed with minimal impact
- **Model without recency**: Test performance without `days_since_last_event` to assess recency dependence

---

## Stage 11: Production Reality Check -- Does It Hold Up?

Cross-validation tells us how well our model generalizes to *similar* data. But production data comes from the *future* -- it may have different patterns due to seasonality, campaign changes, customer behavior shifts, or data quality drift. The scoring pipeline tests our models on a **point-in-time holdout** (the 10% of entities masked before feature computation in Stage 1), simulating true deployment conditions.

[View Scoring Dashboard →](https://aladjov.github.io/CR/tutorial/customer-emails/scoring_dashboard.html)

### Pipeline Training

**Why we retrain through the pipeline:** The generated pipeline retrains from scratch through Bronze/Silver/Gold layers, validating the entire data processing chain. If any transformation was hard-coded or environment-dependent during exploration, it would fail here.

**What the analysis shows:**

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|---------|--------|-----|-----------|--------|
| **XGBoost** | **0.9602** | 0.9567 | 0.8919 | 0.9348 | 0.8527 |
| Logistic Regression | 0.9589 | 0.9541 | 0.8909 | 0.9430 | 0.8442 |
| Random Forest | 0.9555 | 0.9536 | 0.8905 | 0.9319 | 0.8527 |

Results match exploration-phase training (AUC 0.960 vs 0.962). The 0.002 difference is well within expected variance. The pipeline reproduced the exploration-phase findings.

### Holdout Scoring

**Why we score holdout entities:** These entities were masked before feature computation -- they flowed through the same Bronze/Silver/Gold pipeline as training data, but their targets were hidden. This simulates true future data more faithfully than a random split.

**What the analysis shows:**

| Metric | Value |
|--------|-------|
| **Holdout Records** | 499 (10% of data) |
| **Total Predictions** | 1,497 (3 models x 499) |
| **Overall Correct** | 1,371 / 1,497 (**91.6% accuracy**) |

The signal holds on data the model never saw during training.

### Feature Importance (Production)

**Why we verify this:** Training-phase importance rankings must hold in production. If the model relies on different features after retraining through the pipeline, something in the data processing changed.

**What the analysis shows:**

| Rank | Feature | Importance (Gain) |
|------|---------|-------------------|
| 1 | `days_since_last_event` | **2.564** |
| 2 | `opened_mean_365d` | 0.538 |
| 3 | `event_count_all_time` | 0.335 |
| 4 | `opened_mean_all_time` | 0.297 |
| 5 | `send_hour_sum_all_time` | 0.171 |

`days_since_last_event` at importance 2.564 is **4.8x higher** than the second feature. This is consistent with exploration: recency dominates, and its dominance is not an artifact of the exploration environment. This is not overfitting.

### Adversarial Pipeline Validation

**Why we run this:** Verify the scoring pipeline produces identical features to training. Catches scaler re-fitting (should use saved scaler), encoder inconsistencies (should use saved mappings), and feature ordering differences.

**What the analysis shows:** Passed. The pipeline is consistent.

### The Key Lesson

All three models perform remarkably similarly (AUC spread <0.5%), suggesting the email engagement features carry strong, robust signal regardless of model complexity. For this dataset, **model choice matters less than feature engineering** -- the aggregation decisions in Stage 5 determined the outcome.

### Recommendations

1. **Any of the three models is production-viable** -- the performance difference is negligible
2. **Prefer Logistic Regression if interpretability matters** -- nearly identical AUC with simpler explanations
3. **Monitor `days_since_last_event` distribution** -- the dominant feature is also the most drift-prone
4. **Set drift alerts on email volume trends** -- the declining volume (-28%) observed in Stage 2 could affect feature distributions over time

---

## Key Lessons

Looking back across the analysis, a clear arc emerges. We started with 83,000 email events -- raw data that couldn't be modeled directly because each customer had many events and the data was organized by *what happened*, not by *who it happened to*. The first half (Stages 1-4) was entirely about understanding the data well enough to make the aggregation decisions in Stage 5: which time windows to use (driven by the 93-day median cadence from Stage 2), which derived features to create (driven by the recency and momentum signals from Stage 4), and how to handle the heterogeneity across customer segments (driven by Stage 2's eta-squared analysis and validated at every subsequent stage).

Stage 5 was the pivot point -- the transformation from events to entities that every downstream analysis depended on. The second half (Stages 6-11) progressively validated that the aggregated features carried the signal we expected. The zero-inflation in Stage 6 traced back to the quarterly cadence. The false outlier problem in Stage 7 was the segment heterogeneity from Stage 2 manifesting as a practical hazard. The feature importance rankings in Stages 8, 10, and 11 consistently confirmed recency as the dominant predictor, with its Cohen's d growing from 2.23 at the event level (Stage 4) to 2.44 after aggregation (Stage 8) to 4.8x importance dominance in production (Stage 11).

The final models achieved AUC >0.95 regardless of algorithm, which tells us the most important lesson: **aggregation choices mattered more than model choices**. The first five stages -- understanding, validating, and transforming the data -- determined the outcome. The modeling stages confirmed it.

### Core Principles

**Aggregation is the feature engineering.** In entity-level data, features come pre-computed. In event-level data, the aggregation step *is* the feature engineering. The choice of windows (180d, 365d, all_time from Stage 2), functions (sum, mean, max, count), and derived features (recency from Stage 4, lifecycle from Stage 2, momentum from Stage 4) determines what signal the model can access. Stage 5 consolidated all of these into 72 features -- and the model simply learned from what that step provided.

**Window selection drives everything.** The 93-day median inter-event gap from Stage 2 was the single most consequential number in the analysis. It eliminated short windows (7d, 30d, 90d), determined that 180d was the tightest viable recency signal, explained the 57.7% null rate in Stage 5, the zero-inflation in Stage 6, and the feature importance rankings in Stage 10 (365d features dominated because they captured roughly four engagement cycles).

**Absence is signal.** 57.7% of customers had no 180-day activity, and that null pattern is highly predictive (Stages 5-7). `time_to_open_hours` was 77.7% missing because most emails were never opened -- and the missingness encodes engagement quality more directly than any imputed value could (Stage 3). Event-based pipelines must preserve missingness as information rather than imputing it away.

**Event-level targets mislead.** The raw 97.4:2.6 split (Stage 1) was misleading -- most individual emails don't trigger unsubscription. After entity-level aggregation via `max` (Stage 4), the true distribution was 60.5:39.5 -- a manageable imbalance that only needed `class_weight='balanced'` (Stage 7), not the extreme imbalance handling the event-level view would have suggested.

**Recency dominates -- and that's both a strength and a risk.** `days_since_last_event` achieved Cohen's d of 2.23 at the event level (Stage 4), 2.44 after aggregation (Stage 8), 18.5% of model importance (Stage 10), and 4.8x dominance in production (Stage 11). This consistency across stages builds confidence that the signal is real. But over-reliance on a single feature creates fragility -- if recency patterns shift (new campaign cadences, seasonal changes), the model degrades rapidly. The caution from Stage 4 about trailing vs. leading indicators remains unresolved.

### What Wasn't Explored (Future Work)

**Model Improvements:**
1. **Hyperparameter tuning** -- defaults achieved AUC 0.962; tuning could push to 0.97+
2. **Feature selection** -- 29 redundant features could be removed; test impact
3. **Model without recency** -- assess performance when removing `days_since_last_event`
4. **Segment-specific thresholds** -- different probability thresholds per lifecycle quadrant

**Temporal Enhancements:**
5. **Shorter observation windows with activity flags** -- binary "any activity in 30d" may capture recency without sparse aggregation
6. **Sequential modeling** -- RNN/LSTM on raw event sequences instead of aggregation
7. **Campaign response features** -- aggregate by campaign type for campaign-specific engagement rates
8. **Time-between-events features** -- statistics on inter-event gaps (already partially captured by recency)

**Production Readiness:**
9. **Walk-forward validation** -- simulate true temporal deployment
10. **Drift monitoring dashboard** -- track `days_since_last_event` distribution over time
11. **A/B testing framework** -- measure intervention effectiveness
12. **Recency feature stability analysis** -- how recency distributions change month-over-month

---

## Running the Tutorial

```bash
# Clone and install
git clone https://github.com/aladjov/CR.git
cd CR
pip install -e ".[dev,ml]"

# Start with the first notebook
jupyter lab exploration_notebooks/00_start_here.ipynb
```

Set `DATA_PATH = "tests/fixtures/customer_emails.csv"` in the data discovery notebook.

**Note on the temporal track:** When 01_data_discovery detects event-level data, it automatically activates notebooks 01a through 01d. These notebooks handle temporal analysis and aggregation before the standard column analysis (02+) can proceed. The aggregated data is saved as a parquet file that downstream notebooks load automatically.

**Running the generated pipeline:**
```bash
# After completing exploration notebooks, run the generated pipeline
python generated_pipelines/local/customer_churn/run_all.py

# Run scoring on holdout data
python generated_pipelines/local/customer_churn/scoring/run_scoring.py
```

---

## Next Steps

- [[Architecture]] - Understand the medallion architecture (Bronze/Silver/Gold)
- [[Temporal Framework]] - Deep dive into leakage-safe temporal data preparation
- [[Local Track]] - How generated pipelines work (Feast + MLflow)
- [[Databricks Track]] - Deploy to Databricks with Unity Catalog
- [[Tutorial-Retail-Churn]] - Compare with the entity-level retail tutorial
