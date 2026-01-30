# Tutorial: Retail Customer Retention

This tutorial demonstrates a complete customer retention ML pipeline using a synthetic retail dataset. Rather than just showing *what* we do, we focus on *why* each step matters and *what decisions* follow from the analysis.

**[View Interactive Tutorial (HTML)](https://aladjov.github.io/CR/tutorial/retail-churn/)** - Browse all executed notebooks with visualizations

---

## The Business Problem

A retail company wants to predict which customers will churn so they can intervene proactively. The goal is to identify at-risk customers before they leave, enabling targeted retention campaigns.

**Key Questions We'll Answer:**
1. Is the data suitable for ML modeling?
2. What features drive customer retention?
3. How accurately can we predict churn?
4. What's the right model for production?

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | `tests/fixtures/customer_retention_retail.csv` |
| **Customers** | 26,578 (after point-in-time filtering) |
| **Features** | 18 columns |
| **Target** | `retained` (binary: 0=churned, 1=retained) |
| **Retention Rate** | 79.5% (3.8:1 class imbalance) |
| **Time Span** | 2008-2018 (10 years) |

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

## Stage 1: Data Discovery

**Purpose:** Understand dataset structure, detect temporal patterns, and create a point-in-time snapshot that prevents data leakage.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/01_data_discovery.html)

### Why Point-in-Time Matters

The most common mistake in churn modeling is **data leakage** - accidentally using information from the future to predict the past. For example, if we use a customer's behavior from December to predict their January churn status, we're cheating.

The framework automatically:
1. Detects temporal columns (`created`, `firstorder`, `lastorder`)
2. Identifies the **feature timestamp** (when we observe features) vs **label timestamp** (when we know the outcome)
3. Creates a versioned snapshot with a specific cutoff date

### Key Findings

| Metric | Value | Implication |
|--------|-------|-------------|
| Temporal Scenario | Production | Real timestamps available |
| Cutoff Date | ~90/10 train/score split | Adequate holdout for validation |
| Structure | Entity-level | One row per customer (not event-level) |
| Snapshot ID | training_v38 | Tracked for reproducibility |

### Decision Made
- **Selected cutoff** that achieves 90/10 split
- **Label window**: 180 days (standard for retail churn)
- All downstream notebooks load from this snapshot, ensuring consistency

### Alternatives Considered
- **Manual cutoff override**: Useful if business has specific date requirements
- **Different label windows**: 90 days (aggressive) or 365 days (conservative) depending on business definition of churn
- **Event-level aggregation**: If data had multiple rows per customer, we'd use the temporal track (notebooks 01a-01d)

---

## Stage 2: Column Deep Dive

**Purpose:** Analyze each column's distribution, detect issues, and determine appropriate transformations.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/02_column_deep_dive.html)

### Why Distribution Analysis Matters

Not all features are created equal. Understanding distributions helps us:
- Identify **skewed features** that need transformation
- Detect **zero-inflation** (many zeros requiring special handling)
- Choose appropriate **encoding strategies** for categoricals
- Spot **data quality issues** early

### Key Findings

**Numeric Features - Distribution Analysis:**

| Feature | Mean | Skewness | Issue | Recommended Transform |
|---------|------|----------|-------|----------------------|
| esent | 27.9 | -0.18 | None (symmetric) | Standard scaling |
| eopenrate | 24.6% | 1.22 | Right-skewed | Log transform |
| eclickrate | 5.3% | 3.96 | **52% zeros**, highly skewed | Zero-inflation handling |
| avgorder | 62.5 | 12.42 | Extreme outliers (max 2600) | Cap + log transform |
| ordfreq | 0.038 | 10.38 | **61% zeros**, highly skewed | Zero-inflation handling |

**Key Insight:** Two features (`eclickrate`, `ordfreq`) have over 50% zeros. This is **zero-inflation** - we can't just log-transform these. Instead, we create a binary "has_value" indicator plus a log-transformed value for non-zeros.

**Categorical Features:**

| Feature | Categories | Encoding Strategy | Why |
|---------|------------|-------------------|-----|
| city | 4 (DEL, BOM, MAA, BLR) | One-hot encoding | Low cardinality, no natural order |
| favday | 7 (days of week) | Cyclical (sin/cos) | Preserves that Sunday ≈ Saturday |

### Decision Made
- Mark skewed features for log transformation in modeling pipeline
- Apply cyclical encoding for day-of-week (not ordinal, because Monday isn't "greater than" Sunday)
- Create zero-inflation indicators for features with >40% zeros

### Alternatives Considered
- **Winsorization** (cap outliers at 99th percentile): Simpler but loses information
- **Binning**: Converts numeric to categorical, loses granularity but adds robustness
- **Keep as-is**: Tree-based models don't require normality, but interpretability suffers

---

## Stage 3: Quality Assessment

**Purpose:** Deep dive into data quality to ensure we're not modeling noise.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/03_quality_assessment.html)

### Why Quality Matters

Garbage in, garbage out. Before investing in modeling, we need to verify:
- No duplicates (double-counting customers)
- Minimal missing values (or understood patterns)
- Valid value ranges (no impossible values)
- Consistent encoding (no "Male" vs "male" vs "M" issues)

### Key Findings

**Overall Quality Score: 99.7/100** - Excellent!

**Missing Values:**
- Only 20 rows (0.06%) have missing values
- All 4 columns missing together (same 20 customers)
- **Decision:** Safe to drop these rows (negligible impact)

**Target Distribution:**
| Class | Count | Percentage |
|-------|-------|------------|
| Retained (1) | 21,041 | 79.5% |
| Churned (0) | 5,537 | 20.5% |

**Imbalance ratio: 3.8:1** - This is **mild to moderate** imbalance. We'll handle it with class weights rather than oversampling.

**Outlier Analysis:**

| Feature | Outliers | % of Data | Action |
|---------|----------|-----------|--------|
| esent | 7 | 0.03% | Minor, ignore |
| eopenrate | 992 | 3.7% | Moderate, cap at 99th percentile |
| eclickrate | 2,586 | 9.7% | Significant, log transform |
| ordfreq | 3,196 | 12.0% | Significant, log transform |

### Decision Made
- Keep data mostly as-is (quality is excellent)
- Use **balanced class weights** in models to handle 3.8:1 imbalance
- Apply log transforms to high-outlier features (already planned from Stage 2)

### Alternatives Considered
- **SMOTE oversampling**: Creates synthetic minority examples. Not needed here - imbalance is mild and we have 5,537 churned examples (plenty to learn from)
- **Undersampling majority**: Throws away data. Only useful for extreme imbalance (>10:1)
- **Drop outliers entirely**: Too aggressive. Better to transform and preserve information

---

## Stage 4: Relationship Analysis

**Purpose:** Identify which features predict retention and how strongly.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/04_relationship_analysis.html)

### Why Relationship Analysis Matters

Before modeling, we should understand:
- Which features have **predictive signal**
- Are there **multicollinearity** issues (redundant features)?
- What's the **nature of the relationship** (linear? non-linear?)

### Key Findings

**Feature Importance (Effect Size):**

| Feature | Cohen's d | Correlation | Interpretation |
|---------|-----------|-------------|----------------|
| **esent** | **+2.551** | **+0.718** | **LARGE** - dominates! |
| eopenrate | +0.35 | +0.24 | Small |
| eclickrate | +0.23 | +0.18 | Small |
| avgorder | +0.01 | +0.004 | Negligible |
| ordfreq | +0.09 | +0.08 | Negligible |

**Critical Insight:** `esent` (emails sent) has a **massive** effect size (d=2.551). This means retained customers receive, on average, 2.5 standard deviations more emails than churned customers.

**Mean Comparison:**
| Feature | Retained | Churned | Difference |
|---------|----------|---------|------------|
| esent | 34.25 | 4.50 | **+661.7%** |
| eopenrate | 26.68% | 21.21% | +25.8% |
| eclickrate | 5.90% | 4.79% | +23.3% |
| avgorder | $61.96 | $61.55 | +0.7% (no difference!) |

**Retention by City:**
| City | Retention Rate | Lift vs Average |
|------|----------------|-----------------|
| BLR (Bangalore) | 87.4% | 1.10x |
| MAA (Chennai) | 83.2% | 1.05x |
| BOM (Mumbai) | 79.9% | 1.01x |
| **DEL (Delhi)** | **73.6%** | **0.93x** - High risk! |

### Decision Made
- **Prioritize `esent`** - it's the strongest predictor
- **Keep weak features** - they may help in combination
- **Start with Logistic Regression** - strong linear signal (d=2.551) suggests linear models viable
- **Flag Delhi as high-risk segment** - may need targeted intervention

### Alternatives Considered
- **Drop weak features** (avgorder, ordfreq): Risk losing predictive power in combinations
- **Build segment-specific models per city**: Adds complexity, may not be worth it
- **Use only non-linear models**: Would work, but linear interpretability valuable for business

### Caution: Feature Dominance Risk
`esent` accounts for most predictive power. This creates **concentration risk**:
- If email data quality degrades, model fails
- Model may be capturing "customers who receive emails stay" rather than underlying behavior
- Consider: Is this a **leading indicator** (more emails → retention) or **trailing indicator** (retained customers get more emails because they're active)?

---

## Stage 5: Feature Opportunities

**Purpose:** Determine how many features we can safely add and what derived features would help.

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
- **Minority class samples:** 5,751 (churned customers)
- **Current features:** 5
- **EPV:** 1,150 (5,751 / 5)
- **Capacity status:** **ABUNDANT** - we can safely add 570+ features!

**Derived Features Created:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| tenure_days | now - created | Longer tenure → more likely retained |
| days_since_last_activity | now - lastorder | Recency is key churn signal |
| email_engagement_score | 0.6×open + 0.4×click | Composite engagement metric |
| click_to_open_rate | click / open | Quality of engagement |
| service_adoption_score | paperless + refill + doorstep | More services → stickier customer |

**Customer Segmentation Analysis:**

| Segment Type | Distribution | Insight |
|--------------|--------------|---------|
| Dormant (180+ days) | **98.8%** | Most customers inactive! |
| Low Engagement | **77.3%** | Majority disengaged |
| High Value + Frequent | 50% | Evenly split |

**Concerning Finding:** 98.8% of customers haven't ordered in 180+ days. This suggests either:
1. The data is old (snapshot from past)
2. Business has long purchase cycles
3. Retention definition may need revisiting

### Decision Made
- Created 8 derived features (from 5 to 13 total)
- Apply log transforms to skewed features as planned
- Use one-hot encoding for city, cyclical for favday

### Alternatives Considered
- **Aggressive feature engineering** (100+ features): Capacity allows it, but diminishing returns
- **PCA dimensionality reduction**: Not needed with only 13 features
- **Automated feature generation** (featuretools): Overkill for this dataset

---

## Stage 6: Baseline Experiments

**Purpose:** Establish performance benchmarks with standard models.

[View Notebook →](https://aladjov.github.io/CR/tutorial/retail-churn/08_baseline_experiments.html)

### Why Baseline First?

Never start with complex models. Baselines tell us:
- Is the problem solvable at all?
- How much signal exists in the data?
- What's the performance floor to beat?

### Key Findings

**Model Performance Comparison:**

| Model | Test AUC | PR-AUC | Precision | Recall | F1 |
|-------|----------|--------|-----------|--------|-----|
| Logistic Regression | 0.9596 | 0.9782 | 76.4% | 90.9% | 0.915 |
| Random Forest | 0.9783 | 0.9922 | 96.4% | 89.1% | 0.933 |
| **Gradient Boosting** | **0.9858** | **0.9949** | **96.96%** | **88.3%** | **0.980** |

**Winner: Gradient Boosting** with AUC 0.9858 - Excellent!

**What These Numbers Mean:**
- **AUC 0.9858:** Model correctly ranks 98.6% of customer pairs by churn risk
- **Precision 97%:** When we predict someone will churn, we're right 97% of the time
- **Recall 88%:** We catch 88% of actual churners (miss 12%)

**Feature Importance (What Drives Predictions):**

| Rank | Feature | Importance | Cumulative |
|------|---------|------------|------------|
| 1 | **esent** | **58.4%** | 58.4% |
| 2 | eopenrate | 13.5% | 71.9% |
| 3 | eclickrate | 7.8% | 79.7% |
| 4 | Others | <5% each | 100% |

**Critical Insight:** `esent` alone accounts for **58% of predictive power**. This is both good (strong signal) and concerning (concentration risk).

### Interpreting the Model

Feature importance tells us *what* matters, but not *how* it matters. For deeper understanding:

**Partial Dependence Analysis (esent):**
- 0-10 emails: Steep positive effect on retention
- 10-30 emails: Moderate positive effect
- 30+ emails: Diminishing returns (plateau)

**Business Implication:** Focus email campaigns on customers receiving <10 emails - that's where marginal impact is highest.

**Caution on Causality:** High `esent` correlation with retention could mean:
- (A) More emails → Higher retention (emails cause retention)
- (B) Active customers → More emails (activity causes both)
- (C) Both are caused by a third factor (e.g., customer lifetime value)

Only A/B testing can establish causality. The model predicts, it doesn't explain *why*.

### Decision Made
- **Primary model:** Gradient Boosting (best performance)
- **Fallback model:** Logistic Regression (if interpretability needed)
- **Class weights:** Balanced (handles 3.8:1 imbalance well)

### Alternatives Not Explored (Future Work)
- **Neural Networks:** Overkill for tabular data with 13 features
- **SVM with RBF kernel:** Slower, similar performance to tree ensembles
- **Ensemble of all three models:** Could squeeze extra 0.5% AUC
- **Hyperparameter tuning:** Current results are with defaults - tuning could improve

---

## Stage 7: Production Reality Check

**Purpose:** Test models on truly unseen future data to simulate production conditions.

[View Scoring Dashboard →](https://aladjov.github.io/CR/tutorial/retail-churn/scoring_dashboard.html)

### Why Validation Isn't Enough

Cross-validation tells us how well our model generalizes to **similar** data. But production data is from the **future** - it may have different patterns due to:
- Seasonality
- Business changes
- Customer behavior shifts
- Data quality drift

The scoring dashboard tests our models on a **point-in-time holdout** (data after the training cutoff).

### Key Findings

**The Surprising Result:**

| Model | Validation AUC | Scoring AUC | **Gap** |
|-------|----------------|-------------|---------|
| XGBoost | **0.9854** (1st) | 0.9142 (3rd) | **-7.2%** |
| Random Forest | 0.9830 (2nd) | 0.9073 (2nd) | -7.7% |
| **Logistic Regression** | 0.9678 (3rd) | **0.9441 (1st)** | **-2.4%** |

**XGBoost was best in validation but WORST in production!**

### Why Did Complex Models Degrade More?

**Distribution Drift Detected:**

| Feature | Train Mean | Score Mean | Drift |
|---------|------------|------------|-------|
| eopenrate | 24.6% | 31.9% | **+29.8%** |
| eclickrate | 5.3% | 7.7% | **+44.5%** |
| TARGET (retention) | 78.4% | 86.4% | **+10.3%** |

The scoring population has **significantly higher email engagement** than training data. Complex models (XGBoost, Random Forest) learned intricate patterns specific to the training distribution. When that distribution shifted, they failed.

**Logistic Regression's simpler decision boundary generalized better.**

### The Key Lesson

**Simpler models can be more robust to distribution drift.**

This is a crucial production insight. In a controlled validation environment, complex models often win. But in production where data drifts over time, simpler models may be more reliable.

### Recommendations

1. **Consider Logistic Regression for production** - only 2.4% degradation vs 7%+ for tree models
2. **Monitor drift** - track `eopenrate` and `eclickrate` distributions
3. **Set drift alerts** - retrain when feature distributions shift >20%
4. **Recalibrate thresholds** - higher baseline retention requires adjusted decision thresholds

### Improving Robustness to Drift

The drift problem we observed isn't unique to this dataset - it's a fundamental challenge in production ML. Here are techniques to build more drift-resistant models:

**1. Training Strategies**

| Technique | How It Helps | Trade-off |
|-----------|--------------|-----------|
| **Stronger Regularization (L1/L2)** | Penalizes complex patterns that may be distribution-specific | May reduce validation performance |
| **Time-based Cross-Validation** | Use walk-forward validation instead of random splits. Train on months 1-6, validate on 7, then train on 1-7, validate on 8, etc. | Requires more data, slower |
| **Recency Weighting** | Weight recent samples higher (e.g., exponential decay). Recent patterns matter more for future prediction | Loses historical patterns |
| **Adversarial Validation** | Train a model to distinguish train from score data. Features it uses most are drift-prone - consider removing them | May remove predictive features |
| **Domain Adaptation** | Fine-tune model on small sample of target distribution | Requires labeled target data |

**2. Feature Engineering for Stability**

| Approach | Example | Why It Helps |
|----------|---------|--------------|
| **Relative features over absolute** | Use `eopenrate / industry_avg` instead of raw `eopenrate` | Normalizes for population-level shifts |
| **Rank-based features** | Convert to percentiles within time window | Robust to distribution shifts |
| **Ratio features** | `click_to_open = eclickrate / eopenrate` | Captures behavior quality, not volume |
| **Remove unstable features** | Drop features with high drift (eopenrate, eclickrate) | Trades accuracy for stability |
| **Binning continuous features** | Convert numeric to categories | Reduces sensitivity to exact values |

**3. Model Architecture Choices**

| Choice | Drift Robustness | Why |
|--------|------------------|-----|
| Linear models | **High** | Simple decision boundaries generalize |
| Shallow trees (depth ≤ 3) | **High** | Can't memorize complex patterns |
| Deep ensembles (XGBoost) | **Low** | Learn distribution-specific patterns |
| Neural networks | **Variable** | Depends on regularization |

**4. Operational Strategies**

- **Continuous retraining**: Retrain weekly/monthly on recent data
- **Online learning**: Update model incrementally as new data arrives
- **Model ensembles with diversity**: Combine models trained on different time periods
- **Prediction intervals**: Output uncertainty, not just point predictions
- **A/B testing**: Compare model versions on live traffic before full rollout

### Training Differently: What We Could Have Done

Looking back, here's how we could have trained to better handle the drift we observed:

**Walk-Forward Validation (Recommended for Time-Series)**
```
Instead of:  Random 80/20 split
Do this:     Train on 2008-2015, validate on 2016
             Train on 2008-2016, validate on 2017
             Train on 2008-2017, validate on 2018
             Average performance across folds
```
This simulates production conditions where we always predict the future.

**Adversarial Feature Selection**
```
1. Label training data as 0, scoring data as 1
2. Train classifier to distinguish them
3. Features with high importance are drift-prone
4. Remove or down-weight these features
```
In our case, this would have flagged `eopenrate` and `eclickrate` as unstable.

**Ensemble of Time-Period Models**
```
Model_2015 trained on 2008-2015
Model_2016 trained on 2008-2016
Model_2017 trained on 2008-2017
Final prediction = weighted average (more weight to recent models)
```
This hedges against any single time period's patterns dominating.

---

## Model Interpretability: Beyond SHAP

While SHAP (SHapley Additive exPlanations) is powerful, it's not the only interpretability technique. Different methods answer different questions:

### Comparison of Interpretation Methods

| Method | Question It Answers | Scope | Pros | Cons |
|--------|---------------------|-------|------|------|
| **SHAP** | How much did each feature contribute to this prediction? | Local & Global | Theoretically grounded, consistent | Slow for large datasets |
| **LIME** | Which features matter for this specific prediction? | Local | Fast, model-agnostic | Unstable, sensitive to parameters |
| **Permutation Importance** | How much does performance drop if we shuffle this feature? | Global | Simple, reliable | Correlated features problematic |
| **Partial Dependence (PDP)** | What's the average effect of changing this feature? | Global | Easy to understand | Assumes feature independence |
| **ICE (Individual Conditional Expectation)** | How does this feature affect each individual prediction? | Local | Shows heterogeneity | Can be noisy |
| **Accumulated Local Effects (ALE)** | What's the isolated effect of this feature? | Global | Handles correlated features | Harder to interpret |
| **Counterfactuals** | What minimal change would flip this prediction? | Local | Actionable insights | Multiple valid answers |
| **Anchors** | What conditions guarantee this prediction? | Local | Human-readable rules | May not exist for all cases |

### When to Use Each Method

**For Business Stakeholders (need simple explanations):**
- **Partial Dependence Plots**: "Customers who receive more than 20 emails have 90% retention"
- **Permutation Importance**: "Email count is the most important feature overall"
- **Counterfactuals**: "If this customer had received 5 more emails, they would likely have stayed"

**For Model Debugging (need detailed analysis):**
- **SHAP**: Understand exactly how each feature contributed
- **ICE plots**: See if effect varies across customer segments
- **ALE plots**: Handle correlated features correctly

**For Regulatory/Compliance (need audit trail):**
- **SHAP with force plots**: Show contribution breakdown for each decision
- **Anchors**: Provide rule-based explanations ("Customer flagged as high-risk because tenure < 30 days AND email_opens = 0")
- **Counterfactuals**: Demonstrate what would change the outcome

### Interpretation Example for Our Model

**Global View (Permutation Importance):**
```
Feature               Importance (AUC drop if shuffled)
─────────────────────────────────────────────────────
esent                 0.312  ████████████████████
eopenrate             0.089  █████
eclickrate            0.054  ███
city                  0.023  █
avgorder              0.008
```

**Local View (Counterfactual for a churned customer):**
```
Current:    esent=3, eopenrate=5%, prediction=CHURN (82% confidence)
Flip to:    esent=15 (+12 emails) → prediction=RETAIN (71% confidence)

Actionable insight: Send this customer more emails to retain them.
```

**Segment View (PDP showing non-linear effect):**
```
esent     Retention Probability
0-5       ████ 45%
5-15      ████████ 68%
15-30     ████████████ 85%
30+       █████████████ 92%

Insight: Diminishing returns after 30 emails - focus on customers in 5-15 range.
```

### Caution: Interpretation Pitfalls

1. **Correlation ≠ Causation**: `esent` predicts retention, but does sending more emails *cause* retention, or do retained customers simply receive more emails because they're active?

2. **Feature Interactions**: SHAP and Permutation Importance assume features act independently. In reality, `esent × eopenrate` interaction may matter more than either alone.

3. **Out-of-Distribution Explanations**: LIME and SHAP explanations can be misleading for unusual customers (e.g., someone with 200 emails sent - far outside training distribution).

4. **Simpson's Paradox**: A feature may have positive effect overall but negative effect within each segment. Always check segment-level patterns.

---

## Summary: Key Takeaways

### What We Learned

| Stage | Key Insight |
|-------|-------------|
| Data Discovery | Point-in-time snapshots prevent leakage |
| Column Analysis | 52% zero-inflation in `eclickrate` requires special handling |
| Quality | 99.7/100 score - minimal cleaning needed |
| Relationships | `esent` dominates with effect size d=2.551 |
| Capacity | EPV=1,150 allows abundant feature engineering |
| Modeling | AUC 0.9858 - excellent baseline performance |
| **Production** | **Simple models more robust to drift** |

### Decisions Made vs. Alternatives

| Decision | Why | Alternative | When Alternative Better |
|----------|-----|-------------|------------------------|
| Logistic Regression baseline | Strong linear signal (d=2.551) | Neural network | If non-linear patterns dominate |
| Balanced class weights | 3.8:1 is mild imbalance | SMOTE oversampling | If imbalance >10:1 |
| Log transform skewed features | Reduces outlier impact | Winsorization | If outliers are meaningful |
| Keep weak features | May help in combination | Drop features | If interpretability paramount |
| Cyclical encoding for day | Preserves circular structure | One-hot | If days don't wrap (Monday ≠ Sunday) |

### What Wasn't Explored (Future Work)

**Model Improvements:**
1. **Hyperparameter tuning** - Could improve AUC by 1-2%
2. **Feature selection** - SHAP-based or adversarial selection may reduce overfitting
3. **Segment-specific models** - Delhi (73.6% retention) may need different model
4. **Time-series features** - Trend in engagement over time

**Drift Robustness (See Stage 7):**
5. **Walk-forward cross-validation** - Better simulates production conditions
6. **Adversarial feature selection** - Remove drift-prone features like `eopenrate`
7. **Recency-weighted training** - Give more weight to recent samples
8. **Ensemble of time-period models** - Hedge against temporal patterns

**Interpretability (See Interpretability Section):**
9. **LIME for local explanations** - Faster alternative to SHAP
10. **Counterfactual analysis** - "What would change this prediction?"
11. **Partial Dependence Plots** - Visualize feature effects for stakeholders

**Production:**
12. **A/B testing framework** - Measure actual intervention effectiveness
13. **Online learning pipeline** - Continuous model updates
14. **Prediction intervals** - Quantify uncertainty in predictions

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

Set `DATA_PATH = "tests/fixtures/customer_retention_retail.csv"` in the first notebook.

---

## Next Steps

- [[Architecture]] - Understand the medallion architecture
- [[Temporal Framework]] - Leakage-safe data preparation
- [[Local Track]] - Generate production pipelines
- [[Databricks Track]] - Deploy to Databricks
