# Tutorial: Netflix Churn Prediction

> **Status: Prerequisites Only**
> This page describes how to set up the Netflix Churn dataset. For a complete working example with executed notebooks and results, see **[[Tutorial: Retail Customer Retention|Tutorial-Retail-Churn]]**.

## Dataset Overview

The Netflix Customer Churn dataset from Kaggle contains subscriber information and engagement metrics.

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - Netflix Customer Churn](https://www.kaggle.com/datasets/vasifasad/netflix-customer-churn-prediction) |
| **Rows** | ~100,000 subscribers |
| **Target** | `churn` (binary: 0/1) |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | Identifier | Unique subscriber ID |
| `subscription_type` | Categorical | Basic/Standard/Premium |
| `monthly_revenue` | Numeric | Monthly subscription fee |
| `join_date` | Datetime | Subscription start date |
| `last_payment_date` | Datetime | Last payment received |
| `country` | Categorical | Subscriber's country |
| `age` | Numeric | Subscriber's age |
| `gender` | Categorical | Subscriber's gender |
| `device` | Categorical | Primary viewing device |
| `plan_duration` | Numeric | Months subscribed |
| `active_profiles` | Numeric | Number of active profiles |
| `movies_watched` | Numeric | Total movies watched |
| `series_watched` | Numeric | Total series watched |
| `churn` | Target | Churned (0/1) |

---

## Prerequisites

### 1. Install the Framework

```bash
git clone https://github.com/aladjov/CR.git
cd CR
pip install -e ".[dev,ml]"
```

### 2. Download the Dataset

```bash
# Option A: Using Kaggle CLI
pip install kaggle
kaggle datasets download -d vasifasad/netflix-customer-churn-prediction -p tests/fixtures --unzip

# Option B: Manual download
# 1. Go to https://www.kaggle.com/datasets/vasifasad/netflix-customer-churn-prediction
# 2. Download and extract to tests/fixtures/netflix_customer_churn.csv
```

### 3. Run the Notebooks

```bash
jupyter lab exploration_notebooks/00_start_here.ipynb
```

In the first notebook, set:
```python
DATA_PATH = "tests/fixtures/netflix_customer_churn.csv"
```

---

## Expected Characteristics

Based on the dataset structure:

- **Granularity**: Entity-level (one row per subscriber)
- **Temporal columns**: `join_date`, `last_payment_date`
- **Key predictors**: Engagement metrics (movies/series watched), subscription type
- **Derived features**: Content per month, days since last payment

---

## Complete Example

For a fully executed tutorial with:
- All notebook outputs as HTML
- Actual model results (AUC, feature importance)
- Business alignment and ROI analysis

**See: [[Tutorial: Retail Customer Retention|Tutorial-Retail-Churn]]**
