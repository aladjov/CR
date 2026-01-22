# Tutorial: Bank Customer Churn

> **Status: Prerequisites Only**
> This page describes how to set up the Bank Churn dataset. For a complete working example with executed notebooks and results, see **[[Tutorial: Retail Customer Retention|Tutorial-Retail-Churn]]**.

## Dataset Overview

The Bank Customer Churn dataset from Kaggle contains information about bank customers and whether they churned.

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - Bank Customer Churn](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) |
| **Rows** | ~10,000 customers |
| **Target** | `churn` (binary: 0/1) |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | Identifier | Unique customer ID |
| `credit_score` | Numeric | Customer's credit score |
| `country` | Categorical | Customer's country |
| `gender` | Categorical | Customer's gender |
| `age` | Numeric | Customer's age |
| `tenure` | Numeric | Years as customer |
| `balance` | Numeric | Account balance |
| `products_number` | Numeric | Number of products held |
| `credit_card` | Binary | Has credit card (0/1) |
| `active_member` | Binary | Is active member (0/1) |
| `estimated_salary` | Numeric | Estimated annual salary |
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
kaggle datasets download -d gauravtopre/bank-customer-churn-dataset -p tests/fixtures --unzip

# Option B: Manual download
# 1. Go to https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset
# 2. Download and extract to tests/fixtures/bank_customer_churn.csv
```

### 3. Run the Notebooks

```bash
jupyter lab exploration_notebooks/00_start_here.ipynb
```

In the first notebook, set:
```python
DATA_PATH = "tests/fixtures/bank_customer_churn.csv"
```

---

## Expected Characteristics

Based on similar banking datasets:

- **Granularity**: Entity-level (one row per customer)
- **Temporal columns**: Limited (tenure only)
- **Key predictors**: Age, geography, number of products
- **Class imbalance**: Typically ~20% churn rate

---

## Complete Example

For a fully executed tutorial with:
- All notebook outputs as HTML
- Actual model results (AUC, feature importance)
- Business alignment and ROI analysis

**See: [[Tutorial: Retail Customer Retention|Tutorial-Retail-Churn]]**
