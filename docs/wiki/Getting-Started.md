# Getting Started

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Installation Options

```bash
# Basic (exploration & profiling only)
pip install customer-retention

# With ML models (recommended)
pip install "customer-retention[ml]"
# Includes: scikit-learn, xgboost, lightgbm, shap, mlflow, imbalanced-learn

# With text processing (for TEXT column embeddings)
pip install "customer-retention[text]"
# Includes: sentence-transformers

# Full installation (ML + text processing)
pip install "customer-retention[ml,text]"

# Development (includes testing tools)
pip install "customer-retention[dev,ml]"
```

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager:

```bash
pip install uv
uv pip install -e ".[dev,ml]"
```

## Quick Example

```python
from customer_retention.analysis.auto_explorer import DataExplorer

# Point to your data and start exploring
explorer = DataExplorer(visualize=True, save_findings=True)
findings = explorer.explore("your_customer_data.csv")

# What you get:
# - Automatic column type detection
# - Data quality report
# - Target variable identification
# - Initial feature recommendations
```

## Working with Multiple Datasets

Real churn analysis often involves multiple data sources. The framework auto-detects relationships:

```python
from customer_retention.analysis.auto_explorer import ExplorationManager

# After exploring each dataset individually...
manager = ExplorationManager(explorations_dir="./experiments/findings")

# Discover all explored datasets
datasets = manager.list_datasets()
# Shows: customers (entity-level), transactions (event-level), support_tickets (event-level)

# Create multi-dataset analysis
multi = manager.create_multi_dataset_findings()

# Auto-detected:
# - Primary entity: customers (has target column)
# - Event datasets: transactions, support_tickets
# - Join keys: customer_id
```

## Time-Window Aggregations

For event-level data, aggregate to entity level with configurable windows:

```python
from customer_retention.stages.profiling import TimeWindowAggregator

aggregator = TimeWindowAggregator(
    entity_column="customer_id",
    time_column="transaction_date"
)

# Create features across time windows
features = aggregator.aggregate(
    transactions_df,
    windows=["7d", "30d", "90d"],
    value_columns=["amount"],
    agg_funcs=["sum", "mean", "count"],
    include_recency=True  # days_since_last_event
)

# Result: One row per customer with features like:
# - amount_sum_7d, amount_sum_30d, amount_sum_90d
# - amount_mean_7d, amount_mean_30d, amount_mean_90d
# - event_count_7d, event_count_30d, event_count_90d
# - days_since_last_event
```

## Start with the Notebooks

The best way to learn is to work through the exploration notebooks with your own data:

```bash
# Bootstrap a new project with all notebooks
python scripts/notebooks/init_project.py --output ./my_churn_project --name "My Churn Analysis"

# Or open the exploration notebooks directly
jupyter lab exploration_notebooks/01_data_discovery.ipynb
```

## Exploration Notebooks Overview

```
Exploration Notebooks (exploration_notebooks/)
────────────────────────────────────────────────────────
00. Start Here            → Quick orientation and setup
01. Data Discovery        → Get to know your dataset (auto-detects granularity)
02. Column Deep Dive      → Understand each feature's distribution
03. Quality Assessment    → Find and fix data issues
04. Relationship Analysis → Discover feature-target patterns

05. Multi-Dataset         → Combine datasets, select which to include
06. Feature Opportunities → Identify engineering possibilities
07. Modeling Readiness    → Capture recommendations, validate before training
08. Baseline Experiments  → Establish performance benchmarks
09. Business Alignment    → Connect ML to business goals
10. Spec Generation       → Generate production pipeline specs
```

## Next Steps

- [[Architecture]] - Understand the medallion architecture
- [[Exploration Loop]] - Deep dive into the notebook workflow
- [[Tutorial: Bank Customer Churn]] - Hands-on tutorial with real data
