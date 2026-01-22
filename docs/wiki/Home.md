# Customer Retention ML Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/aladjov/CR/blob/master/LICENSE)
[![CI](https://github.com/aladjov/CR/actions/workflows/ci.yaml/badge.svg)](https://github.com/aladjov/CR/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/aladjov/CR/branch/master/graph/badge.svg)](https://codecov.io/gh/aladjov/CR)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Typed](https://img.shields.io/badge/typed-Pydantic-blue)](https://docs.pydantic.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-integrated-0194E2?logo=mlflow)](https://mlflow.org/)
[![Databricks](https://img.shields.io/badge/Databricks-compatible-FF3621?logo=databricks)](https://databricks.com/)

A hands-on framework for learning and implementing customer churn prediction pipelines. Built for data scientists who want to understand the full ML lifecycle—from raw data exploration to production deployment.

## What This Project Is

This is an **educational framework** that walks you through building a customer retention ML pipeline step-by-step. Instead of a black-box solution, you get:

- **Guided notebooks** that teach the *why* behind each step, not just the *how*
- **Modular components** you can understand, modify, and extend
- **A foundation to build on** as you add more sophisticated techniques

## Current Capabilities

| Area | What's Implemented |
|------|-------------------|
| **Data Exploration** | Automatic type detection, quality profiling, basic statistics |
| **Text Processing** | Embeddings + PCA dimensionality reduction for unstructured text columns |
| **Time Series Support** | Entity lifecycle analysis, temporal patterns, trend/seasonality detection |
| **Multi-Dataset Analysis** | Auto-detect relationships, join suggestions, time-window aggregations |
| **Cleaning** | Missing value imputation, outlier handling (IQR, Z-score, Winsorization) |
| **Feature Engineering** | Temporal features, categorical encoding, basic interactions |
| **Modeling** | Logistic Regression, Random Forest, XGBoost, LightGBM baselines |
| **Evaluation** | Standard metrics (AUC, precision, recall), threshold tuning |
| **Interpretability** | SHAP values, feature importance |
| **Monitoring** | Basic drift detection (KS test, PSI), performance tracking |
| **Iteration Support** | Version tracking, recommendation status, feedback loops |
| **Scoring Pipeline** | Production simulation with holdout validation |

## Wiki Contents

- [[Getting Started]] - Installation and quick examples
- [[Architecture]] - Medallion architecture and system design
- [[Exploration Loop]] - Interactive notebook workflow
- [[Temporal Framework]] - Leakage-safe data preparation
- [[Feature Store]] - Feast and Databricks feature management
- [[Local Track]] - Feast + MLFlow execution path
- [[Databricks Track]] - Unity Catalog + Delta Lake execution path

### Tutorials

- [[Tutorial: Retail Customer Retention|Tutorial-Retail-Churn]] - **Complete example with executed notebooks**
  - 17 notebooks executed with actual results
  - Model performance: AUC 0.9858, PR-AUC 0.9949
  - [Browse HTML outputs](tutorial/index.html)
- [[Tutorial: Bank Customer Churn|Tutorial-Bank-Churn]] - Dataset setup instructions
- [[Tutorial: Netflix Churn|Tutorial-Netflix-Churn]] - Dataset setup instructions

## Quick Start

```bash
# Clone the repository
git clone https://github.com/aladjov/CR.git
cd CR

# Install with ML dependencies
pip install -e ".[dev,ml]"

# Start exploring
jupyter lab exploration_notebooks/00_start_here.ipynb
```

## License

Apache 2.0 - See [LICENSE](https://github.com/aladjov/CR/blob/master/LICENSE) for details.
