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

An **ML governance backbone** for customer retention — designed both as an educational walk-through and as a production-grade lifecycle framework. It covers the full loop: **explore → experiment → release → monitor → retrain**.

- **Guided notebooks** that teach the *why* behind each step, not just the *how*
- **Modular components** you can understand, modify, and extend
- **Governance built in** — Delta Lake storage with ACID transactions and time travel, fit/transform separation for scoring integrity, and multi-stage validation gates

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
| **Storage** | Delta Lake at every layer — ACID transactions, time travel, version tracking; shared between experiments and production |
| **Transforms** | Fit/transform separation with artifact persistence (scalers, encoders, power transforms via `ArtifactStore` + `manifest.yaml`) |
| **Monitoring** | Drift detection (KS test, PSI), performance tracking |
| **Iteration Support** | Version tracking, recommendation status, feedback loops |
| **Scoring Pipeline** | Holdout validation, adversarial pipeline checks, transformation consistency gates |
| **Scoring Validation** | Multi-stage gates: feature drift detection, prediction consistency, adversarial checks, Delta version comparison |

## Wiki Contents

- [[Getting Started]] - Installation and quick examples
- [[Architecture]] - Medallion architecture and system design
- [[Exploration Loop]] - Interactive notebook workflow
- [[Model Intent and Objective Support]] - How prediction objectives are declared and validated
- [[Snapshot Grid and Control Variables]] - Leakage-safe temporal grid and control variables
- [[Transforms & Scoring Validation|Transforms-and-Scoring-Validation]] - Fit/transform separation and validation gates
- [[Feature Store]] - Feast and Databricks feature management
- [[Local Track]] - Feast + MLFlow execution path
- [[Databricks Track]] - Unity Catalog + Delta Lake execution path

### Tutorials

- [[Tutorial: Retail Customer Retention|Tutorial-Retail-Churn]] - **Complete example with executed notebooks**
  - 11 notebooks executed with actual results
  - Illustrates distribution drift 
  - [Browse HTML Tutorial](https://aladjov.github.io/CR/tutorial/retail-churn/)
- [[Tutorial: Customer Email Engagement|Tutorial-Customer-Emails]] - **Event-based pipeline with temporal aggregation**
  - 16 notebooks executed with customer emails dataset
  - Demonstrates event-level aggregation
  - [Browse HTML Tutorial](https://aladjov.github.io/CR/tutorial/customer-emails/)
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
