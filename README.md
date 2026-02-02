# Customer Retention ML Framework
A structured backbone for the messy, iterative reality of ML model development. Exploration and production deployment are treated as parts of the same process -- not separate phases -- reflecting how data science actually works: you explore, decide, build, evaluate, learn something new, and circle back.

Handles both entity-level and event-level datasets. Experiments and production can share the same tables without copying data (Delta Lake), features are served consistently across training and inference (Feast / Feature Store), and every experiment is tracked and reproducible (MLflow). Runs locally or deploys to Databricks.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![CI](https://github.com/aladjov/CR/actions/workflows/ci.yaml/badge.svg)](https://github.com/aladjov/CR/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/aladjov/CR/branch/master/graph/badge.svg)](https://codecov.io/gh/aladjov/CR)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Typed](https://img.shields.io/badge/typed-Pydantic-blue)](https://docs.pydantic.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-integrated-0194E2?logo=mlflow)](https://mlflow.org/)
[![Databricks](https://img.shields.io/badge/Databricks-compatible-FF3621?logo=databricks)](https://databricks.com/)

---

## Why This Exists

Most ML tutorials jump straight to `model.fit()`. Real projects fail earlier -- in data issues you didn't notice, leakage you didn't check for, or feature choices you can't explain to your stakeholders three months later. This framework tries to close that gap.

It serves two audiences:

1. **If you're learning**, the notebooks walk through a realistic end-to-end process and explain the reasoning behind each step. Why does a 93-day median inter-event gap rule out short aggregation windows? Why might the model that wins validation degrade in production? The goal is to build intuition for the decisions that don't appear in textbooks.

2. **If you're experienced**, you can `pip install`, point to a new dataset, and get an opinionated exploration scaffold. The output is loosely-coupled production code (Bronze / Silver / Gold) with the provenance of every decision captured in self-contained HTML documentation -- useful when you need to explain *why* the pipeline does what it does.

### The approach

- **Exploration is a first-class concept.** The framework records what it found in the data, what it recommends, and why -- in versioned YAML artifacts. Each downstream transformation traces back to a specific observation in a specific notebook, so nothing happens without a documented reason.
- **Experimentation is version-controlled end to end.** Not just code and features, but the actual data observations and actions taken on them can be frozen in time together. Delta tables support time-travel on live production datasets, so you can always go back to what the data looked like when a decision was made.
- **Iteration is the default.** Model feedback -- feature importances, error analysis, drift signals -- feeds back into the next exploration cycle. The framework tracks iteration lineage rather than treating each experiment as independent.

---

## Quick Start

### 1. Install

```bash
pip install "churnkit[ml]"
```

### 2. Bootstrap notebooks into your project

```bash
churnkit-init --output ./my_project
cd my_project
```

### 3. Point to your data

Open `exploration_notebooks/01_data_discovery.ipynb` and set the data path:

```python
DATA_PATH = "experiments/data/your_file.csv"   # csv, parquet, or delta
```

### 4. Run

Execute cells sequentially. The framework auto-detects column types, data granularity (entity vs event-level), text columns, and temporal patterns -- then routes you through the relevant notebooks.

Findings, recommendations, and production pipeline specs are generated as you go.

---

## Learn More

Detailed documentation lives in the [Wiki](https://github.com/aladjov/CR/wiki):

| Topic | Wiki Page |
|-------|-----------|
| Installation options & environment setup | [Getting Started](https://github.com/aladjov/CR/wiki/Getting-Started) |
| Medallion architecture & system design | [Architecture](https://github.com/aladjov/CR/wiki/Architecture) |
| Notebook workflow & iteration tracking | [Exploration Loop](https://github.com/aladjov/CR/wiki/Exploration-Loop) |
| Leakage-safe temporal data preparation | [Temporal Framework](https://github.com/aladjov/CR/wiki/Temporal-Framework) |
| Feast & Databricks feature management | [Feature Store](https://github.com/aladjov/CR/wiki/Feature-Store) |
| Local execution with Feast + MLFlow | [Local Track](https://github.com/aladjov/CR/wiki/Local-Track) |
| Databricks with Unity Catalog + Delta Lake | [Databricks Track](https://github.com/aladjov/CR/wiki/Databricks-Track) |

### Tutorials

| Tutorial | What it walks through |
|----------|-----------------------|
| [Retail Customer Retention](https://github.com/aladjov/CR/wiki/Tutorial-Retail-Churn) | Entity-level data: point-in-time snapshots, quality assessment, baseline models, and a production scoring check that reveals how distribution drift affects different model families -- [browse HTML](https://aladjov.github.io/CR/tutorial/retail-churn/) |
| [Customer Email Engagement](https://github.com/aladjov/CR/wiki/Tutorial-Customer-Emails) | Event-level data: temporal window selection driven by inter-event cadence, aggregating 83K email events into customer-level features, and tracing each decision from data observation to production pipeline -- [browse HTML](https://aladjov.github.io/CR/tutorial/customer-emails/) |
| [Bank Customer Churn](https://github.com/aladjov/CR/wiki/Tutorial-Bank-Churn) | Dataset setup instructions |
| [Netflix Churn](https://github.com/aladjov/CR/wiki/Tutorial-Netflix-Churn) | Dataset setup instructions |

### [Acknowledgments](https://github.com/aladjov/CR/wiki/Acknowledgments)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 -- See [LICENSE](LICENSE) for details.
