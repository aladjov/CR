# Test Fixtures

Sample datasets for testing and notebook exploration.

## Datasets

| File | Description | Rows | Included | Source |
|------|-------------|------|----------|--------|
| `customer_retention_retail.csv` | Retail customer retention | ~31K | Yes | - |
| `bank_customer_churn.csv` | Bank customer churn | ~10K | No | [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) |
| `netflix_customer_churn.csv` | Netflix subscription churn | ~10K | No | [Kaggle](https://www.kaggle.com/datasets/vasifasad/netflix-customer-churn-prediction) |

## Downloading Kaggle Datasets

The Kaggle datasets are not included in the repository. To download them:

1. Run the prerequisites notebook: `templates/experiments/00_prerequisites.ipynb`
2. Or use the Kaggle CLI directly:
   ```bash
   kaggle datasets download -d gauravtopre/bank-customer-churn-dataset -p tests/fixtures --unzip
   kaggle datasets download -d vasifasad/netflix-customer-churn-prediction -p tests/fixtures --unzip
   ```

## Usage

These datasets can be used with the exploration notebooks:

```python
# In templates/experiments/01_data_discovery.ipynb
DATA_PATH = "../../../tests/fixtures/customer_retention_retail.csv"
DATA_PATH = "../../../tests/fixtures/bank_customer_churn.csv"
DATA_PATH = "../../../tests/fixtures/netflix_customer_churn.csv"
```

## License

These datasets are provided for educational and testing purposes only.
Refer to the original Kaggle dataset pages for licensing information.
