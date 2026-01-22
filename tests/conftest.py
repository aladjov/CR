import os
import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
from customer_retention.core.config import (
    ColumnType, ColumnConfig, DataSourceConfig,
    SourceType, FileFormat, PipelineConfig
)
from customer_retention.core.compat.detection import is_databricks, is_spark_available


def pytest_configure(config):
    """Register custom markers and configure auto-skip for databricks/spark tests."""
    config.addinivalue_line("markers", "databricks: tests requiring Databricks runtime")
    config.addinivalue_line("markers", "spark: tests requiring PySpark")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with databricks or spark when environment not available."""
    skip_databricks = pytest.mark.skip(reason="Requires Databricks runtime (DATABRICKS_RUNTIME_VERSION not set)")
    skip_spark = pytest.mark.skip(reason="Requires PySpark (not installed)")

    for item in items:
        if "databricks" in item.keywords and not is_databricks():
            item.add_marker(skip_databricks)
        if "spark" in item.keywords and not is_spark_available():
            item.add_marker(skip_spark)


@pytest.fixture
def requires_databricks():
    """Skip test if not running in Databricks environment."""
    if not is_databricks():
        pytest.skip("Requires Databricks runtime")


@pytest.fixture
def requires_spark():
    """Skip test if PySpark is not available."""
    if not is_spark_available():
        pytest.skip("Requires PySpark")


@pytest.fixture
def mock_databricks_env(monkeypatch):
    """Mock Databricks environment for testing Databricks-specific code paths."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    yield
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)


@pytest.fixture
def sample_columns():
    return [
        ColumnConfig(name="custid", column_type=ColumnType.IDENTIFIER),
        ColumnConfig(name="retained", column_type=ColumnType.TARGET, nullable=False),
        ColumnConfig(name="created", column_type=ColumnType.DATETIME),
        ColumnConfig(name="firstorder", column_type=ColumnType.DATETIME),
        ColumnConfig(name="lastorder", column_type=ColumnType.DATETIME),
        ColumnConfig(name="esent", column_type=ColumnType.NUMERIC_DISCRETE),
        ColumnConfig(name="eopenrate", column_type=ColumnType.NUMERIC_CONTINUOUS),
        ColumnConfig(name="eclickrate", column_type=ColumnType.NUMERIC_CONTINUOUS),
        ColumnConfig(name="avgorder", column_type=ColumnType.NUMERIC_CONTINUOUS),
        ColumnConfig(name="ordfreq", column_type=ColumnType.NUMERIC_CONTINUOUS),
        ColumnConfig(name="paperless", column_type=ColumnType.BINARY),
        ColumnConfig(name="refill", column_type=ColumnType.BINARY),
        ColumnConfig(name="doorstep", column_type=ColumnType.BINARY),
        ColumnConfig(name="favday", column_type=ColumnType.CATEGORICAL_CYCLICAL, cyclical_max=7),
        ColumnConfig(name="city", column_type=ColumnType.CATEGORICAL_NOMINAL),
    ]


@pytest.fixture
def sample_source_config(sample_columns):
    return DataSourceConfig(
        name="customer_master",
        source_type=SourceType.BATCH_FILE,
        path="/data/customer_retention_retail.csv",
        file_format=FileFormat.CSV,
        primary_key="custid",
        columns=sample_columns,
        timestamp_column="created",
        customer_key="custid"
    )


@pytest.fixture
def sample_pipeline_config(sample_source_config):
    return PipelineConfig(
        project_name="retail_churn_2024",
        data_sources=[sample_source_config]
    )


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "custid": [1, 2, 3, 4, 5],
        "retained": [1, 0, 1, 1, 0],
        "created": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "firstorder": pd.to_datetime(["2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09"]),
        "lastorder": pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-03", "2023-06-04", "2023-06-05"]),
        "esent": [10, 20, 15, 30, 25],
        "eopenrate": [0.5, 0.6, 0.55, 0.7, 0.65],
        "eclickrate": [0.1, 0.15, 0.12, 0.2, 0.18],
        "avgorder": [50.0, 60.0, 55.0, 70.0, 65.0],
        "ordfreq": [2.0, 3.0, 2.5, 4.0, 3.5],
        "paperless": [1, 0, 1, 1, 0],
        "refill": [0, 1, 1, 0, 1],
        "doorstep": [1, 1, 0, 1, 0],
        "favday": [1, 3, 5, 2, 4],
        "city": ["NYC", "LA", "CHI", "NYC", "LA"]
    })


@pytest.fixture
def temp_csv_file(sample_dataframe, tmp_path):
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def temp_parquet_file(sample_dataframe, tmp_path):
    parquet_path = tmp_path / "test_data.parquet"
    sample_dataframe.to_parquet(parquet_path, index=False)
    return str(parquet_path)


@pytest.fixture
def temp_registry_file(tmp_path):
    return str(tmp_path / "registry.json")


@pytest.fixture
def retail_dataset_path():
    return str(Path(__file__).parent / "fixtures" / "customer_retention_retail.csv")


@pytest.fixture
def retail_dataframe(retail_dataset_path):
    return pd.read_csv(retail_dataset_path)
