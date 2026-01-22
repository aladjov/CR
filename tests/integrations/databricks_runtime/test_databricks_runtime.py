"""
Databricks Runtime Integration Tests.

These tests are automatically skipped when not running in a Databricks environment.
They verify that Databricks-specific adapters work correctly with actual Spark/Delta.

To run these tests:
    1. Upload this test file to a Databricks workspace
    2. Run from a notebook or job with: pytest test_databricks_runtime.py -v

The tests use the @pytest.mark.databricks marker which auto-skips when
DATABRICKS_RUNTIME_VERSION environment variable is not set.
"""


import pandas as pd
import pytest


@pytest.mark.databricks
class TestDatabricksDeltaStorage:
    """Test DatabricksDelta adapter with real Spark/Delta."""

    def test_delta_write_and_read(self, tmp_path):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        storage = DatabricksDelta()
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        path = str(tmp_path / "test_delta")
        storage.write(df, path)
        result = storage.read(path)
        assert len(result) == 3
        assert list(result.columns) == ["id", "value"]

    def test_delta_overwrite_mode(self, tmp_path):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        storage = DatabricksDelta()
        path = str(tmp_path / "test_overwrite")
        df1 = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        df2 = pd.DataFrame({"id": [3, 4, 5], "value": ["c", "d", "e"]})
        storage.write(df1, path)
        storage.write(df2, path, mode="overwrite")
        result = storage.read(path)
        assert len(result) == 3

    def test_delta_append_mode(self, tmp_path):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        storage = DatabricksDelta()
        path = str(tmp_path / "test_append")
        df1 = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        df2 = pd.DataFrame({"id": [3, 4], "value": ["c", "d"]})
        storage.write(df1, path)
        storage.write(df2, path, mode="append")
        result = storage.read(path)
        assert len(result) == 4

    def test_delta_partition_by(self, tmp_path):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        storage = DatabricksDelta()
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "category": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40]
        })
        path = str(tmp_path / "test_partitioned")
        storage.write(df, path, partition_by=["category"])
        result = storage.read(path)
        assert len(result) == 4

    def test_delta_version_read(self, tmp_path):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        storage = DatabricksDelta()
        path = str(tmp_path / "test_versions")
        df1 = pd.DataFrame({"id": [1], "value": ["v1"]})
        df2 = pd.DataFrame({"id": [2], "value": ["v2"]})
        storage.write(df1, path)
        storage.write(df2, path, mode="overwrite")
        result_v0 = storage.read(path, version=0)
        result_latest = storage.read(path)
        assert result_v0["value"].iloc[0] == "v1"
        assert result_latest["value"].iloc[0] == "v2"

    def test_delta_history(self, tmp_path):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        storage = DatabricksDelta()
        path = str(tmp_path / "test_history")
        df = pd.DataFrame({"id": [1], "value": ["initial"]})
        storage.write(df, path)
        storage.write(df, path, mode="overwrite")
        history = storage.history(path)
        assert len(history) >= 2
        assert "version" in history[0]

    def test_delta_merge(self, tmp_path):
        from customer_retention.integrations.adapters.storage.databricks import DatabricksDelta
        storage = DatabricksDelta()
        path = str(tmp_path / "test_merge")
        target = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        source = pd.DataFrame({"id": [2, 3], "value": ["updated", "new"]})
        storage.write(target, path)
        storage.merge(source, path, condition="target.id = source.id")
        result = storage.read(path)
        assert len(result) == 3
        row2 = result[result["id"] == 2]["value"].iloc[0]
        assert row2 == "updated"


@pytest.mark.databricks
class TestDatabricksMLflow:
    """Test DatabricksMLflow adapter with real MLflow."""

    def test_mlflow_start_and_end_run(self):
        from customer_retention.integrations.adapters.mlflow.databricks import DatabricksMLflow
        adapter = DatabricksMLflow()
        run_id = adapter.start_run(
            experiment_name="/test/customer_retention_test",
            run_name="test_run"
        )
        assert run_id is not None
        adapter.end_run()

    def test_mlflow_log_params(self):
        from customer_retention.integrations.adapters.mlflow.databricks import DatabricksMLflow
        adapter = DatabricksMLflow()
        adapter.start_run(experiment_name="/test/customer_retention_test")
        adapter.log_params({"learning_rate": 0.01, "max_depth": 5})
        adapter.end_run()

    def test_mlflow_log_metrics(self):
        from customer_retention.integrations.adapters.mlflow.databricks import DatabricksMLflow
        adapter = DatabricksMLflow()
        adapter.start_run(experiment_name="/test/customer_retention_test")
        adapter.log_metrics({"accuracy": 0.95, "f1_score": 0.92})
        adapter.end_run()

    def test_mlflow_log_and_load_model(self):
        from sklearn.linear_model import LogisticRegression

        from customer_retention.integrations.adapters.mlflow.databricks import DatabricksMLflow
        adapter = DatabricksMLflow()
        adapter.start_run(experiment_name="/test/customer_retention_test")
        model = LogisticRegression()
        model.fit([[1, 2], [3, 4]], [0, 1])
        model_uri = adapter.log_model(model, "test_model")
        loaded_model = adapter.load_model(model_uri)
        assert hasattr(loaded_model, "predict")
        adapter.end_run()


@pytest.mark.databricks
class TestDatabricksFeatureStore:
    """Test Databricks Feature Store adapter."""

    def test_feature_store_create_table(self):
        from customer_retention.integrations.adapters.feature_store.databricks import DatabricksFeatureStore
        fs = DatabricksFeatureStore(catalog="hive_metastore", schema="default")
        schema = {
            "customer_id": "int",
            "feature1": "float",
            "feature2": "int"
        }
        result = fs.create_table(
            name="test_customer_features",
            schema=schema,
            primary_keys=["customer_id"]
        )
        assert result.success

    def test_feature_store_write_and_read(self):
        from customer_retention.integrations.adapters.feature_store.databricks import DatabricksFeatureStore
        fs = DatabricksFeatureStore(catalog="hive_metastore", schema="default")
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "feature1": [0.5, 0.6, 0.7],
        })
        fs.write_table(name="test_features_rw", df=df, mode="overwrite")
        result = fs.read_table("test_features_rw")
        assert len(result) == 3


@pytest.mark.spark
class TestSparkIntegration:
    """Test PySpark integration (runs if PySpark available, not necessarily Databricks)."""

    def test_spark_session_available(self):
        from customer_retention.core.compat.detection import get_spark_session
        spark = get_spark_session()
        assert spark is not None
        assert hasattr(spark, "sql")

    def test_spark_dataframe_conversion(self):
        from customer_retention.core.compat.detection import get_spark_session
        spark = get_spark_session()
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        sdf = spark.createDataFrame(pdf)
        result = sdf.toPandas()
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_spark_sql_execution(self):
        from customer_retention.core.compat.detection import get_spark_session
        spark = get_spark_session()
        result = spark.sql("SELECT 1 as value").toPandas()
        assert result["value"].iloc[0] == 1
