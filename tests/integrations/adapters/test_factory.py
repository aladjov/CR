from unittest.mock import patch

import pytest

try:
    import deltalake
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

requires_delta = pytest.mark.skipif(not DELTA_AVAILABLE, reason="deltalake not installed")
requires_mlflow = pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="mlflow not installed")


@requires_delta
class TestFactoryFunctions:
    def test_get_delta_returns_storage(self):
        from customer_retention.integrations.adapters import get_delta
        from customer_retention.integrations.adapters.storage import DeltaStorage
        storage = get_delta()
        assert isinstance(storage, DeltaStorage)

    def test_get_feature_store_returns_adapter(self, tmp_path):
        from customer_retention.integrations.adapters import get_feature_store
        from customer_retention.integrations.adapters.feature_store import FeatureStoreAdapter
        store = get_feature_store(base_path=str(tmp_path))
        assert isinstance(store, FeatureStoreAdapter)

    @requires_mlflow
    def test_get_mlflow_returns_adapter(self, tmp_path):
        from customer_retention.integrations.adapters import get_mlflow
        from customer_retention.integrations.adapters.mlflow import MLflowAdapter
        adapter = get_mlflow(tracking_uri=str(tmp_path))
        assert isinstance(adapter, MLflowAdapter)


@requires_delta
class TestFactoryEnvironmentDetection:
    def test_get_delta_returns_local_by_default(self):
        from customer_retention.core.compat.detection import is_spark_available
        from customer_retention.integrations.adapters import get_delta
        from customer_retention.integrations.adapters.storage import LocalDelta
        if not is_spark_available():
            storage = get_delta()
            assert isinstance(storage, LocalDelta)

    def test_get_delta_returns_databricks_when_available(self):
        from customer_retention.core.compat.detection import is_spark_available
        from customer_retention.integrations.adapters import get_delta
        from customer_retention.integrations.adapters.storage import DatabricksDelta
        if is_spark_available():
            storage = get_delta()
            assert isinstance(storage, DatabricksDelta)


@requires_delta
class TestFactoryWithForcedEnvironment:
    def test_get_delta_local_forced(self):
        from customer_retention.integrations.adapters import get_delta
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = get_delta(force_local=True)
        assert isinstance(storage, LocalDelta)

    def test_get_feature_store_local_forced(self, tmp_path):
        from customer_retention.integrations.adapters import get_feature_store
        from customer_retention.integrations.adapters.feature_store import LocalFeatureStore
        store = get_feature_store(base_path=str(tmp_path), force_local=True)
        assert isinstance(store, LocalFeatureStore)

    @requires_mlflow
    def test_get_mlflow_local_forced(self, tmp_path):
        from customer_retention.integrations.adapters import get_mlflow
        from customer_retention.integrations.adapters.mlflow import LocalMLflow
        adapter = get_mlflow(tracking_uri=str(tmp_path), force_local=True)
        assert isinstance(adapter, LocalMLflow)


class TestFactoryDeltaFallback:
    @patch("customer_retention.integrations.adapters.storage.databricks.is_spark_available", return_value=True)
    @patch("customer_retention.integrations.adapters.factory.is_spark_available", return_value=True)
    @patch("customer_retention.integrations.adapters.storage.local.deltalake_available", return_value=False)
    def test_force_local_falls_back_to_databricks_when_no_deltalake(self, *_):
        from customer_retention.integrations.adapters import get_delta
        from customer_retention.integrations.adapters.storage import DatabricksDelta
        storage = get_delta(force_local=True)
        assert isinstance(storage, DatabricksDelta)

    @requires_delta
    def test_force_local_returns_local_when_deltalake_available(self):
        from customer_retention.integrations.adapters import get_delta
        from customer_retention.integrations.adapters.storage import LocalDelta
        storage = get_delta(force_local=True)
        assert isinstance(storage, LocalDelta)

    @patch("customer_retention.integrations.adapters.factory.is_spark_available", return_value=False)
    @patch("customer_retention.integrations.adapters.storage.local.deltalake_available", return_value=False)
    def test_raises_when_no_backend_available(self, *_):
        from customer_retention.integrations.adapters import get_delta
        with pytest.raises(ImportError):
            get_delta()


@requires_delta
class TestFactoryWithMockedSparkAvailable:
    @patch("customer_retention.integrations.adapters.storage.databricks.is_spark_available", return_value=True)
    @patch("customer_retention.integrations.adapters.factory.is_spark_available", return_value=True)
    def test_get_delta_returns_databricks_when_spark_available(self, mock_factory_spark, mock_storage_spark):
        from customer_retention.integrations.adapters import get_delta
        from customer_retention.integrations.adapters.storage import DatabricksDelta
        storage = get_delta()
        assert isinstance(storage, DatabricksDelta)

    @patch("customer_retention.integrations.adapters.feature_store.databricks.is_spark_available", return_value=True)
    @patch("customer_retention.integrations.adapters.factory.is_spark_available", return_value=True)
    def test_get_feature_store_returns_databricks_when_spark_available(self, mock_factory_spark, mock_fs_spark):
        from customer_retention.integrations.adapters import get_feature_store
        from customer_retention.integrations.adapters.feature_store import DatabricksFeatureStore
        store = get_feature_store()
        assert isinstance(store, DatabricksFeatureStore)

    @requires_mlflow
    @patch("customer_retention.integrations.adapters.mlflow.databricks.is_spark_available", return_value=True)
    @patch("customer_retention.integrations.adapters.factory.is_spark_available", return_value=True)
    def test_get_mlflow_returns_databricks_when_spark_available(self, mock_factory_spark, mock_mlflow_spark):
        from customer_retention.integrations.adapters import get_mlflow
        from customer_retention.integrations.adapters.mlflow import DatabricksMLflow
        adapter = get_mlflow()
        assert isinstance(adapter, DatabricksMLflow)
