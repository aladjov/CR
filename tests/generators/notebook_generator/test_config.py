import pytest
from dataclasses import FrozenInstanceError


class TestMLflowConfig:
    def test_default_values(self):
        from customer_retention.generators.notebook_generator.config import MLflowConfig
        config = MLflowConfig()
        assert config.tracking_uri == "./experiments/mlruns"
        assert config.registry_uri is None
        assert config.experiment_name == "customer_retention"
        assert config.model_name == "churn_model"

    def test_custom_values(self):
        from customer_retention.generators.notebook_generator.config import MLflowConfig
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            registry_uri="databricks-uc",
            experiment_name="custom_exp",
            model_name="custom_model"
        )
        assert config.tracking_uri == "http://localhost:5000"
        assert config.registry_uri == "databricks-uc"


class TestFeatureStoreConfig:
    def test_default_values(self):
        from customer_retention.generators.notebook_generator.config import FeatureStoreConfig
        config = FeatureStoreConfig()
        assert config.base_path == "./experiments/feature_store"
        assert config.catalog == "main"
        assert config.schema == "default"
        assert config.table_name == "customer_features"

    def test_databricks_config(self):
        from customer_retention.generators.notebook_generator.config import FeatureStoreConfig
        config = FeatureStoreConfig(catalog="ml_catalog", schema="features", table_name="churn_features")
        assert config.catalog == "ml_catalog"
        assert config.schema == "features"


class TestNotebookConfig:
    def test_default_values(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig, Platform
        config = NotebookConfig()
        assert config.project_name == "customer_retention"
        assert config.platform == Platform.LOCAL
        assert config.model_type == "xgboost"
        assert config.test_size == 0.2
        assert config.threshold == 0.5

    def test_databricks_platform(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig, Platform
        config = NotebookConfig(platform=Platform.DATABRICKS)
        assert config.platform == Platform.DATABRICKS

    def test_mlflow_config_embedded(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig, MLflowConfig
        mlflow_cfg = MLflowConfig(experiment_name="test_exp")
        config = NotebookConfig(mlflow=mlflow_cfg)
        assert config.mlflow.experiment_name == "test_exp"

    def test_feature_store_config_embedded(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig, FeatureStoreConfig
        fs_cfg = FeatureStoreConfig(table_name="my_features")
        config = NotebookConfig(feature_store=fs_cfg)
        assert config.feature_store.table_name == "my_features"

    def test_use_framework_true_for_local(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig, Platform
        config = NotebookConfig(platform=Platform.LOCAL)
        assert config.use_framework is True

    def test_use_framework_false_for_databricks(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig, Platform
        config = NotebookConfig(platform=Platform.DATABRICKS)
        assert config.use_framework is False


class TestPlatformEnum:
    def test_local_value(self):
        from customer_retention.generators.notebook_generator.config import Platform
        assert Platform.LOCAL.value == "local"

    def test_databricks_value(self):
        from customer_retention.generators.notebook_generator.config import Platform
        assert Platform.DATABRICKS.value == "databricks"

    def test_enum_members(self):
        from customer_retention.generators.notebook_generator.config import Platform
        assert len(Platform) == 2
