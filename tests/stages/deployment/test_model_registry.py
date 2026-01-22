import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from sklearn.ensemble import RandomForestClassifier
from customer_retention.stages.deployment import (
    ModelRegistry, ModelStage, ModelMetadata, RegistrationResult
)


class TestModelRegistration:
    def test_registers_model_successfully(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_mlflow.register_model.return_value = MagicMock(version="1")
            registry = ModelRegistry()
            model = RandomForestClassifier(n_estimators=10)
            model.fit([[0, 1], [1, 0]], [0, 1])
            result = registry.register_model(
                model=model,
                model_name="churn_prediction_model",
                run_id="test_run_123"
            )
            assert result.success is True
            assert result.version is not None

    def test_logs_all_required_artifacts(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            registry = ModelRegistry()
            model = RandomForestClassifier(n_estimators=10)
            model.fit([[0, 1], [1, 0]], [0, 1])
            scaler = MagicMock()
            feature_manifest = {"features": ["f1", "f2"], "version": "1.0"}
            registry.register_model(
                model=model,
                model_name="test_model",
                run_id="test_run",
                scaler=scaler,
                feature_manifest=feature_manifest,
                threshold=0.5,
                metrics={"pr_auc": 0.75}
            )
            assert mock_mlflow.sklearn.log_model.called or mock_mlflow.log_artifact.called or mock_mlflow.log_dict.called

    def test_registration_includes_metadata(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_mlflow.register_model.return_value = MagicMock(version="1")
            mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="run_123"))
            registry = ModelRegistry()
            model = RandomForestClassifier(n_estimators=10)
            model.fit([[0, 1], [1, 0]], [0, 1])
            result = registry.register_model(
                model=model,
                model_name="test_model",
                run_id="run_123",
                tags={"model_type": "random_forest"},
                description="Test model for churn prediction"
            )
            assert result.metadata is not None


class TestModelStages:
    def test_model_stage_enum_values(self):
        assert ModelStage.NONE.value == "None"
        assert ModelStage.STAGING.value == "Staging"
        assert ModelStage.PRODUCTION.value == "Production"
        assert ModelStage.ARCHIVED.value == "Archived"

    def test_promotes_model_to_staging(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            registry.transition_stage("test_model", version="1", stage=ModelStage.STAGING)
            mock_client.transition_model_version_stage.assert_called()

    def test_promotes_model_to_production(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            registry.transition_stage("test_model", version="1", stage=ModelStage.PRODUCTION)
            call_args = mock_client.transition_model_version_stage.call_args
            assert ModelStage.PRODUCTION.value in str(call_args)

    def test_archives_deprecated_model(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            registry.transition_stage("test_model", version="1", stage=ModelStage.ARCHIVED)
            mock_client.transition_model_version_stage.assert_called()


class TestModelMetadata:
    def test_metadata_contains_required_fields(self):
        metadata = ModelMetadata(
            run_id="run_123",
            model_name="churn_model",
            version="1",
            stage=ModelStage.STAGING,
            training_date="2025-01-01",
            feature_table_version="v1.0"
        )
        assert metadata.run_id == "run_123"
        assert metadata.model_name == "churn_model"
        assert metadata.version == "1"
        assert metadata.stage == ModelStage.STAGING

    def test_metadata_includes_optional_tags(self):
        metadata = ModelMetadata(
            run_id="run_123",
            model_name="churn_model",
            version="1",
            stage=ModelStage.NONE,
            tags={"model_type": "xgboost", "dataset": "retail"}
        )
        assert metadata.tags["model_type"] == "xgboost"

    def test_metadata_includes_training_data_range(self):
        metadata = ModelMetadata(
            run_id="run_123",
            model_name="churn_model",
            version="1",
            stage=ModelStage.NONE,
            training_data_range=("2024-01-01", "2024-12-31")
        )
        assert metadata.training_data_range == ("2024-01-01", "2024-12-31")


class TestModelRetrieval:
    def test_loads_model_by_name_and_stage(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_model = MagicMock()
            mock_mlflow.pyfunc.load_model.return_value = mock_model
            registry = ModelRegistry()
            model = registry.load_model("churn_model", stage=ModelStage.PRODUCTION)
            assert model is not None

    def test_loads_model_by_version(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_model = MagicMock()
            mock_mlflow.pyfunc.load_model.return_value = mock_model
            registry = ModelRegistry()
            model = registry.load_model("churn_model", version="3")
            assert model is not None

    def test_retrieves_model_metadata(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_version = MagicMock()
            mock_version.run_id = "run_123"
            mock_version.version = "1"
            mock_version.current_stage = "Production"
            mock_version.tags = {}
            mock_client.get_model_version.return_value = mock_version
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            metadata = registry.get_metadata("churn_model", version="1")
            assert metadata.run_id == "run_123"

    def test_lists_all_model_versions(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_client.search_model_versions.return_value = [
                MagicMock(version="1", current_stage="Archived"),
                MagicMock(version="2", current_stage="Production"),
            ]
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            versions = registry.list_versions("churn_model")
            assert len(versions) == 2


class TestModelValidation:
    def test_validates_model_before_promotion(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            validation_result = registry.validate_for_promotion(
                model_name="churn_model",
                version="1",
                required_metrics={"pr_auc": 0.5}
            )
            assert validation_result.is_valid is not None

    def test_validation_checks_artifacts_exist(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_client.list_artifacts.return_value = [
                MagicMock(path="model"),
                MagicMock(path="scaler.pkl")
            ]
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            validation_result = registry.validate_for_promotion(
                model_name="churn_model",
                version="1",
                required_artifacts=["model", "scaler.pkl"]
            )
            assert validation_result.artifacts_present is True
