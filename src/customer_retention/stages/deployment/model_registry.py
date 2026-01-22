import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn


class ModelStage(Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ModelMetadata:
    run_id: str
    model_name: str
    version: str
    stage: ModelStage
    training_date: Optional[str] = None
    feature_table_version: Optional[str] = None
    training_data_range: Optional[Tuple[str, str]] = None
    tags: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class RegistrationResult:
    success: bool
    version: Optional[str] = None
    model_uri: Optional[str] = None
    metadata: Optional[ModelMetadata] = None
    error: Optional[str] = None


@dataclass
class ValidationResult:
    is_valid: bool
    artifacts_present: bool = True
    metrics_meet_threshold: bool = True
    errors: List[str] = field(default_factory=list)


class ModelRegistry:
    def __init__(self, tracking_uri: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = mlflow.tracking.MlflowClient()
        return self._client

    def register_model(self, model: Any, model_name: str, run_id: str,
                       scaler: Any = None, feature_manifest: Optional[Dict] = None,
                       threshold: Optional[float] = None, metrics: Optional[Dict] = None,
                       tags: Optional[Dict[str, str]] = None, description: Optional[str] = None,
                       config: Optional[Dict] = None) -> RegistrationResult:
        try:
            with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
                if metrics:
                    mlflow.log_metrics(metrics)
                if tags:
                    mlflow.set_tags(tags)
                mlflow.sklearn.log_model(model, "model")
                if scaler is not None:
                    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                        import pickle
                        pickle.dump(scaler, f)
                        scaler_path = f.name
                    mlflow.log_artifact(scaler_path, "scaler")
                    os.unlink(scaler_path)
                if feature_manifest:
                    mlflow.log_dict(feature_manifest, "feature_manifest.json")
                if threshold is not None:
                    mlflow.log_param("optimal_threshold", threshold)
                if config:
                    mlflow.log_dict(config, "config.json")
                active_run = mlflow.active_run()
                current_run_id = active_run.info.run_id if active_run else run_id
                model_uri = f"runs:/{current_run_id}/model"
                result = mlflow.register_model(model_uri, model_name)
                metadata = ModelMetadata(
                    run_id=current_run_id,
                    model_name=model_name,
                    version=str(result.version),
                    stage=ModelStage.NONE,
                    tags=tags or {},
                    description=description
                )
                return RegistrationResult(
                    success=True,
                    version=str(result.version),
                    model_uri=model_uri,
                    metadata=metadata
                )
        except Exception as e:
            return RegistrationResult(success=False, error=str(e))

    def transition_stage(self, model_name: str, version: str, stage: ModelStage,
                         archive_existing: bool = True) -> bool:
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage.value,
            archive_existing_versions=archive_existing
        )
        return True

    def load_model(self, model_name: str, stage: Optional[ModelStage] = None,
                   version: Optional[str] = None) -> Any:
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage.value}"
        else:
            model_uri = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(model_uri)

    def get_metadata(self, model_name: str, version: str) -> ModelMetadata:
        model_version = self.client.get_model_version(model_name, version)
        return ModelMetadata(
            run_id=model_version.run_id,
            model_name=model_name,
            version=model_version.version,
            stage=ModelStage(model_version.current_stage),
            tags=dict(model_version.tags) if model_version.tags else {}
        )

    def list_versions(self, model_name: str) -> List[ModelMetadata]:
        versions = self.client.search_model_versions(f"name='{model_name}'")
        return [
            ModelMetadata(
                run_id=v.run_id,
                model_name=model_name,
                version=v.version,
                stage=ModelStage(v.current_stage),
                tags=dict(v.tags) if v.tags else {}
            )
            for v in versions
        ]

    def validate_for_promotion(self, model_name: str, version: str,
                               required_metrics: Optional[Dict[str, float]] = None,
                               required_artifacts: Optional[List[str]] = None) -> ValidationResult:
        errors = []
        artifacts_present = True
        metrics_meet_threshold = True
        try:
            model_version = self.client.get_model_version(model_name, version)
            run_id = model_version.run_id
            if required_artifacts:
                artifacts = self.client.list_artifacts(run_id)
                artifact_paths = [a.path for a in artifacts]
                for req_artifact in required_artifacts:
                    if req_artifact not in artifact_paths:
                        artifacts_present = False
                        errors.append(f"Missing artifact: {req_artifact}")
            if required_metrics:
                run = self.client.get_run(run_id)
                run_metrics = run.data.metrics
                for metric_name, threshold in required_metrics.items():
                    if metric_name not in run_metrics:
                        metrics_meet_threshold = False
                        errors.append(f"Missing metric: {metric_name}")
                    elif run_metrics[metric_name] < threshold:
                        metrics_meet_threshold = False
                        errors.append(f"Metric {metric_name} below threshold: {run_metrics[metric_name]} < {threshold}")
        except Exception as e:
            errors.append(str(e))
        return ValidationResult(
            is_valid=artifacts_present and metrics_meet_threshold and len(errors) == 0,
            artifacts_present=artifacts_present,
            metrics_meet_threshold=metrics_meet_threshold,
            errors=errors
        )
