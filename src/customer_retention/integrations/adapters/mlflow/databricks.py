from typing import Any, Dict, Optional

from customer_retention.core.compat.detection import is_spark_available

from .base import MLflowAdapter

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
    MLFLOW_MAJOR_VERSION = int(mlflow.__version__.split(".")[0])
except ImportError:
    MLFLOW_AVAILABLE = False
    MLFLOW_MAJOR_VERSION = 0


class DatabricksMLflow(MLflowAdapter):
    def __init__(self, registry_uri: str = "databricks-uc"):
        if not is_spark_available():
            raise ImportError("PySpark required for DatabricksMLflow")
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow package required")
        if MLFLOW_MAJOR_VERSION < 3:
            mlflow.set_registry_uri(registry_uri)
        self.registry_uri = registry_uri
        self._client = MlflowClient()
        self._run_id = None

    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        self._run_id = run.info.run_id
        return self._run_id

    def end_run(self) -> None:
        mlflow.end_run()
        self._run_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str, registered_name: Optional[str] = None) -> str:
        if MLFLOW_MAJOR_VERSION >= 3:
            info = mlflow.sklearn.log_model(model, name=artifact_path, registered_model_name=registered_name)
        else:
            info = mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_name)
        return info.model_uri

    def load_model(self, model_uri: str) -> Any:
        return mlflow.sklearn.load_model(model_uri)

    def transition_stage(self, model_name: str, version: str, stage: str) -> None:
        self._client.set_model_version_tag(name=model_name, version=version, key="stage", value=stage)

    def set_alias(self, model_name: str, alias: str, version: str) -> None:
        self._client.set_registered_model_alias(name=model_name, alias=alias, version=version)

    def get_model_by_alias(self, model_name: str, alias: str) -> Any:
        return self._client.get_model_version_by_alias(name=model_name, alias=alias)
