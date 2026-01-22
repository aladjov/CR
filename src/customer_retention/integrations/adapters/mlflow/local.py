from typing import Any, Dict, Optional

from .base import MLflowAdapter

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class LocalMLflow(MLflowAdapter):
    def __init__(self, tracking_uri: str = "./mlruns"):
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow package required: pip install mlflow")
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self._client = MlflowClient(tracking_uri=tracking_uri)
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
        info = mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_name)
        return info.model_uri

    def load_model(self, model_uri: str) -> Any:
        return mlflow.sklearn.load_model(model_uri)

    def transition_stage(self, model_name: str, version: str, stage: str) -> None:
        self._client.transition_model_version_stage(name=model_name, version=version, stage=stage)
