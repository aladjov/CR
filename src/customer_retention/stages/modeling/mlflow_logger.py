"""MLflow integration for experiment tracking."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class ExperimentConfig:
    experiment_name: str
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None


class MLflowLogger:
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self._run = None

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False

    def start_run(self, run_name: Optional[str] = None):
        if not MLFLOW_AVAILABLE:
            return

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id

        self._run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name or self.run_name,
        )

    def end_run(self):
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
        self._run = None

    def log_params(self, params: Dict[str, Any]):
        if MLFLOW_AVAILABLE:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(local_path, artifact_path)

    def set_tags(self, tags: Dict[str, str]):
        if MLFLOW_AVAILABLE:
            mlflow.set_tags(tags)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        if MLFLOW_AVAILABLE:
            mlflow.log_dict(dictionary, artifact_file)

    def log_model(self, model, artifact_path: str, registered_model_name: Optional[str] = None):
        if MLFLOW_AVAILABLE:
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
            )

    def log_figure(self, figure, artifact_file: str):
        if MLFLOW_AVAILABLE:
            mlflow.log_figure(figure, artifact_file)
