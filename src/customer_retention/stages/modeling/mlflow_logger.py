"""MLflow integration for experiment tracking."""

from contextlib import contextmanager
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
        self._last_run_id: Optional[str] = None

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
        self._last_run_id = self._run.info.run_id

    def end_run(self):
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
        self._run = None

    @property
    def last_run_id(self) -> Optional[str]:
        """Return run_id even after end_run() — set during start_run()."""
        return self._last_run_id

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
        if not MLFLOW_AVAILABLE:
            return
        if hasattr(model, "as_pipeline_model") and model._fitted_model is not None:
            self._log_spark_model(model, artifact_path, registered_model_name)
        else:
            mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name)

    def _log_spark_model(self, wrapper, artifact_path: str, registered_model_name: Optional[str] = None):
        import mlflow.spark as mlflow_spark
        pipeline_model = wrapper.as_pipeline_model()
        mlflow_spark.log_model(pipeline_model, artifact_path, registered_model_name=registered_model_name)
        mlflow.set_tag(f"{artifact_path}.model_flavor", "spark")
        mlflow.log_dict({
            "spark_model_class": wrapper.spark_model_class,
            "spark_model_params": wrapper.spark_model_params,
            "feature_names": wrapper.feature_names,
            "class_weight": wrapper.class_weight,
        }, f"{artifact_path}_wrapper_meta.json")

    def log_figure(self, figure, artifact_file: str):
        if MLFLOW_AVAILABLE:
            mlflow.log_figure(figure, artifact_file)

    @property
    def run_id(self) -> Optional[str]:
        if self._run is not None:
            return self._run.info.run_id
        return None

    @contextmanager
    def nested_run(self, run_name: str):
        if not MLFLOW_AVAILABLE:
            yield
            return
        with mlflow.start_run(run_name=run_name, nested=True):
            yield

    def get_tracking_uri(self) -> Optional[str]:
        if MLFLOW_AVAILABLE:
            return mlflow.get_tracking_uri()
        return self.tracking_uri

    def disable_autolog(self):
        if MLFLOW_AVAILABLE:
            mlflow.autolog(disable=True)

    def end_stale_runs(self):
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
