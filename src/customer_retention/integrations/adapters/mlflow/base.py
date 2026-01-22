from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class MLflowAdapter(ABC):
    @abstractmethod
    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def end_run(self) -> None:
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def log_model(self, model: Any, artifact_path: str, registered_name: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def load_model(self, model_uri: str) -> Any:
        pass

    @abstractmethod
    def transition_stage(self, model_name: str, version: str, stage: str) -> None:
        pass
