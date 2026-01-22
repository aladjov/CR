from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import nbformat

from .config import NotebookConfig


class NotebookStage(str, Enum):
    INGESTION = "01_ingestion"
    PROFILING = "02_profiling"
    CLEANING = "03_cleaning"
    TRANSFORMATION = "04_transformation"
    FEATURE_ENGINEERING = "05_feature_engineering"
    FEATURE_SELECTION = "06_feature_selection"
    MODEL_TRAINING = "07_model_training"
    DEPLOYMENT = "08_deployment"
    MONITORING = "09_monitoring"
    BATCH_INFERENCE = "10_batch_inference"
    FEATURE_STORE = "11_feature_store"


class NotebookGenerator(ABC):
    def __init__(self, config: NotebookConfig, findings: Optional["ExplorationFindings"]):
        self.config = config
        self.findings = findings

    @abstractmethod
    def generate_stage(self, stage: NotebookStage) -> nbformat.NotebookNode:
        pass

    @property
    def available_stages(self) -> List[NotebookStage]:
        if hasattr(self, "stage_generators"):
            return list(self.stage_generators.keys())
        return list(NotebookStage)

    def generate_all(self) -> Dict[NotebookStage, nbformat.NotebookNode]:
        return {stage: self.generate_stage(stage) for stage in self.available_stages}

    def save_all(self, output_dir: str) -> List[str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for stage, notebook in self.generate_all().items():
            file_path = output_path / f"{stage.value}.ipynb"
            with open(file_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
            saved_paths.append(str(file_path))
        return saved_paths
