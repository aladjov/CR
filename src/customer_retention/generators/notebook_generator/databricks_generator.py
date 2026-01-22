from typing import Dict, Optional
import nbformat

from .base import NotebookGenerator, NotebookStage
from .config import NotebookConfig, Platform
from .cell_builder import CellBuilder
from .stages import (
    IngestionStage, ProfilingStage, CleaningStage, TransformationStage,
    FeatureEngineeringStage, FeatureSelectionStage, ModelTrainingStage,
    DeploymentStage, MonitoringStage, BatchInferenceStage,
)


class DatabricksNotebookGenerator(NotebookGenerator):
    def __init__(self, config: NotebookConfig, findings: Optional["ExplorationFindings"]):
        config.platform = Platform.DATABRICKS
        super().__init__(config, findings)
        self.stage_generators = self._build_stage_generators(config, findings)

    def _build_stage_generators(self, config: NotebookConfig, findings) -> dict:
        return {
            NotebookStage.INGESTION: IngestionStage(config, findings),
            NotebookStage.PROFILING: ProfilingStage(config, findings),
            NotebookStage.CLEANING: CleaningStage(config, findings),
            NotebookStage.TRANSFORMATION: TransformationStage(config, findings),
            NotebookStage.FEATURE_ENGINEERING: FeatureEngineeringStage(config, findings),
            NotebookStage.FEATURE_SELECTION: FeatureSelectionStage(config, findings),
            NotebookStage.MODEL_TRAINING: ModelTrainingStage(config, findings),
            NotebookStage.DEPLOYMENT: DeploymentStage(config, findings),
            NotebookStage.MONITORING: MonitoringStage(config, findings),
            NotebookStage.BATCH_INFERENCE: BatchInferenceStage(config, findings),
        }

    def generate_stage(self, stage: NotebookStage) -> nbformat.NotebookNode:
        generator = self.stage_generators[stage]
        cells = generator.generate(Platform.DATABRICKS)
        return CellBuilder.create_notebook(cells)
