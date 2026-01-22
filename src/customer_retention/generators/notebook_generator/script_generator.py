from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .base import NotebookStage

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer import ExplorationFindings

    from .stages.base_stage import StageGenerator
from .config import NotebookConfig, Platform
from .stages import (
    BatchInferenceStage,
    CleaningStage,
    DeploymentStage,
    FeatureEngineeringStage,
    FeatureSelectionStage,
    IngestionStage,
    ModelTrainingStage,
    MonitoringStage,
    ProfilingStage,
    TransformationStage,
)


class ScriptGenerator(ABC):
    def __init__(self, config: NotebookConfig, findings: Optional["ExplorationFindings"]):
        self.config = config
        self.findings = findings
        self.stage_generators = self._create_stage_generators()

    @abstractmethod
    def _create_stage_generators(self) -> Dict[NotebookStage, "StageGenerator"]:
        pass

    @property
    @abstractmethod
    def platform(self) -> Platform:
        pass

    def generate_stage_code(self, stage: NotebookStage) -> str:
        generator = self.stage_generators[stage]
        cells = generator.generate(self.platform)
        return self._cells_to_script(cells, generator.title, generator.description)

    def _cells_to_script(self, cells: list, title: str, description: str) -> str:
        lines = [f'"""{title}', "", description, '"""', ""]
        for cell in cells:
            if cell.cell_type == "code":
                lines.append(cell.source)
                lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    pass")
        return "\n".join(lines)

    def generate_all(self) -> Dict[NotebookStage, str]:
        return {stage: self.generate_stage_code(stage) for stage in self.stage_generators.keys()}

    def save_all(self, output_dir: str) -> List[str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for stage, code in self.generate_all().items():
            file_path = output_path / f"{stage.value}.py"
            file_path.write_text(code, encoding="utf-8")
            saved_paths.append(str(file_path))
        return saved_paths


class LocalScriptGenerator(ScriptGenerator):
    @property
    def platform(self) -> Platform:
        return Platform.LOCAL

    def _create_stage_generators(self) -> Dict[NotebookStage, "StageGenerator"]:
        self.config.platform = Platform.LOCAL
        return {
            NotebookStage.INGESTION: IngestionStage(self.config, self.findings),
            NotebookStage.PROFILING: ProfilingStage(self.config, self.findings),
            NotebookStage.CLEANING: CleaningStage(self.config, self.findings),
            NotebookStage.TRANSFORMATION: TransformationStage(self.config, self.findings),
            NotebookStage.FEATURE_ENGINEERING: FeatureEngineeringStage(self.config, self.findings),
            NotebookStage.FEATURE_SELECTION: FeatureSelectionStage(self.config, self.findings),
            NotebookStage.MODEL_TRAINING: ModelTrainingStage(self.config, self.findings),
            NotebookStage.DEPLOYMENT: DeploymentStage(self.config, self.findings),
            NotebookStage.MONITORING: MonitoringStage(self.config, self.findings),
            NotebookStage.BATCH_INFERENCE: BatchInferenceStage(self.config, self.findings),
        }


class DatabricksScriptGenerator(ScriptGenerator):
    @property
    def platform(self) -> Platform:
        return Platform.DATABRICKS

    def _create_stage_generators(self) -> Dict[NotebookStage, "StageGenerator"]:
        self.config.platform = Platform.DATABRICKS
        return {
            NotebookStage.INGESTION: IngestionStage(self.config, self.findings),
            NotebookStage.PROFILING: ProfilingStage(self.config, self.findings),
            NotebookStage.CLEANING: CleaningStage(self.config, self.findings),
            NotebookStage.TRANSFORMATION: TransformationStage(self.config, self.findings),
            NotebookStage.FEATURE_ENGINEERING: FeatureEngineeringStage(self.config, self.findings),
            NotebookStage.FEATURE_SELECTION: FeatureSelectionStage(self.config, self.findings),
            NotebookStage.MODEL_TRAINING: ModelTrainingStage(self.config, self.findings),
            NotebookStage.DEPLOYMENT: DeploymentStage(self.config, self.findings),
            NotebookStage.MONITORING: MonitoringStage(self.config, self.findings),
            NotebookStage.BATCH_INFERENCE: BatchInferenceStage(self.config, self.findings),
        }
