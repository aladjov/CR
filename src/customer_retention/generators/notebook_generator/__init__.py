from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .base import NotebookGenerator, NotebookStage
from .config import NotebookConfig, Platform, MLflowConfig, FeatureStoreConfig, OutputFormat
from .cell_builder import CellBuilder
from .local_generator import LocalNotebookGenerator
from .databricks_generator import DatabricksNotebookGenerator
from .runner import NotebookRunner, NotebookValidationResult, ValidationReport, validate_generated_notebooks, ScriptRunner
from .script_generator import ScriptGenerator, LocalScriptGenerator, DatabricksScriptGenerator
from .project_init import ProjectInitializer, initialize_project


@dataclass
class GenerationResult:
    platform: Platform
    notebook_paths: List[str]
    validation_report: Optional[ValidationReport] = None

    @property
    def all_valid(self) -> bool:
        return self.validation_report.all_passed if self.validation_report else True


def generate_orchestration_notebooks(
    findings_path: Optional[str] = None,
    output_dir: str = "./generated_pipelines",
    platforms: Optional[List[Platform]] = None,
    config: Optional[NotebookConfig] = None,
    validate: bool = False,
) -> Dict[Platform, List[str]]:
    if platforms is None:
        platforms = [Platform.LOCAL, Platform.DATABRICKS]
    if config is None:
        config = NotebookConfig()

    findings = None
    if findings_path:
        from customer_retention.analysis.auto_explorer import ExplorationFindings
        findings = ExplorationFindings.load(findings_path)

    results = {}
    for platform in platforms:
        generator = create_notebook_generator(platform, findings, config)
        platform_dir = str(Path(output_dir) / platform.value)
        saved_paths = generator.save_all(platform_dir)
        results[platform] = saved_paths

    return results


def generate_and_validate_notebooks(
    findings_path: Optional[str] = None,
    output_dir: str = "./generated_pipelines",
    platforms: Optional[List[Platform]] = None,
    config: Optional[NotebookConfig] = None,
) -> Dict[Platform, GenerationResult]:
    if platforms is None:
        platforms = [Platform.LOCAL, Platform.DATABRICKS]
    if config is None:
        config = NotebookConfig()

    findings = None
    if findings_path:
        from customer_retention.analysis.auto_explorer import ExplorationFindings
        findings = ExplorationFindings.load(findings_path)

    results = {}
    runner = NotebookRunner(dry_run=True)

    for platform in platforms:
        generator = create_notebook_generator(platform, findings, config)
        platform_dir = str(Path(output_dir) / platform.value)
        saved_paths = generator.save_all(platform_dir)
        validation_report = runner.validate_sequence(platform_dir, platform.value)
        results[platform] = GenerationResult(platform, saved_paths, validation_report)
        save_validation_report(platform_dir, validation_report)

    return results


def save_validation_report(output_dir: str, report: ValidationReport) -> str:
    report_path = Path(output_dir) / "VALIDATION_REPORT.md"
    report_path.write_text(report.to_markdown())
    return str(report_path)


def create_notebook_generator(
    platform: Platform,
    findings: Optional["ExplorationFindings"] = None,
    config: Optional[NotebookConfig] = None,
) -> NotebookGenerator:
    if config is None:
        config = NotebookConfig()

    if platform == Platform.LOCAL:
        return LocalNotebookGenerator(config, findings)
    elif platform == Platform.DATABRICKS:
        return DatabricksNotebookGenerator(config, findings)
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def create_script_generator(
    platform: Platform,
    findings: Optional["ExplorationFindings"] = None,
    config: Optional[NotebookConfig] = None,
) -> ScriptGenerator:
    if config is None:
        config = NotebookConfig()

    if platform == Platform.LOCAL:
        return LocalScriptGenerator(config, findings)
    elif platform == Platform.DATABRICKS:
        return DatabricksScriptGenerator(config, findings)
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def generate_orchestration_scripts(
    findings_path: Optional[str] = None,
    output_dir: str = "./generated_pipelines/scripts",
    platforms: Optional[List[Platform]] = None,
    config: Optional[NotebookConfig] = None,
) -> Dict[Platform, List[str]]:
    if platforms is None:
        platforms = [Platform.LOCAL, Platform.DATABRICKS]
    if config is None:
        config = NotebookConfig()

    findings = None
    if findings_path:
        from customer_retention.analysis.auto_explorer import ExplorationFindings
        findings = ExplorationFindings.load(findings_path)

    results = {}
    for platform in platforms:
        generator = create_script_generator(platform, findings, config)
        platform_dir = str(Path(output_dir) / platform.value)
        saved_paths = generator.save_all(platform_dir)
        results[platform] = saved_paths

    return results


__all__ = [
    "NotebookGenerator", "NotebookStage", "NotebookConfig", "Platform",
    "MLflowConfig", "FeatureStoreConfig", "CellBuilder", "OutputFormat",
    "LocalNotebookGenerator", "DatabricksNotebookGenerator",
    "NotebookRunner", "NotebookValidationResult", "ValidationReport", "ScriptRunner",
    "GenerationResult", "generate_orchestration_notebooks",
    "generate_and_validate_notebooks", "create_notebook_generator",
    "validate_generated_notebooks", "save_validation_report",
    "ScriptGenerator", "LocalScriptGenerator", "DatabricksScriptGenerator",
    "create_script_generator", "generate_orchestration_scripts",
    "ProjectInitializer", "initialize_project",
]
