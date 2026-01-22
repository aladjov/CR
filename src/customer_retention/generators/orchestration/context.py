import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from customer_retention.analysis.auto_explorer import ExplorationFindings
from customer_retention.core.config.column_config import ColumnConfig, ColumnType


@dataclass
class PipelineContext:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    project_name: str = "customer_retention"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_data_path: Optional[str] = None
    bronze_path: Optional[str] = None
    silver_path: Optional[str] = None
    gold_path: Optional[str] = None
    exploration_findings: Optional[ExplorationFindings] = None
    profiling_results: Optional[Dict] = None
    cleaning_report: Optional[Dict] = None
    transformation_pipeline: Optional[Any] = None
    feature_manifest: Optional[Any] = None
    selected_features: Optional[List[str]] = None
    model_results: Optional[Dict] = None
    validation_results: Optional[Dict] = None
    current_df: Optional[Any] = None
    current_stage: str = "raw"
    artifacts: Dict[str, str] = field(default_factory=dict)
    _context_dir: str = "./runs"

    @property
    def column_types(self) -> Dict[str, ColumnType]:
        return self.exploration_findings.column_types if self.exploration_findings else {}

    @property
    def column_configs(self) -> Dict[str, ColumnConfig]:
        return self.exploration_findings.column_configs if self.exploration_findings else {}

    @property
    def target_column(self) -> Optional[str]:
        return self.exploration_findings.target_column if self.exploration_findings else None

    def save(self, path: str = None):
        Path(self._context_dir).mkdir(parents=True, exist_ok=True)
        if path is None:
            path = f"{self._context_dir}/{self.run_id}_context.json"
        data = {
            "run_id": self.run_id,
            "project_name": self.project_name,
            "started_at": self.started_at,
            "raw_data_path": self.raw_data_path,
            "bronze_path": self.bronze_path,
            "silver_path": self.silver_path,
            "gold_path": self.gold_path,
            "current_stage": self.current_stage,
            "artifacts": self.artifacts,
            "exploration_findings_path": None
        }
        if self.exploration_findings:
            findings_path = f"{self._context_dir}/{self.run_id}_findings.yaml"
            self.exploration_findings.save(findings_path)
            data["exploration_findings_path"] = findings_path
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Context saved: {path}")

    @classmethod
    def load(cls, path: str) -> "PipelineContext":
        with open(path, "r") as f:
            data = json.load(f)
        exploration_findings = None
        if data.get("exploration_findings_path"):
            exploration_findings = ExplorationFindings.load(data["exploration_findings_path"])
        ctx = cls(
            run_id=data["run_id"],
            project_name=data["project_name"],
            started_at=data["started_at"],
            raw_data_path=data.get("raw_data_path"),
            bronze_path=data.get("bronze_path"),
            silver_path=data.get("silver_path"),
            gold_path=data.get("gold_path"),
            current_stage=data.get("current_stage", "raw"),
            artifacts=data.get("artifacts", {}),
            exploration_findings=exploration_findings
        )
        ctx._context_dir = str(Path(path).parent)
        return ctx


class ContextManager:
    def __init__(self, context: PipelineContext, auto_save: bool = True):
        self.context = context
        self.auto_save = auto_save

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        if self.auto_save:
            self.context.save()

    def add_artifact(self, name: str, path: str):
        self.context.artifacts[name] = path
        if self.auto_save:
            self.context.save()


def setup_notebook_context(
    project_name: str = None,
    exploration_findings: Any = None,
    resume_run: str = None,
    output_dir: str = "./runs"
) -> Tuple[PipelineContext, ContextManager]:
    if resume_run:
        context_path = f"{output_dir}/{resume_run}_context.json"
        context = PipelineContext.load(context_path)
        print(f"Resumed run: {context.run_id}")
    else:
        context = PipelineContext(project_name=project_name or "customer_retention")
        context._context_dir = output_dir
        if exploration_findings:
            if isinstance(exploration_findings, str):
                context.exploration_findings = ExplorationFindings.load(exploration_findings)
            else:
                context.exploration_findings = exploration_findings
            context.raw_data_path = context.exploration_findings.source_path
            print(f"New run: {context.run_id}")
            print(f"Loaded findings: {len(context.column_types)} columns")
            print(f"Target: {context.target_column}")
    return context, ContextManager(context)
