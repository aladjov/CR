import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class IterationStatus(Enum):
    EXPLORING = "exploring"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class IterationTrigger(Enum):
    INITIAL = "initial"
    MANUAL = "manual"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DROP = "performance_drop"
    SCHEDULED = "scheduled"


@dataclass
class IterationContext:
    iteration_id: str
    iteration_number: int
    parent_iteration_id: Optional[str]
    started_at: datetime
    status: IterationStatus
    trigger: IterationTrigger
    findings_path: str
    recommendations_path: str
    model_artifact_path: Optional[str] = None
    model_metrics: Optional[Dict[str, float]] = None
    feature_count: int = 0
    applied_recommendations: List[str] = field(default_factory=list)
    skipped_recommendations: List[str] = field(default_factory=list)
    completed_at: Optional[datetime] = None

    @classmethod
    def create_new(cls, findings_dir: str, trigger: IterationTrigger,
                   iteration_number: int = 1) -> "IterationContext":
        iteration_id = str(uuid.uuid4())
        findings_path = f"{findings_dir}/iterations/findings_{iteration_id}.yaml"
        recommendations_path = f"{findings_dir}/iterations/recommendations_{iteration_id}.yaml"
        return cls(
            iteration_id=iteration_id,
            iteration_number=iteration_number,
            parent_iteration_id=None,
            started_at=datetime.now(),
            status=IterationStatus.EXPLORING,
            trigger=trigger,
            findings_path=findings_path,
            recommendations_path=recommendations_path,
            applied_recommendations=[],
            skipped_recommendations=[]
        )

    def create_child(self, trigger: IterationTrigger) -> "IterationContext":
        findings_dir = str(Path(self.findings_path).parent.parent)
        child = IterationContext.create_new(
            findings_dir=findings_dir,
            trigger=trigger,
            iteration_number=self.iteration_number + 1
        )
        child.parent_iteration_id = self.iteration_id
        return child

    def update_status(self, status: IterationStatus) -> None:
        self.status = status
        if status == IterationStatus.COMPLETED:
            self.completed_at = datetime.now()

    def set_model_metrics(self, metrics: Dict[str, float],
                          artifact_path: Optional[str] = None) -> None:
        self.model_metrics = metrics
        if artifact_path:
            self.model_artifact_path = artifact_path

    def add_applied_recommendation(self, recommendation_id: str) -> None:
        if recommendation_id not in self.applied_recommendations:
            self.applied_recommendations.append(recommendation_id)

    def add_skipped_recommendation(self, recommendation_id: str) -> None:
        if recommendation_id not in self.skipped_recommendations:
            self.skipped_recommendations.append(recommendation_id)

    def get_iteration_filename(self) -> str:
        short_id = self.iteration_id[:8]
        return f"iteration_{self.iteration_number:03d}_{short_id}.yaml"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "iteration_id": self.iteration_id,
            "iteration_number": self.iteration_number,
            "parent_iteration_id": self.parent_iteration_id,
            "started_at": self.started_at.isoformat(),
            "status": self.status.value,
            "trigger": self.trigger.value,
            "findings_path": self.findings_path,
            "recommendations_path": self.recommendations_path,
            "model_artifact_path": self.model_artifact_path,
            "model_metrics": self.model_metrics,
            "feature_count": self.feature_count,
            "applied_recommendations": self.applied_recommendations,
            "skipped_recommendations": self.skipped_recommendations,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationContext":
        return cls(
            iteration_id=data["iteration_id"],
            iteration_number=data["iteration_number"],
            parent_iteration_id=data.get("parent_iteration_id"),
            started_at=datetime.fromisoformat(data["started_at"]),
            status=IterationStatus(data["status"]),
            trigger=IterationTrigger(data["trigger"]),
            findings_path=data["findings_path"],
            recommendations_path=data["recommendations_path"],
            model_artifact_path=data.get("model_artifact_path"),
            model_metrics=data.get("model_metrics"),
            feature_count=data.get("feature_count", 0),
            applied_recommendations=data.get("applied_recommendations", []),
            skipped_recommendations=data.get("skipped_recommendations", []),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "IterationContext":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def compare(self, other: "IterationContext") -> Dict[str, Any]:
        comparison = {
            "iteration_diff": self.iteration_number - other.iteration_number,
            "metric_changes": {},
            "feature_count_change": self.feature_count - other.feature_count,
            "new_recommendations": [
                r for r in self.applied_recommendations
                if r not in other.applied_recommendations
            ]
        }
        if self.model_metrics and other.model_metrics:
            for metric, value in self.model_metrics.items():
                if metric in other.model_metrics:
                    comparison["metric_changes"][metric] = value - other.model_metrics[metric]
        return comparison


class IterationContextManager:
    def __init__(self, iterations_dir: str):
        self.iterations_dir = Path(iterations_dir)
        self._current_path = self.iterations_dir / "current_iteration.yaml"

    def list_iterations(self) -> List[IterationContext]:
        iterations = []
        for path in self.iterations_dir.glob("iteration_*.yaml"):
            if path.name != "current_iteration.yaml":
                iterations.append(IterationContext.load(str(path)))
        iterations.sort(key=lambda x: x.iteration_number)
        return iterations

    def get_current(self) -> Optional[IterationContext]:
        if not self._current_path.exists():
            return None
        with open(self._current_path, "r") as f:
            data = yaml.safe_load(f)
        current_id = data.get("current_iteration_id")
        if current_id:
            return self.get_by_id(current_id)
        return None

    def set_current(self, iteration_id: str) -> None:
        with open(self._current_path, "w") as f:
            yaml.dump({"current_iteration_id": iteration_id}, f)

    def get_by_id(self, iteration_id: str) -> Optional[IterationContext]:
        for path in self.iterations_dir.glob("iteration_*.yaml"):
            if path.name == "current_iteration.yaml":
                continue
            ctx = IterationContext.load(str(path))
            if ctx.iteration_id == iteration_id:
                return ctx
        return None

    def get_iteration_history(self, iteration_id: str) -> List[IterationContext]:
        history = []
        current = self.get_by_id(iteration_id)
        while current is not None:
            history.insert(0, current)
            if current.parent_iteration_id:
                current = self.get_by_id(current.parent_iteration_id)
            else:
                current = None
        return history

    def save_iteration(self, ctx: IterationContext) -> str:
        path = self.iterations_dir / ctx.get_iteration_filename()
        ctx.save(str(path))
        return str(path)
