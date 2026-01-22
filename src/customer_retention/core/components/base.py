from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from customer_retention.generators.orchestration.context import PipelineContext


class ComponentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ComponentResult:
    success: bool
    status: ComponentStatus
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    output_data: Optional[Any] = None

    def get_summary(self) -> str:
        return f"{self.status.value.upper()} in {self.duration_seconds:.1f}s"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
        }


class Component(ABC):
    def __init__(self, name: str, chapters: List[int]):
        self.name = name
        self.chapters = chapters
        self._start_time: Optional[float] = None

    @abstractmethod
    def validate_inputs(self, context: "PipelineContext") -> List[str]:
        pass

    @abstractmethod
    def run(self, context: "PipelineContext") -> ComponentResult:
        pass

    def should_skip(self, context: "PipelineContext") -> bool:
        return False

    def create_result(self, success: bool, artifacts: Optional[Dict[str, str]] = None,
                      metrics: Optional[Dict[str, float]] = None, errors: Optional[List[str]] = None,
                      warnings: Optional[List[str]] = None, output_data: Optional[Any] = None) -> ComponentResult:
        duration = time.time() - self._start_time if self._start_time else 0.0
        status = ComponentStatus.COMPLETED if success else ComponentStatus.FAILED
        return ComponentResult(
            success=success,
            status=status,
            artifacts=artifacts or {},
            metrics=metrics or {},
            errors=errors or [],
            warnings=warnings or [],
            duration_seconds=duration,
            output_data=output_data
        )

    def _start_timer(self) -> None:
        self._start_time = time.time()
