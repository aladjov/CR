from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import time
from .base import Component, ComponentResult, ComponentStatus
from .registry import ComponentRegistry

if TYPE_CHECKING:
    from customer_retention.generators.orchestration.context import PipelineContext


@dataclass
class OrchestratorResult:
    success: bool
    components_run: List[str]
    results: Dict[str, ComponentResult]
    total_duration_seconds: float

    def get_summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"{status}: {len(self.components_run)} components in {self.total_duration_seconds:.1f}s"


class Orchestrator:
    def __init__(self, registry: ComponentRegistry, context: "PipelineContext"):
        self.registry = registry
        self.context = context

    def run_training(self) -> OrchestratorResult:
        return self.run_chapters([1, 2, 3, 4, 5, 6, 7])

    def run_phase(self, phase: str) -> OrchestratorResult:
        start_time = time.time()
        registrations = self.registry.get_phase_components(phase)
        components_run = []
        results = {}
        success = True
        for reg in registrations:
            name = self._get_name_for_registration(reg)
            result = self._run_component(reg.component_class)
            results[name] = result
            components_run.append(name)
            if not result.success:
                success = False
                break
        return OrchestratorResult(
            success=success,
            components_run=components_run,
            results=results,
            total_duration_seconds=time.time() - start_time
        )

    def run_chapters(self, chapters: List[int]) -> OrchestratorResult:
        start_time = time.time()
        registrations = self.registry.get_chapters_components(chapters)
        components_run = []
        results = {}
        success = True
        for reg in registrations:
            name = self._get_name_for_registration(reg)
            result = self._run_component(reg.component_class)
            results[name] = result
            components_run.append(name)
            if not result.success:
                success = False
                break
        return OrchestratorResult(
            success=success,
            components_run=components_run,
            results=results,
            total_duration_seconds=time.time() - start_time
        )

    def run_single(self, component_name: str) -> ComponentResult:
        reg = self.registry.get_component(component_name)
        return self._run_component(reg.component_class)

    def _run_component(self, component_class: type) -> ComponentResult:
        component: Component = component_class()
        errors = component.validate_inputs(self.context)
        if errors:
            return ComponentResult(
                success=False, status=ComponentStatus.FAILED,
                errors=errors
            )
        if component.should_skip(self.context):
            return ComponentResult(success=True, status=ComponentStatus.SKIPPED)
        return component.run(self.context)

    def _get_name_for_registration(self, reg) -> str:
        for name, r in self.registry._components.items():
            if r == reg:
                return name
        return reg.component_class.__name__.lower()
