from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
from .base import Component


@dataclass
class ComponentRegistration:
    component_class: Type[Component]
    phase: str
    dependencies: List[str] = field(default_factory=list)


class ComponentRegistry:
    PHASES = ["discovery", "data_preparation", "model_development", "production"]

    def __init__(self):
        self._components: Dict[str, ComponentRegistration] = {}

    def register(self, name: str, component_class: Type[Component], phase: str,
                 dependencies: Optional[List[str]] = None) -> None:
        self._components[name] = ComponentRegistration(
            component_class=component_class,
            phase=phase,
            dependencies=dependencies or []
        )

    def get_component(self, name: str) -> ComponentRegistration:
        if name not in self._components:
            raise KeyError(f"Component '{name}' not found")
        return self._components[name]

    def get_phase_components(self, phase: str) -> List[ComponentRegistration]:
        return [reg for reg in self._components.values() if reg.phase == phase]

    def get_chapters_components(self, chapters: List[int]) -> List[ComponentRegistration]:
        result = []
        for reg in self._components.values():
            instance = reg.component_class()
            if any(ch in instance.chapters for ch in chapters):
                result.append(reg)
        return result

    def list_components(self) -> List[str]:
        return list(self._components.keys())


def get_default_registry() -> ComponentRegistry:
    from .components import (Ingester, Profiler, Transformer, FeatureEngineer,
                             Trainer, Validator, Explainer, Deployer)
    registry = ComponentRegistry()
    registry.register("ingester", Ingester, "data_preparation")
    registry.register("profiler", Profiler, "data_preparation", ["ingester"])
    registry.register("transformer", Transformer, "data_preparation", ["profiler"])
    registry.register("feature_engineer", FeatureEngineer, "data_preparation", ["transformer"])
    registry.register("trainer", Trainer, "model_development", ["feature_engineer"])
    registry.register("validator", Validator, "model_development", ["trainer"])
    registry.register("explainer", Explainer, "model_development", ["trainer"])
    registry.register("deployer", Deployer, "production", ["validator"])
    return registry
