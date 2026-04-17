"""Harvest pipeline: walk cr.registry, infer stages via phase_map +
fallbacks, validate, return a splice plan.

Stage inference priority (plan § 5.2):
1. `rf.expected_stage` set on the decorator → use it.
2. `rf.cell_id` resolves in the phase_map → use the map's entry.
3. Notebook-number fallback via `NOTEBOOK_FALLBACK_STAGE`.
4. Else: unmappable, with reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .registry import RegisteredFunction, Registry
from .validation import ValidationError, validate_registered_function

NOTEBOOK_FALLBACK_STAGE: Dict[str, str] = {
    "00": "landing_post",
    "01": "bronze_post",
    "02": "bronze_post",
    "03": "bronze_merge",
    "04": "silver_post",
    "05": "silver_post",
    "06": "silver_post",
    "07": "silver_post",
    "08": "training",
}

_NB_NUMBER_RE = re.compile(r"(?P<n>\d{2})[a-z]*_")


@dataclass
class PhaseMap:
    version: int = 1
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PhaseMap":
        return cls()


@dataclass
class HarvestResult:
    functions_by_target: Dict[Tuple[str, str], List[RegisteredFunction]] = field(default_factory=dict)
    cross_dataset_steps: List[RegisteredFunction] = field(default_factory=list)
    unmappable: List[Tuple[RegisteredFunction, str]] = field(default_factory=list)
    validation_errors: List[ValidationError] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "HarvestResult":
        return cls()

    def is_empty(self) -> bool:
        return (
            not self.functions_by_target
            and not self.cross_dataset_steps
            and not self.unmappable
            and not self.validation_errors
        )

    def all_functions(self) -> List[RegisteredFunction]:
        acc: List[RegisteredFunction] = []
        for fns in self.functions_by_target.values():
            acc.extend(fns)
        acc.extend(self.cross_dataset_steps)
        return acc


class Harvester:
    def __init__(self, phase_map: Optional[PhaseMap] = None, registry_: Optional[Registry] = None) -> None:
        from .registry import registry as default_registry
        self._phase_map = phase_map if phase_map is not None else PhaseMap.empty()
        self._registry = registry_ if registry_ is not None else default_registry

    def harvest(self) -> HarvestResult:
        result = HarvestResult.empty()
        for rf in self._registry.get_registered():
            err = validate_registered_function(rf)
            if err is not None:
                result.validation_errors.append(err)
                continue
            stage, reason = self._infer_stage(rf)
            if stage is None:
                result.unmappable.append((rf, reason))
                continue
            rf.inferred_stage = stage
            if rf.scope == "dataset":
                key = (stage, rf.dataset)
                result.functions_by_target.setdefault(key, []).append(rf)
            else:
                result.cross_dataset_steps.append(rf)
        return result

    def _infer_stage(self, rf: RegisteredFunction) -> Tuple[Optional[str], str]:
        if rf.expected_stage:
            return rf.expected_stage, "expected_stage"
        if rf.cell_id:
            entry = self._phase_map.sections.get(rf.cell_id)
            if entry and entry.get("stage"):
                return entry["stage"], "phase_map"
        if rf.notebook_path:
            nb_num = _notebook_number(rf.notebook_path)
            if nb_num and nb_num in NOTEBOOK_FALLBACK_STAGE:
                return NOTEBOOK_FALLBACK_STAGE[nb_num], "notebook_fallback"
        return None, _unmappable_reason(rf)


def _notebook_number(path: Path) -> Optional[str]:
    stem = path.stem if isinstance(path, Path) else Path(path).stem
    m = _NB_NUMBER_RE.match(stem + "_")
    return m.group("n") if m else None


def _unmappable_reason(rf: RegisteredFunction) -> str:
    if not rf.notebook_path and not rf.cell_id:
        return "no notebook path or cell id captured; add expected_stage= on the decorator"
    if rf.notebook_path:
        return (
            f"notebook {rf.notebook_path} has no phase_map entry for cell_id="
            f"{rf.cell_id!r} and no notebook-number fallback matched"
        )
    return f"cell_id={rf.cell_id!r} not found in phase_map"
