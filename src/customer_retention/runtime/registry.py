"""In-process registry of @cr.register-decorated functions.

Decoration appends to the singleton `registry`; harvesters read from it
at codegen time. Re-executing the same cell updates its record in place
— the dedup key is `(name, notebook_path, cell_id)` per plan § 3.1.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

Scope = Literal["dataset", "datasets", "wildcard"]


@dataclass
class RegisteredFunction:
    name: str
    source: str
    scope: Scope
    dataset: Optional[str] = None
    datasets: Optional[List[str]] = None
    primary: Optional[str] = None
    replay_at_scoring: bool = False
    expected_stage: Optional[str] = None
    notebook_path: Optional[Path] = None
    cell_id: Optional[str] = None
    inferred_stage: Optional[str] = None


class Registry:
    def __init__(self) -> None:
        self._records: List[RegisteredFunction] = []

    def register(self, rf: RegisteredFunction) -> None:
        key = (rf.name, rf.notebook_path, rf.cell_id)
        for i, existing in enumerate(self._records):
            if (existing.name, existing.notebook_path, existing.cell_id) == key:
                self._records[i] = rf
                return
        self._records.append(rf)

    def get_registered(self) -> List[RegisteredFunction]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    def validate(self):
        from .validation import validate_registered_function
        errors = []
        for rf in self._records:
            err = validate_registered_function(rf)
            if err is not None:
                errors.append(err)
        return errors


registry = Registry()
