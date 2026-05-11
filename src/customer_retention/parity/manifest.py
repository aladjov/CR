"""Manifest dataclasses: SourceLocation, ManifestEntry, Manifest.

A Manifest is an ordered tuple of ManifestEntries describing every apply-op
call site reachable from a given scope (exploration notebooks or rendered
production scripts). Equality and `diff_key()` ignore source location and
call order so two manifests can be compared structurally.

`fingerprint_kwargs` normalises a kwargs dict into a hashable, deterministic
shape suitable for set-difference comparison: sorted keys, lists → tuples,
None values dropped, DataFrame-like objects dropped (their identity differs
across runs but their structural role is captured by the call site itself).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Iterable, Mapping, Optional, Tuple

from .kinds import ApplyOpKind

_DATAFRAME_ATTRS = ("_internal", "to_spark", "rdd", "_jdf")


def _is_dataframe_like(value: Any) -> bool:
    return any(hasattr(value, attr) for attr in _DATAFRAME_ATTRS)


def fingerprint_kwargs(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        if value is None or _is_dataframe_like(value):
            continue
        cleaned[key] = _normalise(value)
    return cleaned


def _normalise(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalise(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_normalise(v) for v in value)
    if isinstance(value, dict):
        return {k: _normalise(value[k]) for k in sorted(value.keys())}
    return value


@dataclass(frozen=True)
class SourceLocation:
    file: Path
    line: int
    cell_id: Optional[str] = None
    component: Optional[str] = None

    def format(self) -> str:
        parts = [f"{self.file}:{self.line}"]
        if self.cell_id is not None:
            parts.append(f"(cell={self.cell_id})")
        if self.component is not None:
            parts.append(f"[{self.component}]")
        return " ".join(parts)


@dataclass(frozen=True)
class ManifestEntry:
    dataset: str
    kind: ApplyOpKind
    kwargs_fingerprint: Mapping[str, Any]
    call_order: int
    source_location: SourceLocation

    def diff_key(self) -> Tuple[str, ApplyOpKind, Tuple[Tuple[str, Any], ...]]:
        return (self.dataset, self.kind, tuple(sorted(self.kwargs_fingerprint.items())))


@dataclass(frozen=True)
class Manifest:
    entries: Tuple[ManifestEntry, ...]

    def by_dataset(self, dataset: str) -> Tuple[ManifestEntry, ...]:
        return tuple(e for e in self.entries if e.dataset == dataset)

    def kinds_for(self, dataset: str) -> FrozenSet[ApplyOpKind]:
        return frozenset(e.kind for e in self.entries if e.dataset == dataset)

    def datasets(self) -> FrozenSet[str]:
        return frozenset(e.dataset for e in self.entries)

    def extend(self, more: Iterable[ManifestEntry]) -> "Manifest":
        return Manifest(entries=self.entries + tuple(more))
