"""Parse the two schedules that govern a job's apply-op contract.

The **outer schedule** is the engagement's notebook DAG: `-1_parity_contract`
runs first (pre-flight), then `00_start_here`, `01_data_discovery`, ...,
`10_spec_generation`. It is inferred from numeric prefixes on filenames
unless an explicit `workflow.yml` overrides the order.

The **inner schedule** is the generated pipeline's per-stage notebook DAG.
It is extracted by parsing the `pipeline_runner.py` (or `run_all.py`) that
the generator emits: each `from <stage>.<source> import …` line places one
notebook under one stage, and the stage order is the import order.

Both schedules are read-only inputs to the audit; nothing here mutates files.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

_STAGE_NAMES = ("landing", "target_derive", "bronze", "silver", "gold", "training")

_PREFIX_RE = re.compile(r"^(-?\d+[a-z]?(?:_[a-z])*)_")


@dataclass(frozen=True)
class ScheduledNotebook:
    path: Path
    prefix: str
    sort_key: Tuple


@dataclass(frozen=True)
class GeneratedStage:
    name: str
    notebooks: Tuple[Path, ...]


@dataclass(frozen=True)
class JobSchedule:
    outer: Tuple[ScheduledNotebook, ...]
    inner: Tuple[GeneratedStage, ...]


def parse_outer_schedule(
    engagement_dir: Path,
    *,
    explicit_schedule_file: Optional[Path] = None,
) -> Tuple[ScheduledNotebook, ...]:
    engagement_dir = Path(engagement_dir)
    if explicit_schedule_file is not None:
        return _parse_explicit_schedule(engagement_dir, explicit_schedule_file)
    return _parse_numeric_prefix_schedule(engagement_dir)


def parse_inner_schedule(runner_path: Path) -> Tuple[GeneratedStage, ...]:
    runner_path = Path(runner_path)
    if not runner_path.exists():
        return ()
    try:
        tree = ast.parse(runner_path.read_text())
    except SyntaxError:
        return ()
    stage_notebooks: dict[str, list[Path]] = {}
    stage_order: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        head, _, source = node.module.partition(".")
        if head not in _STAGE_NAMES or not source:
            continue
        if head not in stage_notebooks:
            stage_notebooks[head] = []
            stage_order.append(head)
        stage_notebooks[head].append(Path(node.module.replace(".", "/")))
    return tuple(
        GeneratedStage(name=stage, notebooks=tuple(stage_notebooks[stage]))
        for stage in stage_order
    )


# ---------------------------------------------------------------------------
# Outer schedule parsers
# ---------------------------------------------------------------------------


def _parse_numeric_prefix_schedule(engagement_dir: Path) -> Tuple[ScheduledNotebook, ...]:
    scheduled: list[ScheduledNotebook] = []
    for nb in engagement_dir.glob("*.ipynb"):
        prefix = _extract_prefix(nb.name)
        if prefix is None:
            continue
        scheduled.append(
            ScheduledNotebook(
                path=nb,
                prefix=prefix,
                sort_key=_sort_key_for_prefix(prefix),
            )
        )
    scheduled.sort(key=lambda n: n.sort_key)
    return tuple(scheduled)


def _parse_explicit_schedule(
    engagement_dir: Path,
    schedule_file: Path,
) -> Tuple[ScheduledNotebook, ...]:
    text = schedule_file.read_text()
    notebooks: list[ScheduledNotebook] = []
    order = 0
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^-\s*notebook:\s*(.+\.ipynb)\s*$", stripped)
        if not match:
            continue
        name = match.group(1)
        path = engagement_dir / name
        prefix = _extract_prefix(name) or ""
        notebooks.append(
            ScheduledNotebook(path=path, prefix=prefix, sort_key=(order, name))
        )
        order += 1
    return tuple(notebooks)


def _extract_prefix(name: str) -> Optional[str]:
    match = _PREFIX_RE.match(name)
    return match.group(1) if match else None


def _sort_key_for_prefix(prefix: str) -> Tuple:
    match = re.match(r"^(-?\d+)([a-z]?(?:_[a-z])*)$", prefix)
    if not match:
        return (10**6, prefix)
    return (int(match.group(1)), match.group(2))


__all__ = [
    "GeneratedStage",
    "JobSchedule",
    "ScheduledNotebook",
    "parse_inner_schedule",
    "parse_outer_schedule",
]
