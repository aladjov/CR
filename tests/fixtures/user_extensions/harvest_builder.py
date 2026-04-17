"""Programmatic fixture builders for user-extensions tests.

Per plan § 1b.3: `make_harvest(...)` and `make_notebook(...)` live in one
module so every test that needs a `HarvestResult` or a synthetic notebook
constructs one by keyword, not by copy-paste. Keep the surface small — one
factory per type, kwargs with sensible defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import nbformat

from customer_retention.runtime.harvest import HarvestResult, PhaseMap
from customer_retention.runtime.registry import RegisteredFunction


def make_registered_function(
    name: str,
    *,
    source: Optional[str] = None,
    scope: str = "dataset",
    dataset: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    primary: Optional[str] = None,
    replay_at_scoring: bool = False,
    expected_stage: Optional[str] = None,
    notebook_path: Optional[str] = None,
    cell_id: Optional[str] = None,
    inferred_stage: Optional[str] = None,
) -> RegisteredFunction:
    if scope == "dataset" and dataset is None:
        dataset = "request"
    return RegisteredFunction(
        name=name,
        source=source or f"def {name}(spark, df, namespace):\n    return df\n",
        scope=scope,
        dataset=dataset,
        datasets=datasets,
        primary=primary,
        replay_at_scoring=replay_at_scoring,
        expected_stage=expected_stage,
        notebook_path=Path(notebook_path) if notebook_path else None,
        cell_id=cell_id,
        inferred_stage=inferred_stage,
    )


def make_harvest(
    *,
    dataset_scoped: Optional[List[Dict[str, Any]]] = None,
    cross_dataset: Optional[List[Dict[str, Any]]] = None,
    unmappable: Optional[List[Dict[str, Any]]] = None,
) -> HarvestResult:
    """Build a `HarvestResult` from a declarative spec.

    Each entry in `dataset_scoped` needs at minimum `name` and `dataset`
    plus `inferred_stage` (placed into `functions_by_target`). Cross-dataset
    entries go into `cross_dataset_steps`. Unmappable entries are
    `(rf_kwargs, reason)` dicts.
    """
    result = HarvestResult.empty()
    for spec in dataset_scoped or []:
        stage = spec.get("inferred_stage") or "landing_post"
        dataset = spec.get("dataset") or "request"
        rf = make_registered_function(scope="dataset", **spec)
        result.functions_by_target.setdefault((stage, dataset), []).append(rf)
    for spec in cross_dataset or []:
        kwargs = dict(spec)
        kwargs.setdefault("scope", "datasets")
        rf = make_registered_function(**kwargs)
        result.cross_dataset_steps.append(rf)
    for entry in unmappable or []:
        rf = make_registered_function(**entry["spec"])
        result.unmappable.append((rf, entry["reason"]))
    return result


def make_phase_map(entries: Optional[Dict[str, str]] = None) -> PhaseMap:
    """Construct a synthetic PhaseMap from `{cell_id: stage}` pairs.

    Bypasses the YAML loader for the common case where tests want to
    pin a couple of cell-id → stage mappings inline without writing a
    fixture file.
    """
    pm = PhaseMap.empty()
    for cell_id, stage in (entries or {}).items():
        pm.sections[cell_id] = {"stage": stage}
    return pm


def make_notebook(
    cells: Iterable[Dict[str, Any]],
    *,
    kernel_name: str = "python3",
) -> nbformat.NotebookNode:
    """Build an in-memory nbformat notebook for phase-map builder tests.

    Each cell spec is `{"type": "code"|"markdown", "source": "...",
    "id": "hex"}`. Header-shaped markdown cells set section anchors for
    the phase_map builder; code cells carry `# @cr:` tags on line 1.
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": kernel_name, "display_name": kernel_name}
    for spec in cells:
        cell_type = spec.get("type", "code")
        source = spec.get("source", "")
        if cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source=source)
        else:
            cell = nbformat.v4.new_code_cell(source=source)
        if "id" in spec:
            cell["id"] = spec["id"]
        nb.cells.append(cell)
    return nb
