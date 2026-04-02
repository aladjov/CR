#!/usr/bin/env python3
"""Extract per-cell performance profiles from executed Jupyter notebooks.

Reads papermill timing metadata (``cell.metadata.papermill.duration``) and
``@cr:`` tags from executed ``.ipynb`` files, producing a unified JSON
artifact (``cell_profiles.json``) that can be compared across runs and
environments (local vs Databricks).

Usage:
    python scripts/experiments/cell_profiling.py extract \
        --notebooks-dir exploration_notebooks \
        --output experiments/runs/my-run/cell_profiles.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants — notebook ordering & @cr: tag regexes (shared with
# capture_notebook_outputs.py)
# ---------------------------------------------------------------------------

NOTEBOOKS_ORDER = [
    "00_start_here",
    "01_data_discovery",
    "01a_temporal_deep_dive",
    "01a_a_temporal_text_deep_dive",
    "01b_temporal_quality",
    "01c_temporal_patterns",
    "01d_event_aggregation",
    "02_source_integrity",
    "03_dataset_merge",
    "04_column_deep_dive",
    "04a_text_columns_deep_dive",
    "05_relationship_analysis",
    "06_feature_opportunities",
    "07_modeling_readiness",
    "08_baseline_experiments",
    "09_business_alignment",
    "10_spec_generation",
    "11_scoring_validation",
    "12_view_documentation",
]

_CR_CODE_RE = re.compile(
    r"^#\s*@cr:(?P<kind>code|config|user_code|code_system)"
    r"\s+name='(?P<name>[^']+)'\s+id=(?P<id>\w+)"
)
_CR_DOC_RE = re.compile(r"\[//\]:\s*#\s*\(cr:(?:doc)\s+name='(?P<name>[^']+)'\s+id=(?P<id>\w+)\)")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CellProfileEntry:
    cell_name: str
    cell_id: str
    elapsed_sec: float
    status: str  # "completed", "failed", "skipped"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    spark_jobs: Optional[int] = None
    peak_memory_mb: Optional[float] = None


@dataclass
class NotebookProfile:
    total_elapsed: float
    cells: list[CellProfileEntry] = field(default_factory=list)


@dataclass
class CellProfileManifest:
    version: int = 1
    environment: str = "local"
    generated_at: str = ""
    notebooks: dict[str, NotebookProfile] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_cr_tag(cell: dict) -> Optional[dict]:
    """Extract @cr: / cr:doc tag from the first source line of a cell."""
    source = cell.get("source", [])
    if not source:
        return None
    first_line = source[0] if isinstance(source, list) else source.split("\n")[0]
    m = _CR_CODE_RE.match(first_line)
    if m:
        return {"kind": m.group("kind"), "name": m.group("name"), "id": m.group("id")}
    m = _CR_DOC_RE.search(first_line)
    if m:
        return {"kind": "doc", "name": m.group("name"), "id": m.group("id")}
    return None


def _cell_id(cell: dict) -> str:
    """Return the stable cell ID, falling back to 'unknown'."""
    return cell.get("id", "unknown")


# ---------------------------------------------------------------------------
# Local extraction — read papermill metadata from executed .ipynb files
# ---------------------------------------------------------------------------


def extract_profiles_from_notebook(nb_path: Path) -> Optional[NotebookProfile]:
    """Extract per-cell timing from a single executed notebook.

    Returns ``None`` if the notebook has no papermill metadata (never executed).
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    entries: list[CellProfileEntry] = []
    total_elapsed = 0.0
    has_papermill = False

    for cell in cells:
        # Only profile code/config cells, skip markdown
        if cell.get("cell_type") != "code":
            continue

        pm = cell.get("metadata", {}).get("papermill", {})
        if not pm:
            continue
        has_papermill = True

        duration = pm.get("duration") or 0.0
        status = pm.get("status", "completed")
        start_time = pm.get("start_time")
        end_time = pm.get("end_time")
        exception = pm.get("exception", False)
        if exception:
            status = "failed"

        tag = _parse_cr_tag(cell)
        cell_name = tag["name"] if tag else f"cell_{_cell_id(cell)}"
        cell_id_val = tag["id"] if tag else _cell_id(cell)

        entries.append(
            CellProfileEntry(
                cell_name=cell_name,
                cell_id=cell_id_val,
                elapsed_sec=round(duration, 6),
                status=status,
                start_time=start_time,
                end_time=end_time,
            )
        )
        total_elapsed += duration

    if not has_papermill:
        return None
    return NotebookProfile(total_elapsed=round(total_elapsed, 3), cells=entries)


def extract_profiles_from_notebooks(notebooks_dir: Path) -> CellProfileManifest:
    """Walk a directory of executed notebooks and build a profile manifest."""
    manifest = CellProfileManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        environment="local",
    )

    # Discover notebooks in canonical order, then any extras
    found: dict[str, Path] = {}
    for p in sorted(notebooks_dir.glob("*.ipynb")):
        stem = p.stem
        if stem.startswith(".") or stem.startswith("_"):
            continue
        found[stem] = p

    ordered = [n for n in NOTEBOOKS_ORDER if n in found]
    extras = sorted(set(found) - set(NOTEBOOKS_ORDER))
    for name in ordered + extras:
        profile = extract_profiles_from_notebook(found[name])
        if profile is not None:
            manifest.notebooks[name] = profile

    return manifest


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def write_cell_profiles(manifest: CellProfileManifest, output_path: Path) -> None:
    """Write a profile manifest to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_cell_profiles(path: Path) -> Optional[CellProfileManifest]:
    """Load a profile manifest from JSON.  Returns ``None`` if missing."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    notebooks = {}
    for nb_name, nb_data in data.get("notebooks", {}).items():
        cells = [CellProfileEntry(**c) for c in nb_data.get("cells", [])]
        notebooks[nb_name] = NotebookProfile(
            total_elapsed=nb_data.get("total_elapsed", 0.0),
            cells=cells,
        )
    return CellProfileManifest(
        version=data.get("version", 1),
        environment=data.get("environment", "local"),
        generated_at=data.get("generated_at", ""),
        notebooks=notebooks,
    )


# ---------------------------------------------------------------------------
# Merge helper — for multi-dataset runs where same notebook executes twice
# ---------------------------------------------------------------------------


def merge_profiles(a: CellProfileManifest, b: CellProfileManifest) -> CellProfileManifest:
    """Merge two manifests, taking the entry with longer total_elapsed per notebook."""
    merged = CellProfileManifest(
        environment=a.environment or b.environment,
        generated_at=a.generated_at or b.generated_at,
        notebooks={**a.notebooks},
    )
    for nb_name, nb_profile in b.notebooks.items():
        existing = merged.notebooks.get(nb_name)
        if existing is None or nb_profile.total_elapsed > existing.total_elapsed:
            merged.notebooks[nb_name] = nb_profile
    return merged


# ---------------------------------------------------------------------------
# Sidecar merge — combine JSONL metrics (Spark jobs, memory) with timing
# ---------------------------------------------------------------------------

_SIDECAR_GLOB = ".cr_cell_metrics_*.jsonl"


def _load_sidecar(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            entries.append(json.loads(line))
    return entries


def merge_sidecar_metrics(
    manifest: CellProfileManifest,
    sidecar_dir: Path,
) -> CellProfileManifest:
    """Merge JSONL sidecar files into a profile manifest.

    Sidecar files are named ``.cr_cell_metrics_{notebook_stem}.jsonl`` and
    contain one JSON object per cell with ``spark_jobs`` and ``peak_memory_mb``.
    Matching is by ``cell_id`` first, then ``cell_name``.
    """
    for sidecar_path in sorted(sidecar_dir.glob(_SIDECAR_GLOB)):
        nb_name = sidecar_path.stem.removeprefix(".cr_cell_metrics_")
        nb_profile = manifest.notebooks.get(nb_name)
        if nb_profile is None:
            continue
        entries = _load_sidecar(sidecar_path)
        by_id = {e["cell_id"]: e for e in entries if e.get("cell_id")}
        by_name = {e["cell_name"]: e for e in entries if e.get("cell_name")}
        for cell in nb_profile.cells:
            match = by_id.get(cell.cell_id) or by_name.get(cell.cell_name)
            if match:
                cell.spark_jobs = match.get("spark_jobs")
                cell.peak_memory_mb = match.get("peak_memory_mb")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract per-cell performance profiles from executed notebooks.")
    sub = parser.add_subparsers(dest="command")

    extract_p = sub.add_parser("extract", help="Extract profiles from executed .ipynb files")
    extract_p.add_argument(
        "--notebooks-dir",
        type=Path,
        required=True,
        help="Directory containing executed .ipynb files",
    )
    extract_p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for cell_profiles.json",
    )

    args = parser.parse_args(argv)
    if args.command == "extract":
        manifest = extract_profiles_from_notebooks(args.notebooks_dir)
        write_cell_profiles(manifest, args.output)
        nb_count = len(manifest.notebooks)
        cell_count = sum(len(np.cells) for np in manifest.notebooks.values())
        print(f"Extracted profiles: {cell_count} cells across {nb_count} notebooks -> {args.output}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
