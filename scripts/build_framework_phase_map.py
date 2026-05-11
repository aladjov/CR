#!/usr/bin/env python3
"""Build framework/phase_map.yaml from exploration_notebooks/.

Runs deterministically — same notebook input produces byte-identical
YAML output, which is what the pre-commit rail relies on for
regression detection. Because wall-clock timestamps would defeat that
guarantee, the output uses `source_fingerprint` (a content SHA-256 of
the input notebooks, sorted by path) instead of `generated_at` per
plan § 5.3.

Usage:
    python scripts/build_framework_phase_map.py [--check]

Without `--check`: regenerate `framework/phase_map.yaml` in place.
With `--check`: compare existing committed file against a fresh build;
exit 1 with the mismatched keys listed if they differ. The pre-commit
hook uses `--check`.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "exploration_notebooks"
OUTPUT_PATH = REPO_ROOT / "framework" / "phase_map.yaml"

KNOWN_NOTEBOOKS = {
    "-1_parity_contract",
    "00_start_here",
    "01_data_discovery",
    "01a_a_temporal_text_deep_dive",
    "01a_temporal_deep_dive",
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
}

KNOWN_STAGE_KEYWORDS = {
    "landing": "landing_post",
    "loading": "landing_post",
    "filter": "landing_post",
    "lifecycle": "landing_post",
    "enrichment": "landing_post",
    "enrich": "landing_post",
    "semantics": "landing_post",
    "fingerprint": "landing_post",
    "merge": "bronze_merge",
    "dataset_merge": "bronze_merge",
    "join": "bronze_merge",
    "aggregation": "bronze_post",
    "aggregate": "bronze_post",
    "aggregated": "bronze_post",
    "quality": "bronze_post",
    "integrity": "bronze_post",
    "cleaning": "bronze_post",
    "derived": "silver_post",
    "feature": "silver_post",
    "features": "silver_post",
    "relationship": "silver_post",
    "modeling": "silver_post",
    "interaction": "silver_post",
    "target": "target_derive",
    "prediction_objective": "target_derive",
    "training": "training",
    "baseline": "training",
    "experiment": "training",
    "experiments": "training",
    "model": "training",
}

NOTEBOOK_NUMBER_TO_STAGE = {
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

DOC_MARKER_RE = re.compile(r"cr:doc\s+name='(?P<name>[^']+)'\s+id=(?P<id>[^)\s]+)")
ANNOTATION_RE = re.compile(r"^\s*#\s*@cr:code\s+phase=(?P<phase>[a-z_]+)", re.MULTILINE)
NB_NUMBER_RE = re.compile(r"^(?P<n>\d{2})[a-z]*_")
CR_CODE_ID_RE = re.compile(r"#\s*@cr:code[^\n]*id=(?P<id>[\w-]+)")


def build_phase_map(notebook_paths: List[Path]) -> Dict[str, Any]:
    sections: Dict[str, Dict[str, Any]] = {}
    unmappable: List[Dict[str, Any]] = []

    for nb_path in sorted(notebook_paths):
        nb = nbformat.read(nb_path, as_version=4)
        _process_notebook(nb_path, nb, sections, unmappable)

    return {
        "version": 1,
        "source_fingerprint": _fingerprint(notebook_paths),
        "generated_from": [_relpath(p) for p in sorted(notebook_paths)],
        "sections": dict(sorted(sections.items())),
        "unmappable_cells": sorted(unmappable, key=lambda d: (d["notebook"], d["cell_id"])),
    }


def _relpath(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _process_notebook(
    nb_path: Path,
    nb: Any,
    sections: Dict[str, Dict[str, Any]],
    unmappable: List[Dict[str, Any]],
) -> None:
    stem = nb_path.stem
    current_section: Optional[str] = None

    for cell in nb.cells:
        src = cell.source or ""
        if cell.cell_type == "markdown":
            match = DOC_MARKER_RE.search(src)
            if match:
                current_section = match.group("name")
                key = f"{stem}#{current_section}"
                if key not in sections:
                    sections[key] = _infer_section_entry(stem, current_section)
            continue
        if cell.cell_type != "code":
            continue
        annotation_match = ANNOTATION_RE.search(src)
        if annotation_match:
            anchor = current_section or _code_cell_id(src) or "<no_section>"
            key = f"{stem}#{anchor}"
            sections[key] = {
                "stage": annotation_match.group("phase"),
                "notebook": stem,
                "source": "annotation",
            }
            continue
        if current_section is None:
            unmappable.append({
                "notebook": stem,
                "cell_id": _code_cell_id(src) or _cell_metadata_id(cell) or "<unknown>",
                "reason": "no preceding section header within notebook",
            })


def _infer_section_entry(stem: str, section_name: str) -> Dict[str, Any]:
    stage = _infer_stage_from_section(section_name)
    if stage:
        return {"stage": stage, "notebook": stem, "source": "keyword_match"}
    nb_num = _notebook_number(stem)
    if nb_num and nb_num in NOTEBOOK_NUMBER_TO_STAGE:
        return {
            "stage": NOTEBOOK_NUMBER_TO_STAGE[nb_num],
            "notebook": stem,
            "source": "notebook_fallback",
        }
    return {"stage": None, "notebook": stem, "source": "unmappable"}


def _infer_stage_from_section(section_name: str) -> Optional[str]:
    tokens = [t for t in re.split(r"[_\s]+", section_name.lower()) if t]
    for t in tokens:
        if t in KNOWN_STAGE_KEYWORDS:
            return KNOWN_STAGE_KEYWORDS[t]
    return None


def _notebook_number(stem: str) -> Optional[str]:
    m = NB_NUMBER_RE.match(stem)
    return m.group("n") if m else None


def _code_cell_id(src: str) -> Optional[str]:
    m = CR_CODE_ID_RE.search(src)
    return m.group("id") if m else None


def _cell_metadata_id(cell: Any) -> Optional[str]:
    return getattr(cell, "id", None) or cell.get("id") if isinstance(cell, dict) else None


def _fingerprint(notebook_paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(notebook_paths):
        h.update(p.name.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.read_bytes())
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


def find_notebooks(notebook_dir: Path) -> List[Path]:
    return [
        p for p in notebook_dir.glob("*.ipynb")
        if p.stem in KNOWN_NOTEBOOKS
    ]


def serialize(phase_map: Dict[str, Any]) -> str:
    return yaml.safe_dump(phase_map, default_flow_style=False, sort_keys=False, width=120)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare committed file to fresh build; exit 1 on drift")
    parser.add_argument("--notebooks-dir", type=Path, default=NOTEBOOKS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    notebooks = find_notebooks(args.notebooks_dir)
    if not notebooks:
        print(f"error: no notebooks found in {args.notebooks_dir}", file=sys.stderr)
        return 2

    phase_map = build_phase_map(notebooks)
    rendered = serialize(phase_map)

    if args.check:
        if not args.output.exists():
            print(f"error: {args.output} does not exist; run without --check first",
                  file=sys.stderr)
            return 1
        committed = args.output.read_text()
        if committed == rendered:
            return 0
        print(
            f"error: {args.output.relative_to(REPO_ROOT)} is stale. "
            f"Run: python scripts/build_framework_phase_map.py and stage the result.",
            file=sys.stderr,
        )
        _print_diff_summary(committed, rendered)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"wrote {args.output.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"  sections: {len(phase_map['sections'])}", file=sys.stderr)
    print(f"  unmappable_cells: {len(phase_map['unmappable_cells'])}", file=sys.stderr)
    return 0


def _print_diff_summary(old: str, new: str) -> None:
    try:
        old_map = yaml.safe_load(old) or {}
        new_map = yaml.safe_load(new) or {}
    except Exception:
        return
    old_keys = set((old_map.get("sections") or {}).keys())
    new_keys = set((new_map.get("sections") or {}).keys())
    added = new_keys - old_keys
    removed = old_keys - new_keys
    changed: List[str] = []
    for k in old_keys & new_keys:
        if old_map["sections"][k] != new_map["sections"][k]:
            changed.append(k)
    if added:
        print("  added sections:", file=sys.stderr)
        for k in sorted(added)[:10]:
            print(f"    + {k}", file=sys.stderr)
    if removed:
        print("  removed sections:", file=sys.stderr)
        for k in sorted(removed)[:10]:
            print(f"    - {k}", file=sys.stderr)
    if changed:
        print("  changed sections:", file=sys.stderr)
        for k in sorted(changed)[:10]:
            print(f"    ~ {k}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
