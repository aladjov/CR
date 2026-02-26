#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nbformat

from customer_retention.generators.notebook_sync.cell_id_standardizer import (
    split_blended_cell,
    standardize_cell_ids,
)
from customer_retention.generators.notebook_sync.cell_types import CellSyncType, prepend_magic_comment

SplitSpec = Tuple[str, int]

SPLIT_MAP: Dict[str, List[SplitSpec]] = {
    "00_start_here.ipynb": [
        ("cell-1", 6),
        ("cell-5", 4),
        ("cell-11", 4),
        ("sidz4bufnc", 27),
        ("6vilcr3g1hr", 4),
    ],
    "01_data_discovery.ipynb": [
        ("cell-3", 60),
    ],
    "01b_temporal_quality.ipynb": [
        ("cell-3", 1),
        ("cell-5", 3),
    ],
    "01c_temporal_patterns.ipynb": [
        ("19b4e63f", 1),
        ("9e9c32ab", 4),
        ("0e8069c1", 7),
    ],
    "01d_event_aggregation.ipynb": [
        ("cell-3", 1),
        ("cell-7", 5),
    ],
    "10_spec_generation.ipynb": [
        ("b805976d", 35),
    ],
}

TAG_ONLY_MAP: Dict[str, Dict[str, CellSyncType]] = {
    "01_data_discovery.ipynb": {
        "active-dataset-code": CellSyncType.USER_CODE,
    },
    "09_business_alignment.ipynb": {
        "a947050e": CellSyncType.USER_CODE,
        "aec679b2": CellSyncType.USER_CODE,
        "27e74568": CellSyncType.USER_CODE,
        "f6682f6e": CellSyncType.USER_CODE,
        "97c21062": CellSyncType.USER_CODE,
    },
}

DUMMY_SPLIT_MAP: Dict[str, List[SplitSpec]] = {
    "99_dummy_migration.ipynb": [
        ("cell-1", 4),
    ],
}

DUMMY_TAG_ONLY_MAP: Dict[str, Dict[str, CellSyncType]] = {
    "99_dummy_migration.ipynb": {
        "cell-2": CellSyncType.CONFIG,
        "active-dataset-code": CellSyncType.USER_CODE,
    },
}


def _ensure_source_lines(cell) -> List[str]:
    src = cell.source
    if isinstance(src, str):
        return src.splitlines(keepends=True)
    return list(src)



def _find_cell_index_optional(nb: nbformat.NotebookNode, cell_id: str) -> Optional[int]:
    for i, cell in enumerate(nb.cells):
        if cell.id == cell_id:
            return i
    return None


def apply_splits(
    nb: nbformat.NotebookNode,
    filename: str,
    split_map: Dict[str, List[SplitSpec]],
) -> int:
    splits = split_map.get(filename, [])
    offset = 0
    for cell_id, split_after_line in splits:
        idx = _find_cell_index_optional(nb, cell_id)
        if idx is None:
            continue
        cell = nb.cells[idx]
        cell.source = _ensure_source_lines(cell)
        config_cell, code_cell = split_blended_cell(
            cell, split_after_line, f"{cell_id}-cfg", f"{cell_id}-code"
        )
        nb.cells[idx : idx + 1] = [config_cell, code_cell]
        offset += 1
    return offset


def apply_tags(
    nb: nbformat.NotebookNode,
    filename: str,
    tag_only_map: Dict[str, Dict[str, CellSyncType]],
) -> None:
    tags = tag_only_map.get(filename, {})
    for cell_id, cell_type in tags.items():
        idx = _find_cell_index_optional(nb, cell_id)
        if idx is None:
            continue
        cell = nb.cells[idx]
        lines = _ensure_source_lines(cell)
        cell.source = prepend_magic_comment(lines, cell_type)


def apply_config_tags_to_split_cells(nb: nbformat.NotebookNode, filename: str, split_map: Dict[str, List[SplitSpec]]) -> None:
    splits = split_map.get(filename, [])
    for cell_id, _ in splits:
        cfg_id = f"{cell_id}-cfg"
        idx = _find_cell_index_optional(nb, cfg_id)
        if idx is None:
            continue
        cell = nb.cells[idx]
        lines = _ensure_source_lines(cell)
        cell.source = prepend_magic_comment(lines, CellSyncType.CONFIG)


def migrate_notebook(
    nb_path: Path,
    dry_run: bool = False,
    split_map: Optional[Dict[str, List[SplitSpec]]] = None,
    tag_only_map: Optional[Dict[str, Dict[str, CellSyncType]]] = None,
) -> Tuple[int, int]:
    if split_map is None:
        split_map = SPLIT_MAP
    if tag_only_map is None:
        tag_only_map = TAG_ONLY_MAP

    filename = nb_path.name
    nb = nbformat.read(str(nb_path), as_version=4)
    original_count = len(nb.cells)

    apply_splits(nb, filename, split_map)
    apply_config_tags_to_split_cells(nb, filename, split_map)
    apply_tags(nb, filename, tag_only_map)
    nb = standardize_cell_ids(nb, filename)

    nbformat.validate(nb)

    if not dry_run:
        nbformat.write(nb, str(nb_path))

    return original_count, len(nb.cells)


def migrate_all(
    notebooks_dir: Path,
    dry_run: bool = False,
    split_map: Optional[Dict[str, List[SplitSpec]]] = None,
    tag_only_map: Optional[Dict[str, Dict[str, CellSyncType]]] = None,
) -> List[str]:
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}")
        return []

    print(f"Found {len(notebooks)} notebooks in {notebooks_dir}")
    migrated = []
    for nb_path in notebooks:
        try:
            orig, final = migrate_notebook(nb_path, dry_run=dry_run, split_map=split_map, tag_only_map=tag_only_map)
            label = "would migrate" if dry_run else "migrated"
            splits = final - orig
            suffix = f" ({splits} splits)" if splits else ""
            print(f"  {nb_path.name} — {label}{suffix}")
            migrated.append(nb_path.name)
        except Exception as e:
            print(f"  {nb_path.name} — ERROR: {e}")
    return migrated


def main():
    parser = argparse.ArgumentParser(
        description="Migrate notebook cell IDs to standardized format",
    )
    parser.add_argument(
        "--notebooks-dir",
        default="exploration_notebooks",
        help="Directory containing notebooks (default: exploration_notebooks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    args = parser.parse_args()
    notebooks_dir = Path(args.notebooks_dir).resolve()
    if not notebooks_dir.exists():
        print(f"Directory not found: {notebooks_dir}")
        sys.exit(1)
    migrate_all(notebooks_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
