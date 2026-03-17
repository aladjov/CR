#!/usr/bin/env python3
"""Strip cell outputs and Papermill error cells from Jupyter notebooks.

Removes all cell outputs, execution counts, and Papermill error-injection
cells (injected when Papermill encounters an exception) while preserving
the notebook structure, cell source, and metadata.

Usage:
    python scripts/notebooks/clean_notebook_outputs.py
    python scripts/notebooks/clean_notebook_outputs.py --notebooks-dir exploration_notebooks
    python scripts/notebooks/clean_notebook_outputs.py --dry-run
"""
import argparse
import sys
from pathlib import Path
from typing import List

import nbformat

PAPERMILL_ERROR_TAG = "papermill-error-cell-tag"


def _is_papermill_error_cell(cell) -> bool:
    """Check if a cell was injected by Papermill on execution failure."""
    return PAPERMILL_ERROR_TAG in cell.get("metadata", {}).get("tags", [])


def clean_notebook(notebook_path: Path) -> bool:
    """Strip outputs and Papermill error cells. Returns True if modified."""
    nb = nbformat.read(str(notebook_path), as_version=4)
    modified = False

    # Remove Papermill error-injection cells
    original_count = len(nb.cells)
    nb.cells = [cell for cell in nb.cells if not _is_papermill_error_cell(cell)]
    if len(nb.cells) < original_count:
        modified = True

    for cell in nb.cells:
        if cell.cell_type == "code":
            if cell.outputs:
                cell.outputs = []
                modified = True
            if cell.execution_count is not None:
                cell.execution_count = None
                modified = True
    if modified:
        nbformat.write(nb, str(notebook_path))
    return modified


def clean_all(notebooks_dir: Path, dry_run: bool = False) -> List[str]:
    """Clean all notebooks in directory. Returns list of modified filenames."""
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}")
        return []

    print(f"Found {len(notebooks)} notebooks in {notebooks_dir}")
    cleaned = []
    for nb_path in notebooks:
        nb = nbformat.read(str(nb_path), as_version=4)
        has_outputs = any(
            cell.cell_type == "code"
            and (cell.outputs or cell.execution_count is not None)
            for cell in nb.cells
        )
        has_papermill_errors = any(
            _is_papermill_error_cell(cell) for cell in nb.cells
        )
        if has_outputs or has_papermill_errors:
            if dry_run:
                print(f"  {nb_path.name} — would clean")
            else:
                clean_notebook(nb_path)
                print(f"  {nb_path.name} — cleaned")
            cleaned.append(nb_path.name)
        else:
            print(f"  {nb_path.name} — already clean")

    print(f"\n{'Would clean' if dry_run else 'Cleaned'} {len(cleaned)}/{len(notebooks)} notebooks")
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Strip cell outputs from notebooks for clean commits",
    )
    parser.add_argument(
        "--notebooks-dir",
        default="exploration_notebooks",
        help="Directory containing notebooks (default: exploration_notebooks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which notebooks have outputs without cleaning",
    )
    args = parser.parse_args()
    notebooks_dir = Path(args.notebooks_dir).resolve()
    if not notebooks_dir.exists():
        print(f"Directory not found: {notebooks_dir}")
        sys.exit(1)
    clean_all(notebooks_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
