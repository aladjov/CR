#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List

import nbformat

from customer_retention.generators.notebook_sync.cell_types import (
    CellSyncType,
    has_magic_comment,
    prepend_magic_comment,
)


def _source_lines(cell) -> List[str]:
    src = cell.source
    if isinstance(src, str):
        return src.splitlines(keepends=True)
    return list(src)


def tag_untagged_code_cells(nb: nbformat.NotebookNode) -> int:
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = _source_lines(cell)
        if has_magic_comment(lines):
            continue
        tagged = prepend_magic_comment(lines, CellSyncType.CODE)
        cell.source = "".join(tagged) if isinstance(cell.source, str) else tagged
        count += 1
    return count


def tag_notebook(nb_path: Path, dry_run: bool = False) -> int:
    nb = nbformat.read(str(nb_path), as_version=4)
    count = tag_untagged_code_cells(nb)
    if count > 0 and not dry_run:
        nbformat.write(nb, str(nb_path))
    return count


def tag_all(notebooks_dir: Path, dry_run: bool = False) -> int:
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}")
        return 0

    print(f"Found {len(notebooks)} notebooks in {notebooks_dir}")
    total = 0
    for nb_path in notebooks:
        count = tag_notebook(nb_path, dry_run=dry_run)
        total += count
        if count > 0:
            label = "would tag" if dry_run else "tagged"
            print(f"  {nb_path.name} — {label} {count} cells")
        else:
            print(f"  {nb_path.name} — no untagged cells")

    label = "Would tag" if dry_run else "Tagged"
    print(f"\n{label} {total} cells total")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Tag all untagged framework code cells with # @cr:code",
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
    tag_all(notebooks_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
