#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import List

import nbformat

_DOC_TAG_PATTERN = re.compile(
    r"^\[//\]: # \(cr:\w+"
    r"(?:\s+name='[^']*')?"
    r"(?:\s+id=\S+)?"
    r"\)$"
)


def _source_lines(cell) -> List[str]:
    src = cell.source
    if isinstance(src, str):
        return src.splitlines(keepends=True)
    return list(src)


def _has_doc_tag(lines: List[str]) -> bool:
    if not lines:
        return False
    return _DOC_TAG_PATTERN.match(lines[0].rstrip("\n")) is not None


def _derive_name(lines: List[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            name = re.sub(r"[^a-zA-Z0-9]+", "_", heading).strip("_").lower()
            if name:
                return name[:60]
    return "section"


def tag_markdown_cells(nb: nbformat.NotebookNode) -> int:
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        lines = _source_lines(cell)
        if _has_doc_tag(lines):
            continue
        cell_id = cell.get("id", "00000000")
        name = _derive_name(lines)
        tag_line = f"[//]: # (cr:doc name='{name}' id={cell_id})\n"
        tagged = [tag_line, *lines]
        cell.source = "".join(tagged) if isinstance(cell.source, str) else tagged
        count += 1
    return count


def tag_notebook(nb_path: Path, dry_run: bool = False) -> int:
    nb = nbformat.read(str(nb_path), as_version=4)
    count = tag_markdown_cells(nb)
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
            print(f"  {nb_path.name} — {label} {count} markdown cells")
        else:
            print(f"  {nb_path.name} — no untagged markdown cells")

    label = "Would tag" if dry_run else "Tagged"
    print(f"\n{label} {total} markdown cells total")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Tag all untagged markdown cells with [//]: # (cr:doc ...)",
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
