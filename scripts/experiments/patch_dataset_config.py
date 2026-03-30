#!/usr/bin/env python3
"""Toggle the active dataset in 00_start_here.ipynb.

Finds the ``@cr:config name='dataset_paths'`` cell and activates the
requested dataset block while commenting out all others.

Usage:
    # Activate retail dataset
    python scripts/experiments/patch_dataset_config.py \
        --notebook exploration_notebooks/00_start_here.ipynb \
        --activate customer_retention_retail

    # Restore from backup
    python scripts/experiments/patch_dataset_config.py \
        --notebook exploration_notebooks/00_start_here.ipynb \
        --restore
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_CELL_TAG = "dataset_paths"
_CELL_ID = "f1eb641c"


def _find_config_cell(nb: dict) -> int | None:
    """Return the index of the dataset_paths config cell."""
    for i, cell in enumerate(nb.get("cells", [])):
        cid = cell.get("id", "")
        src = cell.get("source", [])
        first_line = src[0] if src else ""
        if cid == _CELL_ID or _CELL_TAG in first_line:
            return i
    return None


def _parse_dataset_blocks(source_lines: list[str]) -> list[dict]:
    """Parse the cell source into dataset blocks.

    Returns a list of dicts:
        {
            "header_idx": int,       # index of the `datasets = {` line
            "end_idx": int,          # index of the closing `}` line
            "name": str,             # dataset key name (first key)
            "active": bool,          # True if NOT commented out
        }
    """
    blocks = []
    i = 0
    while i < len(source_lines):
        line = source_lines[i]

        # Match `datasets = {` or `# datasets = {`
        if re.match(r"^#?\s*datasets\s*=\s*\{", line.strip()):
            is_active = not line.strip().startswith("#")
            header_idx = i

            # Check if it's a single-line dict
            if "}" in line and "{" in line:
                end_idx = i
            else:
                # Find the closing brace
                end_idx = i
                j = i + 1
                while j < len(source_lines):
                    if "}" in source_lines[j]:
                        end_idx = j
                        break
                    j += 1
                else:
                    end_idx = j - 1

            # Extract dataset name from keys
            block_text = "".join(
                ln.lstrip("# ") for ln in source_lines[header_idx : end_idx + 1]
            )
            name_match = re.search(r'"(\w+)"', block_text)
            name = name_match.group(1) if name_match else f"block_{header_idx}"

            blocks.append(
                {
                    "header_idx": header_idx,
                    "end_idx": end_idx,
                    "name": name,
                    "active": is_active,
                }
            )
            i = end_idx + 1
        else:
            i += 1

    return blocks


def _toggle_block(
    source_lines: list[str], block: dict, activate: bool
) -> list[str]:
    """Comment or uncomment lines in a dataset block."""
    result = list(source_lines)
    for idx in range(block["header_idx"], block["end_idx"] + 1):
        line = result[idx]
        is_commented = line.lstrip().startswith("#")

        if activate and is_commented:
            # Uncomment: remove the first `# ` or `#`
            result[idx] = re.sub(r"^(\s*)#\s?", r"\1", line, count=1)
        elif not activate and not is_commented:
            # Comment: add `# ` at the start (preserving indent)
            m = re.match(r"^(\s*)", line)
            indent = m.group(1) if m else ""
            result[idx] = f"{indent}# {line.lstrip()}"

    return result


def patch_notebook(nb_path: Path, target_dataset: str) -> None:
    """Activate *target_dataset* in the notebook and comment out others."""
    backup = nb_path.with_suffix(".ipynb.bak")
    if not backup.exists():
        shutil.copy2(nb_path, backup)
        print(f"Backup saved to {backup}", file=sys.stderr)

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cell_idx = _find_config_cell(nb)
    if cell_idx is None:
        print(f"Error: could not find {_CELL_TAG} cell in {nb_path}", file=sys.stderr)
        sys.exit(1)

    cell = nb["cells"][cell_idx]
    source = cell["source"]

    # Normalize source to list of lines
    if isinstance(source, str):
        lines = source.split("\n")
        lines = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
    else:
        lines = list(source)

    blocks = _parse_dataset_blocks(lines)
    if not blocks:
        print("Error: no dataset blocks found in cell", file=sys.stderr)
        sys.exit(1)

    target_block = None
    for b in blocks:
        if target_dataset in b["name"]:
            target_block = b
            break

    if target_block is None:
        available = [b["name"] for b in blocks]
        print(
            f"Error: dataset '{target_dataset}' not found. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Apply: comment out all, then uncomment target
    for b in blocks:
        lines = _toggle_block(lines, b, activate=False)

    # Re-parse after commenting out all (indices are the same)
    # Now uncomment the target
    lines = _toggle_block(lines, target_block, activate=True)

    cell["source"] = lines
    nb["cells"][cell_idx] = cell

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Activated dataset '{target_dataset}' in {nb_path}", file=sys.stderr)


def restore_notebook(nb_path: Path) -> None:
    """Restore the notebook from its .bak backup."""
    backup = nb_path.with_suffix(".ipynb.bak")
    if not backup.exists():
        print(f"No backup found at {backup}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(backup, nb_path)
    backup.unlink()
    print(f"Restored {nb_path} from backup", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toggle active dataset in NB00.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("exploration_notebooks/00_start_here.ipynb"),
        help="Path to 00_start_here.ipynb",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--activate",
        type=str,
        help="Dataset name to activate (e.g. customer_retention_retail)",
    )
    group.add_argument(
        "--restore",
        action="store_true",
        help="Restore notebook from .bak backup",
    )
    args = parser.parse_args()

    nb_path = args.notebook.resolve()
    if not nb_path.exists():
        print(f"Error: {nb_path} not found", file=sys.stderr)
        sys.exit(1)

    if args.restore:
        restore_notebook(nb_path)
    else:
        patch_notebook(nb_path, args.activate)


if __name__ == "__main__":
    main()
