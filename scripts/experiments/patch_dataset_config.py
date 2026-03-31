#!/usr/bin/env python3
"""Toggle the active dataset in 00_start_here.ipynb.

Finds the ``@cr:config name='dataset_paths'`` cell and activates the
requested dataset block while commenting out all others.

Usage:
    # Activate retail dataset
    python scripts/experiments/patch_dataset_config.py \
        --notebook exploration_notebooks/00_start_here.ipynb \
        --activate customer_retention_retail

    # With absolute fixture paths (for running notebooks from a different dir)
    python scripts/experiments/patch_dataset_config.py \
        --notebook /path/to/copy/00_start_here.ipynb \
        --activate customer_emails \
        --fixture-root /abs/path/to/tests/fixtures
"""

from __future__ import annotations

import argparse
import json
import re
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
    # Fallback: find the first code cell containing a `datasets = {` block
    # (older notebooks lack the @cr:config tag)
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        if re.search(r"^#?\s*datasets\s*=\s*\{", text, re.MULTILINE):
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


def _rewrite_relative_paths(lines: list[str], fixture_root: Path | None) -> list[str]:
    """Replace ``../tests/fixtures/`` with an absolute *fixture_root* prefix."""
    if fixture_root is None:
        return lines
    root_str = str(fixture_root).rstrip("/")
    return [ln.replace("../tests/fixtures/", f"{root_str}/") for ln in lines]


def patch_notebook(
    nb_path: Path, target_dataset: str, *, fixture_root: Path | None = None
) -> None:
    """Activate *target_dataset* in the notebook and comment out others."""
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

    # When multiple blocks share the same dataset name (e.g. a single-line
    # Databricks /Volumes/ path and a multi-line local fixture path), prefer
    # the block whose values use relative paths (start with . or ..).
    candidates = [b for b in blocks if target_dataset in b["name"]]
    target_block = None
    if len(candidates) == 1:
        target_block = candidates[0]
    elif candidates:
        def _block_paths(b):
            text = "".join(lines[b["header_idx"]:b["end_idx"] + 1])
            return re.findall(r':\s*"([^"]+)"', text)
        local = [b for b in candidates
                 if any(p.startswith(".") for p in _block_paths(b))]
        target_block = local[0] if local else candidates[-1]

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
    lines = _rewrite_relative_paths(lines, fixture_root)

    cell["source"] = lines
    nb["cells"][cell_idx] = cell

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Activated dataset '{target_dataset}' in {nb_path}", file=sys.stderr)


def patch_data_path(
    nb_path: Path, target_dataset: str, *, fixture_root: Path | None = None
) -> None:
    """Activate *target_dataset* DATA_PATH in older NB01 notebooks.

    Older notebooks use a flat ``DATA_PATH = "..."`` line instead of the
    ``datasets = {}`` block.  This function comments out all DATA_PATH lines
    and uncomments the one whose path contains *target_dataset*.
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    patched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src)
        if "DATA_PATH" not in text:
            continue
        # Only the config cell (contains DATA_PATH and TARGET_COLUMN)
        if "TARGET_COLUMN" not in text and "ENTITY_COLUMN" not in text:
            continue

        lines = list(src)
        activated = False
        for idx, line in enumerate(lines):
            # Match DATA_PATH = "..." or #DATA_PATH = "..."
            m = re.match(r"^(\s*)#?\s*(DATA_PATH\s*=\s*\"[^\"]+\")", line)
            if not m:
                continue
            indent, assignment = m.group(1), m.group(2)
            if target_dataset in line:
                lines[idx] = f"{indent}{assignment}\n"
                activated = True
            else:
                lines[idx] = f"{indent}#{assignment}\n"

        if activated:
            cell["source"] = _rewrite_relative_paths(lines, fixture_root)
            patched = True
            break

    if not patched:
        print(
            f"Warning: no DATA_PATH matching '{target_dataset}' in {nb_path}",
            file=sys.stderr,
        )
        return

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Activated DATA_PATH for '{target_dataset}' in {nb_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toggle active dataset in NB00.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("exploration_notebooks/00_start_here.ipynb"),
        help="Path to 00_start_here.ipynb",
    )
    parser.add_argument(
        "--activate",
        type=str,
        required=True,
        help="Dataset name to activate (e.g. customer_retention_retail)",
    )
    parser.add_argument(
        "--also-patch",
        type=Path,
        action="append",
        default=[],
        help="Additional notebook whose DATA_PATH line to patch (repeatable)",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=None,
        help="Rewrite ../tests/fixtures/ to this absolute path (for running from a different dir)",
    )
    args = parser.parse_args()

    fixture_root = args.fixture_root.resolve() if args.fixture_root else None
    nb_path = args.notebook.resolve()
    if not nb_path.exists():
        print(f"Error: {nb_path} not found", file=sys.stderr)
        sys.exit(1)

    patch_notebook(nb_path, args.activate, fixture_root=fixture_root)
    for extra in args.also_patch:
        p = extra.resolve()
        if p.exists():
            patch_data_path(p, args.activate, fixture_root=fixture_root)


if __name__ == "__main__":
    main()
