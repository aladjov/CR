#!/usr/bin/env python3
"""Capture non-chart cell outputs from executed Jupyter notebooks into a structured Markdown manifest.

The manifest is designed for diffing between two exploration runs to produce
a drift report.  Charts (Plotly, matplotlib images) are excluded; text,
HTML tables, and stream output are preserved with cell-ID and @cr: tag
annotations so the comparison tool can match cells across versions.

Usage:
    python scripts/experiments/capture_notebook_outputs.py \
        --notebooks-dir exploration_notebooks \
        --output experiments/runs/my-run/cell_outputs.md \
        [--notebooks 00,01,08]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
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

# MIME types that indicate a chart / image (excluded from capture)
_CHART_MIMES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/svg+xml",
        "application/vnd.plotly.v1+json",
    }
)

_MAX_OUTPUT_LINES = 200

# Regex for @cr: tags in code cells
_CR_CODE_RE = re.compile(
    r"^#\s*@cr:(?P<kind>code|config)\s+name='(?P<name>[^']+)'\s+id=(?P<id>\w+)"
)
# Regex for cr:doc tags in markdown cells
_CR_DOC_RE = re.compile(
    r"\[//\]:\s*#\s*\(cr:doc\s+name='(?P<name>[^']+)'\s+id=(?P<id>\w+)\)"
)


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


def _truncate(text: str, max_lines: int = _MAX_OUTPUT_LINES) -> str:
    """Truncate text to *max_lines* and append a marker if needed."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = "\n".join(lines[:max_lines])
    return f"{kept}\n[...truncated {len(lines) - max_lines} lines]"


def _is_chart_only(output: dict) -> bool:
    """Return True if the output contains only chart MIME types."""
    if output.get("output_type") == "stream":
        return False
    if output.get("output_type") == "error":
        return False
    data = output.get("data", {})
    if not data:
        return False
    return all(mime in _CHART_MIMES for mime in data)


def _join_text(fragments: list | str) -> str:
    if isinstance(fragments, list):
        return "".join(fragments)
    return fragments


# ---------------------------------------------------------------------------
# Core capture logic
# ---------------------------------------------------------------------------


def capture_notebook(nb_path: Path) -> Optional[str]:
    """Capture cell outputs from a single executed notebook.

    Returns a Markdown string, or None if the notebook has no outputs.
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    parts: list[str] = []
    current_section: Optional[str] = None
    has_any_output = False

    for cell in cells:
        tag = _parse_cr_tag(cell)
        cid = _cell_id(cell)

        # Track the current section from markdown doc tags
        if cell["cell_type"] == "markdown" and tag and tag["kind"] == "doc":
            current_section = tag["name"]
            continue

        # Only process code cells
        if cell["cell_type"] != "code":
            continue

        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        # Build cell header
        cell_name = tag["name"] if tag else f"cell_{cid}"
        cell_kind = tag["kind"] if tag else "code"
        section_ctx = f" (section: {current_section})" if current_section else ""
        header = f"## {cell_kind}: {cell_name} (id: {cid}){section_ctx}"

        cell_parts: list[str] = []

        for output in outputs:
            otype = output.get("output_type", "")

            if _is_chart_only(output):
                continue

            if otype == "stream":
                stream_name = output.get("name", "stdout")
                text = _join_text(output.get("text", ""))
                text = _truncate(text.rstrip())
                if text:
                    cell_parts.append(f"### Stream ({stream_name})\n```\n{text}\n```")

            elif otype == "error":
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                tb = _join_text(output.get("traceback", []))
                # Strip ANSI escape codes from traceback
                tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
                tb = _truncate(tb.rstrip())
                cell_parts.append(
                    f"### Error: {ename}\n```\n{evalue}\n\n{tb}\n```"
                )

            elif otype in ("execute_result", "display_data"):
                data = output.get("data", {})

                if "text/html" in data:
                    html = _join_text(data["text/html"])
                    html = _truncate(html.rstrip())
                    cell_parts.append(f"### Output (text/html)\n{html}")
                elif "text/markdown" in data:
                    md = _join_text(data["text/markdown"])
                    md = _truncate(md.rstrip())
                    if md:
                        cell_parts.append(f"### Output (text/markdown)\n{md}")
                    elif "text/plain" in data:
                        text = _join_text(data["text/plain"])
                        text = _truncate(text.rstrip())
                        if text:
                            cell_parts.append(f"### Output (text/plain)\n```\n{text}\n```")
                elif "text/plain" in data:
                    text = _join_text(data["text/plain"])
                    text = _truncate(text.rstrip())
                    if text:
                        cell_parts.append(f"### Output (text/plain)\n```\n{text}\n```")

        if cell_parts:
            has_any_output = True
            parts.append(header)
            parts.extend(cell_parts)

    if not has_any_output:
        return None
    return "\n\n".join(parts)


def capture_all_notebooks(
    notebooks_dir: Path,
    filter_stems: Optional[set[str]] = None,
) -> str:
    """Capture outputs from all exploration notebooks in order.

    Returns the full Markdown manifest including a section map header.
    """
    manifest_parts: list[str] = []
    section_map_lines: list[str] = ["<!-- Section Map -->"]
    notebooks_found: list[tuple[str, Path]] = []

    for stem in NOTEBOOKS_ORDER:
        if filter_stems and stem not in filter_stems:
            continue
        # Notebooks may have a leading `-1_` prefix in some cases
        candidates = list(notebooks_dir.glob(f"{stem}.ipynb")) + list(
            notebooks_dir.glob(f"-1_{stem}.ipynb")
        )
        if not candidates:
            continue
        nb_path = candidates[0]
        notebooks_found.append((stem, nb_path))

    # Also pick up any notebooks not in the standard order
    known_stems = {s for s, _ in notebooks_found}
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        stem = nb_path.stem
        if stem.startswith("-1_"):
            stem = stem[3:]
        if stem not in known_stems and (not filter_stems or stem in filter_stems):
            notebooks_found.append((stem, nb_path))

    for stem, nb_path in notebooks_found:
        try:
            nb = json.load(open(nb_path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            section_map_lines.append(f"<!-- {stem}: UNREADABLE -->")
            continue
        cell_count = len(nb.get("cells", []))
        output_count = sum(
            1
            for c in nb.get("cells", [])
            if c.get("outputs")
        )
        section_map_lines.append(
            f"<!-- {stem} ({cell_count} cells, {output_count} with output) -->"
        )

        captured = capture_notebook(nb_path)
        if captured:
            manifest_parts.append(f"# {stem}\n\n{captured}")
        else:
            manifest_parts.append(f"# {stem}\n\n_No captured outputs._")

    header = "\n".join(section_map_lines)
    body = "\n\n---\n\n".join(manifest_parts)
    return f"{header}\n\n{body}\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture non-chart cell outputs from executed notebooks."
    )
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        required=True,
        help="Directory containing executed .ipynb files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown file (default: stdout)",
    )
    parser.add_argument(
        "--notebooks",
        type=str,
        default=None,
        help="Comma-separated notebook stems to capture (e.g. 00,01,08)",
    )
    args = parser.parse_args()

    notebooks_dir = args.notebooks_dir.resolve()
    if not notebooks_dir.is_dir():
        print(f"Error: {notebooks_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    filter_stems: Optional[set[str]] = None
    if args.notebooks:
        raw = [s.strip() for s in args.notebooks.split(",")]
        # Expand short names like "08" to match "08_baseline_experiments"
        filter_stems = set()
        for r in raw:
            matched = [s for s in NOTEBOOKS_ORDER if s.startswith(r)]
            if matched:
                filter_stems.update(matched)
            else:
                filter_stems.add(r)

    manifest = capture_all_notebooks(notebooks_dir, filter_stems)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(manifest, encoding="utf-8")
        print(f"Manifest written to {args.output}", file=sys.stderr)
    else:
        print(manifest)


if __name__ == "__main__":
    main()
