"""Capture cell outputs from a Databricks job run into structured Markdown.

Produces the same format as ``scripts/experiments/capture_notebook_outputs.py``
so that outputs from Databricks job runs can be compared with local runs using
``scripts/experiments/compare_exploration_runs.py``.

Usage from a Databricks notebook cell::

    from customer_retention.integrations.databricks_job_capture import capture_job_run

    manifest = capture_job_run(run_id=123456789)
    print(manifest)

    # Or save to workspace / DBFS:
    capture_job_run(run_id=123456789, output_path="/Workspace/Users/me/cell_outputs.md")

Charts (Plotly, matplotlib images, SVG, Bokeh) are excluded; text, HTML
tables, and stream output are preserved with ``@cr:`` tag annotations when
present.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy import — WorkspaceClient is only available on Databricks
# ---------------------------------------------------------------------------

try:
    from databricks.sdk import WorkspaceClient
except ImportError:  # pragma: no cover
    WorkspaceClient = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_OUTPUT_LINES = 200

_CHART_INDICATORS = (
    "plotly-graph-div",
    "plotly.newplot",
    "data:image/png",
    "data:image/jpeg",
    "data:image/svg",
    "<svg ",
    "<svg>",
    "bokeh-plot-",
    "bk-root",
    "vega-embed",
    'class="output-image"',
)

_CR_TAG_RE = re.compile(
    r"^#\s*@cr:(?P<kind>code|config|user_code|code_system)"
    r"\s+name='(?P<name>[^']+)'\s+id=(?P<id>\w+)"
)

_COMMAND_SPLIT_RE = re.compile(r'(?=<div[^>]*(?:id="command-[^"]*"|class="[^"]*\bcommand\b)[^>]*>)')
_RUNTIME_ATTR_RE = re.compile(r'data-commandruntime="(\d+)"')
_TIMING_TEXT_RE = re.compile(r"(?:Command took|Took)\s+([\d.]+)\s+seconds", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ParsedCell:
    index: int
    cell_type: str  # "code" or "markdown"
    source: str
    cr_tag: Optional[dict] = None
    outputs: list = field(default_factory=list)
    elapsed_sec: Optional[float] = None


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _strip_tags(html_str: str) -> str:
    """Strip HTML tags, decode entities, return plain text."""
    text = re.sub(r"<[^>]+>", "", html_str)
    for entity, char in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
        text = text.replace(entity, char)
    return text.strip()


def _truncate(text: str, max_lines: int = _MAX_OUTPUT_LINES) -> str:
    """Truncate to *max_lines*, append marker if needed."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = "\n".join(lines[:max_lines])
    return f"{kept}\n[...truncated {len(lines) - max_lines} lines]"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_cr_tag_from_source(source: str) -> Optional[dict]:
    """Extract ``@cr:`` tag from the first line of cell source."""
    if not source:
        return None
    first_line = source.split("\n")[0]
    m = _CR_TAG_RE.match(first_line)
    if m:
        return {"kind": m.group("kind"), "name": m.group("name"), "id": m.group("id")}
    return None


def _is_chart_block(block: str) -> bool:
    """Return True if *block* contains primarily chart/image content."""
    lower = block.lower()
    return any(ind.lower() in lower for ind in _CHART_INDICATORS)


def _extract_source_from_block(block: str) -> tuple[str, str]:
    """Extract source code and cell type from a command block.

    Returns ``(source_text, cell_type)`` where *cell_type* is ``"code"`` or
    ``"markdown"``.
    """
    if re.search(r'class="[^"]*\bmarkdown\b', block):
        pre = re.search(r"<h[1-6][^>]*>(.*?)</h[1-6]>", block, re.DOTALL)
        return (_strip_tags(pre.group(0)) if pre else "", "markdown")
    pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", block, re.DOTALL)
    if pre_match:
        return (_strip_tags(pre_match.group(1)), "code")
    return ("", "code")


def _extract_outputs_from_block(block: str) -> list[dict]:
    """Extract non-chart outputs from a command block."""
    result_match = re.search(
        r'class="[^"]*(?:commandResult|command[_-]?results?)[^"]*"[^>]*>(.*)',
        block,
        re.DOTALL,
    )
    content = result_match.group(1) if result_match else block
    if _is_chart_block(content):
        return []

    outputs: list[dict] = []

    # Stream / ansiout text
    for m in re.finditer(
        r'class="[^"]*(?:ansiout|stream|text[_-]?output)[^"]*"[^>]*>(.*?)</(?:div|pre)>',
        content,
        re.DOTALL,
    ):
        text = _strip_tags(m.group(1)).strip()
        if text:
            outputs.append({"type": "stream", "content": _truncate(text)})

    # HTML tables
    for m in re.finditer(r"(<table[^>]*>.*?</table>)", content, re.DOTALL):
        outputs.append({"type": "html", "content": _truncate(m.group(1))})

    # Error output
    for m in re.finditer(
        r'class="[^"]*(?:error|traceback)[^"]*"[^>]*>(.*?)</(?:div|pre)>',
        content,
        re.DOTALL,
    ):
        text = _strip_tags(m.group(1)).strip()
        if text:
            outputs.append({"type": "error", "content": _truncate(text)})

    # Fallback: plain text results not already captured
    if not outputs:
        cleaned = re.sub(r"<table[^>]*>.*?</table>", "", content, flags=re.DOTALL)
        cleaned = re.sub(r"<img[^>]*>", "", cleaned)
        cleaned = re.sub(r"<svg[^>]*>.*?</svg>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<script[^>]*>.*?</script>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL)
        text = _strip_tags(cleaned).strip()
        if text:
            outputs.append({"type": "text", "content": _truncate(text)})

    return outputs


def _extract_timing_from_block(block: str) -> Optional[float]:
    """Extract cell execution time in seconds from a command block.

    Tries ``data-commandruntime`` attribute (milliseconds), then falls back
    to text patterns like "Command took X.XX seconds".
    """
    m = _RUNTIME_ATTR_RE.search(block)
    if m:
        return int(m.group(1)) / 1000.0
    m = _TIMING_TEXT_RE.search(block)
    if m:
        return float(m.group(1))
    return None


def _parse_exported_html(html_content: str) -> list[ParsedCell]:
    """Parse Databricks exported HTML notebook into structured cells."""
    cleaned = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL)

    blocks = _COMMAND_SPLIT_RE.split(cleaned)
    # First element is the preamble (before any command div)
    blocks = [b for b in blocks if re.search(r'(?:id="command-|class="[^"]*\bcommand\b)', b)]

    cells: list[ParsedCell] = []
    for i, block in enumerate(blocks):
        source, cell_type = _extract_source_from_block(block)
        cr_tag = _parse_cr_tag_from_source(source) if cell_type == "code" else None
        outputs = _extract_outputs_from_block(block) if cell_type == "code" else []
        elapsed = _extract_timing_from_block(block) if cell_type == "code" else None
        cells.append(
            ParsedCell(index=i, cell_type=cell_type, source=source, cr_tag=cr_tag, outputs=outputs, elapsed_sec=elapsed)
        )
    return cells


# ---------------------------------------------------------------------------
# Formatting (matches capture_notebook_outputs.py format)
# ---------------------------------------------------------------------------


def _format_notebook_manifest(notebook_name: str, cells: list[ParsedCell]) -> str:
    """Format parsed cells into Markdown matching ``capture_notebook_outputs.py``."""
    cell_count = len(cells)
    output_count = sum(1 for c in cells if c.outputs)
    section_map = f"<!-- {notebook_name} ({cell_count} cells, {output_count} with output) -->"

    parts: list[str] = []
    has_any_output = False

    for cell in cells:
        if cell.cell_type != "code" or not cell.outputs:
            continue

        has_any_output = True
        tag = cell.cr_tag
        cell_name = tag["name"] if tag else f"cell_{cell.index}"
        cell_kind = tag["kind"] if tag else "code"
        cell_id = tag["id"] if tag else f"idx{cell.index}"
        header = f"## {cell_kind}: {cell_name} (id: {cell_id})"

        cell_parts: list[str] = [header]
        for output in cell.outputs:
            otype = output["type"]
            content = output["content"]
            if otype == "stream":
                cell_parts.append(f"### Stream (stdout)\n```\n{content}\n```")
            elif otype == "html":
                cell_parts.append(f"### Output (text/html)\n{content}")
            elif otype == "error":
                cell_parts.append(f"### Error\n```\n{content}\n```")
            elif otype == "text":
                cell_parts.append(f"### Output (text/plain)\n```\n{content}\n```")
        parts.append("\n\n".join(cell_parts))

    if not has_any_output:
        body = "_No captured outputs._"
    else:
        body = "\n\n".join(parts)
    return f"{section_map}\n\n# {notebook_name}\n\n{body}\n"


# ---------------------------------------------------------------------------
# Databricks API interaction
# ---------------------------------------------------------------------------


@dataclass
class CellProfileEntry:
    cell_name: str
    cell_id: str
    elapsed_sec: float
    status: str = "completed"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    spark_jobs: Optional[int] = None
    peak_memory_mb: Optional[float] = None


@dataclass
class NotebookProfile:
    total_elapsed: float
    cells: list[CellProfileEntry] = field(default_factory=list)


def extract_cell_profiles_from_parsed_cells(notebook_name: str, cells: list[ParsedCell]) -> list[CellProfileEntry]:
    entries: list[CellProfileEntry] = []
    for cell in cells:
        if cell.cell_type != "code":
            continue
        tag = cell.cr_tag
        name = tag["name"] if tag else f"cell_{cell.index}"
        cid = tag["id"] if tag else f"idx{cell.index}"
        elapsed = cell.elapsed_sec or 0.0
        entries.append(CellProfileEntry(cell_name=name, cell_id=cid, elapsed_sec=elapsed))
    return entries


def _build_profiles_json(task_profiles: dict[str, list[CellProfileEntry]]) -> dict:
    notebooks = {}
    for nb_name, entries in task_profiles.items():
        total = sum(e.elapsed_sec for e in entries)
        notebooks[nb_name] = {
            "total_elapsed": round(total, 3),
            "cells": [
                {
                    "cell_name": e.cell_name,
                    "cell_id": e.cell_id,
                    "elapsed_sec": round(e.elapsed_sec, 6),
                    "status": e.status,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "spark_jobs": e.spark_jobs,
                    "peak_memory_mb": e.peak_memory_mb,
                }
                for e in entries
            ],
        }
    return {
        "version": 1,
        "environment": "databricks",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notebooks": notebooks,
    }


def _fetch_task_output(w, task_run_id: int, task_name: str) -> tuple[str, str, list[ParsedCell]]:
    """Fetch and parse one task's notebook output.

    Returns ``(task_name, markdown_manifest, parsed_cells)``.
    """

    def _result(name: str, cells: list[ParsedCell]) -> tuple[str, str, list[ParsedCell]]:
        return name, _format_notebook_manifest(name, cells), cells

    try:
        export = w.jobs.export_run(run_id=task_run_id)
    except Exception:
        try:
            resp = w.api_client.do(
                "GET", "/api/2.0/jobs/runs/export", query={"run_id": task_run_id, "views_to_export": "ALL"}
            )
            export_views = resp.get("views", [])
            html_content = export_views[0]["content"] if export_views else ""
            return _result(task_name, _parse_exported_html(html_content))
        except Exception as e:
            return task_name, f"# {task_name}\n\n_Failed to export: {e}_\n", []

    if not export.views:
        return task_name, f"# {task_name}\n\n_No views returned._\n", []

    html_content = export.views[0].content or ""
    return _result(task_name, _parse_exported_html(html_content))


def capture_job_run(
    run_id: int,
    output_path: Optional[str] = None,
    profiles_path: Optional[str] = None,
    max_workers: int = 8,
    notebook_filter: Optional[set[str]] = None,
) -> str:
    """Capture all cell outputs (and optionally profiles) from a Databricks job run.

    Args:
        run_id: Databricks job run ID (parent run for multi-task jobs).
        output_path: Optional path to write the manifest.
        profiles_path: Optional path to write ``cell_profiles.json``.
        max_workers: Maximum parallel API calls for multi-task jobs.
        notebook_filter: Optional set of task keys to include.

    Returns:
        Markdown manifest string.
    """
    if WorkspaceClient is None:
        raise ImportError("databricks-sdk is required. Install with: pip install databricks-sdk")

    w = WorkspaceClient()
    run = w.jobs.get_run(run_id)

    tasks: list[tuple[str, int]] = []
    if run.tasks:
        for task in run.tasks:
            name = task.task_key or f"task_{task.run_id}"
            if notebook_filter and name not in notebook_filter:
                continue
            tasks.append((name, task.run_id))
    else:
        name = "notebook"
        if hasattr(run, "notebook_task") and run.notebook_task:
            nb_path = run.notebook_task.notebook_path or ""
            name = Path(nb_path).stem or "notebook"
        tasks.append((name, run.run_id))

    manifests: dict[str, str] = {}
    task_cells: dict[str, list[ParsedCell]] = {}
    worker_count = min(max_workers, len(tasks)) if tasks else 1
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_fetch_task_output, w, tid, tname): tname for tname, tid in tasks}
        for future in as_completed(futures):
            tname, manifest, cells = future.result()
            manifests[tname] = manifest
            task_cells[tname] = cells

    # Assemble in task order (preserves dependency ordering)
    task_order = [name for name, _ in tasks]
    all_parts = [manifests[name] for name in task_order if name in manifests]

    header = "<!-- Section Map -->\n" + "\n".join(
        line for part in all_parts for line in part.split("\n") if line.startswith("<!-- ")
    )
    body_parts = []
    for part in all_parts:
        lines = part.split("\n")
        # Skip the section map line (first line starting with <!--)
        body_lines = [ln for ln in lines if not ln.startswith("<!-- ")]
        body_parts.append("\n".join(body_lines).strip())
    body = "\n\n---\n\n".join(body_parts)
    result = f"{header}\n\n{body}\n"

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(result, encoding="utf-8")

    if profiles_path:
        task_profiles = {
            tname: extract_cell_profiles_from_parsed_cells(tname, task_cells.get(tname, []))
            for tname in task_order
            if tname in task_cells
        }
        profiles_json = _build_profiles_json(task_profiles)
        # Merge runner-written Spark job counts if the run was instrumented
        profiles_json = _merge_runner_profiles(w, run, profiles_json)
        p = Path(profiles_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(profiles_json, indent=2), encoding="utf-8")

    return result


def _merge_runner_profiles(w, run, profiles_json: dict) -> dict:
    """Merge Spark job counts from runner-written cell_profiles.json (if present)."""
    try:
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        exp_dir = None
        if run.tasks:
            for task in run.tasks:
                if hasattr(task, "notebook_task") and task.notebook_task:
                    params = getattr(task.notebook_task, "base_parameters", {}) or {}
                    exp_dir = params.get("experiments_dir")
                    run_id = params.get("run_id")
                    if exp_dir and run_id:
                        break
        if not exp_dir:
            return profiles_json

        ns = RunNamespace(root=Path(exp_dir), run_id=run_id)
        if not ns.cell_profiles_path.exists():
            return profiles_json

        runner_data = json.loads(ns.cell_profiles_path.read_text(encoding="utf-8"))
        runner_nbs = runner_data.get("notebooks", {})
        for nb_name, nb_data in runner_nbs.items():
            runner_cells = nb_data.get("cells", [])
            if not runner_cells:
                continue
            runner_spark_jobs = runner_cells[0].get("spark_jobs")
            if runner_spark_jobs is None:
                continue
            # Distribute per-notebook Spark jobs to the HTML-extracted per-cell profiles
            target = profiles_json.get("notebooks", {}).get(nb_name)
            if target and target.get("cells"):
                target["cells"][0]["spark_jobs"] = runner_spark_jobs
    except Exception:
        pass  # Runner profiles are best-effort enrichment
    return profiles_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Capture cell outputs from a Databricks job run.")
    parser.add_argument("--run-id", type=int, required=True, help="Databricks job run ID")
    parser.add_argument("--output", required=True, help="Output path for cell_outputs.md")
    parser.add_argument("--profiles", default=None, help="Output path for cell_profiles.json")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    result = capture_job_run(
        run_id=args.run_id,
        output_path=args.output,
        profiles_path=args.profiles,
        max_workers=args.max_workers,
    )
    print(f"Captured {result.count(chr(10))} lines -> {args.output}", file=sys.stderr)
    if args.profiles:
        print(f"Profiles -> {args.profiles}", file=sys.stderr)


if __name__ == "__main__":
    main()
