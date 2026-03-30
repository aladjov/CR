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

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    "plotly-graph-div", "plotly.newplot",
    "data:image/png", "data:image/jpeg", "data:image/svg",
    "<svg ", "<svg>",
    "bokeh-plot-", "bk-root",
    "vega-embed",
    "class=\"output-image\"",
)

_CR_TAG_RE = re.compile(
    r"^#\s*@cr:(?P<kind>code|config|user_code|code_system)"
    r"\s+name='(?P<name>[^']+)'\s+id=(?P<id>\w+)"
)

_COMMAND_SPLIT_RE = re.compile(
    r'(?=<div[^>]*(?:id="command-[^"]*"|class="[^"]*\bcommand\b)[^>]*>)'
)


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


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _strip_tags(html_str: str) -> str:
    """Strip HTML tags, decode entities, return plain text."""
    text = re.sub(r"<[^>]+>", "", html_str)
    for entity, char in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&quot;", '"'), ("&#39;", "'")):
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
        block, re.DOTALL,
    )
    content = result_match.group(1) if result_match else block
    if _is_chart_block(content):
        return []

    outputs: list[dict] = []

    # Stream / ansiout text
    for m in re.finditer(
        r'class="[^"]*(?:ansiout|stream|text[_-]?output)[^"]*"[^>]*>(.*?)</(?:div|pre)>',
        content, re.DOTALL,
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
        content, re.DOTALL,
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
        cells.append(ParsedCell(index=i, cell_type=cell_type, source=source,
                                cr_tag=cr_tag, outputs=outputs))
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


def _fetch_task_output(w, task_run_id: int, task_name: str) -> tuple[str, str]:
    """Fetch and parse one task's notebook output.

    Returns ``(task_name, markdown_manifest)``.
    """
    try:
        export = w.jobs.export_run(run_id=task_run_id)
    except Exception:
        # Fallback: try the REST API directly
        try:
            resp = w.api_client.do("GET", "/api/2.0/jobs/runs/export",
                                   query={"run_id": task_run_id, "views_to_export": "ALL"})
            export_views = resp.get("views", [])
            html_content = export_views[0]["content"] if export_views else ""
            cells = _parse_exported_html(html_content)
            return task_name, _format_notebook_manifest(task_name, cells)
        except Exception as e:
            return task_name, f"# {task_name}\n\n_Failed to export: {e}_\n"

    if not export.views:
        return task_name, f"# {task_name}\n\n_No views returned._\n"

    html_content = export.views[0].content or ""
    cells = _parse_exported_html(html_content)
    return task_name, _format_notebook_manifest(task_name, cells)


def capture_job_run(
    run_id: int,
    output_path: Optional[str] = None,
    max_workers: int = 8,
    notebook_filter: Optional[set[str]] = None,
) -> str:
    """Capture all cell outputs from a Databricks job run.

    Args:
        run_id: Databricks job run ID (parent run for multi-task jobs).
        output_path: Optional path to write the manifest. Supports workspace
            paths (``/Workspace/...``) and DBFS paths (``/dbfs/...``).
        max_workers: Maximum parallel API calls for multi-task jobs.
        notebook_filter: Optional set of task keys to include. If ``None``,
            all tasks are captured.

    Returns:
        Markdown manifest string in the same format as
        ``capture_notebook_outputs.py``.
    """
    if WorkspaceClient is None:
        raise ImportError(
            "databricks-sdk is required. Install with: pip install databricks-sdk"
        )

    w = WorkspaceClient()
    run = w.jobs.get_run(run_id)

    # Build task list: multi-task or single-task
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

    # Parallel fetch all task outputs
    manifests: dict[str, str] = {}
    worker_count = min(max_workers, len(tasks)) if tasks else 1
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_fetch_task_output, w, tid, tname): tname
            for tname, tid in tasks
        }
        for future in as_completed(futures):
            tname, manifest = future.result()
            manifests[tname] = manifest

    # Assemble in task order (preserves dependency ordering)
    task_order = [name for name, _ in tasks]
    all_parts = [manifests[name] for name in task_order if name in manifests]

    header = "<!-- Section Map -->\n" + "\n".join(
        line for part in all_parts for line in part.split("\n")
        if line.startswith("<!-- ")
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

    return result
