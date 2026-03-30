"""Tests for Databricks job run cell output capture utility.

Tests HTML parsing, @cr: tag extraction, chart skipping, and Markdown
formatting. All Databricks API calls are mocked — no real cluster needed.
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

from customer_retention.integrations.databricks_job_capture import (
    ParsedCell,
    _extract_outputs_from_block,
    _format_notebook_manifest,
    _is_chart_block,
    _parse_cr_tag_from_source,
    _parse_exported_html,
    _strip_tags,
    _truncate,
    capture_job_run,
)

# ---------------------------------------------------------------------------
# Fixtures: mock Databricks HTML fragments
# ---------------------------------------------------------------------------

SIMPLE_CODE_CELL_HTML = textwrap.dedent("""\
<div id="command-1234" class="command">
  <div class="commandInput">
    <div class="code"><pre># @cr:code name='load_data' id=abcd1234
df = spark.read.table("my_table")
print(df.count())</pre></div>
  </div>
  <div class="commandResult">
    <div class="ansiout"><pre>42000</pre></div>
  </div>
</div>
""")

MARKDOWN_CELL_HTML = textwrap.dedent("""\
<div id="command-5678" class="command">
  <div class="commandInput">
    <div class="markdown"><h2>Section 1.2 — Data Loading</h2></div>
  </div>
</div>
""")

HTML_TABLE_OUTPUT_HTML = textwrap.dedent("""\
<div id="command-9012" class="command">
  <div class="commandInput">
    <div class="code"><pre># @cr:config name='show_stats' id=ef567890
display(df.describe())</pre></div>
  </div>
  <div class="commandResult">
    <div class="display-html-output">
      <table><tr><th>col</th><th>count</th></tr><tr><td>A</td><td>100</td></tr></table>
    </div>
  </div>
</div>
""")

CHART_CELL_HTML = textwrap.dedent("""\
<div id="command-3456" class="command">
  <div class="commandInput">
    <div class="code"><pre>fig.show()</pre></div>
  </div>
  <div class="commandResult">
    <div class="plotly-graph-div" id="plotly-123">
      <script>Plotly.newPlot(...)</script>
    </div>
  </div>
</div>
""")

IMAGE_CHART_HTML = textwrap.dedent("""\
<div id="command-7890" class="command">
  <div class="commandInput">
    <div class="code"><pre>plt.show()</pre></div>
  </div>
  <div class="commandResult">
    <div class="output-image">
      <img src="data:image/png;base64,iVBOR..." />
    </div>
  </div>
</div>
""")

ERROR_CELL_HTML = textwrap.dedent("""\
<div id="command-error1" class="command">
  <div class="commandInput">
    <div class="code"><pre>1/0</pre></div>
  </div>
  <div class="commandResult">
    <div class="errorOutput">
      <pre class="error-message">ZeroDivisionError: division by zero</pre>
    </div>
  </div>
</div>
""")

EMPTY_OUTPUT_CELL_HTML = textwrap.dedent("""\
<div id="command-noop" class="command">
  <div class="commandInput">
    <div class="code"><pre>x = 1 + 1</pre></div>
  </div>
  <div class="commandResult"></div>
</div>
""")


def _full_html(*blocks: str) -> str:
    """Wrap HTML blocks in a minimal Databricks notebook page."""
    body = "\n".join(blocks)
    return f"<html><body><div id='notebook'>{body}</div></body></html>"


# ---------------------------------------------------------------------------
# Tests: _strip_tags
# ---------------------------------------------------------------------------


class TestStripTags:
    def test_removes_simple_tags(self):
        assert _strip_tags("<b>bold</b>") == "bold"

    def test_preserves_plain_text(self):
        assert _strip_tags("no tags here") == "no tags here"

    def test_handles_nested_tags(self):
        assert _strip_tags("<div><span>text</span></div>") == "text"

    def test_handles_empty_string(self):
        assert _strip_tags("") == ""

    def test_handles_self_closing_tags(self):
        assert _strip_tags("before<br/>after") == "beforeafter"


# ---------------------------------------------------------------------------
# Tests: _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("line1\nline2", 10) == "line1\nline2"

    def test_long_text_truncated(self):
        text = "\n".join(f"line{i}" for i in range(50))
        result = _truncate(text, 10)
        assert "[...truncated 40 lines]" in result
        assert result.count("\n") == 10  # 10 kept lines + truncation marker

    def test_exact_limit_not_truncated(self):
        text = "\n".join(f"line{i}" for i in range(10))
        assert _truncate(text, 10) == text


# ---------------------------------------------------------------------------
# Tests: _parse_cr_tag_from_source
# ---------------------------------------------------------------------------


class TestParseCrTag:
    def test_code_tag(self):
        source = "# @cr:code name='load_data' id=abcd1234\ndf = spark.read.table('t')"
        tag = _parse_cr_tag_from_source(source)
        assert tag == {"kind": "code", "name": "load_data", "id": "abcd1234"}

    def test_config_tag(self):
        source = "# @cr:config name='settings' id=12345678"
        tag = _parse_cr_tag_from_source(source)
        assert tag == {"kind": "config", "name": "settings", "id": "12345678"}

    def test_user_code_tag(self):
        source = "# @cr:user_code name='custom' id=aabbccdd"
        tag = _parse_cr_tag_from_source(source)
        assert tag == {"kind": "user_code", "name": "custom", "id": "aabbccdd"}

    def test_no_tag(self):
        source = "x = 1 + 1"
        assert _parse_cr_tag_from_source(source) is None

    def test_empty_source(self):
        assert _parse_cr_tag_from_source("") is None


# ---------------------------------------------------------------------------
# Tests: _is_chart_block
# ---------------------------------------------------------------------------


class TestIsChartBlock:
    def test_plotly_detected(self):
        assert _is_chart_block('<div class="plotly-graph-div">...</div>')

    def test_image_detected(self):
        assert _is_chart_block('<img src="data:image/png;base64,abc"/>')

    def test_svg_detected(self):
        assert _is_chart_block('<svg width="100"><rect/></svg>')

    def test_text_not_chart(self):
        assert not _is_chart_block('<pre>42000</pre>')

    def test_table_not_chart(self):
        assert not _is_chart_block('<table><tr><td>val</td></tr></table>')

    def test_bokeh_detected(self):
        assert _is_chart_block('<div class="bk-root" id="bokeh-plot-123">...</div>')

    def test_matplotlib_inline_detected(self):
        assert _is_chart_block('<img src="data:image/png;base64,iVBOR..." />')


# ---------------------------------------------------------------------------
# Tests: _extract_outputs_from_block
# ---------------------------------------------------------------------------


class TestExtractOutputs:
    def test_stream_output(self):
        outputs = _extract_outputs_from_block(SIMPLE_CODE_CELL_HTML)
        assert len(outputs) >= 1
        stream = [o for o in outputs if o["type"] == "stream"]
        assert len(stream) == 1
        assert "42000" in stream[0]["content"]

    def test_html_table_output(self):
        outputs = _extract_outputs_from_block(HTML_TABLE_OUTPUT_HTML)
        html_outputs = [o for o in outputs if o["type"] == "html"]
        assert len(html_outputs) == 1
        assert "<table>" in html_outputs[0]["content"]

    def test_chart_skipped(self):
        outputs = _extract_outputs_from_block(CHART_CELL_HTML)
        assert len(outputs) == 0

    def test_image_skipped(self):
        outputs = _extract_outputs_from_block(IMAGE_CHART_HTML)
        assert len(outputs) == 0

    def test_error_captured(self):
        outputs = _extract_outputs_from_block(ERROR_CELL_HTML)
        errors = [o for o in outputs if o["type"] == "error"]
        assert len(errors) == 1
        assert "ZeroDivisionError" in errors[0]["content"]

    def test_empty_output(self):
        outputs = _extract_outputs_from_block(EMPTY_OUTPUT_CELL_HTML)
        assert len(outputs) == 0


# ---------------------------------------------------------------------------
# Tests: _parse_exported_html
# ---------------------------------------------------------------------------


class TestParseExportedHtml:
    def test_multiple_cells(self):
        html = _full_html(
            SIMPLE_CODE_CELL_HTML,
            MARKDOWN_CELL_HTML,
            HTML_TABLE_OUTPUT_HTML,
        )
        cells = _parse_exported_html(html)
        assert len(cells) == 3
        assert cells[0].cell_type == "code"
        assert cells[1].cell_type == "markdown"
        assert cells[2].cell_type == "code"

    def test_cr_tags_extracted(self):
        html = _full_html(SIMPLE_CODE_CELL_HTML)
        cells = _parse_exported_html(html)
        assert cells[0].cr_tag is not None
        assert cells[0].cr_tag["name"] == "load_data"
        assert cells[0].cr_tag["id"] == "abcd1234"

    def test_charts_have_no_outputs(self):
        html = _full_html(CHART_CELL_HTML, IMAGE_CHART_HTML)
        cells = _parse_exported_html(html)
        for cell in cells:
            assert len(cell.outputs) == 0

    def test_empty_html(self):
        cells = _parse_exported_html("<html><body></body></html>")
        assert cells == []

    def test_source_extracted(self):
        html = _full_html(SIMPLE_CODE_CELL_HTML)
        cells = _parse_exported_html(html)
        assert "spark.read.table" in cells[0].source


# ---------------------------------------------------------------------------
# Tests: _format_notebook_manifest
# ---------------------------------------------------------------------------


class TestFormatNotebookManifest:
    def test_basic_format(self):
        cells = [
            ParsedCell(
                index=0, cell_type="code",
                source="# @cr:code name='load' id=aabb1122\ndf = load()",
                cr_tag={"kind": "code", "name": "load", "id": "aabb1122"},
                outputs=[{"type": "stream", "content": "Loaded 100 rows"}],
            ),
        ]
        result = _format_notebook_manifest("00_start_here", cells)
        assert "# 00_start_here" in result
        assert "## code: load (id: aabb1122)" in result
        assert "### Stream (stdout)" in result
        assert "Loaded 100 rows" in result

    def test_html_output_format(self):
        cells = [
            ParsedCell(
                index=0, cell_type="code", source="display(df)",
                cr_tag=None,
                outputs=[{"type": "html", "content": "<table><tr><td>A</td></tr></table>"}],
            ),
        ]
        result = _format_notebook_manifest("01_discovery", cells)
        assert "### Output (text/html)" in result
        assert "<table>" in result

    def test_error_format(self):
        cells = [
            ParsedCell(
                index=0, cell_type="code", source="1/0",
                cr_tag=None,
                outputs=[{"type": "error", "content": "ZeroDivisionError"}],
            ),
        ]
        result = _format_notebook_manifest("08_experiments", cells)
        assert "### Error" in result
        assert "ZeroDivisionError" in result

    def test_no_output_cells_skipped(self):
        cells = [
            ParsedCell(index=0, cell_type="code", source="x = 1", outputs=[]),
            ParsedCell(index=1, cell_type="markdown", source="## Title", outputs=[]),
        ]
        result = _format_notebook_manifest("00_start_here", cells)
        assert "# 00_start_here" in result
        assert "_No captured outputs._" in result

    def test_section_map_header(self):
        cells = [
            ParsedCell(
                index=0, cell_type="code", source="print(1)",
                cr_tag=None,
                outputs=[{"type": "stream", "content": "1"}],
            ),
        ]
        result = _format_notebook_manifest("00_start_here", cells)
        assert "<!-- 00_start_here" in result


# ---------------------------------------------------------------------------
# Tests: capture_job_run (mocked API)
# ---------------------------------------------------------------------------


class TestCaptureJobRun:
    def _mock_workspace_client(self, tasks=None, html_content=""):
        """Build a mock WorkspaceClient with configurable tasks and HTML."""
        w = MagicMock()
        run = MagicMock()
        if tasks is None:
            run.tasks = None
            run.run_id = 100
        else:
            run.tasks = []
            for task_key, task_run_id in tasks:
                t = MagicMock()
                t.task_key = task_key
                t.run_id = task_run_id
                t.state = MagicMock()
                t.state.result_state = MagicMock()
                t.state.result_state.value = "SUCCESS"
                nb_task = MagicMock()
                nb_task.notebook_path = f"/notebooks/{task_key}"
                t.notebook_task = nb_task
                run.tasks.append(t)
        w.jobs.get_run.return_value = run

        export = MagicMock()
        view = MagicMock()
        view.content = html_content
        view.name = "notebook"
        view.type = "NOTEBOOK"
        export.views = [view]
        w.jobs.export_run.return_value = export
        return w

    @patch("customer_retention.integrations.databricks_job_capture.WorkspaceClient")
    def test_single_task_job(self, mock_ws_cls):
        html = _full_html(SIMPLE_CODE_CELL_HTML)
        mock_ws_cls.return_value = self._mock_workspace_client(html_content=html)
        result = capture_job_run(run_id=100)
        assert "42000" in result
        assert "load_data" in result

    @patch("customer_retention.integrations.databricks_job_capture.WorkspaceClient")
    def test_multi_task_job(self, mock_ws_cls):
        html = _full_html(SIMPLE_CODE_CELL_HTML)
        tasks = [("nb_00", 101), ("nb_01", 102)]
        mock_ws_cls.return_value = self._mock_workspace_client(tasks=tasks, html_content=html)
        result = capture_job_run(run_id=100)
        assert "nb_00" in result
        assert "nb_01" in result

    @patch("customer_retention.integrations.databricks_job_capture.WorkspaceClient")
    def test_charts_excluded(self, mock_ws_cls):
        html = _full_html(CHART_CELL_HTML, IMAGE_CHART_HTML, SIMPLE_CODE_CELL_HTML)
        mock_ws_cls.return_value = self._mock_workspace_client(html_content=html)
        result = capture_job_run(run_id=100)
        assert "plotly" not in result.lower()
        assert "data:image" not in result
        assert "42000" in result

    @patch("customer_retention.integrations.databricks_job_capture.WorkspaceClient")
    def test_output_path_writes_file(self, mock_ws_cls, tmp_path):
        html = _full_html(SIMPLE_CODE_CELL_HTML)
        mock_ws_cls.return_value = self._mock_workspace_client(html_content=html)
        out = tmp_path / "output.md"
        result = capture_job_run(run_id=100, output_path=str(out))
        assert out.exists()
        assert out.read_text() == result

    @patch("customer_retention.integrations.databricks_job_capture.WorkspaceClient")
    def test_empty_notebook(self, mock_ws_cls):
        html = "<html><body></body></html>"
        mock_ws_cls.return_value = self._mock_workspace_client(html_content=html)
        result = capture_job_run(run_id=100)
        assert "_No captured outputs._" in result
