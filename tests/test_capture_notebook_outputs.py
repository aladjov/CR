"""Tests for scripts/experiments/capture_notebook_outputs.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from capture_notebook_outputs import capture_notebook


def _make_notebook(tmp_path, cells):
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb))
    return path


def _code_cell(cell_id, outputs, source="# @cr:code name='test_cell' id=aabbccdd\npass"):
    return {
        "cell_type": "code",
        "id": cell_id,
        "source": source,
        "outputs": outputs,
        "metadata": {},
    }


class TestTextMarkdownCapture:
    def test_captures_text_markdown_output(self, tmp_path):
        outputs = [{"output_type": "display_data", "data": {
            "text/markdown": "**Bold result**: 42 rows processed",
            "text/plain": "<IPython.core.display.Markdown object>",
        }}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert "**Bold result**: 42 rows processed" in result
        assert "<IPython.core.display.Markdown object>" not in result

    def test_html_preferred_over_markdown(self, tmp_path):
        outputs = [{"output_type": "display_data", "data": {
            "text/html": "<table><tr><td>HTML</td></tr></table>",
            "text/markdown": "| Markdown table |",
            "text/plain": "<IPython.core.display.Markdown object>",
        }}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert "<table>" in result
        assert "Markdown table" not in result

    def test_markdown_preferred_over_plain(self, tmp_path):
        outputs = [{"output_type": "display_data", "data": {
            "text/markdown": "# Important finding",
            "text/plain": "<IPython.core.display.Markdown object>",
        }}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert "# Important finding" in result
        assert "Markdown object" not in result

    def test_plain_text_still_captured_when_no_markdown(self, tmp_path):
        outputs = [{"output_type": "execute_result", "data": {
            "text/plain": "42",
        }}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert "42" in result

    def test_empty_markdown_falls_through_to_plain(self, tmp_path):
        outputs = [{"output_type": "display_data", "data": {
            "text/markdown": "",
            "text/plain": "fallback text",
        }}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert "fallback text" in result

    def test_markdown_output_labeled_correctly(self, tmp_path):
        outputs = [{"output_type": "display_data", "data": {
            "text/markdown": "Dataset has 100 rows",
        }}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert "### Output (text/markdown)" in result


class TestExistingCaptureBehavior:
    def test_stream_output_captured(self, tmp_path):
        outputs = [{"output_type": "stream", "name": "stdout", "text": "hello world"}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert "hello world" in result

    def test_chart_only_output_skipped(self, tmp_path):
        outputs = [{"output_type": "display_data", "data": {
            "image/png": "base64data",
        }}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert result is None

    def test_error_output_captured(self, tmp_path):
        outputs = [{"output_type": "error", "ename": "ValueError", "evalue": "bad", "traceback": ["trace"]}]
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", outputs)])
        result = capture_notebook(path)
        assert "ValueError" in result

    def test_no_outputs_returns_none(self, tmp_path):
        path = _make_notebook(tmp_path, [_code_cell("aabbccdd", [])])
        result = capture_notebook(path)
        assert result is None
