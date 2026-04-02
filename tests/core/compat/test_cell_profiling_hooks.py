"""Tests for cell_profiling_hooks — permanent Spark job counter in notebook init cells."""

from __future__ import annotations

import json
from pathlib import Path

from customer_retention.core.compat.cell_profiling_hooks import (
    _SENTINEL_END,
    _SENTINEL_START,
    PROFILER_BLOCK,
    add_profiler,
    has_profiler,
    remove_profiler,
)


def _make_notebook(cells: list[dict] | None = None) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3"}},
        "cells": cells
        or [
            {
                "cell_type": "code",
                "id": "aaa111",
                "source": "# @cr:code name='init' id=aaa111\nimport os\n",
                "metadata": {},
                "outputs": [],
            },
            {
                "cell_type": "code",
                "id": "bbb222",
                "source": "# @cr:code name='c2' id=bbb222\ny=2\n",
                "metadata": {},
                "outputs": [],
            },
        ],
    }


class TestAddProfiler:
    def test_appends_to_first_code_cell(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(_make_notebook()))
        assert add_profiler(nb_path) is True
        nb = json.loads(nb_path.read_text())
        assert len(nb["cells"]) == 2  # no new cell
        first_src = nb["cells"][0]["source"]
        assert _SENTINEL_START in first_src
        assert "pre_run_cell" in first_src
        assert "import os" in first_src  # original preserved

    def test_skips_markdown_cells(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        cells = [
            {"cell_type": "markdown", "id": "md1", "source": "# Title\n", "metadata": {}},
            {"cell_type": "code", "id": "c1", "source": "x = 1\n", "metadata": {}, "outputs": []},
        ]
        nb_path.write_text(json.dumps(_make_notebook(cells)))
        add_profiler(nb_path)
        nb = json.loads(nb_path.read_text())
        assert _SENTINEL_START not in nb["cells"][0]["source"]
        assert _SENTINEL_START in nb["cells"][1]["source"]

    def test_idempotent(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(_make_notebook()))
        add_profiler(nb_path)
        assert add_profiler(nb_path) is False
        nb = json.loads(nb_path.read_text())
        assert nb["cells"][0]["source"].count(_SENTINEL_START) == 1

    def test_block_is_guarded_by_batch_env(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(_make_notebook()))
        add_profiler(nb_path)
        nb = json.loads(nb_path.read_text())
        assert "CR_BATCH_EXECUTION" in nb["cells"][0]["source"]


class TestHasProfiler:
    def test_true_when_present(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(_make_notebook()))
        add_profiler(nb_path)
        assert has_profiler(nb_path) is True

    def test_false_when_absent(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(_make_notebook()))
        assert has_profiler(nb_path) is False


class TestRemoveProfiler:
    def test_strips_profiler_block(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(_make_notebook()))
        add_profiler(nb_path)
        assert remove_profiler(nb_path) is True
        nb = json.loads(nb_path.read_text())
        assert _SENTINEL_START not in nb["cells"][0]["source"]
        assert "import os" in nb["cells"][0]["source"]

    def test_noop_when_absent(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(_make_notebook()))
        assert remove_profiler(nb_path) is False
