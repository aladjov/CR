"""Tests for scripts/experiments/cell_profiling.py — per-cell performance profile extraction."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from cell_profiling import (
    CellProfileEntry,
    CellProfileManifest,
    NotebookProfile,
    extract_profiles_from_notebook,
    extract_profiles_from_notebooks,
    load_cell_profiles,
    merge_profiles,
    write_cell_profiles,
)

# ---------------------------------------------------------------------------
# Helpers — create mock notebook JSON
# ---------------------------------------------------------------------------

def _make_notebook(cells: list[dict]) -> dict:
    """Build a minimal nbformat-like notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": cells,
    }


def _code_cell(
    name: str,
    cell_id: str,
    duration: float = 1.0,
    status: str = "completed",
    exception: bool = False,
    start_time: str = "2026-04-01T10:00:00",
    end_time: str = "2026-04-01T10:00:01",
    kind: str = "code",
) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "source": [f"# @cr:{kind} name='{name}' id={cell_id}\n", "x = 1\n"],
        "metadata": {
            "papermill": {
                "duration": duration,
                "status": status,
                "exception": exception,
                "start_time": start_time,
                "end_time": end_time,
            }
        },
        "outputs": [],
    }


def _markdown_cell(cell_id: str = "md001") -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "source": ["# Title\n"],
        "metadata": {
            "papermill": {
                "duration": 0.003,
                "status": "completed",
                "exception": False,
                "start_time": "2026-04-01T10:00:00",
                "end_time": "2026-04-01T10:00:00.003",
            }
        },
    }


def _code_cell_no_cr_tag(cell_id: str, duration: float = 0.5) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "source": ["import os\n"],
        "metadata": {
            "papermill": {
                "duration": duration,
                "status": "completed",
                "exception": False,
                "start_time": "2026-04-01T10:00:00",
                "end_time": "2026-04-01T10:00:00.5",
            }
        },
        "outputs": [],
    }


# ---------------------------------------------------------------------------
# Tests — single notebook extraction
# ---------------------------------------------------------------------------


class TestExtractProfilesFromNotebook:
    def test_basic_extraction(self, tmp_path):
        nb = _make_notebook([
            _markdown_cell(),
            _code_cell("load_data", "abc123", duration=3.21),
            _code_cell("validate", "def456", duration=1.05),
        ])
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb))
        result = extract_profiles_from_notebook(nb_path)

        assert result is not None
        assert len(result.cells) == 2
        assert result.cells[0].cell_name == "load_data"
        assert result.cells[0].cell_id == "abc123"
        assert result.cells[0].elapsed_sec == pytest.approx(3.21, abs=0.001)
        assert result.cells[1].cell_name == "validate"
        assert result.total_elapsed == pytest.approx(4.26, abs=0.01)

    def test_skips_markdown_cells(self, tmp_path):
        nb = _make_notebook([
            _markdown_cell(),
            _markdown_cell("md002"),
            _code_cell("code1", "aaa111", duration=1.0),
        ])
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb))
        result = extract_profiles_from_notebook(nb_path)

        assert result is not None
        assert len(result.cells) == 1
        assert result.cells[0].cell_name == "code1"

    def test_handles_failed_cells(self, tmp_path):
        nb = _make_notebook([
            _code_cell("ok_cell", "ok1", duration=1.0),
            _code_cell("bad_cell", "bad1", duration=0.5, exception=True),
        ])
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb))
        result = extract_profiles_from_notebook(nb_path)

        assert result is not None
        assert result.cells[0].status == "completed"
        assert result.cells[1].status == "failed"

    def test_fallback_cell_name_without_cr_tag(self, tmp_path):
        nb = _make_notebook([
            _code_cell_no_cr_tag("xyz789", duration=0.5),
        ])
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb))
        result = extract_profiles_from_notebook(nb_path)

        assert result is not None
        assert result.cells[0].cell_name == "cell_xyz789"
        assert result.cells[0].cell_id == "xyz789"

    def test_returns_none_for_unexecuted_notebook(self, tmp_path):
        nb = _make_notebook([
            {"cell_type": "code", "id": "nopm", "source": ["x=1"], "metadata": {}, "outputs": []},
        ])
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb))
        result = extract_profiles_from_notebook(nb_path)

        assert result is None

    def test_null_duration_treated_as_zero(self, tmp_path):
        nb = _make_notebook([
            _code_cell("cell1", "c1", duration=None),
        ])
        # Manually patch duration to None
        nb["cells"][0]["metadata"]["papermill"]["duration"] = None
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb))
        result = extract_profiles_from_notebook(nb_path)

        assert result is not None
        assert result.cells[0].elapsed_sec == 0.0

    def test_config_kind_cells_extracted(self, tmp_path):
        nb = _make_notebook([
            _code_cell("discovery_config", "cfg1", duration=0.01, kind="config"),
        ])
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb))
        result = extract_profiles_from_notebook(nb_path)

        assert result is not None
        assert result.cells[0].cell_name == "discovery_config"


# ---------------------------------------------------------------------------
# Tests — directory extraction
# ---------------------------------------------------------------------------


class TestExtractProfilesFromNotebooks:
    def test_discovers_and_orders_notebooks(self, tmp_path):
        for name in ["01_data_discovery", "00_start_here", "extra_notebook"]:
            nb = _make_notebook([_code_cell(f"cell_{name}", f"id_{name}", duration=1.0)])
            (tmp_path / f"{name}.ipynb").write_text(json.dumps(nb))

        manifest = extract_profiles_from_notebooks(tmp_path)
        names = list(manifest.notebooks.keys())
        # Canonical order first, then extras
        assert names == ["00_start_here", "01_data_discovery", "extra_notebook"]

    def test_skips_hidden_notebooks(self, tmp_path):
        nb = _make_notebook([_code_cell("cell1", "id1")])
        (tmp_path / ".hidden.ipynb").write_text(json.dumps(nb))
        (tmp_path / "_private.ipynb").write_text(json.dumps(nb))
        (tmp_path / "visible.ipynb").write_text(json.dumps(nb))

        manifest = extract_profiles_from_notebooks(tmp_path)
        assert list(manifest.notebooks.keys()) == ["visible"]


# ---------------------------------------------------------------------------
# Tests — serialisation round-trip
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_write_and_load_round_trip(self, tmp_path):
        entry = CellProfileEntry(
            cell_name="load_data", cell_id="abc123",
            elapsed_sec=3.21, status="completed",
            start_time="2026-04-01T10:00:00", end_time="2026-04-01T10:00:03",
            spark_jobs=5, peak_memory_mb=12.4,
        )
        manifest = CellProfileManifest(
            environment="local",
            generated_at="2026-04-02T10:00:00",
            notebooks={"01_data_discovery": NotebookProfile(total_elapsed=3.21, cells=[entry])},
        )

        out = tmp_path / "cell_profiles.json"
        write_cell_profiles(manifest, out)

        loaded = load_cell_profiles(out)
        assert loaded is not None
        assert loaded.version == 1
        assert loaded.environment == "local"
        assert "01_data_discovery" in loaded.notebooks
        nb = loaded.notebooks["01_data_discovery"]
        assert nb.total_elapsed == pytest.approx(3.21)
        assert len(nb.cells) == 1
        c = nb.cells[0]
        assert c.cell_name == "load_data"
        assert c.spark_jobs == 5
        assert c.peak_memory_mb == pytest.approx(12.4)

    def test_load_missing_file_returns_none(self, tmp_path):
        result = load_cell_profiles(tmp_path / "nonexistent.json")
        assert result is None

    def test_json_structure(self, tmp_path):
        entry = CellProfileEntry(
            cell_name="test", cell_id="id1",
            elapsed_sec=1.0, status="completed",
        )
        manifest = CellProfileManifest(
            notebooks={"nb1": NotebookProfile(total_elapsed=1.0, cells=[entry])},
        )
        out = tmp_path / "test.json"
        write_cell_profiles(manifest, out)

        data = json.loads(out.read_text())
        assert data["version"] == 1
        assert "notebooks" in data
        assert "nb1" in data["notebooks"]
        cells = data["notebooks"]["nb1"]["cells"]
        assert len(cells) == 1
        assert cells[0]["spark_jobs"] is None  # null in JSON when not set


# ---------------------------------------------------------------------------
# Tests — merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_takes_longer_run(self):
        a = CellProfileManifest(
            environment="local",
            notebooks={
                "nb1": NotebookProfile(total_elapsed=10.0, cells=[
                    CellProfileEntry("c1", "id1", 10.0, "completed"),
                ]),
            },
        )
        b = CellProfileManifest(
            environment="local",
            notebooks={
                "nb1": NotebookProfile(total_elapsed=20.0, cells=[
                    CellProfileEntry("c1", "id1", 20.0, "completed"),
                ]),
                "nb2": NotebookProfile(total_elapsed=5.0, cells=[
                    CellProfileEntry("c2", "id2", 5.0, "completed"),
                ]),
            },
        )
        merged = merge_profiles(a, b)
        assert merged.notebooks["nb1"].total_elapsed == 20.0
        assert "nb2" in merged.notebooks


# ---------------------------------------------------------------------------
# Tests — drift report integration
# ---------------------------------------------------------------------------


class TestDriftReportIntegration:
    """Test that cell profiles integrate correctly into the drift report."""

    def _make_profiles(self, notebooks: dict[str, list[tuple[str, str, float]]]) -> CellProfileManifest:
        nbs = {}
        for nb_name, cells in notebooks.items():
            entries = [CellProfileEntry(name, cid, elapsed, "completed") for name, cid, elapsed in cells]
            total = sum(e.elapsed_sec for e in entries)
            nbs[nb_name] = NotebookProfile(total_elapsed=total, cells=entries)
        return CellProfileManifest(environment="local", notebooks=nbs)

    def _minimal_drift_with_profiles(self, old_profiles, new_profiles):
        from compare_exploration_runs import DriftResult, ParsedManifest, RunMetadata
        return DriftResult(
            old_meta=RunMetadata("aaa", "1.0", "Jan 01"),
            new_meta=RunMetadata("bbb", "2.0", "Feb 01"),
            old_manifest=ParsedManifest({}, ""),
            new_manifest=ParsedManifest({}, ""),
            notebook_drifts=[],
            snapshot_grid_old=None, snapshot_grid_new=None,
            project_context_old=None, project_context_new=None,
            feature_profile_old=None, feature_profile_new=None,
            old_timings={}, new_timings={},
            metric_tables=[],
            old_cell_profiles=old_profiles,
            new_cell_profiles=new_profiles,
        )

    def test_md_render_cell_profiling_with_data(self):
        from compare_exploration_runs import MarkdownDriftRenderer
        old = self._make_profiles({"nb1": [("load", "id1", 3.0), ("validate", "id2", 1.0)]})
        new = self._make_profiles({"nb1": [("load", "id1", 2.5), ("validate", "id2", 5.0)]})
        drift = self._minimal_drift_with_profiles(old, new)
        md = MarkdownDriftRenderer(drift).render_cell_profiling()

        assert "## Per-Cell Profiling" in md
        assert "nb1" in md
        assert "load" in md
        assert "validate" in md
        assert "id1" in md
        # Check timing values present
        assert "3.0s" in md
        assert "2.5s" in md

    def test_md_render_cell_profiling_no_profiles(self):
        from compare_exploration_runs import MarkdownDriftRenderer
        drift = self._minimal_drift_with_profiles(None, None)
        md = MarkdownDriftRenderer(drift).render_cell_profiling()
        assert "No cell profiles found" in md

    def test_html_render_cell_profiling_with_data(self):
        from compare_exploration_runs import HtmlDriftRenderer
        old = self._make_profiles({"nb1": [("load", "id1", 3.0)]})
        new = self._make_profiles({"nb1": [("load", "id1", 2.0)]})
        drift = self._minimal_drift_with_profiles(old, new)
        html = HtmlDriftRenderer(drift).render_cell_profiling()

        assert "<h2>Per-Cell Profiling</h2>" in html
        assert "<table>" in html
        assert "load" in html

    def test_cross_environment_note(self):
        from compare_exploration_runs import MarkdownDriftRenderer
        old = self._make_profiles({"nb1": [("c1", "id1", 1.0)]})
        old.environment = "local"
        new = self._make_profiles({"nb1": [("c1", "id1", 2.0)]})
        new.environment = "databricks"
        drift = self._minimal_drift_with_profiles(old, new)
        md = MarkdownDriftRenderer(drift).render_cell_profiling()
        assert "Cross-environment" in md
        assert "different hardware" in md

    def test_cell_timing_annotation_in_diff(self):
        from compare_exploration_runs import (
            CellMatch,
            CellMatchStatus,
            ManifestCell,
            MarkdownDriftRenderer,
            NotebookDrift,
        )
        old_profiles = self._make_profiles({"nb1": [("cell_a", "id_a", 2.0)]})
        new_profiles = self._make_profiles({"nb1": [("cell_a", "id_a", 5.0)]})

        old_cell = ManifestCell(name="cell_a", cell_id="id_a", section="", content="old output")
        new_cell = ManifestCell(name="cell_a", cell_id="id_a", section="", content="new output")
        nd = NotebookDrift("nb1", [
            CellMatch(old_cell, new_cell, CellMatchStatus.MODIFIED, "- old\n+ new"),
        ])

        drift = self._minimal_drift_with_profiles(old_profiles, new_profiles)
        drift.notebook_drifts = [nd]
        md = MarkdownDriftRenderer(drift).render_cell_by_cell_diff()

        # Should have inline timing annotation on the cell header
        assert "2.0s" in md
        assert "5.0s" in md
        assert "+3.0s" in md

    def test_full_render_pipeline_with_profiles(self):
        from compare_exploration_runs import HtmlDriftRenderer, MarkdownDriftRenderer
        old = self._make_profiles({"nb1": [("c1", "id1", 1.0)]})
        new = self._make_profiles({"nb1": [("c1", "id1", 2.0)]})
        drift = self._minimal_drift_with_profiles(old, new)

        md = MarkdownDriftRenderer(drift).render()
        html = HtmlDriftRenderer(drift).render()
        assert "Per-Cell Profiling" in md
        assert "Per-Cell Profiling" in html
