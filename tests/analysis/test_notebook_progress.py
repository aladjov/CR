"""Tests for notebook progress tracker."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from customer_retention.analysis.notebook_progress import (
    _ensure_databricks_config_loaded,
    publish_skip_flags,
    track_and_export_previous,
)


def _patch_experiments_dir(tmp_path):
    return patch(
        "customer_retention.analysis.notebook_progress.get_notebook_experiments_dir",
        return_value=tmp_path,
    )


class TestTrackAndExportPrevious:
    def test_first_run_no_export(self, tmp_path):
        """No progress file exists → returns None, creates progress file."""
        with _patch_experiments_dir(tmp_path):
            result = track_and_export_previous("00_start_here.ipynb")

        assert result is None
        progress = tmp_path / "notebook_progress.json"
        assert progress.exists()
        data = json.loads(progress.read_text())
        assert data["last_notebook"] == "00_start_here.ipynb"

    def test_exports_previous_notebook(self, tmp_path):
        """Progress says '01.ipynb', current is '02.ipynb' → dispatches export in background, returns None."""
        progress = tmp_path / "notebook_progress.json"
        progress.write_text(json.dumps({"last_notebook": "01_data_discovery.ipynb"}))

        with _patch_experiments_dir(tmp_path), \
             patch("customer_retention.analysis.notebook_progress.threading") as mock_threading:
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread
            result = track_and_export_previous("04_column_deep_dive.ipynb")

        assert result is None
        mock_threading.Thread.assert_called_once()
        mock_thread.start.assert_called_once()

    def test_updates_progress_after_export(self, tmp_path):
        """Progress file content reflects the current notebook after call."""
        progress = tmp_path / "notebook_progress.json"
        progress.write_text(json.dumps({"last_notebook": "01_data_discovery.ipynb"}))

        with _patch_experiments_dir(tmp_path), \
             patch("customer_retention.analysis.notebook_progress.threading"):
            track_and_export_previous("04_column_deep_dive.ipynb")

        data = json.loads(progress.read_text())
        assert data["last_notebook"] == "04_column_deep_dive.ipynb"

    def test_handles_missing_previous_notebook(self, tmp_path):
        """Previous notebook file doesn't exist → export dispatched but result is None."""
        progress = tmp_path / "notebook_progress.json"
        progress.write_text(json.dumps({"last_notebook": "nonexistent.ipynb"}))

        with _patch_experiments_dir(tmp_path), \
             patch("customer_retention.analysis.notebook_progress.threading"):
            result = track_and_export_previous("04_column_deep_dive.ipynb")

        assert result is None
        data = json.loads(progress.read_text())
        assert data["last_notebook"] == "04_column_deep_dive.ipynb"

    def test_handles_corrupt_progress_file(self, tmp_path):
        """Bad JSON → no export, creates fresh progress."""
        progress = tmp_path / "notebook_progress.json"
        progress.write_text("not valid json {{{")

        with _patch_experiments_dir(tmp_path):
            result = track_and_export_previous("04_column_deep_dive.ipynb")

        assert result is None
        data = json.loads(progress.read_text())
        assert data["last_notebook"] == "04_column_deep_dive.ipynb"

    def test_handles_oserror_on_mkdir_gracefully(self):
        mock_dir = MagicMock(spec=Path)
        mock_dir.mkdir.side_effect = OSError(95, "Operation not supported")
        with patch(
            "customer_retention.analysis.notebook_progress.get_notebook_experiments_dir",
            return_value=mock_dir,
        ):
            result = track_and_export_previous("01.ipynb")
        assert result is None

    def test_creates_experiments_dir_if_missing(self, tmp_path):
        """Experiments dir doesn't exist → created."""
        experiments_dir = tmp_path / "nested" / "experiments"

        with patch(
            "customer_retention.analysis.notebook_progress.get_notebook_experiments_dir",
            return_value=experiments_dir,
        ):
            track_and_export_previous("00_start_here.ipynb")

        assert experiments_dir.exists()
        assert (experiments_dir / "notebook_progress.json").exists()

    def test_progress_updated_before_export_starts(self, tmp_path):
        """Progress file must contain current notebook before export thread runs."""
        progress_file = tmp_path / "notebook_progress.json"
        progress_file.write_text(json.dumps({"last_notebook": "01.ipynb"}))

        progress_during_export = {}

        def fake_export(notebook_name, docs_dir):
            data = json.loads(progress_file.read_text())
            progress_during_export.update(data)

        with _patch_experiments_dir(tmp_path), \
             patch("customer_retention.analysis.notebook_progress._export_notebook", side_effect=fake_export):
            # Use real threading so the thread actually calls fake_export
            track_and_export_previous("02.ipynb")
            # Give the daemon thread a moment to run
            import time
            time.sleep(0.1)

        assert progress_during_export.get("last_notebook") == "02.ipynb"

    def test_export_runs_in_daemon_thread(self, tmp_path):
        """Export should be dispatched as a daemon thread."""
        progress_file = tmp_path / "notebook_progress.json"
        progress_file.write_text(json.dumps({"last_notebook": "01.ipynb"}))

        with _patch_experiments_dir(tmp_path), \
             patch("customer_retention.analysis.notebook_progress.threading") as mock_threading:
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread
            track_and_export_previous("02.ipynb")

        _, kwargs = mock_threading.Thread.call_args
        assert kwargs.get("daemon") is True
        mock_thread.start.assert_called_once()

    def test_export_exception_does_not_propagate(self, tmp_path):
        """If export raises, it must not crash the caller."""
        progress_file = tmp_path / "notebook_progress.json"
        progress_file.write_text(json.dumps({"last_notebook": "01.ipynb"}))

        def boom(notebook_name, docs_dir):
            raise RuntimeError("export failed")

        with _patch_experiments_dir(tmp_path), \
             patch("customer_retention.analysis.notebook_progress._export_notebook", side_effect=boom):
            # Should not raise — the exception is in a daemon thread
            result = track_and_export_previous("02.ipynb")
            import time
            time.sleep(0.1)

        assert result is None
        data = json.loads(progress_file.read_text())
        assert data["last_notebook"] == "02.ipynb"


class TestPublishSkipFlags:
    def test_noop_outside_databricks(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        findings = MagicMock()
        publish_skip_flags(findings)
        findings.assert_not_called()

    def test_sets_task_values_on_databricks(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        mock_dbutils = MagicMock()
        findings = MagicMock()
        findings.time_series_metadata = None
        findings.text_processing = None
        findings.column_types = {}
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_skip_flags(findings)
        calls = {c.kwargs["key"]: c.kwargs["value"] for c in mock_dbutils.jobs.taskValues.set.call_args_list}
        assert calls == {"has_event_data": False, "has_text_columns": False}

    def test_detects_event_level_data(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        from customer_retention.core.config.column_config import DatasetGranularity
        mock_dbutils = MagicMock()
        findings = MagicMock()
        findings.time_series_metadata.granularity = DatasetGranularity.EVENT_LEVEL
        findings.text_processing = None
        findings.column_types = {}
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_skip_flags(findings)
        calls = {c.kwargs["key"]: c.kwargs["value"] for c in mock_dbutils.jobs.taskValues.set.call_args_list}
        assert calls["has_event_data"] is True

    def test_detects_text_columns(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        from customer_retention.core.config.column_config import ColumnType
        mock_dbutils = MagicMock()
        findings = MagicMock()
        findings.time_series_metadata = None
        findings.text_processing = None
        findings.column_types = {"notes": ColumnType.TEXT}
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_skip_flags(findings)
        calls = {c.kwargs["key"]: c.kwargs["value"] for c in mock_dbutils.jobs.taskValues.set.call_args_list}
        assert calls["has_text_columns"] is True

    def test_noop_when_dbutils_unavailable(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        findings = MagicMock()
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=None):
            publish_skip_flags(findings)


class TestEnsureDatabricksConfigLoaded:
    def test_reloads_config_on_databricks(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        with patch("customer_retention.analysis.notebook_progress.reload_config") as mock_reload:
            _ensure_databricks_config_loaded()
        mock_reload.assert_called_once()

    def test_skips_reload_outside_databricks(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        with patch("customer_retention.analysis.notebook_progress.reload_config") as mock_reload:
            _ensure_databricks_config_loaded()
        mock_reload.assert_not_called()
