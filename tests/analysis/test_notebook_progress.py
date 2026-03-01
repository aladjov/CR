"""Tests for notebook progress tracker."""
import ast
import getpass
import inspect
from unittest.mock import MagicMock, patch

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.analysis.auto_explorer.session import SessionState
from customer_retention.analysis.notebook_progress import (
    _ensure_databricks_config_loaded,
    accept_workflow_params,
    guard_skip,
    publish_workflow_metadata,
    resolve_config,
    track_and_export_previous,
)


def _make_namespace(tmp_path, run_id="test-run-001"):
    ns = RunNamespace(root=tmp_path, run_id=run_id)
    ns.setup()
    return ns


def _write_session(ns, username, last_notebook=None, active_dataset=None):
    state = SessionState(
        active_dataset=active_dataset,
        active_run_id=ns.run_id,
        last_notebook=last_notebook,
    )
    state.save(ns.user_session_path(username))
    return state


_PATCH_NS = "customer_retention.analysis.auto_explorer.run_namespace.RunNamespace.from_env_or_latest"


class TestTrackAndExportPrevious:
    def test_returns_none_when_no_namespace(self):
        with patch(_PATCH_NS, return_value=None):
            result = track_and_export_previous("01_data_discovery.ipynb")
        assert result is None

    def test_first_run_records_current_notebook(self, tmp_path):
        ns = _make_namespace(tmp_path)
        username = getpass.getuser()

        with patch(_PATCH_NS, return_value=ns):
            result = track_and_export_previous("00_start_here.ipynb")

        assert result is None
        state = SessionState.load(ns.user_session_path(username))
        assert state is not None
        assert state.last_notebook == "00_start_here.ipynb"

    def test_exports_previous_notebook(self, tmp_path):
        ns = _make_namespace(tmp_path)
        username = getpass.getuser()
        _write_session(ns, username, last_notebook="01_data_discovery.ipynb")

        with patch(_PATCH_NS, return_value=ns), \
             patch("customer_retention.analysis.notebook_progress.threading") as mock_threading:
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread
            result = track_and_export_previous("04_column_deep_dive.ipynb")

        assert result is None
        mock_threading.Thread.assert_called_once()
        mock_thread.start.assert_called_once()

    def test_updates_session_after_call(self, tmp_path):
        ns = _make_namespace(tmp_path)
        username = getpass.getuser()
        _write_session(ns, username, last_notebook="01_data_discovery.ipynb")

        with patch(_PATCH_NS, return_value=ns), \
             patch("customer_retention.analysis.notebook_progress.threading"):
            track_and_export_previous("04_column_deep_dive.ipynb")

        state = SessionState.load(ns.user_session_path(username))
        assert state.last_notebook == "04_column_deep_dive.ipynb"

    def test_no_export_when_no_previous(self, tmp_path):
        ns = _make_namespace(tmp_path)

        with patch(_PATCH_NS, return_value=ns), \
             patch("customer_retention.analysis.notebook_progress.threading") as mock_threading:
            track_and_export_previous("01_data_discovery.ipynb")

        mock_threading.Thread.assert_not_called()

    def test_no_export_on_databricks(self, tmp_path):
        ns = _make_namespace(tmp_path)
        username = getpass.getuser()
        _write_session(ns, username, last_notebook="01_data_discovery.ipynb")

        with patch(_PATCH_NS, return_value=ns), \
             patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.analysis.notebook_progress.threading") as mock_threading:
            track_and_export_previous("04_column_deep_dive.ipynb")

        mock_threading.Thread.assert_not_called()

    def test_export_runs_in_daemon_thread(self, tmp_path):
        ns = _make_namespace(tmp_path)
        username = getpass.getuser()
        _write_session(ns, username, last_notebook="01.ipynb")

        with patch(_PATCH_NS, return_value=ns), \
             patch("customer_retention.analysis.notebook_progress.threading") as mock_threading:
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread
            track_and_export_previous("02.ipynb")

        _, kwargs = mock_threading.Thread.call_args
        assert kwargs.get("daemon") is True
        mock_thread.start.assert_called_once()

    def test_calls_ensure_config(self):
        with patch(_PATCH_NS, return_value=None), \
             patch("customer_retention.analysis.notebook_progress._ensure_databricks_config_loaded") as mock_ensure:
            track_and_export_previous("01.ipynb")
        mock_ensure.assert_called_once()

    def test_no_notebook_progress_json_created(self, tmp_path):
        ns = _make_namespace(tmp_path)

        with patch(_PATCH_NS, return_value=ns):
            track_and_export_previous("01.ipynb")

        assert not list(tmp_path.rglob("notebook_progress.json"))

    def test_session_updated_before_export_starts(self, tmp_path):
        ns = _make_namespace(tmp_path)
        username = getpass.getuser()
        _write_session(ns, username, last_notebook="01.ipynb")

        session_during_export = {}

        def fake_export(_notebook_name, _docs_dir):
            state = SessionState.load(ns.user_session_path(username))
            session_during_export["last_notebook"] = state.last_notebook

        with patch(_PATCH_NS, return_value=ns), \
             patch("customer_retention.analysis.notebook_progress._export_notebook", side_effect=fake_export):
            track_and_export_previous("02.ipynb")
            import time
            time.sleep(0.1)

        assert session_during_export.get("last_notebook") == "02.ipynb"

    def test_export_exception_does_not_propagate(self, tmp_path):
        ns = _make_namespace(tmp_path)
        username = getpass.getuser()
        _write_session(ns, username, last_notebook="01.ipynb")

        def boom(_notebook_name, _docs_dir):
            raise RuntimeError("export failed")

        with patch(_PATCH_NS, return_value=ns), \
             patch("customer_retention.analysis.notebook_progress._export_notebook", side_effect=boom):
            result = track_and_export_previous("02.ipynb")
            import time
            time.sleep(0.1)

        assert result is None
        state = SessionState.load(ns.user_session_path(username))
        assert state.last_notebook == "02.ipynb"


class TestNoImportSideEffects:
    def test_module_has_no_toplevel_function_calls(self):
        import customer_retention.analysis.notebook_progress as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                raise AssertionError(
                    f"Module-level function call at line {node.lineno}: "
                    f"{ast.unparse(node.value)}"
                )


class TestAcceptWorkflowParams:
    def test_noop_outside_databricks(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        accept_workflow_params()

    def test_sets_env_from_widgets(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        mock_dbutils = MagicMock()
        mock_dbutils.widgets.get.side_effect = lambda name: {
            "dataset_id": "my_dataset",
            "run_id": "run-123",
        }[name]
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            accept_workflow_params()
        import os
        assert os.environ.get("CR_DATASET_ID") == "my_dataset"
        assert os.environ.get("CR_RUN_ID") == "run-123"
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        monkeypatch.delenv("CR_RUN_ID", raising=False)

    def test_ignores_missing_widgets(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        mock_dbutils = MagicMock()
        mock_dbutils.widgets.get.side_effect = Exception("Widget not found")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            accept_workflow_params()
        import os
        assert os.environ.get("CR_DATASET_ID") is None

    def test_noop_when_dbutils_unavailable(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=None):
            accept_workflow_params()

    def test_skips_empty_widget_values(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        mock_dbutils = MagicMock()
        mock_dbutils.widgets.get.side_effect = lambda name: {
            "dataset_id": "",
            "run_id": "",
        }[name]
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            accept_workflow_params()
        import os
        assert os.environ.get("CR_DATASET_ID") is None

    def test_clears_stale_cr_dataset_id_when_widget_absent(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setenv("CR_DATASET_ID", "stale_account_dataset")
        monkeypatch.setenv("CR_RUN_ID", "run-previous")
        mock_dbutils = MagicMock()
        mock_dbutils.widgets.get.side_effect = Exception("Widget not found")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            accept_workflow_params()
        import os
        assert os.environ.get("CR_DATASET_ID") is None

    def test_cr_run_id_not_cleared_when_widget_absent(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setenv("CR_RUN_ID", "run-from-initialize")
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        mock_dbutils = MagicMock()
        mock_dbutils.widgets.get.side_effect = Exception("Widget not found")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            accept_workflow_params()
        import os
        assert os.environ.get("CR_RUN_ID") == "run-from-initialize"

    def test_clears_stale_cr_dataset_id_when_widget_empty(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setenv("CR_DATASET_ID", "stale_value")
        mock_dbutils = MagicMock()
        mock_dbutils.widgets.get.side_effect = lambda name: {
            "dataset_id": "",
            "run_id": "run-123",
        }[name]
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            accept_workflow_params()
        import os
        assert os.environ.get("CR_DATASET_ID") is None
        assert os.environ.get("CR_RUN_ID") == "run-123"
        monkeypatch.delenv("CR_RUN_ID", raising=False)

    def test_consecutive_calls_update_cr_dataset_id(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        mock_dbutils = MagicMock()

        mock_dbutils.widgets.get.side_effect = lambda name: {
            "dataset_id": "emails",
            "run_id": "run-1",
        }[name]
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            accept_workflow_params()
        import os
        assert os.environ.get("CR_DATASET_ID") == "emails"

        mock_dbutils.widgets.get.side_effect = lambda name: {
            "dataset_id": "transactions",
            "run_id": "run-1",
        }[name]
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            accept_workflow_params()
        assert os.environ.get("CR_DATASET_ID") == "transactions"
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        monkeypatch.delenv("CR_RUN_ID", raising=False)


class TestPublishWorkflowMetadata:
    def _make_context(self, dataset_names, target_dataset=None, run_id=None):
        ctx = MagicMock()
        ctx.datasets = {name: MagicMock() for name in dataset_names}
        ctx.target_dataset = target_dataset
        ctx.run_id = run_id
        return ctx

    def test_noop_outside_databricks(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        ctx = self._make_context(["ds1", "ds2"])
        publish_workflow_metadata(ctx)

    def test_publishes_dataset_names(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        mock_dbutils = MagicMock()
        ctx = self._make_context(["customers", "transactions"], target_dataset="customers")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_workflow_metadata(ctx)
        calls = {c.kwargs["key"]: c.kwargs["value"] for c in mock_dbutils.jobs.taskValues.set.call_args_list}
        assert calls["dataset_names"] == ["customers", "transactions"]
        assert calls["dataset_count"] == 2

    def test_publishes_target_dataset(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        mock_dbutils = MagicMock()
        ctx = self._make_context(["ds1"], target_dataset="ds1")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_workflow_metadata(ctx)
        calls = {c.kwargs["key"]: c.kwargs["value"] for c in mock_dbutils.jobs.taskValues.set.call_args_list}
        assert calls["target_dataset"] == "ds1"

    def test_publishes_run_id(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        mock_dbutils = MagicMock()
        ctx = self._make_context(["ds1"], run_id="run-abc-123")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_workflow_metadata(ctx)
        calls = {c.kwargs["key"]: c.kwargs["value"] for c in mock_dbutils.jobs.taskValues.set.call_args_list}
        assert calls["run_id"] == "run-abc-123"

    def test_no_run_id_when_absent(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        mock_dbutils = MagicMock()
        ctx = self._make_context(["ds1"], run_id=None)
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_workflow_metadata(ctx)
        keys = [c.kwargs["key"] for c in mock_dbutils.jobs.taskValues.set.call_args_list]
        assert "run_id" not in keys

    def test_noop_when_dbutils_unavailable(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        ctx = self._make_context(["ds1"])
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=None):
            publish_workflow_metadata(ctx)

    def test_single_dataset(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        mock_dbutils = MagicMock()
        ctx = self._make_context(["only_one"], target_dataset="only_one")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_workflow_metadata(ctx)
        calls = {c.kwargs["key"]: c.kwargs["value"] for c in mock_dbutils.jobs.taskValues.set.call_args_list}
        assert calls["dataset_names"] == ["only_one"]
        assert calls["dataset_count"] == 1

    def test_empty_target_dataset(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        mock_dbutils = MagicMock()
        ctx = self._make_context(["ds1"], target_dataset=None)
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            publish_workflow_metadata(ctx)
        calls = {c.kwargs["key"]: c.kwargs["value"] for c in mock_dbutils.jobs.taskValues.set.call_args_list}
        assert calls["target_dataset"] == ""


class TestGuardSkip:
    def test_noop_outside_databricks(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        guard_skip("01a_temporal_deep_dive")

    def test_noop_without_dataset_id(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.delenv("CR_DATASET_ID", raising=False)
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True):
            guard_skip("01a_temporal_deep_dive")

    def test_exits_when_notebook_in_skip_set(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setenv("CR_DATASET_ID", "entity_dataset")
        mock_dbutils = MagicMock()
        skip_set = {"01a_temporal_deep_dive"}
        skip_reasons = {"01a_temporal_deep_dive": "no event-level data (entity_dataset)"}
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils), \
             patch("customer_retention.analysis.auto_explorer.skip_logic.detect_skip_set_for_dataset",
                   return_value=(skip_set, skip_reasons)):
            guard_skip("01a_temporal_deep_dive")
        mock_dbutils.notebook.exit.assert_called_once()
        assert "SKIPPED" in mock_dbutils.notebook.exit.call_args[0][0]

    def test_continues_when_notebook_not_in_skip_set(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setenv("CR_DATASET_ID", "event_dataset")
        mock_dbutils = MagicMock()
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils), \
             patch("customer_retention.analysis.auto_explorer.skip_logic.detect_skip_set_for_dataset",
                   return_value=(set(), {})):
            guard_skip("01a_temporal_deep_dive")
        mock_dbutils.notebook.exit.assert_not_called()

    def test_noop_when_dbutils_unavailable(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setenv("CR_DATASET_ID", "ds1")
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=None):
            guard_skip("01a_temporal_deep_dive")

    def test_passes_namespace_to_detect_skip_set(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        monkeypatch.setenv("CR_DATASET_ID", "my_dataset")
        monkeypatch.setenv("CR_RUN_ID", "test-run-001")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path))
        _make_namespace(tmp_path, run_id="test-run-001")
        mock_dbutils = MagicMock()
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils), \
             patch("customer_retention.analysis.auto_explorer.skip_logic.detect_skip_set_for_dataset",
                   return_value=(set(), {})) as mock_detect:
            guard_skip("01a_temporal_deep_dive")
        _, kwargs = mock_detect.call_args
        assert kwargs.get("namespace") is not None
        assert kwargs["namespace"].run_id == "test-run-001"

    def test_event_level_not_skipped_with_namespace_findings(self, monkeypatch, tmp_path):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings, TimeSeriesMetadata
        from customer_retention.core.config.column_config import DatasetGranularity

        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        monkeypatch.setenv("CR_DATASET_ID", "events")
        monkeypatch.setenv("CR_RUN_ID", "run-ev")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path))
        ns = _make_namespace(tmp_path, run_id="run-ev")
        findings = ExplorationFindings(
            column_count=5, columns={}, target_column=None,
            source_path="events.csv", source_format="csv",
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column="cid", time_column="ts", temporal_pattern="recurring",
            ),
        )
        findings_dir = ns.dataset_findings_dir("events")
        findings_dir.mkdir(parents=True, exist_ok=True)
        findings.save(str(findings_dir / "events_findings.yaml"))
        mock_dbutils = MagicMock()
        with patch("customer_retention.analysis.notebook_progress.is_databricks", return_value=True), \
             patch("customer_retention.core.compat.detection.get_dbutils", return_value=mock_dbutils):
            guard_skip("01a_temporal_deep_dive")
        mock_dbutils.notebook.exit.assert_not_called()


class TestResolveConfig:
    def test_scalar_returns_as_is(self):
        assert resolve_config("D", "any_dataset") == "D"

    def test_dict_returns_matching_key(self):
        cfg = {"emails": "H", "transactions": "D"}
        assert resolve_config(cfg, "emails") == "H"
        assert resolve_config(cfg, "transactions") == "D"

    def test_dict_returns_default_for_missing_key(self):
        cfg = {"emails": "H"}
        assert resolve_config(cfg, "unknown", default="D") == "D"

    def test_dict_returns_none_when_no_default(self):
        cfg = {"emails": "H"}
        assert resolve_config(cfg, "unknown") is None

    def test_none_value_returns_none(self):
        assert resolve_config(None, "ds") is None

    def test_numeric_scalar(self):
        assert resolve_config(90, "ds") == 90

    def test_dict_with_numeric_values(self):
        cfg = {"fast": 30, "slow": 180}
        assert resolve_config(cfg, "fast") == 30

    def test_bool_scalar(self):
        assert resolve_config(True, "ds") is True

    def test_list_scalar(self):
        val = [7, 30, 90]
        assert resolve_config(val, "ds") == [7, 30, 90]

    def test_dict_with_list_values(self):
        cfg = {"emails": [7, 30], "profiles": [90]}
        assert resolve_config(cfg, "emails") == [7, 30]


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
