"""Tests for the ``causal_notebooks_path`` wiring across init paths.

Mirrors ``test_get_playbooks_dir.py``: every place that already accepts
``exploration_notebooks_path`` and ``playbooks_path`` (databricks_init,
ProjectInitializer, churnkit-init CLI, core/config/experiments) must
accept the new ``causal_notebooks_path`` in parallel.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    monkeypatch.delenv("CR_CAUSAL_NOTEBOOKS_DIR", raising=False)
    monkeypatch.delenv("CR_PLAYBOOKS_DIR", raising=False)
    monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
    yield


class TestGetCausalNotebooksDir:
    def test_env_var_takes_precedence(self, monkeypatch):
        from customer_retention.core.config.experiments import get_causal_notebooks_dir

        monkeypatch.setenv("CR_CAUSAL_NOTEBOOKS_DIR", "/Workspace/u/custom_causal")
        assert str(get_causal_notebooks_dir()) == "/Workspace/u/custom_causal"

    def test_explicit_default_argument(self):
        from customer_retention.core.config.experiments import get_causal_notebooks_dir

        assert str(get_causal_notebooks_dir(default="/some/explicit/path")) == "/some/explicit/path"

    def test_local_fallback_resolves_to_project_root_causal_notebooks(self):
        from customer_retention.core.config.experiments import get_causal_notebooks_dir

        result = get_causal_notebooks_dir()
        assert str(result).endswith("causal_notebooks")

    def test_reload_config_picks_up_env_change(self, monkeypatch):
        from customer_retention.core.config import experiments

        monkeypatch.setenv("CR_CAUSAL_NOTEBOOKS_DIR", "/tmp/test_reload_causal")
        experiments.reload_config()
        assert str(experiments.CAUSAL_NOTEBOOKS_DIR) == "/tmp/test_reload_causal"


class TestCausalNotebooksDirReExports:
    def test_re_exported_from_core_config(self):
        from customer_retention.core.config import (
            CAUSAL_NOTEBOOKS_DIR,
            get_causal_notebooks_dir,
        )

        assert get_causal_notebooks_dir is not None
        assert CAUSAL_NOTEBOOKS_DIR is not None


class TestProjectInitializerWiring:
    def test_dataclass_has_causal_notebooks_path_field(self):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

        init = ProjectInitializer(project_name="t")
        assert hasattr(init, "causal_notebooks_path")
        assert init.causal_notebooks_path == "causal_notebooks"

    def test_custom_causal_notebooks_path(self):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

        init = ProjectInitializer(project_name="t", causal_notebooks_path="my_causal")
        assert init.causal_notebooks_path == "my_causal"

    def test_initialize_project_creates_causal_dir_and_copies_notebooks(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import initialize_project

        result = initialize_project(
            output_dir=str(tmp_path / "new_project"),
            project_name="new_project",
            causal_notebooks_path="my_causal",
        )
        assert (tmp_path / "new_project" / "my_causal").is_dir()
        # Causal notebooks should be copied (the framework ships four of them)
        copied = result.get("causal_notebooks", [])
        assert any(p.endswith("c01_publish_definitions.ipynb") for p in copied)
        assert any(p.endswith("c04_snapshot_and_dashboard.ipynb") for p in copied)

    def test_readme_mentions_causal_notebooks_dir(self, tmp_path):
        from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

        init = ProjectInitializer(project_name="t", causal_notebooks_path="cn")
        assert "cn/" in init._readme_content()


class TestDatabricksInitWiring:
    def test_signature_has_causal_notebooks_path(self):
        import inspect

        from customer_retention.integrations.databricks_init import databricks_init

        sig = inspect.signature(databricks_init)
        assert "causal_notebooks_path" in sig.parameters
        assert sig.parameters["causal_notebooks_path"].default == "causal_notebooks"

    def test_result_dataclass_has_causal_notebooks_path(self):
        from customer_retention.integrations.databricks_init import DatabricksInitResult

        assert "causal_notebooks_path" in DatabricksInitResult.__dataclass_fields__
        assert (
            DatabricksInitResult.__dataclass_fields__["causal_notebooks_path"].default
            == "causal_notebooks"
        )

    def test_environment_variables_includes_causal_dir_when_workspace_set(self):
        from customer_retention.integrations.databricks_init import DatabricksInitResult

        result = DatabricksInitResult(
            catalog="cat",
            schema="sch",
            experiment_name="exp",
            workspace_path="Users/me",
            model_name="m",
            causal_notebooks_path="my_causal",
        )
        env = result.environment_variables
        assert env["CR_CAUSAL_NOTEBOOKS_DIR"] == "/Workspace/Users/me/my_causal"

    def test_no_causal_env_var_without_workspace_path(self):
        from customer_retention.integrations.databricks_init import DatabricksInitResult

        result = DatabricksInitResult(
            catalog="cat",
            schema="sch",
            experiment_name="exp",
            workspace_path=None,
            model_name="m",
        )
        assert "CR_CAUSAL_NOTEBOOKS_DIR" not in result.environment_variables


class TestPersistDatabricksConfigCausalDir:
    def test_persist_writes_causal_notebooks_dir(self, tmp_path, monkeypatch):
        from customer_retention.core.config import experiments as exp

        # Redirect _workspace_config_path to write into tmp_path
        target = tmp_path / "config.json"
        monkeypatch.setattr(exp, "_workspace_config_path", lambda _ws: target)
        exp.persist_databricks_config(
            "/Volumes/cat/sch/experiments",
            "cat",
            "sch",
            workspace_path="Users/me",
            causal_notebooks_dir="/Workspace/Users/me/causal_notebooks",
        )
        import json
        data = json.loads(target.read_text())
        assert data["causal_notebooks_dir"] == "/Workspace/Users/me/causal_notebooks"


class TestCliWiring:
    def test_cli_has_causal_notebooks_path_argument(self):
        import inspect

        import customer_retention.cli as cli

        source = inspect.getsource(cli)
        assert "--causal-notebooks-path" in source
        assert 'default="causal_notebooks"' in source


class TestSyncCliCausalSupport:
    def test_help_mentions_causal(self):
        import inspect

        import customer_retention.generators.notebook_sync.cli as sync_cli

        source = inspect.getsource(sync_cli)
        assert "--causal-repo-dir" in source
        assert "--causal-user-dir" in source

    def test_resolve_causal_repo_dir_uses_sibling(self, tmp_path):
        from customer_retention.generators.notebook_sync.cli import _resolve_causal_repo_dir

        repo_root = tmp_path / "framework"
        (repo_root / "exploration_notebooks").mkdir(parents=True)
        (repo_root / "causal_notebooks").mkdir(parents=True)

        class _Args:
            no_causal = False
            causal_repo_dir = None

        result = _resolve_causal_repo_dir(_Args, repo_root / "exploration_notebooks")
        assert result == (repo_root / "causal_notebooks").resolve()

    def test_resolve_causal_repo_dir_returns_none_when_disabled(self, tmp_path):
        from customer_retention.generators.notebook_sync.cli import _resolve_causal_repo_dir

        class _Args:
            no_causal = True
            causal_repo_dir = None

        assert _resolve_causal_repo_dir(_Args, tmp_path) is None

    def test_resolve_causal_repo_dir_explicit_path_wins(self, tmp_path):
        from customer_retention.generators.notebook_sync.cli import _resolve_causal_repo_dir

        explicit = tmp_path / "explicit_causal"
        explicit.mkdir()

        class _Args:
            no_causal = False
            causal_repo_dir = str(explicit)

        assert _resolve_causal_repo_dir(_Args, tmp_path) == explicit.resolve()


class TestJupyterSaveHookCausal:
    def test_export_triggers_for_causal_notebooks(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from customer_retention.analysis.jupyter_save_hook import post_save_export

        nb_path = tmp_path / "causal_notebooks" / "c04_snapshot_and_dashboard.ipynb"
        nb_path.parent.mkdir()
        nb_path.write_text("{}")
        with patch(
            "customer_retention.analysis.jupyter_save_hook.export_notebook_html"
        ) as mock_export:
            post_save_export({"type": "notebook"}, str(nb_path), MagicMock())
            mock_export.assert_called_once()

    def test_export_skips_other_dirs(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from customer_retention.analysis.jupyter_save_hook import post_save_export

        nb_path = tmp_path / "random" / "x.ipynb"
        nb_path.parent.mkdir()
        nb_path.write_text("{}")
        with patch(
            "customer_retention.analysis.jupyter_save_hook.export_notebook_html"
        ) as mock_export:
            post_save_export({"type": "notebook"}, str(nb_path), MagicMock())
            mock_export.assert_not_called()
