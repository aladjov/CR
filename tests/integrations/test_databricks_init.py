import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import nbformat
import pytest


@pytest.fixture(autouse=True)
def _clean_env_vars(monkeypatch):
    for var in ("CR_CATALOG", "CR_SCHEMA", "CR_WORKSPACE_PATH", "CR_EXPERIMENT_NAME", "CR_EXPERIMENTS_DIR"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def databricks_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")


class TestDatabricksInitEnvironmentVariables:
    def test_sets_cr_catalog(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(catalog="analytics", copy_notebooks=False)
        assert os.environ["CR_CATALOG"] == "analytics"

    def test_sets_cr_schema(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(schema="churn", copy_notebooks=False)
        assert os.environ["CR_SCHEMA"] == "churn"

    def test_sets_cr_workspace_path(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(workspace_path="Users/me/project", copy_notebooks=False)
        assert os.environ["CR_WORKSPACE_PATH"] == "Users/me/project"

    def test_sets_cr_experiment_name(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(experiment_name="my_exp", copy_notebooks=False)
        assert os.environ["CR_EXPERIMENT_NAME"] == "my_exp"

    def test_sets_cr_experiments_dir_from_catalog_and_schema(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(catalog="churnkit", schema="prod", copy_notebooks=False)
        assert os.environ["CR_EXPERIMENTS_DIR"] == "/Volumes/churnkit/prod/experiments"

    def test_sets_cr_experiments_dir_with_defaults(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(copy_notebooks=False)
        assert os.environ["CR_EXPERIMENTS_DIR"] == "/Volumes/main/default/experiments"

    def test_idempotent_multiple_calls(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(catalog="first", copy_notebooks=False)
        databricks_init(catalog="first", copy_notebooks=False)
        assert os.environ["CR_CATALOG"] == "first"

    def test_overrides_previous_values(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(catalog="first", copy_notebooks=False)
        databricks_init(catalog="second", copy_notebooks=False)
        assert os.environ["CR_CATALOG"] == "second"


class TestNormalizeWorkspacePath:
    def test_strips_workspace_prefix(self):
        from customer_retention.integrations.databricks_init import _normalize_workspace_path

        assert _normalize_workspace_path("/Workspace/Users/me/project") == "Users/me/project"

    def test_leaves_relative_path_unchanged(self):
        from customer_retention.integrations.databricks_init import _normalize_workspace_path

        assert _normalize_workspace_path("Users/me/project") == "Users/me/project"

    def test_env_vars_correct_when_workspace_prefix_passed(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(workspace_path="/Workspace/Users/me/project", copy_notebooks=False)
        assert os.environ["CR_WORKSPACE_PATH"] == "Users/me/project"
        assert os.environ["CR_EXPERIMENTS_DIR"] == "/Volumes/main/default/experiments"

    def test_result_workspace_path_normalized(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(workspace_path="/Workspace/Users/me/project", copy_notebooks=False)
        assert result.workspace_path == "Users/me/project"


class TestDatabricksInitValidation:
    def test_raises_runtime_error_on_non_databricks(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        from customer_retention.integrations.databricks_init import databricks_init

        with pytest.raises(RuntimeError, match="DATABRICKS_RUNTIME_VERSION"):
            databricks_init()

    def test_succeeds_on_databricks_environment(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert result.catalog == "main"


class TestMakeAbsoluteExperimentPath:
    def test_already_absolute_path_unchanged(self):
        from customer_retention.integrations.databricks_init import _make_absolute_experiment_path

        assert _make_absolute_experiment_path("/Users/me/exp", None) == "/Users/me/exp"

    def test_relative_name_with_workspace_path(self):
        from customer_retention.integrations.databricks_init import _make_absolute_experiment_path

        result = _make_absolute_experiment_path("my_exp", "Users/me/project")
        assert result == "/Users/me/project/my_exp"

    def test_relative_name_with_workspace_path_including_workspace_prefix(self):
        from customer_retention.integrations.databricks_init import _make_absolute_experiment_path

        result = _make_absolute_experiment_path("my_exp", "/Workspace/Users/me/project")
        assert result == "/Users/me/project/my_exp"

    def test_relative_name_without_workspace_path_unchanged(self):
        from customer_retention.integrations.databricks_init import _make_absolute_experiment_path

        assert _make_absolute_experiment_path("my_exp", None) == "my_exp"

    def test_absolute_path_ignores_workspace_path(self):
        from customer_retention.integrations.databricks_init import _make_absolute_experiment_path

        result = _make_absolute_experiment_path("/Users/other/exp", "Users/me/project")
        assert result == "/Users/other/exp"


class TestDatabricksInitMLflowConfiguration:
    @patch("customer_retention.integrations.databricks_init.mlflow", create=True)
    def test_calls_mlflow_set_experiment(self, mock_mlflow_module, monkeypatch, databricks_env):
        import customer_retention.integrations.databricks_init as mod

        mock_mlflow = MagicMock()
        monkeypatch.setattr(mod, "_configure_mlflow_experiment", lambda name: mock_mlflow.set_experiment(name))
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(experiment_name="test_exp", copy_notebooks=False)
        mock_mlflow.set_experiment.assert_called_once_with("test_exp")

    def test_custom_experiment_name_used(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(experiment_name="custom_exp", copy_notebooks=False)
        assert result.experiment_name == "custom_exp"

    def test_experiment_name_made_absolute_with_workspace_path(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(
            experiment_name="my_exp", workspace_path="Users/me/project", copy_notebooks=False,
        )
        assert result.experiment_name == "/Users/me/project/my_exp"

    def test_auto_resolves_experiment_name_from_notebook_path(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        mock_dbutils = MagicMock()
        (
            mock_dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get.return_value
        ) = "/Users/me/my_notebook"

        with patch("customer_retention.integrations.databricks_init._get_dbutils", return_value=mock_dbutils):
            result = databricks_init(copy_notebooks=False)
        assert result.experiment_name == "my_notebook"


class TestDatabricksInitNotebookCopy:
    def _mock_copy_notebooks(self, source_dir, dest_dir):
        mock_initializer_cls = MagicMock()
        mock_initializer_cls.return_value._get_exploration_source_dir.return_value = source_dir
        return mock_initializer_cls

    def test_copies_new_notebooks(self, monkeypatch, databricks_env, tmp_path):
        nb1 = _make_test_notebook([("c1", "code", "# @cr:code\nprint(1)")])
        nb2 = _make_test_notebook([("c1", "code", "# @cr:code\nprint(2)")])

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _write_notebook(source_dir / "notebook1.ipynb", nb1)
        _write_notebook(source_dir / "notebook2.ipynb", nb2)

        dest_dir = tmp_path / "Workspace" / "Users" / "me" / "project" / "exploration_notebooks"
        dest_dir.mkdir(parents=True)

        mock_cls = self._mock_copy_notebooks(source_dir, dest_dir)

        import customer_retention.integrations.databricks_init as mod

        real_path = mod.Path

        def redirect_path(p):
            if "/Workspace/" in str(p):
                return dest_dir
            return real_path(p)

        monkeypatch.setattr(mod, "Path", redirect_path)
        with patch.dict("sys.modules", {}):
            with patch(
                "customer_retention.generators.notebook_generator.project_init.ProjectInitializer",
                mock_cls,
            ):
                copied, synced = mod._sync_exploration_notebooks("Users/me/project")
        assert len(copied) == 2
        assert synced == []

    def test_skips_copy_when_disabled(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert result.notebooks_copied == []

    def test_syncs_existing_notebooks(self, monkeypatch, databricks_env, tmp_path):
        repo_nb = _make_test_notebook([
            ("cell1", "code", "# @cr:config\nFOO = 1"),
            ("cell2", "code", "# @cr:code\nnew_code()"),
        ])
        user_nb = _make_test_notebook([
            ("cell1", "code", "# @cr:config\nFOO = 42"),
            ("cell2", "code", "# @cr:code\nold_code()"),
        ])

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _write_notebook(source_dir / "existing.ipynb", repo_nb)

        dest_dir = tmp_path / "Workspace" / "Users" / "me" / "project" / "exploration_notebooks"
        dest_dir.mkdir(parents=True)
        _write_notebook(dest_dir / "existing.ipynb", user_nb)

        mock_cls = self._mock_copy_notebooks(source_dir, dest_dir)

        import customer_retention.integrations.databricks_init as mod

        real_path = mod.Path

        def redirect_path(p):
            if "/Workspace/" in str(p):
                return dest_dir
            return real_path(p)

        monkeypatch.setattr(mod, "Path", redirect_path)
        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer",
            mock_cls,
        ):
            copied, synced = mod._sync_exploration_notebooks("Users/me/project")
        assert copied == []
        assert len(synced) == 1

    def test_handles_missing_package_notebooks(self, monkeypatch, databricks_env):
        mock_cls = MagicMock()
        mock_cls.return_value._get_exploration_source_dir.return_value = None

        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer",
            mock_cls,
        ):
            from customer_retention.integrations.databricks_init import _sync_exploration_notebooks

            copied, synced = _sync_exploration_notebooks("Users/me/project")
        assert copied == []
        assert synced == []


class TestDatabricksInitResult:
    def test_result_contains_all_fields(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(
            catalog="analytics",
            schema="churn",
            experiment_name="exp",
            workspace_path="Users/me/project",
            model_name="my_model",
            copy_notebooks=False,
        )
        assert result.catalog == "analytics"
        assert result.schema == "churn"
        assert result.experiment_name == "/Users/me/project/exp"
        assert result.workspace_path == "Users/me/project"
        assert result.model_name == "my_model"

    def test_result_environment_variables_dict_complete(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(
            catalog="analytics",
            schema="churn",
            experiment_name="exp",
            workspace_path="Users/me/project",
            copy_notebooks=False,
        )
        env_vars = result.environment_variables
        assert env_vars["CR_CATALOG"] == "analytics"
        assert env_vars["CR_SCHEMA"] == "churn"
        assert env_vars["CR_EXPERIMENT_NAME"] == "/Users/me/project/exp"
        assert env_vars["CR_WORKSPACE_PATH"] == "Users/me/project"
        assert env_vars["CR_EXPERIMENTS_DIR"] == "/Volumes/analytics/churn/experiments"


class TestDatabricksInitDefaults:
    def test_default_catalog_is_main(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert result.catalog == "main"

    def test_default_schema_is_default(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert result.schema == "default"

    def test_default_model_name_is_customer_retention(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert result.model_name == "customer_retention"


class TestDatabricksInitCellScenario:
    def test_works_when_dbutils_global_unavailable(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        with patch("customer_retention.integrations.databricks_init._get_dbutils", return_value=None):
            result = databricks_init(copy_notebooks=False)
        assert result.experiment_name == "customer_retention"

    def test_falls_back_when_dbutils_chain_raises(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        mock_dbutils = MagicMock()
        mock_dbutils.notebook.entry_point.getDbutils.side_effect = AttributeError("no notebook context")

        with patch("customer_retention.integrations.databricks_init._get_dbutils", return_value=mock_dbutils):
            result = databricks_init(copy_notebooks=False)
        assert result.experiment_name == "customer_retention"

    def test_no_workspace_path_omits_path_env_vars(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert "CR_WORKSPACE_PATH" not in os.environ
        assert result.workspace_path is None

    def test_no_workspace_path_skips_notebook_copy(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(workspace_path=None, copy_notebooks=True)
        assert result.notebooks_copied == []

    def test_experiments_dir_set_without_workspace_path(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert result.environment_variables["CR_EXPERIMENTS_DIR"] == "/Volumes/main/default/experiments"

    def test_refreshes_config_module_constants(self, monkeypatch, databricks_env):
        import customer_retention.core.config.experiments as exp_module
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(catalog="churnkit", schema="prod", copy_notebooks=False)
        assert exp_module.EXPERIMENTS_DIR == Path("/Volumes/churnkit/prod/experiments")
        assert exp_module.FINDINGS_DIR == Path("/Volumes/churnkit/prod/experiments/findings")
        assert exp_module.CATALOG == "churnkit"
        assert exp_module.SCHEMA == "prod"

    def test_mlflow_import_missing_does_not_raise(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import _configure_mlflow_experiment

        with patch.dict("sys.modules", {"mlflow": None}):
            _configure_mlflow_experiment("test_exp")

    def test_display_summary_prints_output(self, monkeypatch, databricks_env, capsys):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(catalog="test_cat", schema="test_sch", copy_notebooks=False)
        captured = capsys.readouterr()
        assert "test_cat" in captured.out
        assert "test_sch" in captured.out
        assert "Initialization Complete" in captured.out

    def test_display_summary_includes_version(self, monkeypatch, databricks_env, capsys):
        from customer_retention import __version__
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(copy_notebooks=False)
        captured = capsys.readouterr()
        assert __version__ in captured.out


class TestDatabricksInitConfigPersistence:
    def test_persists_config_to_workspace(self, monkeypatch, databricks_env, tmp_path):
        config_file = tmp_path / ".churnkit_config.json"
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(
            catalog="churnkit", schema="analysis", workspace_path="Users/me/proj", copy_notebooks=False,
        )
        data = json.loads(config_file.read_text())
        assert data["experiments_dir"] == "/Volumes/churnkit/analysis/experiments"
        assert data["catalog"] == "churnkit"
        assert data["schema"] == "analysis"

    def test_persisted_config_survives_for_subsequent_import(self, monkeypatch, databricks_env, tmp_path):
        config_file = tmp_path / ".churnkit_config.json"
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(catalog="churnkit", schema="prod", workspace_path="Users/me/proj", copy_notebooks=False)
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        from customer_retention.core.config.experiments import get_experiments_dir

        assert str(get_experiments_dir()) == "/Volumes/churnkit/prod/experiments"

    def test_no_config_persisted_without_workspace_path(self, monkeypatch, databricks_env, tmp_path):
        config_file = tmp_path / ".churnkit_config.json"
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(copy_notebooks=False)
        assert not config_file.exists()


class TestDatabricksInitExperimentStructure:
    def test_invokes_setup_experiments_structure(self, monkeypatch, databricks_env):
        with patch("customer_retention.core.config.experiments.setup_experiments_structure") as mock_setup:
            from customer_retention.integrations.databricks_init import databricks_init

            databricks_init(copy_notebooks=False)
        mock_setup.assert_called_once()

    def test_handles_oserror_gracefully(self, monkeypatch, databricks_env):
        with patch(
            "customer_retention.core.config.experiments.setup_experiments_structure", side_effect=OSError("read-only"),
        ):
            from customer_retention.integrations.databricks_init import databricks_init

            databricks_init(copy_notebooks=False)


class TestEnsureExperimentsVolumeExists:
    def test_creates_volume_via_spark_sql(self, databricks_env):
        mock_spark = MagicMock()
        with patch("customer_retention.core.compat.detection.get_spark_session", return_value=mock_spark):
            from customer_retention.integrations.databricks_init import _ensure_experiments_volume_exists

            _ensure_experiments_volume_exists("churnkit", "analysis")
        mock_spark.sql.assert_called_once_with("CREATE VOLUME IF NOT EXISTS churnkit.analysis.experiments")

    def test_handles_no_spark_session(self, databricks_env):
        with patch("customer_retention.core.compat.detection.get_spark_session", return_value=None):
            from customer_retention.integrations.databricks_init import _ensure_experiments_volume_exists

            _ensure_experiments_volume_exists("churnkit", "analysis")

    def test_handles_spark_sql_error(self, databricks_env):
        mock_spark = MagicMock()
        mock_spark.sql.side_effect = Exception("permission denied")
        with patch("customer_retention.core.compat.detection.get_spark_session", return_value=mock_spark):
            from customer_retention.integrations.databricks_init import _ensure_experiments_volume_exists

            _ensure_experiments_volume_exists("churnkit", "analysis")

    def test_called_during_init(self, databricks_env):
        with patch("customer_retention.integrations.databricks_init._ensure_experiments_volume_exists") as mock_ensure:
            from customer_retention.integrations.databricks_init import databricks_init

            databricks_init(catalog="churnkit", schema="analysis", copy_notebooks=False)
        mock_ensure.assert_called_once_with("churnkit", "analysis")


class TestEnsureWorkspaceDirectory:
    def test_creates_workspace_directory(self, monkeypatch, databricks_env, tmp_path):
        import customer_retention.integrations.databricks_init as mod

        real_path = mod.Path

        def redirect_path(p):
            if "/Workspace/" in str(p):
                return tmp_path / "workspace_dir"
            return real_path(p)

        monkeypatch.setattr(mod, "Path", redirect_path)
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: tmp_path / ".churnkit_config.json",
        )
        mod.databricks_init(workspace_path="Users/me/project", copy_notebooks=False)
        assert (tmp_path / "workspace_dir").exists()

    def test_no_workspace_path_skips_directory_creation(self, monkeypatch, databricks_env):
        with patch("customer_retention.integrations.databricks_init._ensure_workspace_directory") as mock_ensure:
            from customer_retention.integrations.databricks_init import databricks_init

            databricks_init(copy_notebooks=False)
        mock_ensure.assert_not_called()


def _make_test_notebook(cells):
    nb = nbformat.v4.new_notebook()
    for cell_id, cell_type, source in cells:
        if cell_type == "code":
            cell = nbformat.v4.new_code_cell(source=source)
        else:
            cell = nbformat.v4.new_markdown_cell(source=source)
        cell.id = cell_id
        nb.cells.append(cell)
    return nb


def _write_notebook(path, nb):
    with open(path, "w") as f:
        nbformat.write(nb, f)


def _read_notebook(path):
    with open(path) as f:
        return nbformat.read(f, as_version=4)


class TestDatabricksInitNotebookSync:
    def _setup_sync(self, tmp_path, repo_cells, user_cells, notebook_name="test.ipynb"):
        repo_nb = _make_test_notebook(repo_cells)
        user_nb = _make_test_notebook(user_cells)

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _write_notebook(source_dir / notebook_name, repo_nb)

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir(parents=True)
        _write_notebook(dest_dir / notebook_name, user_nb)

        return source_dir, dest_dir

    def _run_sync(self, monkeypatch, source_dir, dest_dir):
        mock_cls = MagicMock()
        mock_cls.return_value._get_exploration_source_dir.return_value = source_dir

        import customer_retention.integrations.databricks_init as mod

        real_path = mod.Path

        def redirect_path(p):
            if "/Workspace/" in str(p):
                return dest_dir
            return real_path(p)

        monkeypatch.setattr(mod, "Path", redirect_path)
        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer",
            mock_cls,
        ):
            return mod._sync_exploration_notebooks("Users/me/project")

    def test_updates_code_cells(self, monkeypatch, databricks_env, tmp_path):
        repo_cells = [("cell1", "code", "# @cr:code\nnew_code()")]
        user_cells = [("cell1", "code", "# @cr:code\nold_code()")]
        source_dir, dest_dir = self._setup_sync(tmp_path, repo_cells, user_cells)

        copied, synced = self._run_sync(monkeypatch, source_dir, dest_dir)

        assert copied == []
        assert len(synced) == 1
        result_nb = _read_notebook(dest_dir / "test.ipynb")
        assert "new_code()" in result_nb.cells[0].source

    def test_preserves_config_cells(self, monkeypatch, databricks_env, tmp_path):
        repo_cells = [
            ("cell1", "code", "# @cr:config\nFOO = 'default'"),
            ("cell2", "code", "# @cr:code\nrun()"),
        ]
        user_cells = [
            ("cell1", "code", "# @cr:config\nFOO = 'custom'"),
            ("cell2", "code", "# @cr:code\nrun()"),
        ]
        source_dir, dest_dir = self._setup_sync(tmp_path, repo_cells, user_cells)

        self._run_sync(monkeypatch, source_dir, dest_dir)

        result_nb = _read_notebook(dest_dir / "test.ipynb")
        assert "FOO = 'custom'" in result_nb.cells[0].source

    def test_preserves_user_code_cells(self, monkeypatch, databricks_env, tmp_path):
        repo_cells = [
            ("cell1", "code", "# @cr:user_code\n# add your code here"),
            ("cell2", "code", "# @cr:code\nrun()"),
        ]
        user_cells = [
            ("cell1", "code", "# @cr:user_code\nmy_custom_analysis()"),
            ("cell2", "code", "# @cr:code\nrun()"),
        ]
        source_dir, dest_dir = self._setup_sync(tmp_path, repo_cells, user_cells)

        self._run_sync(monkeypatch, source_dir, dest_dir)

        result_nb = _read_notebook(dest_dir / "test.ipynb")
        assert "my_custom_analysis()" in result_nb.cells[0].source

    def test_no_write_when_unchanged(self, monkeypatch, databricks_env, tmp_path):
        cells = [("cell1", "code", "# @cr:code\nsame_code()")]
        source_dir, dest_dir = self._setup_sync(tmp_path, cells, cells)
        dest_file = dest_dir / "test.ipynb"
        mtime_before = dest_file.stat().st_mtime

        self._run_sync(monkeypatch, source_dir, dest_dir)

        assert dest_file.stat().st_mtime == mtime_before

    def test_no_write_reports_empty_synced(self, monkeypatch, databricks_env, tmp_path):
        cells = [("cell1", "code", "# @cr:code\nsame_code()")]
        source_dir, dest_dir = self._setup_sync(tmp_path, cells, cells)

        copied, synced = self._run_sync(monkeypatch, source_dir, dest_dir)

        assert copied == []
        assert synced == []

    def test_adds_new_cells_from_repo(self, monkeypatch, databricks_env, tmp_path):
        repo_cells = [
            ("cell1", "code", "# @cr:code\noriginal()"),
            ("cell2", "code", "# @cr:code\nnew_feature()"),
        ]
        user_cells = [("cell1", "code", "# @cr:code\noriginal()")]
        source_dir, dest_dir = self._setup_sync(tmp_path, repo_cells, user_cells)

        copied, synced = self._run_sync(monkeypatch, source_dir, dest_dir)

        assert len(synced) == 1
        result_nb = _read_notebook(dest_dir / "test.ipynb")
        assert len(result_nb.cells) == 2
        assert "new_feature()" in result_nb.cells[1].source

    def test_result_has_notebooks_synced(self, monkeypatch, databricks_env, tmp_path):
        repo_nb = _make_test_notebook([("c1", "code", "# @cr:code\nnew()")])
        user_nb = _make_test_notebook([("c1", "code", "# @cr:code\nold()")])

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _write_notebook(source_dir / "nb.ipynb", repo_nb)

        dest_dir = tmp_path / "dest" / "exploration_notebooks"
        dest_dir.mkdir(parents=True)
        _write_notebook(dest_dir / "nb.ipynb", user_nb)

        mock_cls = MagicMock()
        mock_cls.return_value._get_exploration_source_dir.return_value = source_dir

        import customer_retention.integrations.databricks_init as mod

        real_path = mod.Path

        def redirect_path(p):
            if "/Workspace/" in str(p):
                return dest_dir
            return real_path(p)

        monkeypatch.setattr(mod, "Path", redirect_path)
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: tmp_path / ".churnkit_config.json",
        )

        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer",
            mock_cls,
        ):
            result = mod.databricks_init(
                workspace_path="Users/me/project", copy_notebooks=True,
            )

        assert len(result.notebooks_synced) == 1

    def test_display_summary_shows_synced(self, monkeypatch, databricks_env, tmp_path, capsys):
        repo_nb = _make_test_notebook([("c1", "code", "# @cr:code\nnew()")])
        user_nb = _make_test_notebook([("c1", "code", "# @cr:code\nold()")])

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _write_notebook(source_dir / "nb.ipynb", repo_nb)

        dest_dir = tmp_path / "dest" / "exploration_notebooks"
        dest_dir.mkdir(parents=True)
        _write_notebook(dest_dir / "nb.ipynb", user_nb)

        mock_cls = MagicMock()
        mock_cls.return_value._get_exploration_source_dir.return_value = source_dir

        import customer_retention.integrations.databricks_init as mod

        real_path = mod.Path

        def redirect_path(p):
            if "/Workspace/" in str(p):
                return dest_dir
            return real_path(p)

        monkeypatch.setattr(mod, "Path", redirect_path)
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: tmp_path / ".churnkit_config.json",
        )

        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer",
            mock_cls,
        ):
            mod.databricks_init(workspace_path="Users/me/project", copy_notebooks=True)

        captured = capsys.readouterr()
        assert "Synced" in captured.out

    def test_syncs_via_embedded_ids_when_nbformat_ids_stripped(self, monkeypatch, databricks_env, tmp_path):
        repo_nb = _make_test_notebook([
            ("cell1", "code", "# @cr:config name='settings' id=a1b2c3d4\nFOO = 'default'"),
            ("cell2", "code", "# @cr:code name='run' id=e5f6a7b8\nnew_code()"),
        ])
        user_nb = nbformat.v4.new_notebook()
        c1 = nbformat.v4.new_code_cell(source="# @cr:config name='settings' id=a1b2c3d4\nFOO = 'custom'")
        c2 = nbformat.v4.new_code_cell(source="# @cr:code name='run' id=e5f6a7b8\nold_code()")
        if "id" in c1:
            del c1["id"]
        if "id" in c2:
            del c2["id"]
        user_nb.cells.extend([c1, c2])

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _write_notebook(source_dir / "test.ipynb", repo_nb)

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir(parents=True)
        _write_notebook(dest_dir / "test.ipynb", user_nb)

        copied, synced = self._run_sync(monkeypatch, source_dir, dest_dir)

        assert copied == []
        assert len(synced) == 1
        result_nb = _read_notebook(dest_dir / "test.ipynb")
        assert "FOO = 'custom'" in result_nb.cells[0].source
        assert "new_code()" in result_nb.cells[1].source

    def test_handles_corrupt_user_notebook(self, monkeypatch, databricks_env, tmp_path):
        repo_nb = _make_test_notebook([("c1", "code", "# @cr:code\nrun()")])

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _write_notebook(source_dir / "test.ipynb", repo_nb)

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir(parents=True)
        (dest_dir / "test.ipynb").write_text("not valid json{{{")

        copied, synced = self._run_sync(monkeypatch, source_dir, dest_dir)

        assert copied == []
        assert synced == []
