import os
from unittest.mock import MagicMock, patch

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

    def test_sets_cr_experiments_dir_from_workspace_path(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        databricks_init(workspace_path="Users/me/project", copy_notebooks=False)
        assert os.environ["CR_EXPERIMENTS_DIR"] == "/Workspace/Users/me/project/experiments"

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

    def test_copies_notebooks_when_enabled(self, monkeypatch, databricks_env, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "notebook1.ipynb").write_text("{}")
        (source_dir / "notebook2.ipynb").write_text("{}")

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
                copied = mod._copy_exploration_notebooks("Users/me/project")
        assert len(copied) == 2

    def test_skips_copy_when_disabled(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert result.notebooks_copied == []

    def test_skips_existing_notebooks(self, monkeypatch, databricks_env, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "existing.ipynb").write_text("{}")

        dest_dir = tmp_path / "Workspace" / "Users" / "me" / "project" / "exploration_notebooks"
        dest_dir.mkdir(parents=True)
        (dest_dir / "existing.ipynb").write_text("{}")

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
            copied = mod._copy_exploration_notebooks("Users/me/project")
        assert copied == []

    def test_handles_missing_package_notebooks(self, monkeypatch, databricks_env):
        mock_cls = MagicMock()
        mock_cls.return_value._get_exploration_source_dir.return_value = None

        with patch(
            "customer_retention.generators.notebook_generator.project_init.ProjectInitializer",
            mock_cls,
        ):
            from customer_retention.integrations.databricks_init import _copy_exploration_notebooks

            copied = _copy_exploration_notebooks("Users/me/project")
        assert copied == []


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
        assert result.experiment_name == "exp"
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
        assert env_vars["CR_EXPERIMENT_NAME"] == "exp"
        assert env_vars["CR_WORKSPACE_PATH"] == "Users/me/project"
        assert "CR_EXPERIMENTS_DIR" in env_vars


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

    def test_result_env_vars_empty_experiments_dir_without_workspace(self, monkeypatch, databricks_env):
        from customer_retention.integrations.databricks_init import databricks_init

        result = databricks_init(copy_notebooks=False)
        assert result.environment_variables["CR_EXPERIMENTS_DIR"] == ""

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
