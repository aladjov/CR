import json
from unittest.mock import MagicMock, patch


class TestGetCatalog:
    def test_returns_env_var_when_set(self, monkeypatch):
        from customer_retention.core.config.experiments import get_catalog

        monkeypatch.setenv("CR_CATALOG", "analytics")
        assert get_catalog() == "analytics"

    def test_returns_default_when_not_set(self, monkeypatch):
        from customer_retention.core.config.experiments import get_catalog

        monkeypatch.delenv("CR_CATALOG", raising=False)
        assert get_catalog() == "main"

    def test_returns_custom_default(self, monkeypatch):
        from customer_retention.core.config.experiments import get_catalog

        monkeypatch.delenv("CR_CATALOG", raising=False)
        assert get_catalog(default="production") == "production"


class TestGetSchema:
    def test_returns_env_var_when_set(self, monkeypatch):
        from customer_retention.core.config.experiments import get_schema

        monkeypatch.setenv("CR_SCHEMA", "churn")
        assert get_schema() == "churn"

    def test_returns_default_when_not_set(self, monkeypatch):
        from customer_retention.core.config.experiments import get_schema

        monkeypatch.delenv("CR_SCHEMA", raising=False)
        assert get_schema() == "default"


class TestGetWorkspacePath:
    def test_returns_env_var_when_set(self, monkeypatch):
        from customer_retention.core.config.experiments import get_workspace_path

        monkeypatch.setenv("CR_WORKSPACE_PATH", "/Workspace/Users/me/project")
        assert get_workspace_path() == "/Workspace/Users/me/project"

    def test_returns_none_when_not_set(self, monkeypatch):
        from customer_retention.core.config.experiments import get_workspace_path

        monkeypatch.delenv("CR_WORKSPACE_PATH", raising=False)
        assert get_workspace_path() is None

    def test_returns_custom_default(self, monkeypatch):
        from customer_retention.core.config.experiments import get_workspace_path

        monkeypatch.delenv("CR_WORKSPACE_PATH", raising=False)
        assert get_workspace_path(default="/Workspace/shared") == "/Workspace/shared"


class TestGetExperimentName:
    def test_returns_env_var_when_set(self, monkeypatch):
        from customer_retention.core.config.experiments import get_experiment_name

        monkeypatch.setenv("CR_EXPERIMENT_NAME", "my_experiment")
        assert get_experiment_name() == "my_experiment"

    def test_returns_default_when_not_set(self, monkeypatch):
        from customer_retention.core.config.experiments import get_experiment_name

        monkeypatch.delenv("CR_EXPERIMENT_NAME", raising=False)
        assert get_experiment_name() == "customer_retention"

    def test_reads_from_persisted_config(self, tmp_path, monkeypatch):
        from customer_retention.core.config.experiments import get_experiment_name

        monkeypatch.delenv("CR_EXPERIMENT_NAME", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me/project")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({"experiment_name": "/Users/me/project/my_exp"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        assert get_experiment_name() == "/Users/me/project/my_exp"

    def test_env_var_takes_precedence_over_persisted(self, tmp_path, monkeypatch):
        from customer_retention.core.config.experiments import get_experiment_name

        monkeypatch.setenv("CR_EXPERIMENT_NAME", "from_env")
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me/project")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({"experiment_name": "/Users/me/project/persisted"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        assert get_experiment_name() == "from_env"


class TestGetMlflowDfsTmpdir:
    def test_returns_uc_volume_path_on_databricks(self, monkeypatch):
        from customer_retention.core.config.experiments import get_mlflow_dfs_tmpdir

        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        monkeypatch.setenv("CR_CATALOG", "analytics")
        monkeypatch.setenv("CR_SCHEMA", "churn")
        mock_spark = MagicMock()
        with patch("customer_retention.core.compat.detection.get_spark_session", return_value=mock_spark):
            result = get_mlflow_dfs_tmpdir()
        assert result == "/Volumes/analytics/churn/mlflow_tmp"
        mock_spark.sql.assert_called_once_with(
            "CREATE VOLUME IF NOT EXISTS `analytics`.`churn`.`mlflow_tmp`"
        )

    def test_returns_none_outside_databricks(self, monkeypatch):
        from customer_retention.core.config.experiments import get_mlflow_dfs_tmpdir

        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        assert get_mlflow_dfs_tmpdir() is None

    def test_uses_default_catalog_schema(self, monkeypatch):
        from customer_retention.core.config.experiments import get_mlflow_dfs_tmpdir

        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        monkeypatch.delenv("CR_CATALOG", raising=False)
        monkeypatch.delenv("CR_SCHEMA", raising=False)
        mock_spark = MagicMock()
        with patch("customer_retention.core.compat.detection.get_spark_session", return_value=mock_spark):
            result = get_mlflow_dfs_tmpdir()
        assert result == "/Volumes/main/default/mlflow_tmp"

    def test_respects_env_override(self, monkeypatch):
        from customer_retention.core.config.experiments import get_mlflow_dfs_tmpdir

        monkeypatch.setenv("MLFLOW_DFS_TMP", "/Volumes/custom/path/vol")
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        assert get_mlflow_dfs_tmpdir() == "/Volumes/custom/path/vol"
