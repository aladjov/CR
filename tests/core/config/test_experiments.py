import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from customer_retention.core.config.experiments import (
    DATA_DIR,
    EXPERIMENTS_DIR,
    FEATURE_STORE_DIR,
    FINDINGS_DIR,
    MLRUNS_DIR,
    OUTPUT_DIR,
    _find_project_root,
    _load_persisted_databricks_config,
    get_active_run_dir,
    get_data_dir,
    get_experiments_dir,
    get_feature_store_dir,
    get_findings_dir,
    get_mlruns_dir,
    get_runs_dir,
    persist_databricks_config,
    setup_experiments_structure,
)


class TestFindProjectRoot:
    def test_finds_project_root_with_pyproject(self):
        root = _find_project_root()
        assert (root / "pyproject.toml").exists()

    def test_finds_project_root_with_git(self):
        root = _find_project_root()
        assert (root / ".git").exists() or (root / "pyproject.toml").exists()

    def test_fallback_to_cwd_when_no_markers(self, tmp_path, monkeypatch):
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "h" / "i" / "j" / "k"
        deep_path.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        fake_module_path = deep_path / "fake.py"
        fake_module_path.touch()
        with patch(
            "customer_retention.core.config.experiments.__file__",
            str(fake_module_path),
        ):
            import importlib

            import customer_retention.core.config.experiments as exp_module

            importlib.reload(exp_module)
            result = exp_module._find_project_root()
            assert result == tmp_path or result.exists()
            importlib.reload(exp_module)


class TestGetExperimentsDir:
    def test_returns_default_experiments_dir(self):
        with patch.dict(os.environ, {}, clear=False):
            if "CR_EXPERIMENTS_DIR" in os.environ:
                del os.environ["CR_EXPERIMENTS_DIR"]
            result = get_experiments_dir()
            assert result.name == "experiments"

    def test_uses_env_var_override(self, tmp_path, monkeypatch):
        custom = str(tmp_path / "custom_exp")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", custom)
        result = get_experiments_dir()
        assert str(result) == custom

    def test_uses_default_parameter(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_experiments_dir(default="/my/custom/path")
        assert str(result) == "/my/custom/path"

    def test_env_var_takes_precedence_over_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path / "from_env"))
        result = get_experiments_dir(default="/should/be/ignored")
        assert str(result) == str(tmp_path / "from_env")


class TestGetFindingsDir:
    def test_returns_findings_subdir(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_findings_dir()
        assert result.name == "findings"
        assert result.parent.name == "experiments"

    def test_uses_default_parameter(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_findings_dir(default="/custom/exp")
        assert str(result) == "/custom/exp/findings"


class TestGetDataDir:
    def test_returns_data_subdir(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_data_dir()
        assert result.name == "data"

    def test_uses_default_parameter(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_data_dir(default="/custom/exp")
        assert str(result) == "/custom/exp/data"


class TestGetMlrunsDir:
    def test_returns_mlruns_subdir(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_mlruns_dir()
        assert result.name == "mlruns"

    def test_uses_default_parameter(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_mlruns_dir(default="/custom/exp")
        assert str(result) == "/custom/exp/mlruns"


class TestGetFeatureStoreDir:
    def test_returns_feature_repo_subdir(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_feature_store_dir()
        assert result.name == "feature_repo"

    def test_uses_default_parameter(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_feature_store_dir(default="/custom/exp")
        assert str(result) == "/custom/exp/feature_repo"


class TestModuleLevelConstants:
    def test_experiments_dir_is_path(self):
        assert isinstance(EXPERIMENTS_DIR, Path)

    def test_findings_dir_is_path(self):
        assert isinstance(FINDINGS_DIR, Path)

    def test_data_dir_is_path(self):
        assert isinstance(DATA_DIR, Path)

    def test_mlruns_dir_is_path(self):
        assert isinstance(MLRUNS_DIR, Path)

    def test_feature_store_dir_is_path(self):
        assert isinstance(FEATURE_STORE_DIR, Path)

    def test_output_dir_equals_findings_dir(self):
        assert OUTPUT_DIR == FINDINGS_DIR


class TestReloadConfig:
    def test_refreshes_experiments_dir_from_env(self, tmp_path, monkeypatch):
        import customer_retention.core.config.experiments as exp_module

        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path / "refreshed"))
        exp_module.reload_config()
        assert exp_module.EXPERIMENTS_DIR == tmp_path / "refreshed"
        assert exp_module.FINDINGS_DIR == tmp_path / "refreshed" / "findings"
        assert exp_module.DATA_DIR == tmp_path / "refreshed" / "data"
        assert exp_module.MLRUNS_DIR == tmp_path / "refreshed" / "mlruns"
        assert exp_module.FEATURE_STORE_DIR == tmp_path / "refreshed" / "feature_repo"
        assert exp_module.OUTPUT_DIR == exp_module.FINDINGS_DIR

    def test_refreshes_catalog_and_schema_from_env(self, monkeypatch):
        import customer_retention.core.config.experiments as exp_module

        monkeypatch.setenv("CR_CATALOG", "my_catalog")
        monkeypatch.setenv("CR_SCHEMA", "my_schema")
        exp_module.reload_config()
        assert exp_module.CATALOG == "my_catalog"
        assert exp_module.SCHEMA == "my_schema"


class TestSetupExperimentsStructure:
    def test_creates_all_directories(self, tmp_path):
        setup_experiments_structure(tmp_path)
        expected = [
            "data/bronze",
            "data/silver",
            "data/gold",
            "data/scoring",
            "mlruns",
            "feature_repo/data",
        ]
        for subdir in expected:
            assert (tmp_path / subdir).exists(), f"Missing: {subdir}"

    def test_uses_get_experiments_dir_when_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path / "auto"))
        import importlib

        import customer_retention.core.config.experiments as exp_module

        importlib.reload(exp_module)
        exp_module.setup_experiments_structure()
        assert (tmp_path / "auto" / "data" / "bronze").exists()
        monkeypatch.delenv("CR_EXPERIMENTS_DIR")
        importlib.reload(exp_module)

    def test_idempotent_creation(self, tmp_path):
        setup_experiments_structure(tmp_path)
        setup_experiments_structure(tmp_path)
        assert (tmp_path / "data" / "bronze").exists()



class TestPersistedDatabricksConfig:
    def test_load_returns_none_outside_databricks(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        assert _load_persisted_databricks_config() is None

    def test_load_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me/project")
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: tmp_path / "nonexistent.json",
        )
        assert _load_persisted_databricks_config() is None

    def test_load_reads_via_workspace_path_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me/project")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({
            "experiments_dir": "/Volumes/cat/sch/experiments",
            "catalog": "cat", "schema": "sch",
        }))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        result = _load_persisted_databricks_config()
        assert result["experiments_dir"] == "/Volumes/cat/sch/experiments"
        assert result["catalog"] == "cat"

    def test_load_walks_up_from_cwd(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.delenv("CR_WORKSPACE_PATH", raising=False)
        project_dir = tmp_path / "project"
        notebooks_dir = project_dir / "exploration_notebooks"
        notebooks_dir.mkdir(parents=True)
        (project_dir / ".churnkit_config.json").write_text(json.dumps({
            "experiments_dir": "/Volumes/cat/sch/experiments",
        }))
        monkeypatch.chdir(notebooks_dir)
        result = _load_persisted_databricks_config()
        assert result["experiments_dir"] == "/Volumes/cat/sch/experiments"

    def test_load_returns_none_on_corrupt_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text("not json{{{")
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        assert _load_persisted_databricks_config() is None

    def test_persist_writes_config_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".churnkit_config.json"
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        persist_databricks_config("/Volumes/mycat/mysch/experiments", "mycat", "mysch", "Users/me")
        data = json.loads(config_file.read_text())
        assert data["experiments_dir"] == "/Volumes/mycat/mysch/experiments"
        assert data["catalog"] == "mycat"
        assert data["schema"] == "mysch"

    def test_persist_skips_without_workspace_path(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".churnkit_config.json"
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        persist_databricks_config("/Volumes/c/s/experiments", "c", "s")
        assert not config_file.exists()

    def test_persist_overwrites_existing(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({"experiments_dir": "/old"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        persist_databricks_config("/Volumes/new/sch/experiments", "new", "sch", "Users/me")
        data = json.loads(config_file.read_text())
        assert data["experiments_dir"] == "/Volumes/new/sch/experiments"

    def test_persist_handles_write_error_gracefully(self, monkeypatch):
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: Path("/nonexistent_dir/subdir/config.json"),
        )
        persist_databricks_config("/Volumes/c/s/experiments", "c", "s", "Users/me")

    def test_persist_treats_empty_string_workspace_as_no_workspace(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".churnkit_config.json"
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        persist_databricks_config("/Volumes/c/s/experiments", "c", "s", "")
        assert not config_file.exists()

    def test_persist_propagates_non_os_exception(self, monkeypatch):
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: Path("/nonexistent_dir/subdir/deep/config.json"),
        )
        with patch("json.dumps", side_effect=RuntimeError("Py4J error")):
            with pytest.raises(RuntimeError, match="Py4J error"):
                persist_databricks_config("/Volumes/c/s/experiments", "c", "s", "Users/me")

    def test_read_config_propagates_non_json_os_exception(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value"}')
        from customer_retention.core.config.experiments import _read_config_file
        with patch("json.loads", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError, match="unexpected"):
                _read_config_file(config_file)

    def test_load_returns_none_when_config_missing_experiments_dir_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({"catalog": "cat", "schema": "sch"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        result = _load_persisted_databricks_config()
        assert result is not None
        assert "experiments_dir" not in result

    def test_load_cwd_walk_returns_none_when_no_config_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.delenv("CR_WORKSPACE_PATH", raising=False)
        empty_dir = tmp_path / "deep" / "nested" / "dir"
        empty_dir.mkdir(parents=True)
        monkeypatch.chdir(empty_dir)
        assert _load_persisted_databricks_config() is None

    def test_get_experiments_dir_ignores_config_without_experiments_dir_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({"catalog": "cat"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        result = get_experiments_dir()
        assert result.name == "experiments"


class TestGetExperimentsDirWithPersistedConfig:
    def test_env_var_takes_precedence_over_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path / "from_env"))
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me/project")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({"experiments_dir": "/Volumes/other/path/experiments"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        assert str(get_experiments_dir()) == str(tmp_path / "from_env")

    def test_uses_persisted_config_when_no_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me/project")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({"experiments_dir": "/Volumes/cat/sch/experiments"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        assert str(get_experiments_dir()) == "/Volumes/cat/sch/experiments"

    def test_default_param_takes_precedence_over_persisted(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me/project")
        config_file = tmp_path / ".churnkit_config.json"
        config_file.write_text(json.dumps({"experiments_dir": "/Volumes/cat/sch/experiments"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        assert str(get_experiments_dir(default="/explicit/default")) == "/explicit/default"

    def test_falls_back_to_project_root_when_no_persisted_config(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        result = get_experiments_dir()
        assert result.name == "experiments"


class TestGetRunsDir:
    def test_returns_runs_subdir_of_experiments(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_runs_dir()
        assert result.name == "runs"
        assert result.parent.name == "experiments"

    def test_uses_default_parameter(self, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_runs_dir(default="/custom/exp")
        assert str(result) == "/custom/exp/runs"

    def test_uses_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path / "exp"))
        result = get_runs_dir()
        assert result == tmp_path / "exp" / "runs"


class TestGetActiveRunDir:
    def test_returns_none_without_env_var(self, monkeypatch):
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        assert get_active_run_dir() is None

    def test_returns_run_dir_with_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_RUN_ID", "my-run-12345678")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path / "exp"))
        result = get_active_run_dir()
        assert result == tmp_path / "exp" / "runs" / "my-run-12345678"

    def test_run_id_preserved_in_path(self, monkeypatch):
        monkeypatch.setenv("CR_RUN_ID", "retention-abcd1234")
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        result = get_active_run_dir()
        assert result.name == "retention-abcd1234"
        assert result.parent.name == "runs"
