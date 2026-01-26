import os
from pathlib import Path
from unittest.mock import patch

from customer_retention.core.config.experiments import (
    DATA_DIR,
    EXPERIMENTS_DIR,
    FEATURE_STORE_DIR,
    FINDINGS_DIR,
    MLRUNS_DIR,
    OUTPUT_DIR,
    _find_project_root,
    get_data_dir,
    get_experiments_dir,
    get_feature_store_dir,
    get_findings_dir,
    get_mlruns_dir,
    get_notebook_experiments_dir,
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


class TestSetupExperimentsStructure:
    def test_creates_all_directories(self, tmp_path):
        setup_experiments_structure(tmp_path)
        expected = [
            "findings/snapshots",
            "findings/unified",
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
        assert (tmp_path / "auto" / "findings" / "snapshots").exists()
        monkeypatch.delenv("CR_EXPERIMENTS_DIR")
        importlib.reload(exp_module)

    def test_idempotent_creation(self, tmp_path):
        setup_experiments_structure(tmp_path)
        setup_experiments_structure(tmp_path)
        assert (tmp_path / "findings" / "snapshots").exists()


class TestGetNotebookExperimentsDir:
    def test_uses_env_var_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", str(tmp_path / "from_env"))
        result = get_notebook_experiments_dir()
        assert str(result) == str(tmp_path / "from_env")

    def test_finds_experiments_in_parent_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        (tmp_path / "experiments").mkdir()
        notebook_dir = tmp_path / "notebooks"
        notebook_dir.mkdir()
        monkeypatch.chdir(notebook_dir)
        result = get_notebook_experiments_dir()
        assert result == tmp_path / "experiments"

    def test_finds_experiments_in_current_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        (tmp_path / "experiments").mkdir()
        monkeypatch.chdir(tmp_path)
        result = get_notebook_experiments_dir()
        assert result == tmp_path / "experiments"

    def test_falls_back_to_get_experiments_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_EXPERIMENTS_DIR", raising=False)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)
        result = get_notebook_experiments_dir()
        assert result.name == "experiments"
