import os
from pathlib import Path
from typing import Optional


def _find_project_root() -> Path:
    path = Path(__file__).parent
    for _ in range(10):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
        path = path.parent
    return Path.cwd()


def get_experiments_dir(default: Optional[str] = None) -> Path:
    if "CR_EXPERIMENTS_DIR" in os.environ:
        return Path(os.environ["CR_EXPERIMENTS_DIR"])
    if default:
        return Path(default)
    return _find_project_root() / "experiments"


def get_findings_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "findings"


def get_data_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "data"


def get_mlruns_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "mlruns"


def get_feature_store_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "feature_repo"


def get_catalog(default: str = "main") -> str:
    return os.environ.get("CR_CATALOG", default)


def get_schema(default: str = "default") -> str:
    return os.environ.get("CR_SCHEMA", default)


def get_workspace_path(default: str | None = None) -> str | None:
    return os.environ.get("CR_WORKSPACE_PATH", default)


def get_experiment_name(default: str = "customer_retention") -> str:
    return os.environ.get("CR_EXPERIMENT_NAME", default)


EXPERIMENTS_DIR = get_experiments_dir()
FINDINGS_DIR = get_findings_dir()
DATA_DIR = get_data_dir()
MLRUNS_DIR = get_mlruns_dir()
FEATURE_STORE_DIR = get_feature_store_dir()
OUTPUT_DIR = FINDINGS_DIR
CATALOG = get_catalog()
SCHEMA = get_schema()
WORKSPACE_PATH = get_workspace_path()
EXPERIMENT_NAME = get_experiment_name()


def setup_experiments_structure(experiments_dir: Optional[Path] = None) -> None:
    base = experiments_dir or get_experiments_dir()
    directories = [
        base / "findings" / "snapshots",
        base / "findings" / "unified",
        base / "data" / "bronze",
        base / "data" / "silver",
        base / "data" / "gold",
        base / "data" / "scoring",
        base / "mlruns",
        base / "feature_repo" / "data",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_notebook_experiments_dir() -> Path:
    if "CR_EXPERIMENTS_DIR" in os.environ:
        return Path(os.environ["CR_EXPERIMENTS_DIR"])
    cwd = Path.cwd()
    if (cwd.parent / "experiments").exists():
        return cwd.parent / "experiments"
    elif (cwd / "experiments").exists():
        return cwd / "experiments"
    return get_experiments_dir()
