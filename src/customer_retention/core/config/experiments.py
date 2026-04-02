import json
import os
from pathlib import Path
from typing import Optional, Union

from customer_retention.core.compat.remote_path import RemotePath, make_path

_DATABRICKS_CONFIG_FILENAME = ".churnkit_config.json"


def _workspace_config_path(workspace_path: str) -> Path:
    return Path(f"/Workspace/{workspace_path}") / _DATABRICKS_CONFIG_FILENAME


def _read_config_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def _load_persisted_databricks_config() -> dict | None:
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return None
    workspace_path = os.environ.get("CR_WORKSPACE_PATH")
    if workspace_path:
        return _read_config_file(_workspace_config_path(workspace_path))
    cwd = Path.cwd()
    for _ in range(5):
        result = _read_config_file(cwd / _DATABRICKS_CONFIG_FILENAME)
        if result:
            return result
        if cwd.parent == cwd:
            break
        cwd = cwd.parent
    return None


def persist_databricks_config(
    experiments_dir: str, catalog: str, schema: str,
    workspace_path: str | None = None, experiment_name: str | None = None,
    framework_repo_path: str | None = None,
) -> None:
    if not workspace_path:
        return
    data: dict = {"experiments_dir": experiments_dir, "catalog": catalog, "schema": schema}
    if experiment_name:
        data["experiment_name"] = experiment_name
    if framework_repo_path:
        data["framework_repo_path"] = framework_repo_path
    try:
        _workspace_config_path(workspace_path).write_text(json.dumps(data))
    except OSError:
        pass


def _find_project_root() -> Path:
    path = Path(__file__).parent
    for _ in range(10):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
        path = path.parent
    return Path.cwd()


def get_experiments_dir(default: Optional[str] = None) -> Union[Path, RemotePath]:
    if "CR_EXPERIMENTS_DIR" in os.environ:
        return make_path(os.environ["CR_EXPERIMENTS_DIR"])
    if default:
        return make_path(default)
    persisted = _load_persisted_databricks_config()
    if persisted and "experiments_dir" in persisted:
        return make_path(persisted["experiments_dir"])
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
    if "CR_CATALOG" in os.environ:
        return os.environ["CR_CATALOG"]
    persisted = _load_persisted_databricks_config()
    if persisted and "catalog" in persisted:
        return persisted["catalog"]
    return default


def get_schema(default: str = "default") -> str:
    if "CR_SCHEMA" in os.environ:
        return os.environ["CR_SCHEMA"]
    persisted = _load_persisted_databricks_config()
    if persisted and "schema" in persisted:
        return persisted["schema"]
    return default


def get_workspace_path(default: str | None = None) -> str | None:
    return os.environ.get("CR_WORKSPACE_PATH", default)


def get_experiment_name(default: str = "customer_retention") -> str:
    if "CR_EXPERIMENT_NAME" in os.environ:
        return os.environ["CR_EXPERIMENT_NAME"]
    persisted = _load_persisted_databricks_config()
    if persisted and "experiment_name" in persisted:
        return persisted["experiment_name"]
    return default


def get_framework_repo_path() -> str | None:
    if "CR_FRAMEWORK_REPO_PATH" in os.environ:
        return os.environ["CR_FRAMEWORK_REPO_PATH"]
    persisted = _load_persisted_databricks_config()
    if persisted and "framework_repo_path" in persisted:
        return persisted["framework_repo_path"]
    return None


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


def reload_config() -> None:
    global EXPERIMENTS_DIR, FINDINGS_DIR, DATA_DIR, MLRUNS_DIR, FEATURE_STORE_DIR
    global OUTPUT_DIR, CATALOG, SCHEMA, WORKSPACE_PATH, EXPERIMENT_NAME
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


def get_mlflow_dfs_tmpdir() -> str | None:
    if os.environ.get("MLFLOW_DFS_TMP"):
        return os.environ["MLFLOW_DFS_TMP"]
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return None
    path = f"/Volumes/{get_catalog()}/{get_schema()}/mlflow_tmp"
    _ensure_uc_volume(path)
    return path


def _parse_uc_volume(path_str: str) -> tuple[str, str, str] | None:
    parts = str(path_str).strip("/").split("/")
    if len(parts) >= 4 and parts[0] == "Volumes":
        return parts[1], parts[2], parts[3]
    return None


def _ensure_uc_volume(base_path: Union[str, Path, "RemotePath"]) -> None:
    parsed = _parse_uc_volume(str(base_path))
    if not parsed:
        return
    catalog, schema, volume = parsed
    try:
        from customer_retention.core.compat.detection import get_spark_session
        spark = get_spark_session()
    except ImportError:
        return
    if not spark:
        return
    spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}`")


def setup_experiments_structure(experiments_dir: Optional[Path] = None) -> None:
    base = experiments_dir or get_experiments_dir()
    _ensure_uc_volume(base)
    directories = [
        base / "data" / "bronze",
        base / "data" / "silver",
        base / "data" / "gold",
        base / "data" / "scoring",
        base / "mlruns",
        base / "feature_repo" / "data",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_runs_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "runs"


def get_active_run_dir() -> Path | None:
    run_id = os.environ.get("CR_RUN_ID")
    if not run_id:
        return None
    return get_runs_dir() / run_id
