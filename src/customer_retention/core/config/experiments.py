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
    playbooks_dir: str | None = None,
    causal_notebooks_dir: str | None = None,
) -> None:
    if not workspace_path:
        return
    data: dict = {
        "experiments_dir": experiments_dir,
        "catalog": catalog,
        "schema": schema,
        "workspace_path": workspace_path,
    }
    if experiment_name:
        data["experiment_name"] = experiment_name
    if framework_repo_path:
        data["framework_repo_path"] = framework_repo_path
    if playbooks_dir:
        data["playbooks_dir"] = playbooks_dir
    if causal_notebooks_dir:
        data["causal_notebooks_dir"] = causal_notebooks_dir
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


_RUN_POINTER_FILENAME = ".cr_active_run.json"


def _read_project_pointer_experiments_root() -> Optional[str]:
    """Return ``experiments_root`` from ``<project_root>/.cr_active_run.json``.

    The pointer is the file-based signal NB00 (and any other run-creating
    notebook) writes — it carries both the active ``run_id`` and the
    ``experiments_root`` path. Notebook-job tasks cannot rely on env vars
    propagating across tasks, so file-tracked discovery is mandatory.

    Returns None when the pointer is absent, malformed, or stale (the
    referenced directory no longer exists). The staleness check protects
    against leftover pointers from prior test runs and from runs whose
    Volume mount has been deleted.
    """
    try:
        pointer = _find_project_root() / _RUN_POINTER_FILENAME
        if not pointer.exists():
            return None
        data = json.loads(pointer.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    root_str = data.get("experiments_root")
    if not (isinstance(root_str, str) and root_str):
        return None
    # Defend against stale pointers — only honor when the directory exists.
    try:
        if not Path(root_str).exists():
            return None
    except OSError:
        return None
    return root_str


def get_experiments_dir(default: Optional[str] = None) -> Union[Path, RemotePath]:
    if "CR_EXPERIMENTS_DIR" in os.environ:
        return make_path(os.environ["CR_EXPERIMENTS_DIR"])
    if default:
        return make_path(default)
    persisted = _load_persisted_databricks_config()
    if persisted and "experiments_dir" in persisted:
        return make_path(persisted["experiments_dir"])
    # File-tracked tier: read from the project pointer NB00 writes. This is
    # the only signal that survives across notebook-job tasks (env vars do
    # not). Falls through to the project-root fallback when absent.
    pointer_root = _read_project_pointer_experiments_root()
    if pointer_root:
        return make_path(pointer_root)
    return _find_project_root() / "experiments"


def get_findings_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "findings"


def get_data_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "data"


def get_mlruns_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "mlruns"


def get_feature_store_dir(default: Optional[str] = None) -> Path:
    return get_experiments_dir(default) / "feature_repo"


def get_playbooks_dir(default: Optional[str] = None) -> Union[Path, RemotePath]:
    """Resolve the playbooks volume / directory.

    Resolution order, mirroring ``get_experiments_dir``:

    1. ``CR_PLAYBOOKS_DIR`` environment variable
    2. Explicit ``default`` argument
    3. ``playbooks_dir`` from the persisted Databricks config (set by
       ``databricks_init`` and friends)
    4. Repo-local fallback: project root ``/playbooks``

    The fallback uses the same project root discovery as ``get_experiments_dir``
    so local development "just works" against the gitignored ``playbooks/``
    directory in the repo.
    """
    if "CR_PLAYBOOKS_DIR" in os.environ:
        return make_path(os.environ["CR_PLAYBOOKS_DIR"])
    if default:
        return make_path(default)
    persisted = _load_persisted_databricks_config()
    if persisted and "playbooks_dir" in persisted:
        return make_path(persisted["playbooks_dir"])
    return _find_project_root() / "playbooks"


def get_causal_notebooks_dir(default: Optional[str] = None) -> Union[Path, RemotePath]:
    """Resolve the causal-notebooks directory.

    Sibling of ``exploration_notebooks/`` that holds the four hand-authored
    causal-track notebooks (``c01..c04``). Resolution order mirrors
    ``get_playbooks_dir``:

    1. ``CR_CAUSAL_NOTEBOOKS_DIR`` environment variable
    2. Explicit ``default`` argument
    3. ``causal_notebooks_dir`` from the persisted Databricks config (set by
       ``databricks_init``)
    4. Repo-local fallback: project root ``/causal_notebooks``
    """
    if "CR_CAUSAL_NOTEBOOKS_DIR" in os.environ:
        return make_path(os.environ["CR_CAUSAL_NOTEBOOKS_DIR"])
    if default:
        return make_path(default)
    persisted = _load_persisted_databricks_config()
    if persisted and "causal_notebooks_dir" in persisted:
        return make_path(persisted["causal_notebooks_dir"])
    return _find_project_root() / "causal_notebooks"


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
    if "CR_WORKSPACE_PATH" in os.environ:
        return os.environ["CR_WORKSPACE_PATH"]
    persisted = _load_persisted_databricks_config()
    if persisted and "workspace_path" in persisted:
        return persisted["workspace_path"]
    return default


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
PLAYBOOKS_DIR = get_playbooks_dir()
CAUSAL_NOTEBOOKS_DIR = get_causal_notebooks_dir()
OUTPUT_DIR = FINDINGS_DIR
CATALOG = get_catalog()
SCHEMA = get_schema()
WORKSPACE_PATH = get_workspace_path()
EXPERIMENT_NAME = get_experiment_name()


def reload_config() -> None:
    global EXPERIMENTS_DIR, FINDINGS_DIR, DATA_DIR, MLRUNS_DIR, FEATURE_STORE_DIR
    global PLAYBOOKS_DIR, CAUSAL_NOTEBOOKS_DIR
    global OUTPUT_DIR, CATALOG, SCHEMA, WORKSPACE_PATH, EXPERIMENT_NAME
    EXPERIMENTS_DIR = get_experiments_dir()
    FINDINGS_DIR = get_findings_dir()
    DATA_DIR = get_data_dir()
    MLRUNS_DIR = get_mlruns_dir()
    FEATURE_STORE_DIR = get_feature_store_dir()
    PLAYBOOKS_DIR = get_playbooks_dir()
    CAUSAL_NOTEBOOKS_DIR = get_causal_notebooks_dir()
    OUTPUT_DIR = FINDINGS_DIR
    CATALOG = get_catalog()
    SCHEMA = get_schema()
    WORKSPACE_PATH = get_workspace_path()
    EXPERIMENT_NAME = get_experiment_name()


def get_mlflow_dfs_tmpdir() -> str | None:
    if os.environ.get("MLFLOW_DFS_TMP"):
        path = os.environ["MLFLOW_DFS_TMP"]
        _ensure_uc_volume(path)
        return path
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
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
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
