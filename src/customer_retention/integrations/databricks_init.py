from __future__ import annotations

import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatabricksInitResult:
    catalog: str
    schema: str
    experiment_name: str
    workspace_path: str | None
    model_name: str
    framework_repo_path: str | None = None
    exploration_notebooks_path: str = "exploration_notebooks"
    causal_notebooks_path: str = "causal_notebooks"
    experiments_path: str = "experiments"
    playbooks_path: str = "playbooks"
    notebooks_copied: list[str] = field(default_factory=list)
    notebooks_synced: list[str] = field(default_factory=list)
    causal_notebooks_copied: list[str] = field(default_factory=list)
    causal_notebooks_synced: list[str] = field(default_factory=list)

    @property
    def environment_variables(self) -> dict[str, str]:
        env_vars = {
            "CR_CATALOG": self.catalog,
            "CR_SCHEMA": self.schema,
            "CR_EXPERIMENT_NAME": self.experiment_name,
            "CR_EXPERIMENTS_DIR": f"/Volumes/{self.catalog}/{self.schema}/{self.experiments_path}",
            "CR_PLAYBOOKS_DIR": f"/Volumes/{self.catalog}/{self.schema}/{self.playbooks_path}",
        }
        if self.workspace_path:
            env_vars["CR_WORKSPACE_PATH"] = self.workspace_path
            env_vars["CR_CAUSAL_NOTEBOOKS_DIR"] = (
                f"/Workspace/{self.workspace_path}/{self.causal_notebooks_path}"
            )
        if self.framework_repo_path:
            env_vars["CR_FRAMEWORK_REPO_PATH"] = self.framework_repo_path
        return env_vars


def databricks_init(
    catalog: str = "main",
    schema: str = "default",
    experiment_name: str | None = None,
    workspace_path: str | None = None,
    copy_notebooks: bool = True,
    model_name: str = "customer_retention",
    framework_repo_path: str | None = None,
    exploration_notebooks_path: str = "exploration_notebooks",
    causal_notebooks_path: str = "causal_notebooks",
    experiments_path: str = "experiments",
    playbooks_path: str = "playbooks",
) -> DatabricksInitResult:
    _validate_databricks_environment()
    if workspace_path:
        workspace_path = _normalize_workspace_path(workspace_path)
        _ensure_workspace_directory(workspace_path)
    _set_environment_variables(
        catalog, schema, workspace_path, framework_repo_path,
        experiments_path, playbooks_path, causal_notebooks_path,
    )
    resolved_experiment_name = experiment_name or _resolve_experiment_name_from_notebook_path()
    resolved_experiment_name = _make_absolute_experiment_path(resolved_experiment_name, workspace_path)
    _set_experiment_name_env_var(resolved_experiment_name)
    _persist_config(
        catalog, schema, workspace_path, resolved_experiment_name, framework_repo_path,
        experiments_path, playbooks_path, causal_notebooks_path,
    )
    _reload_config_constants()
    _ensure_experiments_volume_exists(catalog, schema, experiments_path)
    _ensure_playbooks_volume_exists(catalog, schema, playbooks_path)
    _setup_experiment_directories()
    notebooks_copied: list[str] = []
    notebooks_synced: list[str] = []
    causal_copied: list[str] = []
    causal_synced: list[str] = []
    if copy_notebooks and workspace_path:
        notebooks_copied, notebooks_synced = _sync_exploration_notebooks(
            workspace_path,
            framework_repo_path=framework_repo_path,
            exploration_notebooks_path=exploration_notebooks_path,
        )
        causal_copied, causal_synced = _sync_causal_notebooks(
            workspace_path,
            framework_repo_path=framework_repo_path,
            causal_notebooks_path=causal_notebooks_path,
        )
    if workspace_path:
        _write_requirements_files(workspace_path)
    result = DatabricksInitResult(
        catalog=catalog,
        schema=schema,
        experiment_name=resolved_experiment_name,
        workspace_path=workspace_path,
        model_name=model_name,
        framework_repo_path=framework_repo_path,
        exploration_notebooks_path=exploration_notebooks_path,
        causal_notebooks_path=causal_notebooks_path,
        experiments_path=experiments_path,
        playbooks_path=playbooks_path,
        notebooks_copied=notebooks_copied,
        notebooks_synced=notebooks_synced,
        causal_notebooks_copied=causal_copied,
        causal_notebooks_synced=causal_synced,
    )
    _display_init_summary(result)
    return result


def _validate_databricks_environment() -> None:
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        raise RuntimeError(
            "databricks_init() must be called from a Databricks notebook. "
            "DATABRICKS_RUNTIME_VERSION not found in environment."
        )


def _set_environment_variables(
    catalog: str, schema: str, workspace_path: str | None, framework_repo_path: str | None = None,
    experiments_path: str = "experiments",
    playbooks_path: str = "playbooks",
    causal_notebooks_path: str = "causal_notebooks",
) -> None:
    os.environ["CR_CATALOG"] = catalog
    os.environ["CR_SCHEMA"] = schema
    os.environ["CR_EXPERIMENTS_DIR"] = f"/Volumes/{catalog}/{schema}/{experiments_path}"
    os.environ["CR_PLAYBOOKS_DIR"] = f"/Volumes/{catalog}/{schema}/{playbooks_path}"
    if workspace_path:
        os.environ["CR_WORKSPACE_PATH"] = workspace_path
        os.environ["CR_CAUSAL_NOTEBOOKS_DIR"] = (
            f"/Workspace/{workspace_path}/{causal_notebooks_path}"
        )
    if framework_repo_path:
        os.environ["CR_FRAMEWORK_REPO_PATH"] = framework_repo_path


def _set_experiment_name_env_var(experiment_name: str) -> None:
    os.environ["CR_EXPERIMENT_NAME"] = experiment_name


def _persist_config(
    catalog: str, schema: str, workspace_path: str | None,
    experiment_name: str | None = None, framework_repo_path: str | None = None,
    experiments_path: str = "experiments",
    playbooks_path: str = "playbooks",
    causal_notebooks_path: str = "causal_notebooks",
) -> None:
    from customer_retention.core.config.experiments import persist_databricks_config

    causal_dir = (
        f"/Workspace/{workspace_path}/{causal_notebooks_path}" if workspace_path else None
    )
    persist_databricks_config(
        f"/Volumes/{catalog}/{schema}/{experiments_path}", catalog, schema, workspace_path,
        experiment_name, framework_repo_path=framework_repo_path,
        playbooks_dir=f"/Volumes/{catalog}/{schema}/{playbooks_path}",
        causal_notebooks_dir=causal_dir,
    )


def _reload_config_constants() -> None:
    from customer_retention.core.config.experiments import reload_config

    reload_config()


def _ensure_experiments_volume_exists(
    catalog: str, schema: str, experiments_path: str = "experiments",
) -> None:
    from customer_retention.core.config.experiments import _ensure_uc_volume

    _ensure_uc_volume(f"/Volumes/{catalog}/{schema}/{experiments_path}")


def _ensure_playbooks_volume_exists(
    catalog: str, schema: str, playbooks_path: str = "playbooks",
) -> None:
    from customer_retention.core.config.experiments import _ensure_uc_volume

    _ensure_uc_volume(f"/Volumes/{catalog}/{schema}/{playbooks_path}")


def _setup_experiment_directories() -> None:
    from customer_retention.core.config.experiments import setup_experiments_structure

    try:
        setup_experiments_structure()
    except (OSError, RuntimeError):
        pass


def _resolve_experiment_name_from_notebook_path() -> str:
    try:
        dbutils = _get_dbutils()
        if dbutils:
            notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
            return notebook_path.rsplit("/", 1)[-1]
    except (AttributeError, RuntimeError, KeyError, TypeError):
        pass
    return "customer_retention"


def _get_dbutils() -> Any | None:
    try:
        from customer_retention.core.compat.detection import get_dbutils

        return get_dbutils()
    except (ImportError, AttributeError, RuntimeError):
        return None


def _normalize_workspace_path(workspace_path: str) -> str:
    if workspace_path.startswith("/Workspace/"):
        return workspace_path[len("/Workspace/"):]
    return workspace_path


def _make_absolute_experiment_path(experiment_name: str, workspace_path: str | None) -> str:
    if experiment_name.startswith("/"):
        return experiment_name
    if not workspace_path:
        return experiment_name
    base = workspace_path.removeprefix("/Workspace")
    if not base.startswith("/"):
        base = f"/{base}"
    return f"{base}/{experiment_name}"


def _ensure_workspace_directory(workspace_path: str) -> None:
    try:
        Path(f"/Workspace/{workspace_path}").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _sync_exploration_notebooks(
    workspace_path: str,
    *,
    framework_repo_path: str | None = None,
    exploration_notebooks_path: str = "exploration_notebooks",
) -> tuple[list[str], list[str]]:
    from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

    source_dir = ProjectInitializer(project_name="")._get_exploration_source_dir()
    return _sync_notebook_directory(
        source_dir,
        workspace_path,
        exploration_notebooks_path,
        framework_repo_path=framework_repo_path,
    )


def _sync_causal_notebooks(
    workspace_path: str,
    *,
    framework_repo_path: str | None = None,
    causal_notebooks_path: str = "causal_notebooks",
) -> tuple[list[str], list[str]]:
    """Mirror :func:`_sync_exploration_notebooks` for ``causal_notebooks/``.

    Causal notebooks live alongside the exploration notebooks in the
    framework package and use the same cell-id sync engine. The destination
    is a sibling workspace folder so operators see two parallel directories
    inside their Databricks workspace.
    """
    from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

    source_dir = ProjectInitializer(project_name="")._get_causal_source_dir()
    return _sync_notebook_directory(
        source_dir,
        workspace_path,
        causal_notebooks_path,
        framework_repo_path=framework_repo_path,
    )


def _sync_notebook_directory(
    source_dir: Path | None,
    workspace_path: str,
    workspace_subpath: str,
    *,
    framework_repo_path: str | None,
) -> tuple[list[str], list[str]]:
    if not source_dir or not source_dir.exists():
        return [], []

    dest_dir = Path(f"/Workspace/{workspace_path}/{workspace_subpath}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    synced: list[str] = []
    for notebook in source_dir.glob("*.ipynb"):
        dest_path = dest_dir / notebook.name
        if not dest_path.exists():
            shutil.copy2(notebook, dest_path)
            _inject_system_cell(dest_path, framework_repo_path)
            copied.append(str(dest_path))
            continue
        try:
            if _sync_notebook(notebook, dest_path, framework_repo_path=framework_repo_path):
                synced.append(str(dest_path))
        except (OSError, ValueError):
            warnings.warn(
                f"Could not sync notebook '{notebook.name}', skipping",
                stacklevel=2,
            )

    return copied, synced


def _sync_notebook(
    repo_path: Path, user_path: Path, *, framework_repo_path: str | None = None,
) -> bool:
    import nbformat

    from customer_retention.generators.notebook_sync.sync_engine import NotebookSyncEngine

    try:
        repo_nb = nbformat.read(str(repo_path), as_version=4)
        user_nb = nbformat.read(str(user_path), as_version=4)
    except (OSError, ValueError, KeyError):
        return False

    engine = NotebookSyncEngine()
    merged, report = engine.sync(repo_nb, user_nb)

    system_cell_changed = engine.ensure_system_cell(merged, framework_repo_path)

    if not report.has_changes and not system_cell_changed:
        return False

    nbformat.write(merged, str(user_path))
    return True


def _inject_system_cell(notebook_path: Path, framework_repo_path: str | None) -> None:
    if not framework_repo_path:
        return
    import nbformat

    from customer_retention.generators.notebook_sync.sync_engine import NotebookSyncEngine

    try:
        nb = nbformat.read(str(notebook_path), as_version=4)
    except (OSError, ValueError, KeyError):
        return

    if NotebookSyncEngine.ensure_system_cell(nb, framework_repo_path):
        nbformat.write(nb, str(notebook_path))


def _write_requirements_files(workspace_path: str) -> None:
    try:
        dest_dir = Path(f"/Workspace/{workspace_path}")
        for name in ("requirements.txt", "requirements-databricks.txt"):
            src = _find_requirements_file(name)
            if src and src.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest_dir / name)
    except OSError:
        pass


def _find_requirements_file(name: str) -> Path | None:
    try:
        import importlib.resources
        pkg_root = Path(importlib.resources.files("customer_retention").__str__()).parent.parent
        candidate = pkg_root / name
        if candidate.exists():
            return candidate
    except (ImportError, AttributeError, TypeError, OSError):
        pass
    return None


def _display_init_summary(result: DatabricksInitResult) -> None:
    from customer_retention import __version__

    print(f"ChurnKit v{__version__} Databricks Initialization Complete")
    print("=" * 45)
    print(f"  Catalog:          {result.catalog}")
    print(f"  Schema:           {result.schema}")
    print(f"  Experiment:       {result.experiment_name}")
    print(f"  Experiments Dir:  /Volumes/{result.catalog}/{result.schema}/{result.experiments_path}")
    print(f"  Experiments Path: {result.experiments_path}")
    print(f"  Playbooks Dir:    /Volumes/{result.catalog}/{result.schema}/{result.playbooks_path}")
    print(f"  Playbooks Path:   {result.playbooks_path}")
    print(f"  Workspace Path:   {result.workspace_path or '(not set)'}")
    print(f"  Notebooks Path:   {result.exploration_notebooks_path}")
    print(f"  Causal NB Path:   {result.causal_notebooks_path}")
    print(f"  Model Name:       {result.model_name}")
    if result.framework_repo_path:
        print(f"  Framework Repo:   {result.framework_repo_path}")
    else:
        print("  Framework:        PyPI (cluster library)")
    if result.notebooks_copied:
        print(f"  Notebooks Copied: {len(result.notebooks_copied)}")
        for nb in result.notebooks_copied:
            print(f"    - {nb}")
    if result.notebooks_synced:
        print(f"  Notebooks Synced: {len(result.notebooks_synced)}")
        for nb in result.notebooks_synced:
            print(f"    - {nb}")
    if result.causal_notebooks_copied:
        print(f"  Causal Copied:    {len(result.causal_notebooks_copied)}")
        for nb in result.causal_notebooks_copied:
            print(f"    - {nb}")
    if result.causal_notebooks_synced:
        print(f"  Causal Synced:    {len(result.causal_notebooks_synced)}")
        for nb in result.causal_notebooks_synced:
            print(f"    - {nb}")
    print("=" * 45)
