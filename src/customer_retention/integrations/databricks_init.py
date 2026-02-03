from __future__ import annotations

import os
import shutil
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
    notebooks_copied: list[str] = field(default_factory=list)

    @property
    def environment_variables(self) -> dict[str, str]:
        env_vars = {
            "CR_CATALOG": self.catalog,
            "CR_SCHEMA": self.schema,
            "CR_EXPERIMENT_NAME": self.experiment_name,
            "CR_EXPERIMENTS_DIR": f"/Workspace/{self.workspace_path}/experiments" if self.workspace_path else "",
        }
        if self.workspace_path:
            env_vars["CR_WORKSPACE_PATH"] = self.workspace_path
        return env_vars


def databricks_init(
    catalog: str = "main",
    schema: str = "default",
    experiment_name: str | None = None,
    workspace_path: str | None = None,
    copy_notebooks: bool = True,
    model_name: str = "customer_retention",
) -> DatabricksInitResult:
    _validate_databricks_environment()
    _set_environment_variables(catalog, schema, workspace_path)
    resolved_experiment_name = experiment_name or _resolve_experiment_name_from_notebook_path()
    resolved_experiment_name = _make_absolute_experiment_path(resolved_experiment_name, workspace_path)
    _set_experiment_name_env_var(resolved_experiment_name)
    _configure_mlflow_experiment(resolved_experiment_name)
    notebooks_copied: list[str] = []
    if copy_notebooks and workspace_path:
        notebooks_copied = _copy_exploration_notebooks(workspace_path)
    result = DatabricksInitResult(
        catalog=catalog,
        schema=schema,
        experiment_name=resolved_experiment_name,
        workspace_path=workspace_path,
        model_name=model_name,
        notebooks_copied=notebooks_copied,
    )
    _display_init_summary(result)
    return result


def _validate_databricks_environment() -> None:
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        raise RuntimeError(
            "databricks_init() must be called from a Databricks notebook. "
            "DATABRICKS_RUNTIME_VERSION not found in environment."
        )


def _set_environment_variables(catalog: str, schema: str, workspace_path: str | None) -> None:
    os.environ["CR_CATALOG"] = catalog
    os.environ["CR_SCHEMA"] = schema
    if workspace_path:
        os.environ["CR_WORKSPACE_PATH"] = workspace_path
        os.environ["CR_EXPERIMENTS_DIR"] = f"/Workspace/{workspace_path}/experiments"


def _set_experiment_name_env_var(experiment_name: str) -> None:
    os.environ["CR_EXPERIMENT_NAME"] = experiment_name


def _resolve_experiment_name_from_notebook_path() -> str:
    try:
        dbutils = _get_dbutils()
        if dbutils:
            notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
            return notebook_path.rsplit("/", 1)[-1]
    except Exception:
        pass
    return "customer_retention"


def _get_dbutils() -> Any | None:
    try:
        from customer_retention.core.compat.detection import get_dbutils

        return get_dbutils()
    except Exception:
        return None


def _make_absolute_experiment_path(experiment_name: str, workspace_path: str | None) -> str:
    if experiment_name.startswith("/"):
        return experiment_name
    if not workspace_path:
        return experiment_name
    base = workspace_path.removeprefix("/Workspace")
    if not base.startswith("/"):
        base = f"/{base}"
    return f"{base}/{experiment_name}"


def _configure_mlflow_experiment(experiment_name: str) -> None:
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
    except ImportError:
        pass


def _copy_exploration_notebooks(workspace_path: str) -> list[str]:
    from customer_retention.generators.notebook_generator.project_init import ProjectInitializer

    source_dir = ProjectInitializer(project_name="")._get_exploration_source_dir()
    if not source_dir or not source_dir.exists():
        return []

    dest_dir = Path(f"/Workspace/{workspace_path}/exploration_notebooks")
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for notebook in source_dir.glob("*.ipynb"):
        dest_path = dest_dir / notebook.name
        if not dest_path.exists():
            shutil.copy2(notebook, dest_path)
            copied.append(str(dest_path))

    return copied


def _display_init_summary(result: DatabricksInitResult) -> None:
    print("ChurnKit Databricks Initialization Complete")
    print("=" * 45)
    print(f"  Catalog:          {result.catalog}")
    print(f"  Schema:           {result.schema}")
    print(f"  Experiment:       {result.experiment_name}")
    print(f"  Workspace Path:   {result.workspace_path or '(not set)'}")
    print(f"  Model Name:       {result.model_name}")
    if result.notebooks_copied:
        print(f"  Notebooks Copied: {len(result.notebooks_copied)}")
        for nb in result.notebooks_copied:
            print(f"    - {nb}")
    print("=" * 45)
