"""Track notebook execution progress and export previous notebook on start."""
import os
import threading
from pathlib import Path
from typing import Optional

from customer_retention.core.compat import is_databricks, is_remote_spark
from customer_retention.core.config.experiments import get_experiments_dir, reload_config


def _ensure_databricks_config_loaded() -> None:
    if not is_databricks():
        return
    reload_config()


def accept_workflow_params() -> None:
    if not is_databricks():
        return
    from customer_retention.core.compat.detection import get_dbutils

    dbutils = get_dbutils()
    if not dbutils:
        return
    resolved = {}
    for widget_name, env_name in [
        ("dataset_id", "CR_DATASET_ID"),
        ("run_id", "CR_RUN_ID"),
        ("experiments_dir", "CR_EXPERIMENTS_DIR"),
    ]:
        try:
            val = dbutils.widgets.get(widget_name)
            if val:
                os.environ[env_name] = val
                resolved[widget_name] = val
            elif env_name == "CR_DATASET_ID":
                os.environ.pop(env_name, None)
        except Exception:
            if env_name == "CR_DATASET_ID":
                os.environ.pop(env_name, None)
    if resolved:
        print(f"[CR] workflow params: {resolved}")


def _disable_mlflow_autolog() -> None:
    """Suppress MLflow autologging so profiling sklearn calls don't create noise runs."""
    try:
        import mlflow
        mlflow.autolog(disable=True)
    except ImportError:
        pass


def track_and_export_previous(current_notebook: str) -> None:
    _disable_mlflow_autolog()
    _ensure_databricks_config_loaded()
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    from customer_retention.analysis.auto_explorer.session import (
        SessionState,
        _is_automated_databricks_run,
        get_current_username,
        mark_notebook,
    )

    namespace = RunNamespace.from_env_or_latest()
    if namespace is None:
        return

    if _is_automated_databricks_run():
        return

    username = get_current_username()
    session_path = namespace.user_session_path(username)
    state = SessionState.load(session_path)
    previous = state.last_notebook if state else None

    mark_notebook(namespace, current_notebook, username=username)

    if previous and not is_databricks() and not is_remote_spark():
        docs_dir = get_experiments_dir() / "docs"
        _export_in_background(previous, docs_dir)


def _export_notebook(notebook_name: str, docs_dir: Path) -> Optional[Path]:
    """Export *notebook_name* to HTML in *docs_dir*."""
    from customer_retention.analysis.notebook_html_exporter import export_notebook_html

    return export_notebook_html(Path(notebook_name), docs_dir)


def _export_in_background(notebook_name: str, docs_dir: Path) -> None:
    """Dispatch export as a daemon thread so the notebook cell does not block."""
    threading.Thread(
        target=_export_notebook,
        args=(notebook_name, docs_dir),
        daemon=True,
    ).start()


def publish_workflow_metadata(project_context) -> None:
    if not is_databricks():
        return
    from customer_retention.core.compat.detection import get_dbutils

    dbutils = get_dbutils()
    if not dbutils:
        return
    dataset_names = list(project_context.datasets.keys())
    dbutils.jobs.taskValues.set(key="dataset_names", value=dataset_names)
    dbutils.jobs.taskValues.set(key="dataset_count", value=len(dataset_names))
    dbutils.jobs.taskValues.set(key="target_dataset", value=project_context.target_dataset or "")
    if project_context.run_id:
        dbutils.jobs.taskValues.set(key="run_id", value=project_context.run_id)
    dbutils.jobs.taskValues.set(key="experiments_dir", value=str(get_experiments_dir()))


def guard_skip(notebook_stem: str) -> None:
    if not is_databricks():
        return
    dataset_id = os.environ.get("CR_DATASET_ID")
    if not dataset_id:
        return
    from customer_retention.core.compat.detection import get_dbutils

    dbutils = get_dbutils()
    if not dbutils:
        return
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    from customer_retention.analysis.auto_explorer.skip_logic import detect_skip_set_for_dataset

    namespace = RunNamespace.from_env_or_latest()
    findings_dir = get_experiments_dir() / "findings"
    skip_set, skip_reasons = detect_skip_set_for_dataset(
        findings_dir, dataset_id, namespace=namespace,
    )
    if notebook_stem in skip_set:
        dbutils.notebook.exit(f"SKIPPED: {skip_reasons.get(notebook_stem, 'skipped')}")


def resolve_config(value, dataset_name: str, default=None):
    if isinstance(value, dict):
        return value.get(dataset_name, default)
    return value
