"""Track notebook execution progress and export previous notebook on start."""
import json
import os
import threading
from pathlib import Path
from typing import Optional

from customer_retention.core.compat import is_databricks, is_remote_spark
from customer_retention.core.config.experiments import get_notebook_experiments_dir, reload_config


def _ensure_databricks_config_loaded() -> None:
    if not is_databricks():
        return
    reload_config()


def _accept_workflow_params() -> None:
    if not is_databricks():
        return
    from customer_retention.core.compat.detection import get_dbutils

    dbutils = get_dbutils()
    if not dbutils:
        return
    for widget_name, env_name in [
        ("dataset_id", "CR_DATASET_ID"),
        ("run_id", "CR_RUN_ID"),
        ("experiments_dir", "CR_EXPERIMENTS_DIR"),
    ]:
        try:
            val = dbutils.widgets.get(widget_name)
            if val:
                os.environ[env_name] = val
            elif env_name == "CR_DATASET_ID":
                os.environ.pop(env_name, None)
        except Exception:
            if env_name == "CR_DATASET_ID":
                os.environ.pop(env_name, None)


_ensure_databricks_config_loaded()
_accept_workflow_params()


def track_and_export_previous(current_notebook: str) -> None:
    """Record the current notebook and export the previous one in the background.

    Called at the top of each notebook.  Progress is written *before* the
    export thread starts so that the current notebook is already recorded
    even if export is slow or fails.

    Returns ``None`` — the export runs asynchronously.
    """
    _ensure_databricks_config_loaded()
    _accept_workflow_params()
    experiments_dir = get_notebook_experiments_dir()
    try:
        experiments_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    progress_file = experiments_dir / "notebook_progress.json"

    previous = _read_last_notebook(progress_file)
    _write_current_notebook(progress_file, current_notebook)

    if previous and not is_databricks() and not is_remote_spark():
        docs_dir = experiments_dir / "docs"
        _export_in_background(previous, docs_dir)


def _read_last_notebook(progress_file: Path) -> Optional[str]:
    """Return the last-run notebook name, or ``None`` if missing/corrupt."""
    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        return data.get("last_notebook")
    except Exception:
        return None


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
    from customer_retention.core.config.experiments import get_experiments_dir

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
    from customer_retention.analysis.auto_explorer.skip_logic import detect_skip_set_for_dataset
    from customer_retention.core.config.experiments import get_experiments_dir

    findings_dir = get_experiments_dir() / "findings"
    skip_set, skip_reasons = detect_skip_set_for_dataset(findings_dir, dataset_id)
    if notebook_stem in skip_set:
        dbutils.notebook.exit(f"SKIPPED: {skip_reasons.get(notebook_stem, 'skipped')}")


def resolve_config(value, dataset_name: str, default=None):
    if isinstance(value, dict):
        return value.get(dataset_name, default)
    return value


def _write_current_notebook(progress_file: Path, current_notebook: str) -> None:
    """Write the current notebook name to the progress file."""
    try:
        progress_file.write_text(
            json.dumps({"last_notebook": current_notebook}),
            encoding="utf-8",
        )
    except Exception:
        pass
