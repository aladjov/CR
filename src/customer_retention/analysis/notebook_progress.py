"""Track notebook execution progress and export previous notebook on start."""
import json
import threading
from pathlib import Path
from typing import Optional

from customer_retention.core.compat import is_databricks, is_remote_spark
from customer_retention.core.config.experiments import get_notebook_experiments_dir, reload_config


def _ensure_databricks_config_loaded() -> None:
    if not is_databricks():
        return
    reload_config()


_ensure_databricks_config_loaded()


def track_and_export_previous(current_notebook: str) -> None:
    """Record the current notebook and export the previous one in the background.

    Called at the top of each notebook.  Progress is written *before* the
    export thread starts so that the current notebook is already recorded
    even if export is slow or fails.

    Returns ``None`` — the export runs asynchronously.
    """
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
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
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


def publish_skip_flags(findings) -> None:
    if not is_databricks():
        return
    from customer_retention.core.compat.detection import get_dbutils
    from customer_retention.core.config.column_config import ColumnType, DatasetGranularity
    dbutils = get_dbutils()
    if not dbutils:
        return
    has_event_data = bool(
        findings.time_series_metadata
        and findings.time_series_metadata.granularity == DatasetGranularity.EVENT_LEVEL
    )
    has_text = bool(findings.text_processing) or ColumnType.TEXT in findings.column_types.values()
    dbutils.jobs.taskValues.set(key="has_event_data", value=has_event_data)
    dbutils.jobs.taskValues.set(key="has_text_columns", value=bool(has_text))


def _write_current_notebook(progress_file: Path, current_notebook: str) -> None:
    """Write the current notebook name to the progress file."""
    progress_file.write_text(
        json.dumps({"last_notebook": current_notebook}),
        encoding="utf-8",
    )
