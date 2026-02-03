"""Track notebook execution progress and export previous notebook on start."""
import json
import threading
from pathlib import Path
from typing import Optional

from customer_retention.core.compat import is_databricks
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
    experiments_dir.mkdir(parents=True, exist_ok=True)
    progress_file = experiments_dir / "notebook_progress.json"
    docs_dir = experiments_dir / "docs"

    previous = _read_last_notebook(progress_file)
    _write_current_notebook(progress_file, current_notebook)

    if previous and not is_databricks():
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


def _write_current_notebook(progress_file: Path, current_notebook: str) -> None:
    """Write the current notebook name to the progress file."""
    progress_file.write_text(
        json.dumps({"last_notebook": current_notebook}),
        encoding="utf-8",
    )
