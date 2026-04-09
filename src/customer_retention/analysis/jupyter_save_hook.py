"""Jupyter post-save hook that exports exploration + causal notebooks to HTML.

Add to jupyter_notebook_config.py or jupyter_server_config.py::

    from customer_retention.analysis.jupyter_save_hook import post_save_export
    c.ContentsManager.post_save_hook = post_save_export
"""
import logging
from pathlib import Path

from customer_retention.analysis.notebook_html_exporter import export_notebook_html
from customer_retention.core.config.experiments import get_experiments_dir

logger = logging.getLogger(__name__)

EXPLORATION_DIR_NAME = "exploration_notebooks"
CAUSAL_DIR_NAME = "causal_notebooks"
_TRACKED_DIR_NAMES = (EXPLORATION_DIR_NAME, CAUSAL_DIR_NAME)


def post_save_export(model, os_path, contents_manager, **kwargs):
    if model.get("type") != "notebook":
        return
    path = Path(os_path)
    if not any(part in _TRACKED_DIR_NAMES for part in path.parts):
        return
    try:
        export_notebook_html(path, get_experiments_dir() / "docs")
    except Exception:
        logger.warning("HTML export failed for %s", path.name, exc_info=True)
