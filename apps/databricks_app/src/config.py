"""Runtime configuration — reads env vars set by app.yaml (or .env locally).

When ``CR_PROFILE_TEMPLATE_PATH`` is unset (e.g. on a Databricks cluster where
env vars are not configured), we fall back to **file-based discovery**: scan
``/Volumes/{catalog}/{schema}/dashboard_templates/`` for the newest ``.html``
template and use that path. This mirrors the file-tracked discovery the
framework already uses for ``RunNamespace`` and avoids forcing operators to
juggle env vars across job tasks (which strip them on Databricks anyway).

The fallback is purely additive — the env var still wins when set, and an
empty result simply means the app keeps using its bundled default template
(existing behaviour).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    catalog: str
    schema: str
    warehouse_id: str
    profile_template_path: str

    @property
    def fqn_prefix(self) -> str:
        return f"{self.catalog}.{self.schema}"


def _discover_profile_template(catalog: str, schema: str) -> str:
    """Pick the newest ``.html`` under ``/Volumes/{cat}/{sch}/dashboard_templates/``.

    Returns "" when the directory is missing, empty, or unreadable — in which
    case the app keeps using its bundled default template (the empty-state
    panel will explain that no dataset-specific override is wired).
    """
    candidate_dir = Path(f"/Volumes/{catalog}/{schema}/dashboard_templates")
    try:
        if not candidate_dir.is_dir():
            return ""
        html_files = [p for p in candidate_dir.iterdir() if p.is_file() and p.suffix == ".html"]
    except OSError as exc:
        logger.info("profile template discovery skipped (%s)", exc)
        return ""
    if not html_files:
        return ""
    try:
        newest = max(html_files, key=lambda p: p.stat().st_mtime)
    except OSError as exc:
        logger.info("profile template discovery skipped (%s)", exc)
        return ""
    logger.info("profile template auto-discovered -> %s", newest)
    return str(newest)


def _resolve_profile_template_path(catalog: str, schema: str) -> str:
    explicit = os.environ.get("CR_PROFILE_TEMPLATE_PATH", "").strip()
    if explicit:
        return explicit
    return _discover_profile_template(catalog, schema)


def load_config() -> AppConfig:
    catalog = os.environ.get("CR_CATALOG", "churnkit")
    schema = os.environ.get("CR_SCHEMA", "analysis")
    return AppConfig(
        catalog=catalog,
        schema=schema,
        warehouse_id=os.environ.get("CR_WAREHOUSE_ID", ""),
        profile_template_path=_resolve_profile_template_path(catalog, schema),
    )
