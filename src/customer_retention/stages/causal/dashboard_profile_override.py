"""Generic publisher for project-side dashboard profile overrides.

The framework ships a generic ``default_profile.html`` and the generic
``v_account_explanation`` view; everything dataset-specific (the
"customer profile" panel and the SQL view that backs it) lives in the
project's own override files.  This module is the **single entry point**
that turns those two strings into a published view + a Volume HTML file
ready to be served by the Streamlit app.

Operators paste a small config cell into ``c05`` (right after
``publish_dashboard_views``); that cell defines two triple-quoted
constants -- ``PROFILE_SQL`` and ``PROFILE_HTML`` -- and calls
``apply_profile_override(...)``.  The framework does the rest:
placeholder substitution, multi-statement splitting, view publishing,
HTML write to Volume, and the ``CR_PROFILE_TEMPLATE_PATH`` reminder.

No framework code knows about email-churn or SPS or any other dataset.
The override docs (``docs/ech_notebook_ux_overrides.md``,
``docs/sps_notebook_ux_overrides.md``) supply the concrete strings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .dashboard_views import split_view_statements

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession


@dataclass
class ProfileOverrideResult:
    """Outcome of one ``apply_profile_override`` call."""

    published_views:        List[str]
    template_path:          str
    env_hint:               str

    def __str__(self) -> str:
        lines = [f"published {len(self.published_views)} profile view(s):"]
        for v in self.published_views:
            lines.append(f"  - {v}")
        lines.append(f"wrote profile template -> {self.template_path}")
        lines.append(f"set on the Databricks App: {self.env_hint}")
        return "\n".join(lines)


_RESERVED_PLACEHOLDERS = ("catalog", "schema", "composite_name")


def render_profile_sql(
    sql_text: str,
    *,
    catalog: str,
    schema: str,
    composite_name: Optional[str] = None,
    placeholders: Optional[Dict[str, str]] = None,
) -> str:
    r"""Substitute ``{catalog}`` / ``{schema}`` (and optionally ``{composite_name}``)
    into the operator-supplied SQL text.  Mirrors the contract of the
    framework's ``render_dashboard_view_sql`` so override authors can rely
    on the same set of placeholders.

    ``placeholders`` is an open-ended dict the operator can use to inject
    runtime-specific tokens such as Volume paths (e.g.
    ``{"volume_run_data": "/Volumes/.../runs/<run_id>/data"}``) so the
    profile SQL can reference Delta paths via
    ``delta.\`{volume_run_data}/silver/silver_merged\``` without coupling
    the framework to per-cluster storage layout.  Reserved keys
    ``catalog``, ``schema``, ``composite_name`` raise ``ValueError`` to
    avoid silent collisions.
    """
    text = sql_text
    if placeholders:
        for reserved in _RESERVED_PLACEHOLDERS:
            if reserved in placeholders:
                raise ValueError(
                    f"placeholders may not override reserved key {reserved!r}"
                )
        for key, value in placeholders.items():
            text = text.replace("{" + key + "}", str(value))
    if composite_name is not None:
        text = text.replace("{composite_name}", composite_name)
    return text.replace("{catalog}", catalog).replace("{schema}", schema)


def _has_executable(stmt: str) -> bool:
    """Return True when ``stmt`` has at least one non-comment, non-blank line."""
    for line in stmt.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("--"):
            continue
        return True
    return False


def _extract_view_names(statements: List[str]) -> List[str]:
    """Best-effort extraction of ``catalog.schema.view`` names from each DDL.

    Used only for the human-readable summary; never gates the actual
    publish.  When extraction fails we fall back to the statement index.
    """
    out: List[str] = []
    for idx, stmt in enumerate(statements):
        upper = stmt.upper()
        marker = "CREATE OR REPLACE VIEW"
        pos = upper.find(marker)
        if pos < 0:
            out.append(f"<statement_{idx}>")
            continue
        rest = stmt[pos + len(marker):].strip()
        token = rest.split()[0] if rest else f"<statement_{idx}>"
        out.append(token.rstrip("("))
    return out


def apply_profile_override(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    *,
    profile_sql: str,
    profile_html: str,
    template_volume_path: str,
    composite_name: Optional[str] = None,
    placeholders: Optional[Dict[str, str]] = None,
) -> ProfileOverrideResult:
    """Publish a project-side profile view set and drop the HTML on Volume.

    Parameters
    ----------
    spark
        Active Spark session.  When ``None`` the call is a no-op (returns a
        result with empty lists) so the cell stays safe in local dev.
    catalog, schema
        Unity Catalog target.  Substituted into ``{catalog}`` / ``{schema}``
        placeholders inside ``profile_sql``.
    profile_sql
        One or more ``CREATE OR REPLACE VIEW`` statements separated by
        semicolons.  Use ``{catalog}``, ``{schema}``, and optionally
        ``{composite_name}`` placeholders — they are substituted before
        execution.
    profile_html
        The complete dashboard profile template (Handlebars).  Written
        verbatim to ``template_volume_path``; the operator points
        ``CR_PROFILE_TEMPLATE_PATH`` at that path on the Databricks App.
    template_volume_path
        Absolute path on a FUSE-mounted Volume (e.g.
        ``/Volumes/cat/sch/dashboard_templates/customer_emails_profile.html``).
        Parent directories must already exist.
    composite_name
        When the SQL references ``{composite_name}`` (e.g. for
        ``gold_features_<cn>``) supply the run's composite name.  Omit when
        the SQL has no per-run binding.
    """
    if spark is None:
        logger.warning("apply_profile_override: spark is None — skipping publish")
        return ProfileOverrideResult([], "", "")

    rendered = render_profile_sql(
        profile_sql,
        catalog=catalog,
        schema=schema,
        composite_name=composite_name,
        placeholders=placeholders,
    )
    if "{catalog}" in rendered or "{schema}" in rendered:
        raise ValueError(
            "profile_sql still contains unsubstituted placeholders after rendering"
        )
    if composite_name is None and "{composite_name}" in rendered:
        raise ValueError(
            "profile_sql references {composite_name} but none was supplied"
        )
    statements = [s for s in split_view_statements(rendered) if _has_executable(s)]
    if not statements:
        raise ValueError("profile_sql contained no executable statements")

    for stmt in statements:
        spark.sql(stmt)
        logger.info("published profile view (%d chars)", len(stmt))

    template_path = Path(template_volume_path)
    if not template_path.parent.exists():
        raise FileNotFoundError(
            f"parent directory does not exist: {template_path.parent}. "
            "Create the Volume directory before calling apply_profile_override."
        )
    template_path.write_text(profile_html, encoding="utf-8")
    logger.info("wrote profile template -> %s", template_path)

    return ProfileOverrideResult(
        published_views=_extract_view_names(statements),
        template_path=str(template_path),
        env_hint=f"CR_PROFILE_TEMPLATE_PATH={template_path}",
    )
