"""Generic publisher for project-side dashboard profile overrides.

The framework ships a generic ``default_profile.html`` and the generic
``v_account_explanation`` view; everything dataset-specific (the
"customer profile" panel and the SQL view that backs it) lives in the
project's own override files. This module is the **single entry point**
that turns those two strings into a published SQL view set + a UC-table
row holding the rendered HTML body, ready to be served by the
Databricks App.

Operators paste a small config cell into ``c05`` (right after
``publish_dashboard_views``); that cell defines two triple-quoted
constants -- ``PROFILE_SQL`` and ``PROFILE_HTML`` -- and calls
``apply_profile_override(...)``. The framework does the rest:
placeholder substitution, multi-statement splitting, view publishing,
and an append-only write into ``dashboard_template_overrides``.

The HTML body lives as a row in the Delta table
``{catalog}.{schema}.dashboard_template_overrides`` (created by
``publish_dashboard_views``); the active row per ``composite_name`` is
exposed by ``v_dashboard_template_active``. The Databricks App reads
the template through the same SQL warehouse it already uses for every
``v_*`` view -- this avoids the Volume-FUSE access path that proved
unreliable from Apps even with READ_VOLUME granted.

No framework code knows about email-churn or SPS or any other dataset.
The override docs (``docs/ech_notebook_ux_overrides.md``,
``docs/sps_notebook_ux_overrides.md``) supply the concrete strings.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from .dashboard_views import (
    MaterializedViewSpec,
    materialize_view_as_table,
    split_view_statements,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession


_TEMPLATE_TABLE_NAME = "dashboard_template_overrides"


@dataclass
class ProfileOverrideResult:
    """Outcome of one ``apply_profile_override`` call."""

    published_views:        List[str]
    template_table_fqn:     str
    composite_name:         str
    updated_at:             str
    updated_by:             str
    materialized_views:     List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.materialized_views is None:
            self.materialized_views = []

    def __str__(self) -> str:
        lines = [f"published {len(self.published_views)} profile view(s):"]
        for v in self.published_views:
            lines.append(f"  - {v}")
        if self.materialized_views:
            lines.append(
                f"materialized {len(self.materialized_views)} view(s) as Delta tables:"
            )
            for v in self.materialized_views:
                lines.append(f"  - {v}")
        lines.append(
            f"appended template row to {self.template_table_fqn} "
            f"(composite_name={self.composite_name!r}, updated_by={self.updated_by!r}, "
            f"updated_at={self.updated_at})"
        )
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


def _resolve_updated_by(spark: "SparkSession") -> str:
    """Best-effort identity for the ``updated_by`` audit column.

    Tries Spark's ``current_user()`` (works on serverless + classic), falls
    back to ``CR_USERNAME`` env var, then ``USER`` env var, then the literal
    ``"unknown"``. Diagnostics only — never gates the write.
    """
    try:
        row = spark.sql("SELECT current_user() AS u").first()
        if row and row["u"]:
            return str(row["u"])
    except Exception:  # noqa: BLE001
        pass
    import os
    return os.environ.get("CR_USERNAME") or os.environ.get("USER") or "unknown"


def _ensure_template_table(spark: "SparkSession", table_fqn: str) -> None:
    """Create the template-overrides table if the publisher hasn't already.

    ``publish_dashboard_views`` ships the same DDL, so this is a defensive
    no-op when c05 was run end-to-end. Defined here so ad-hoc operator
    cells that call ``apply_profile_override`` directly without a fresh
    publish still succeed.
    """
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_fqn} (
            composite_name  STRING NOT NULL,
            profile_html    STRING NOT NULL,
            updated_at      TIMESTAMP NOT NULL,
            updated_by      STRING
        ) USING DELTA
        """
    )


def apply_profile_override(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    *,
    profile_sql: str,
    profile_html: str,
    composite_name: str,
    placeholders: Optional[Dict[str, str]] = None,
    template_volume_path: Optional[str] = None,
    materialize_views: Optional[Sequence[MaterializedViewSpec]] = None,
) -> ProfileOverrideResult:
    """Publish a project-side profile view set and append the HTML row.

    Parameters
    ----------
    spark
        Active Spark session. When ``None`` the call is a no-op (returns
        an empty result) so the cell stays safe in local dev.
    catalog, schema
        Unity Catalog target. Substituted into ``{catalog}`` / ``{schema}``
        placeholders inside ``profile_sql``.
    profile_sql
        One or more ``CREATE OR REPLACE VIEW`` statements separated by
        semicolons. Use ``{catalog}``, ``{schema}``, and optionally
        ``{composite_name}`` placeholders -- they are substituted before
        execution.
    profile_html
        The complete dashboard profile template (Handlebars). Appended
        verbatim as a row in
        ``{catalog}.{schema}.dashboard_template_overrides``; the active
        row per ``composite_name`` is then exposed by
        ``v_dashboard_template_active`` for the Databricks App to read
        through its SQL warehouse.
    composite_name
        Required. Identifies the dataset's model run. Used both as the
        UC-row key for the template and as the ``{composite_name}``
        placeholder substitution in ``profile_sql``.
    placeholders
        Optional extra placeholders for runtime-specific tokens (Volume
        paths for inline Delta references, etc.). Keys ``catalog``,
        ``schema``, ``composite_name`` are reserved.
    template_volume_path
        DEPRECATED. Previously the absolute path on a FUSE-mounted Volume
        where the HTML body was written. Kept as a no-op kwarg so existing
        ``c05`` operator cells generated against the old framework signature
        keep running unchanged -- the App now reads the body from the
        ``dashboard_template_overrides`` Delta table via its SQL warehouse,
        which is reliable from Databricks Apps; the Volume route was not.
        Pass ``None`` (or omit) in new cells; the parameter will be removed
        in a future release.
    materialize_views
        Optional sequence of ``MaterializedViewSpec``. For each spec, after
        the view DDLs in ``profile_sql`` are published, a
        ``CREATE OR REPLACE TABLE <table_name> USING DELTA AS SELECT * FROM
        <view_name>`` + ``OPTIMIZE ZORDER BY (<zorder_col>)`` + view re-point
        pass collapses per-click reads from re-executing the view body into
        a point lookup on a Z-ORDER'd Delta table. Use this for project-side
        views that the dashboard reads on every L4 click (e.g. per-account
        profile views) -- without it, each click re-runs the full multi-join
        view body and stalls in the tens of seconds.
    """
    if template_volume_path is not None:
        logger.warning(
            "apply_profile_override: ``template_volume_path`` is deprecated "
            "and ignored -- the HTML body is now appended to the UC table "
            "``%s.%s.dashboard_template_overrides`` and exposed via "
            "``v_dashboard_template_active``. Drop the kwarg from your "
            "c05 user-code cell on the next edit.",
            catalog, schema,
        )
    if spark is None:
        logger.warning("apply_profile_override: spark is None — skipping publish")
        return ProfileOverrideResult([], "", "", "", "")

    if not composite_name:
        raise ValueError(
            "composite_name is required (used both as the UC-row key for the "
            "template and as the {composite_name} substitution in profile_sql)"
        )

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
    statements = [s for s in split_view_statements(rendered) if _has_executable(s)]
    if not statements:
        raise ValueError("profile_sql contained no executable statements")

    for stmt in statements:
        spark.sql(stmt)
        logger.info("published profile view (%d chars)", len(stmt))

    table_fqn = f"{catalog}.{schema}.{_TEMPLATE_TABLE_NAME}"
    _ensure_template_table(spark, table_fqn)

    updated_by = _resolve_updated_by(spark)
    updated_at = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)

    # Append-only write: the v_dashboard_template_active view selects the most
    # recent row per composite_name via ROW_NUMBER. History stays in the table
    # so operators can audit who changed the template when, and revert by
    # re-publishing an older HTML body.
    from pyspark.sql import Row
    df = spark.createDataFrame([Row(
        composite_name=composite_name,
        profile_html=profile_html,
        updated_at=updated_at,
        updated_by=updated_by,
    )])
    df.write.mode("append").saveAsTable(table_fqn)
    logger.info(
        "appended profile template row -> %s (composite_name=%r, %d chars)",
        table_fqn, composite_name, len(profile_html),
    )

    materialized: List[str] = []
    for spec in (materialize_views or ()):
        if materialize_view_as_table(spark, catalog, schema, spec):
            materialized.append(spec.view_name)

    return ProfileOverrideResult(
        published_views=_extract_view_names(statements),
        template_table_fqn=table_fqn,
        composite_name=composite_name,
        updated_at=updated_at.isoformat(timespec="seconds"),
        updated_by=updated_by,
        materialized_views=materialized,
    )
