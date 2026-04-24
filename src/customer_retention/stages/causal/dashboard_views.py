"""Loader + publisher for the six causal-track dashboard SQL views.

The view DDLs live as a single ``sql/dashboard_views.sql`` file shipped
inside the package. This module reads the file, substitutes the
``{catalog}`` / ``{schema}`` placeholders, splits it into individual
``CREATE OR REPLACE VIEW`` statements, and submits each one to Spark.

Keeping the SQL in a static file (rather than concatenated string
literals in Python) makes the views diffable, lintable with standard SQL
tooling, and reviewable in a dedicated ``.sql`` file by SQL-fluent
operators.
"""

from __future__ import annotations

import logging
from importlib.resources import files
from typing import TYPE_CHECKING, List

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession


_VIEW_FILE_NAME = "dashboard_views.sql"
DASHBOARD_VIEW_NAMES: tuple[str, ...] = (
    "v_ranked_at_risk_customers",
    "v_archetype_overview",
    "v_playbook_eligibility_rules",
    "v_holdout_assignments",
    "v_capacity_utilization",
    "v_run_anchor_history",
    "v_portfolio_risk_matrix",
    "v_playbook_archetype_rollup",
    "v_eligible_all_playbooks",
    "v_account_explanation",
    "v_run_context",
)


def load_dashboard_view_sql() -> str:
    """Return the raw SQL text shipped at ``stages/causal/sql/dashboard_views.sql``."""
    resource = files("customer_retention.stages.causal.sql") / _VIEW_FILE_NAME
    return resource.read_text(encoding="utf-8")


def render_dashboard_view_sql(catalog: str, schema: str) -> str:
    """Substitute ``{catalog}`` / ``{schema}`` into the raw SQL template.

    Uses ``str.replace`` rather than ``str.format`` so other braces in the
    SQL (none today, but possible for JSON / map literals later) are
    preserved untouched.
    """
    return load_dashboard_view_sql().replace("{catalog}", catalog).replace("{schema}", schema)


def split_view_statements(sql_text: str) -> List[str]:
    """Split a multi-statement SQL string on semicolons.

    Comments (``--`` lines) are kept on the preceding statement so view
    headers stay readable in error messages. Empty trailing statements
    after the final semicolon are dropped.
    """
    statements: List[str] = []
    for raw in sql_text.split(";"):
        stripped = raw.strip()
        if not stripped:
            continue
        statements.append(stripped)
    return statements


def publish_dashboard_views(spark: "SparkSession", catalog: str, schema: str) -> List[str]:
    """Execute every dashboard view DDL in order.

    Returns the list of statements that were submitted (one per view).
    Each statement is a ``CREATE OR REPLACE VIEW`` so re-running is a
    no-op when the underlying schema is unchanged.
    """
    rendered = render_dashboard_view_sql(catalog, schema)
    statements = split_view_statements(rendered)
    submitted: List[str] = []
    for stmt in statements:
        spark.sql(stmt)
        submitted.append(stmt)
        logger.info("published dashboard view (%d chars)", len(stmt))
    return submitted
