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
import re
from importlib.resources import files
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession


_VIEW_FILE_NAME = "dashboard_views.sql"

# Sentinel-delimited block in the SQL file that is only emitted when a concrete
# composite_name is supplied. Keep the markers in sync with the .sql file.
_DEVIATION_BLOCK_OPEN = "-- @cr:deviation-block:open"
_DEVIATION_BLOCK_CLOSE = "-- @cr:deviation-block:close"

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
DASHBOARD_DEVIATION_VIEW_NAMES: tuple[str, ...] = (
    "v_account_feature_deviation",
    "v_account_feature_deviation_topn",
)


def load_dashboard_view_sql() -> str:
    """Return the raw SQL text shipped at ``stages/causal/sql/dashboard_views.sql``."""
    resource = files("customer_retention.stages.causal.sql") / _VIEW_FILE_NAME
    return resource.read_text(encoding="utf-8")


def _strip_deviation_block(sql_text: str) -> str:
    """Drop the ``@cr:deviation-block`` section so the rendered SQL parses.

    The deviation views reference ``gold_features_{composite_name}`` which has
    no inert default; when no composite name is supplied the block is excised
    rather than left with an unbound placeholder.
    """
    pattern = re.compile(
        re.escape(_DEVIATION_BLOCK_OPEN) + r".*?" + re.escape(_DEVIATION_BLOCK_CLOSE),
        re.DOTALL,
    )
    return pattern.sub("", sql_text)


def _strip_deviation_markers(sql_text: str) -> str:
    """Drop the sentinel-marker lines themselves so they don't survive as
    standalone comment-only chunks after ``split_view_statements``.
    """
    return "\n".join(
        line for line in sql_text.splitlines()
        if line.strip() not in (_DEVIATION_BLOCK_OPEN, _DEVIATION_BLOCK_CLOSE)
    )


def render_dashboard_view_sql(
    catalog: str,
    schema: str,
    *,
    composite_name: Optional[str] = None,
) -> str:
    """Substitute ``{catalog}`` / ``{schema}`` (and optionally ``{composite_name}``)
    into the raw SQL template.

    When ``composite_name`` is omitted, the deviation views (which reference
    ``gold_features_{composite_name}``) are stripped from the output so the
    remaining DDL stays parseable.  Existing callers that don't supply the
    parameter keep their previous behaviour.
    """
    text = load_dashboard_view_sql()
    if composite_name:
        text = _strip_deviation_markers(text).replace("{composite_name}", composite_name)
    else:
        text = _strip_deviation_block(text)
    return text.replace("{catalog}", catalog).replace("{schema}", schema)


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


def publish_dashboard_views(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    *,
    composite_name: Optional[str] = None,
) -> List[str]:
    """Execute every dashboard view DDL in order.

    Returns the list of statements that were submitted (one per view).
    Each statement is a ``CREATE OR REPLACE VIEW`` so re-running is a
    no-op when the underlying schema is unchanged.

    Before submitting, the ``run_context`` Delta table is ensured via
    idempotent ``CREATE TABLE IF NOT EXISTS`` — Spark validates the
    ``v_run_context`` view body against its referenced table at creation
    time, so this makes publish order-independent from the per-run
    ``write_run_context`` write.

    When ``composite_name`` is supplied, the deviation views
    (``v_account_feature_deviation`` / ``_topn``) are also published.  When
    omitted, those views are skipped — the gold table FQN is per-run and
    cannot be inferred from ``catalog`` / ``schema`` alone.
    """
    from .run_context_writer import ensure_run_context_table

    ensure_run_context_table(spark, f"{catalog}.{schema}.run_context")
    rendered = render_dashboard_view_sql(catalog, schema, composite_name=composite_name)
    statements = split_view_statements(rendered)
    submitted: List[str] = []
    for stmt in statements:
        spark.sql(stmt)
        submitted.append(stmt)
        logger.info("published dashboard view (%d chars)", len(stmt))
    if composite_name is None:
        logger.info(
            "deviation views skipped (no composite_name supplied; pass composite_name= to enable)"
        )
    return submitted
