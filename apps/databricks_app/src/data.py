"""SQL data access — reads from the causal-track Unity Catalog views.

One SQL Warehouse connection, one cached DataFrame per query. The connection
uses workspace identity in Databricks Apps (no credentials passed) and falls
back to DATABRICKS_HOST / DATABRICKS_TOKEN locally.

The queries here mirror the Unity Catalog views we already publish from
`src/customer_retention/stages/causal/sql/dashboard_views.sql`:
  - v_portfolio_risk_matrix      — (playbook, risk_tier) rollup
  - v_playbook_archetype_rollup  — (playbook, archetype, risk_tier) rollup
  - v_eligible_all_playbooks     — per-account eligible pairs (top 500 per slice)
  - v_account_explanation        — per-account joined context for detail panel
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

from .config import AppConfig, load_config


def _warehouse_http_path(warehouse_id: str) -> str:
    return f"/sql/1.0/warehouses/{warehouse_id}"


@contextmanager
def _connect(cfg: AppConfig):
    """Open a SQL connection using the Databricks SDK's unified auth chain.

    In Databricks Apps this resolves to the app's on-behalf-of identity.
    Locally it reads DATABRICKS_HOST / DATABRICKS_TOKEN from env or `.databrickscfg`.
    """
    if not cfg.warehouse_id:
        raise RuntimeError("CR_WAREHOUSE_ID is not set. Update app.yaml or your .env.")
    sdk_cfg = Config()  # host/token/etc. resolved from standard chain
    conn = sql.connect(
        server_hostname=sdk_cfg.host.replace("https://", "").rstrip("/"),
        http_path=_warehouse_http_path(cfg.warehouse_id),
        credentials_provider=lambda: sdk_cfg.authenticate,
    )
    try:
        yield conn
    finally:
        conn.close()


def _query(cfg: AppConfig, sql_text: str, params: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    with _connect(cfg) as conn:
        cur = conn.cursor()
        try:
            if params:
                cur.execute(sql_text, params)
            else:
                cur.execute(sql_text)
            return cur.fetchall_arrow().to_pandas()
        finally:
            cur.close()


# ---------------------------------------------------------------------------
# Cached readers
# ---------------------------------------------------------------------------
# All queries are lightweight (< a few thousand rows) because upstream views
# are pre-aggregated or capped at 500 rows per slice. Cache for 60s so the
# app is responsive without going stale during a scoring run.


@st.cache_data(ttl=60, show_spinner=False)
def portfolio_by_playbook_risk_tier() -> pd.DataFrame:
    cfg = load_config()
    return _query(cfg, f"""
        SELECT playbook_name, risk_tier,
               SUM(eligible_count) AS eligible_count,
               SUM(recommended_count) AS recommended_count,
               SUM(holdout_count) AS holdout_count,
               SUM(total_value_at_risk) AS total_value_at_risk,
               AVG(mean_churn_probability) AS mean_churn_probability
        FROM {cfg.fqn_prefix}.v_portfolio_risk_matrix
        GROUP BY playbook_name, risk_tier
    """)


@st.cache_data(ttl=60, show_spinner=False)
def portfolio_totals() -> pd.DataFrame:
    """Per-account L1 KPI tiles for the latest scoring run.

    Each tile counts ENTITIES, not snapshot rows. ``eligibility_snapshot``
    has one row per ``(scoring_run_id, entity_id, playbook_id)`` so a row-sum
    multi-counts accounts that match more than one play. Filtering to
    ``policy_rank_among_eligible = 1`` collapses to the primary play per
    entity, so ``total_eligible`` / ``total_recommended`` answer "of all
    customers right now, how many are eligible / recommended" rather than
    "how many account-recommendations did the model emit".

    ``v_portfolio_risk_matrix`` is intentionally untouched -- its row-grain
    semantics are correct for capacity planning charts. We just stop using
    it as the source for the L1 tiles.
    """
    cfg = load_config()
    return _query(cfg, f"""
        WITH latest_run AS (
            SELECT scoring_run_id
            FROM {cfg.fqn_prefix}.eligibility_snapshot
            WHERE as_of_date = (
                SELECT MAX(as_of_date) FROM {cfg.fqn_prefix}.eligibility_snapshot
            )
            LIMIT 1
        ),
        primary_only AS (
            SELECT s.entity_id, s.recommended, s.value_at_risk, s.risk_tier
            FROM {cfg.fqn_prefix}.eligibility_snapshot s
            JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
            WHERE s.policy_rank_among_eligible = 1
              AND COALESCE(s.is_dashboard_visible, TRUE) = TRUE
        ),
        active_pb AS (
            SELECT COUNT(DISTINCT s.playbook_id) AS active_playbooks
            FROM {cfg.fqn_prefix}.eligibility_snapshot s
            JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
            WHERE s.recommended
        )
        SELECT
            COUNT(*)                                                      AS total_eligible,
            SUM(CASE WHEN recommended           THEN 1 ELSE 0 END)         AS total_recommended,
            SUM(COALESCE(value_at_risk, 0))                                AS total_value_at_risk,

            SUM(CASE WHEN risk_tier = 'High'   THEN 1 ELSE 0 END)         AS eligible_high,
            SUM(CASE WHEN risk_tier = 'Medium' THEN 1 ELSE 0 END)         AS eligible_medium,
            SUM(CASE WHEN risk_tier = 'Low'    THEN 1 ELSE 0 END)         AS eligible_low,

            SUM(CASE WHEN risk_tier = 'High'   AND recommended THEN 1 ELSE 0 END) AS recommended_high,
            SUM(CASE WHEN risk_tier = 'Medium' AND recommended THEN 1 ELSE 0 END) AS recommended_medium,
            SUM(CASE WHEN risk_tier = 'Low'    AND recommended THEN 1 ELSE 0 END) AS recommended_low,

            SUM(CASE WHEN risk_tier = 'High'   THEN COALESCE(value_at_risk, 0) ELSE 0 END) AS value_at_risk_high,
            SUM(CASE WHEN risk_tier = 'Medium' THEN COALESCE(value_at_risk, 0) ELSE 0 END) AS value_at_risk_medium,
            SUM(CASE WHEN risk_tier = 'Low'    THEN COALESCE(value_at_risk, 0) ELSE 0 END) AS value_at_risk_low,

            (SELECT active_playbooks FROM active_pb) AS active_playbooks
        FROM primary_only
    """)


@st.cache_data(ttl=60, show_spinner=False)
def archetypes_for_playbook(playbook_name: str) -> pd.DataFrame:
    cfg = load_config()
    return _query(cfg, f"""
        SELECT playbook_name, archetype_name, risk_tier,
               SUM(account_count) AS account_count,
               SUM(recommended_count) AS recommended_count,
               AVG(mean_churn_probability) AS mean_churn_probability,
               SUM(total_value_at_risk) AS total_value_at_risk
        FROM {cfg.fqn_prefix}.v_playbook_archetype_rollup
        WHERE playbook_name = :playbook_name
        GROUP BY playbook_name, archetype_name, risk_tier
    """, {"playbook_name": playbook_name})


@st.cache_data(ttl=60, show_spinner=False)
def accounts_in_scope(
    playbook_name: Optional[str] = None,
    archetype_name: Optional[str] = None,
    risk_tier: Optional[str] = None,
    limit: int = 500,
) -> pd.DataFrame:
    cfg = load_config()
    filters, params = [], {"limit": limit}
    if playbook_name:
        filters.append("playbook_name = :playbook_name")
        params["playbook_name"] = playbook_name
    if archetype_name:
        filters.append("archetype_name = :archetype_name")
        params["archetype_name"] = archetype_name
    if risk_tier:
        filters.append("risk_tier = :risk_tier")
        params["risk_tier"] = risk_tier
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    return _query(cfg, f"""
        SELECT entity_id, playbook_name, archetype_name, risk_tier,
               churn_probability, value_at_risk, expected_loss,
               policy_rank_among_eligible, eligible_playbook_count,
               recommended, is_holdout, eligibility_evidence
        FROM {cfg.fqn_prefix}.v_eligible_all_playbooks
        {where}
        ORDER BY expected_loss DESC
        LIMIT :limit
    """, params)


@st.cache_data(ttl=60, show_spinner=False)
def account_explanation(entity_id: str) -> pd.DataFrame:
    """Single-row view of a customer, joined to archetype + playbook + policy context."""
    cfg = load_config()
    return _query(cfg, f"""
        SELECT *
        FROM {cfg.fqn_prefix}.v_account_explanation
        WHERE entity_id = :entity_id
        ORDER BY policy_rank_among_eligible ASC
        LIMIT 1
    """, {"entity_id": entity_id})


@st.cache_data(ttl=300, show_spinner=False)
def run_context() -> pd.DataFrame:
    """Single-row projection for the app masthead.

    Returns an empty DataFrame when the view or the underlying table is absent
    (e.g. the causal track hasn't been re-run since ``v_run_context`` was
    introduced). The masthead gracefully degrades to a minimal title in that
    case.
    """
    cfg = load_config()
    try:
        return _query(cfg, f"SELECT * FROM {cfg.fqn_prefix}.v_run_context LIMIT 1")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def feature_provenance() -> pd.DataFrame:
    """Per-feature lineage + business definition for the latest run.

    One row per feature_name. Cached for 5 min — feature metadata is slow-
    changing and the table is small (<10k rows). Returns an empty DataFrame
    when ``v_feature_provenance`` is missing (e.g. older causal-track build).

    The view exposes both ``source_table`` (raw column from feature_meta)
    and ``source_dataset`` (alias for display, same value) plus
    ``composite_name`` (the run-level identifier surfaced as a hover
    tooltip beside the dataset name in the Feature dictionary table).
    """
    cfg = load_config()
    try:
        return _query(cfg, f"SELECT * FROM {cfg.fqn_prefix}.v_feature_provenance")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_template_html_from_uc(composite_name: Optional[str] = None) -> Optional[str]:
    """Read the active customer-profile HTML body from Unity Catalog.

    Reads from ``v_dashboard_template_active`` -- a view that returns the
    most recent row per ``composite_name`` from
    ``dashboard_template_overrides`` (where ``apply_profile_override``
    appends every published template). Replaces the prior volume-based
    auto-discovery (``Path("/Volumes/<cat>/<sch>/dashboard_templates").iterdir()``),
    which was unreliable from Databricks Apps because the App service
    principal cannot consistently read Volume FUSE mounts even with
    ``READ_VOLUME`` granted.

    Returns the HTML body as a string when found, ``None`` when the view
    has no matching row OR the view itself doesn't exist (older
    causal-track build, table not yet created). Callers fall back to the
    bundled ``default_profile.html`` on ``None``.
    """
    cfg = load_config()
    try:
        if composite_name:
            df = _query(
                cfg,
                f"""
                SELECT profile_html
                FROM {cfg.fqn_prefix}.v_dashboard_template_active
                WHERE composite_name = :composite_name
                LIMIT 1
                """,
                {"composite_name": composite_name},
            )
        else:
            # No composite_name supplied: take the most recently updated row
            # across all datasets. Single-dataset deploys hit this path.
            df = _query(
                cfg,
                f"""
                SELECT profile_html
                FROM {cfg.fqn_prefix}.v_dashboard_template_active
                ORDER BY updated_at DESC
                LIMIT 1
                """,
            )
    except Exception:
        return None
    if df.empty:
        return None
    html = df.iloc[0].get("profile_html")
    return str(html) if html else None


@st.cache_data(ttl=300, show_spinner=False)
def run_history() -> pd.DataFrame:
    cfg = load_config()
    return _query(cfg, f"""
        SELECT scoring_run_id, as_of_date, model_name, model_version,
               total_eligible_rows, recommended_rows, holdout_rows,
               mean_churn_probability
        FROM {cfg.fqn_prefix}.v_run_anchor_history
        ORDER BY as_of_date DESC
        LIMIT 20
    """)
