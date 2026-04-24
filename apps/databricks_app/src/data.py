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
    cfg = load_config()
    return _query(cfg, f"""
        SELECT SUM(eligible_count) AS total_eligible,
               SUM(recommended_count) AS total_recommended,
               SUM(total_value_at_risk) AS total_value_at_risk,
               COUNT(DISTINCT playbook_id) AS active_playbooks
        FROM {cfg.fqn_prefix}.v_portfolio_risk_matrix
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
