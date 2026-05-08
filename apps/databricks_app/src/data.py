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
    # Multi-key ORDER BY so the table stays useful even when expected_loss
    # ties at 0 across thousands of rows (Low-tier slices with NULL
    # value_at_risk). Risk-tier rank (High before Medium before Low) is the
    # primary fallback so the user's click on "High" surfaces High rows at
    # the top even when the click hasn't propagated as a hard filter.
    # entity_id ASC is the deterministic final tiebreaker so the same K
    # rows return run-over-run.
    return _query(cfg, f"""
        SELECT entity_id, playbook_name, archetype_name, risk_tier,
               churn_probability, value_at_risk, expected_loss,
               policy_rank_among_eligible, eligible_playbook_count,
               recommended, is_holdout, eligibility_evidence
        FROM {cfg.fqn_prefix}.v_eligible_all_playbooks
        {where}
        ORDER BY
            expected_loss DESC,
            CASE risk_tier
                 WHEN 'High'   THEN 0
                 WHEN 'Medium' THEN 1
                 WHEN 'Low'    THEN 2
                 ELSE 3
            END ASC,
            churn_probability DESC,
            value_at_risk DESC NULLS LAST,
            entity_id ASC
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


# ---------------------------------------------------------------------------
# Self-assignment — ``account_assignments`` Delta table
# ---------------------------------------------------------------------------
# A CSM clicks "Assign to me" on the L4 customer profile to claim ownership
# of an account. One row per assigned entity: no row means unassigned. The
# table is additive (no view modification, no schema migration) and lazy-
# created on first read or write so the dashboard can drop into existing
# deployments without an out-of-band DDL step.
#
# Race semantics: two CSMs clicking the same unassigned entity within the
# same second land on a Delta MERGE with ``WHEN NOT MATCHED THEN INSERT``,
# so exactly one INSERT wins and the loser's next rerun reads the winner's
# row and renders "Claimed by …". Unassign is restricted to the current
# owner -- you cannot clear someone else's claim from the UI.
_ASSIGNMENTS_TABLE = "account_assignments"

_TOGGLE_ASSIGNED = "assigned"
_TOGGLE_UNASSIGNED = "unassigned"
_TOGGLE_CLAIMED_BY_OTHER = "claimed_by_other"


def _assignments_fqn(cfg: AppConfig) -> str:
    return f"{cfg.fqn_prefix}.{_ASSIGNMENTS_TABLE}"


def _ensure_assignments_table(cfg: AppConfig) -> None:
    """Idempotent ``CREATE TABLE IF NOT EXISTS`` for the assignments table.

    Runs once per process — the Streamlit session keeps a flag so we don't
    fire DDL on every rerun. Safe to call concurrently: ``CREATE TABLE IF
    NOT EXISTS`` is a no-op on a pre-existing table.
    """
    flag_key = f"_assignments_table_ready::{cfg.fqn_prefix}"
    try:
        if st.session_state.get(flag_key):
            return
    except Exception:
        pass
    with _connect(cfg) as conn:
        cur = conn.cursor()
        try:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {_assignments_fqn(cfg)} (
                    entity_id    STRING NOT NULL,
                    assigned_to  STRING NOT NULL,
                    assigned_at  TIMESTAMP NOT NULL
                ) USING DELTA
            """)
        finally:
            cur.close()
    try:
        st.session_state[flag_key] = True
    except Exception:
        pass


@st.cache_data(ttl=10, show_spinner=False)
def assignments() -> pd.DataFrame:
    """Return all current account assignments.

    Two columns: ``entity_id`` and ``assigned_to`` (the owner's email).
    Cached for 10s so the In-scope table stays fresh without hammering the
    warehouse on every rerun. Bust the cache via ``data.assignments.clear()``
    after a toggle so the UI reflects the change immediately.

    Returns an empty DataFrame on the first call before any rows exist —
    callers can safely left-merge against it.
    """
    cfg = load_config()
    _ensure_assignments_table(cfg)
    return _query(
        cfg,
        f"SELECT entity_id, assigned_to FROM {_assignments_fqn(cfg)}",
    )


def toggle_assignment(entity_id: str, user_email: str) -> str:
    """Self-assign or unassign ``entity_id`` for ``user_email``.

    Returns one of:
      - ``"assigned"``           — row was inserted (entity now owned by user)
      - ``"unassigned"``         — row was deleted (user released their own claim)
      - ``"claimed_by_other"``   — entity is owned by someone else; no DML

    The MERGE-on-INSERT path protects against the read-then-write race
    where another CSM claimed the entity between our SELECT and our
    INSERT: ``WHEN NOT MATCHED`` collapses to a no-op and we surface
    ``claimed_by_other`` on the next rerun via the cached read.
    """
    if not entity_id or not user_email:
        raise ValueError("entity_id and user_email are required")
    cfg = load_config()
    _ensure_assignments_table(cfg)
    fqn = _assignments_fqn(cfg)
    user = user_email.strip().lower()

    with _connect(cfg) as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT assigned_to FROM {fqn} WHERE entity_id = :eid LIMIT 1",
                {"eid": entity_id},
            )
            rows = cur.fetchall()
            current = rows[0][0] if rows else None

            if current is not None and current.strip().lower() != user:
                return _TOGGLE_CLAIMED_BY_OTHER

            if current is not None:
                cur.execute(
                    f"DELETE FROM {fqn} WHERE entity_id = :eid AND assigned_to = :user",
                    {"eid": entity_id, "user": user},
                )
                return _TOGGLE_UNASSIGNED

            # No prior row -- MERGE so a concurrent claim by another CSM
            # cannot produce a duplicate row. The source row carries the
            # candidate (entity, user, now()); the target only inserts when
            # ``entity_id`` is still absent.
            cur.execute(
                f"""
                MERGE INTO {fqn} t
                USING (
                    SELECT :eid AS entity_id, :user AS assigned_to, current_timestamp() AS assigned_at
                ) s
                ON t.entity_id = s.entity_id
                WHEN NOT MATCHED THEN INSERT (entity_id, assigned_to, assigned_at)
                                  VALUES (s.entity_id, s.assigned_to, s.assigned_at)
                """,
                {"eid": entity_id, "user": user},
            )

            cur.execute(
                f"SELECT assigned_to FROM {fqn} WHERE entity_id = :eid LIMIT 1",
                {"eid": entity_id},
            )
            verify = cur.fetchall()
            winner = verify[0][0] if verify else None
            if winner is None:
                return _TOGGLE_CLAIMED_BY_OTHER
            if winner.strip().lower() == user:
                return _TOGGLE_ASSIGNED
            return _TOGGLE_CLAIMED_BY_OTHER
        finally:
            cur.close()


def assignment_for(entity_id: str) -> str | None:
    """Return the current owner email for ``entity_id``, or ``None``.

    Reads through the cached ``assignments()`` frame so the L4 button can
    label itself ("Assign to me" / "Unassign" / "Claimed") without firing
    its own SQL on every rerun.
    """
    if not entity_id:
        return None
    df = assignments()
    if df is None or df.empty or "entity_id" not in df.columns:
        return None
    hit = df[df["entity_id"].astype(str) == str(entity_id)]
    if hit.empty:
        return None
    return str(hit.iloc[0]["assigned_to"])


@st.cache_data(ttl=60, show_spinner=False)
def entity_exists_in_latest_run(entity_id: str) -> bool:
    """Return ``True`` when ``entity_id`` was scored in the latest run.

    Used by the Search tab to validate the typed ID before rendering the
    profile. Reads ``eligibility_snapshot`` rather than ``v_account_explanation``
    because the snapshot is the canonical "scored entities" pool — covers
    every entity the model produced a prediction for, not just the in-scope
    top-K. SHAP availability is a separate concern handled by the template
    (which gates the driver block on ``account_top_shap_features``).
    """
    if not entity_id or not entity_id.strip():
        return False
    cfg = load_config()
    df = _query(
        cfg,
        f"""
        WITH latest_run AS (
            SELECT scoring_run_id
            FROM {cfg.fqn_prefix}.eligibility_snapshot
            WHERE as_of_date = (
                SELECT MAX(as_of_date) FROM {cfg.fqn_prefix}.eligibility_snapshot
            )
            LIMIT 1
        )
        SELECT 1 AS hit
        FROM {cfg.fqn_prefix}.eligibility_snapshot s
        JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
        WHERE s.entity_id = :eid
        LIMIT 1
        """,
        {"eid": entity_id.strip()},
    )
    return not df.empty


def account_is_holdout(entity_id: str) -> bool:
    """Return ``True`` when ``entity_id`` is a holdout row.

    Reads the ``is_holdout`` field from the same cached ``account_explanation``
    call that ``customer_profile.render()`` already issues for the L4 panel,
    so this lookup adds zero extra SQL when the profile is open. Holdout
    accounts are deliberately excluded from CSM intervention to measure
    model lift; the L4 button uses this to refuse self-assignment.
    """
    if not entity_id:
        return False
    df = account_explanation(entity_id)
    if df is None or df.empty or "is_holdout" not in df.columns:
        return False
    val = df.iloc[0]["is_holdout"]
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return bool(val)
