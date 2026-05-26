"""SQL data access — reads from the causal-track Unity Catalog views.

One SQL Warehouse connection per Streamlit session (held via
``st.cache_resource``) is reused across every cached reader and every
per-render query. Opening a Databricks SQL warehouse connection costs
~1-2s of handshake; with 3-5 queries per L4 render that adds up to
multi-second visible latency before any real query work happens.
Reusing one connection collapses that to a single handshake at session
start.

The connection uses workspace identity in Databricks Apps (no
credentials passed) and falls back to DATABRICKS_HOST /
DATABRICKS_TOKEN locally.

The queries here mirror the Unity Catalog views we already publish from
`src/customer_retention/stages/causal/sql/dashboard_views.sql`:
  - v_portfolio_risk_matrix      — (playbook, risk_tier) rollup
  - v_playbook_archetype_rollup  — (playbook, archetype, risk_tier) rollup
  - v_eligible_all_playbooks     — per-account eligible pairs (top 500 per slice)
  - v_account_explanation        — per-account joined context for detail panel
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Optional

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

from . import diagnostics
from .config import AppConfig, load_config


def _warehouse_http_path(warehouse_id: str) -> str:
    return f"/sql/1.0/warehouses/{warehouse_id}"


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------
# Every query in this module flows through ``_query`` and emits a
# ``diagnostics.record(...)`` event with timing + sql preview + attempt
# number. Connection rebuilds, stale-hint matches, and retries get their
# own events so the operator can see the full picture in the Diagnostics
# tab. ``diagnostics.record`` short-circuits when CR_SHOW_DIAGNOSTICS is
# off, so this instrumentation costs nothing in production.


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _short_sql(sql_text: str, limit: int = 100) -> str:
    one_line = " ".join(sql_text.split())
    return one_line[:limit] + ("..." if len(one_line) > limit else "")


# Substrings of the driver's exception messages that historically mean
# "the cached connection is no longer usable" -- Thrift-level closures
# from idle timeouts. After ``_query`` switched to a blanket retry on
# any first-attempt failure (see its docstring), these hints are no
# longer used as a retry GATE; they're kept for diagnostic labeling so
# the Diagnostics tab can distinguish "the obvious stale-conn" cases
# from blanket retries. Expanded with the longer-idle failure modes
# we observed empirically (token expiration, warehouse cold-start,
# cluster terminations, generic network) so the diagnostic events
# stay readable across all of them.
_STALE_CONNECTION_HINTS: tuple[str, ...] = (
    # Thrift / driver transport.
    "session was closed",
    "session is closed",
    "connection is closed",
    "connection was closed",
    "invalid session",
    "session handle is not present",
    "operation was canceled",
    "broken pipe",
    "thrifttransport",
    "connectionreseterror",
    "cursor was closed",
    # Auth / token expiry (multi-day idle eats PAT and OAuth tokens).
    "401",
    "403",
    "unauthorized",
    "authentication failed",
    "invalid_grant",
    "token has expired",
    "token expired",
    "could not be authenticated",
    # Warehouse state (long idle scales the warehouse down).
    "warehouse is not running",
    "warehouse must be running",
    "warehouse is stopped",
    "warehouse is starting",
    "warehouse is stopping",
    "warehouse is terminating",
    "no warehouses available",
    "compute is not running",
    # Network / transport at the HTTP layer.
    "name or service not known",
    "name resolution failure",
    "connection refused",
    "connection reset",
    "timed out",
    "read timed out",
    "request timed out",
    "service unavailable",
    "bad gateway",
    "502",
    "503",
    "504",
)


def _looks_like_stale_connection(exc: BaseException) -> bool:
    msg = str(exc).lower()
    for hint in _STALE_CONNECTION_HINTS:
        if hint in msg:
            diagnostics.record(
                "stale_hint_matched",
                hint=hint,
                exc_type=type(exc).__name__,
            )
            return True
    return False


@st.cache_resource(show_spinner=False)
def _shared_warehouse_connection(warehouse_id: str) -> Any:
    """Return a single Databricks SQL warehouse connection per session.

    ``st.cache_resource`` keeps one instance for the lifetime of the
    Streamlit session (across reruns and across users in shared apps it
    keys on the warehouse_id, so multiple deployments stay isolated).
    Every dashboard query borrows this connection via ``_connect``
    instead of paying the ~1-2s handshake cost on each query, which is
    what made the L4 render visibly slow (3-5 connect/close cycles per
    page click).

    Stale-connection recovery lives in ``_query`` -- when a borrowed
    cursor raises a "session closed" / "broken pipe" error we clear this
    cache entry and retry once against a fresh connection.
    """
    t0 = time.perf_counter()
    sdk_cfg = Config()
    conn = sql.connect(
        server_hostname=sdk_cfg.host.replace("https://", "").rstrip("/"),
        http_path=_warehouse_http_path(warehouse_id),
        credentials_provider=lambda: sdk_cfg.authenticate,
    )
    diagnostics.record("conn_rebuild", elapsed_ms=_ms(t0), reason="cache_miss")
    return conn


@contextmanager
def _connect(cfg: AppConfig):
    """Yield the session-shared SQL connection.

    No-op finalizer: closing the connection here would defeat the cache
    -- the next call would reopen and pay the handshake cost again. The
    connection lives until ``_shared_warehouse_connection.clear()`` is
    called (stale-connection recovery in ``_query``) or the Streamlit
    session ends.
    """
    if not cfg.warehouse_id:
        raise RuntimeError("CR_WAREHOUSE_ID is not set. Update app.yaml or your .env.")
    conn = _shared_warehouse_connection(cfg.warehouse_id)
    yield conn


def _query(cfg: AppConfig, sql_text: str, params: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    """Run a parameterized query against the shared warehouse connection.

    Stale-connection retry: ANY first-attempt failure drops the cached
    connection and reruns once against a fresh one. Previously this
    was gated on a string-match heuristic against
    ``_STALE_CONNECTION_HINTS``, but the hint set only covered Thrift
    transport-level errors ("session closed", "broken pipe"). After
    multi-day idle gaps the dominant failure modes are token expiry
    ("401 Unauthorized", "invalid_grant", "token has expired"), cold
    warehouse ("warehouse is not running", "warehouse is starting"),
    and network ("connection refused", "name resolution failure") --
    none of which matched the original hints, so the L1 panel would
    fail with no automatic recovery after a long weekend.

    Blanket retry on first failure costs one extra rebuild + one extra
    query attempt (~2-3s total) on legitimate errors (syntax error,
    missing table, permission denied); those errors fail identically on
    retry and propagate. The cost is negligible compared to the
    user-visible benefit of never serving stale-connection errors.

    The stale-hint substring match is kept as a DIAGNOSTIC label so the
    Diagnostics tab can distinguish "obviously stale" rebuilds from
    blanket first-attempt retries, but the retry no longer depends on
    a hint match.

    Tracing: when CR_QUERY_TRACE=1 in the environment, each attempt
    logs ``elapsed_ms``, attempt number, retry status, and the first
    100 chars of the SQL so the operator can see where an L4 render's
    time actually goes.
    """
    last_exc: Optional[BaseException] = None
    overall_t0 = time.perf_counter()
    for attempt in (0, 1):
        attempt_t0 = time.perf_counter()
        with _connect(cfg) as conn:
            cur = conn.cursor()
            try:
                if params:
                    cur.execute(sql_text, params)
                else:
                    cur.execute(sql_text)
                df = cur.fetchall_arrow().to_pandas()
                diagnostics.record(
                    "query_ok",
                    elapsed_ms=_ms(attempt_t0),
                    total_ms=_ms(overall_t0),
                    attempt=attempt,
                    rows=len(df),
                    sql=_short_sql(sql_text),
                )
                return df
            except Exception as exc:  # noqa: BLE001 -- classified below
                last_exc = exc
                will_retry = attempt == 0
                stale_hint = _looks_like_stale_connection(exc)
                diagnostics.record(
                    "query_fail",
                    elapsed_ms=_ms(attempt_t0),
                    attempt=attempt,
                    will_retry=will_retry,
                    stale_hint_matched=stale_hint,
                    exc_type=type(exc).__name__,
                    sql=_short_sql(sql_text),
                )
                if will_retry:
                    diagnostics.record(
                        "conn_clear",
                        reason="stale_hint" if stale_hint else "first_attempt_fail",
                        exc_type=type(exc).__name__,
                    )
                    _shared_warehouse_connection.clear()
                    continue
                raise
            finally:
                try:
                    cur.close()
                except Exception:  # noqa: BLE001 -- cursor close best-effort
                    pass
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_query exited the retry loop without returning")


def fetch_template_data_source(
    source: str,
    join_key: str,
    entity_id: str,
    *,
    order_by: Optional[str] = None,
    limit: int = 1,
) -> pd.DataFrame:
    """Fetch rows of a customer-profile template data source.

    Built on the shared connection via ``_query`` so the L4 render path
    doesn't open a second Databricks SQL connection on top of the ones
    the cached readers already borrow. Returns the raw DataFrame --
    list vs single-row context shaping is the caller's job.
    """
    cfg = load_config()
    order_clause = f"ORDER BY {order_by}" if order_by else ""
    limit_clause = f"LIMIT {int(limit)}"
    fqn = f"{cfg.fqn_prefix}.{source}"
    return _query(
        cfg,
        f"SELECT * FROM {fqn} WHERE `{join_key}` = :eid {order_clause} {limit_clause}",
        {"eid": entity_id},
    )


# ---------------------------------------------------------------------------
# Cached readers
# ---------------------------------------------------------------------------
# All queries are lightweight (< a few thousand rows) because upstream views
# are pre-aggregated or capped at 500 rows per slice. Cache for 60s so the
# app is responsive without going stale during a scoring run.


@st.cache_data(ttl=60, show_spinner=False)
def portfolio_by_archetype_risk_tier() -> pd.DataFrame:
    """Per-(archetype, risk_tier) rollup for the L1 treemap.

    Archetype is per-entity (one ``archetype_id`` per scored account in
    ``eligibility_snapshot``), so this is entity-grain by construction —
    tiles never overlap and summing across tiles equals the unique
    eligible-account count. That's the right grain for an L1 "where
    does the portfolio sit" view that doubles as a filter into the
    Playbook-recommendations treemap below.

    Counts are taken once per (entity, archetype, risk_tier) using
    ``policy_rank_among_eligible = 1`` -- otherwise an account matching
    N playbooks would be counted N times even though its archetype
    assignment is the same in every row.
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
        archetype_names AS (
            SELECT archetype_id, MAX(name) AS archetype_name
            FROM {cfg.fqn_prefix}.archetype_catalog
            WHERE status = 'active'
            GROUP BY archetype_id
        ),
        primary_only AS (
            SELECT s.entity_id, s.archetype_id, s.risk_tier,
                   s.value_at_risk, s.churn_probability,
                   s.recommended, s.is_holdout
            FROM {cfg.fqn_prefix}.eligibility_snapshot s
            JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
            WHERE s.policy_rank_among_eligible = 1
              AND COALESCE(s.is_dashboard_visible, TRUE) = TRUE
        )
        SELECT
            COALESCE(an.archetype_name, p.archetype_id) AS archetype_name,
            p.risk_tier,
            COUNT(*)                                       AS eligible_count,
            SUM(CASE WHEN p.recommended THEN 1 ELSE 0 END) AS recommended_count,
            SUM(CASE WHEN p.is_holdout  THEN 1 ELSE 0 END) AS holdout_count,
            SUM(COALESCE(p.value_at_risk, 0))              AS total_value_at_risk,
            AVG(p.churn_probability)                       AS mean_churn_probability
        FROM primary_only p
        LEFT JOIN archetype_names an ON p.archetype_id = an.archetype_id
        GROUP BY an.archetype_name, p.archetype_id, p.risk_tier
    """)


@st.cache_data(ttl=60, show_spinner=False)
def playbooks_for_archetype(archetype_name: str, risk_tier: Optional[str] = None) -> pd.DataFrame:
    """Per-(playbook, risk_tier) rollup for the L2 treemap, scoped to an archetype.

    Tile size = number of ENTITIES in the archetype whose PRIMARY play
    is this playbook. The earlier implementation counted rows of
    ``eligibility_snapshot`` -- which has one row per (entity, playbook)
    eligibility match -- so when every entity in an archetype was
    eligible for every play, every tile under that archetype showed an
    identical count (the full archetype's cohort), not its own. Sourcing
    from ``v_account_primary_recommendation`` (one row per entity at
    its primary play) restores the intended grain: summing tiles within
    one archetype/tier equals the L1 archetype/tier count, never more.

    Colour = ``fit_score``, the model's per-(playbook, archetype)
    goodness-of-match value (already joined onto the source view, so we
    don't need a separate ``eligibility_policy`` explode here).

    When ``risk_tier`` is supplied the L1 click drilled into a specific
    tier; we subset here so the L2 treemap shows only that slice across
    every playbook -- matching the "whatever classes are visible on the
    treemap remain on lower levels" intent.
    """
    cfg = load_config()
    params: dict[str, Any] = {"archetype_name": archetype_name}
    risk_tier_filter = ""
    if risk_tier:
        risk_tier_filter = "AND p.risk_tier = :risk_tier"
        params["risk_tier"] = risk_tier
    return _query(cfg, f"""
        SELECT
            p.archetype_name,
            p.playbook_name,
            p.playbook_id,
            p.risk_tier,
            COUNT(*)                                            AS eligible_count,
            SUM(CASE WHEN p.recommended THEN 1 ELSE 0 END)      AS recommended_count,
            SUM(COALESCE(p.value_at_risk, 0))                   AS total_value_at_risk,
            AVG(p.churn_probability)                            AS mean_churn_probability,
            MAX(p.fit_score)                                    AS fit_score,
            MAX(p.expected_uplift_pct)                          AS expected_uplift_pct
        FROM {cfg.fqn_prefix}.v_account_primary_recommendation p
        WHERE p.archetype_name = :archetype_name
          {risk_tier_filter}
        GROUP BY p.archetype_name, p.playbook_name, p.playbook_id, p.risk_tier
    """, params)


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
            SELECT s.entity_id, s.recommended, s.value_at_risk, s.risk_tier,
                   s.churn_probability
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
            -- Probability-weighted dollar exposure across the eligible
            -- cohort. churn_probability is 0..1 and value_at_risk is
            -- per-account ARR -- summing the product gives "if we did
            -- nothing this cycle, this is the realistic $ projection of
            -- what we'd lose". Strictly more useful for portfolio
            -- managers than raw value_at_risk (which is worst-case).
            SUM(COALESCE(churn_probability, 0.0) * COALESCE(value_at_risk, 0.0))
                                                                          AS total_expected_loss,

            SUM(CASE WHEN risk_tier = 'High'   THEN 1 ELSE 0 END)         AS eligible_high,
            SUM(CASE WHEN risk_tier = 'Medium' THEN 1 ELSE 0 END)         AS eligible_medium,
            SUM(CASE WHEN risk_tier = 'Low'    THEN 1 ELSE 0 END)         AS eligible_low,

            SUM(CASE WHEN risk_tier = 'High'   AND recommended THEN 1 ELSE 0 END) AS recommended_high,
            SUM(CASE WHEN risk_tier = 'Medium' AND recommended THEN 1 ELSE 0 END) AS recommended_medium,
            SUM(CASE WHEN risk_tier = 'Low'    AND recommended THEN 1 ELSE 0 END) AS recommended_low,

            SUM(CASE WHEN risk_tier = 'High'   THEN COALESCE(value_at_risk, 0) ELSE 0 END) AS value_at_risk_high,
            SUM(CASE WHEN risk_tier = 'Medium' THEN COALESCE(value_at_risk, 0) ELSE 0 END) AS value_at_risk_medium,
            SUM(CASE WHEN risk_tier = 'Low'    THEN COALESCE(value_at_risk, 0) ELSE 0 END) AS value_at_risk_low,

            SUM(CASE WHEN risk_tier = 'High'   THEN COALESCE(churn_probability, 0.0) * COALESCE(value_at_risk, 0.0) ELSE 0 END) AS expected_loss_high,
            SUM(CASE WHEN risk_tier = 'Medium' THEN COALESCE(churn_probability, 0.0) * COALESCE(value_at_risk, 0.0) ELSE 0 END) AS expected_loss_medium,
            SUM(CASE WHEN risk_tier = 'Low'    THEN COALESCE(churn_probability, 0.0) * COALESCE(value_at_risk, 0.0) ELSE 0 END) AS expected_loss_low,

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
def playbook_catalog_all() -> pd.DataFrame:
    """Latest-version row per playbook from ``playbook_catalog``.

    The catalog is append-only — every YAML re-ingest writes a new
    ``(playbook_id, version)`` row. We project the most-recently-active
    row per playbook so the master table shows one line per intervention
    design. Eligible / recommended counts come from
    ``v_capacity_utilization`` for the latest scoring run; left-joined so
    a freshly-loaded playbook with no scoring yet still renders.
    """
    cfg = load_config()
    return _query(cfg, f"""
        WITH latest_pb AS (
            SELECT
                playbook_id,
                version,
                name,
                description,
                when_applicable,
                policy_summary,
                expected_effect,
                cost_per_customer_default,
                expected_uplift_pct_default,
                outcome_windows_days,
                outcome_definition_version,
                time_zero_definition,
                grace_period_days,
                followup_start_rule,
                followup_end_rule,
                default_estimand,
                analysis_population_rule,
                active_from,
                active_to,
                ROW_NUMBER() OVER (
                    PARTITION BY playbook_id
                    ORDER BY active_from DESC NULLS LAST, version DESC
                ) AS rn
            FROM {cfg.fqn_prefix}.playbook_catalog
        ),
        capacity AS (
            SELECT playbook_id,
                   SUM(eligible_count)    AS eligible_count,
                   SUM(recommended_count) AS recommended_count,
                   SUM(holdout_count)     AS holdout_count
            FROM {cfg.fqn_prefix}.v_capacity_utilization
            GROUP BY playbook_id
        ),
        step_counts AS (
            SELECT playbook_id, playbook_version, COUNT(*) AS step_count
            FROM {cfg.fqn_prefix}.playbook_steps
            GROUP BY playbook_id, playbook_version
        )
        SELECT
            p.playbook_id,
            p.version,
            p.name,
            p.description,
            p.when_applicable,
            p.policy_summary,
            p.expected_effect,
            p.cost_per_customer_default,
            p.expected_uplift_pct_default,
            p.outcome_windows_days,
            p.outcome_definition_version,
            p.time_zero_definition,
            p.grace_period_days,
            p.followup_start_rule,
            p.followup_end_rule,
            p.default_estimand,
            p.analysis_population_rule,
            p.active_from,
            p.active_to,
            COALESCE(c.eligible_count, 0)    AS eligible_count,
            COALESCE(c.recommended_count, 0) AS recommended_count,
            COALESCE(c.holdout_count, 0)     AS holdout_count,
            COALESCE(sc.step_count, 0)       AS step_count
        FROM latest_pb p
        LEFT JOIN capacity   c  ON c.playbook_id = p.playbook_id
        LEFT JOIN step_counts sc ON sc.playbook_id = p.playbook_id
                               AND sc.playbook_version = p.version
        WHERE p.rn = 1
        ORDER BY COALESCE(c.recommended_count, 0) DESC, p.name ASC
    """)


@st.cache_data(ttl=300, show_spinner=False)
def playbook_steps_for(playbook_id: str, version: str) -> pd.DataFrame:
    """Ordered step rows for one ``(playbook_id, version)`` pair."""
    cfg = load_config()
    return _query(cfg, f"""
        SELECT step_id, step_sequence, step_name, action_type,
               automation_level, owner_role, cadence_trigger,
               cadence_relative_to, cadence_offset_days, cadence_condition,
               timeout_days, response_schema_id, intensity_param_name,
               skip_conditions, stop_conditions
        FROM {cfg.fqn_prefix}.playbook_steps
        WHERE playbook_id = :pid AND playbook_version = :ver
        ORDER BY step_sequence ASC
    """, {"pid": playbook_id, "ver": version})


@st.cache_data(ttl=300, show_spinner=False)
def archetype_catalog_all() -> pd.DataFrame:
    """Active archetypes from ``v_archetype_overview``.

    One row per active archetype with cluster stats and the top-SHAP
    driver array. The view is already filtered to ``status = 'active'``
    so the master table reflects what's deployed against the latest
    model build.
    """
    cfg = load_config()
    return _query(cfg, f"""
        SELECT
            archetype_id,
            archetype_version,
            archetype_name,
            archetype_description,
            rationale,
            cluster_size,
            cluster_mean_churn_probability,
            top_shap_features,
            feature_thresholds,
            model_name,
            model_version,
            derivation_method,
            stability_vs_prior_version,
            status,
            valid_from,
            valid_to
        FROM {cfg.fqn_prefix}.v_archetype_overview
        ORDER BY cluster_mean_churn_probability DESC NULLS LAST, cluster_size DESC
    """)


@st.cache_data(ttl=600, show_spinner=False)
def playbook_yaml_runbook(playbook_id: str) -> dict | None:
    """Return the parsed playbook YAML (or ``None`` when unavailable).

    The framework's playbook loader writes the ``catalog`` and ``steps``
    blocks to UC but deliberately drops ``operator_runbook`` -- that block
    is operator content, not model-derived. To surface it in the dashboard
    without a schema migration we read the YAML directly from the same
    playbooks directory the loader uses.

    Resolution order for the playbooks root:
      1. ``CR_PLAYBOOKS_DIR`` env var (Volume path like
         ``/Volumes/<catalog>/<schema>/playbooks`` or any filesystem dir).
         Volume paths are read via the Databricks SDK ``files`` API; local
         paths via ``Path.read_text``.
      2. Repo-relative ``playbooks/`` discovered by walking up from this
         file (works in local dev).
      3. Give up — return ``None`` so the UI gracefully falls back to the
         catalog-only layout.

    Per-playbook caching is keyed on ``playbook_id`` so a click into a
    different playbook only hits one file read.
    """
    import os
    from pathlib import Path

    import yaml

    candidates = []
    env_dir = os.environ.get("CR_PLAYBOOKS_DIR")
    if env_dir:
        candidates.append(env_dir)
    here = Path(__file__).resolve()
    # apps/databricks_app/src/data.py -> repo_root/playbooks
    for parent in here.parents:
        candidate = parent / "playbooks"
        if candidate.is_dir():
            candidates.append(str(candidate))
            break

    for root in candidates:
        text = _read_playbook_yaml(root, playbook_id)
        if text is None:
            continue
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _read_playbook_yaml(root: str, playbook_id: str) -> str | None:
    """Find ``<playbook_id>.yaml`` under ``root`` and return its contents.

    Searches the top level and the first level of subdirectories so the
    SPS/email split (``playbooks/sps/foo.yaml`` vs ``playbooks/email/bar.yaml``)
    works without configuration. Volume paths (starting ``/Volumes/``)
    go through the Databricks SDK file API; everything else uses the
    local filesystem. Returns ``None`` when nothing matches.
    """
    target = f"{playbook_id}.yaml"
    if root.startswith("/Volumes/"):
        return _read_yaml_from_volume(root, target)
    from pathlib import Path
    base = Path(root)
    if not base.is_dir():
        return None
    candidates = [base / target]
    for child in sorted(base.iterdir()):
        if child.is_dir():
            candidates.append(child / target)
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
    return None


def _read_yaml_from_volume(root: str, target: str) -> str | None:
    """Read ``<root>/<sub>/<target>`` from a Unity Catalog Volume.

    Uses ``WorkspaceClient.files.download`` -- the supported SDK path that
    works reliably from Databricks Apps where FUSE is unreliable. Walks
    the top level then one level of subdirectories so the SPS/email split
    is auto-discovered.
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        return None
    try:
        w = WorkspaceClient()
    except Exception:
        return None

    def _try_path(path: str) -> str | None:
        try:
            resp = w.files.download(path)
        except Exception:
            return None
        try:
            data = resp.contents.read()
        except Exception:
            return None
        try:
            return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
        except Exception:
            return None

    # Try top-level first.
    direct = _try_path(f"{root.rstrip('/')}/{target}")
    if direct is not None:
        return direct
    # Walk one level of subdirectories.
    try:
        listing = list(w.files.list_directory_contents(root))
    except Exception:
        return None
    for entry in listing:
        # The SDK returns a DirectoryEntry-like object; both ``is_directory``
        # and ``path`` are present on the variants we care about.
        try:
            is_dir = bool(getattr(entry, "is_directory", False))
            entry_path = getattr(entry, "path", None)
        except Exception:
            continue
        if not is_dir or not entry_path:
            continue
        found = _try_path(f"{entry_path.rstrip('/')}/{target}")
        if found is not None:
            return found
    return None


@st.cache_data(ttl=300, show_spinner=False)
def feature_business_phrases() -> pd.DataFrame:
    """``feature_name → business_phrase`` map for archetype detail.

    Joined client-side onto the raw SHAP-feature list so the Archetypes
    tab can show a readable phrase next to each technical column name.
    Reads ``v_feature_provenance``; returns an empty DataFrame when the
    view is missing.
    """
    cfg = load_config()
    try:
        return _query(cfg, f"""
            SELECT feature_name, business_phrase, source_dataset,
                   aggregation_kind, window_phrase, polarity
            FROM {cfg.fqn_prefix}.v_feature_provenance
        """)
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


@st.cache_data(ttl=600, show_spinner=False)
def sample_entity_id_for_placeholder() -> Optional[str]:
    """Return one real ``entity_id`` from the latest scoring run, or ``None``.

    Used as the Search-tab placeholder so the operator sees a working ID
    they can copy verbatim instead of a fabricated example that won't
    resolve. Long-cached (10 min) -- the placeholder is decorative; it
    doesn't need to track the snapshot in real time. Returns ``None`` when
    the snapshot is empty or the query fails so the caller can fall back
    to a generic placeholder.
    """
    cfg = load_config()
    try:
        df = _query(cfg, f"""
            WITH latest_run AS (
                SELECT scoring_run_id
                FROM {cfg.fqn_prefix}.eligibility_snapshot
                WHERE as_of_date = (
                    SELECT MAX(as_of_date) FROM {cfg.fqn_prefix}.eligibility_snapshot
                )
                LIMIT 1
            )
            SELECT s.entity_id
            FROM {cfg.fqn_prefix}.eligibility_snapshot s
            JOIN latest_run lr ON s.scoring_run_id = lr.scoring_run_id
            ORDER BY s.entity_id
            LIMIT 1
        """)
    except Exception:
        return None
    if df.empty:
        return None
    eid = df.iloc[0]["entity_id"]
    if eid is None:
        return None
    return str(eid)


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
