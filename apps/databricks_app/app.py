"""Streamlit entry — progressive-disclosure CSM triage.

Four levels, one long scroll, strict detail gating:
  L1 Portfolio        — always visible: churn risk over the model horizon
  L2 Playbook drill   — appears after a playbook treemap click
  L3 Customer list    — appears alongside L2, scoped by the selection
  L4 Customer profile — appears after a row click

Nothing downstream is ever rendered until the upstream decision has been made.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from src.masthead import l1_title_html, masthead_title

from src import accounts_view, archetype_view, customer_profile, data, state, treemap


def _load_run_context() -> tuple[dict, str | None]:
    """Return ``(ctx, diagnostic)`` for the masthead.

    ``diagnostic`` is ``None`` on the happy path. When the run-context view
    is missing or empty the diagnostic surfaces the operator-actionable
    reason so the dashboard never silently degrades to a generic title.
    """
    try:
        df = data.run_context()
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim to operator
        return {}, f"v_run_context query failed: {type(exc).__name__}: {exc}"
    if df is None or df.empty:
        return {}, "v_run_context returned 0 rows — re-run c05 to publish run_context"
    row = df.iloc[0]

    def _get(col):
        if col not in df.columns:
            return None
        v = row[col]
        return None if pd.isna(v) else v

    ctx = {
        "horizon_days":       _get("horizon_days"),
        "primary_objective":  _get("primary_objective"),
        "temporal_posture":   _get("temporal_posture"),
        "model_type":         _get("model_type"),
        "model_name":         _get("model_name"),
    }
    # Only fields that drive the masthead/L1 title trigger the warning.
    # ``model_type`` is best-effort MLflow metadata and ``model_name`` is
    # decorative -- neither is required to render the dashboard, so a
    # missing one shouldn't surface a "context unavailable" banner.
    _title_driving = ("horizon_days", "primary_objective", "temporal_posture")
    missing = [k for k in _title_driving if ctx.get(k) is None]
    diag = (
        f"v_run_context row has NULL fields: {', '.join(missing)}"
        if missing else None
    )
    return ctx, diag


st.set_page_config(
    page_title="Churn Risk · CSM Triage",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="collapsed",
)
state.init()


# ---------------------------------------------------------------------------
# Theme injection
# ---------------------------------------------------------------------------
_theme_path = Path(__file__).parent / "src" / "theme.css"
st.markdown(f"<style>{_theme_path.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Masthead with inline breadcrumb
# ---------------------------------------------------------------------------
def _render_masthead(ctx: dict) -> None:
    parts = state.breadcrumb_parts()
    crumbs: list[str] = []
    for i, (label, level) in enumerate(parts):
        if i > 0:
            crumbs.append('<span class="crumb-sep">›</span>')
        crumbs.append(
            f'<span class="crumb crumb-{level}">{escape(str(label))}</span>'
        )
    trail_html = "".join(crumbs) if crumbs else ""

    title, segments = masthead_title(ctx)
    sub_html = (
        " &middot; ".join(escape(s) for s in segments)
        if segments
        else "CSM Triage"
    )
    st.markdown(
        f"""
        <header class="masthead">
          <div class="masthead-brand">
            <span class="brand-mark"></span>
            <span class="brand-title">{escape(title)}</span>
            <span class="brand-sub">{sub_html}</span>
          </div>
          <nav class="masthead-trail">{trail_html}</nav>
        </header>
        """,
        unsafe_allow_html=True,
    )


def _render_reset_bar() -> None:
    if len(state.breadcrumb_parts()) <= 1:
        return
    _, right = st.columns([6, 1])
    with right:
        if st.button("Reset drill", key="reset_btn", use_container_width=True):
            state.clear_all()
            st.rerun()


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------
def _level_header(
    level: int,
    eyebrow: str,
    title_html: str,
    lead: str | None = None,
) -> None:
    lead_html = f'<p class="level-lead">{escape(lead)}</p>' if lead else ""
    st.markdown(
        f"""
        <section class="level level-{level}">
          <div class="level-eyebrow"><span class="dot"></span>{escape(eyebrow)}</div>
          <h2 class="level-title">{title_html}</h2>
          {lead_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def _section_head(eyebrow: str, title: str, lead: str) -> None:
    st.markdown(
        f"""
        <section class="section-head">
          <div class="section-eyebrow">{escape(eyebrow)}</div>
          <h3 class="section-title">{escape(title)}</h3>
          <p class="section-lead">{escape(lead)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _compact_number(v) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{int(v):,}"


def _compact_currency(v) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:,.0f}"


def _safe_float(v) -> float:
    try:
        return float(v) if v is not None and not pd.isna(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _stat_breakdown_html(
    high: float,
    medium: float,
    low: float,
    *,
    formatter,
    high_label: str = "high",
    medium_label: str = "med",
    low_label: str = "low",
) -> str:
    total = high + medium + low
    if total <= 0:
        return ""
    pct_high = high / total * 100
    pct_med = medium / total * 100
    pct_low = max(0.0, 100 - pct_high - pct_med)
    return (
        '<div class="stat-bar" role="img" aria-label="risk-tier composition">'
        f'<span class="stat-bar-seg stat-bar-high" style="width:{pct_high:.4f}%"></span>'
        f'<span class="stat-bar-seg stat-bar-med" style="width:{pct_med:.4f}%"></span>'
        f'<span class="stat-bar-seg stat-bar-low" style="width:{pct_low:.4f}%"></span>'
        '</div>'
        '<div class="stat-tier-line">'
        f'<span class="stat-tier stat-tier-high"><span class="stat-tier-dot"></span>{formatter(high)} {high_label}</span>'
        f'<span class="stat-tier stat-tier-med"><span class="stat-tier-dot"></span>{formatter(medium)} {medium_label}</span>'
        f'<span class="stat-tier stat-tier-low"><span class="stat-tier-dot"></span>{formatter(low)} {low_label}</span>'
        '</div>'
    )


def _render_stat_row() -> None:
    try:
        df = data.portfolio_totals()
    except Exception as exc:
        st.warning(f"Could not load portfolio totals: {exc}")
        return
    if df.empty:
        return
    r = df.iloc[0]

    eligible = _compact_number(r.get("total_eligible"))
    recommended = _compact_number(r.get("total_recommended"))
    value_at_risk = _compact_currency(r.get("total_value_at_risk"))
    playbooks = _compact_number(r.get("active_playbooks"))

    eligible_breakdown = _stat_breakdown_html(
        _safe_float(r.get("eligible_high")),
        _safe_float(r.get("eligible_medium")),
        _safe_float(r.get("eligible_low")),
        formatter=_compact_number,
    )
    recommended_breakdown = _stat_breakdown_html(
        _safe_float(r.get("recommended_high")),
        _safe_float(r.get("recommended_medium")),
        _safe_float(r.get("recommended_low")),
        formatter=_compact_number,
    )
    var_breakdown = _stat_breakdown_html(
        _safe_float(r.get("value_at_risk_high")),
        _safe_float(r.get("value_at_risk_medium")),
        _safe_float(r.get("value_at_risk_low")),
        formatter=lambda v: f"${_compact_currency(v)}",
    )

    st.markdown(
        f"""
        <div class="stat-row">
          <div class="stat">
            <span class="stat-label">Eligible accounts</span>
            <span class="stat-value">{eligible}</span>
            {eligible_breakdown}
          </div>
          <div class="stat">
            <span class="stat-label">Recommended</span>
            <span class="stat-value"><em>{recommended}</em></span>
            {recommended_breakdown}
          </div>
          <div class="stat">
            <span class="stat-label">Value at risk</span>
            <span class="stat-value">${value_at_risk}</span>
            {var_breakdown}
          </div>
          <div class="stat">
            <span class="stat-label">Active playbooks</span>
            <span class="stat-value">{playbooks}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Render
# ===========================================================================
_ctx, _ctx_diagnostic = _load_run_context()
_render_masthead(_ctx)
_render_reset_bar()

if _ctx_diagnostic:
    st.warning(f"Dashboard run context unavailable — {_ctx_diagnostic}")

# --- Level 1 · Portfolio ----------------------------------------------------
_level_header(
    level=1,
    eyebrow="Level 01 · Portfolio",
    title_html=l1_title_html(_ctx),
)
_render_stat_row()

_section_head(
    eyebrow="Playbooks breakdown",
    title="Where the cohort sits today",
    lead=(
        "Every eligible customer across every active playbook. Tile size is the "
        "eligible cohort; tile colour tracks mean churn probability — pale green "
        "is safe, amber-yellow needs a look, deeper hues mean act first."
    ),
)

try:
    treemap.render()
except Exception as exc:
    st.error(f"Treemap failed: {exc}")

st.markdown(
    '<p class="chart-caption">'
    '<span class="accent-green">▬</span>&nbsp;low risk · '
    '<span class="accent-yellow">▬</span>&nbsp;medium · '
    '<span class="accent-blue">▬</span>&nbsp;high · click any playbook to drill'
    '</p>',
    unsafe_allow_html=True,
)


# --- Level 2 · Playbook drill -----------------------------------------------
_selected_playbook = state.get("selected_playbook")
if _selected_playbook:
    _level_header(
        level=2,
        eyebrow="Level 02 · Playbook",
        title_html=escape(str(_selected_playbook)),
        lead=(
            "Archetypes within this playbook. Each tile is a behavioural cluster "
            "of customers who share risk drivers. Click an archetype to narrow "
            "the list below."
        ),
    )
    try:
        archetype_view.render()
    except Exception as exc:
        st.error(f"Archetype view failed: {exc}")


# --- Level 3 · Customer list ------------------------------------------------
if _selected_playbook:
    _archetype = state.get("selected_archetype")
    if _archetype:
        lead_l3 = (
            f"Customers in {_selected_playbook} · {_archetype}, "
            "sorted by expected loss. Click a row to open the profile below."
        )
    else:
        lead_l3 = (
            f"Customers in {_selected_playbook}, sorted by expected loss. "
            "Click a row to open the profile below, or narrow further by "
            "picking an archetype tile above."
        )
    _level_header(
        level=3,
        eyebrow="Level 03 · Customers",
        title_html="In scope",
        lead=lead_l3,
    )
    try:
        accounts_view.render()
    except Exception as exc:
        st.error(f"Accounts list failed: {exc}")


# --- Level 4 · Customer profile --------------------------------------------
_selected_entity = state.get("selected_entity")
if _selected_entity:
    _level_header(
        level=4,
        eyebrow="Level 04 · Profile",
        title_html=escape(str(_selected_entity)),
        lead="Everything we know about this customer and why the model flagged them.",
    )
    try:
        customer_profile.render()
    except Exception as exc:
        st.error(f"Customer profile failed: {exc}")
