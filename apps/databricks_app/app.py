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


def _load_run_context() -> dict:
    """Return a dict with the masthead fields; empty on any failure."""
    try:
        df = data.run_context()
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    row = df.iloc[0]

    def _get(col):
        if col not in df.columns:
            return None
        v = row[col]
        return None if pd.isna(v) else v

    return {
        "horizon_days":       _get("horizon_days"),
        "primary_objective":  _get("primary_objective"),
        "temporal_posture":   _get("temporal_posture"),
        "model_type":         _get("model_type"),
        "model_name":         _get("model_name"),
    }


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
def _level_header(level: int, eyebrow: str, title_html: str, lead: str) -> None:
    st.markdown(
        f"""
        <section class="level level-{level}">
          <div class="level-eyebrow"><span class="dot"></span>{escape(eyebrow)}</div>
          <h2 class="level-title">{title_html}</h2>
          <p class="level-lead">{escape(lead)}</p>
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
    st.markdown(
        f"""
        <div class="stat-row">
          <div class="stat">
            <span class="stat-label">Eligible accounts</span>
            <span class="stat-value">{eligible}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Recommended</span>
            <span class="stat-value"><em>{recommended}</em></span>
          </div>
          <div class="stat">
            <span class="stat-label">Value at risk</span>
            <span class="stat-value">${value_at_risk}</span>
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
_ctx = _load_run_context()
_render_masthead(_ctx)
_render_reset_bar()

# --- Level 1 · Portfolio ----------------------------------------------------
_level_header(
    level=1,
    eyebrow="Level 01 · Portfolio",
    title_html=l1_title_html(_ctx),
    lead=(
        "Every eligible customer across every active playbook. Tile size is the "
        "eligible cohort; tile colour tracks mean churn probability — pale green "
        "is safe, amber-yellow needs a look, deeper hues mean act first."
    ),
)
_render_stat_row()

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
