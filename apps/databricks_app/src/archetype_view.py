"""L2 — Archetype breakdown for the selected playbook, click-to-drill.

Same pastel palette as L1, scoped to one playbook.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from . import data, state
from .treemap import PASTEL_COLORSCALE


def render() -> None:
    pb = state.get("selected_playbook")
    if not pb:
        return
    df = data.archetypes_for_playbook(pb)
    if df.empty:
        st.info(f"No archetypes found for playbook **{pb}**.")
        return

    fig = px.treemap(
        df,
        path=[px.Constant(pb), "archetype_name", "risk_tier"],
        values="account_count",
        color="mean_churn_probability",
        color_continuous_scale=PASTEL_COLORSCALE,
        range_color=(0.0, 1.0),
        custom_data=["archetype_name", "risk_tier", "total_value_at_risk", "mean_churn_probability"],
    )
    fig.update_traces(
        textposition="middle center",
        texttemplate="<b>%{label}</b><br><span style='font-size:11px; opacity:0.75'>%{value:,} accts</span>",
        textfont=dict(family="Geist, system-ui, sans-serif", size=12, color="#1b2230"),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}"
            "<br>Accounts: %{value:,}"
            "<br>Value at risk: $%{customdata[2]:,.0f}"
            "<br>Mean churn prob: %{customdata[3]:.1%}<extra></extra>"
        ),
        marker=dict(
            line=dict(color="#faf9f4", width=3),
            cornerradius=5,
        ),
        root=dict(color="rgba(0,0,0,0)"),
    )
    fig.update_layout(
        margin=dict(t=8, l=0, r=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, system-ui, sans-serif", size=12, color="#1b2230"),
        height=380,
        coloraxis_showscale=False,
    )

    selected = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"archetype_treemap_{pb}",
        on_select="rerun",
        selection_mode="points",
    )

    archetype, risk_tier = _extract_click(selected)
    if archetype and archetype != pb:
        # Update archetype if it changed; ALWAYS update risk_tier (None
        # when the user clicked the archetype tile itself rather than a
        # risk-tier leaf -- so re-clicking the parent archetype unfilters
        # the previously-selected tier).
        rerun = False
        if archetype != state.get("selected_archetype"):
            state.set_archetype(archetype)
            rerun = True
        if risk_tier != state.get("selected_risk_tier"):
            state.set_risk_tier(risk_tier)
            rerun = True
        if rerun:
            st.rerun()


_RISK_TIER_LABELS = ("High", "Medium", "Low")


def _extract_click(event) -> tuple[str | None, str | None]:
    """Return ``(archetype_name, risk_tier_or_None)`` for the clicked tile.

    Tree layout: ``<playbook> → archetype_name → risk_tier``. Clicking the
    risk-tier leaf must resolve BOTH to the parent archetype AND to the
    leaf's risk_tier so the In-Scope table can filter by tier (the user
    just told us which slice they care about). Clicking the archetype
    tile itself returns ``(archetype, None)`` so the table shows every
    tier under that archetype.
    """
    if not event:
        return None, None
    points = (event.get("selection") or {}).get("points") or []
    if not points:
        return None, None
    p = points[0]
    label = (p.get("label") or "").strip()
    parent = (p.get("parent") or "").strip()
    if not label:
        return None, None
    pb = state.get("selected_playbook") or ""
    # Root tile (playbook name used as synthetic root) — ignore
    if label == pb:
        return None, None
    # Top-level archetype tile — parent is the root (playbook name or "")
    if parent in ("", pb) or parent.endswith(f"/{pb}") or parent == pb:
        return label, None
    # Risk-tier leaf — label is the tier, parent encodes the archetype
    if label in _RISK_TIER_LABELS:
        archetype = parent.rsplit("/", 1)[-1] if "/" in parent else parent
        return archetype, label
    # Defensive fallback (unexpected layout): treat parent as archetype, no
    # tier filter. Better to under-filter than to lose the click entirely.
    if "/" in parent:
        return parent.rsplit("/", 1)[-1], None
    return parent, None
