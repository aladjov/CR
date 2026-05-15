"""L1 — Archetype × risk-tier treemap, click-to-drill.

Pastel editorial palette: green (safe) → yellow (caution) → soft blue (focus),
no red anywhere. Tile size encodes the eligible-account count for the
archetype; colour tracks mean churn probability for the cohort inside the
tile.

Archetype is per-entity (one ``archetype_id`` per scored account), so
this treemap is entity-grain by construction — tiles never overlap and
summing across the chart equals the unique-account total in the L1
stat row. Clicking a tile sets ``selected_archetype`` (and
``selected_risk_tier`` when the click landed on a risk-tier leaf), and
the L2 treemap below renders the playbook recommendations for that
exact slice.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from . import data, state

# Shared with archetype_view and theme.css :root
PASTEL_COLORSCALE = [
    (0.00, "#b9ddc0"),   # green-200  · low risk
    (0.35, "#dcefdf"),   # green-100  · pale green
    (0.50, "#fbf1c8"),   # yellow-100 · pale yellow
    (0.70, "#f6e299"),   # yellow-200 · medium
    (1.00, "#8bb5d3"),   # blue-300   · high (ink-attention, no red)
]

# L2 fit-score gradient (plum). Same lightness/saturation envelope as
# the L1 palette so the two charts read as siblings, but a clearly
# different hue family so the eye registers L2 as a "different split"
# at a glance. Values mirror --plum-* in theme.css :root.
PASTEL_FIT_COLORSCALE = [
    (0.00, "#f4eff5"),   # plum-50    · pale lavender (low fit)
    (0.30, "#e6d8ec"),   # plum-100   · soft tint
    (0.60, "#cdb3da"),   # plum-200   · clear mauve
    (0.85, "#a07cbb"),   # plum-400   · saturated plum
    (1.00, "#6e4a8c"),   # plum-600   · deep mulberry (best fit)
]


_LAYOUT_COMMON = dict(
    margin=dict(t=44, l=0, r=60, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist, system-ui, sans-serif", size=12, color="#1b2230"),
    height=480,
)


def render() -> None:
    df = data.portfolio_by_archetype_risk_tier()
    if df.empty:
        st.info(
            "No portfolio data yet — run c05 (publish_dashboard_views) against a "
            "scored snapshot, and confirm c02/c03 have published active archetypes."
        )
        return

    fig = px.treemap(
        df,
        path=[px.Constant("All"), "archetype_name", "risk_tier"],
        values="eligible_count",
        color="mean_churn_probability",
        color_continuous_scale=PASTEL_COLORSCALE,
        range_color=(0.0, 1.0),
        custom_data=[
            "archetype_name",
            "risk_tier",
            "total_value_at_risk",
            "mean_churn_probability",
        ],
    )
    fig.update_traces(
        textposition="middle center",
        texttemplate=(
            "<span style='font-size:10px; opacity:0.55'>%{parent}</span>"
            "<br><b>%{label}</b>"
            "<br><span style='font-size:11px; opacity:0.75'>%{value:,} accts</span>"
        ),
        textfont=dict(family="Geist, system-ui, sans-serif", size=13, color="#1b2230"),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}"
            "<br>Eligible: %{value:,}"
            "<br>Value at risk: $%{customdata[2]:,.0f}"
            "<br>Mean churn prob: %{customdata[3]:.1%}<extra></extra>"
        ),
        marker=dict(
            line=dict(color="#faf9f4", width=3),
            cornerradius=6,
        ),
        root=dict(color="rgba(0,0,0,0)"),
        pathbar=dict(
            visible=True,
            side="top",
            thickness=24,
            edgeshape=">",
            textfont=dict(
                family="JetBrains Mono, ui-monospace, monospace",
                size=11,
                color="#404853",
            ),
        ),
    )
    fig.update_layout(
        **_LAYOUT_COMMON,
        coloraxis_colorbar=dict(
            title=dict(text="", font=dict(size=10, color="#98a0ac")),
            thickness=6,
            len=0.55,
            tickfont=dict(size=10, color="#98a0ac", family="JetBrains Mono, ui-monospace, monospace"),
            outlinewidth=0,
            ticks="outside",
            tickformat=".0%",
            x=1.02,
        ),
    )

    selected = st.plotly_chart(
        fig,
        use_container_width=True,
        key="portfolio_treemap",
        on_select="rerun",
        selection_mode="points",
    )

    archetype, risk_tier = _extract_click(selected)
    if archetype and archetype != "All":
        rerun = False
        if archetype != state.get("selected_archetype"):
            state.set_archetype(archetype)
            rerun = True
        # Clicking a risk-tier LEAF inside an archetype must pin the
        # tier so L2 / L3 see the same slice. Clicking the archetype
        # TILE itself (risk_tier=None) should clear any prior leaf
        # selection so the L2 view shows every tier under the archetype.
        if risk_tier != state.get("selected_risk_tier"):
            state.set_risk_tier(risk_tier)
            rerun = True
        if rerun:
            st.rerun()


_RISK_TIER_LABELS = ("High", "Medium", "Low")


def _extract_click(event) -> tuple[str | None, str | None]:
    """Return ``(archetype_name, risk_tier_or_None)`` for the clicked tile.

    Tree layout: ``All → archetype_name → risk_tier``. Clicking the
    risk-tier leaf must resolve BOTH to the parent archetype AND to the
    leaf's risk_tier so downstream views narrow to that slice. Clicking
    the archetype tile returns ``(archetype, None)`` so the L2 treemap
    shows every tier under that archetype.
    """
    if not event:
        return None, None
    points = (event.get("selection") or {}).get("points") or []
    if not points:
        return None, None
    p = points[0]
    label = (p.get("label") or "").strip()
    parent = (p.get("parent") or "").strip()
    if not label or label == "All":
        return None, None
    # Top-level archetype tile (parent is the synthetic root)
    if parent in ("", "All"):
        return label, None
    # Risk-tier leaf — label is the tier, parent encodes the archetype
    if label in _RISK_TIER_LABELS:
        archetype = parent.rsplit("/", 1)[-1] if "/" in parent else parent
        return archetype, label
    # Defensive: unexpected layout — treat parent as the archetype.
    if "/" in parent:
        return parent.rsplit("/", 1)[-1], None
    return parent, None
