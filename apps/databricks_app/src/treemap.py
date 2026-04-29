"""L1 — Playbook × risk-tier treemap, click-to-drill.

Pastel editorial palette: green (safe) → yellow (caution) → soft blue (focus),
no red anywhere. Tile size encodes eligible-account count; colour tracks mean
churn probability.
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


_LAYOUT_COMMON = dict(
    margin=dict(t=44, l=0, r=60, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist, system-ui, sans-serif", size=12, color="#1b2230"),
    height=480,
)


def render() -> None:
    df = data.portfolio_by_playbook_risk_tier()
    if df.empty:
        st.info("No portfolio data yet — run c05 (publish_dashboard_views) against a scored snapshot.")
        return

    fig = px.treemap(
        df,
        path=[px.Constant("All"), "playbook_name", "risk_tier"],
        values="eligible_count",
        color="mean_churn_probability",
        color_continuous_scale=PASTEL_COLORSCALE,
        range_color=(0.0, 1.0),
        custom_data=["playbook_name", "risk_tier", "total_value_at_risk", "mean_churn_probability"],
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

    clicked = _extract_click(selected)
    if clicked and clicked != "All":
        if clicked != state.get("selected_playbook"):
            state.set_playbook(clicked)
            st.rerun()


def _extract_click(event) -> str | None:
    """Return the clicked playbook name.

    Tree layout: All → playbook_name → risk_tier. Use `parent` so the drill-down
    resolves to the playbook regardless of which level the user clicked — clicking
    a risk-tier leaf must still select its parent playbook.
    """
    if not event:
        return None
    points = (event.get("selection") or {}).get("points") or []
    if not points:
        return None
    p = points[0]
    label = (p.get("label") or "").strip()
    parent = (p.get("parent") or "").strip()
    if not label or label == "All":
        return None
    # Top-level playbook tile — parent is the synthetic root
    if parent in ("", "All"):
        return label
    # Drilled into risk-tier leaf — parent IS the playbook
    if parent == "All" or parent == "":
        return label
    # Leaf with parent = "All/<playbook>" (plotly sometimes prefixes with root)
    if parent.startswith("All/"):
        return parent.split("/", 1)[1]
    return parent
