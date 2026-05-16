"""L2 — Playbook recommendations for the selected archetype, click-to-drill.

Despite the filename (kept for git-history continuity), this view is
about PLAYBOOKS now: once a CSM picks an archetype at L1, the L2
treemap shows every playbook the eligibility policy mapped to that
archetype, sized by the count of accounts in the slice, coloured by
``fit_score`` -- the model's per-(playbook, archetype) goodness-of-fit
score. The plum gradient is a deliberate sibling to L1's green/yellow/
blue: same lightness/saturation envelope, distinct hue family, so the
eye reads L2 as "this is a different split of the same cohort".

When the L1 click pinned a risk_tier (user clicked a tier-leaf rather
than the archetype tile), we pass that through so this view shows only
that slice -- matching the "whatever tiers are visible at the upper
level remain on all lower levels" intent.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from . import data, state
from .treemap import PASTEL_FIT_COLORSCALE


def render() -> None:
    archetype = state.get("selected_archetype")
    if not archetype:
        return
    risk_tier = state.get("selected_risk_tier")
    df = data.playbooks_for_archetype(archetype, risk_tier=risk_tier)
    if df.empty:
        # The view honours the (archetype, risk_tier) subset, so empty
        # means "no playbooks matched this archetype at this tier" --
        # explicit enough that the user understands it's a real zero,
        # not a missing-data hiccup.
        if risk_tier:
            st.info(
                f"No playbooks matched **{archetype}** at **{risk_tier}** risk in the latest run."
            )
        else:
            st.info(f"No playbooks matched **{archetype}** in the latest run.")
        return

    # Pre-fill missing fit_score / uplift values so the hover template
    # never renders "None" -- the LEFT JOIN in the SQL leaves them NULL
    # when an archetype-version isn't represented in eligibility_policy.
    df = df.copy()
    df["fit_score"] = df["fit_score"].fillna(0.0)
    df["expected_uplift_pct"] = df["expected_uplift_pct"].fillna(0.0)

    # The root tile carries the selected archetype's name so the tree
    # reads as "archetype → playbook → tier" rather than "→ playbook →
    # tier" floating in space. When the L1 click also pinned a tier we
    # collapse the third level (everything is that tier already).
    path = [px.Constant(archetype), "playbook_name"]
    if not risk_tier:
        path.append("risk_tier")

    fig = px.treemap(
        df,
        path=path,
        values="eligible_count",
        color="fit_score",
        color_continuous_scale=PASTEL_FIT_COLORSCALE,
        range_color=(0.0, 1.0),
        custom_data=[
            "playbook_name",
            "risk_tier",
            "total_value_at_risk",
            "mean_churn_probability",
            "fit_score",
            "expected_uplift_pct",
        ],
    )

    # Tile label: name + accts + fit-score so the headline number is
    # legible on the chart itself. The leaf rows (risk_tier tiles) get
    # a slimmer label because they're narrow; the parent playbook tiles
    # get the full readout.
    fig.update_traces(
        textposition="middle center",
        texttemplate=(
            "<b>%{label}</b>"
            "<br><span style='font-size:11px; opacity:0.78'>%{value:,} accts</span>"
            "<br><span style='font-size:10.5px; opacity:0.7'>fit %{customdata[4]:.2f} · +%{customdata[5]:.0%} uplift</span>"
        ),
        textfont=dict(family="Geist, system-ui, sans-serif", size=12, color="#1b2230"),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}"
            "<br>Eligible accounts: %{value:,}"
            "<br>Fit score: %{customdata[4]:.2f}"
            "<br>Expected uplift: %{customdata[5]:.1%}"
            "<br>Value at risk: $%{customdata[2]:,.0f}"
            "<br>Mean churn prob: %{customdata[3]:.1%}<extra></extra>"
        ),
        marker=dict(
            line=dict(color="#faf9f4", width=3),
            cornerradius=5,
        ),
        root=dict(color="rgba(0,0,0,0)"),
    )

    # Per-tile risk-tier border. Plotly express flattens the tree into
    # one trace, so `fig.data[0].labels` carries roots + parents + leaves
    # in one array. We walk that array and stamp a tier-coloured border
    # only on the tier-leaf nodes (label is one of High/Medium/Low AND
    # parent is a playbook, not the synthetic archetype root). Parent
    # playbook tiles and the archetype root keep the existing paper
    # border. Width bumps to 4px on tier leaves so the colour is visible
    # even on thin tiles -- the L1 risk palette (blue-300 / yellow-400 /
    # green-400) is reused verbatim so the same colour means the same
    # thing in both charts.
    _apply_risk_tier_borders(fig, archetype=archetype, has_tier_level=not risk_tier)
    fig.update_layout(
        margin=dict(t=8, l=0, r=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, system-ui, sans-serif", size=12, color="#1b2230"),
        height=380,
        # Surface a slim colorbar on the right so the plum gradient is
        # decoded: "darker = stronger fit". Same minimal styling as the
        # L1 colorbar so the two charts feel like one report.
        coloraxis_colorbar=dict(
            title=dict(text="fit", font=dict(size=10, color="#98a0ac")),
            thickness=6,
            len=0.55,
            tickfont=dict(
                size=10, color="#98a0ac",
                family="JetBrains Mono, ui-monospace, monospace",
            ),
            outlinewidth=0,
            ticks="outside",
            tickformat=".1f",
            x=1.02,
        ),
    )

    selected = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"l2_playbook_treemap::{archetype}::{risk_tier or 'all'}",
        on_select="rerun",
        selection_mode="points",
    )

    playbook, tier_from_click = _extract_click(selected, archetype=archetype)
    if not playbook or playbook == archetype:
        return
    rerun = False
    if playbook != state.get("selected_playbook"):
        state.set_playbook(playbook)
        rerun = True
    # The L1 click may already have pinned a risk_tier. If the user
    # clicked a tier leaf here we override; otherwise we keep what L1
    # set. Clicking the playbook tile itself (tier_from_click=None)
    # while L1 already pinned a tier should leave the tier alone --
    # they're narrowing within the same tier.
    if tier_from_click and tier_from_click != state.get("selected_risk_tier"):
        state.set_risk_tier(tier_from_click)
        rerun = True
    if rerun:
        st.rerun()


_RISK_TIER_LABELS = ("High", "Medium", "Low")

# L1 risk-tier palette, re-used verbatim on the L2 tile borders so the
# same hue means the same thing in both charts. Sourced from the
# ``--blue-300 / --yellow-400 / --green-400`` tokens in ``theme.css``.
_TIER_BORDER_COLORS = {
    "High":   "#8bb5d3",   # blue-300
    "Medium": "#e8c259",   # yellow-400
    "Low":    "#6bac7a",   # green-400
}
_DEFAULT_BORDER_COLOR = "#faf9f4"  # --paper (the existing tile separator)


def _apply_risk_tier_borders(fig, *, archetype: str, has_tier_level: bool) -> None:
    """Stamp per-tile border colour by risk_tier on the leaf nodes.

    Plotly express flattens the (archetype → playbook → tier) tree into
    one trace where ``fig.data[0].labels`` / ``.parents`` carry every
    node including roots and intermediate parents. We compute a colour
    array the same length and width array of equal length, then assign
    both onto ``marker.line``. Tier leaves get a 4-px tier-coloured
    stroke; everything else keeps the existing paper-coloured 3-px
    separator so the tree's visual rhythm doesn't change.

    ``has_tier_level=False`` is the "L1 click already pinned a tier"
    case: the tree collapses to two levels and there are no tier
    leaves to stamp -- the function short-circuits.
    """
    if not has_tier_level:
        return
    if not fig.data:
        return
    trace = fig.data[0]
    # ``trace.labels`` / ``.parents`` come back as numpy arrays from
    # plotly. We can't use the ``getattr(...) or default`` pattern here
    # because ``bool(ndarray)`` with len > 1 raises "truth value of an
    # array with more than one element is ambiguous". Resolve to None
    # explicitly, then convert via ``list(...)`` only when present.
    labels_attr = getattr(trace, "labels", None)
    parents_attr = getattr(trace, "parents", None)
    if labels_attr is None or parents_attr is None:
        return
    labels = list(labels_attr)
    parents = list(parents_attr)
    if len(labels) == 0 or len(parents) == 0:
        return

    line_colors: list[str] = []
    line_widths: list[float] = []
    for label, parent in zip(labels, parents):
        label_s = (label or "").strip()
        parent_s = (parent or "").strip()
        # A tier leaf has the tier name as its label AND a non-root,
        # non-archetype parent (= the playbook that owns it). The
        # synthetic archetype root sits at the top of the tree, so any
        # parent that isn't blank and isn't the archetype must be the
        # playbook owner.
        is_tier_leaf = (
            label_s in _RISK_TIER_LABELS
            and parent_s
            and parent_s != archetype
        )
        if is_tier_leaf:
            line_colors.append(_TIER_BORDER_COLORS.get(label_s, _DEFAULT_BORDER_COLOR))
            line_widths.append(4.0)
        else:
            line_colors.append(_DEFAULT_BORDER_COLOR)
            line_widths.append(3.0)
    trace.marker.line.color = line_colors
    trace.marker.line.width = line_widths


def _extract_click(event, *, archetype: str) -> tuple[str | None, str | None]:
    """Return ``(playbook_name, risk_tier_or_None)`` for the clicked tile.

    Tree layout: ``<archetype> → playbook_name → risk_tier``. Clicking
    the risk-tier leaf must resolve BOTH to the parent playbook AND to
    the tier so the L3 table narrows to that slice. Clicking the
    playbook tile returns ``(playbook, None)``.
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
    # Root tile (archetype as synthetic root) — ignore.
    if label == archetype:
        return None, None
    # Top-level playbook tile — parent is the archetype root.
    if parent in ("", archetype) or parent.endswith(f"/{archetype}"):
        return label, None
    # Risk-tier leaf — label is the tier, parent encodes the playbook.
    if label in _RISK_TIER_LABELS:
        playbook = parent.rsplit("/", 1)[-1] if "/" in parent else parent
        return playbook, label
    if "/" in parent:
        return parent.rsplit("/", 1)[-1], None
    return parent, None
