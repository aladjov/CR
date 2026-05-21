"""L2 — Playbook recommendations for the selected archetype, click-to-drill.

Despite the filename (kept for git-history continuity), this view is
about PLAYBOOKS now: once a CSM picks an archetype at L1, the L2
treemap shows every playbook the eligibility policy mapped to that
archetype, sized by the count of accounts in the slice, coloured by a
DUAL encoding: tile depth tracks ``fit_score`` (deeper plum = stronger
match) and tile saturation tracks ``risk_tier`` (vivid plum = High
risk, gray-ish plum = Low). Combined, the most vivid AND deepest tile
is the top-priority recommendation — one visual cell = one decision.

When the L1 click pinned a risk_tier (user clicked a tier-leaf rather
than the archetype tile), we pass that through so this view shows only
that slice -- matching the "whatever tiers are visible at the upper
level remain on all lower levels" intent. In that mode the saturation
of every tile is fixed by the pinned tier and only the depth varies.
"""
from __future__ import annotations

import colorsys

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

    # Per-tile fill that combines fit_score (depth) and risk_tier
    # (saturation) into one visual cue. See ``_apply_priority_fill``
    # for the math. The single white separator (3 px paper colour)
    # stays so adjacent tiles remain visually distinct.
    _apply_priority_fill(fig, df, archetype=archetype, pinned_tier=risk_tier)
    fig.update_layout(
        margin=dict(t=8, l=0, r=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, system-ui, sans-serif", size=12, color="#1b2230"),
        height=380,
    )
    # Continuous colorbar is misleading now -- fit and tier are encoded
    # jointly per-tile, not on a single axis. Hide it; the visual legend
    # below the chart documents the dual encoding instead.
    fig.update_coloraxes(showscale=False)

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

    # Same consumption-tracking guard as treemap.render() (L1). Plotly
    # keeps each chart's selection across reruns via its widget key, so
    # without this both L1 and L2 re-fire their state writes on every
    # rerun and whichever runs second clobbers the other. Per-chart
    # "last consumed" tracking confines each handler to its OWN clicks.
    current = (playbook, tier_from_click)
    last_consumed = st.session_state.get("_l2_consumed_selection")
    if current == last_consumed:
        return

    rerun = False
    if playbook != state.get("selected_playbook"):
        state.set_playbook(playbook)
        rerun = True
    # Playbook-frame click (``tier_from_click is None``) widens the L3
    # cohort to every tier under the chosen playbook -- so an inherited
    # L1 tier pin is cleared. Tier-leaf click pins that specific tier.
    # This matches the rule "click the playbook to see all tiers; click
    # a tier within the playbook to narrow to that tier" and gives the
    # operator a reliable way to widen back out without losing the
    # archetype context.
    target_tier = tier_from_click  # may be None
    if target_tier != state.get("selected_risk_tier"):
        state.set_risk_tier(target_tier)
        rerun = True
    st.session_state["_l2_consumed_selection"] = current
    if rerun:
        st.rerun()


_RISK_TIER_LABELS = ("High", "Medium", "Low")

# Per-tier saturation multipliers applied to the plum fill. 1.0 leaves
# the original plum gradient untouched (High); 0.18 collapses the
# colour almost to gray (Low). Combined with the existing depth
# gradient on fit_score this produces a 2D priority cue: vividness ×
# depth = where to act first. Tuned to keep the visual envelope BELOW
# the original max-saturation plum (per user direction) so saturation
# only ever subtracts.
_TIER_SATURATION = {
    "High":   1.0,
    "Medium": 0.5,
    "Low":    0.18,
}

# Same stops as PASTEL_FIT_COLORSCALE in treemap.py. Kept locally so
# we can interpolate continuous RGB without paying a plotly round-trip
# for every tile.
_PLUM_STOPS: tuple[tuple[float, str], ...] = (
    (0.00, "#f4eff5"),
    (0.30, "#e6d8ec"),
    (0.60, "#cdb3da"),
    (0.85, "#a07cbb"),
    (1.00, "#6e4a8c"),
)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _interpolate_plum_rgb(fit_score: float) -> tuple[int, int, int]:
    """Linear interpolation between adjacent ``_PLUM_STOPS`` pairs."""
    try:
        f = float(fit_score)
    except (TypeError, ValueError):
        f = 0.0
    f = max(0.0, min(1.0, f))
    for i in range(len(_PLUM_STOPS) - 1):
        lo_pos, lo_hex = _PLUM_STOPS[i]
        hi_pos, hi_hex = _PLUM_STOPS[i + 1]
        if f <= hi_pos:
            span = max(hi_pos - lo_pos, 1e-9)
            t = (f - lo_pos) / span
            lo = _hex_to_rgb(lo_hex)
            hi = _hex_to_rgb(hi_hex)
            return tuple(int(round(lo[k] + t * (hi[k] - lo[k]))) for k in range(3))
    return _hex_to_rgb(_PLUM_STOPS[-1][1])


def _desaturate_rgb(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale saturation in HLS space. ``factor`` 1.0 = unchanged; 0.0 = gray."""
    r, g, b = (c / 255 for c in rgb)
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    sat *= max(0.0, min(1.0, factor))
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _priority_color(fit_score: float, tier_sat: float) -> str:
    return _rgb_to_hex(_desaturate_rgb(_interpolate_plum_rgb(fit_score), tier_sat))


def _apply_priority_fill(
    fig, df, *, archetype: str, pinned_tier: str | None
) -> None:
    """Stamp the per-tile fill colour that combines fit and risk tier.

    Tier leaves: colour = ``plum(fit_score)`` desaturated by the leaf's
    own tier factor. Playbook parents (only present when the tree has a
    tier level): colour = ``plum(weighted_mean_fit)`` at full saturation
    so the parent reads as a neutral aggregate while the tier-saturated
    children sit beneath it. When the L1 click already pinned a tier
    the tree collapses to (archetype → playbook); every playbook tile
    IS the tier-fixed slice and we apply the pinned tier's saturation
    factor directly to it.

    Defensive against numpy-truthiness: ``trace.labels`` / ``.parents``
    come back as numpy arrays, so we resolve to None explicitly before
    converting.
    """
    if not fig.data:
        return
    trace = fig.data[0]
    labels_attr = getattr(trace, "labels", None)
    parents_attr = getattr(trace, "parents", None)
    if labels_attr is None or parents_attr is None:
        return
    labels = list(labels_attr)
    parents = list(parents_attr)
    if len(labels) == 0 or len(parents) == 0:
        return

    # (playbook, tier) -> fit_score lookups from the source frame.
    leaf_fit: dict[tuple[str, str], float] = {}
    pb_fit_num: dict[str, float] = {}
    pb_fit_den: dict[str, float] = {}
    for _, row in df.iterrows():
        pb = str(row.get("playbook_name") or "")
        rt = str(row.get("risk_tier") or "")
        try:
            fit = float(row.get("fit_score") or 0.0)
        except (TypeError, ValueError):
            fit = 0.0
        try:
            n = float(row.get("eligible_count") or 0.0)
        except (TypeError, ValueError):
            n = 0.0
        leaf_fit[(pb, rt)] = fit
        pb_fit_num[pb] = pb_fit_num.get(pb, 0.0) + fit * n
        pb_fit_den[pb] = pb_fit_den.get(pb, 0.0) + n
    pb_avg_fit = {
        pb: (pb_fit_num[pb] / pb_fit_den[pb] if pb_fit_den[pb] else 0.0)
        for pb in pb_fit_num
    }

    transparent = "rgba(0,0,0,0)"
    colors: list[str] = []
    for label, parent in zip(labels, parents):
        label_s = str(label).strip()
        parent_s = str(parent).strip()

        if pinned_tier:
            # Tree is [archetype -> playbook]; every playbook tile is
            # the tier-fixed slice.
            if label_s in pb_avg_fit:
                colors.append(_priority_color(
                    pb_avg_fit[label_s],
                    _TIER_SATURATION.get(pinned_tier, 1.0),
                ))
            else:
                colors.append(transparent)
            continue

        # Tree is [archetype -> playbook -> tier].
        is_tier_leaf = (
            label_s in _RISK_TIER_LABELS
            and parent_s
            and parent_s != archetype
        )
        if is_tier_leaf:
            pb = parent_s.rsplit("/", 1)[-1] if "/" in parent_s else parent_s
            colors.append(_priority_color(
                leaf_fit.get((pb, label_s), 0.0),
                _TIER_SATURATION.get(label_s, 1.0),
            ))
        elif label_s in pb_avg_fit:
            # Parent playbook tile -- aggregate fit at full saturation.
            colors.append(_priority_color(pb_avg_fit[label_s], 1.0))
        else:
            colors.append(transparent)

    trace.marker.colors = colors


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
