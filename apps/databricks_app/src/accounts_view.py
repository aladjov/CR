"""L3 — Ranked accounts in the current scope, row-click-to-open-profile.

Uses `st.dataframe(on_select="rerun", selection_mode="single-row")` so a single
click picks an entity and sets the L4 selection. Native column sort always
available — click a header to resort.
"""
from __future__ import annotations

import streamlit as st

from . import data, state

_TIER_GLYPH = {"Low": "● ", "Medium": "● ", "High": "● "}  # dot prefix, coloured via column_config? Streamlit doesn't style per-row — use unicode dots.


def _tier_label(tier) -> str:
    if not tier:
        return "—"
    glyph = {"Low": "🟢", "Medium": "🟡", "High": "🔵"}.get(str(tier), "○")
    return f"{glyph}  {tier}"


def render() -> None:
    pb = state.get("selected_playbook")
    ar = state.get("selected_archetype")
    df = data.accounts_in_scope(playbook_name=pb, archetype_name=ar)
    if df.empty:
        st.info("No accounts in this scope.")
        return

    display = df[[
        "entity_id", "churn_probability", "expected_loss", "value_at_risk",
        "risk_tier", "archetype_name",
        "policy_rank_among_eligible", "eligible_playbook_count",
        "recommended", "is_holdout",
    ]].copy()
    display["risk_tier"] = display["risk_tier"].map(_tier_label)

    display = display.rename(columns={
        "entity_id":                  "Entity",
        "churn_probability":          "Churn prob",
        "expected_loss":              "Expected loss",
        "value_at_risk":              "Value at risk",
        "risk_tier":                  "Tier",
        "archetype_name":             "Archetype",
        "policy_rank_among_eligible": "Rank",
        "eligible_playbook_count":    "Plays",
        "recommended":                "Rec.",
        "is_holdout":                 "Holdout",
    })

    col_config = {
        "Entity":        st.column_config.TextColumn(width="small", help="Customer identifier"),
        "Churn prob":    st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0, width="small"),
        "Expected loss": st.column_config.NumberColumn(format="$%,d", width="small"),
        "Value at risk": st.column_config.NumberColumn(format="$%,d", width="small"),
        "Tier":          st.column_config.TextColumn(width="small"),
        "Archetype":     st.column_config.TextColumn(width="medium"),
        "Rank":          st.column_config.NumberColumn(format="%d", width="small"),
        "Plays":         st.column_config.NumberColumn(format="%d", width="small"),
        "Rec.":          st.column_config.CheckboxColumn(width="small"),
        "Holdout":       st.column_config.CheckboxColumn(width="small"),
    }

    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        on_select="rerun",
        selection_mode="single-row",
        height=480,
    )

    rows = (event.selection or {}).get("rows") if event else None
    if rows:
        idx = rows[0]
        entity = str(display.iloc[idx]["Entity"])
        if entity != state.get("selected_entity"):
            state.set_entity(entity)
            st.rerun()

    st.markdown(
        f'<p class="chart-caption">'
        f'{len(df):,}&nbsp;rows · ranked by <em style="font-family:var(--font-serif); font-style:italic; color:var(--ink-2)">expected loss</em> '
        f'= churn&nbsp;prob × value&nbsp;at&nbsp;risk'
        f'</p>',
        unsafe_allow_html=True,
    )
