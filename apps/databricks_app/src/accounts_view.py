"""L3 — Ranked accounts in the current scope, row-click-to-open-profile.

Uses `st.dataframe(on_select="rerun", selection_mode="single-row")` so a single
click picks an entity and sets the L4 selection. Native column sort always
available — click a header to resort.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import auth, data, state

_TIER_GLYPH = {"Low": "● ", "Medium": "● ", "High": "● "}  # dot prefix, coloured via column_config? Streamlit doesn't style per-row — use unicode dots.


def _tier_label(tier) -> str:
    if not tier:
        return "—"
    glyph = {"Low": "🟢", "Medium": "🟡", "High": "🔵"}.get(str(tier), "○")
    return f"{glyph}  {tier}"


_ME_LABEL = "Me"
# Very low-saturation green wash so a CSM can spot their own rows at a
# glance without the row screaming louder than the High-tier risk dots.
# Tuned to read against both the table's default white background and the
# selected-row pink wash without becoming muddy.
_MINE_ROW_STYLE = "background-color: rgba(34, 139, 34, 0.08); font-weight: 600"


def _assignment_label(assigned_to, me: str | None) -> str:
    """Render the ``assigned_to`` cell.

    ``Me`` is reserved for the current user so a CSM can scan their own
    column at a glance; everyone else surfaces the local-part of their
    email (``jane.doe`` from ``jane.doe@churnkit.com``) for compactness.
    Returns an empty string for unassigned rows so the column visually
    recedes when most rows are blank.
    """
    if assigned_to is None or (isinstance(assigned_to, float) and pd.isna(assigned_to)):
        return ""
    text = str(assigned_to).strip()
    if not text:
        return ""
    if me and text.lower() == me.lower():
        return _ME_LABEL
    return text.split("@", 1)[0] if "@" in text else text


def _row_style(row) -> list[str]:
    """Per-row CSS for ``Styler.apply(axis=1)``.

    Highlights rows owned by the current user so the at-a-glance scan
    answers "which of these is mine" before any reading. Other rows return
    empty strings so the default table styling applies.
    """
    if row.get("Assigned to") == _ME_LABEL:
        return [_MINE_ROW_STYLE] * len(row)
    return [""] * len(row)


def render() -> None:
    pb = state.get("selected_playbook")
    ar = state.get("selected_archetype")
    rt = state.get("selected_risk_tier")
    df = data.accounts_in_scope(playbook_name=pb, archetype_name=ar, risk_tier=rt)
    if df.empty:
        st.info("No accounts in this scope.")
        return

    # Left-merge the cached assignments frame so the In-scope table can
    # surface "Me" / owner handle / blank without modifying the upstream
    # ``v_eligible_all_playbooks`` view. Empty assignments DataFrame leaves
    # every row unassigned.
    me = auth.current_user_email()
    try:
        assigns = data.assignments()
    except Exception:
        assigns = pd.DataFrame(columns=["entity_id", "assigned_to"])
    if assigns is None or assigns.empty:
        df = df.assign(assigned_to=None)
    else:
        df = df.merge(
            assigns[["entity_id", "assigned_to"]],
            on="entity_id",
            how="left",
        )

    display = df[[
        "entity_id", "churn_probability", "expected_loss", "value_at_risk",
        "risk_tier", "archetype_name",
        "policy_rank_among_eligible", "eligible_playbook_count",
        "recommended", "is_holdout", "assigned_to",
    ]].copy()
    display["risk_tier"] = display["risk_tier"].map(_tier_label)
    display["assigned_to"] = display["assigned_to"].map(
        lambda v: _assignment_label(v, me)
    )

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
        "assigned_to":                "Assigned to",
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
        "Assigned to":   st.column_config.TextColumn(
            width="small",
            help="CSM who has self-assigned this account. 'Me' = you.",
        ),
    }

    # Styler-applied row tint for entries owned by the current user.
    # ``st.dataframe`` accepts a Styler and still respects column_config and
    # selection events (the events index back into the underlying frame).
    #
    # The explicit ``key`` is load-bearing: without it Streamlit hashes the
    # widget identity from the (Styler, kwargs) tuple. ``display.style.apply``
    # returns a fresh Styler on every rerun, so the auto-key changes each
    # time, the widget is re-created from scratch, and the user's row
    # selection is lost on the rerun the click itself triggers. That makes
    # the L3 click look like a no-op: ``event.selection.rows`` comes back
    # empty after the rerun, ``set_entity`` never fires, and the L4 panel's
    # ``if _selected_entity:`` guard stays False so the profile never
    # renders. A stable ``key`` pins the widget identity so the selection
    # survives the rerun.
    styled = display.style.apply(_row_style, axis=1)
    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        on_select="rerun",
        selection_mode="single-row",
        height=480,
        key="dashboard_accounts_table",
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
