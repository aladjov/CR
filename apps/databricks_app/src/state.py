"""Session-state wiring — the single source of truth for the cascade.

Four selectors drive the whole UI:
  * selected_playbook   — set by a click on the L1 treemap
  * selected_archetype  — set by a click on the L2 treemap (archetype tile)
  * selected_risk_tier  — set by a click on an L2 risk-tier leaf
  * selected_entity     — set by a row click on the L3 accounts table

Each downstream section reads these from `st.session_state` and re-queries.
Each selector also exposes a "clear" button so the user can pop back up the
drill-down without starting over.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

_KEYS = (
    "selected_playbook",
    "selected_archetype",
    "selected_risk_tier",
    "selected_entity",
)


def init() -> None:
    for k in _KEYS:
        st.session_state.setdefault(k, None)


def get(key: str) -> Optional[str]:
    return st.session_state.get(key)


def set_playbook(name: Optional[str]) -> None:
    st.session_state.selected_playbook = name
    # Picking a different playbook invalidates downstream selections
    st.session_state.selected_archetype = None
    st.session_state.selected_risk_tier = None
    st.session_state.selected_entity = None


def set_archetype(name: Optional[str]) -> None:
    st.session_state.selected_archetype = name
    # Switching archetype clears the risk-tier slice (the user is asking
    # about the whole archetype again, not a specific tier inside the
    # previously-selected one).
    st.session_state.selected_risk_tier = None
    st.session_state.selected_entity = None


def set_risk_tier(tier: Optional[str]) -> None:
    """Pin the In-Scope table to a single risk_tier (High/Medium/Low)
    when the user clicks the corresponding L2 treemap leaf.

    Clears the entity selection so the L4 panel doesn't show a stale
    profile from a different tier than the one the operator just narrowed
    to. Setting tier=None unfilters back to the whole archetype.
    """
    st.session_state.selected_risk_tier = tier
    st.session_state.selected_entity = None


def set_entity(name: Optional[str]) -> None:
    st.session_state.selected_entity = name


def clear_all() -> None:
    for k in _KEYS:
        st.session_state[k] = None


def breadcrumb_parts() -> list[tuple[str, str]]:
    """Return [(label, level)] for the currently-selected drill path."""
    parts = [("Portfolio", "portfolio")]
    if pb := get("selected_playbook"):
        parts.append((pb, "playbook"))
    if ar := get("selected_archetype"):
        parts.append((ar, "archetype"))
    if rt := get("selected_risk_tier"):
        parts.append((f"{rt} risk", "risk_tier"))
    if en := get("selected_entity"):
        parts.append((en, "entity"))
    return parts
