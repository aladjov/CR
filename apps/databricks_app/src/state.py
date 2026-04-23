"""Session-state wiring — the single source of truth for the cascade.

Three selectors drive the whole UI:
  * selected_playbook  — set by a click on the L1 treemap
  * selected_archetype — set by a click on the L2 treemap
  * selected_entity    — set by a row click on the L3 accounts table

Each downstream section reads these from `st.session_state` and re-queries.
Each selector also exposes a "clear" button so the user can pop back up the
drill-down without starting over.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

_KEYS = ("selected_playbook", "selected_archetype", "selected_entity")


def init() -> None:
    for k in _KEYS:
        st.session_state.setdefault(k, None)


def get(key: str) -> Optional[str]:
    return st.session_state.get(key)


def set_playbook(name: Optional[str]) -> None:
    st.session_state.selected_playbook = name
    # Picking a different playbook invalidates downstream selections
    st.session_state.selected_archetype = None
    st.session_state.selected_entity = None


def set_archetype(name: Optional[str]) -> None:
    st.session_state.selected_archetype = name
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
    if en := get("selected_entity"):
        parts.append((en, "entity"))
    return parts
