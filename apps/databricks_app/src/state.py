"""Session-state wiring — the single source of truth for the cascade.

Four selectors drive the whole UI. The drill order is archetype-first:

  * selected_archetype  — set by a click on the L1 (archetype) treemap
  * selected_playbook   — set by a click on the L2 (playbook-recommendations
                          -for-archetype) treemap
  * selected_risk_tier  — set whenever either treemap's click resolves to
                          a risk_tier leaf (or by clicking the same leaf
                          at the deeper level)
  * selected_entity     — set by a row click on the L3 accounts table

Each downstream section reads these from ``st.session_state`` and
re-queries. Clicking ``Reset drill`` clears everything via ``clear_all``.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

_KEYS = (
    "selected_archetype",
    "selected_playbook",
    "selected_risk_tier",
    "selected_entity",
    "searched_entity",
)


def init() -> None:
    for k in _KEYS:
        st.session_state.setdefault(k, None)


def get(key: str) -> Optional[str]:
    return st.session_state.get(key)


def set_archetype(name: Optional[str]) -> None:
    """Set the L1 archetype selection.

    Picking a different archetype invalidates the L2 playbook pick and
    everything below it. ``selected_risk_tier`` is NOT cleared here
    because the same click that landed us on the new archetype may
    have also pinned a tier (the treemap returns both when the user
    clicked a tier leaf inside the archetype). The caller in
    ``treemap.render`` handles tier (un)pinning explicitly via
    ``set_risk_tier`` so the two decisions stay independent.
    """
    st.session_state.selected_archetype = name
    st.session_state.selected_playbook = None
    st.session_state.selected_entity = None


def set_playbook(name: Optional[str]) -> None:
    """Set the L2 playbook selection within the current archetype.

    Doesn't touch ``selected_archetype`` (we're drilling within it).
    Doesn't touch ``selected_risk_tier`` either -- the L1 click may
    have pre-pinned a tier and we want to honour it down the
    cascade; ``archetype_view.render`` will overwrite the tier when
    the L2 click landed on a risk_tier leaf.
    """
    st.session_state.selected_playbook = name
    st.session_state.selected_entity = None


def set_risk_tier(tier: Optional[str]) -> None:
    """Pin (or clear) the risk tier shared across L1 / L2 / L3.

    The L1 and L2 treemaps both let the user click a tier leaf. We
    store the result in one place so the L3 accounts table can
    subset uniformly. Setting ``tier=None`` clears the pin so the
    drill widens back to every tier under the current archetype +
    playbook selection.
    """
    st.session_state.selected_risk_tier = tier
    st.session_state.selected_entity = None


def set_entity(name: Optional[str]) -> None:
    st.session_state.selected_entity = name


def set_searched_entity(name: Optional[str]) -> None:
    """Pin the entity displayed by the Search tab.

    Kept distinct from ``selected_entity`` so switching tabs doesn't
    fold a search-bar pick into the drill-down breadcrumb (which would
    skip L1/L2/L3 and confuse the trail) and vice-versa.
    """
    st.session_state.searched_entity = name


def clear_all() -> None:
    for k in _KEYS:
        st.session_state[k] = None


def breadcrumb_parts() -> list[tuple[str, str]]:
    """Return [(label, level)] for the currently-selected drill path.

    Order is Portfolio → Archetype → Playbook → Tier → Entity. The
    masthead renders each part with a class based on the second
    element (``crumb-<level>``), so the level strings here have to
    match the ``crumb-*`` classes in ``theme.css``.
    """
    parts = [("Portfolio", "portfolio")]
    if ar := get("selected_archetype"):
        parts.append((ar, "archetype"))
    if pb := get("selected_playbook"):
        parts.append((pb, "playbook"))
    if rt := get("selected_risk_tier"):
        parts.append((f"{rt} risk", "risk_tier"))
    if en := get("selected_entity"):
        parts.append((en, "entity"))
    return parts
