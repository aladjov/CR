"""In-app diagnostic event log — per-session, opt-in via ``CR_SHOW_DIAGNOSTICS``.

Every data-layer query (data.py) and L4 render boundary (customer_profile.py)
pushes a structured event here via ``record(event, **fields)``. The Diagnostics
tab in ``app.py`` reads the per-session deque and renders it as a sortable
table so the operator can see *exactly* where each render's time goes without
leaving the app or scraping Databricks logs.

Hidden by default: the Diagnostics tab is only included in the top-level
``st.tabs(...)`` list when ``CR_SHOW_DIAGNOSTICS=1`` is set in the app
environment. ``record(...)`` itself short-circuits when the flag is off so
there is zero overhead on the hot path in production deployments.

The store lives in ``st.session_state["_cr_diag_events"]`` so it is naturally
per-user and cleared on session end. Bounded to ``_MAX_EVENTS`` (500) to keep
the dataframe lookup cheap even after a long debug session.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Optional

import pandas as pd
import streamlit as st

_MAX_EVENTS = 500
_SESSION_KEY = "_cr_diag_events"


def is_enabled() -> bool:
    """Return ``True`` when the Diagnostics tab should be rendered.

    Gated on the ``CR_SHOW_DIAGNOSTICS`` env var so a production deployment
    pays nothing for the tab while a dev / debug instance can flip it on
    without code changes. Accepts the usual truthy strings; anything else
    (including the var being unset) keeps the tab hidden.
    """
    return os.environ.get("CR_SHOW_DIAGNOSTICS", "0") not in ("", "0", "false", "False")


def _store() -> Optional[deque]:
    """Get or create the per-session event deque.

    Returns ``None`` when called outside a Streamlit script context (e.g.
    module import in a test) so callers can short-circuit without raising.
    """
    try:
        store = st.session_state.get(_SESSION_KEY)
    except Exception:  # noqa: BLE001 -- outside Streamlit script ctx
        return None
    if store is None:
        store = deque(maxlen=_MAX_EVENTS)
        st.session_state[_SESSION_KEY] = store
    return store


def record(event: str, **fields: Any) -> None:
    """Push one event into the per-session diagnostic store.

    No-op when ``CR_SHOW_DIAGNOSTICS`` is unset / falsy, when called outside
    a Streamlit script context, or when the session_state write itself
    fails (best-effort -- the dashboard must never crash because diagnostics
    couldn't be written).

    ``ts`` is added automatically as a Unix timestamp (seconds, float). All
    other fields land verbatim in the row so the Diagnostics tab can sort
    on whichever columns the operator cares about.
    """
    if not is_enabled():
        return
    store = _store()
    if store is None:
        return
    try:
        store.append({"ts": time.time(), "event": event, **fields})
    except Exception:  # noqa: BLE001 -- diagnostics are best-effort
        pass


def clear() -> None:
    """Empty the per-session event store (called from the Clear button)."""
    store = _store()
    if store is not None:
        store.clear()


def _format_time(ts: float) -> str:
    """Human-readable time-of-day with millisecond precision."""
    hms = time.strftime("%H:%M:%S", time.localtime(ts))
    millis = int((ts % 1) * 1000)
    return f"{hms}.{millis:03d}"


def render() -> None:
    """Render the Diagnostics tab body.

    Top-of-tab controls (clear, optional level filter) + a dataframe of the
    captured events. Empty state explains how to populate it so an operator
    landing here for the first time isn't confused by a blank table.
    """
    st.markdown(
        '<p class="catalog-lead">'
        "Per-session timing log. Every data-layer query and L1-L4 render "
        "boundary writes a row here when the dashboard is exercised. Use it "
        "to find slow queries, count cache rebuilds, and verify that the "
        "retry path isn’t firing on benign cancellations."
        "</p>",
        unsafe_allow_html=True,
    )

    store = _store()
    events = list(store) if store is not None else []

    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("Clear", key="diag_clear_btn", use_container_width=True):
            clear()
            st.rerun()
    with cols[1]:
        st.metric("Events captured", len(events))

    if not events:
        st.info(
            "No events captured yet. Click around L1 → L2 → L3 → L4 "
            "and come back here — every query and render boundary will appear."
        )
        return

    df = pd.DataFrame(events)
    # Pretty time column at the front; drop the raw timestamp.
    if "ts" in df.columns:
        df.insert(0, "time", df["ts"].apply(_format_time))
        df = df.drop(columns=["ts"])
    # Stable column order: time, event, then everything else alphabetically
    # so the per-event extras (elapsed_ms, sql, attempt, ...) stay grouped.
    front = [c for c in ("time", "event") if c in df.columns]
    rest = sorted(c for c in df.columns if c not in front)
    df = df[front + rest]

    st.dataframe(df, use_container_width=True, hide_index=True, height=520)
