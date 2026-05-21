"""Tests for the consumption-tracking guard that confines L1/L2 chart
state writes to their own clicks.

Background: Plotly preserves each chart's selection across reruns via its
widget key. Before this guard, both ``treemap.render()`` (L1) and
``archetype_view.render()`` (L2) re-fired their state writes on every
rerun -- whichever ran second wins. The symptom users hit: pin a risk
tier at L2, the next rerun fires L1's handler whose chart still has an
archetype-tile-only selection (risk_tier=None), and L1 calls
``set_risk_tier(None)`` within ~10ms of L2 setting it -- so L2's tier
filter never sticks.

These tests are source-shape assertions: they pin the contract that each
file (a) reads its own ``_l*_consumed_selection`` sentinel from
``st.session_state``, (b) skips state writes when the current selection
equals the last consumed selection, and (c) updates the sentinel after
consuming a new selection.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parents[3] / "apps" / "databricks_app"
_TREEMAP_PY = _APP_DIR / "src" / "treemap.py"
_ARCHETYPE_VIEW_PY = _APP_DIR / "src" / "archetype_view.py"


@pytest.fixture(scope="module")
def treemap_src() -> str:
    return _TREEMAP_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def archetype_src() -> str:
    return _ARCHETYPE_VIEW_PY.read_text(encoding="utf-8")


class TestL1ConsumptionGuard:
    """``treemap.py`` (L1 portfolio treemap)."""

    def test_reads_l1_consumed_sentinel(self, treemap_src):
        assert "_l1_consumed_selection" in treemap_src
        assert 'st.session_state.get("_l1_consumed_selection")' in treemap_src

    def test_returns_early_when_current_matches_last_consumed(self, treemap_src):
        # The smoking-gun guard: when the chart's current selection equals
        # what we already consumed, the handler must not re-write state.
        # Pin both halves of the guard verbatim so accidental refactors
        # surface here.
        assert "if current == last_consumed:" in treemap_src
        # The early return on identity must precede ANY state.set_* call
        # below it -- otherwise the guard is dead.
        guard_idx = treemap_src.index("if current == last_consumed:")
        first_setter_idx = min(
            treemap_src.index("state.set_archetype("),
            treemap_src.index("state.set_risk_tier("),
        )
        assert guard_idx < first_setter_idx, (
            "consumption guard must precede state.set_* calls"
        )

    def test_writes_sentinel_after_consuming_selection(self, treemap_src):
        # Without this, the guard is one-shot: it would skip on the first
        # rerun and then re-fire forever because the sentinel never
        # advances past its initial value.
        assert 'st.session_state["_l1_consumed_selection"] = current' in treemap_src


class TestL2ConsumptionGuard:
    """``archetype_view.py`` (L2 playbook treemap)."""

    def test_reads_l2_consumed_sentinel(self, archetype_src):
        assert "_l2_consumed_selection" in archetype_src
        assert 'st.session_state.get("_l2_consumed_selection")' in archetype_src

    def test_returns_early_when_current_matches_last_consumed(self, archetype_src):
        assert "if current == last_consumed:" in archetype_src
        guard_idx = archetype_src.index("if current == last_consumed:")
        first_setter_idx = min(
            archetype_src.index("state.set_playbook("),
            archetype_src.index("state.set_risk_tier("),
        )
        assert guard_idx < first_setter_idx, (
            "consumption guard must precede state.set_* calls"
        )

    def test_writes_sentinel_after_consuming_selection(self, archetype_src):
        assert 'st.session_state["_l2_consumed_selection"] = current' in archetype_src


class TestSentinelKeysDoNotCollide:
    """L1 and L2 must NOT share a single consumed-selection key -- if they
    did, the cross-chart interference this whole pattern prevents would
    re-emerge in a different shape."""

    def test_l1_and_l2_use_distinct_keys(self, treemap_src, archetype_src):
        assert "_l1_consumed_selection" in treemap_src
        assert "_l1_consumed_selection" not in archetype_src
        assert "_l2_consumed_selection" in archetype_src
        assert "_l2_consumed_selection" not in treemap_src
