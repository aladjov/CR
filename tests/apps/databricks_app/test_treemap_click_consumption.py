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

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parents[3] / "apps" / "databricks_app"
_TREEMAP_PY = _APP_DIR / "src" / "treemap.py"
_ARCHETYPE_VIEW_PY = _APP_DIR / "src" / "archetype_view.py"
_STATE_PY = _APP_DIR / "src" / "state.py"


class _SessionStateDict(dict):
    """Streamlit's session_state behaves like both a dict and a namespace.
    The state module uses attribute-style writes (``st.session_state.x = v``)
    and ``.get(...)`` reads, so we shim both surfaces from a plain dict."""

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


def _load_state_module() -> tuple[object, _SessionStateDict]:
    """Load the project's ``state.py`` against a fake streamlit module so
    its calls into ``st.session_state`` mutate a dict we can inspect."""
    fake = types.ModuleType("streamlit")
    fake.session_state = _SessionStateDict()
    fake.cache_data = lambda *a, **k: (lambda f: f)
    sys.modules["streamlit"] = fake
    spec = importlib.util.spec_from_file_location("_state_under_test", _STATE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, fake.session_state


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


class TestExplorationCycle:
    """End-to-end trace of the natural exploration cycle the user
    described: click L1 → drill L2 → pick a row → see L4 → switch L2 →
    L4 follows → switch L1 → cohort resets but tier follows → Reset
    drill button clears everything (including the click-consumption
    sentinels). Exercises the ``state`` setters in the order the
    handlers call them and pins each step's resulting session state.
    """

    def test_full_exploration_cycle_leaves_session_state_consistent(self):
        state, session = _load_state_module()
        state.init()

        # Step 1: L1 archetype click. New archetype, no tier specified.
        # set_archetype clears any prior playbook + entity but leaves
        # selected_risk_tier alone (callers explicitly call set_risk_tier
        # when needed).
        state.set_archetype("NOA")
        assert session["selected_archetype"] == "NOA"
        assert session["selected_playbook"] is None
        assert session["selected_entity"] is None

        # Step 2: L2 playbook frame click. Playbook is set, the entity
        # the operator might already have open at L4 stays put (fixed
        # in this iteration -- earlier behaviour wiped it here, which is
        # what made L4 "stop loading" the moment the operator clicked
        # any L2 tile).
        state.set_playbook("R")
        assert session["selected_playbook"] == "R"
        assert session["selected_entity"] is None  # not yet picked

        # Step 3: L3 row click → entity is set, L4 renders.
        state.set_entity("E5")
        assert session["selected_entity"] == "E5"

        # Step 4: Switch L2 playbook (same archetype). Playbook flips
        # but the open L4 profile is preserved.
        state.set_playbook("O")
        assert session["selected_playbook"] == "O"
        assert session["selected_entity"] == "E5", (
            "switching L2 playbook must not close the open L4 profile"
        )

        # Step 5: Pin a tier at L2 (tier-leaf click inside the playbook).
        # Tier is set; L4 profile still preserved.
        state.set_risk_tier("Low")
        assert session["selected_risk_tier"] == "Low"
        assert session["selected_entity"] == "E5"

        # Step 6: Switch L1 archetype. set_archetype clears playbook
        # and entity (top-level context shift) but leaves tier alone --
        # the chart's handler decides whether to also reset tier based
        # on what the user actually clicked at L1.
        state.set_archetype("NOA2")
        assert session["selected_archetype"] == "NOA2"
        assert session["selected_playbook"] is None
        assert session["selected_entity"] is None
        assert session["selected_risk_tier"] == "Low"  # left alone

        # Step 7: Reset drill -- the "go back to everything" action.
        # Must clear both the public selectors AND the private
        # consumption sentinels, otherwise the next click on a tile the
        # chart had previously selected would short-circuit as
        # "already consumed" and silently do nothing.
        session["_l1_consumed_selection"] = ("NOA2", None)
        session["_l2_consumed_selection"] = ("O", "Low")
        state.clear_all()
        for k in (
            "selected_archetype", "selected_playbook",
            "selected_risk_tier", "selected_entity", "searched_entity",
        ):
            assert session[k] is None, f"clear_all left {k} populated"
        assert session["_l1_consumed_selection"] is None, (
            "clear_all must wipe the L1 chart's last-consumed sentinel"
        )
        assert session["_l2_consumed_selection"] is None, (
            "clear_all must wipe the L2 chart's last-consumed sentinel"
        )
