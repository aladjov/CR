"""Source-shape tests for the L4 self-assign button wiring in app.py."""
from __future__ import annotations

from pathlib import Path

import pytest

_APP_PY = Path(__file__).resolve().parents[3] / "apps" / "databricks_app" / "app.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _APP_PY.read_text(encoding="utf-8")


class TestAssignButtonWiring:
    def test_imports_auth_module(self, src):
        assert "auth" in src
        # in the explicit ``from src import …`` line, not just incidentally
        assert "auth," in src or ", auth" in src

    def test_button_renderer_exists(self, src):
        assert "_render_assign_button" in src

    def test_button_renders_inside_l4_block(self, src):
        # The button must appear after the L4 header and before the
        # customer_profile.render() call so it sits visually next to the
        # profile, scoped to the selected entity.
        l4_idx = src.find("Level 04 · Profile")
        button_call_idx = src.find("_render_assign_button(str(_selected_entity))")
        profile_render_idx = src.find("customer_profile.render()")
        assert l4_idx > 0
        assert button_call_idx > l4_idx
        assert profile_render_idx > button_call_idx

    def test_button_invokes_toggle_with_email(self, src):
        assert "data.toggle_assignment(entity_id, me)" in src

    def test_button_clears_cache_after_toggle(self, src):
        # Without busting the assignments cache the In-scope table keeps
        # showing the pre-toggle state for up to 10s.
        assert "data.assignments.clear()" in src

    def test_button_disabled_for_other_owner(self, src):
        # Disabling on ``other`` is the safety net that prevents one CSM
        # accidentally clearing another's claim from the UI.
        assert "disabled=other or not me" in src

    def test_button_label_branches_on_three_states(self, src):
        for label in ("Assign to me", "Unassign", "Claimed"):
            assert label in src, f"button label {label!r} missing"

    def test_button_warns_on_concurrent_claim(self, src):
        # The race-condition feedback path: another CSM beat us to it.
        assert '"claimed_by_other"' in src

    def test_button_keyed_per_entity(self, src):
        # Streamlit reuses widgets by key. Without entity-scoped keys, a
        # button retains state from the previous selection and clicks fire
        # against the wrong entity.
        assert 'key=f"assign_btn::{entity_id}"' in src
