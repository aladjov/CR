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
        # Inside the shared ``_render_l4_panel`` helper, the button must
        # render after the L4 header and before customer_profile.render()
        # so it sits visually next to the profile, scoped to the entity
        # passed in by either tab.
        helper_idx = src.find("def _render_l4_panel(")
        assert helper_idx > 0
        l4_idx = src.find("Level 04 · Profile", helper_idx)
        button_idx = src.find("_render_assign_button(", helper_idx)
        profile_idx = src.find(
            "customer_profile.render(entity_id=entity_id)", helper_idx
        )
        assert l4_idx > helper_idx
        assert button_idx > l4_idx
        assert profile_idx > button_idx

    def test_button_invokes_toggle_with_email(self, src):
        assert "data.toggle_assignment(entity_id, me)" in src

    def test_button_clears_cache_after_toggle(self, src):
        # Without busting the assignments cache the In-scope table keeps
        # showing the pre-toggle state for up to 10s.
        assert "data.assignments.clear()" in src

    def test_button_disabled_for_other_owner(self, src):
        # Disabling on ``other`` is the safety net that prevents one CSM
        # accidentally clearing another's claim from the UI. The expression
        # also folds in ``holdout`` and ``not me`` so all three lockout
        # cases route through the same widget flag.
        assert "disabled = holdout or other or not me" in src
        assert "disabled=disabled" in src

    def test_button_label_branches_on_three_states(self, src):
        for label in ("Assign to me", "Unassign", "Claimed"):
            assert label in src, f"button label {label!r} missing"

    def test_button_warns_on_concurrent_claim(self, src):
        # The race-condition feedback path: another CSM beat us to it.
        assert '"claimed_by_other"' in src

    def test_button_keyed_per_entity_and_tab(self, src):
        # Streamlit reuses widgets by key. The key must scope on
        # ``entity_id`` (so clicks fire against the right account) AND
        # on a tab namespace (so the same entity selected in both
        # Dashboard and Search tabs doesn't collide -- both tabs are
        # rendered on every run, not lazily, so duplicate keys crash
        # the page with StreamlitDuplicateElementKey).
        assert 'key=f"assign_btn::{key_namespace}::{entity_id}"' in src

    def test_l4_panel_threads_distinct_namespaces_per_tab(self, src):
        # Dashboard and Search must pass different namespaces; otherwise
        # the duplicate-key crash returns the moment a user picks the
        # same entity in both tabs.
        assert 'key_namespace="dash"' in src
        assert 'key_namespace="search"' in src


class TestHeaderInlineLayout:
    def test_header_and_button_share_a_columns_row(self, src):
        # The button must render inside the same ``st.columns`` row as the
        # L4 header so it sits visually next to the title rather than
        # taking a full-width band of its own. Helper-scope locals
        # ``hdr_col`` and ``btn_col`` are what the panel uses now.
        helper_idx = src.find("def _render_l4_panel(")
        assert helper_idx > 0
        helper_block = src[helper_idx:src.find("\n\n\n", helper_idx) or len(src)]
        assert "st.columns([6, 1]" in helper_block
        assert "hdr_col" in helper_block and "btn_col" in helper_block

    def test_columns_are_vertically_centered(self, src):
        # Without vertical centering the button floats at the top of its
        # column while the header text occupies the rest -- visually
        # disconnected from the title.
        assert 'vertical_alignment="center"' in src

    def test_level_header_renders_inside_left_column(self, src):
        # ``_level_header`` must be called inside ``with hdr_col:`` so the
        # rendered HTML lands in the column, not full-width above it.
        helper_idx = src.find("def _render_l4_panel(")
        helper_block = src[helper_idx:src.find("\n\n\n", helper_idx) or len(src)]
        with_hdr_idx = helper_block.find("with hdr_col:")
        level_header_idx = helper_block.find(
            "_level_header(\n            level=4"
        )
        with_btn_idx = helper_block.find("with btn_col:")
        assert with_hdr_idx > 0 and level_header_idx > with_hdr_idx
        assert with_btn_idx > level_header_idx


class TestHoldoutDisabled:
    def test_holdout_lookup_runs_before_render(self, src):
        # Reading is_holdout from the cached profile call avoids extra SQL
        # and gives the button the info it needs to disable itself.
        assert "data.account_is_holdout(" in src

    def test_holdout_passed_to_button(self, src):
        # The helper computes is_holdout once and forwards it; both tabs
        # benefit from the holdout policy without duplicating the lookup.
        # The forward is expressed as ``holdout=is_holdout`` regardless of
        # how the call is wrapped across lines.
        assert "holdout=is_holdout" in src
        assert "data.account_is_holdout(entity_id)" in src

    def test_holdout_label_visible(self, src):
        # The label must explain *why* the button is disabled -- a plain
        # greyed-out button without context just looks broken.
        assert "Holdout · no action" in src

    def test_holdout_disables_regardless_of_owner(self, src):
        # ``disabled = holdout or other or not me`` -- holdout dominates.
        assert "disabled = holdout or other or not me" in src

    def test_holdout_blocks_toggle_dml(self, src):
        # Even if a stale UI sneaks a click through (race with cache),
        # the click handler must short-circuit before MERGE on a holdout.
        assert "and not holdout" in src
