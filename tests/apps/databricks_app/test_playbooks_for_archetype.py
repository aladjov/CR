"""Regression tests for the L2 treemap query (``playbooks_for_archetype``).

The L2 treemap breaks one archetype down by primary playbook x risk_tier.
``eligibility_snapshot`` has one row per (entity, playbook) eligibility
match, so when every entity in an archetype is eligible for every play,
counting rows makes every playbook tile show the FULL archetype's
cohort -- they all look identical. The fix sources from
``v_account_primary_recommendation`` (one row per entity at its primary
play) so tile sizes reflect "of the entities in this archetype/tier, how
many have THIS play as their primary".

These tests assert against the static source of the query rather than
executing it, mirroring the pattern in ``test_portfolio_totals.py`` so
the dashboard app's runtime deps (streamlit, databricks SDK) stay
optional in the test venv.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

_DATA_PY = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "databricks_app"
    / "src"
    / "data.py"
)


def _playbooks_for_archetype_source() -> str:
    """Return the body of the ``playbooks_for_archetype`` function."""
    src = _DATA_PY.read_text(encoding="utf-8")
    match = re.search(
        r"def playbooks_for_archetype\([^)]*\) -> pd\.DataFrame:\n(.*?)(?:\n@|\ndef |\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert match, "playbooks_for_archetype() not found in data.py"
    return textwrap.dedent(match.group(1))


class TestPlaybooksForArchetypeShape:
    """SQL-shape assertions that prevent the fan-out regression."""

    def test_sources_from_primary_recommendation_view(self):
        # Reading from ``v_account_primary_recommendation`` is what
        # collapses snapshot fan-out: that view emits exactly one row per
        # entity (its primary play), so COUNT(*) per (archetype, playbook,
        # risk_tier) is a distinct-entity count by construction.
        body = _playbooks_for_archetype_source()
        assert "v_account_primary_recommendation" in body

    def test_does_not_source_directly_from_eligibility_snapshot(self):
        # The earlier broken version grouped over
        # ``eligibility_snapshot`` (one row per (entity, playbook)) and
        # counted rows -- so when every entity in an archetype was
        # eligible for every play, every playbook tile in the treemap
        # showed the full archetype's cohort. Pin the source-table swap
        # by matching the actual SQL clause (the docstring still
        # mentions the snapshot table for historical context, so a bare
        # substring check would over-fire).
        body = _playbooks_for_archetype_source()
        assert not re.search(
            r"FROM\s+\{cfg\.fqn_prefix\}\.eligibility_snapshot",
            body,
        )

    def test_groups_by_archetype_playbook_and_risk_tier(self):
        body = _playbooks_for_archetype_source()
        normalized = " ".join(body.split())
        assert "GROUP BY p.archetype_name, p.playbook_name, p.playbook_id, p.risk_tier" in normalized

    def test_filters_to_supplied_archetype(self):
        body = _playbooks_for_archetype_source()
        assert ":archetype_name" in body
        assert "p.archetype_name = :archetype_name" in body

    def test_optionally_filters_to_supplied_risk_tier(self):
        body = _playbooks_for_archetype_source()
        # The risk-tier filter is appended only when the L1 click pinned
        # a tier; the source includes the conditional injection point.
        assert "p.risk_tier = :risk_tier" in body

    def test_projects_fit_score_and_expected_uplift(self):
        # The L2 treemap colours tiles by fit_score and surfaces uplift
        # in the hover. Both fields live on ``v_account_primary_recommendation``
        # already (the view joins ``eligibility_policy.expected_uplift_pct``
        # and the snapshot's ``fit_score``), so no extra explode of the
        # policy table is needed.
        body = _playbooks_for_archetype_source()
        assert "fit_score" in body
        assert "expected_uplift_pct" in body
