"""Regression tests for the L1 KPI tile query.

The L1 tiles answer "of all customers right now, how many are eligible /
recommended", which is per-account semantics. ``eligibility_snapshot`` is
keyed on ``(scoring_run_id, entity_id, playbook_id)``, so an account
matching N plays appears N times -- summing rows multi-counts. These
tests pin the SQL shape that prevents that regression and lock in the
returned column contract.

We assert against the static source of ``portfolio_totals`` rather than
executing it, because the dashboard app's runtime deps (streamlit,
databricks SDK) are intentionally absent from the test venv -- importing
the live module would require stubbing the entire app surface.
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


def _portfolio_totals_source() -> str:
    """Return just the body of the ``portfolio_totals`` function."""
    src = _DATA_PY.read_text(encoding="utf-8")
    match = re.search(
        r"def portfolio_totals\([^)]*\) -> pd\.DataFrame:\n(.*?)(?:\n@|\ndef |\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert match, "portfolio_totals() not found in data.py"
    return textwrap.dedent(match.group(1))


class TestPortfolioTotalsShape:
    """SQL-shape assertions that prevent the multi-counting regression."""

    def test_collapses_to_primary_play_per_entity(self):
        # Entity-grain collapse must be explicit -- without this the
        # query sums ``eligibility_snapshot`` rows and double-counts every
        # account that matches more than one playbook (the original bug).
        body = _portfolio_totals_source()
        assert "policy_rank_among_eligible = 1" in body

    def test_pins_to_a_single_scoring_run(self):
        body = _portfolio_totals_source()
        assert "latest_run" in body
        assert "MAX(as_of_date)" in body
        assert "LIMIT 1" in body

    def test_does_not_source_from_rollup_view(self):
        # ``v_portfolio_risk_matrix`` is row-grain (per playbook x risk_tier
        # COUNT(*)). Summing its eligible_count gives row counts, not
        # entity counts. The fix queries eligibility_snapshot directly --
        # check the actual SQL string, ignoring the docstring (which
        # mentions the rollup view to explain why we don't use it).
        body = _portfolio_totals_source()
        # Strip the leading docstring before checking.
        sql_body = re.sub(r'""".*?"""', "", body, count=1, flags=re.DOTALL)
        assert "v_portfolio_risk_matrix" not in sql_body
        assert "eligibility_snapshot" in sql_body

    def test_active_playbooks_uses_count_distinct_on_recommended(self):
        body = _portfolio_totals_source()
        assert "COUNT(DISTINCT" in body
        assert "playbook_id" in body
        assert "recommended" in body

    def test_honours_dashboard_visibility_flag(self):
        # COALESCE keeps pre-migration rows visible; post-migration rows
        # with is_dashboard_visible = FALSE must be excluded so the L1
        # tiles match what the L2/L3 drill shows.
        body = _portfolio_totals_source()
        assert "is_dashboard_visible" in body

    def test_emits_the_four_tile_columns(self):
        # app.py reads these four keys via ``r.get(...)``. Renaming any of
        # them silently breaks the L1 tiles -- pin the contract here.
        body = _portfolio_totals_source()
        for col in (
            "total_eligible",
            "total_recommended",
            "total_value_at_risk",
            "active_playbooks",
        ):
            assert f"AS {col}" in body, f"column {col!r} missing from portfolio_totals SQL"
