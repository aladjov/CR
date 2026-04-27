"""Structural assertions for the Phase 2 v_account_explanation upgrades.

These tests inspect the rendered SQL text rather than executing it (which would
require Spark), since the existing dashboard_views test suite uses the same
text-level approach.  Phase 5's probe notebook will execute the SQL against
real Databricks output to validate runtime semantics.
"""
from __future__ import annotations

import re

from customer_retention.stages.causal.dashboard_views import (
    DASHBOARD_VIEW_NAMES,
    render_dashboard_view_sql,
)


def _account_explanation_block(rendered: str) -> str:
    statements = re.split(r";\s*(?=CREATE OR REPLACE VIEW)", rendered)
    matches = [s for s in statements if "v_account_explanation" in s]
    assert matches, "v_account_explanation must appear in rendered SQL"
    return matches[0]


def test_v_account_explanation_is_listed_in_dashboard_view_names():
    assert "v_account_explanation" in DASHBOARD_VIEW_NAMES


def test_v_account_explanation_exposes_playbook_detail_columns():
    block = _account_explanation_block(render_dashboard_view_sql("c", "s"))
    for column in (
        "playbook_description",
        "playbook_time_zero_definition",
        "playbook_analysis_population_rule",
        "playbook_when_applicable",
    ):
        assert column in block, f"missing column {column!r}"


def test_v_account_explanation_exposes_archetype_detail_columns():
    block = _account_explanation_block(render_dashboard_view_sql("c", "s"))
    for column in (
        "archetype_description",
        "archetype_rationale",
        "archetype_cluster_size",
        "archetype_mean_churn_probability",
    ):
        assert column in block, f"missing column {column!r}"


def test_v_account_explanation_exposes_policy_fit_columns():
    block = _account_explanation_block(render_dashboard_view_sql("c", "s"))
    for column in (
        "policy_fit_score",
        "policy_fit_tier",
        "policy_eligibility_rules_prose",
    ):
        assert column in block, f"missing column {column!r}"


def test_v_account_explanation_emits_alternate_playbooks():
    block = _account_explanation_block(render_dashboard_view_sql("c", "s"))
    assert "alternate_playbooks" in block
    # Built from per-archetype set, excluding the chosen playbook, sorted desc by fit.
    assert "exploded_policies" in block
    assert "archetype_playbook_set" in block
    assert "playbook_id != s.playbook_id" in block
    assert "ARRAY_SORT" in block
    assert "fit_score" in block


def test_alternate_playbooks_uses_lateral_explode_on_archetype_ids():
    block = _account_explanation_block(render_dashboard_view_sql("c", "s"))
    assert "LATERAL VIEW EXPLODE(e.archetype_ids)" in block


def test_existing_columns_remain_for_backwards_compatibility():
    # Phase 2 must NOT remove any column the dashboard relies on today.
    block = _account_explanation_block(render_dashboard_view_sql("c", "s"))
    for column in (
        "entity_id",
        "playbook_id",
        "playbook_name",
        "archetype_id",
        "archetype_name",
        "archetype_feature_thresholds",
        "archetype_top_shap_features",
        "risk_tier",
        "churn_probability",
        "value_at_risk",
        "expected_loss",
        "eligibility_evidence",
        "account_top_shap_features",
        "eligibility_rules_sql",
        "expected_uplift_pct",
        "policy_rationale",
        "recommended",
        "is_holdout",
        "policy_rank_among_eligible",
        "eligible_playbook_count",
        "scoring_run_id",
        "as_of_date",
    ):
        assert column in block, f"regression: column {column!r} disappeared"
