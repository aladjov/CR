"""Optional Handlebars render check — runs only if pybars is installed.

Confirms the bundled default_profile.html still compiles and renders both the
recommended-action card and the archetype card with the new Phase 2 fields.
"""
from pathlib import Path

import pytest

pybars = pytest.importorskip("pybars")

from src.template import HELPERS, _parse_frontmatter  # noqa: E402

_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "databricks_app"
    / "src"
    / "default_profile.html"
)


@pytest.fixture(scope="module")
def compiled_template():
    text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    _front, body = _parse_frontmatter(text)
    return pybars.Compiler().compile(body)


@pytest.fixture
def base_context():
    return {
        "entity_id":                 "C53DF2",
        "risk_tier":                 "High",
        "churn_probability":         0.42,
        "value_at_risk":             18000,
        "expected_loss":             7560,
        "policy_rank_among_eligible": 1,
        "eligible_playbook_count":    3,
        "playbook_name":             "Onboarding Recovery",
        "playbook_description":      "Rebuild momentum after a stalled start.",
        "playbook_when_applicable":  "First 90 days, low engagement.",
        "playbook_time_zero_definition": "First product login below threshold.",
        "playbook_analysis_population_rule": "Tenure < 90d, no contract.",
        "expected_uplift_pct":       0.10,
        "recommended":               True,
        "is_holdout":                False,
        "archetype_name":            "Stalled Onboarders",
        "archetype_description":     "Customers who never reached activation.",
        "archetype_rationale":       "Low feature_count_30d and short tenure.",
        "archetype_cluster_size":    427,
        "archetype_mean_churn_probability": 0.38,
        "policy_fit_score":          0.84,
        "policy_fit_tier":           "auto",
        "policy_rationale":          "Top prose match against onboarding play.",
        "policy_eligibility_rules_prose": "Tenure under 90 days and zero logins last 14 days.",
        "eligibility_rules_sql":     "tenure_days < 90 AND logins_14d = 0",
        "eligibility_evidence":      "Tenure 47d, 0 logins last 14 days.",
        "alternate_playbooks": [
            {"playbook_id": "RETAIN_DISCOUNT", "playbook_name": "Retention Discount",
             "fit_score": 0.71, "expected_uplift_pct": 0.06},
            {"playbook_id": "CSM_OUTREACH",    "playbook_name": "CSM Outreach",
             "fit_score": 0.58, "expected_uplift_pct": 0.04},
        ],
    }


def test_renders_recommended_action_card(compiled_template, base_context):
    html = compiled_template(base_context, helpers=HELPERS)
    assert 'cr-card-section cr-action' in html
    assert 'cr-action-ok' in html
    assert "Recommended action" in html
    assert "Onboarding Recovery" in html
    assert "10.0%" in html
    assert "Other applicable playbooks" in html
    assert "Retention Discount" in html
    assert "CSM Outreach" in html


def test_renders_archetype_card_with_fit_pill(compiled_template, base_context):
    html = compiled_template(base_context, helpers=HELPERS)
    assert 'cr-card-section cr-archetype' in html
    assert "Stalled Onboarders" in html
    assert "84.0% match" in html
    assert "Auto-fit" in html
    assert "cr-pill-fit-auto" in html
    assert "427" in html
    assert "Technical details" in html
    assert "Eligibility rules · prose" in html
    assert "Tenure under 90 days" in html


def test_holdout_branch_styles_action_as_holdout(compiled_template, base_context):
    base_context["recommended"] = False
    base_context["is_holdout"] = True
    html = compiled_template(base_context, helpers=HELPERS)
    assert "cr-action-holdout" in html
    assert "Holdout" in html


def test_no_alternates_skips_alts_block(compiled_template, base_context):
    base_context["alternate_playbooks"] = []
    html = compiled_template(base_context, helpers=HELPERS)
    assert "Other applicable playbooks" not in html


def test_missing_fit_tier_skips_pill(compiled_template, base_context):
    base_context["policy_fit_tier"] = None
    base_context["policy_fit_score"] = None
    html = compiled_template(base_context, helpers=HELPERS)
    assert "match" not in html.split('cr-card-section cr-archetype')[1].split('</summary>')[0]
    assert "Auto-fit" not in html
