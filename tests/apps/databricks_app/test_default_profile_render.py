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


def test_renders_deviation_panel_with_bidirectional_bars(compiled_template, base_context):
    base_context["feature_deviation"] = [
        {"feature_name": "event_count_all_time", "feature_value": 0.321, "z": -1.5},
        {"feature_name": "event_count_365d",     "feature_value": 0.855, "z":  0.6},
        {"feature_name": "regularity_score",     "feature_value": -1.174, "z": 0.0},
    ]
    html = compiled_template(base_context, helpers=HELPERS)
    assert "cr-grid cr-grid-2col" in html
    assert "cr-deviation" in html
    # SHAP became the primary right-pane content; deviation now sits in
    # the secondary "vs. peers" subsection.
    assert "How this customer compares" in html
    assert "event_count_all_time" in html
    assert "dev-neg" in html
    assert "dev-pos" in html
    assert "dev-zero" in html
    assert "width:50.0%" in html
    assert "width:20.0%" in html
    assert "-1.50σ" in html
    assert "+0.60σ" in html


def test_no_feature_deviation_data_hides_panel(compiled_template, base_context):
    base_context["feature_deviation"] = []
    html = compiled_template(base_context, helpers=HELPERS)
    assert "How this customer compares" not in html
    assert "cr-deviation" not in html


def test_renders_shap_panel_with_bidirectional_bars(compiled_template, base_context):
    base_context["account_top_shap_features"] = [
        {
            "feature": "active_span_days", "value": 4.0, "shap_contribution":  0.40,
            "direction": "positive",
            "raw_value": 150, "q05": 10, "q25": 50, "q50": 120, "q75": 200, "q95": 400,
        },
        {
            "feature": "open_rate", "value": 0.1, "shap_contribution": -0.20,
            "direction": "negative",
            "raw_value": 0.85, "q05": 0.0, "q25": 0.1, "q50": 0.3, "q75": 0.55, "q95": 0.8,
        },
        {
            "feature": "regularity_score", "value": 0.0, "shap_contribution":  0.10,
            "direction": "positive",
            # No deviation row available — band falls back to empty (no pill).
        },
    ]
    html = compiled_template(base_context, helpers=HELPERS)
    assert "cr-shap" in html
    assert "Top SHAP drivers" in html
    assert "active_span_days" in html
    assert "shap-pos" in html
    assert "shap-neg" in html
    # Largest absolute contribution → 100% bar; half-magnitude → 50%.
    assert "width:100.0%" in html
    assert "width:50.0%" in html
    # Raw values surface unscaled in the new layout.
    assert "150" in html
    # Trailing ``cr-shap-score`` cell shows the signed log-odds contribution
    # (real, unscaled). Bar magnitude is normalized to the row's strongest
    # driver -- but the trailing label is the actual value, not a derived
    # percentage, so analysts can compare across customers.
    assert "cr-shap-score" in html
    assert "+0.400" in html
    assert "-0.200" in html
    assert "+0.100" in html
    # Glyph / percentage / band pill / in-bar label / model-math disclosure
    # are all GONE -- the layout reverts to the original image-#3 shape:
    # name | raw | bar | signed-contribution. Direction is encoded purely
    # via the bar's colour (yellow = pushes toward churn, green = away).
    assert "cr-shap-bar-glyph" not in html
    assert "cr-shap-bar-label" not in html
    assert "cr-shap-math" not in html
    assert "cr-shap-direction" not in html
    # No quantile band pill in the bundled row body.
    assert "cr-band-typical" not in html
    assert "cr-band-very-high" not in html
    # No share-of-total percentage label and no arrow glyphs.
    assert "57%" not in html
    assert "29%" not in html
    assert "↑" not in html
    assert "↓" not in html


def test_no_account_top_shap_features_hides_shap_panel(compiled_template, base_context):
    base_context["account_top_shap_features"] = []
    html = compiled_template(base_context, helpers=HELPERS)
    assert "Model results interpretation" not in html
    assert "cr-shap" not in html


def test_customer_left_panel_renders_nothing_when_no_data(compiled_template, base_context):
    # The empty-state diagnostic block has been removed. With the UC-table
    # template route, the per-dataset HTML loads reliably (or the bundled
    # fallback runs); the diagnostic was vestigial. When no ``customer``
    # data source is wired, the left panel should now be empty -- no
    # ``cr-customer-profile`` shell, no ``CR_PROFILE_TEMPLATE_PATH`` hint,
    # no ``cr-customer-empty`` block.
    base_context.pop("customer", None)
    html = compiled_template(base_context, helpers=HELPERS)
    assert "cr-customer-empty" not in html
    assert "CR_PROFILE_TEMPLATE_PATH" not in html
    assert "Dataset-specific panel pending" not in html


def test_customer_left_panel_renders_real_data_when_provided(compiled_template, base_context):
    # When a dataset-specific template HAS wired the ``customer`` source
    # the visible-fields-pending shell still renders; covers the customer
    # data path end-to-end.
    base_context["customer"] = {"customer_since": "2024-01-15", "lifecycle_quadrant": "Active"}
    html = compiled_template(base_context, helpers=HELPERS)
    assert "cr-customer-profile" in html
    assert "Dataset-specific panel pending" not in html


def test_two_column_grid_always_present(compiled_template, base_context):
    base_context["feature_deviation"] = []
    html = compiled_template(base_context, helpers=HELPERS)
    # The grid scaffold itself stays — only its contents vary.
    assert "cr-grid cr-grid-2col" in html
