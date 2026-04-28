from src.masthead import (
    context_segments,
    horizon_phrase,
    l1_title_html,
    masthead_title,
)


def test_horizon_phrase_uses_int_horizon():
    assert horizon_phrase({"horizon_days": 270}) == "Churn Risk in next 270 days"


def test_horizon_phrase_coerces_numeric_strings():
    assert horizon_phrase({"horizon_days": "90"}) == "Churn Risk in next 90 days"


def test_horizon_phrase_returns_none_when_missing():
    assert horizon_phrase({}) is None
    assert horizon_phrase({"horizon_days": None}) is None


def test_horizon_phrase_returns_none_when_unparsable():
    assert horizon_phrase({"horizon_days": "n/a"}) is None


def test_context_segments_orders_objective_posture_model():
    ctx = {
        "primary_objective": "immediate_risk",
        "temporal_posture":  "short_memory",
        "model_type":        "xgboost",
    }
    assert context_segments(ctx) == ["Immediate risk", "Reactive posture", "XGBoost"]


def test_context_segments_skips_empty_fields():
    assert context_segments({"primary_objective": "renewal_risk"}) == ["Renewal risk"]


def test_context_segments_handles_unknown_enum_gracefully():
    out = context_segments({"primary_objective": "weird_value"})
    assert out == ["Weird Value"]


def test_masthead_title_full_context():
    ctx = {
        "horizon_days":      270,
        "primary_objective": "immediate_risk",
        "temporal_posture":  "long_memory",
        "model_type":        "lightgbm",
    }
    title, segments = masthead_title(ctx)
    assert title == "Churn Risk in next 270 days"
    assert segments == ["LightGBM", "Immediate risk", "Stable posture"]


def test_masthead_title_falls_back_when_empty():
    title, segments = masthead_title({})
    assert title == "Churn Risk"
    assert segments == []


def test_l1_title_html_falls_back_to_actionable_insights_when_no_horizon():
    out = l1_title_html({})
    assert "Churn Risk" in out
    assert "Actionable insights" in out
    assert "<em>" in out and "</em>" in out
    assert "book" not in out.lower()
    assert "glance" not in out.lower()


def test_l1_title_html_renders_horizon_as_single_run():
    # The masthead at the top of the page already shows objective/posture
    # segments, so the L1 hero stays scoped to the horizon line. The whole
    # horizon phrase renders as one styled run -- the earlier two-tone
    # treatment (italic display face + sans annotation) split the title
    # visually, so we revert to a single coherent string.
    ctx = {
        "horizon_days":      270,
        "primary_objective": "immediate_risk",
        "temporal_posture":  "short_memory",
        "model_type":        "xgboost",
    }
    html = l1_title_html(ctx)
    assert html == "Churn Risk in next 270 days"
    for token in ("Immediate risk", "Reactive posture", "XGBoost"):
        assert token not in html
    assert "<em>" not in html
    assert "<span" not in html


def test_l1_title_html_horizon_only_renders_plain_string():
    html = l1_title_html({"horizon_days": 30})
    assert html == "Churn Risk in next 30 days"


def test_l1_title_html_partial_horizon_only_renders_dynamic_no_static():
    out = l1_title_html({"horizon_days": 60})
    assert out == "Churn Risk in next 60 days"
    assert "book" not in out.lower()


def test_l1_title_html_unparsable_horizon_falls_back():
    out = l1_title_html({"horizon_days": "thirty"})
    assert "Churn Risk" in out
    assert "Actionable insights" in out
