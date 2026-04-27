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


def test_l1_title_html_renders_dynamic_with_flourish():
    ctx = {
        "horizon_days":      270,
        "primary_objective": "immediate_risk",
        "temporal_posture":  "short_memory",
        "model_type":        "xgboost",
    }
    html = l1_title_html(ctx)
    assert html.startswith("Churn Risk in next 270 days ")
    assert "<em>" in html and "</em>" in html
    for token in ("Immediate risk", "Reactive posture", "XGBoost"):
        assert token in html


def test_l1_title_html_horizon_only_has_no_em():
    html = l1_title_html({"horizon_days": 30})
    assert html == "Churn Risk in next 30 days"


def test_l1_title_html_partial_horizon_only_renders_dynamic_no_static():
    # When horizon is present but every other field is NULL, we must still
    # render the horizon-bearing dynamic title (no static fallback string).
    out = l1_title_html({"horizon_days": 60})
    assert out == "Churn Risk in next 60 days"
    assert "book" not in out.lower()


def test_l1_title_html_escapes_unknown_segment_values():
    html = l1_title_html({
        "horizon_days":      90,
        "primary_objective": "<script>",
    })
    assert "<script>" not in html
    assert "&lt;" in html and "&gt;" in html
