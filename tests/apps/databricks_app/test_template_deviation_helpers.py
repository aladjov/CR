from src.template import HELPERS, DataSource, _parse_frontmatter


def _call(name, *args):
    return HELPERS[name](None, *args)


def test_dev_bar_pct_zero_for_missing():
    assert _call("dev_bar_pct", None) == "0"
    assert _call("dev_bar_pct", float("nan")) == "0"


def test_dev_bar_pct_scales_to_3_sigma():
    # |z| = 3 saturates to 100%
    assert _call("dev_bar_pct", 3.0) == "100.0"
    assert _call("dev_bar_pct", -3.0) == "100.0"
    # Beyond 3σ is still capped at 100
    assert _call("dev_bar_pct", 5.0) == "100.0"
    # Mid-range
    assert _call("dev_bar_pct", 1.5) == "50.0"
    assert _call("dev_bar_pct", -0.6) == "20.0"


def test_dev_bar_pct_handles_unparseable():
    assert _call("dev_bar_pct", "not a number") == "0"


def test_dev_sign_class_cases():
    # ``cr-`` prefix matches the CSS selectors in ``default_profile.css``
    # (``.cr-dev-pos .cr-dev-bar`` etc.) so the bidirectional bars actually
    # paint at the correct colour.
    assert _call("dev_sign_class", 0.5) == "cr-dev-pos"
    assert _call("dev_sign_class", -0.5) == "cr-dev-neg"
    assert _call("dev_sign_class", 0.0) == "cr-dev-zero"
    assert _call("dev_sign_class", None) == "cr-dev-zero"


def test_fmt_signed_z_formats_with_sigma():
    assert _call("fmt_signed_z", 1.234) == "+1.23σ"
    assert _call("fmt_signed_z", -0.50000) == "-0.50σ"
    assert _call("fmt_signed_z", 0.0) == "+0.00σ"
    assert _call("fmt_signed_z", None) == "—"


# ---------------------------------------------------------------------------
# SHAP panel helpers
#
# The right-hand panel renders ``account_top_shap_features`` (already on the
# v_account_explanation row) as bidirectional bars. ``shap_bar_pct`` scales
# each bar relative to the row's largest |contribution|, ``shap_sign_class``
# colours it by causal direction, and ``fmt_signed_shap`` / ``fmt_shap_value``
# format the labels.
# ---------------------------------------------------------------------------


_SHAP_DRIVERS = [
    {"feature": "active_span_days", "value": 4.0, "shap_contribution": 0.40, "direction": "positive"},
    {"feature": "open_rate",        "value": 0.1, "shap_contribution": -0.20, "direction": "negative"},
    {"feature": "regularity_score", "value": 0.0, "shap_contribution": 0.10, "direction": "positive"},
    {"feature": "bounces_total",    "value": 12,  "shap_contribution": None,  "direction": None},
]


def test_shap_bar_pct_normalises_to_row_max():
    # 0.40 is the largest |contribution|, so it shows at 100%
    assert _call("shap_bar_pct", 0.40, _SHAP_DRIVERS) == "100.0"
    # 0.20 is half the local max → 50%
    assert _call("shap_bar_pct", -0.20, _SHAP_DRIVERS) == "50.0"
    # 0.10 is a quarter of the local max → 25%
    assert _call("shap_bar_pct", 0.10, _SHAP_DRIVERS) == "25.0"


def test_shap_bar_pct_handles_missing_contribution():
    assert _call("shap_bar_pct", None, _SHAP_DRIVERS) == "0"
    assert _call("shap_bar_pct", "not a number", _SHAP_DRIVERS) == "0"


def test_shap_bar_pct_safe_when_all_drivers_zero():
    # Degenerate row (all zero / all None) must not divide by zero --
    # falls back to a 1.0 cap so the bar renders cleanly at the input
    # contribution's relative position.
    drivers = [{"shap_contribution": 0.0}, {"shap_contribution": None}]
    assert _call("shap_bar_pct", 0.0, drivers) == "0.0"
    assert _call("shap_bar_pct", 0.5, drivers) == "50.0"


def test_shap_sign_class_cases():
    # ``cr-`` prefix matches the CSS selectors in ``default_profile.css``
    # (``.cr-shap-pos .cr-shap-bar`` etc.) so the bidirectional bars actually
    # paint at the correct colour.
    assert _call("shap_sign_class", 0.4) == "cr-shap-pos"
    assert _call("shap_sign_class", -0.4) == "cr-shap-neg"
    assert _call("shap_sign_class", 0.0) == "cr-shap-zero"
    assert _call("shap_sign_class", None) == "cr-shap-zero"
    assert _call("shap_sign_class", "garbage") == "cr-shap-zero"


def test_fmt_signed_shap_uses_three_decimals():
    assert _call("fmt_signed_shap", 0.1234) == "+0.123"
    assert _call("fmt_signed_shap", -0.5) == "-0.500"
    assert _call("fmt_signed_shap", 0.0) == "+0.000"
    assert _call("fmt_signed_shap", None) == "—"


def test_fmt_shap_value_picks_format_by_magnitude():
    assert _call("fmt_shap_value", 12.0) == "12"           # whole numbers as ints with thousands
    assert _call("fmt_shap_value", 1234.0) == "1,234"
    assert _call("fmt_shap_value", 0.123) == "0.123"        # float
    assert _call("fmt_shap_value", 0.000456) == "4.56e-04"  # tiny values get scientific
    assert _call("fmt_shap_value", "Active") == "Active"    # non-numeric pass-through
    assert _call("fmt_shap_value", None) == "—"


def test_data_source_dataclass_defaults():
    ds = DataSource(name="x", source="t", join_key="entity_id")
    assert ds.as_list is False
    assert ds.limit == 1
    assert ds.order_by is None


def test_template_loader_parses_as_list_with_default_limit_50():
    text = (
        "---\n"
        "data:\n"
        "  feature_deviation:\n"
        "    source: 'v_account_feature_deviation_topn'\n"
        "    join_key: 'entity_id'\n"
        "    as_list: true\n"
        "---\n"
        "<div>{{entity_id}}</div>\n"
    )
    front, body = _parse_frontmatter(text)
    assert front["data"]["feature_deviation"]["as_list"] is True
    assert "<div>" in body


def test_template_loader_keeps_limit_at_1_for_non_list_default():
    # Belt-and-braces: we don't accidentally bump non-list sources to 50.
    # Templates are now loaded directly from raw HTML text (the orchestrator
    # fetches the body from the UC ``v_dashboard_template_active`` view) --
    # no temp-file path detour needed for this contract.
    from src.template import load_template_from_text
    fm = (
        "---\n"
        "data:\n"
        "  account:\n"
        "    source: 'v_account_profile_x'\n"
        "    join_key: 'entity_id'\n"
        "---\n"
        "<div>x</div>\n"
    )
    tpl = load_template_from_text(fm)
    assert len(tpl.data_sources) == 1
    ds = tpl.data_sources[0]
    assert ds.as_list is False
    assert ds.limit == 1


def test_template_loader_list_source_uses_limit_50_default():
    from src.template import load_template_from_text
    fm = (
        "---\n"
        "data:\n"
        "  feature_deviation:\n"
        "    source: 'v_account_feature_deviation_topn'\n"
        "    join_key: 'entity_id'\n"
        "    as_list: true\n"
        "---\n"
        "<div>x</div>\n"
    )
    tpl = load_template_from_text(fm)
    ds = tpl.data_sources[0]
    assert ds.as_list is True
    assert ds.limit == 50


def test_template_loader_explicit_limit_overrides_default():
    from src.template import load_template_from_text
    fm = (
        "---\n"
        "data:\n"
        "  feature_deviation:\n"
        "    source: 'v'\n"
        "    join_key: 'entity_id'\n"
        "    as_list: true\n"
        "    limit: 5\n"
        "---\n"
        "<div>x</div>\n"
    )
    tpl = load_template_from_text(fm)
    assert tpl.data_sources[0].limit == 5
