from src.template import HELPERS


def _call(name, *args):
    """Pybars helpers receive a `this` context as the first arg."""
    return HELPERS[name](None, *args)


def test_fit_tier_label_known_values():
    assert _call("fit_tier_label", "auto") == "Auto-fit"
    assert _call("fit_tier_label", "review") == "Manual review"
    assert _call("fit_tier_label", "manual") == "Manual"
    assert _call("fit_tier_label", "catch_all") == "Default"


def test_fit_tier_label_unknown_value_falls_back_titlecased():
    assert _call("fit_tier_label", "weird_thing") == "Weird Thing"


def test_fit_tier_label_missing_returns_empty():
    assert _call("fit_tier_label", None) == ""
    assert _call("fit_tier_label", float("nan")) == ""


def test_fit_tier_class_maps_known_values():
    assert _call("fit_tier_class", "auto") == "fit-auto"
    assert _call("fit_tier_class", "review") == "fit-review"
    assert _call("fit_tier_class", "catch_all") == "fit-catch_all"


def test_fit_tier_class_handles_caps_and_spaces():
    assert _call("fit_tier_class", "Auto") == "fit-auto"
    assert _call("fit_tier_class", "catch all") == "fit-catch_all"


def test_fit_tier_class_unknown_returns_unknown_class():
    assert _call("fit_tier_class", "ad_hoc") == "fit-unknown"
    assert _call("fit_tier_class", None) == "fit-unknown"


# ----------------------------------------------------------------------------
# quantile_band — 5-band classification (very low / low / typical / high / very high)
# ----------------------------------------------------------------------------


def test_quantile_band_below_q05_is_very_low():
    assert _call("quantile_band", 5, 10, 50, 200, 400) == "very low"


def test_quantile_band_between_q05_and_q25_is_low():
    assert _call("quantile_band", 30, 10, 50, 200, 400) == "low"


def test_quantile_band_inside_iqr_is_typical():
    assert _call("quantile_band", 150, 10, 50, 200, 400) == "typical"
    # Boundary: value == q25 lands in typical (not low).
    assert _call("quantile_band", 50, 10, 50, 200, 400) == "typical"


def test_quantile_band_above_q75_is_high():
    assert _call("quantile_band", 250, 10, 50, 200, 400) == "high"


def test_quantile_band_above_q95_is_very_high():
    assert _call("quantile_band", 500, 10, 50, 200, 400) == "very high"
    # Boundary: value == q95 saturates to very high.
    assert _call("quantile_band", 400, 10, 50, 200, 400) == "very high"


def test_quantile_band_missing_inputs_return_empty_string():
    # Empty string lets the template's ``{{#if quantile_band ...}}`` branch
    # collapse the band pill cleanly via CSS ``:empty`` instead of rendering
    # a visible em-dash placeholder. Stale "—" assertions caught a real
    # regression — the panel kept showing dashes on rows without population
    # stats, which the redesign explicitly removed.
    assert _call("quantile_band", None, 10, 50, 200, 400) == ""
    assert _call("quantile_band", 100, None, 50, 200, 400) == ""
    assert _call("quantile_band", 100, 10, 50, 200, None) == ""
    assert _call("quantile_band", "not_a_number", 10, 50, 200, 400) == ""


# ----------------------------------------------------------------------------
# band_class — pill class names matching the 5 bands + unknown fallback
# ----------------------------------------------------------------------------


def test_band_class_maps_each_band():
    assert _call("band_class", "very low") == "cr-band-very-low"
    assert _call("band_class", "low") == "cr-band-low"
    assert _call("band_class", "typical") == "cr-band-typical"
    assert _call("band_class", "high") == "cr-band-high"
    assert _call("band_class", "very high") == "cr-band-very-high"


def test_band_class_unknown_inputs_fall_back():
    # Empty string is the new "unknown" sentinel emitted by quantile_band when
    # population stats are missing; the band_class helper still maps it to the
    # ``cr-band-unknown`` CSS class so paired pill rendering stays consistent.
    assert _call("band_class", "") == "cr-band-unknown"
    assert _call("band_class", None) == "cr-band-unknown"
    assert _call("band_class", "garbage") == "cr-band-unknown"


# ----------------------------------------------------------------------------
# direction_phrase / direction_class — "protective" / "risk-driving" / "neutral"
# ----------------------------------------------------------------------------


def test_direction_phrase_positive_contribution_is_risk_driving():
    assert _call("direction_phrase", 0.42) == "risk-driving"


def test_direction_phrase_negative_contribution_is_protective():
    assert _call("direction_phrase", -0.42) == "protective"


def test_direction_phrase_near_zero_is_neutral():
    # Tiny contributions get rounded to neutral so a 0.001 push doesn't
    # get labelled as "risk-driving" with a barely-visible bar.
    assert _call("direction_phrase", 1e-5) == "neutral"
    assert _call("direction_phrase", 0.0) == "neutral"
    assert _call("direction_phrase", None) == "neutral"


def test_direction_class_matches_phrase():
    assert _call("direction_class", 0.42) == "cr-dir-risk"
    assert _call("direction_class", -0.42) == "cr-dir-protective"
    assert _call("direction_class", 0.0) == "cr-dir-neutral"


# ----------------------------------------------------------------------------
# derivation_phrase — aggregation_kind → CSM-readable phrase, with fallback
# to feature-name pattern hints when feature_meta is missing.
# ----------------------------------------------------------------------------


def test_derivation_phrase_uses_aggregation_kind_when_present():
    out = _call(
        "derivation_phrase", "count", "event_count_30d", ["event"], "last 30 days",
    )
    assert "Count" in out
    assert "event" in out
    assert "last 30 days" in out


def test_derivation_phrase_recency_days_has_canonical_wording():
    out = _call(
        "derivation_phrase", "recency_days", "days_since_last_login", ["login"], "lifetime",
    )
    assert "Days since" in out
    assert "login" in out


def test_derivation_phrase_falls_back_to_name_pattern_when_aggregation_kind_missing():
    out = _call(
        "derivation_phrase", None, "active_span_velocity", ["event_timestamp"], None,
    )
    assert "Rate" in out or "velocity" in out.lower()


def test_derivation_phrase_uses_feature_name_when_source_columns_empty():
    out = _call("derivation_phrase", "count", "event_count_all_time", [], None)
    # Falls back to the feature name as the {source} placeholder.
    assert "event_count_all_time" in out


def test_derivation_phrase_returns_empty_when_nothing_known():
    assert _call("derivation_phrase", None, None, [], None) == ""


# ----------------------------------------------------------------------------
# fmt_raw_value — readable customer value (integer / float / scientific)
# ----------------------------------------------------------------------------


def test_fmt_raw_value_renders_whole_number_as_integer():
    assert _call("fmt_raw_value", 312.0) == "312"


def test_fmt_raw_value_renders_float_three_decimals():
    assert _call("fmt_raw_value", 0.456789) == "0.457"


def test_fmt_raw_value_missing_returns_empty_string():
    # Empty string lets the SHAP-row raw-value cell collapse via CSS
    # ``:empty`` instead of rendering a visible em-dash placeholder when
    # the customer's raw value isn't available.
    assert _call("fmt_raw_value", None) == ""


# ----------------------------------------------------------------------------
# direction_glyph / direction_tooltip — the bar-edge glyph + hover phrase
# that replaced the permanent direction pill column.
# ----------------------------------------------------------------------------


def test_direction_glyph_arrows_match_sign():
    assert _call("direction_glyph", 0.42) == "↑"   # up arrow → risk
    assert _call("direction_glyph", -0.42) == "↓"  # down arrow → protective
    assert _call("direction_glyph", 0.0) == ""
    assert _call("direction_glyph", None) == ""


def test_direction_tooltip_long_phrases():
    risk = _call("direction_tooltip", 0.42)
    prot = _call("direction_tooltip", -0.42)
    assert "Risk-driving" in risk and "churn" in risk
    assert "Protective" in prot and "retention" in prot


# ----------------------------------------------------------------------------
# shap_share_pct — share of total visible |SHAP|, sums to 100% across rows
# ----------------------------------------------------------------------------


def test_shap_share_pct_sums_to_100_across_visible_rows():
    drivers = [
        {"shap_contribution": 0.40},
        {"shap_contribution": -0.20},
        {"shap_contribution": 0.10},
    ]
    # Total |shap| = 0.7. Shares: 0.40/0.7 ≈ 57%, 0.20/0.7 ≈ 29%, 0.10/0.7 ≈ 14%.
    assert _call("shap_share_pct", 0.40, drivers) == "57%"
    assert _call("shap_share_pct", -0.20, drivers) == "29%"
    assert _call("shap_share_pct", 0.10, drivers) == "14%"


def test_shap_share_pct_tiny_value_renders_lt_1pct():
    drivers = [
        {"shap_contribution": 1.0},
        {"shap_contribution": 0.001},  # 0.1% share → <1%
    ]
    assert _call("shap_share_pct", 0.001, drivers) == "<1%"


def test_shap_share_pct_missing_returns_empty_string():
    # Paired with a {{#if ...}} in the template so the label disappears
    # cleanly on rows that have no SHAP contribution at all.
    assert _call("shap_share_pct", None, [{"shap_contribution": 1.0}]) == ""
