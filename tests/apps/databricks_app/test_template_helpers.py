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
