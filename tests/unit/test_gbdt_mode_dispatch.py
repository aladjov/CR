import pytest

from customer_retention.stages.features.feature_selector import (
    resolve_gbdt_selection_mode,
    run_chi_squared_rescue_selection,
    run_gbdt_importance_selection,
)


@pytest.mark.parametrize(
    "mode,expected_fn",
    [
        ("standalone", run_gbdt_importance_selection),
        ("chi_squared_rescue", run_chi_squared_rescue_selection),
        ("chain", run_gbdt_importance_selection),
    ],
)
def test_gbdt_mode_dispatch_known_modes(mode, expected_fn):
    assert resolve_gbdt_selection_mode(mode) is expected_fn


def test_gbdt_mode_dispatch_unknown_mode_raises():
    with pytest.raises(ValueError, match="GBDT_SELECTION_MODE"):
        resolve_gbdt_selection_mode("rescue_only")


def test_gbdt_mode_dispatch_none_raises():
    with pytest.raises(ValueError, match="GBDT_SELECTION_MODE"):
        resolve_gbdt_selection_mode(None)


def test_gbdt_mode_dispatch_error_lists_valid_modes():
    with pytest.raises(ValueError) as exc:
        resolve_gbdt_selection_mode("bogus")
    assert "standalone" in str(exc.value)
    assert "chi_squared_rescue" in str(exc.value)


def test_gbdt_mode_dispatch_rejects_empty_string():
    with pytest.raises(ValueError, match="GBDT_SELECTION_MODE"):
        resolve_gbdt_selection_mode("")
