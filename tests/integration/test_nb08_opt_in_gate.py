import pytest

from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
from customer_retention.stages.profiling.recommendation_filter import (
    collect_opt_in_prefixes,
    filter_recommendations_by_opt_in,
)


def _rec(action: str, target_column: str) -> LayeredRecommendation:
    return LayeredRecommendation(
        id=f"{action}-{target_column}",
        layer="gold",
        category="transformation",
        action=action,
        target_column=target_column,
        parameters={},
        rationale="",
        source_notebook="nb04",
    )


def test_filter_keeps_opt_in_zero_inflation_and_other_actions():
    recs = [
        _rec("zero_inflation_handling", "X_count_30d"),
        _rec("zero_inflation_handling", "OPT_IN_COL_count_30d"),
        _rec("log_transform", "Y"),
    ]
    kept = filter_recommendations_by_opt_in(recs, {"some_dataset": ["OPT_IN_COL"]})
    assert [r.target_column for r in kept] == ["OPT_IN_COL_count_30d", "Y"]


def test_filter_empty_opt_in_blocks_all_zero_inflation():
    recs = [
        _rec("zero_inflation_handling", "A"),
        _rec("zero_inflation_handling", "B"),
        _rec("log_transform", "C"),
    ]
    kept = filter_recommendations_by_opt_in(recs, {})
    assert [r.target_column for r in kept] == ["C"]


def test_filter_matches_lag_and_velocity_prefixes():
    recs = [
        _rec("zero_inflation_handling", "lag1_ORDERS_count_30d"),
        _rec("zero_inflation_handling", "velocity_ORDERS_sum_7d"),
        _rec("zero_inflation_handling", "UNRELATED_count_30d"),
    ]
    kept = filter_recommendations_by_opt_in(recs, {"trans": ["ORDERS"]})
    assert {r.target_column for r in kept} == {
        "lag1_ORDERS_count_30d",
        "velocity_ORDERS_sum_7d",
    }


def test_filter_none_opt_in_raises():
    with pytest.raises(ValueError, match="opt_in"):
        filter_recommendations_by_opt_in([], None)


def test_filter_returns_new_list():
    recs = [_rec("log_transform", "X")]
    kept = filter_recommendations_by_opt_in(recs, {})
    assert kept is not recs
    assert kept == recs


def test_filter_preserves_order():
    recs = [
        _rec("log_transform", "first"),
        _rec("zero_inflation_handling", "KEEP_count_30d"),
        _rec("scaling", "second"),
    ]
    kept = filter_recommendations_by_opt_in(recs, {"ds": ["KEEP"]})
    assert [r.target_column for r in kept] == ["first", "KEEP_count_30d", "second"]


def test_collect_opt_in_prefixes_includes_base_and_underscore_forms():
    prefixes = collect_opt_in_prefixes({"ds": ["ALPHA", "BETA"]})
    assert "ALPHA" in prefixes
    assert "ALPHA_" in prefixes
    assert "BETA" in prefixes
    assert "BETA_" in prefixes


def test_collect_opt_in_prefixes_empty_dict():
    assert collect_opt_in_prefixes({}) == []


def test_collect_opt_in_prefixes_none_raises():
    with pytest.raises(ValueError, match="opt_in"):
        collect_opt_in_prefixes(None)


def test_filter_gate_logic_matches_findings_parser():
    from customer_retention.generators.pipeline_generator.findings_parser import _matches_any_prefix

    recs = [_rec("zero_inflation_handling", "RESOLUTION_TARGET_DATE_TIME_count_30d")]
    prefixes = collect_opt_in_prefixes({"case": ["RESOLUTION_TARGET_DATE_TIME"]})
    assert _matches_any_prefix("RESOLUTION_TARGET_DATE_TIME_count_30d", prefixes)
    kept = filter_recommendations_by_opt_in(recs, {"case": ["RESOLUTION_TARGET_DATE_TIME"]})
    assert len(kept) == 1
