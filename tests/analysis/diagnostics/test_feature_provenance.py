"""Tests for the NB09 feature provenance helper."""

from __future__ import annotations

import pytest

from customer_retention.analysis.diagnostics.feature_provenance import (
    build_overall_source_distribution,
    build_provenance_table,
    cached_target_correlations,
    feature_origin_for,
    feature_origins_for,
    parse_feature_provenance,
    source_histogram,
)
from customer_retention.stages.modeling.feature_profile import FeatureOrigin


@pytest.fixture
def source_columns():
    return {
        "subscription": {"NET_PRICE", "DOCUMENT_ALLOTMENT", "NUMBER_OF_ASINS"},
        "contract": {"ACTIVATED_DATE", "CONTRACT_VALUE", "PROBABILITY_PERCENTAGE"},
        "case": {"CASE_AGE_HOURS", "STATUS"},
    }


class TestParseFeatureProvenance:
    def test_exact_match_to_raw_column(self, source_columns):
        p = parse_feature_provenance("NET_PRICE", source_columns)
        assert p.source == "subscription"
        assert p.base_column == "NET_PRICE"
        assert p.family == ""
        assert p.lag_prefix is None
        assert p.is_resolved

    def test_simple_aggregation_suffix(self, source_columns):
        p = parse_feature_provenance("NET_PRICE_sum_180d", source_columns)
        assert p.source == "subscription"
        assert p.base_column == "NET_PRICE"
        assert p.family == "sum_180d"

    def test_zero_inflation_flag_suffix(self, source_columns):
        p = parse_feature_provenance("NET_PRICE_count_365d_is_zero", source_columns)
        assert p.source == "subscription"
        assert p.base_column == "NET_PRICE"
        assert p.family == "count_365d_is_zero"

    def test_lag_prefix_stripped(self, source_columns):
        p = parse_feature_provenance("lag2_DOCUMENT_ALLOTMENT_count_30d", source_columns)
        assert p.lag_prefix == "lag2"
        assert p.source == "subscription"
        assert p.base_column == "DOCUMENT_ALLOTMENT"
        assert p.family == "count_30d"

    def test_velocity_prefix_stripped(self, source_columns):
        p = parse_feature_provenance("velocity_NUMBER_OF_ASINS", source_columns)
        assert p.lag_prefix == "velocity"
        assert p.source == "subscription"
        assert p.base_column == "NUMBER_OF_ASINS"

    def test_datetime_derivation_family(self, source_columns):
        p = parse_feature_provenance("ACTIVATED_DATE_delta_hours_max_all_time", source_columns)
        assert p.source == "contract"
        assert p.base_column == "ACTIVATED_DATE"
        assert p.family == "delta_hours_max_all_time"

    def test_cohort_relative_family(self, source_columns):
        p = parse_feature_provenance("CONTRACT_VALUE_vs_cohort_pct", source_columns)
        assert p.source == "contract"
        assert p.base_column == "CONTRACT_VALUE"
        assert p.family == "vs_cohort_pct"

    def test_unresolved_when_no_source_matches(self, source_columns):
        p = parse_feature_provenance("UNKNOWN_COLUMN_sum_30d", source_columns)
        assert p.source is None
        assert p.base_column is None
        assert not p.is_resolved

    def test_longest_match_wins_when_one_column_prefixes_another(self):
        # NET_PRICE prefixes NET_PRICE_TIER. The longest column wins.
        cols = {"subscription": {"NET_PRICE", "NET_PRICE_TIER"}}
        p = parse_feature_provenance("NET_PRICE_TIER_max_180d", cols)
        assert p.base_column == "NET_PRICE_TIER"
        assert p.family == "max_180d"

    def test_no_match_when_column_is_a_substring_but_not_a_prefix_segment(self, source_columns):
        # `NET_PRICEY_sum` should NOT match NET_PRICE (no underscore boundary).
        p = parse_feature_provenance("NET_PRICEY_sum_180d", source_columns)
        assert p.source is None


class TestCachedTargetCorrelations:
    def test_none_recommendations_returns_empty(self):
        assert cached_target_correlations(None) == {}

    def test_extracts_prioritize_correlations(self):
        class _Rec:
            def __init__(self, action, target, corr):
                self.action = action
                self.target_column = target
                self.parameters = {"correlation": corr}

        class _Gold:
            feature_selection = [
                _Rec("prioritize", "feat_a", 0.85),
                _Rec("prioritize", "feat_b", -0.42),  # absolute value
                _Rec("drop_weak", "feat_c", 0.05),     # ignored — not prioritize
            ]

        class _Recs:
            gold = _Gold()

        result = cached_target_correlations(_Recs())
        assert result == {"feat_a": 0.85, "feat_b": 0.42}

    def test_handles_missing_correlation_param(self):
        class _Rec:
            action = "prioritize"
            target_column = "feat_a"
            parameters = {}  # no correlation key

        class _Gold:
            feature_selection = [_Rec()]

        class _Recs:
            gold = _Gold()

        assert cached_target_correlations(_Recs()) == {}

    def test_handles_non_numeric_correlation(self):
        class _Rec:
            action = "prioritize"
            target_column = "feat_a"
            parameters = {"correlation": "not_a_number"}

        class _Gold:
            feature_selection = [_Rec()]

        class _Recs:
            gold = _Gold()

        assert cached_target_correlations(_Recs()) == {}


class TestBuildProvenanceTable:
    def test_orders_by_importance_and_caps_at_top_n(self, source_columns):
        importances = {
            "NET_PRICE_sum_180d": 0.42,
            "DOCUMENT_ALLOTMENT_count_30d": 0.30,
            "CONTRACT_VALUE_max_all_time": 0.18,
            "ACTIVATED_DATE_delta_hours_max_all_time": 0.05,
        }
        rows = build_provenance_table(importances, source_columns, {}, top_n=3)
        assert [r.feature for r in rows] == [
            "NET_PRICE_sum_180d",
            "DOCUMENT_ALLOTMENT_count_30d",
            "CONTRACT_VALUE_max_all_time",
        ]

    def test_status_confirmed_when_cached_correlation_present(self, source_columns):
        importances = {"NET_PRICE_sum_180d": 0.5}
        cached = {"NET_PRICE_sum_180d": 0.65}
        rows = build_provenance_table(importances, source_columns, cached, top_n=10)
        assert rows[0].nb05_status == "confirmed"
        assert rows[0].nb05_correlation == 0.65

    def test_status_new_when_no_cached_correlation(self, source_columns):
        importances = {"NET_PRICE_count_365d_is_zero": 0.5}
        rows = build_provenance_table(importances, source_columns, {}, top_n=10)
        assert rows[0].nb05_status == "new"
        assert rows[0].nb05_correlation is None

    def test_status_missing_when_cached_correlation_below_threshold(self, source_columns):
        importances = {"NET_PRICE_sum_180d": 0.5}
        cached = {"NET_PRICE_sum_180d": 0.05}  # below 0.1
        rows = build_provenance_table(importances, source_columns, cached, top_n=10)
        assert rows[0].nb05_status == "missing"

    def test_falls_back_to_base_column_in_cache(self, source_columns):
        # NB05 may have cached the raw column, not the derived form.
        importances = {"NET_PRICE_sum_180d": 0.5}
        cached = {"NET_PRICE": 0.55}
        rows = build_provenance_table(importances, source_columns, cached, top_n=10)
        assert rows[0].nb05_status == "confirmed"
        assert rows[0].nb05_correlation == 0.55

    def test_unresolved_features_marked(self, source_columns):
        importances = {"UNKNOWN_FEATURE_sum_30d": 0.5}
        rows = build_provenance_table(importances, source_columns, {}, top_n=10)
        assert rows[0].source == "(unresolved)"
        assert rows[0].base_column == "UNKNOWN_FEATURE_sum_30d"

    def test_empty_importances_returns_empty(self, source_columns):
        assert build_provenance_table({}, source_columns, {}, top_n=10) == []


class TestSourceHistogram:
    def test_aggregates_by_source(self, source_columns):
        importances = {
            "NET_PRICE_sum_180d": 0.5,
            "DOCUMENT_ALLOTMENT_count_30d": 0.3,
            "CONTRACT_VALUE_max_all_time": 0.2,
        }
        rows = build_provenance_table(importances, source_columns, {}, top_n=10)
        hist = source_histogram(rows)
        # Sorted by count descending
        assert hist[0][0] == "subscription"
        assert hist[0][1] == 2
        assert hist[1][0] == "contract"
        assert hist[1][1] == 1

    def test_empty_rows_returns_empty(self):
        assert source_histogram([]) == []


@pytest.fixture
def source_column_types():
    return {
        "subscription": {
            "NET_PRICE": "numeric_continuous",
            "DOCUMENT_ALLOTMENT": "numeric_discrete",
            "NUMBER_OF_ASINS": "numeric_discrete",
        },
        "contract": {
            "ACTIVATED_DATE": "datetime",
            "CONTRACT_VALUE": "numeric_continuous",
            "PROBABILITY_PERCENTAGE": "numeric_continuous",
        },
        "case": {
            "CASE_AGE_HOURS": "numeric_continuous",
            "STATUS": "categorical_nominal",
        },
    }


class TestOriginalDtypeResolution:
    def test_dtype_populated_for_resolved_feature(self, source_columns, source_column_types):
        importances = {"NET_PRICE_sum_180d": 0.5}
        rows = build_provenance_table(
            importances, source_columns, {}, top_n=10,
            source_column_types=source_column_types,
        )
        assert rows[0].original_dtype == "numeric_continuous"

    def test_dtype_populated_for_categorical_base(self, source_columns, source_column_types):
        importances = {"STATUS_count_30d": 0.5}
        rows = build_provenance_table(
            importances, source_columns, {}, top_n=10,
            source_column_types=source_column_types,
        )
        assert rows[0].original_dtype == "categorical_nominal"

    def test_dtype_none_for_interaction_feature(self, source_columns, source_column_types):
        # `_x_` in the name signals an interaction combining two inputs.
        # Source still resolves (to the first prefix match) so the overall
        # pie chart can count it, but dtype is ambiguous → None.
        importances = {"NET_PRICE_sum_180d_x_CONTRACT_VALUE_max_180d": 0.5}
        rows = build_provenance_table(
            importances, source_columns, {}, top_n=10,
            source_column_types=source_column_types,
        )
        assert rows[0].original_dtype is None

    def test_dtype_none_for_ratio_feature(self, source_columns, source_column_types):
        importances = {"NET_PRICE_ratio_CONTRACT_VALUE": 0.5}
        rows = build_provenance_table(
            importances, source_columns, {}, top_n=10,
            source_column_types=source_column_types,
        )
        assert rows[0].original_dtype is None

    def test_dtype_none_when_truly_unresolved(self, source_columns, source_column_types):
        importances = {"UNKNOWN_FEATURE_sum_30d": 0.5}
        rows = build_provenance_table(
            importances, source_columns, {}, top_n=10,
            source_column_types=source_column_types,
        )
        assert rows[0].source == "(unresolved)"
        assert rows[0].original_dtype is None

    def test_dtype_none_when_types_not_provided(self, source_columns):
        importances = {"NET_PRICE_sum_180d": 0.5}
        rows = build_provenance_table(importances, source_columns, {}, top_n=10)
        assert rows[0].original_dtype is None

    def test_dtype_none_when_base_missing_from_types(self, source_columns):
        types = {"subscription": {"DOCUMENT_ALLOTMENT": "numeric_discrete"}}
        importances = {"NET_PRICE_sum_180d": 0.5}
        rows = build_provenance_table(
            importances, source_columns, {}, top_n=10,
            source_column_types=types,
        )
        assert rows[0].source == "subscription"
        assert rows[0].original_dtype is None


class TestTopNBump:
    def test_top_n_fifty_returned_when_available(self, source_columns):
        importances = {f"NET_PRICE_sum_{i}d": 1.0 / (i + 1) for i in range(60)}
        rows = build_provenance_table(importances, source_columns, {}, top_n=50)
        assert len(rows) == 50
        assert rows[0].importance > rows[-1].importance


class TestBuildOverallSourceDistribution:
    def test_aggregates_unique_features_across_models(self, source_columns):
        model_importances = {
            "xgb": {
                "NET_PRICE_sum_180d": 0.5,
                "DOCUMENT_ALLOTMENT_count_30d": 0.3,
                "CONTRACT_VALUE_max_all_time": 0.2,
            },
            "rf": {
                "NET_PRICE_sum_180d": 0.4,  # duplicate — counted once
                "NUMBER_OF_ASINS_mean_90d": 0.25,
                "CASE_AGE_HOURS_sum_30d": 0.1,
            },
        }
        dist = build_overall_source_distribution(model_importances, source_columns)
        as_dict = {src: count for src, count in dist}
        assert as_dict["subscription"] == 3
        assert as_dict["contract"] == 1
        assert as_dict["case"] == 1

    def test_unresolved_features_grouped(self, source_columns):
        model_importances = {
            "xgb": {
                "NET_PRICE_sum_180d": 0.5,
                "MYSTERY_FEATURE": 0.2,
            },
        }
        dist = build_overall_source_distribution(model_importances, source_columns)
        as_dict = {src: count for src, count in dist}
        assert as_dict["(unresolved)"] == 1
        assert as_dict["subscription"] == 1

    def test_sorted_by_count_descending(self, source_columns):
        model_importances = {
            "xgb": {
                "NET_PRICE_sum_180d": 0.5,
                "NUMBER_OF_ASINS_mean_90d": 0.3,
                "CONTRACT_VALUE_max_all_time": 0.2,
            },
        }
        dist = build_overall_source_distribution(model_importances, source_columns)
        counts = [count for _src, count in dist]
        assert counts == sorted(counts, reverse=True)

    def test_empty_inputs_return_empty_list(self):
        assert build_overall_source_distribution({}, {}) == []


class TestFeatureOriginFor:
    def test_bronze_aggregate_resolved(self, source_columns):
        origin = feature_origin_for("NET_PRICE_sum_180d", source_columns)
        assert isinstance(origin, FeatureOrigin)
        assert origin.source == "subscription"
        assert origin.base_column == "NET_PRICE"
        assert origin.family == "sum_180d"
        assert origin.derivation == "bronze_aggregate"
        assert origin.parents == ()

    def test_lag_prefix_carried_through(self, source_columns):
        origin = feature_origin_for("lag2_DOCUMENT_ALLOTMENT_count_30d", source_columns)
        assert origin.lag_prefix == "lag2"
        assert origin.source == "subscription"
        assert origin.derivation == "bronze_aggregate"

    def test_unresolved_feature(self, source_columns):
        origin = feature_origin_for("UNKNOWN_sum_30d", source_columns)
        assert origin.source is None
        assert origin.base_column is None
        assert origin.derivation is None

    def test_derived_ratio_parents_extracted(self, source_columns):
        origin = feature_origin_for(
            "CREATED_DATE_delta_hours_to_ANNIVERSARY_DATE__C_dow_ratio", source_columns,
        )
        assert origin.derivation == "derived_ratio"
        assert origin.family == "ratio"
        assert origin.parents == ("CREATED_DATE_delta_hours", "ANNIVERSARY_DATE__C_dow")

    def test_interaction_feature_marked(self, source_columns):
        origin = feature_origin_for("NET_PRICE_x_NUMBER_OF_ASINS", source_columns)
        assert origin.derivation == "interaction"
        assert "NET_PRICE" in origin.parents
        assert "NUMBER_OF_ASINS" in origin.parents

    def test_temporal_derived_family(self, source_columns):
        origin = feature_origin_for("subscription__recency_ratio", source_columns)
        assert origin.derivation == "temporal_derived"
        assert origin.source == "subscription"
        assert origin.family == "recency_ratio"

    def test_gold_transform_suffix(self, source_columns):
        origin = feature_origin_for("NET_PRICE_sum_180d_log", source_columns)
        assert origin.derivation == "gold_transform"
        assert origin.source == "subscription"
        assert origin.base_column == "NET_PRICE"


class TestFeatureOriginsFor:
    def test_bulk_resolves_all_features(self, source_columns):
        features = [
            "NET_PRICE_sum_180d",
            "CONTRACT_VALUE_max_all_time",
            "UNKNOWN_FEATURE",
        ]
        result = feature_origins_for(features, source_columns)
        assert set(result.keys()) == set(features)
        assert result["NET_PRICE_sum_180d"].source == "subscription"
        assert result["CONTRACT_VALUE_max_all_time"].source == "contract"
        assert result["UNKNOWN_FEATURE"].source is None

    def test_empty_feature_list_returns_empty(self, source_columns):
        assert feature_origins_for([], source_columns) == {}

    def test_duplicate_features_deduplicated(self, source_columns):
        result = feature_origins_for(
            ["NET_PRICE_sum_180d", "NET_PRICE_sum_180d"], source_columns,
        )
        assert list(result.keys()) == ["NET_PRICE_sum_180d"]

    def test_ratio_parents_in_bulk(self, source_columns):
        result = feature_origins_for(
            ["A_to_B_ratio", "NET_PRICE_sum_180d"], source_columns,
        )
        assert result["A_to_B_ratio"].derivation == "derived_ratio"
        assert result["A_to_B_ratio"].parents == ("A", "B")
