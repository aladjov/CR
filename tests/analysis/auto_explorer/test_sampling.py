import math
from datetime import timedelta

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.project_context import CadenceInterval, IntentConfig
from customer_retention.analysis.auto_explorer.sampling import (
    _compute_group_budget,
    apply_sample_filters,
    apply_temporal_lookback,
    estimate_sampling_accuracy,
    resolve_segment_entity_ids,
    stratified_entity_sample,
    stratified_holdout_split,
)


def _make_intent(lookback_periods=None, cadence=CadenceInterval.WEEKLY, upper_limit=None):
    return IntentConfig(
        lookback_periods=lookback_periods,
        cadence_interval=cadence,
        history_upper_limit=upper_limit,
    )


class TestApplyTemporalLookback:
    def test_no_lookback_returns_unchanged(self):
        df = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=100, freq="D"), "v": range(100)})
        result = apply_temporal_lookback(df, "ts", _make_intent(lookback_periods=None))
        assert len(result) == 100

    def test_filters_to_lookback_window(self):
        dates = pd.date_range("2020-01-01", periods=365 * 5, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(len(dates))})
        intent = _make_intent(lookback_periods=156, cadence=CadenceInterval.WEEKLY)
        result = apply_temporal_lookback(df, "ts", intent)
        lookback_days = 156 * 7
        expected_lower = dates.max() - timedelta(days=lookback_days)
        assert len(result) < len(df)
        assert result["ts"].min() >= expected_lower

    def test_all_data_within_window_returns_full(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(30)})
        intent = _make_intent(lookback_periods=156, cadence=CadenceInterval.WEEKLY)
        result = apply_temporal_lookback(df, "ts", intent)
        assert len(result) == 30

    def test_history_upper_limit_caps_upper_bound(self):
        dates = pd.date_range("2020-01-01", periods=365 * 5, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(len(dates))})
        intent = _make_intent(
            lookback_periods=52, cadence=CadenceInterval.WEEKLY,
            upper_limit="2023-06-30",
        )
        result = apply_temporal_lookback(df, "ts", intent)
        assert result["ts"].max() <= pd.Timestamp("2023-06-30")
        lookback_days = 52 * 7
        expected_lower = pd.Timestamp("2023-06-30") - timedelta(days=lookback_days)
        assert result["ts"].min() >= expected_lower

    def test_all_nat_returns_unchanged(self):
        df = pd.DataFrame({"ts": [pd.NaT, pd.NaT, pd.NaT], "v": [1, 2, 3]})
        intent = _make_intent(lookback_periods=52)
        result = apply_temporal_lookback(df, "ts", intent)
        assert len(result) == 3

    def test_monthly_cadence(self):
        dates = pd.date_range("2018-01-01", periods=365 * 5, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(len(dates))})
        intent = _make_intent(lookback_periods=36, cadence=CadenceInterval.MONTHLY)
        result = apply_temporal_lookback(df, "ts", intent)
        lookback_days = 36 * 30
        expected_lower = dates.max() - timedelta(days=lookback_days)
        assert result["ts"].min() >= expected_lower
        assert len(result) < len(df)

    def test_daily_cadence(self):
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(len(dates))})
        intent = _make_intent(lookback_periods=365, cadence=CadenceInterval.DAILY)
        result = apply_temporal_lookback(df, "ts", intent)
        assert len(result) == 365 + 1  # inclusive boundary

    def test_preserves_non_time_columns(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(100), "cat": ["a"] * 100})
        intent = _make_intent(lookback_periods=4, cadence=CadenceInterval.WEEKLY)
        result = apply_temporal_lookback(df, "ts", intent)
        assert list(result.columns) == ["ts", "v", "cat"]

    def test_boundary_row_included(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])
        df = pd.DataFrame({"ts": dates, "v": [1, 2, 3]})
        intent = _make_intent(lookback_periods=1, cadence=CadenceInterval.WEEKLY)
        result = apply_temporal_lookback(df, "ts", intent)
        expected_lower = pd.Timestamp("2024-01-15") - timedelta(days=7)
        assert result["ts"].min() == expected_lower

    def test_upper_limit_beyond_data_has_no_effect(self):
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(365)})
        intent_with = _make_intent(lookback_periods=26, cadence=CadenceInterval.WEEKLY, upper_limit="2030-01-01")
        intent_without = _make_intent(lookback_periods=26, cadence=CadenceInterval.WEEKLY)
        result_with = apply_temporal_lookback(df, "ts", intent_with)
        result_without = apply_temporal_lookback(df, "ts", intent_without)
        assert len(result_with) == len(result_without)

    def test_string_timestamps_parsed(self):
        df = pd.DataFrame({
            "ts": ["2020-01-01", "2022-06-15", "2024-12-31"],
            "v": [1, 2, 3],
        })
        intent = _make_intent(lookback_periods=52, cadence=CadenceInterval.WEEKLY)
        result = apply_temporal_lookback(df, "ts", intent)
        assert len(result) < 3

    def test_mixed_nat_preserves_valid_rows(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2020-01-01", pd.NaT, "2024-12-01", "2024-12-15"]),
            "v": [1, 2, 3, 4],
        })
        intent = _make_intent(lookback_periods=4, cadence=CadenceInterval.WEEKLY)
        result = apply_temporal_lookback(df, "ts", intent)
        assert 1 not in result["v"].values
        assert 3 in result["v"].values
        assert 4 in result["v"].values

    def test_upper_limit_before_all_data_returns_empty(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(30)})
        intent = _make_intent(lookback_periods=1, cadence=CadenceInterval.WEEKLY, upper_limit="2020-01-01")
        result = apply_temporal_lookback(df, "ts", intent)
        assert len(result) == 0


class TestApplySampleFilters:
    def test_no_filter_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = apply_sample_filters(df, "ds1", {})
        assert len(result) == 3

    def test_matching_filter_applied(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = apply_sample_filters(df, "ds1", {"ds1": "a > 3"})
        assert list(result["a"]) == [4, 5]

    def test_non_matching_dataset_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = apply_sample_filters(df, "ds2", {"ds1": "a > 1"})
        assert len(result) == 3

    def test_string_equality_filter(self):
        df = pd.DataFrame({"region": ["US", "UK", "US", "FR"]})
        result = apply_sample_filters(df, "sales", {"sales": "region == 'US'"})
        assert len(result) == 2

    def test_compound_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
        result = apply_sample_filters(df, "ds", {"ds": "a > 1 and b == 'x'"})
        assert len(result) == 1
        assert result.iloc[0]["a"] == 3

    def test_none_filters_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = apply_sample_filters(df, "ds", None)
        assert len(result) == 2

    def test_in_operator(self):
        df = pd.DataFrame({"status": ["active", "cancelled", "pending", "active"]})
        result = apply_sample_filters(df, "ds", {"ds": "status in ['active', 'pending']"})
        assert len(result) == 3

    def test_numeric_range(self):
        df = pd.DataFrame({"amount": [10, 50, 100, 200, 500]})
        result = apply_sample_filters(df, "ds", {"ds": "amount >= 50 and amount <= 200"})
        assert list(result["amount"]) == [50, 100, 200]

    def test_not_equal_filter(self):
        df = pd.DataFrame({"type": ["A", "B", "C", "A"]})
        result = apply_sample_filters(df, "ds", {"ds": "type != 'B'"})
        assert len(result) == 3

    def test_multiple_datasets_only_matching_applied(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        filters = {"ds1": "x > 2", "ds2": "x < 2"}
        result = apply_sample_filters(df, "ds1", filters)
        assert list(result["x"]) == [3]

    def test_preserves_dataframe_type(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = apply_sample_filters(df, "ds", {"ds": "a > 1"})
        assert isinstance(result, pd.DataFrame)

    def test_filter_all_rows_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = apply_sample_filters(df, "ds", {"ds": "a > 100"})
        assert len(result) == 0

    def test_in_operator_with_tuple_syntax(self):
        df = pd.DataFrame({"region": ["US", "UK", "FR", "DE"]})
        result = apply_sample_filters(df, "ds", {"ds": "region in ('US', 'UK')"})
        assert len(result) == 2


class TestResolveSegmentEntityIds:
    def test_no_filters_returns_none(self):
        frames = {"ds": pd.DataFrame({"eid": [1, 2], "x": [10, 20]})}
        assert resolve_segment_entity_ids(frames, {}, {"ds": "eid"}) is None
        assert resolve_segment_entity_ids(frames, None, {"ds": "eid"}) is None

    def test_single_dataset_all_pass(self):
        df = pd.DataFrame({"eid": [1, 2, 3], "status": ["active", "active", "active"]})
        result = resolve_segment_entity_ids(
            {"customers": df}, {"customers": "status == 'active'"}, {"customers": "eid"},
        )
        assert result == {1, 2, 3}

    def test_single_dataset_some_excluded(self):
        df = pd.DataFrame({"eid": [1, 2, 3], "region": ["US", "UK", "US"]})
        result = resolve_segment_entity_ids(
            {"customers": df}, {"customers": "region == 'US'"}, {"customers": "eid"},
        )
        assert result == {1, 3}

    def test_event_level_all_rows_must_match(self):
        df = pd.DataFrame({
            "eid": [1, 1, 1, 2, 2, 2],
            "amount": [100, 200, 50, 150, 300, 200],
        })
        result = resolve_segment_entity_ids(
            {"txn": df}, {"txn": "amount >= 100"}, {"txn": "eid"},
        )
        assert result == {2}

    def test_event_level_single_non_matching_row_excludes(self):
        df = pd.DataFrame({
            "eid": [1, 1, 2, 2, 3, 3],
            "status": ["ok", "ok", "ok", "cancelled", "ok", "ok"],
        })
        result = resolve_segment_entity_ids(
            {"orders": df}, {"orders": "status == 'ok'"}, {"orders": "eid"},
        )
        assert result == {1, 3}

    def test_multiple_datasets_intersection(self):
        customers = pd.DataFrame({"eid": [1, 2, 3], "region": ["US", "UK", "US"]})
        orders = pd.DataFrame({
            "eid": [1, 1, 2, 3, 3],
            "status": ["ok", "cancelled", "ok", "ok", "ok"],
        })
        result = resolve_segment_entity_ids(
            {"customers": customers, "orders": orders},
            {"customers": "region == 'US'", "orders": "status == 'ok'"},
            {"customers": "eid", "orders": "eid"},
        )
        assert result == {3}

    def test_filter_dataset_not_in_frames_skipped(self):
        df = pd.DataFrame({"eid": [1, 2], "x": [10, 20]})
        result = resolve_segment_entity_ids(
            {"ds1": df}, {"missing_ds": "x > 5"}, {"ds1": "eid"},
        )
        assert result is None

    def test_entity_column_missing_in_dataset_raises(self):
        df = pd.DataFrame({"other_col": [1, 2], "x": [10, 20]})
        with pytest.raises(KeyError):
            resolve_segment_entity_ids(
                {"ds": df}, {"ds": "x > 5"}, {"ds": "eid"},
            )

    def test_filter_removes_all_entities(self):
        df = pd.DataFrame({"eid": [1, 2, 3], "val": [1, 2, 3]})
        result = resolve_segment_entity_ids(
            {"ds": df}, {"ds": "val > 100"}, {"ds": "eid"},
        )
        assert result == set()

    def test_compound_filter(self):
        df = pd.DataFrame({
            "eid": [1, 2, 3, 4],
            "amount": [100, 50, 200, 30],
            "status": ["active", "active", "cancelled", "active"],
        })
        result = resolve_segment_entity_ids(
            {"ds": df},
            {"ds": "amount >= 50 and status == 'active'"},
            {"ds": "eid"},
        )
        assert result == {1, 2}

    def test_in_operator_filter(self):
        df = pd.DataFrame({"eid": [1, 2, 3, 4], "region": ["US", "UK", "FR", "US"]})
        result = resolve_segment_entity_ids(
            {"ds": df},
            {"ds": "region in ['US', 'UK']"},
            {"ds": "eid"},
        )
        assert result == {1, 2, 4}

    def test_entity_with_no_matching_rows_excluded(self):
        df = pd.DataFrame({
            "eid": [1, 1, 2, 2],
            "val": [0, 0, 10, 20],
        })
        result = resolve_segment_entity_ids(
            {"ds": df}, {"ds": "val > 5"}, {"ds": "eid"},
        )
        assert result == {2}

    def test_different_entity_columns_per_dataset(self):
        customers = pd.DataFrame({"customer_id": [1, 2, 3], "tier": ["gold", "silver", "gold"]})
        orders = pd.DataFrame({
            "account_id": [1, 2, 3, 3],
            "amount": [100, 200, 50, 300],
        })
        result = resolve_segment_entity_ids(
            {"customers": customers, "orders": orders},
            {"customers": "tier == 'gold'", "orders": "amount >= 100"},
            {"customers": "customer_id", "orders": "account_id"},
        )
        assert result == {1}

    def test_one_filtered_one_unfiltered_dataset(self):
        customers = pd.DataFrame({"eid": [1, 2, 3], "region": ["US", "UK", "US"]})
        orders = pd.DataFrame({"eid": [1, 2, 3], "val": [10, 20, 30]})
        result = resolve_segment_entity_ids(
            {"customers": customers, "orders": orders},
            {"customers": "region == 'US'"},
            {"customers": "eid", "orders": "eid"},
        )
        assert result == {1, 3}

    def test_inner_join_avoids_fillna_type_issues(self):
        """Entities with no matching rows after filter must be excluded
        without relying on fillna (which can fail on pyspark.pandas)."""
        df = pd.DataFrame({
            "eid": [1, 1, 2, 2, 3, 3],
            "status": ["ok", "ok", "ok", "bad", "bad", "bad"],
        })
        result = resolve_segment_entity_ids(
            {"ds": df}, {"ds": "status == 'ok'"}, {"ds": "eid"},
        )
        assert result == {1}
        assert 2 not in result  # partial match excluded
        assert 3 not in result  # no match excluded

    def test_entity_level_single_row_per_entity(self):
        """Entity-level data: each entity has exactly 1 row. Matching row = passes."""
        df = pd.DataFrame({
            "eid": [10, 20, 30, 40],
            "account_type": ["Enterprise", "Strategic", "SMB", "Enterprise"],
        })
        result = resolve_segment_entity_ids(
            {"accounts": df},
            {"accounts": "account_type in ['Enterprise', 'Strategic']"},
            {"accounts": "eid"},
        )
        assert result == {10, 20, 40}

    def test_not_equal_filter(self):
        df = pd.DataFrame({
            "eid": [1, 2, 3],
            "status": ["Test", "Active", "Active"],
        })
        result = resolve_segment_entity_ids(
            {"ds": df}, {"ds": "status != 'Test'"}, {"ds": "eid"},
        )
        assert result == {2, 3}

    def test_all_entities_pass_filter(self):
        """When all rows pass, all entities are returned."""
        df = pd.DataFrame({
            "eid": [1, 1, 2, 2],
            "val": [10, 20, 30, 40],
        })
        result = resolve_segment_entity_ids(
            {"ds": df}, {"ds": "val > 0"}, {"ds": "eid"},
        )
        assert result == {1, 2}

    def test_string_entity_ids(self):
        """Entity IDs that are strings (common in Salesforce-like systems)."""
        df = pd.DataFrame({
            "eid": ["ACC001", "ACC002", "ACC003"],
            "region": ["US", "UK", "US"],
        })
        result = resolve_segment_entity_ids(
            {"ds": df}, {"ds": "region == 'US'"}, {"ds": "eid"},
        )
        assert result == {"ACC001", "ACC003"}


class TestEstimateSamplingAccuracy:
    def test_correct_ci_formula(self):
        results = estimate_sampling_accuracy(10000, 0.5, [1000], n_cohorts=1)
        expected_ci = 1.96 * math.sqrt(0.5 * 0.5 / 1000)
        assert abs(results[0]["churn_rate_ci"] - expected_ci) < 1e-6

    def test_correlation_error_formula(self):
        results = estimate_sampling_accuracy(10000, 0.5, [1000], n_cohorts=1)
        expected = 1 / math.sqrt(1000)
        assert abs(results[0]["correlation_error"] - expected) < 1e-6

    def test_sample_equals_total(self):
        results = estimate_sampling_accuracy(500, 0.3, [500], n_cohorts=1)
        assert results[0]["pct_of_total"] == 1.0
        assert results[0]["sample_size"] == 500

    def test_target_rate_zero(self):
        results = estimate_sampling_accuracy(1000, 0.0, [100], n_cohorts=1)
        assert results[0]["churn_rate_ci"] == 0.0

    def test_target_rate_one(self):
        results = estimate_sampling_accuracy(1000, 1.0, [100], n_cohorts=1)
        assert results[0]["churn_rate_ci"] == 0.0

    def test_multiple_sample_sizes(self):
        results = estimate_sampling_accuracy(10000, 0.2, [500, 1000, 5000], n_cohorts=4)
        assert len(results) == 3
        assert results[0]["sample_size"] == 500
        assert results[1]["sample_size"] == 1000
        assert results[2]["sample_size"] == 5000

    def test_cohort_ok_threshold(self):
        results = estimate_sampling_accuracy(10000, 0.5, [100], n_cohorts=4)
        assert results[0]["entities_per_cohort"] == 25
        assert results[0]["cohort_ok"] is False

        results = estimate_sampling_accuracy(10000, 0.5, [120], n_cohorts=4)
        assert results[0]["entities_per_cohort"] == 30
        assert results[0]["cohort_ok"] is True

    def test_minority_expected(self):
        results = estimate_sampling_accuracy(10000, 0.1, [1000], n_cohorts=1)
        assert results[0]["minority_expected"] == 100.0

    def test_empty_sample_sizes(self):
        results = estimate_sampling_accuracy(1000, 0.5, [], n_cohorts=1)
        assert results == []

    def test_sample_capped_at_total(self):
        results = estimate_sampling_accuracy(100, 0.5, [200], n_cohorts=1)
        assert results[0]["sample_size"] == 100


class TestStratifiedEntitySample:
    @pytest.fixture
    def entity_df(self):
        return pd.DataFrame({
            "entity_id": list(range(200)),
            "churned": [1] * 40 + [0] * 160,
            "signup_date": pd.date_range("2020-01-01", periods=200, freq="D"),
            "region": (["east"] * 50 + ["west"] * 50 + ["north"] * 50 + ["south"] * 50),
        })

    def test_returns_correct_count(self, entity_df):
        ids = stratified_entity_sample(entity_df, 50, "entity_id", "churned")
        assert len(ids) == 50

    def test_target_proportions_preserved(self, entity_df):
        ids = stratified_entity_sample(entity_df, 100, "entity_id", "churned")
        sampled = entity_df[entity_df["entity_id"].isin(ids)]
        rate = sampled["churned"].mean()
        assert 0.1 <= rate <= 0.3

    def test_reproducibility(self, entity_df):
        ids1 = stratified_entity_sample(entity_df, 50, "entity_id", "churned", random_state=42)
        ids2 = stratified_entity_sample(entity_df, 50, "entity_id", "churned", random_state=42)
        assert ids1 == ids2

    def test_different_seed_different_result(self, entity_df):
        ids1 = stratified_entity_sample(entity_df, 50, "entity_id", "churned", random_state=42)
        ids2 = stratified_entity_sample(entity_df, 50, "entity_id", "churned", random_state=99)
        assert ids1 != ids2

    def test_n_entities_exceeds_total(self, entity_df):
        ids = stratified_entity_sample(entity_df, 999, "entity_id", "churned")
        assert len(ids) == 200

    def test_n_entities_zero(self, entity_df):
        ids = stratified_entity_sample(entity_df, 0, "entity_id", "churned")
        assert ids == []

    def test_rare_class_floor(self):
        df = pd.DataFrame({
            "entity_id": list(range(100)),
            "target": [1] * 3 + [0] * 97,
        })
        ids = stratified_entity_sample(df, 20, "entity_id", "target", min_rare_count=5)
        sampled = df[df["entity_id"].isin(ids)]
        rare_count = (sampled["target"] == 1).sum()
        assert rare_count == 3

    def test_cohort_coverage(self, entity_df):
        ids = stratified_entity_sample(
            entity_df, 80, "entity_id", "churned", time_col="signup_date",
        )
        sampled = entity_df[entity_df["entity_id"].isin(ids)]
        quarters = (
            pd.to_datetime(sampled["signup_date"]).dt.year.astype(str) + "-Q"
            + pd.to_datetime(sampled["signup_date"]).dt.quarter.astype(str)
        )
        assert quarters.nunique() >= 2

    def test_extra_strat_cols_respected(self, entity_df):
        ids = stratified_entity_sample(
            entity_df, 100, "entity_id", "churned",
            extra_strat_cols=["region"],
        )
        sampled = entity_df[entity_df["entity_id"].isin(ids)]
        assert set(sampled["region"].unique()) == {"east", "west", "north", "south"}

    def test_no_target_column(self, entity_df):
        ids = stratified_entity_sample(entity_df, 50, "entity_id")
        assert len(ids) == 50

    def test_duplicate_entities_deduplicated(self):
        df = pd.DataFrame({
            "entity_id": [1, 1, 2, 2, 3, 3],
            "value": [10, 20, 30, 40, 50, 60],
        })
        ids = stratified_entity_sample(df, 2, "entity_id")
        assert len(ids) == 2
        assert len(set(ids)) == 2

    def test_combined_strat_target_time_and_extra(self, entity_df):
        ids = stratified_entity_sample(
            entity_df, 80, "entity_id", "churned",
            time_col="signup_date", extra_strat_cols=["region"],
        )
        assert len(ids) == 80
        sampled = entity_df[entity_df["entity_id"].isin(ids)]
        assert set(sampled["region"].unique()) == {"east", "west", "north", "south"}
        rate = sampled["churned"].mean()
        assert 0.1 <= rate <= 0.3


class TestComputeGroupBudget:
    def test_proportional_allocation(self):
        budget = _compute_group_budget({"a": 60, "b": 40}, 50, 100)
        assert budget["a"] == 30
        assert budget["b"] == 20
        assert sum(budget.values()) == 50

    def test_min_one_guarantee(self):
        budget = _compute_group_budget({"a": 90, "b": 5, "c": 5}, 10, 100)
        assert budget.get("b", 0) >= 1

    def test_last_group_gets_remainder(self):
        budget = _compute_group_budget({"a": 50, "b": 50}, 11, 100)
        assert sum(budget.values()) == 11

    def test_cap_at_group_size(self):
        budget = _compute_group_budget({"a": 3, "b": 100}, 50, 103)
        assert budget["a"] <= 3

    def test_total_matches_requested(self):
        counts = {"x": 100, "y": 200, "z": 50}
        budget = _compute_group_budget(counts, 70, 350)
        assert sum(budget.values()) == 70

    def test_single_group(self):
        budget = _compute_group_budget({"only": 100}, 30, 100)
        assert budget == {"only": 30}

    def test_zero_remaining(self):
        budget = _compute_group_budget({"a": 10, "b": 10}, 0, 20)
        assert sum(budget.values()) == 0

    def test_all_groups_represented(self):
        counts = {"a": 20, "b": 20, "c": 20, "d": 20}
        budget = _compute_group_budget(counts, 20, 80)
        for key in counts:
            assert budget.get(key, 0) >= 1


    def test_no_ndarray_column_assignment(self):
        df = pd.DataFrame({
            "entity_id": list(range(50)),
            "target": [1] * 10 + [0] * 40,
            "ts": pd.date_range("2022-01-01", periods=50, freq="D"),
        })
        ids = stratified_entity_sample(df, 20, "entity_id", "target", time_col="ts")
        assert len(ids) == 20
        assert all(isinstance(i, int) for i in ids)


class TestStratifiedHoldoutSplit:
    def _make_df(self, n=100):
        return pd.DataFrame({
            "entity_id": list(range(n)),
            "target": [1] * (n // 5) + [0] * (n - n // 5),
            "ts": pd.date_range("2022-01-01", periods=n, freq="D"),
        })

    def test_basic_split_sizes(self):
        df = self._make_df(100)
        all_ids = list(range(100))
        train, holdout = stratified_holdout_split(
            df, all_ids, holdout_fraction=0.2, entity_col="entity_id",
        )
        assert len(holdout) == 20
        assert len(train) == 80
        assert set(train) | set(holdout) == set(all_ids)
        assert set(train) & set(holdout) == set()

    def test_no_overlap(self):
        df = self._make_df(200)
        all_ids = list(range(200))
        train, holdout = stratified_holdout_split(
            df, all_ids, holdout_fraction=0.1, entity_col="entity_id",
            target_col="target",
        )
        assert set(train).isdisjoint(set(holdout))
        assert len(train) + len(holdout) == 200

    def test_zero_fraction_returns_all_train(self):
        df = self._make_df(50)
        all_ids = list(range(50))
        train, holdout = stratified_holdout_split(
            df, all_ids, holdout_fraction=0.0, entity_col="entity_id",
        )
        assert len(train) == 50
        assert len(holdout) == 0

    def test_full_fraction_returns_all_holdout(self):
        df = self._make_df(50)
        all_ids = list(range(50))
        train, holdout = stratified_holdout_split(
            df, all_ids, holdout_fraction=1.0, entity_col="entity_id",
        )
        assert len(train) == 0
        assert len(holdout) == 50

    def test_preserves_target_distribution(self):
        n = 500
        df = pd.DataFrame({
            "entity_id": list(range(n)),
            "target": [1] * 100 + [0] * 400,
        })
        all_ids = list(range(n))
        train, holdout = stratified_holdout_split(
            df, all_ids, holdout_fraction=0.2, entity_col="entity_id",
            target_col="target",
        )
        train_set = set(train)
        holdout_set = set(holdout)
        train_rate = sum(1 for i in train if df.loc[df["entity_id"] == i, "target"].iloc[0] == 1) / len(train)
        holdout_rate = sum(1 for i in holdout if df.loc[df["entity_id"] == i, "target"].iloc[0] == 1) / len(holdout)
        # Both should be close to 0.2 (100/500)
        assert abs(train_rate - 0.2) < 0.05
        assert abs(holdout_rate - 0.2) < 0.05

    def test_with_time_and_extra_cols(self):
        n = 200
        df = pd.DataFrame({
            "entity_id": list(range(n)),
            "target": [1] * 40 + [0] * 160,
            "ts": pd.date_range("2022-01-01", periods=n, freq="D"),
            "region": (["east", "west"] * 100)[:n],
        })
        all_ids = list(range(n))
        train, holdout = stratified_holdout_split(
            df, all_ids, holdout_fraction=0.15, entity_col="entity_id",
            target_col="target", time_col="ts", extra_strat_cols=["region"],
        )
        assert len(train) + len(holdout) == n
        assert set(train).isdisjoint(set(holdout))

    def test_small_dataset(self):
        df = pd.DataFrame({"entity_id": [1, 2, 3], "target": [1, 0, 0]})
        train, holdout = stratified_holdout_split(
            df, [1, 2, 3], holdout_fraction=0.3, entity_col="entity_id",
            target_col="target",
        )
        assert len(train) + len(holdout) == 3
        assert len(holdout) >= 1

    def test_reproducible(self):
        df = self._make_df(100)
        all_ids = list(range(100))
        t1, h1 = stratified_holdout_split(
            df, all_ids, holdout_fraction=0.2, entity_col="entity_id",
            target_col="target", random_state=42,
        )
        t2, h2 = stratified_holdout_split(
            df, all_ids, holdout_fraction=0.2, entity_col="entity_id",
            target_col="target", random_state=42,
        )
        assert t1 == t2
        assert h1 == h2
