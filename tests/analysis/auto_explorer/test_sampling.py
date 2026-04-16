import math
from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.project_context import CadenceInterval, IntentConfig
from customer_retention.analysis.auto_explorer.sampling import (
    PopulationSummary,
    SegmentEntitySelection,
    _build_strat_key_column,
    _compute_group_budget,
    apply_sample_filters,
    apply_temporal_lookback,
    candidate_sample_sizes,
    estimate_sampling_accuracy,
    prepare_sample_frame,
    render_population_markdown,
    render_sample_result_markdown,
    render_segment_filter_markdown,
    resolve_segment_entity_ids,
    save_sample_ids,
    stratified_entity_sample,
    stratified_holdout_split,
    summarize_population,
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

    def test_raises_when_large_dataset_collapses_to_few_rows(self):
        old_dates = pd.date_range("2005-01-01", periods=10_000, freq="h")
        recent_dates = pd.to_datetime(["2024-12-20", "2024-12-21"])
        df = pd.DataFrame({
            "ts": list(old_dates) + list(recent_dates),
            "v": range(10_000 + 2),
        })
        intent = _make_intent(lookback_periods=36, cadence=CadenceInterval.MONTHLY)
        with pytest.raises(ValueError, match="retained only 2"):
            apply_temporal_lookback(df, "ts", intent)

    def test_error_message_suggests_interval_start_time_fix(self):
        old_dates = pd.date_range("2005-01-01", periods=10_000, freq="D")
        recent_dates = pd.to_datetime(["2036-07-01"])
        df = pd.DataFrame({
            "ts": list(old_dates) + list(recent_dates),
            "v": range(10_001),
        })
        intent = _make_intent(lookback_periods=4, cadence=CadenceInterval.WEEKLY)
        with pytest.raises(ValueError, match="INTERVAL_START_TIME"):
            apply_temporal_lookback(df, "ts", intent)

    def test_error_message_mentions_lookback_periods_remedy(self):
        old_dates = pd.date_range("2005-01-01", periods=10_000, freq="D")
        recent_dates = pd.to_datetime(["2036-07-01"])
        df = pd.DataFrame({
            "ts": list(old_dates) + list(recent_dates),
            "v": range(10_001),
        })
        intent = _make_intent(lookback_periods=4, cadence=CadenceInterval.WEEKLY)
        with pytest.raises(ValueError, match="lookback_periods"):
            apply_temporal_lookback(df, "ts", intent)

    def test_error_message_mentions_null_time_column(self):
        nulls = [pd.NaT] * 10_000
        recent_dates = list(pd.to_datetime(["2024-12-20", "2024-12-21"]))
        df = pd.DataFrame({"ts": nulls + recent_dates, "v": range(10_000 + 2)})
        intent = _make_intent(lookback_periods=4, cadence=CadenceInterval.WEEKLY)
        with pytest.raises(ValueError, match="populated"):
            apply_temporal_lookback(df, "ts", intent)

    def test_small_input_skips_retention_guard(self):
        dates = pd.date_range("2005-01-01", periods=100, freq="D")
        df = pd.DataFrame({"ts": dates, "v": range(100)})
        intent = _make_intent(lookback_periods=4, cadence=CadenceInterval.WEEKLY)
        result = apply_temporal_lookback(df, "ts", intent)
        assert len(result) == 29

    def test_passes_when_retention_meets_threshold(self):
        recent_dates = pd.date_range("2024-10-01", periods=120, freq="D")
        old_dates = pd.date_range("2010-01-01", periods=1400, freq="D")
        df = pd.DataFrame({
            "ts": list(recent_dates) + list(old_dates),
            "v": range(120 + 1400),
        })
        intent = _make_intent(lookback_periods=26, cadence=CadenceInterval.WEEKLY)
        result = apply_temporal_lookback(df, "ts", intent)
        assert len(result) >= 100

    def test_error_message_includes_window_bounds(self):
        old_dates = pd.date_range("2005-01-01", periods=10_000, freq="D")
        recent_dates = pd.to_datetime(["2036-07-01"])
        df = pd.DataFrame({
            "ts": list(old_dates) + list(recent_dates),
            "v": range(10_001),
        })
        intent = _make_intent(lookback_periods=4, cadence=CadenceInterval.WEEKLY)
        with pytest.raises(ValueError, match=r"Window: \["):
            apply_temporal_lookback(df, "ts", intent)


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


class TestSegmentEntitySelection:
    def test_pandas_set_supports_len_and_contains(self):
        sel = SegmentEntitySelection.from_set({1, 2, 3})
        assert len(sel) == 3
        assert 1 in sel
        assert 99 not in sel

    def test_pandas_set_equals_plain_set(self):
        sel = SegmentEntitySelection.from_set({"a", "b"})
        assert sel == {"a", "b"}
        assert sel != {"a"}

    def test_pandas_intersection(self):
        a = SegmentEntitySelection.from_set({1, 2, 3})
        b = SegmentEntitySelection.from_set({2, 3, 4})
        result = a.intersect(b)
        assert result == {2, 3}

    def test_filter_frame_pandas_semantics(self):
        sel = SegmentEntitySelection.from_set({1, 3})
        df = pd.DataFrame({"eid": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
        filtered = sel.filter_frame(df, "eid")
        assert set(filtered["eid"]) == {1, 3}

    def test_distributed_len_uses_count_without_collect(self):
        # For Spark-backed selections the row count comes from a single .count()
        # action — never a .collect() of entity IDs (would OOM on large tables).
        mock_sdf = MagicMock()
        mock_sdf.withColumnRenamed.return_value = mock_sdf
        mock_sdf.select.return_value = mock_sdf
        mock_sdf.cache.return_value = mock_sdf
        mock_sdf.count.return_value = 1_250_000
        sel = SegmentEntitySelection.from_spark_df(mock_sdf, "eid")
        assert len(sel) == 1_250_000
        mock_sdf.collect.assert_not_called()

    def test_from_spark_df_caches_result_to_avoid_rescans(self):
        # Without cache, `.count()` (len) + `.join()` (filter_frame) each
        # re-execute the filter subquery from scratch. Caching pins the
        # passing-entity IDs so downstream actions reuse the same block.
        mock_sdf = MagicMock()
        renamed = MagicMock()
        selected = MagicMock()
        cached = MagicMock(name="cached")
        mock_sdf.withColumnRenamed.return_value = renamed
        renamed.select.return_value = selected
        selected.cache.return_value = cached

        sel = SegmentEntitySelection.from_spark_df(mock_sdf, "eid")
        selected.cache.assert_called_once()
        assert sel._data is cached

    def test_intersect_caches_join_result(self):
        left_sdf = MagicMock(name="left_sdf")
        left_sdf.withColumnRenamed.return_value = left_sdf
        left_sdf.select.return_value = left_sdf
        left_sdf.cache.return_value = left_sdf
        right_sdf = MagicMock(name="right_sdf")
        right_sdf.withColumnRenamed.return_value = right_sdf
        right_sdf.select.return_value = right_sdf
        right_sdf.cache.return_value = right_sdf
        joined = MagicMock(name="joined")
        joined.cache.return_value = MagicMock(name="joined_cached")
        left_sdf.join.return_value = joined

        a = SegmentEntitySelection.from_spark_df(left_sdf, "eid")
        b = SegmentEntitySelection.from_spark_df(right_sdf, "eid")
        a.intersect(b)
        joined.cache.assert_called_once()

    def test_distributed_len_cached(self):
        mock_sdf = MagicMock()
        mock_sdf.withColumnRenamed.return_value = mock_sdf
        mock_sdf.select.return_value = mock_sdf
        mock_sdf.cache.return_value = mock_sdf
        mock_sdf.count.return_value = 42
        sel = SegmentEntitySelection.from_spark_df(mock_sdf, "eid")
        assert len(sel) == 42
        assert len(sel) == 42
        # One .count() job only, regardless of how many times len() is called.
        assert mock_sdf.count.call_count == 1

    def test_distributed_iteration_raises(self):
        mock_sdf = MagicMock()
        mock_sdf.withColumnRenamed.return_value = mock_sdf
        mock_sdf.select.return_value = mock_sdf
        mock_sdf.cache.return_value = mock_sdf
        sel = SegmentEntitySelection.from_spark_df(mock_sdf, "eid")
        with pytest.raises(TypeError, match="distributed"):
            list(sel)
        with pytest.raises(TypeError, match="distributed"):
            _ = 5 in sel

    def test_distributed_intersection_uses_inner_join(self):
        # Intersection between two distributed selections must stay in Spark —
        # a join, not collecting both sides to the driver.
        left_sdf = MagicMock(name="left_sdf")
        left_sdf.withColumnRenamed.return_value = left_sdf
        left_sdf.select.return_value = left_sdf
        left_sdf.cache.return_value = left_sdf
        right_sdf = MagicMock(name="right_sdf")
        right_sdf.withColumnRenamed.return_value = right_sdf
        right_sdf.select.return_value = right_sdf
        right_sdf.cache.return_value = right_sdf
        joined = MagicMock(name="joined")
        joined.cache.return_value = joined
        left_sdf.join.return_value = joined

        a = SegmentEntitySelection.from_spark_df(left_sdf, "eid")
        b = SegmentEntitySelection.from_spark_df(right_sdf, "eid")
        result = a.intersect(b)

        left_sdf.join.assert_called_once()
        _, kwargs = left_sdf.join.call_args
        assert kwargs.get("how") == "inner"
        assert result.is_distributed is True
        left_sdf.collect.assert_not_called()
        right_sdf.collect.assert_not_called()

    def test_distributed_filter_frame_uses_left_semi_join(self, monkeypatch):
        # Applying the filter back to a frame must use a left-semi join —
        # never materialize the entity ID list.
        mock_sdf = MagicMock()
        mock_sdf.withColumnRenamed.return_value = mock_sdf
        mock_sdf.select.return_value = mock_sdf
        mock_sdf.cache.return_value = mock_sdf
        sel = SegmentEntitySelection.from_spark_df(mock_sdf, "eid")

        captured: dict = {}

        def fake_as_spark_df(df):
            captured["input"] = df
            return MagicMock(name="target_sdf", **{
                "join.return_value": MagicMock(name="joined"),
            })

        def fake_as_pandas_api(result):
            captured["result"] = result
            return "wrapped"

        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.sampling.as_spark_df",
            fake_as_spark_df,
        )
        monkeypatch.setattr(
            "customer_retention.core.compat.spark_backend._as_pandas_api",
            fake_as_pandas_api,
        )

        out = sel.filter_frame("any-psdf", "eid")
        assert out == "wrapped"
        assert captured["input"] == "any-psdf"
        mock_sdf.collect.assert_not_called()


class TestResolveSegmentEntityIdsRegistersViews:
    def test_spark_frames_exposed_as_temp_views_by_dataset_name(self, monkeypatch):
        # SQL subqueries in filter expressions (e.g. `... in (select ACCOUNT_ID
        # from contract where ...)`) resolve through the session catalog.
        # Every Spark-backed frame must be registered as a temp view keyed by
        # its dataset name so those references resolve.
        account_sdf = MagicMock(name="account_sdf")
        contract_sdf = MagicMock(name="contract_sdf")

        class FakeSparkFrame:
            def __init__(self, sdf):
                self._sdf = sdf

            def to_spark(self):
                return self._sdf

        account_frame = FakeSparkFrame(account_sdf)
        contract_frame = FakeSparkFrame(contract_sdf)

        # Stub out the heavy Spark execution path — we only want to prove that
        # temp view registration happened before any filter evaluation.
        monkeypatch.setattr(
            "customer_retention.analysis.auto_explorer.sampling._spark_passing_entities",
            lambda df, expr, col: MagicMock(name="passing_ids_sdf"),
        )

        resolve_segment_entity_ids(
            {"account": account_frame, "contract": contract_frame},
            {"account": "ACCOUNT_ID in (select ACCOUNT_ID from contract)"},
            {"account": "ACCOUNT_ID", "contract": "ACCOUNT_ID"},
        )

        account_sdf.createOrReplaceTempView.assert_called_once_with("account")
        contract_sdf.createOrReplaceTempView.assert_called_once_with("contract")

    def test_pandas_frames_do_not_touch_spark_catalog(self):
        df = pd.DataFrame({"eid": [1, 2], "x": [10, 20]})
        # Pure pandas path — no Spark session exists. Must not raise.
        result = resolve_segment_entity_ids(
            {"ds": df}, {"ds": "x > 5"}, {"ds": "eid"},
        )
        assert result == {1, 2}


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


class TestBuildStratKeyColumn:
    def _df(self, n=20):
        return pd.DataFrame({
            "entity_id": range(n),
            "target": [1, 0] * (n // 2),
            "ts": pd.date_range("2023-01-01", periods=n, freq="45D"),
            "region": (["a", "b", "c", "d"] * ((n // 4) + 1))[:n],
            "amount": list(range(10, n + 10)),
        })

    def test_returns_none_when_no_dimensions(self):
        key = _build_strat_key_column(self._df(), None, None, None)
        assert key is None

    def test_target_only(self):
        df = self._df(4)
        key = _build_strat_key_column(df, "target", None, None)
        assert list(key) == ["1", "0", "1", "0"]

    def test_target_and_time_produces_pipe_joined_cohort(self):
        df = self._df(8)
        key = _build_strat_key_column(df, "target", "ts", None)
        assert all("|" in v for v in key)
        assert all(v.split("|")[1].startswith("20") and "-Q" in v.split("|")[1] for v in key)

    def test_extra_categorical_is_stringified(self):
        df = self._df(8)
        key = _build_strat_key_column(df, None, None, ["region"])
        assert sorted(set(key)) == ["a", "b", "c", "d"]

    def test_extra_numeric_is_qcut_binned(self):
        df = self._df(20)
        key = _build_strat_key_column(df, None, None, ["amount"])
        assert len(set(key)) <= 4

    def test_missing_extra_column_is_ignored(self):
        df = self._df(4)
        key = _build_strat_key_column(df, "target", None, ["nonexistent"])
        assert list(key) == ["1", "0", "1", "0"]

    def test_target_missing_in_df_is_ignored(self):
        df = self._df(4).drop(columns=["target"])
        key = _build_strat_key_column(df, "target", None, None)
        assert key is None

    def test_time_column_with_nat_produces_deterministic_key_per_row(self):
        # Original behavior: NaT year/quarter stringify to "nan" — fillna never
        # triggers. The refactor preserves this; the key is still stable.
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01", pd.NaT, "2024-04-01"]),
        })
        key = _build_strat_key_column(df, None, "ts", None)
        vals = list(key)
        assert len(vals) == 3
        assert vals[0] != vals[2]
        assert "Q" in vals[0] and "Q" in vals[2]


class TestSummarizePopulation:
    def test_none_target_df_returns_zero(self):
        s = summarize_population(None, "entity_id", "target", "ts")
        assert s == PopulationSummary(0, 0.5, 1, None)

    def test_missing_entity_col_returns_zero(self):
        df = pd.DataFrame({"x": [1, 2]})
        s = summarize_population(df, "entity_id", "target", "ts")
        assert s.total_entities == 0

    def test_duplicates_deduped(self):
        df = pd.DataFrame({
            "entity_id": [1, 1, 2, 2, 3],
            "target": [1, 1, 0, 0, 1],
        })
        s = summarize_population(df, "entity_id", "target", None)
        assert s.total_entities == 3

    def test_target_rate_computed_on_entities(self):
        df = pd.DataFrame({
            "entity_id": [1, 2, 3, 4],
            "target": [1, 1, 0, 0],
        })
        s = summarize_population(df, "entity_id", "target", None)
        assert s.target_rate == 0.5

    def test_no_target_column_falls_back_to_default(self):
        df = pd.DataFrame({"entity_id": [1, 2]})
        s = summarize_population(df, "entity_id", "target", None)
        assert s.target_rate == 0.5

    def test_cohort_count_from_quarters(self):
        df = pd.DataFrame({
            "entity_id": range(8),
            "ts": pd.to_datetime([
                "2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01",
                "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01",
            ]),
        })
        s = summarize_population(df, "entity_id", None, "ts")
        assert s.n_cohorts == 8
        assert s.time_column == "ts"

    def test_time_column_missing_returns_one_cohort(self):
        df = pd.DataFrame({"entity_id": [1, 2]})
        s = summarize_population(df, "entity_id", None, "ts")
        assert s.n_cohorts == 1
        assert s.time_column is None


class TestCandidateSampleSizes:
    def test_returns_sorted_candidates_under_total(self):
        assert candidate_sample_sizes(3000) == [500, 1000, 2000, 3000]

    def test_includes_total_when_not_in_defaults(self):
        sizes = candidate_sample_sizes(7500)
        assert 7500 in sizes
        assert max(sizes) == 7500

    def test_total_equal_to_default_not_duplicated(self):
        sizes = candidate_sample_sizes(1000)
        assert sizes.count(1000) == 1

    def test_total_zero_returns_empty(self):
        assert candidate_sample_sizes(0) == []

    def test_very_large_total_includes_all_defaults(self):
        sizes = candidate_sample_sizes(1_000_000)
        assert sizes == [500, 1000, 2000, 5000, 10000, 1_000_000]


class TestPrepareSampleFrame:
    def _df(self):
        return pd.DataFrame({
            "entity_id": [1, 2, 3],
            "target": [1, 0, 1],
            "ts": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-09-01"]),
            "region": ["a", "b", "a"],
            "other": [10, 20, 30],
        })

    def test_projects_to_entity_target_time_and_extras(self):
        out = prepare_sample_frame(self._df(), "entity_id", "target", "ts", ["region"], None)
        assert set(out.columns) == {"entity_id", "target", "ts", "region"}

    def test_omits_missing_extras(self):
        out = prepare_sample_frame(self._df(), "entity_id", "target", None, ["missing"], None)
        assert set(out.columns) == {"entity_id", "target"}

    def test_without_target_or_time(self):
        out = prepare_sample_frame(self._df(), "entity_id", None, None, None, None)
        assert list(out.columns) == ["entity_id"]

    def test_segment_filter_applied(self):
        sel = SegmentEntitySelection.from_set({2})
        out = prepare_sample_frame(self._df(), "entity_id", "target", None, None, sel)
        assert list(out["entity_id"]) == [2]

    def test_no_duplicate_columns_when_extra_overlaps_target(self):
        out = prepare_sample_frame(self._df(), "entity_id", "target", None, ["target"], None)
        assert list(out.columns).count("target") == 1


class TestRenderMarkdowns:
    def _estimate(self, **overrides):
        base = {
            "sample_size": 1000,
            "pct_of_total": 0.1,
            "churn_rate_ci": 0.015,
            "correlation_error": 0.032,
            "minority_expected": 80,
            "cohort_ok": True,
        }
        base.update(overrides)
        return base

    def test_population_markdown_contains_counts_and_table_header(self):
        summary = PopulationSummary(total_entities=10000, target_rate=0.08, n_cohorts=12, time_column="ts")
        text = render_population_markdown(summary, 10000, False, 10000, [self._estimate()])
        assert "**Population:** 10,000 entities" in text
        assert "target rate 8.0%" in text
        assert "12 cohorts" in text
        assert "Sample Size" in text
        assert "| 1,000 |" in text

    def test_population_markdown_shows_filtered_suffix(self):
        summary = PopulationSummary(total_entities=10000, target_rate=0.1, n_cohorts=1, time_column=None)
        text = render_population_markdown(summary, 2000, True, 10000, [self._estimate()])
        assert "2,000 entities (filtered from 10,000)" in text

    def test_cohort_ok_renders_yes_or_no(self):
        summary = PopulationSummary(total_entities=5, target_rate=0.5, n_cohorts=1, time_column=None)
        text = render_population_markdown(summary, 5, False, 5, [self._estimate(cohort_ok=False)])
        assert " no |" in text

    def test_sample_result_markdown_without_extras(self):
        text = render_sample_result_markdown(5000, 10000, 4000, 1000, 0.2, False, None)
        assert "**Sampled:** 5,000 / 10,000 entities (50%)" in text
        assert "stratified by target" in text
        assert "**Train:** 4,000" in text
        assert "**Holdout:** 1,000 entities (20%)" in text

    def test_sample_result_markdown_with_time_and_extras(self):
        text = render_sample_result_markdown(500, 1000, 400, 100, 0.2, True, ["region", "segment"])
        assert "stratified by target + cohort + region, segment" in text

    def test_sample_result_markdown_handles_zero_total(self):
        text = render_sample_result_markdown(0, 0, 0, 0, 0.0, False, None)
        assert "(0%)" in text

    def test_segment_filter_markdown(self):
        text = render_segment_filter_markdown({"account": "status == 'Active'", "orders": "amount > 0"})
        assert "**Segment filters:**" in text
        assert "**account**" in text
        assert "`status == 'Active'`" in text


class TestSaveSampleIds:
    def _namespace(self, tmp_path):
        class FakeNamespace:
            sample_entity_ids_path = tmp_path / "sub" / "sample_entity_ids.json"
            holdout_entity_ids_path = tmp_path / "sub" / "holdout_entity_ids.json"
        return FakeNamespace()

    def test_writes_train_and_holdout(self, tmp_path):
        import json as _json
        ns = self._namespace(tmp_path)
        save_sample_ids(ns, [1, 2, 3], [4, 5])
        assert _json.loads(ns.sample_entity_ids_path.read_text()) == [1, 2, 3]
        assert _json.loads(ns.holdout_entity_ids_path.read_text()) == [4, 5]

    def test_skips_holdout_when_empty(self, tmp_path):
        ns = self._namespace(tmp_path)
        save_sample_ids(ns, [1, 2], [])
        assert ns.sample_entity_ids_path.exists()
        assert not ns.holdout_entity_ids_path.exists()

    def test_creates_parent_dir(self, tmp_path):
        ns = self._namespace(tmp_path)
        assert not ns.sample_entity_ids_path.parent.exists()
        save_sample_ids(ns, [1], [])
        assert ns.sample_entity_ids_path.parent.exists()

    def test_serializes_non_json_native_ids_via_default_str(self, tmp_path):
        ns = self._namespace(tmp_path)
        import numpy as np
        save_sample_ids(ns, [np.int64(7), np.int64(9)], [])
        assert "7" in ns.sample_entity_ids_path.read_text()


class TestAssertSampledIdsHaveLabels:
    """Fail-fast validation: every sampled ID must resolve to a non-null target."""

    def _entity_df(self):
        return pd.DataFrame({
            "account_id": ["A", "B", "C", "D"],
            "churned":    [1, 0, 1, 0],
        })

    def test_empty_ids_is_noop(self):
        from customer_retention.analysis.auto_explorer.sampling import (
            _assert_sampled_ids_have_labels,
        )
        _assert_sampled_ids_have_labels(
            entity_df=self._entity_df(), entity_col="account_id",
            target_col="churned", ids=[], pool_label="training",
        )

    def test_all_ids_resolvable_is_noop(self):
        from customer_retention.analysis.auto_explorer.sampling import (
            _assert_sampled_ids_have_labels,
        )
        _assert_sampled_ids_have_labels(
            entity_df=self._entity_df(), entity_col="account_id",
            target_col="churned", ids=["A", "B"], pool_label="training",
        )

    def test_raises_when_id_missing_from_entity_df(self):
        from customer_retention.analysis.auto_explorer.sampling import (
            _assert_sampled_ids_have_labels,
        )
        with pytest.raises(ValueError, match=r"training.*NULL.*churned"):
            _assert_sampled_ids_have_labels(
                entity_df=self._entity_df(), entity_col="account_id",
                target_col="churned", ids=["A", "ORPHAN"], pool_label="training",
            )

    def test_raises_when_target_is_null(self):
        from customer_retention.analysis.auto_explorer.sampling import (
            _assert_sampled_ids_have_labels,
        )
        df = pd.DataFrame({
            "account_id": ["A", "B", "C"],
            "churned":    [1, None, 0],
        })
        with pytest.raises(ValueError, match=r"null target"):
            _assert_sampled_ids_have_labels(
                entity_df=df, entity_col="account_id", target_col="churned",
                ids=["A", "B"], pool_label="training",
            )

    def test_error_message_includes_offending_ids(self):
        from customer_retention.analysis.auto_explorer.sampling import (
            _assert_sampled_ids_have_labels,
        )
        with pytest.raises(ValueError) as excinfo:
            _assert_sampled_ids_have_labels(
                entity_df=self._entity_df(), entity_col="account_id",
                target_col="churned", ids=["A", "X", "Y", "Z"], pool_label="holdout",
            )
        msg = str(excinfo.value)
        assert "holdout" in msg
        assert "X" in msg
        assert "3 absent" in msg or "3 missing" in msg

    def test_pool_label_surfaces_in_error(self):
        from customer_retention.analysis.auto_explorer.sampling import (
            _assert_sampled_ids_have_labels,
        )
        with pytest.raises(ValueError, match=r"holdout"):
            _assert_sampled_ids_have_labels(
                entity_df=self._entity_df(), entity_col="account_id",
                target_col="churned", ids=["ORPHAN"], pool_label="holdout",
            )

    def test_does_not_validate_when_target_col_absent(self):
        from customer_retention.analysis.auto_explorer.sampling import (
            _assert_sampled_ids_have_labels,
        )
        df = pd.DataFrame({"account_id": ["A"]})
        _assert_sampled_ids_have_labels(
            entity_df=df, entity_col="account_id",
            target_col="churned", ids=["A"], pool_label="training",
        )

    def test_reports_null_and_missing_counts_independently(self):
        from customer_retention.analysis.auto_explorer.sampling import (
            _assert_sampled_ids_have_labels,
        )
        df = pd.DataFrame({
            "account_id": ["A", "B", "C"],
            "churned":    [1, None, 0],
        })
        with pytest.raises(ValueError) as excinfo:
            _assert_sampled_ids_have_labels(
                entity_df=df, entity_col="account_id", target_col="churned",
                ids=["A", "B", "ORPHAN"], pool_label="training",
            )
        msg = str(excinfo.value)
        assert "1 absent" in msg
        assert "1 with null target" in msg


class TestSaveSampleIdsValidation:
    def _namespace(self, tmp_path):
        class FakeNamespace:
            sample_entity_ids_path = tmp_path / "sub" / "sample_entity_ids.json"
            holdout_entity_ids_path = tmp_path / "sub" / "holdout_entity_ids.json"
        return FakeNamespace()

    def _entity_df(self):
        return pd.DataFrame({
            "account_id": ["A", "B", "C", "D"],
            "churned":    [1, 0, 1, 0],
        })

    def test_without_validation_args_backward_compat(self, tmp_path):
        ns = self._namespace(tmp_path)
        save_sample_ids(ns, ["ORPHAN"], [])
        assert ns.sample_entity_ids_path.exists()

    def test_with_validation_passes_when_ids_valid(self, tmp_path):
        ns = self._namespace(tmp_path)
        save_sample_ids(
            ns, ["A", "B"], ["C"],
            entity_df=self._entity_df(), entity_col="account_id", target_col="churned",
        )
        assert ns.sample_entity_ids_path.exists()
        assert ns.holdout_entity_ids_path.exists()

    def test_with_validation_raises_on_orphan_train_id(self, tmp_path):
        ns = self._namespace(tmp_path)
        with pytest.raises(ValueError, match=r"training"):
            save_sample_ids(
                ns, ["A", "ORPHAN"], ["C"],
                entity_df=self._entity_df(), entity_col="account_id", target_col="churned",
            )
        assert not ns.sample_entity_ids_path.exists()
        assert not ns.holdout_entity_ids_path.exists()

    def test_with_validation_raises_on_orphan_holdout_id(self, tmp_path):
        ns = self._namespace(tmp_path)
        with pytest.raises(ValueError, match=r"holdout"):
            save_sample_ids(
                ns, ["A", "B"], ["ORPHAN"],
                entity_df=self._entity_df(), entity_col="account_id", target_col="churned",
            )
        assert not ns.sample_entity_ids_path.exists()

    def test_validation_raises_before_any_write(self, tmp_path):
        """No partial writes: train file should not be created when holdout fails."""
        ns = self._namespace(tmp_path)
        with pytest.raises(ValueError):
            save_sample_ids(
                ns, ["A"], ["ORPHAN"],
                entity_df=self._entity_df(), entity_col="account_id", target_col="churned",
            )
        assert not ns.sample_entity_ids_path.exists()


class TestSegmentFilterStats:
    def test_pass_rate_calculation(self):
        from customer_retention.analysis.auto_explorer.sampling import SegmentFilterStats
        stats = SegmentFilterStats(
            dataset="account", filter_expr="has_contract=1",
            input_entities=402_384, output_entities=2_893,
        )
        assert abs(stats.pass_rate - 0.00719) < 1e-4

    def test_pass_rate_zero_input(self):
        from customer_retention.analysis.auto_explorer.sampling import SegmentFilterStats
        stats = SegmentFilterStats(
            dataset="account", filter_expr="x=1",
            input_entities=0, output_entities=0,
        )
        assert stats.pass_rate == 0.0

    def test_is_suspicious_below_threshold(self):
        from customer_retention.analysis.auto_explorer.sampling import SegmentFilterStats
        stats = SegmentFilterStats("account", "x=1", input_entities=100, output_entities=1)
        assert stats.is_suspicious(threshold=0.05)

    def test_is_not_suspicious_above_threshold(self):
        from customer_retention.analysis.auto_explorer.sampling import SegmentFilterStats
        stats = SegmentFilterStats("account", "x=1", input_entities=100, output_entities=50)
        assert not stats.is_suspicious(threshold=0.05)


class TestResolveSegmentEntityIdsDiagnostics:
    def _frames(self):
        return {
            "account": pd.DataFrame({
                "account_id": ["A", "B", "C", "D", "E"],
                "has_contract": [1, 1, 0, 0, 0],
            }),
        }

    def test_diagnostics_captured_when_param_provided(self):
        diagnostics: list = []
        resolve_segment_entity_ids(
            frames=self._frames(),
            filters={"account": "has_contract == 1"},
            entity_columns={"account": "account_id"},
            diagnostics=diagnostics,
        )
        assert len(diagnostics) == 1
        assert diagnostics[0].dataset == "account"
        assert diagnostics[0].input_entities == 5
        assert diagnostics[0].output_entities == 2
        assert abs(diagnostics[0].pass_rate - 0.4) < 1e-6

    def test_no_diagnostics_when_param_none(self):
        # backward compat: calling without diagnostics= works unchanged
        result = resolve_segment_entity_ids(
            frames=self._frames(),
            filters={"account": "has_contract == 1"},
            entity_columns={"account": "account_id"},
        )
        assert result is not None
        assert len(result) == 2

    def test_diagnostics_per_filter_dataset(self):
        frames = {
            "account": pd.DataFrame({
                "account_id": ["A", "B", "C"],
                "has_contract": [1, 0, 1],
            }),
            "contract": pd.DataFrame({
                "account_id": ["A", "A", "B"],
                "status": ["active", "active", "cancelled"],
            }),
        }
        diagnostics: list = []
        resolve_segment_entity_ids(
            frames=frames,
            filters={"account": "has_contract == 1", "contract": "status == 'active'"},
            entity_columns={"account": "account_id", "contract": "account_id"},
            diagnostics=diagnostics,
        )
        assert len(diagnostics) == 2
        datasets = {d.dataset for d in diagnostics}
        assert datasets == {"account", "contract"}


class TestRenderSegmentFilterMarkdown:
    def test_renders_without_diagnostics_shows_expressions(self):
        md = render_segment_filter_markdown({"account": "has_contract == 1"})
        assert "account" in md and "has_contract == 1" in md

    def test_renders_with_diagnostics_shows_counts_and_pass_rate(self):
        from customer_retention.analysis.auto_explorer.sampling import SegmentFilterStats
        stats = [SegmentFilterStats("account", "has_contract == 1", 402384, 2893)]
        md = render_segment_filter_markdown({"account": "has_contract == 1"}, diagnostics=stats)
        assert "2,893" in md and "402,384" in md
        assert "0.7%" in md

    def test_flags_suspicious_pass_rate(self):
        from customer_retention.analysis.auto_explorer.sampling import SegmentFilterStats
        stats = [SegmentFilterStats("account", "x=1", 1000, 10)]  # 1%
        md = render_segment_filter_markdown({"account": "x=1"}, diagnostics=stats)
        assert "⚠" in md or "warn" in md.lower() or "suspicious" in md.lower()

    def test_no_flag_when_pass_rate_reasonable(self):
        from customer_retention.analysis.auto_explorer.sampling import SegmentFilterStats
        stats = [SegmentFilterStats("account", "x=1", 1000, 500)]  # 50%
        md = render_segment_filter_markdown({"account": "x=1"}, diagnostics=stats)
        assert "⚠" not in md
