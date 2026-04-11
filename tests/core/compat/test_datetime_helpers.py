from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import (
    _infer_epoch_unit,
    _normalize_timestamp_columns,
    as_tz_naive,
    ensure_datetime_column,
    groupby_multi_agg,
    groupby_multi_col_agg,
    normalize_timestamps,
    safe_to_datetime,
    timestamp_diff_days,
    timestamp_diff_seconds,
    timestamp_diffs_seconds,
)


class TestGroupbyMultiAgg:
    def test_min_max_count(self):
        df = pd.DataFrame({
            "entity": ["A", "A", "A", "B", "B"],
            "value": [10, 20, 30, 40, 50],
        })
        result = groupby_multi_agg(df, "entity", "value", ["min", "max", "count"])
        result.columns = ["entity", "min", "max", "count"]
        a = result[result["entity"] == "A"].iloc[0]
        b = result[result["entity"] == "B"].iloc[0]
        assert a["min"] == 10
        assert a["max"] == 30
        assert a["count"] == 3
        assert b["min"] == 40
        assert b["max"] == 50
        assert b["count"] == 2

    def test_sum_count_mean(self):
        df = pd.DataFrame({
            "group": ["X", "X", "Y", "Y", "Y"],
            "val": [10.0, 20.0, 30.0, 60.0, 90.0],
        })
        result = groupby_multi_agg(df, "group", "val", ["sum", "count", "mean"])
        result.columns = ["group", "sum", "count", "mean"]
        x = result[result["group"] == "X"].iloc[0]
        y = result[result["group"] == "Y"].iloc[0]
        assert x["sum"] == 30.0
        assert x["count"] == 2
        assert x["mean"] == 15.0
        assert y["sum"] == 180.0
        assert y["mean"] == 60.0

    def test_result_columns(self):
        df = pd.DataFrame({"g": ["a", "b"], "v": [1, 2]})
        result = groupby_multi_agg(df, "g", "v", ["min", "max"])
        assert list(result.columns) == ["g", "min", "max"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"g": pd.Series([], dtype=str), "v": pd.Series([], dtype=float)})
        result = groupby_multi_agg(df, "g", "v", ["sum", "count"])
        assert len(result) == 0
        assert list(result.columns) == ["g", "sum", "count"]


class TestGroupbyMultiColAgg:
    def test_multi_col_sum_mean(self):
        df = pd.DataFrame({
            "entity": ["A", "A", "B", "B"],
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        result = groupby_multi_col_agg(df, "entity", ["x", "y"], ["sum", "mean"])
        a = result[result["entity"] == "A"].iloc[0]
        assert a["x_sum"] == 30.0
        assert a["x_mean"] == 15.0
        assert a["y_sum"] == 3.0
        assert a["y_mean"] == 1.5

    def test_col_prefix(self):
        df = pd.DataFrame({"g": ["A", "A"], "v": [5.0, 15.0]})
        result = groupby_multi_col_agg(df, "g", ["v"], ["sum", "count"], col_prefix="lag0_")
        assert "lag0_v_sum" in result.columns
        assert "lag0_v_count" in result.columns
        assert result.iloc[0]["lag0_v_sum"] == 20.0

    def test_empty_dataframe(self):
        df = pd.DataFrame({"g": pd.Series([], dtype=str), "a": pd.Series([], dtype=float)})
        result = groupby_multi_col_agg(df, "g", ["a"], ["sum", "count"])
        assert len(result) == 0

    def test_count_returns_integers(self):
        df = pd.DataFrame({
            "g": ["A", "A", "B"],
            "v1": [1.0, 2.0, 3.0],
            "v2": [4.0, 5.0, 6.0],
        })
        result = groupby_multi_col_agg(df, "g", ["v1", "v2"], ["count"])
        assert result[result["g"] == "A"].iloc[0]["v1_count"] == 2
        assert result[result["g"] == "B"].iloc[0]["v1_count"] == 1


class TestTimestampDiffsSeconds:
    def test_consecutive_diffs(self):
        ts = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-04"]))
        result = timestamp_diffs_seconds(ts)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 86400.0
        assert result.iloc[2] == 172800.0

    def test_first_element_is_nan(self):
        ts = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02"]))
        result = timestamp_diffs_seconds(ts)
        assert pd.isna(result.iloc[0])
        assert not pd.isna(result.iloc[1])

    def test_single_element(self):
        ts = pd.Series(pd.to_datetime(["2023-01-01"]))
        result = timestamp_diffs_seconds(ts)
        assert len(result) == 1
        assert pd.isna(result.iloc[0])

    def test_uniform_intervals(self):
        ts = pd.Series(pd.to_datetime([f"2023-01-{d:02d}" for d in range(1, 6)]))
        result = timestamp_diffs_seconds(ts).dropna()
        assert (result == 86400.0).all()
        assert len(result) == 4

    def test_sub_day_precision(self):
        ts = pd.Series(pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 01:30:00"]))
        result = timestamp_diffs_seconds(ts)
        assert result.iloc[1] == 5400.0


class TestTimestampDiffSeconds:
    def test_basic_diff(self):
        a = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"]))
        b = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-01"]))
        result = timestamp_diff_seconds(a, b)
        assert result.iloc[0] == 86400.0
        assert result.iloc[1] == 172800.0

    def test_negative_diff(self):
        a = pd.Series(pd.to_datetime(["2024-01-01"]))
        b = pd.Series(pd.to_datetime(["2024-01-02"]))
        result = timestamp_diff_seconds(a, b)
        assert result.iloc[0] == -86400.0

    def test_same_timestamps(self):
        a = pd.Series(pd.to_datetime(["2024-01-01"]))
        b = pd.Series(pd.to_datetime(["2024-01-01"]))
        result = timestamp_diff_seconds(a, b)
        assert result.iloc[0] == 0.0

    def test_sub_day_precision(self):
        a = pd.Series(pd.to_datetime(["2024-01-01 01:30:00"]))
        b = pd.Series(pd.to_datetime(["2024-01-01 00:00:00"]))
        result = timestamp_diff_seconds(a, b)
        assert result.iloc[0] == 5400.0


class TestTimestampDiffDays:
    def test_basic_diff(self):
        a = pd.Series(pd.to_datetime(["2024-01-11", "2024-01-05"]))
        b = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-01"]))
        result = timestamp_diff_days(a, b)
        assert np.isclose(result.iloc[0], 10.0)
        assert np.isclose(result.iloc[1], 4.0)

    def test_negative_diff(self):
        a = pd.Series(pd.to_datetime(["2024-01-01"]))
        b = pd.Series(pd.to_datetime(["2024-01-04"]))
        result = timestamp_diff_days(a, b)
        assert np.isclose(result.iloc[0], -3.0)

    def test_fractional_days(self):
        a = pd.Series(pd.to_datetime(["2024-01-01 12:00:00"]))
        b = pd.Series(pd.to_datetime(["2024-01-01 00:00:00"]))
        result = timestamp_diff_days(a, b)
        assert np.isclose(result.iloc[0], 0.5)


class TestInferEpochUnit:
    def test_seconds(self):
        assert _infer_epoch_unit(1_672_531_200) == "s"

    def test_milliseconds(self):
        assert _infer_epoch_unit(1_672_531_200_000) == "ms"

    def test_microseconds(self):
        assert _infer_epoch_unit(1_672_531_200_000_000) == "us"

    def test_nanoseconds(self):
        assert _infer_epoch_unit(1_672_531_200_000_000_000) == "ns"

    def test_negative_timestamp(self):
        assert _infer_epoch_unit(-1_672_531_200_000) == "ms"

    def test_zero(self):
        assert _infer_epoch_unit(0) == "s"


class TestSafeToDatetimeSparkPandasTimestamp:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_spark_pandas_datetime_series_returned_without_iter(self):
        series = pd.Series(pd.to_datetime(["2023-01-01", "2023-06-15"]))
        series.spark = True
        series.to_spark = lambda: None
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 2

    def test_spark_pandas_string_series_falls_to_spark_branch(self):
        from unittest.mock import MagicMock, patch
        mock_series = MagicMock()
        mock_series.spark = MagicMock()
        mock_series.to_spark = MagicMock()
        mock_series.dtype = "object"
        transformed = MagicMock()
        mock_series.spark.transform.return_value = transformed
        with patch("customer_retention.core.compat._is_spark_pandas", return_value=True):
            with patch("pandas.api.types.is_datetime64_any_dtype", return_value=False):
                result = safe_to_datetime(mock_series)
        mock_series.spark.transform.assert_called_once()
        assert result is transformed


class TestSafeToDatetime:
    def test_already_datetime(self):
        series = pd.Series(pd.to_datetime(["2023-01-01", "2023-06-15"]))
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 2

    def test_string_dates(self):
        series = pd.Series(["2023-01-01", "2023-06-15"])
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2023-01-01")

    def test_epoch_seconds_int64(self):
        ts = int(pd.Timestamp("2023-01-01").timestamp())
        series = pd.Series([ts, ts + 86400], dtype="int64")
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0].year == 2023

    def test_epoch_milliseconds_int64(self):
        ts_ms = int(pd.Timestamp("2023-01-01").timestamp() * 1000)
        series = pd.Series([ts_ms, ts_ms + 86_400_000], dtype="int64")
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0].year == 2023

    def test_epoch_microseconds_int64(self):
        ts_us = int(pd.Timestamp("2023-01-01").timestamp() * 1_000_000)
        series = pd.Series([ts_us, ts_us + 86_400_000_000], dtype="int64")
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0].year == 2023

    def test_epoch_nanoseconds_int64(self):
        ts_ns = int(pd.Timestamp("2023-01-01").value)
        series = pd.Series([ts_ns, ts_ns + 86_400_000_000_000], dtype="int64")
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0].year == 2023

    def test_with_nan_values(self):
        ts = int(pd.Timestamp("2023-01-01").timestamp())
        series = pd.array([ts, pd.NA, ts + 86400], dtype="Int64")
        result = safe_to_datetime(pd.Series(series))
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert pd.isna(result.iloc[1])
        assert result.iloc[0].year == 2023

    def test_all_nan_integer(self):
        series = pd.Series([None, None, None], dtype="Int64")
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.isna().all()

    def test_errors_coerce_kwarg_passthrough(self):
        series = pd.Series(["2023-01-01", "not-a-date", "2023-06-15"])
        result = safe_to_datetime(series, errors="coerce")
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert pd.isna(result.iloc[1])

    def test_mixed_iso8601_with_and_without_fractional_seconds(self):
        series = pd.Series([
            "2024-07-29T15:48:00.123Z",
            "2024-07-29T15:48:00Z",
            "2024-08-01T10:00:00.000Z",
        ])
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0].year == 2024
        assert result.iloc[1] == pd.Timestamp("2024-07-29T15:48:00")
        assert result.dt.tz is None

    def test_mixed_iso8601_without_timezone(self):
        series = pd.Series([
            "2024-01-01T00:00:00.500",
            "2024-01-02T12:00:00",
        ])
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 2
        assert result.dt.tz is None

    def test_iso8601_uniform_no_fractional(self):
        series = pd.Series(["2024-07-29T15:48:00Z", "2024-07-30T10:00:00Z"])
        result = safe_to_datetime(series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0].day == 29
        assert result.dt.tz is None

    def test_mixed_iso8601_comparable_with_naive_timestamp(self):
        series = pd.Series([
            "2024-07-29T15:48:00.123Z",
            "2024-07-29T15:48:00Z",
            "2024-08-01T10:00:00.000Z",
        ])
        result = safe_to_datetime(series)
        cutoff = pd.Timestamp("2024-07-30")
        mask = result < cutoff
        assert mask.sum() == 2

    def test_already_datetime_tz_aware_stripped(self):
        series = pd.Series(pd.to_datetime(["2023-01-01", "2023-06-15"]).tz_localize("UTC"))
        result = safe_to_datetime(series)
        assert result.dt.tz is None
        cutoff = pd.Timestamp("2023-03-01")
        mask = result < cutoff
        assert mask.sum() == 1


class TestEnsureDatetimeColumn:
    def test_converts_string_column(self):
        df = pd.DataFrame({"ts": ["2023-01-01", "2023-06-15"], "val": [1, 2]})
        ensure_datetime_column(df, "ts")
        assert pd.api.types.is_datetime64_any_dtype(df["ts"])

    def test_noop_on_datetime_column(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2023-01-01", "2023-06-15"]), "val": [1, 2]})
        ensure_datetime_column(df, "ts")
        assert pd.api.types.is_datetime64_any_dtype(df["ts"])
        assert df["ts"].iloc[0] == pd.Timestamp("2023-01-01")

    def test_converts_epoch_seconds_column(self):
        ts = int(pd.Timestamp("2023-01-01").timestamp())
        df = pd.DataFrame({"ts": [ts, ts + 86400], "val": [1, 2]})
        ensure_datetime_column(df, "ts")
        assert pd.api.types.is_datetime64_any_dtype(df["ts"])
        assert df["ts"].iloc[0].year == 2023

    def test_converts_epoch_millis_column(self):
        ts_ms = int(pd.Timestamp("2023-01-01").timestamp() * 1000)
        df = pd.DataFrame({"ts": [ts_ms, ts_ms + 86_400_000]})
        ensure_datetime_column(df, "ts")
        assert pd.api.types.is_datetime64_any_dtype(df["ts"])
        assert df["ts"].iloc[0].year == 2023

    def test_returns_same_dataframe(self):
        df = pd.DataFrame({"ts": ["2023-01-01"], "val": [1]})
        result = ensure_datetime_column(df, "ts")
        assert result is df

    def test_does_not_affect_other_columns(self):
        df = pd.DataFrame({"ts": ["2023-01-01"], "val": [42], "name": ["test"]})
        ensure_datetime_column(df, "ts")
        assert df["val"].iloc[0] == 42
        assert df["name"].iloc[0] == "test"

    def test_mixed_iso8601_with_and_without_fractional_seconds(self):
        df = pd.DataFrame({"ts": [
            "2024-07-29T15:48:00.123Z",
            "2024-07-29T15:48:00Z",
            "2024-08-01T10:00:00.000Z",
        ], "val": [1, 2, 3]})
        ensure_datetime_column(df, "ts")
        assert pd.api.types.is_datetime64_any_dtype(df["ts"])
        assert df["ts"].iloc[0].year == 2024


class TestAsTzNaive:
    def test_naive_scalar_unchanged(self):
        ts = pd.Timestamp("2023-06-15")
        assert as_tz_naive(ts) == ts

    def test_aware_scalar_stripped(self):
        ts = pd.Timestamp("2023-06-15", tz="UTC")
        result = as_tz_naive(ts)
        assert result.tzinfo is None
        assert result == pd.Timestamp("2023-06-15")

    def test_datetime_scalar_naive_unchanged(self):
        dt = datetime(2023, 6, 15)
        assert as_tz_naive(dt) == dt

    def test_datetime_scalar_aware_stripped(self):
        from zoneinfo import ZoneInfo
        dt = datetime(2023, 6, 15, tzinfo=ZoneInfo("UTC"))
        result = as_tz_naive(dt)
        assert result.tzinfo is None

    def test_naive_series_unchanged(self):
        series = pd.Series(pd.to_datetime(["2023-01-01", "2023-06-15"]))
        result = as_tz_naive(series)
        pd.testing.assert_series_equal(result, series)

    def test_aware_series_stripped(self):
        series = pd.Series(pd.to_datetime(["2023-01-01", "2023-06-15"]).tz_localize("UTC"))
        result = as_tz_naive(series)
        assert not hasattr(result.dtype, "tz") or result.dtype.tz is None
        assert result.iloc[0] == pd.Timestamp("2023-01-01")

    def test_non_datetime_passthrough(self):
        assert as_tz_naive(42) == 42
        assert as_tz_naive("hello") == "hello"

    def test_aware_series_comparison_with_naive_scalar(self):
        series = pd.Series(pd.to_datetime(["2023-01-01", "2025-01-01"]).tz_localize("UTC"))
        cutoff = pd.Timestamp("2024-01-01")
        mask = as_tz_naive(series) > as_tz_naive(cutoff)
        assert mask.sum() == 1


class TestTimeSeriesProfilerWithEpochIntegers:
    def test_profile_with_epoch_seconds(self):
        from customer_retention.stages.profiling.time_series_profiler import TimeSeriesProfiler
        base_ts = int(pd.Timestamp("2023-01-01").timestamp())
        df = pd.DataFrame({
            "customer_id": ["A", "A", "B", "B", "B"],
            "event_time": [base_ts, base_ts + 86400, base_ts, base_ts + 86400, base_ts + 172800],
        })
        profiler = TimeSeriesProfiler("customer_id", "event_time")
        profile = profiler.profile(df)
        assert profile.total_events == 5
        assert profile.unique_entities == 2
        assert profile.time_span_days == 2

    def test_profile_with_epoch_milliseconds(self):
        from customer_retention.stages.profiling.time_series_profiler import TimeSeriesProfiler
        base_ts = int(pd.Timestamp("2023-01-01").timestamp() * 1000)
        df = pd.DataFrame({
            "customer_id": ["A", "A", "B"],
            "event_time": [base_ts, base_ts + 86_400_000, base_ts],
        })
        profiler = TimeSeriesProfiler("customer_id", "event_time")
        profile = profiler.profile(df)
        assert profile.total_events == 3
        assert profile.unique_entities == 2

    def test_quality_check_with_epoch_seconds(self):
        from customer_retention.stages.profiling.temporal_quality_checks import TemporalGapCheck
        base_ts = int(pd.Timestamp("2023-01-01").timestamp())
        df = pd.DataFrame({
            "event_time": [base_ts + i * 86400 for i in range(10)],
        })
        check = TemporalGapCheck("event_time", expected_frequency="D")
        result = check.run(df)
        assert result.passed

    def test_time_window_aggregator_with_epoch_seconds(self):
        from customer_retention.stages.profiling.time_window_aggregator import TimeWindowAggregator
        base_ts = int(pd.Timestamp("2023-01-01").timestamp())
        df = pd.DataFrame({
            "customer_id": ["A"] * 5 + ["B"] * 3,
            "event_time": [base_ts + i * 86400 for i in range(5)] + [base_ts + i * 86400 for i in range(3)],
            "amount": np.random.rand(8) * 100,
        })
        agg = TimeWindowAggregator("customer_id", "event_time")
        result = agg.aggregate(df, windows=["30d"], include_recency=True, include_tenure=True)
        assert "days_since_last_event" in result.columns
        assert "days_since_first_event" in result.columns
        assert len(result) == 2


class TestAsTzNaiveDatetimeIndex:
    def test_naive_index_unchanged(self):
        idx = pd.DatetimeIndex(["2023-01-01", "2023-06-15"])
        result = as_tz_naive(idx)
        pd.testing.assert_index_equal(result, idx)

    def test_aware_index_stripped(self):
        idx = pd.DatetimeIndex(["2023-01-01", "2023-06-15"]).tz_localize("UTC")
        result = as_tz_naive(idx)
        assert result.tz is None
        assert result[0] == pd.Timestamp("2023-01-01")

    def test_non_utc_timezone_stripped(self):
        idx = pd.DatetimeIndex(["2023-01-01", "2023-06-15"]).tz_localize("US/Eastern")
        result = as_tz_naive(idx)
        assert result.tz is None


class TestNormalizeTimestampColumns:
    def test_naive_passthrough(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01", "2023-06-15"]),
            "val": [1, 2],
        })
        result = _normalize_timestamp_columns(df)
        pd.testing.assert_frame_equal(result, df)

    def test_tz_aware_stripped(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01", "2023-06-15"]).tz_localize("UTC"),
            "val": [1, 2],
        })
        result = _normalize_timestamp_columns(df)
        assert result["ts"].dt.tz is None
        assert result["ts"].iloc[0] == pd.Timestamp("2023-01-01")

    def test_mixed_columns(self):
        df = pd.DataFrame({
            "ts1": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
            "ts2": pd.to_datetime(["2023-06-15"]),
            "val": [42],
        })
        result = _normalize_timestamp_columns(df)
        assert result["ts1"].dt.tz is None
        assert result["ts2"].dt.tz is None
        assert result["val"].iloc[0] == 42

    def test_non_datetime_untouched(self):
        df = pd.DataFrame({"name": ["alice"], "age": [30]})
        result = _normalize_timestamp_columns(df)
        pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = _normalize_timestamp_columns(df)
        assert result.empty

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
        })
        _normalize_timestamp_columns(df)
        assert df["ts"].dt.tz is not None


class TestNormalizeTimestampColumnsDecimal:
    def test_decimal_column_converted_to_numeric(self):
        from decimal import Decimal
        df = pd.DataFrame({"amount": [Decimal("10.50"), Decimal("20.75"), Decimal("30.00")]})
        result = _normalize_timestamp_columns(df)
        assert pd.api.types.is_numeric_dtype(result["amount"])
        assert result["amount"].iloc[0] == pytest.approx(10.50)

    def test_decimal_with_nulls(self):
        from decimal import Decimal
        df = pd.DataFrame({"amount": [Decimal("10.50"), None, Decimal("30.00")]})
        result = _normalize_timestamp_columns(df)
        assert pd.api.types.is_numeric_dtype(result["amount"])
        assert pd.isna(result["amount"].iloc[1])
        assert result["amount"].iloc[2] == pytest.approx(30.00)

    def test_string_column_not_converted(self):
        df = pd.DataFrame({"name": ["alice", "bob"]})
        result = _normalize_timestamp_columns(df)
        assert result["name"].dtype == "object"

    def test_decimal_and_timestamp_mixed(self):
        from decimal import Decimal
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
            "amount": [Decimal("99.99")],
            "label": ["ok"],
        })
        result = _normalize_timestamp_columns(df)
        assert result["ts"].dt.tz is None
        assert pd.api.types.is_numeric_dtype(result["amount"])
        assert result["label"].dtype == "object"

    def test_all_null_object_column_untouched(self):
        df = pd.DataFrame({"x": [None, None]})
        result = _normalize_timestamp_columns(df)
        assert result["x"].dtype == "object"


class TestPandasDtypeToSparkSchema:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_int_columns(self):
        from pyspark.sql.types import IntegerType, LongType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"a": pd.array([1, 2], dtype="int32"), "b": pd.array([3, 4], dtype="int64")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, IntegerType)
        assert isinstance(schema.fields[1].dataType, LongType)

    def test_float_columns(self):
        from pyspark.sql.types import DoubleType, FloatType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"a": pd.array([1.0], dtype="float32"), "b": pd.array([2.0], dtype="float64")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, FloatType)
        assert isinstance(schema.fields[1].dataType, DoubleType)

    def test_bool_column(self):
        from pyspark.sql.types import BooleanType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"flag": [True, False]})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, BooleanType)

    def test_string_column(self):
        from pyspark.sql.types import StringType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"name": ["alice", "bob"]})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, StringType)

    def test_datetime_column(self):
        from pyspark.sql.types import TimestampNTZType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"ts": pd.to_datetime(["2023-01-01"])})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, TimestampNTZType)

    def test_nullable_int_column(self):
        from pyspark.sql.types import LongType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"a": pd.array([1, pd.NA, 3], dtype="Int64")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, LongType)

    def test_empty_dataframe(self):
        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame()
        schema = pandas_dtype_to_spark_schema(df)
        assert len(schema.fields) == 0

    def test_unknown_dtype_falls_back_to_string(self):
        from pyspark.sql.types import StringType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"])})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, StringType)

    def test_decimal_column_maps_to_double_after_normalize(self):
        from decimal import Decimal

        from pyspark.sql.types import DoubleType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = _normalize_timestamp_columns(
            pd.DataFrame({"amount": [Decimal("10.50"), Decimal("20.75")]})
        )
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, DoubleType)

    def test_object_column_with_floats_inferred_as_double(self):
        from pyspark.sql.types import DoubleType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"val": pd.array([1.5, None, 3.0], dtype="object")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, DoubleType)

    def test_object_column_with_ints_inferred_as_long(self):
        from pyspark.sql.types import LongType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"val": pd.array([1, None, 3], dtype="object")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, LongType)

    def test_object_column_with_mixed_int_float_inferred_as_double(self):
        from pyspark.sql.types import DoubleType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"val": pd.array([1, 2.5, None], dtype="object")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, DoubleType)

    def test_object_column_with_bools_inferred_as_boolean(self):
        from pyspark.sql.types import BooleanType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"val": pd.array([True, None, False], dtype="object")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, BooleanType)

    def test_object_column_with_datetimes_inferred_as_timestamp(self):
        import datetime

        from pyspark.sql.types import TimestampNTZType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"val": pd.array([datetime.datetime(2023, 1, 1), None], dtype="object")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, TimestampNTZType)

    def test_object_column_with_strings_stays_string(self):
        from pyspark.sql.types import StringType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"val": ["hello", None, "world"]})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, StringType)

    def test_object_column_all_none_defaults_to_string(self):
        from pyspark.sql.types import StringType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"val": [None, None, None]})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, StringType)

    def test_object_column_mixed_types_defaults_to_string(self):
        from pyspark.sql.types import StringType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"val": pd.array([1.0, "hello", None], dtype="object")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, StringType)

    def test_object_column_numpy_float64_values(self):
        import numpy as np
        from pyspark.sql.types import DoubleType

        from customer_retention.core.compat import pandas_dtype_to_spark_schema
        df = pd.DataFrame({"hour_mean_90d": pd.array([np.float64(12.5), None, np.float64(8.3)], dtype="object")})
        schema = pandas_dtype_to_spark_schema(df)
        assert isinstance(schema.fields[0].dataType, DoubleType)


class TestBuildSpineWithTzAwareDates:
    def test_tz_aware_grid_dates_produce_naive_spine(self):
        from customer_retention.stages.temporal.temporal_merger import TemporalMerger
        merger = TemporalMerger()
        entities = pd.Series(["A", "B"])
        tz_dates = [
            pd.Timestamp("2023-01-01", tz="UTC").isoformat(),
            pd.Timestamp("2023-02-01", tz="UTC").isoformat(),
        ]
        spine = merger.build_spine(entities, tz_dates)
        assert spine[merger.config.as_of_column].dt.tz is None

    def test_naive_grid_dates_still_work(self):
        from customer_retention.stages.temporal.temporal_merger import TemporalMerger
        merger = TemporalMerger()
        entities = pd.Series(["A"])
        naive_dates = ["2023-01-01", "2023-02-01"]
        spine = merger.build_spine(entities, naive_dates)
        assert spine[merger.config.as_of_column].dt.tz is None
        assert len(spine) == 2


class TestTimedeltaToDays:
    def test_timedelta_series(self):
        from customer_retention.core.compat import timedelta_to_days
        s = pd.Series(pd.to_datetime(["2023-01-10", "2023-01-05"])) - pd.Timestamp("2023-01-01")
        result = timedelta_to_days(s)
        assert list(result) == [9, 4]

    def test_integer_seconds_series(self):
        from customer_retention.core.compat import timedelta_to_days
        s = pd.Series([86400 * 3, 86400 * 7, 0])
        result = timedelta_to_days(s)
        assert list(result) == [3, 7, 0]

    def test_negative_timedelta(self):
        from customer_retention.core.compat import timedelta_to_days
        s = pd.Series([pd.Timedelta(days=-2), pd.Timedelta(days=5)])
        result = timedelta_to_days(s)
        assert list(result) == [-2, 5]

    def test_negative_integer_seconds(self):
        from customer_retention.core.compat import timedelta_to_days
        s = pd.Series([-86400 * 2, 86400 * 5])
        result = timedelta_to_days(s)
        assert list(result) == [-2, 5]

    def test_diff_on_datetime_column(self):
        from customer_retention.core.compat import timedelta_to_days
        dates = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-04", "2023-01-10"]))
        result = timedelta_to_days(dates.diff()).dropna()
        assert list(result) == [3, 6]


class TestTimedeltaToSeconds:
    def test_timedelta_series(self):
        from customer_retention.core.compat import timedelta_to_seconds
        s = pd.Series([pd.Timedelta(hours=1), pd.Timedelta(minutes=30)])
        result = timedelta_to_seconds(s)
        assert list(result) == [3600.0, 1800.0]

    def test_integer_seconds_series(self):
        from customer_retention.core.compat import timedelta_to_seconds
        s = pd.Series([3600, 1800, 0])
        result = timedelta_to_seconds(s)
        assert list(result) == [3600.0, 1800.0, 0.0]

    def test_diff_on_datetime_column(self):
        from customer_retention.core.compat import timedelta_to_seconds
        dates = pd.Series(pd.to_datetime(["2023-01-01T00:00:00", "2023-01-01T01:00:00"]))
        result = timedelta_to_seconds(dates.diff()).dropna()
        assert list(result) == [3600.0]

    def test_fallback_microsecond_integer_series(self):
        from customer_retention.core.compat import timedelta_to_seconds

        microseconds = pd.Series([93600 * 1_000_000, 10800 * 1_000_000])

        class FakeSeries:
            def __init__(self, data):
                self._data = data

            def astype(self, dtype):
                return self._data.astype(dtype)

            @property
            def dt(self):
                raise AttributeError

        fake = FakeSeries(microseconds)
        result = timedelta_to_seconds(fake)
        assert list(result) == [93600.0, 10800.0]


class TestDataMaterializerDatetimeConversion:
    def _make_recommendation(self, target_column, parameters):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        return LayeredRecommendation(
            id="test_conv_1",
            action="convert",
            target_column=target_column,
            parameters=parameters,
            layer="bronze",
            category="type_conversion",
            rationale="test",
            source_notebook="test",
        )

    def _make_registry_with_type_conversion(self, target_column, parameters):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            BronzeRecommendations,
            RecommendationRegistry,
        )
        registry = RecommendationRegistry()
        registry.bronze = BronzeRecommendations(
            source_file="test",
            type_conversions=[self._make_recommendation(target_column, parameters)],
        )
        return registry

    def test_type_conversion_uses_safe_to_datetime(self):
        from customer_retention.generators.orchestration.data_materializer import DataMaterializer

        registry = self._make_registry_with_type_conversion("ts", {"target_type": "datetime"})
        materializer = DataMaterializer(registry, storage=None)
        df = pd.DataFrame({"ts": ["2023-01-01", "2023-06-15"], "val": [1, 2]})
        result = materializer.apply_bronze(df)
        assert pd.api.types.is_datetime64_any_dtype(result["ts"])

    def test_epoch_seconds_conversion(self):
        from customer_retention.generators.orchestration.data_materializer import DataMaterializer

        ts_epoch = int(pd.Timestamp("2023-01-01").timestamp())
        registry = self._make_registry_with_type_conversion("ts", {"target_type": "datetime"})
        materializer = DataMaterializer(registry, storage=None)
        df = pd.DataFrame({"ts": [ts_epoch, ts_epoch + 86400], "val": [1, 2]})
        result = materializer.apply_bronze(df)
        assert pd.api.types.is_datetime64_any_dtype(result["ts"])
        assert result["ts"].iloc[0].year == 2023


class TestNormalizeTimestamps:
    def test_pandas_tz_stripped(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01", "2023-06-15"]).tz_localize("UTC"),
            "val": [1, 2],
        })
        result = normalize_timestamps(df)
        assert result["ts"].dt.tz is None
        assert result["ts"].iloc[0] == pd.Timestamp("2023-01-01")

    def test_pandas_no_tz_unchanged(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01"]),
            "val": [1],
        })
        result = normalize_timestamps(df)
        assert result["ts"].dt.tz is None
        assert result["val"].iloc[0] == 1

    def test_pandas_does_not_mutate_original(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01"]).tz_localize("US/Eastern"),
        })
        normalize_timestamps(df)
        assert df["ts"].dt.tz is not None

    def test_pandas_empty_df(self):
        result = normalize_timestamps(pd.DataFrame())
        assert result.empty

    def test_pandas_mixed_tz_and_plain(self):
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
            "ts_naive": pd.to_datetime(["2023-01-01"]),
            "val": [42],
        })
        result = normalize_timestamps(df)
        assert result["ts_utc"].dt.tz is None
        assert result["ts_naive"].dt.tz is None

    def test_delegates_to_normalize_timestamp_columns_for_pandas(self):
        from decimal import Decimal
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
            "amount": [Decimal("10.50")],
        })
        result = normalize_timestamps(df)
        assert result["ts"].dt.tz is None
        assert pd.api.types.is_numeric_dtype(result["amount"])


class TestStripSparkTimestampTz:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_casts_timestamp_to_ntz(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType

        from customer_retention.core.compat import strip_spark_timestamp_tz

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampType()),
            StructField("name", StringType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = strip_spark_timestamp_tz(mock_df)
        mock_df.select.assert_called_once()
        assert result is mock_df.select.return_value

    def test_no_timestamp_columns_returns_original(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StringType, StructField, StructType

        from customer_retention.core.compat import strip_spark_timestamp_tz

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = strip_spark_timestamp_tz(mock_df)
        assert result is mock_df
        mock_df.select.assert_not_called()

    def test_ntz_columns_returns_original(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StructField, StructType, TimestampNTZType

        from customer_retention.core.compat import strip_spark_timestamp_tz

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampNTZType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = strip_spark_timestamp_tz(mock_df)
        assert result is mock_df
        mock_df.select.assert_not_called()

    def test_mixed_timestamp_and_ntz(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import (
            IntegerType,
            StructField,
            StructType,
            TimestampNTZType,
            TimestampType,
        )

        from customer_retention.core.compat import strip_spark_timestamp_tz

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts_tz", TimestampType()),
            StructField("ts_ntz", TimestampNTZType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = strip_spark_timestamp_tz(mock_df)
        mock_df.select.assert_called_once()
        assert result is mock_df.select.return_value


class TestClampSparkTimestamps:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_no_timestamp_columns_returns_original(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StringType, StructField, StructType

        from customer_retention.core.compat import clamp_spark_timestamps

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = clamp_spark_timestamps(mock_df)
        assert result is mock_df
        mock_df.select.assert_not_called()

    def test_clamps_ntz_columns(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StructField, StructType, TimestampNTZType

        from customer_retention.core.compat import clamp_spark_timestamps

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampNTZType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = clamp_spark_timestamps(mock_df)
        mock_df.select.assert_called_once()
        assert result is mock_df.select.return_value

    def test_clamps_tz_columns(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StructField, StructType, TimestampType

        from customer_retention.core.compat import clamp_spark_timestamps

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = clamp_spark_timestamps(mock_df)
        mock_df.select.assert_called_once()
        assert result is mock_df.select.return_value

    def test_mixed_ts_and_non_ts(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import (
            IntegerType,
            StringType,
            StructField,
            StructType,
            TimestampNTZType,
        )

        from customer_retention.core.compat import clamp_spark_timestamps

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampNTZType()),
            StructField("name", StringType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = clamp_spark_timestamps(mock_df)
        mock_df.select.assert_called_once()
        args = mock_df.select.call_args[0][0]
        assert len(args) == 3


class TestClampDistributedTimestamps:
    def test_returns_native_pandas_unchanged(self):
        from customer_retention.core.compat import clamp_distributed_timestamps

        df = pd.DataFrame({"id": [1], "ts": pd.to_datetime(["2020-01-01"])})
        result = clamp_distributed_timestamps(df)
        assert result is df

    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_clamps_datetime_columns(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.core.compat import clamp_distributed_timestamps

        mock_df = MagicMock()
        mock_df.columns = ["id", "ts"]
        mock_id_series = MagicMock()
        mock_id_series.dtype = "int64"
        mock_ts_series = MagicMock()
        mock_ts_series.dtype = "datetime64[ns]"
        mock_transformed = MagicMock()
        mock_ts_series.spark.transform.return_value = mock_transformed

        def getitem(key):
            return {"id": mock_id_series, "ts": mock_ts_series}[key]
        mock_df.__getitem__ = MagicMock(side_effect=getitem)

        with patch("customer_retention.core.compat._is_spark_pandas", return_value=True):
            result = clamp_distributed_timestamps(mock_df)

        mock_ts_series.spark.transform.assert_called_once()
        assert result is mock_df

    def test_skips_non_timestamp_columns(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.core.compat import clamp_distributed_timestamps

        mock_df = MagicMock()
        mock_df.columns = ["id", "name"]
        mock_id = MagicMock()
        mock_id.dtype = "int64"
        mock_name = MagicMock()
        mock_name.dtype = "object"

        def getitem(key):
            return {"id": mock_id, "name": mock_name}[key]
        mock_df.__getitem__ = MagicMock(side_effect=getitem)

        with patch("customer_retention.core.compat._is_spark_pandas", return_value=True):
            clamp_distributed_timestamps(mock_df)

        mock_id.spark.transform.assert_not_called()
        mock_name.spark.transform.assert_not_called()


class TestSanitizeSparkTimestamps:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_strips_tz_and_clamps(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StructField, StructType, TimestampNTZType, TimestampType

        from customer_retention.core.compat import sanitize_spark_timestamps

        ts_schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampType()),
        ])
        ntz_schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampNTZType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = ts_schema.fields
        mock_stripped = MagicMock()
        mock_stripped.schema.fields = ntz_schema.fields
        mock_df.select.return_value = mock_stripped
        mock_clamped = MagicMock()
        mock_stripped.select.return_value = mock_clamped

        result = sanitize_spark_timestamps(mock_df)
        mock_df.select.assert_called_once()
        mock_stripped.select.assert_called_once()
        assert result is mock_clamped

    def test_no_timestamp_columns_returns_original(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StringType, StructField, StructType

        from customer_retention.core.compat import sanitize_spark_timestamps

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        result = sanitize_spark_timestamps(mock_df)
        assert result is mock_df
        mock_df.select.assert_not_called()

    def test_no_schema_returns_original(self):
        from customer_retention.core.compat import sanitize_spark_timestamps

        class FakeDF:
            pass

        df = FakeDF()
        assert sanitize_spark_timestamps(df) is df

    def test_ntz_only_still_clamps(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StructField, StructType, TimestampNTZType

        from customer_retention.core.compat import sanitize_spark_timestamps

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampNTZType()),
        ])
        mock_df = MagicMock()
        mock_df.schema.fields = schema.fields
        mock_clamped = MagicMock()
        mock_df.select.return_value = mock_clamped

        result = sanitize_spark_timestamps(mock_df)
        mock_df.select.assert_called_once()
        assert result is mock_clamped


class TestAsPandasApiSanitizes:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_sanitizes_before_wrapping(self):
        from unittest.mock import MagicMock, call, patch

        from customer_retention.core.compat import as_pandas_api

        mock_spark_df = MagicMock()
        mock_sanitized = MagicMock()
        mock_ps_result = MagicMock()
        mock_sanitized.pandas_api.return_value = mock_ps_result

        with patch("customer_retention.core.compat.sanitize_spark_timestamps", return_value=mock_sanitized) as mock_sanitize:
            result = as_pandas_api(mock_spark_df)

        mock_sanitize.assert_called_once_with(mock_spark_df)
        assert result is mock_ps_result


class TestNormalizeTimestampsDistributed:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_strips_and_clamps_spark_timestamps(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import (
            IntegerType,
            StringType,
            StructField,
            StructType,
            TimestampNTZType,
            TimestampType,
        )

        ts_schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampType()),
            StructField("name", StringType()),
        ])
        ntz_schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampNTZType()),
            StructField("name", StringType()),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema.fields = ts_schema.fields
        mock_stripped = MagicMock()
        mock_stripped.schema.fields = ntz_schema.fields
        mock_spark_df.select.return_value = mock_stripped
        mock_clamped = MagicMock()
        mock_stripped.select.return_value = mock_clamped
        mock_ps_result = MagicMock()
        mock_clamped.pandas_api.return_value = mock_ps_result

        mock_df = MagicMock()
        mock_df.to_spark.return_value = mock_spark_df

        with (
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "customer_retention.core.compat._is_spark_pandas", return_value=True,
            ),
        ):
            result = normalize_timestamps(mock_df)
        mock_spark_df.select.assert_called_once()
        mock_stripped.select.assert_called_once()
        assert result is mock_ps_result

    def test_clamps_ntz_columns(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StructField, StructType, TimestampNTZType

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("ts", TimestampNTZType()),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema.fields = schema.fields
        mock_clamped = MagicMock()
        mock_spark_df.select.return_value = mock_clamped
        mock_ps_result = MagicMock()
        mock_clamped.pandas_api.return_value = mock_ps_result

        mock_df = MagicMock()
        mock_df.to_spark.return_value = mock_spark_df

        with (
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "customer_retention.core.compat._is_spark_pandas", return_value=True,
            ),
        ):
            result = normalize_timestamps(mock_df)
        mock_spark_df.select.assert_called_once()
        assert result is mock_ps_result

    def test_no_timestamp_columns_returns_original(self):
        from unittest.mock import MagicMock

        from pyspark.sql.types import IntegerType, StringType, StructField, StructType

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema.fields = schema.fields

        mock_df = MagicMock()
        mock_df.to_spark.return_value = mock_spark_df

        with (
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "customer_retention.core.compat._is_spark_pandas", return_value=True,
            ),
        ):
            result = normalize_timestamps(mock_df)
        assert result is mock_df
