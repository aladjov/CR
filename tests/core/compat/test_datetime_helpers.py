from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import (
    _infer_epoch_unit,
    as_tz_naive,
    ensure_datetime_column,
    safe_to_datetime,
)


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
