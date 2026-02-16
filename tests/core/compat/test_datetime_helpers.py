from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import (
    _infer_epoch_unit,
    as_tz_naive,
    ensure_datetime_column,
    normalize_timestamp_columns,
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
        result = normalize_timestamp_columns(df)
        pd.testing.assert_frame_equal(result, df)

    def test_tz_aware_stripped(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01", "2023-06-15"]).tz_localize("UTC"),
            "val": [1, 2],
        })
        result = normalize_timestamp_columns(df)
        assert result["ts"].dt.tz is None
        assert result["ts"].iloc[0] == pd.Timestamp("2023-01-01")

    def test_mixed_columns(self):
        df = pd.DataFrame({
            "ts1": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
            "ts2": pd.to_datetime(["2023-06-15"]),
            "val": [42],
        })
        result = normalize_timestamp_columns(df)
        assert result["ts1"].dt.tz is None
        assert result["ts2"].dt.tz is None
        assert result["val"].iloc[0] == 42

    def test_non_datetime_untouched(self):
        df = pd.DataFrame({"name": ["alice"], "age": [30]})
        result = normalize_timestamp_columns(df)
        pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = normalize_timestamp_columns(df)
        assert result.empty

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
        })
        normalize_timestamp_columns(df)
        assert df["ts"].dt.tz is not None


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

    def test_fallback_without_total_seconds(self):
        from unittest.mock import PropertyMock, patch

        from customer_retention.core.compat import timedelta_to_seconds
        s = pd.Series([pd.Timedelta(days=1, hours=2), pd.Timedelta(days=0, hours=3)])
        with patch.object(type(s.dt), "total_seconds", new_callable=PropertyMock, side_effect=AttributeError):
            result = timedelta_to_seconds(s)
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
