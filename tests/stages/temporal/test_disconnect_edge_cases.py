"""Tests for disconnects between temporal pipeline components.

Each test covers a specific disconnect that could silently produce incorrect
results if the pipeline components make conflicting assumptions.
"""

import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.temporal.cutoff_analyzer import CutoffAnalyzer
from customer_retention.stages.temporal.snapshot_manager import SnapshotManager
from customer_retention.stages.temporal.timestamp_discovery import DatetimeOrderAnalyzer
from customer_retention.stages.temporal.timestamp_manager import (
    TimestampConfig,
    TimestampManager,
    TimestampStrategy,
)


class TestDoubleFilterMismatch:
    """split_at_cutoff() uses only timestamp, but create_snapshot() also
    filters on label_available_flag. These can produce different counts."""

    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    def test_split_at_cutoff_ignores_label_available_flag(self):
        n = 100
        ts = pd.Series(pd.date_range("2022-01-01", periods=n, freq="D"), name="ts")
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_available_flag": [True] * 50 + [False] * 50,
            "target": [0] * n,
        })

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        # Use fixed cutoff that's clearly past the flag boundary (row 50)
        cutoff = datetime(2022, 3, 15)  # 73 days in → rows 0-73 in train
        split = analysis.split_at_cutoff(cutoff)

        # split_at_cutoff includes rows with flag=False (ignores flag)
        assert split.train_count > 50

    def test_snapshot_respects_label_available_flag(self, temp_dir):
        n = 100
        ts = pd.Series(pd.date_range("2022-01-01", periods=n, freq="D"))
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=90),
            "label_available_flag": [True] * 50 + [False] * 50,
            "target": [0] * n,
        })

        manager = SnapshotManager(temp_dir)
        cutoff = datetime(2022, 4, 1)  # ~90 days in
        meta = manager.create_snapshot(df, cutoff, "target", timestamp_series=ts)

        # Only first 50 rows have label_available_flag=True
        assert meta.row_count <= 50

    def test_split_count_can_exceed_snapshot_count(self, temp_dir):
        n = 200
        ts = pd.Series(pd.date_range("2022-01-01", periods=n, freq="D"))
        # Only first 100 rows have label available
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=90),
            "label_available_flag": [True] * 100 + [False] * 100,
            "target": [0] * n,
        })

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)

        manager = SnapshotManager(temp_dir)
        meta = manager.create_snapshot(df, cutoff, "target", timestamp_series=ts)

        # split counts all 180 rows; snapshot only counts those with flag=True
        assert split.train_count >= meta.row_count


class TestTimezoneAwareNaiveComparison:
    """Timezone-aware timestamps compared with naive cutoff dates."""

    def test_naive_cutoff_with_naive_series(self):
        ts = pd.Series(pd.date_range("2022-01-01", periods=100, freq="D"), name="ts")
        df = pd.DataFrame({"ts": ts, "val": range(100)})

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        cutoff = datetime(2022, 3, 1)  # naive

        split = analysis.split_at_cutoff(cutoff)
        assert split.train_count + split.score_count == 100

    def test_tz_aware_series_with_naive_cutoff_raises_or_works(self):
        ts = pd.Series(
            pd.date_range("2022-01-01", periods=50, freq="D", tz="UTC"),
            name="ts",
        )
        df = pd.DataFrame({"ts": ts, "val": range(50)})

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        cutoff = datetime(2022, 2, 1)  # naive

        # Should either work (by coercion) or raise a clear error
        try:
            split = analysis.split_at_cutoff(cutoff)
            # If it works, counts should still be correct
            assert split.train_count + split.score_count == 50
        except TypeError:
            # Acceptable: explicit failure is better than silent wrong results
            pass

    def test_tz_aware_series_with_tz_aware_cutoff(self):
        ts = pd.Series(
            pd.date_range("2022-01-01", periods=50, freq="D", tz="UTC"),
            name="ts",
        )
        df = pd.DataFrame({"ts": ts, "val": range(50)})

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        cutoff = datetime(2022, 2, 1, tzinfo=timezone.utc)

        split = analysis.split_at_cutoff(cutoff)
        assert split.train_count + split.score_count == 50
        assert split.train_count == 32  # Jan 1 through Feb 1 inclusive (ts <= cutoff)


class TestExactCutoffBoundary:
    """Many rows sharing exact same timestamp as cutoff date."""

    def test_rows_at_exact_cutoff_are_in_train(self):
        # 50 rows all with the exact cutoff timestamp
        cutoff = datetime(2022, 6, 15)
        timestamps = [cutoff] * 50 + [cutoff + timedelta(days=i) for i in range(1, 51)]
        ts = pd.Series(pd.to_datetime(timestamps), name="ts")
        df = pd.DataFrame({"ts": ts, "val": range(100)})

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        split = analysis.split_at_cutoff(cutoff)

        # All 50 rows at exact cutoff should be in train (<=)
        assert split.train_count == 50
        assert split.score_count == 50

    def test_all_rows_same_timestamp_as_cutoff(self):
        cutoff = datetime(2022, 6, 15)
        ts = pd.Series([cutoff] * 100, name="ts")
        df = pd.DataFrame({"ts": ts, "val": range(100)})

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        split = analysis.split_at_cutoff(cutoff)

        # All rows at cutoff → all in train
        assert split.train_count == 100
        assert split.score_count == 0

    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    def test_snapshot_boundary_matches_split_boundary(self, temp_dir):
        cutoff = datetime(2022, 6, 15)
        timestamps = [cutoff - timedelta(days=1)] * 30 + [cutoff] * 40 + [cutoff + timedelta(days=1)] * 30
        ts = pd.Series(pd.to_datetime(timestamps))
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=90),
            "label_available_flag": [True] * 100,
            "target": [0] * 100,
        })

        manager = SnapshotManager(temp_dir)
        meta = manager.create_snapshot(df, cutoff, "target", timestamp_series=ts)

        # 30 before + 40 at cutoff = 70
        assert meta.row_count == 70


class TestObservationWindowMismatch:
    """ScenarioDetector label_window_days vs TimestampConfig observation_window_days."""

    def test_label_window_propagates_to_observation_window(self):
        from customer_retention.stages.temporal import ScenarioDetector

        df = pd.DataFrame({
            "obs_date": pd.date_range("2022-01-01", periods=100, freq="D"),
            "target": [0] * 100,
        })

        detector = ScenarioDetector(label_window_days=60)
        _, config, _ = detector.detect(df, "target")

        assert config.observation_window_days == 60

    def test_different_window_produces_different_label_available(self):
        now = datetime.now()
        feature_ts = now - timedelta(days=100)  # 100 days ago
        df = pd.DataFrame({
            "feature_date": [feature_ts] * 5,
            "event_date": pd.to_datetime([None] * 5),
            "value": range(5),
        })

        # With 90-day window: feature + 90 = 10 days ago → available
        config_90 = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="feature_date",
            label_timestamp_column="event_date",
            observation_window_days=90,
        )
        result_90 = TimestampManager(config_90).ensure_timestamps(df)

        # With 120-day window: feature + 120 = 20 days in future → NOT available
        config_120 = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="feature_date",
            label_timestamp_column="event_date",
            observation_window_days=120,
        )
        result_120 = TimestampManager(config_120).ensure_timestamps(df)

        assert result_90["label_available_flag"].all()
        assert not result_120["label_available_flag"].any()


class TestIndexAlignmentEdgeCases:
    """Series index must match DataFrame index for split_at_cutoff."""

    def test_custom_string_index_preserved(self):
        idx = [f"row_{i}" for i in range(50)]
        ts = pd.Series(
            pd.date_range("2022-01-01", periods=50, freq="D"),
            index=idx, name="ts",
        )
        df = pd.DataFrame({"val": range(50)}, index=idx)

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        split = analysis.split_at_cutoff(datetime(2022, 2, 1))

        all_indices = sorted(
            split.train_df.index.tolist()
            + split.score_df.index.tolist()
            + split.unresolvable_df.index.tolist()
        )
        assert all_indices == sorted(idx)

    def test_non_contiguous_integer_index(self):
        idx = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ts = pd.Series(
            pd.date_range("2022-01-01", periods=10, freq="ME"),
            index=idx, name="ts",
        )
        df = pd.DataFrame({"val": range(10)}, index=idx)

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        split = analysis.split_at_cutoff(datetime(2022, 6, 15))

        assert sorted(
            split.train_df.index.tolist()
            + split.score_df.index.tolist()
        ) == idx

    def test_misaligned_series_index_raises(self):
        ts = pd.Series(
            pd.date_range("2022-01-01", periods=10, freq="D"),
            index=range(100, 110), name="ts",
        )
        df = pd.DataFrame({"val": range(10)}, index=range(0, 10))

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)

        with pytest.raises(Exception):
            analysis.split_at_cutoff(datetime(2022, 1, 5))

    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    def test_snapshot_with_misaligned_series_index(self, temp_dir):
        ts = pd.Series(
            pd.date_range("2022-01-01", periods=10, freq="D"),
            index=range(100, 110),
        )
        df = pd.DataFrame({
            "feature_timestamp": pd.date_range("2022-01-01", periods=10, freq="D"),
            "label_timestamp": pd.date_range("2022-04-01", periods=10, freq="D"),
            "label_available_flag": [True] * 10,
            "target": [0] * 10,
        }, index=range(0, 10))

        manager = SnapshotManager(temp_dir)
        # Misaligned series — should fallback gracefully or produce 0 rows
        meta = manager.create_snapshot(
            df, datetime(2022, 1, 5), "target", timestamp_series=ts
        )
        # Misaligned index means no rows match
        assert meta.row_count == 0


class TestAutoDetectionPicksWrongColumn:
    """After ensure_timestamps() adds synthetic columns, auto-detection
    may pick the wrong column for cutoff analysis."""

    def test_auto_detect_prefers_feature_timestamp_over_original(self):
        df = pd.DataFrame({
            "original_date": pd.date_range("2020-01-01", periods=50, freq="D"),
            "feature_timestamp": pd.date_range("2024-01-01", periods=50, freq="D"),
            "value": range(50),
        })

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df)  # No explicit column

        # Auto-detection should pick "feature_timestamp" first (matches pattern)
        assert analysis.timestamp_column == "feature_timestamp"

    def test_explicit_column_overrides_auto_detection(self):
        df = pd.DataFrame({
            "original_date": pd.date_range("2020-01-01", periods=50, freq="D"),
            "feature_timestamp": pd.date_range("2024-01-01", periods=50, freq="D"),
            "value": range(50),
        })

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_column="original_date")

        assert analysis.timestamp_column == "original_date"
        # Date range should be from original column
        assert analysis.date_range[0].year == 2020

    def test_timestamp_series_overrides_all(self):
        original_ts = pd.Series(
            pd.date_range("2019-01-01", periods=50, freq="D"), name="custom_ts"
        )
        df = pd.DataFrame({
            "feature_timestamp": pd.date_range("2024-01-01", periods=50, freq="D"),
            "value": range(50),
        })

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=original_ts)

        assert analysis.timestamp_column == "custom_ts"
        assert analysis.date_range[0].year == 2019


class TestAllNaTUnresolvableRows:
    """Rows where all datetime source columns are NaT."""

    def test_fully_unresolvable_rows_in_coalesced_series(self):
        df = pd.DataFrame({
            "date_a": pd.to_datetime(["2022-01-01", None, "2022-03-01", None, None]),
            "date_b": pd.to_datetime(["2022-01-15", None, "2022-03-15", None, "2022-05-15"]),
            "value": range(5),
        })

        analyzer = DatetimeOrderAnalyzer()
        coalesced = analyzer.derive_last_action_date(df)

        # Row 1 and 3 have NaT in both columns → unresolvable
        # Row 4 has NaT in date_a but value in date_b → resolvable
        assert coalesced.isna().sum() == 2

    def test_split_at_cutoff_captures_unresolvable(self):
        ts = pd.Series(
            pd.to_datetime(["2022-01-01", None, "2022-03-01", None, "2022-05-01"]),
            name="ts",
        )
        df = pd.DataFrame({"val": range(5)})

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        split = analysis.split_at_cutoff(datetime(2022, 2, 15))

        assert split.unresolvable_count == 2
        assert split.train_count == 1  # Only 2022-01-01
        assert split.score_count == 2  # 2022-03-01 and 2022-05-01

    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    def test_snapshot_silently_drops_unresolvable(self, temp_dir):
        ts = pd.Series(pd.to_datetime(["2022-01-01", None, "2022-03-01", None, "2022-05-01"]))
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=90),
            "label_available_flag": [True] * 5,
            "target": [0] * 5,
        })

        manager = SnapshotManager(temp_dir)
        meta = manager.create_snapshot(
            df, datetime(2022, 6, 1), "target", timestamp_series=ts
        )

        # NaT rows are excluded by ts <= cutoff (NaT comparison is False)
        assert meta.row_count == 3  # Only the 3 non-null rows


class TestLabelAvailableFlagTemporalDrift:
    """label_available_flag depends on datetime.now(), so results change over time."""

    def test_recent_data_has_unavailable_labels(self):
        now = datetime.now()
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="obs_date",
            label_timestamp_column="event_date",
            observation_window_days=90,
        )
        manager = TimestampManager(config)

        # Data from 10 days ago — not enough time for 90-day window
        recent = now - timedelta(days=10)
        df = pd.DataFrame({
            "obs_date": [recent] * 5,
            "event_date": pd.to_datetime([None] * 5),
            "value": range(5),
        })

        result = manager.ensure_timestamps(df)
        assert not result["label_available_flag"].any()

    def test_old_data_always_available(self):
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="obs_date",
            label_timestamp_column="event_date",
            observation_window_days=90,
        )
        manager = TimestampManager(config)

        # Data from 2 years ago — well past any observation window
        old = datetime(2020, 1, 1)
        df = pd.DataFrame({
            "obs_date": [old] * 5,
            "event_date": pd.to_datetime([None] * 5),
            "value": range(5),
        })

        result = manager.ensure_timestamps(df)
        assert result["label_available_flag"].all()

    def test_mixed_recent_and_old_partially_available(self):
        now = datetime.now()
        config = TimestampConfig(
            strategy=TimestampStrategy.PRODUCTION,
            feature_timestamp_column="obs_date",
            label_timestamp_column="event_date",
            observation_window_days=90,
        )
        manager = TimestampManager(config)

        df = pd.DataFrame({
            "obs_date": [
                now - timedelta(days=200),  # old → available
                now - timedelta(days=100),  # old enough → available
                now - timedelta(days=50),   # too recent → NOT available
                now - timedelta(days=10),   # too recent → NOT available
            ],
            "event_date": pd.to_datetime([None, None, None, None]),
        })

        result = manager.ensure_timestamps(df)
        assert result["label_available_flag"].iloc[0] == True
        assert result["label_available_flag"].iloc[1] == True
        assert result["label_available_flag"].iloc[2] == False
        assert result["label_available_flag"].iloc[3] == False


class TestCutoffBeforeAllData:
    """When cutoff is before all timestamps, snapshot should be empty."""

    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    def test_cutoff_before_all_gives_empty_train(self):
        ts = pd.Series(
            pd.date_range("2022-06-01", periods=100, freq="D"), name="ts"
        )
        df = pd.DataFrame({"val": range(100)})

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        split = analysis.split_at_cutoff(datetime(2022, 1, 1))

        assert split.train_count == 0
        assert split.score_count == 100

    def test_cutoff_before_all_gives_empty_snapshot(self, temp_dir):
        ts = pd.Series(pd.date_range("2022-06-01", periods=50, freq="D"))
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=90),
            "label_available_flag": [True] * 50,
            "target": [0] * 50,
        })

        manager = SnapshotManager(temp_dir)
        meta = manager.create_snapshot(
            df, datetime(2020, 1, 1), "target", timestamp_series=ts
        )
        assert meta.row_count == 0


class TestDuplicateTimestampHandling:
    """Datasets with many rows sharing timestamps (event-level data)."""

    def test_daily_granularity_many_events_per_day(self):
        # 100 customers, each with one event per day for 10 days
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        ts = pd.Series(np.tile(dates, 100), name="ts")
        df = pd.DataFrame({"customer": np.repeat(range(100), 10), "val": range(1000)})

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        cutoff = datetime(2022, 1, 5)  # After 5 days

        split = analysis.split_at_cutoff(cutoff)
        # 5 days * 100 customers = 500 train
        assert split.train_count == 500
        assert split.score_count == 500

    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)

    def test_snapshot_matches_split_with_duplicates(self, temp_dir):
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        ts = pd.Series(np.tile(dates, 50))
        df = pd.DataFrame({
            "feature_timestamp": ts,
            "label_timestamp": ts + pd.Timedelta(days=90),
            "label_available_flag": [True] * 500,
            "target": [0] * 500,
        })

        cutoff = datetime(2022, 1, 5)
        manager = SnapshotManager(temp_dir)
        meta = manager.create_snapshot(df, cutoff, "target", timestamp_series=ts)

        analyzer = CutoffAnalyzer()
        analysis = analyzer.analyze(df, timestamp_series=ts)
        split = analysis.split_at_cutoff(cutoff)

        assert meta.row_count == split.train_count


class TestCoalescedSeriesColumnContamination:
    """derive_last_action_date on prepared df picks up synthetic columns."""

    def test_prepared_df_has_more_datetime_columns(self):
        raw_df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "activity_date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
            "target": [0, 1, 0],
        })

        analyzer = DatetimeOrderAnalyzer()
        raw_series = analyzer.derive_last_action_date(raw_df)

        # Simulate what prepare_from_raw does
        prepared = raw_df.copy()
        prepared["feature_timestamp"] = prepared["activity_date"]
        prepared["label_timestamp"] = prepared["activity_date"] + pd.Timedelta(days=90)

        prepared_series = analyzer.derive_last_action_date(prepared)

        # Prepared series coalesces MORE columns (including label_timestamp)
        # This can shift values forward
        assert not raw_series.equals(prepared_series)

    def test_using_raw_series_on_prepared_df_gives_consistent_results(self):
        raw_df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "activity_date": pd.to_datetime(["2022-01-01", "2022-06-01", "2022-12-01"]),
            "target": [0, 1, 0],
        })

        analyzer = DatetimeOrderAnalyzer()
        raw_series = analyzer.derive_last_action_date(raw_df)

        cutoff_analyzer = CutoffAnalyzer()
        analysis = cutoff_analyzer.analyze(raw_df, timestamp_series=raw_series)
        cutoff = analysis.suggest_cutoff(train_ratio=0.9)
        split = analysis.split_at_cutoff(cutoff)

        # Simulated prepared df (same index)
        prepared = raw_df.copy()
        prepared["feature_timestamp"] = prepared["activity_date"]
        prepared["label_timestamp"] = prepared["activity_date"] + pd.Timedelta(days=90)
        prepared["label_available_flag"] = True

        # Using RAW series for snapshot filtering → consistent with analysis
        mask = (raw_series <= cutoff) & prepared["label_available_flag"]
        snapshot_count = mask.sum()

        assert snapshot_count == split.train_count
