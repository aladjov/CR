from __future__ import annotations

import pytest
import yaml

from customer_retention.analysis.auto_explorer.dataset_fingerprinter import DatasetFingerprint
from customer_retention.analysis.auto_explorer.project_context import (
    CadenceInterval,
    DatasetRegistryEntry,
    IntentConfig,
)
from customer_retention.analysis.auto_explorer.snapshot_grid import (
    DatasetGridVote,
    GridAdjustmentMode,
    SnapshotGrid,
    compute_boundaries,
)
from customer_retention.core.config.column_config import DatasetGranularity


def _minimal_vote(**overrides):
    defaults = dict(
        dataset_name="transactions",
        granularity=DatasetGranularity.EVENT_LEVEL,
    )
    defaults.update(overrides)
    return DatasetGridVote(**defaults)


def _minimal_grid(**overrides):
    defaults = dict(
        cadence_interval=CadenceInterval.WEEKLY,
        observation_window_days=270,
    )
    defaults.update(overrides)
    return SnapshotGrid(**defaults)


def _minimal_intent(**overrides):
    defaults = dict(
        cadence_interval=CadenceInterval.WEEKLY,
        observation_window_days=270,
    )
    defaults.update(overrides)
    return IntentConfig(**defaults)


def _minimal_entry(name, granularity):
    return DatasetRegistryEntry(
        name=name,
        path=f"data/{name}.csv",
        storage_format="csv",
        granularity=granularity,
    )


class TestGridAdjustmentModeEnum:
    def test_values(self):
        assert GridAdjustmentMode.NO_ADJUSTMENTS.value == "no_adjustments"
        assert GridAdjustmentMode.ALLOW_ADJUSTMENTS.value == "allow_adjustments"

    def test_from_string_round_trip(self):
        for mode in GridAdjustmentMode:
            assert GridAdjustmentMode(mode.value) == mode


class TestDatasetGridVoteCreation:
    def test_minimal(self):
        vote = _minimal_vote()
        assert vote.dataset_name == "transactions"
        assert vote.granularity == DatasetGranularity.EVENT_LEVEL

    def test_all_fields(self):
        vote = DatasetGridVote(
            dataset_name="events",
            granularity=DatasetGranularity.EVENT_LEVEL,
            voted=True,
            suggested_cadence=CadenceInterval.DAILY,
            suggested_start="2024-01-01",
            data_span_start="2023-06-01",
            data_span_end="2024-06-01",
        )
        assert vote.voted is True
        assert vote.suggested_cadence == CadenceInterval.DAILY
        assert vote.suggested_start == "2024-01-01"
        assert vote.data_span_start == "2023-06-01"
        assert vote.data_span_end == "2024-06-01"

    def test_defaults(self):
        vote = _minimal_vote()
        assert vote.voted is False
        assert vote.suggested_cadence is None
        assert vote.suggested_start is None
        assert vote.data_span_start is None
        assert vote.data_span_end is None


class TestSnapshotGridCreation:
    def test_minimal(self):
        grid = _minimal_grid()
        assert grid.cadence_interval == CadenceInterval.WEEKLY
        assert grid.observation_window_days == 270

    def test_with_dates(self):
        grid = _minimal_grid(grid_start="2024-01-01", grid_end="2024-06-01")
        assert grid.grid_start == "2024-01-01"
        assert grid.grid_end == "2024-06-01"

    def test_defaults(self):
        grid = _minimal_grid()
        assert grid.mode == GridAdjustmentMode.NO_ADJUSTMENTS
        assert grid.grid_start is None
        assert grid.grid_end is None
        assert grid.grid_dates == []
        assert grid.dataset_votes == {}
        assert grid.locked is False


class TestFromIntent:
    def test_extracts_cadence_and_observation_window(self):
        intent = _minimal_intent(
            cadence_interval=CadenceInterval.MONTHLY,
            observation_window_days=180,
        )
        grid = SnapshotGrid.from_intent(intent, datasets={})
        assert grid.cadence_interval == CadenceInterval.MONTHLY
        assert grid.observation_window_days == 180

    def test_entity_datasets_auto_voted(self):
        datasets = {
            "profiles": _minimal_entry("profiles", DatasetGranularity.ENTITY_LEVEL),
        }
        grid = SnapshotGrid.from_intent(_minimal_intent(), datasets=datasets)
        assert grid.dataset_votes["profiles"].voted is True

    def test_event_datasets_not_voted(self):
        datasets = {
            "transactions": _minimal_entry("transactions", DatasetGranularity.EVENT_LEVEL),
        }
        grid = SnapshotGrid.from_intent(_minimal_intent(), datasets=datasets)
        assert grid.dataset_votes["transactions"].voted is False

    def test_mode_passthrough(self):
        grid = SnapshotGrid.from_intent(
            _minimal_intent(),
            datasets={},
            mode=GridAdjustmentMode.ALLOW_ADJUSTMENTS,
        )
        assert grid.mode == GridAdjustmentMode.ALLOW_ADJUSTMENTS

    def test_mixed_granularities(self):
        datasets = {
            "profiles": _minimal_entry("profiles", DatasetGranularity.ENTITY_LEVEL),
            "transactions": _minimal_entry("transactions", DatasetGranularity.EVENT_LEVEL),
            "tickets": _minimal_entry("tickets", DatasetGranularity.EVENT_LEVEL),
        }
        grid = SnapshotGrid.from_intent(_minimal_intent(), datasets=datasets)
        assert grid.dataset_votes["profiles"].voted is True
        assert grid.dataset_votes["transactions"].voted is False
        assert grid.dataset_votes["tickets"].voted is False

    def test_empty_datasets(self):
        grid = SnapshotGrid.from_intent(_minimal_intent(), datasets={})
        assert grid.dataset_votes == {}

    def test_unknown_granularity_treated_as_entity(self):
        datasets = {
            "mystery": _minimal_entry("mystery", DatasetGranularity.UNKNOWN),
        }
        grid = SnapshotGrid.from_intent(_minimal_intent(), datasets=datasets)
        assert grid.dataset_votes["mystery"].voted is True


class TestCadenceToDays:
    def test_daily(self):
        grid = _minimal_grid(cadence_interval=CadenceInterval.DAILY)
        assert grid.cadence_to_days() == 1

    def test_weekly(self):
        grid = _minimal_grid(cadence_interval=CadenceInterval.WEEKLY)
        assert grid.cadence_to_days() == 7

    def test_biweekly(self):
        grid = _minimal_grid(cadence_interval=CadenceInterval.BIWEEKLY)
        assert grid.cadence_to_days() == 14

    def test_monthly(self):
        grid = _minimal_grid(cadence_interval=CadenceInterval.MONTHLY)
        assert grid.cadence_to_days() == 30

    def test_loop_invariant_all_values(self):
        expected = {"daily": 1, "weekly": 7, "biweekly": 14, "monthly": 30}
        for cadence in CadenceInterval:
            grid = _minimal_grid(cadence_interval=cadence)
            assert grid.cadence_to_days() == expected[cadence.value]


class TestGenerateGridDates:
    def test_weekly_range(self):
        grid = _minimal_grid(
            cadence_interval=CadenceInterval.WEEKLY,
            grid_start="2024-01-01",
            grid_end="2024-01-29",
        )
        grid.generate_grid_dates()
        assert grid.grid_dates == [
            "2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29",
        ]

    def test_daily_range(self):
        grid = _minimal_grid(
            cadence_interval=CadenceInterval.DAILY,
            grid_start="2024-03-01",
            grid_end="2024-03-04",
        )
        grid.generate_grid_dates()
        assert grid.grid_dates == [
            "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04",
        ]

    def test_monthly_range(self):
        grid = _minimal_grid(
            cadence_interval=CadenceInterval.MONTHLY,
            grid_start="2024-01-01",
            grid_end="2024-04-01",
        )
        grid.generate_grid_dates()
        assert len(grid.grid_dates) >= 3
        assert grid.grid_dates[0] == "2024-01-01"
        assert all(d <= "2024-04-01" for d in grid.grid_dates)

    def test_none_start_end_returns_empty(self):
        grid = _minimal_grid()
        grid.generate_grid_dates()
        assert grid.grid_dates == []

    def test_end_before_start_returns_empty(self):
        grid = _minimal_grid(grid_start="2024-06-01", grid_end="2024-01-01")
        grid.generate_grid_dates()
        assert grid.grid_dates == []

    def test_same_start_end_single_date(self):
        grid = _minimal_grid(grid_start="2024-03-15", grid_end="2024-03-15")
        grid.generate_grid_dates()
        assert grid.grid_dates == ["2024-03-15"]

    def test_updates_in_place(self):
        grid = _minimal_grid(
            cadence_interval=CadenceInterval.WEEKLY,
            grid_start="2024-01-01",
            grid_end="2024-01-15",
        )
        grid.generate_grid_dates()
        first_result = list(grid.grid_dates)
        assert len(first_result) == 3

        grid.grid_start = "2024-01-01"
        grid.grid_end = "2024-01-08"
        grid.generate_grid_dates()
        assert len(grid.grid_dates) == 2
        assert grid.grid_dates != first_result

    def test_last_date_lte_grid_end(self):
        grid = _minimal_grid(
            cadence_interval=CadenceInterval.WEEKLY,
            grid_start="2024-01-01",
            grid_end="2024-01-10",
        )
        grid.generate_grid_dates()
        assert grid.grid_dates[-1] <= "2024-01-10"


class TestRecordVote:
    def test_basic_record(self):
        grid = _minimal_grid()
        vote = _minimal_vote(voted=True)
        grid.record_vote("transactions", vote)
        assert grid.dataset_votes["transactions"].voted is True

    def test_skips_when_locked(self):
        grid = _minimal_grid(locked=True)
        grid.dataset_votes["transactions"] = _minimal_vote(voted=False)
        vote = _minimal_vote(voted=True, data_span_start="2024-01-01")
        grid.record_vote("transactions", vote)
        assert grid.dataset_votes["transactions"].voted is False
        assert grid.dataset_votes["transactions"].data_span_start is None

    def test_overwrites_existing(self):
        grid = _minimal_grid()
        vote1 = _minimal_vote(voted=False)
        grid.record_vote("transactions", vote1)
        assert grid.dataset_votes["transactions"].voted is False

        vote2 = _minimal_vote(voted=True, data_span_start="2024-01-01")
        grid.record_vote("transactions", vote2)
        assert grid.dataset_votes["transactions"].voted is True
        assert grid.dataset_votes["transactions"].data_span_start == "2024-01-01"

    def test_with_suggestions(self):
        grid = _minimal_grid()
        vote = _minimal_vote(
            voted=True,
            suggested_cadence=CadenceInterval.DAILY,
            suggested_start="2024-01-15",
            data_span_start="2023-06-01",
            data_span_end="2024-06-30",
        )
        grid.record_vote("transactions", vote)
        stored = grid.dataset_votes["transactions"]
        assert stored.suggested_cadence == CadenceInterval.DAILY
        assert stored.suggested_start == "2024-01-15"


class TestIsReadyForAggregation:
    def test_no_adjustments_always_ready(self):
        grid = _minimal_grid(mode=GridAdjustmentMode.NO_ADJUSTMENTS)
        grid.dataset_votes["events"] = _minimal_vote(voted=False)
        ready, missing = grid.is_ready_for_aggregation()
        assert ready is True
        assert missing == []

    def test_allow_all_voted(self):
        grid = _minimal_grid(mode=GridAdjustmentMode.ALLOW_ADJUSTMENTS)
        grid.dataset_votes["events"] = _minimal_vote(
            granularity=DatasetGranularity.EVENT_LEVEL, voted=True,
        )
        ready, missing = grid.is_ready_for_aggregation()
        assert ready is True
        assert missing == []

    def test_allow_some_missing(self):
        grid = _minimal_grid(mode=GridAdjustmentMode.ALLOW_ADJUSTMENTS)
        grid.dataset_votes["events1"] = _minimal_vote(
            dataset_name="events1", granularity=DatasetGranularity.EVENT_LEVEL, voted=True,
        )
        grid.dataset_votes["events2"] = _minimal_vote(
            dataset_name="events2", granularity=DatasetGranularity.EVENT_LEVEL, voted=False,
        )
        ready, missing = grid.is_ready_for_aggregation()
        assert ready is False
        assert "events2" in missing

    def test_entity_datasets_dont_block(self):
        grid = _minimal_grid(mode=GridAdjustmentMode.ALLOW_ADJUSTMENTS)
        grid.dataset_votes["profiles"] = _minimal_vote(
            dataset_name="profiles",
            granularity=DatasetGranularity.ENTITY_LEVEL,
            voted=False,
        )
        grid.dataset_votes["events"] = _minimal_vote(
            dataset_name="events",
            granularity=DatasetGranularity.EVENT_LEVEL,
            voted=True,
        )
        ready, missing = grid.is_ready_for_aggregation()
        assert ready is True
        assert missing == []

    def test_empty_votes(self):
        grid = _minimal_grid(mode=GridAdjustmentMode.ALLOW_ADJUSTMENTS)
        ready, missing = grid.is_ready_for_aggregation()
        assert ready is True
        assert missing == []


class TestLock:
    def test_sets_flag(self):
        grid = _minimal_grid()
        assert grid.locked is False
        grid.lock()
        assert grid.locked is True

    def test_generates_dates_when_start_end_set(self):
        grid = _minimal_grid(
            cadence_interval=CadenceInterval.WEEKLY,
            grid_start="2024-01-01",
            grid_end="2024-01-15",
        )
        assert grid.grid_dates == []
        grid.lock()
        assert len(grid.grid_dates) == 3

    def test_record_after_lock_is_noop(self):
        grid = _minimal_grid()
        grid.lock()
        grid.record_vote("events", _minimal_vote(voted=True))
        assert "events" not in grid.dataset_votes

    def test_lock_without_dates_stays_empty(self):
        grid = _minimal_grid()
        grid.lock()
        assert grid.grid_dates == []


class TestEstimatedGridSize:
    def test_basic(self):
        grid = _minimal_grid(grid_dates=["2024-01-01", "2024-01-08", "2024-01-15"])
        assert grid.estimated_grid_size(100) == 300

    def test_empty_grid_returns_zero(self):
        grid = _minimal_grid()
        assert grid.estimated_grid_size(100) == 0


class TestSerialization:
    def test_save_creates_file(self, tmp_path):
        grid = _minimal_grid()
        path = tmp_path / "grid.yaml"
        grid.save(path)
        assert path.exists()

    def test_yaml_structure(self, tmp_path):
        grid = _minimal_grid(
            grid_start="2024-01-01",
            grid_end="2024-06-01",
        )
        path = tmp_path / "grid.yaml"
        grid.save(path)
        data = yaml.safe_load(path.read_text())
        assert data["cadence_interval"] == "weekly"
        assert data["observation_window_days"] == 270
        assert data["grid_start"] == "2024-01-01"
        assert data["mode"] == "no_adjustments"

    def test_load(self, tmp_path):
        grid = _minimal_grid(grid_start="2024-01-01", grid_end="2024-06-01")
        path = tmp_path / "grid.yaml"
        grid.save(path)
        loaded = SnapshotGrid.load(path)
        assert loaded.cadence_interval == CadenceInterval.WEEKLY
        assert loaded.grid_start == "2024-01-01"

    def test_round_trip(self, tmp_path):
        grid = _minimal_grid(
            grid_start="2024-01-01",
            grid_end="2024-06-01",
            mode=GridAdjustmentMode.ALLOW_ADJUSTMENTS,
        )
        path = tmp_path / "grid.yaml"
        grid.save(path)
        loaded = SnapshotGrid.load(path)
        assert loaded.mode == GridAdjustmentMode.ALLOW_ADJUSTMENTS
        assert loaded.cadence_interval == grid.cadence_interval
        assert loaded.observation_window_days == grid.observation_window_days
        assert loaded.grid_start == grid.grid_start
        assert loaded.grid_end == grid.grid_end

    def test_round_trip_with_votes(self, tmp_path):
        grid = _minimal_grid()
        grid.dataset_votes["events"] = _minimal_vote(
            voted=True,
            data_span_start="2023-06-01",
            data_span_end="2024-06-01",
        )
        path = tmp_path / "grid.yaml"
        grid.save(path)
        loaded = SnapshotGrid.load(path)
        assert "events" in loaded.dataset_votes
        assert loaded.dataset_votes["events"].voted is True
        assert loaded.dataset_votes["events"].data_span_start == "2023-06-01"

    def test_round_trip_locked(self, tmp_path):
        grid = _minimal_grid(
            grid_start="2024-01-01",
            grid_end="2024-01-15",
            cadence_interval=CadenceInterval.WEEKLY,
        )
        grid.lock()
        path = tmp_path / "grid.yaml"
        grid.save(path)
        loaded = SnapshotGrid.load(path)
        assert loaded.locked is True
        assert len(loaded.grid_dates) == 3

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SnapshotGrid.load(tmp_path / "missing.yaml")

    def test_enums_as_strings(self, tmp_path):
        grid = _minimal_grid(mode=GridAdjustmentMode.ALLOW_ADJUSTMENTS)
        grid.dataset_votes["ev"] = _minimal_vote(
            voted=True,
            suggested_cadence=CadenceInterval.DAILY,
        )
        path = tmp_path / "grid.yaml"
        grid.save(path)
        raw = yaml.safe_load(path.read_text())
        assert raw["mode"] == "allow_adjustments"
        assert raw["cadence_interval"] == "weekly"
        assert raw["dataset_votes"]["ev"]["granularity"] == "event_level"
        assert raw["dataset_votes"]["ev"]["suggested_cadence"] == "daily"

    def test_creates_parent_dirs(self, tmp_path):
        grid = _minimal_grid()
        path = tmp_path / "nested" / "deep" / "grid.yaml"
        grid.save(path)
        assert path.exists()


class TestComputeBoundaries:
    def test_training_safe_formula(self):
        spans = [("2022-01-01", "2024-12-31")]
        obs = 270
        purge = 104
        label = 90
        start, end = compute_boundaries(spans, obs, purge, label)
        assert start == "2022-09-28"
        assert end == "2024-06-20"

    def test_multiple_datasets_uses_min_start_max_end(self):
        spans = [("2023-01-01", "2024-06-01"), ("2022-06-01", "2024-12-31")]
        start, end = compute_boundaries(spans, observation_window_days=270, purge_gap_days=104, label_window_days=90)
        from datetime import date, timedelta
        expected_start = str(date.fromisoformat("2022-06-01") + timedelta(days=270))
        expected_end = str(date.fromisoformat("2024-12-31") - timedelta(days=104 + 90))
        assert start == expected_start
        assert end == expected_end

    def test_empty_spans_returns_none(self):
        start, end = compute_boundaries([], observation_window_days=270, purge_gap_days=0, label_window_days=0)
        assert start is None
        assert end is None

    def test_insufficient_data_returns_none(self):
        spans = [("2024-01-01", "2024-02-01")]
        start, end = compute_boundaries(spans, observation_window_days=270, purge_gap_days=104, label_window_days=90)
        assert start is None
        assert end is None

    def test_exact_boundary_single_date(self):
        from datetime import date, timedelta
        obs, purge, label = 10, 5, 5
        base = date(2024, 1, 1)
        data_end = str(base + timedelta(days=obs + purge + label))
        spans = [("2024-01-01", data_end)]
        start, end = compute_boundaries(spans, observation_window_days=obs, purge_gap_days=purge, label_window_days=label)
        assert start == end
        assert start is not None

    def test_purge_zero_label_zero(self):
        spans = [("2023-01-01", "2024-12-31")]
        start, end = compute_boundaries(spans, observation_window_days=270, purge_gap_days=0, label_window_days=0)
        assert end == "2024-12-31"
        from datetime import date, timedelta
        assert start == str(date.fromisoformat("2023-01-01") + timedelta(days=270))


class TestComputeBoundariesFromVotes:
    def test_computes_from_recorded_votes(self):
        grid = _minimal_grid()
        grid.dataset_votes["events"] = _minimal_vote(
            voted=True, data_span_start="2022-01-01", data_span_end="2024-12-31",
        )
        start, end = grid.compute_boundaries_from_votes(purge_gap_days=104, label_window_days=90)
        assert start is not None
        assert end is not None

    def test_ignores_votes_without_spans(self):
        grid = _minimal_grid()
        grid.dataset_votes["events"] = _minimal_vote(
            voted=True, data_span_start="2022-01-01", data_span_end="2024-12-31",
        )
        grid.dataset_votes["profiles"] = _minimal_vote(
            dataset_name="profiles", granularity=DatasetGranularity.ENTITY_LEVEL, voted=True,
        )
        start, end = grid.compute_boundaries_from_votes(purge_gap_days=104, label_window_days=90)
        assert start is not None
        assert end is not None

    def test_no_votes_returns_none(self):
        grid = _minimal_grid()
        start, end = grid.compute_boundaries_from_votes(purge_gap_days=0, label_window_days=0)
        assert start is None
        assert end is None


def _fp(name, data_start=None, data_end=None, granularity=DatasetGranularity.EVENT_LEVEL):
    return DatasetFingerprint(
        name=name,
        row_count=100,
        column_count=5,
        entity_columns=["customer_id"],
        time_columns=["event_timestamp"] if data_start else [],
        target_candidates=[],
        granularity=granularity,
        data_start=data_start,
        data_end=data_end,
    )


class TestFromIntentWithFingerprints:
    def test_fingerprints_set_boundaries(self):
        intent = _minimal_intent(
            observation_window_days=270, purge_gap_days=104, label_window_days=90,
        )
        datasets = {
            "events": _minimal_entry("events", DatasetGranularity.EVENT_LEVEL),
        }
        fps = {"events": _fp("events", data_start="2020-01-01", data_end="2024-12-31")}
        grid = SnapshotGrid.from_intent(intent, datasets, fingerprints=fps)
        assert grid.grid_start is not None
        assert grid.grid_end is not None

    def test_no_fingerprints_boundaries_none(self):
        intent = _minimal_intent()
        datasets = {
            "events": _minimal_entry("events", DatasetGranularity.EVENT_LEVEL),
        }
        grid = SnapshotGrid.from_intent(intent, datasets)
        assert grid.grid_start is None
        assert grid.grid_end is None

    def test_fingerprints_without_dates_boundaries_none(self):
        intent = _minimal_intent()
        datasets = {
            "profiles": _minimal_entry("profiles", DatasetGranularity.ENTITY_LEVEL),
        }
        fps = {"profiles": _fp("profiles")}
        grid = SnapshotGrid.from_intent(intent, datasets, fingerprints=fps)
        assert grid.grid_start is None
        assert grid.grid_end is None

    def test_mixed_fingerprints(self):
        intent = _minimal_intent(
            observation_window_days=270, purge_gap_days=104, label_window_days=90,
        )
        datasets = {
            "events": _minimal_entry("events", DatasetGranularity.EVENT_LEVEL),
            "profiles": _minimal_entry("profiles", DatasetGranularity.ENTITY_LEVEL),
        }
        fps = {
            "events": _fp("events", data_start="2020-01-01", data_end="2024-12-31"),
            "profiles": _fp("profiles"),
        }
        grid = SnapshotGrid.from_intent(intent, datasets, fingerprints=fps)
        assert grid.grid_start is not None
        assert grid.grid_end is not None


class TestLockWithBoundaryFallback:
    def test_lock_computes_from_votes_when_boundaries_none(self):
        grid = _minimal_grid()
        grid.dataset_votes["events"] = _minimal_vote(
            voted=True, data_span_start="2020-01-01", data_span_end="2024-12-31",
        )
        grid.lock(purge_gap_days=104, label_window_days=90)
        assert grid.locked is True
        assert grid.grid_start is not None
        assert grid.grid_end is not None
        assert len(grid.grid_dates) > 0

    def test_lock_preserves_existing_boundaries(self):
        grid = _minimal_grid(grid_start="2024-01-01", grid_end="2024-06-01")
        grid.dataset_votes["events"] = _minimal_vote(
            voted=True, data_span_start="2020-01-01", data_span_end="2024-12-31",
        )
        grid.lock(purge_gap_days=104, label_window_days=90)
        assert grid.grid_start == "2024-01-01"
        assert grid.grid_end == "2024-06-01"

    def test_lock_default_params_backward_compatible(self):
        grid = _minimal_grid()
        grid.lock()
        assert grid.locked is True

    def test_lock_insufficient_data_no_crash(self):
        grid = _minimal_grid()
        grid.dataset_votes["events"] = _minimal_vote(
            voted=True, data_span_start="2024-01-01", data_span_end="2024-02-01",
        )
        grid.lock(purge_gap_days=104, label_window_days=90)
        assert grid.locked is True
        assert grid.grid_dates == []
