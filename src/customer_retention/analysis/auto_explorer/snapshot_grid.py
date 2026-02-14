from __future__ import annotations

import datetime as _dt
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from customer_retention.analysis.auto_explorer.project_context import (
    CadenceInterval,
    DatasetRegistryEntry,
    IntentConfig,
)
from customer_retention.core.config.column_config import DatasetGranularity


class GridAdjustmentMode(str, Enum):
    NO_ADJUSTMENTS = "no_adjustments"
    ALLOW_ADJUSTMENTS = "allow_adjustments"


_CADENCE_DAYS = {
    CadenceInterval.DAILY: 1,
    CadenceInterval.WEEKLY: 7,
    CadenceInterval.BIWEEKLY: 14,
    CadenceInterval.MONTHLY: 30,
}


def compute_boundaries(
    data_spans: list[tuple[str, str]],
    observation_window_days: int,
    purge_gap_days: int,
    label_window_days: int,
) -> tuple[Optional[str], Optional[str]]:
    if not data_spans:
        return None, None
    starts = [_dt.date.fromisoformat(s) for s, _ in data_spans]
    ends = [_dt.date.fromisoformat(e) for _, e in data_spans]
    grid_start = min(starts) + _dt.timedelta(days=observation_window_days)
    grid_end = max(ends) - _dt.timedelta(days=purge_gap_days + label_window_days)
    if grid_end < grid_start:
        return None, None
    return grid_start.isoformat(), grid_end.isoformat()


class DatasetGridVote(BaseModel):
    dataset_name: str
    granularity: DatasetGranularity
    voted: bool = False
    suggested_cadence: Optional[CadenceInterval] = None
    suggested_start: Optional[str] = None
    data_span_start: Optional[str] = None
    data_span_end: Optional[str] = None


class SnapshotGrid(BaseModel):
    mode: GridAdjustmentMode = GridAdjustmentMode.NO_ADJUSTMENTS
    cadence_interval: CadenceInterval
    observation_window_days: int
    grid_start: Optional[str] = None
    grid_end: Optional[str] = None
    grid_dates: list[str] = Field(default_factory=list)
    dataset_votes: dict[str, DatasetGridVote] = Field(default_factory=dict)
    locked: bool = False

    @classmethod
    def from_intent(
        cls,
        intent: IntentConfig,
        datasets: dict[str, DatasetRegistryEntry],
        mode: GridAdjustmentMode = GridAdjustmentMode.NO_ADJUSTMENTS,
        fingerprints: Optional[dict] = None,
    ) -> SnapshotGrid:
        from customer_retention.analysis.auto_explorer.dataset_fingerprinter import DatasetFingerprint

        votes: dict[str, DatasetGridVote] = {}
        for name, entry in datasets.items():
            auto_voted = entry.granularity != DatasetGranularity.EVENT_LEVEL
            votes[name] = DatasetGridVote(
                dataset_name=name,
                granularity=entry.granularity or DatasetGranularity.UNKNOWN,
                voted=auto_voted,
            )

        grid_start = None
        grid_end = None
        if fingerprints:
            spans = [
                (fp.data_start, fp.data_end)
                for fp in fingerprints.values()
                if isinstance(fp, DatasetFingerprint) and fp.data_start and fp.data_end
            ]
            if spans:
                grid_start, grid_end = compute_boundaries(
                    spans,
                    observation_window_days=intent.observation_window_days,
                    purge_gap_days=intent.purge_gap_days,
                    label_window_days=intent.label_window_days,
                )

        return cls(
            mode=mode,
            cadence_interval=intent.cadence_interval,
            observation_window_days=intent.observation_window_days,
            dataset_votes=votes,
            grid_start=grid_start,
            grid_end=grid_end,
        )

    def cadence_to_days(self) -> int:
        return _CADENCE_DAYS[self.cadence_interval]

    def generate_grid_dates(self) -> None:
        if self.grid_start is None or self.grid_end is None:
            self.grid_dates = []
            return
        start = _dt.date.fromisoformat(self.grid_start)
        end = _dt.date.fromisoformat(self.grid_end)
        if end < start:
            self.grid_dates = []
            return
        step = _dt.timedelta(days=self.cadence_to_days())
        dates: list[str] = []
        current = start
        while current <= end:
            dates.append(current.isoformat())
            current += step
        self.grid_dates = dates

    def record_vote(self, name: str, vote: DatasetGridVote) -> None:
        if self.locked:
            return
        self.dataset_votes[name] = vote

    def is_ready_for_aggregation(self) -> tuple[bool, list[str]]:
        if self.mode == GridAdjustmentMode.NO_ADJUSTMENTS:
            return True, []
        missing = [
            name
            for name, vote in self.dataset_votes.items()
            if vote.granularity == DatasetGranularity.EVENT_LEVEL and not vote.voted
        ]
        return len(missing) == 0, missing

    def compute_boundaries_from_votes(
        self,
        purge_gap_days: int,
        label_window_days: int,
    ) -> tuple[Optional[str], Optional[str]]:
        spans = [
            (v.data_span_start, v.data_span_end)
            for v in self.dataset_votes.values()
            if v.data_span_start and v.data_span_end
        ]
        return compute_boundaries(
            spans,
            observation_window_days=self.observation_window_days,
            purge_gap_days=purge_gap_days,
            label_window_days=label_window_days,
        )

    def lock(self, purge_gap_days: int = 0, label_window_days: int = 0) -> None:
        if self.grid_start is None or self.grid_end is None:
            start, end = self.compute_boundaries_from_votes(purge_gap_days, label_window_days)
            if start is not None and end is not None:
                self.grid_start = start
                self.grid_end = end
        self.locked = True
        if not self.grid_dates and self.grid_start is not None and self.grid_end is not None:
            self.generate_grid_dates()

    def estimated_grid_size(self, n_entities: int) -> int:
        return n_entities * len(self.grid_dates)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> SnapshotGrid:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Snapshot grid file not found: {p}")
        data = yaml.safe_load(p.read_text())
        return cls.model_validate(data)
