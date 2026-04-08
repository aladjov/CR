from __future__ import annotations

import datetime as _dt
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from customer_retention.analysis.auto_explorer.project_context import (
    CadenceInterval,
    IntentConfig,
)
from customer_retention.core.compat.remote_path import RemotePath
from customer_retention.core.config.column_config import DatasetGranularity


def _coerce_path(path):
    if isinstance(path, (Path, RemotePath)):
        return path
    return Path(str(path))


def _is_fuse_volume(path) -> bool:
    return str(path).startswith("/Volumes/")


def _atomic_write_text(path, content: str) -> None:
    if isinstance(path, Path) and not _is_fuse_volume(path):
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_text(content)
        os.replace(str(tmp), str(path))
    else:
        path.write_text(content)


def _cap_grid_dates(dates: list[str], max_dates_override: Optional[int] = None) -> list[str]:
    max_dates: Optional[int] = max_dates_override
    if max_dates is None:
        raw = os.environ.get("CR_GRID_MAX_DATES")
        if raw:
            max_dates = int(raw)
    if max_dates is None or max_dates < 2 or len(dates) <= max_dates:
        return dates
    step = (len(dates) - 1) / (max_dates - 1)
    indices = [round(i * step) for i in range(max_dates)]
    return [dates[i] for i in indices]


class GridAdjustmentMode(str, Enum):
    NO_ADJUSTMENTS = "no_adjustments"
    ALLOW_ADJUSTMENTS = "allow_adjustments"


CADENCE_DAYS = {
    CadenceInterval.DAILY: 1,
    CadenceInterval.WEEKLY: 7,
    CadenceInterval.BIWEEKLY: 14,
    CadenceInterval.MONTHLY: 30,
    CadenceInterval.BIMONTHLY: 60,
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
    purge_gap_days: int = 0
    label_window_days: int = 0
    grid_start: Optional[str] = None
    grid_end: Optional[str] = None
    grid_dates: list[str] = Field(default_factory=list)
    dataset_votes: dict[str, DatasetGridVote] = Field(default_factory=dict)
    locked: bool = False
    max_grid_dates: Optional[int] = None
    lookback_periods: Optional[int] = None

    @classmethod
    def from_intent(
        cls,
        intent: IntentConfig,
        granularities: dict[str, DatasetGranularity],
        mode: GridAdjustmentMode = GridAdjustmentMode.NO_ADJUSTMENTS,
        fingerprints: Optional[dict] = None,
    ) -> SnapshotGrid:
        """Build a snapshot grid from intent + per-dataset granularities.

        Decoupled from the dataset registry so the grid can be built BEFORE
        the registry exists in NB00 (after intent / semantics, before relationship
        scaffold + registry build). This lets preprocessing cells that depend
        on the grid (e.g. SCD history reconstruction) run in the same
        block as lifecycle enrichment, with their changes naturally captured
        by the later registry build and project_context save.
        """
        from customer_retention.analysis.auto_explorer.dataset_fingerprinter import DatasetFingerprint

        votes: dict[str, DatasetGridVote] = {}
        for name, granularity in granularities.items():
            resolved = granularity or DatasetGranularity.UNKNOWN
            auto_voted = resolved != DatasetGranularity.EVENT_LEVEL
            votes[name] = DatasetGridVote(
                dataset_name=name,
                granularity=resolved,
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

        if intent.history_upper_limit is not None and grid_end is not None:
            grid_end = min(grid_end, intent.history_upper_limit)

        if intent.lookback_periods is not None and grid_end is not None:
            lookback_total = intent.lookback_periods * CADENCE_DAYS[intent.cadence_interval]
            earliest = (_dt.date.fromisoformat(grid_end) - _dt.timedelta(days=lookback_total)).isoformat()
            grid_start = max(grid_start, earliest) if grid_start is not None else earliest

        grid = cls(
            mode=mode,
            cadence_interval=intent.cadence_interval,
            observation_window_days=intent.observation_window_days,
            purge_gap_days=intent.purge_gap_days,
            label_window_days=intent.label_window_days,
            dataset_votes=votes,
            grid_start=grid_start,
            grid_end=grid_end,
            lookback_periods=intent.lookback_periods,
        )
        grid.generate_grid_dates()
        return grid

    def cadence_to_days(self) -> int:
        return CADENCE_DAYS[self.cadence_interval]

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
        self.grid_dates = _cap_grid_dates(dates, max_dates_override=self.max_grid_dates)

    def record_vote(self, name: str, vote: DatasetGridVote) -> None:
        if self.locked or self.mode == GridAdjustmentMode.NO_ADJUSTMENTS:
            return
        self.dataset_votes[name] = vote
        self._refresh_grid()

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
            and str(v.data_span_start) != "NaT" and str(v.data_span_end) != "NaT"
        ]
        return compute_boundaries(
            spans,
            observation_window_days=self.observation_window_days,
            purge_gap_days=purge_gap_days,
            label_window_days=label_window_days,
        )

    def lock(self, purge_gap_days: int = 0, label_window_days: int = 0) -> None:
        self.purge_gap_days = purge_gap_days
        self.label_window_days = label_window_days
        start, end = self.compute_boundaries_from_votes(purge_gap_days, label_window_days)
        if start is not None and end is not None:
            self.grid_start = start
            self.grid_end = end
        if self.lookback_periods is not None and self.grid_end is not None:
            lookback_total = self.lookback_periods * self.cadence_to_days()
            earliest = (_dt.date.fromisoformat(self.grid_end) - _dt.timedelta(days=lookback_total)).isoformat()
            self.grid_start = max(self.grid_start, earliest) if self.grid_start is not None else earliest
        self.generate_grid_dates()
        self.locked = True

    def estimated_grid_size(self, n_entities: int) -> int:
        return n_entities * len(self.grid_dates)

    def save(self, path) -> None:
        p = _coerce_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.dump(self.model_dump(mode="json"), default_flow_style=False, sort_keys=False)
        _atomic_write_text(p, content)

    def apply_votes(self, votes: dict[str, DatasetGridVote]) -> None:
        if self.locked:
            return
        for name, vote in votes.items():
            self.dataset_votes[name] = vote
        self._refresh_grid()

    @classmethod
    def save_vote(cls, vote_dir, dataset_name: str, vote: DatasetGridVote) -> None:
        p = _coerce_path(vote_dir)
        p.mkdir(parents=True, exist_ok=True)
        content = yaml.dump(vote.model_dump(mode="json"), default_flow_style=False, sort_keys=False)
        _atomic_write_text(p / f"{dataset_name}.yaml", content)

    @classmethod
    def load_votes(cls, vote_dir) -> dict[str, DatasetGridVote]:
        p = _coerce_path(vote_dir)
        if not p.is_dir():
            return {}
        result: dict[str, DatasetGridVote] = {}
        for f in sorted(p.glob("*.yaml")):
            data = yaml.safe_load(f.read_text())
            result[f.stem] = DatasetGridVote.model_validate(data)
        return result

    def _refresh_grid(self) -> None:
        if self.grid_start is None or self.grid_end is None:
            start, end = self.compute_boundaries_from_votes(
                self.purge_gap_days, self.label_window_days,
            )
            if start is not None and end is not None:
                self.grid_start = start
                self.grid_end = end
        self.generate_grid_dates()

    @classmethod
    def load(cls, path) -> SnapshotGrid:
        p = _coerce_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Snapshot grid file not found: {p}")
        data = yaml.safe_load(p.read_text())
        return cls.model_validate(data)
