from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


@dataclass
class DatasetSnapshot:
    dataset_name: str
    snapshot_id: str
    cutoff_date: datetime
    source_path: str
    row_count: int
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "snapshot_id": self.snapshot_id,
            "cutoff_date": self.cutoff_date.isoformat(),
            "source_path": self.source_path,
            "row_count": self.row_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSnapshot":
        return cls(
            dataset_name=data["dataset_name"],
            snapshot_id=data["snapshot_id"],
            cutoff_date=datetime.fromisoformat(data["cutoff_date"]),
            source_path=data["source_path"],
            row_count=data["row_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class ConsistencyReport:
    is_consistent: bool
    reference_cutoff: Optional[datetime]
    inconsistent_datasets: list[str]
    message: str


class PointInTimeRegistry:
    REGISTRY_FILENAME = "point_in_time_registry.json"

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.registry_path = self.output_dir / self.REGISTRY_FILENAME
        self.snapshots: dict[str, DatasetSnapshot] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                self.snapshots = {
                    name: DatasetSnapshot.from_dict(snap) for name, snap in data.get("snapshots", {}).items()
                }

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = {"snapshots": {name: snap.to_dict() for name, snap in self.snapshots.items()}}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_reference_cutoff(self) -> Optional[datetime]:
        if not self.snapshots:
            return None
        return next(iter(self.snapshots.values())).cutoff_date

    def check_consistency(self) -> ConsistencyReport:
        if not self.snapshots:
            return ConsistencyReport(
                is_consistent=True, reference_cutoff=None, inconsistent_datasets=[], message="No datasets registered"
            )

        reference_cutoff = self.get_reference_cutoff()
        inconsistent = [
            name for name, snap in self.snapshots.items() if snap.cutoff_date.date() != reference_cutoff.date()
        ]

        if inconsistent:
            return ConsistencyReport(
                is_consistent=False,
                reference_cutoff=reference_cutoff,
                inconsistent_datasets=inconsistent,
                message=f"Inconsistent cutoff dates detected. Reference: {reference_cutoff.date()}. "
                f"Out of sync: {', '.join(inconsistent)}. Re-run exploration for these datasets.",
            )

        return ConsistencyReport(
            is_consistent=True,
            reference_cutoff=reference_cutoff,
            inconsistent_datasets=[],
            message=f"All {len(self.snapshots)} datasets use consistent cutoff: {reference_cutoff.date()}",
        )

    def validate_cutoff(self, proposed_cutoff: datetime) -> tuple[bool, str]:
        reference = self.get_reference_cutoff()
        if reference is None:
            return True, "First dataset - cutoff date will be set as reference"

        if proposed_cutoff.date() != reference.date():
            return False, (
                f"Cutoff date mismatch. Existing datasets use {reference.date()}. "
                f"Proposed: {proposed_cutoff.date()}. Change will require re-exploration of all datasets."
            )

        return True, f"Cutoff date matches reference: {reference.date()}"

    def register_snapshot(
        self, dataset_name: str, snapshot_id: str, cutoff_date: datetime, source_path: str, row_count: int
    ) -> DatasetSnapshot:
        snapshot = DatasetSnapshot(
            dataset_name=dataset_name,
            snapshot_id=snapshot_id,
            cutoff_date=cutoff_date,
            source_path=source_path,
            row_count=row_count,
        )
        self.snapshots[dataset_name] = snapshot
        self._save()
        return snapshot

    def get_snapshot(self, dataset_name: str) -> Optional[DatasetSnapshot]:
        return self.snapshots.get(dataset_name)

    def list_snapshots(self) -> list[DatasetSnapshot]:
        return list(self.snapshots.values())

    def get_out_of_sync_datasets(self, reference_cutoff: datetime) -> list[str]:
        return [name for name, snap in self.snapshots.items() if snap.cutoff_date.date() != reference_cutoff.date()]

    def clear_registry(self) -> None:
        self.snapshots = {}
        if self.registry_path.exists():
            self.registry_path.unlink()

    def update_cutoff_for_all(self, new_cutoff: datetime) -> list[str]:
        affected = list(self.snapshots.keys())
        for name in affected:
            self.snapshots[name].cutoff_date = new_cutoff
        self._save()
        return affected
