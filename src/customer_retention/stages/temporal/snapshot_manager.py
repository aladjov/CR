"""Versioned training snapshot management with integrity verification.

This module provides infrastructure for creating, versioning, and verifying
training data snapshots. Each snapshot includes:

- Point-in-time filtered data (only label_available=True records)
- SHA256 integrity hash for tamper detection
- Comprehensive metadata (cutoff date, feature columns, row counts)
- Version tracking for reproducibility

Example:
    >>> from customer_retention.stages.temporal import SnapshotManager
    >>> from datetime import datetime
    >>>
    >>> manager = SnapshotManager(base_path="./output")
    >>> metadata = manager.create_snapshot(
    ...     df=prepared_df,
    ...     cutoff_date=datetime(2024, 6, 1),
    ...     target_column="churn"
    ... )
    >>> print(f"Created {metadata.snapshot_id} with hash {metadata.data_hash}")
    >>>
    >>> # Load with integrity verification
    >>> df, meta = manager.load_snapshot("training_v1")
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from customer_retention.core.compat import is_datetime64_any_dtype, is_extension_array_dtype, native_pd


def _strip_tz_series(s: Any) -> Any:
    if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
        return s.dt.tz_localize(None)
    return s


def _as_naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def compute_composite_dataset_name(dataset_names: list[str]) -> str:
    cleaned = [n.strip() for n in dataset_names if n.strip()]
    if not cleaned:
        raise ValueError("dataset_names must contain at least one non-blank name")
    joined = "\0".join(sorted(cleaned))
    return f"combined_{hashlib.md5(joined.encode()).hexdigest()[:6]}"


def require_consistent_cutoffs(registry, dataset_names: list[str]) -> str:
    composite_name = compute_composite_dataset_name(dataset_names)
    missing = [n for n in dataset_names if registry.get_snapshot(n) is None]
    if missing:
        raise ValueError(f"Datasets not registered in PIT registry: {', '.join(missing)}")
    cutoffs = {n: registry.get_snapshot(n).cutoff_date for n in dataset_names}
    reference_date = next(iter(cutoffs.values())).date()
    inconsistent = [n for n, c in cutoffs.items() if c.date() != reference_date]
    if inconsistent:
        raise ValueError(
            f"Inconsistent cutoff dates for merge. Reference: {reference_date}. "
            f"Out of sync: {', '.join(inconsistent)}. Re-run exploration for these datasets."
        )
    return composite_name


@dataclass
class SnapshotMetadata:
    """Metadata for a training data snapshot.

    Attributes:
        snapshot_id: Unique identifier (e.g., "training_v1")
        version: Numeric version number
        created_at: When the snapshot was created
        cutoff_date: Point-in-time cutoff for label availability
        label_availability_filter: Filter expression used
        row_count: Number of rows in the snapshot
        column_count: Number of columns in the snapshot
        data_hash: SHA256 hash of the data for integrity verification
        feature_columns: List of feature column names
        target_column: Name of the target column
        timestamp_config: Configuration used for timestamp handling
    """
    snapshot_id: str
    version: int
    created_at: datetime
    cutoff_date: datetime
    label_availability_filter: str
    row_count: int
    column_count: int
    data_hash: str
    feature_columns: list[str]
    target_column: str
    timestamp_config: dict[str, Any]


class SnapshotManager:
    """Manages versioned training data snapshots with integrity verification.

    The SnapshotManager creates point-in-time correct training snapshots,
    maintaining version history and providing SHA256 integrity verification
    on load to detect any data modifications.

    Example:
        >>> manager = SnapshotManager(base_path="./output")
        >>> # Create a new snapshot
        >>> meta = manager.create_snapshot(df, cutoff_date, "churn")
        >>> # List all snapshots
        >>> print(manager.list_snapshots())  # ["training_v1", "training_v2"]
        >>> # Load with verification
        >>> df, meta = manager.load_snapshot("training_v1")
    """

    def __init__(self, base_path: Path, storage=None, dataset_name=None):
        self.base_path = Path(base_path) if not isinstance(base_path, Path) else base_path
        base = self.base_path / "snapshots"
        self.snapshots_dir = base / dataset_name if dataset_name else base
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.storage = storage or _get_storage()

    def create_snapshot(
        self, df: Any, cutoff_date: datetime, target_column: str,
        snapshot_name: str = "training", timestamp_series: Optional[Any] = None,
    ) -> SnapshotMetadata:
        if timestamp_series is not None:
            ts = _strip_tz_series(timestamp_series)
        else:
            ts = _strip_tz_series(df["feature_timestamp"])
        naive_cutoff = _as_naive(cutoff_date)
        snapshot_df = df[
            (df["label_available_flag"] == True) & (ts <= naive_cutoff)
        ]

        table_path = self.snapshots_dir / snapshot_name
        version = self._next_version(table_path, snapshot_name)
        snapshot_id = f"{snapshot_name}_v{version}"
        data_hash = self._compute_hash(snapshot_df, cutoff_date)

        metadata_cols = ["feature_timestamp", "label_timestamp", "label_available_flag"]
        feature_cols = [c for c in snapshot_df.columns if c not in metadata_cols and c != target_column]

        self._write_snapshot(snapshot_df, str(table_path), snapshot_id)

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id, version=version, created_at=datetime.now(),
            cutoff_date=cutoff_date, label_availability_filter="label_available_flag == True AND feature_timestamp <= cutoff",
            row_count=len(snapshot_df), column_count=len(snapshot_df.columns),
            data_hash=data_hash, feature_columns=feature_cols, target_column=target_column,
            timestamp_config={"cutoff_date": cutoff_date.isoformat(), "label_available_only": True},
        )

        self._save_metadata(metadata, snapshot_id)
        return metadata

    def load_snapshot(self, snapshot_id: str) -> tuple[Any, SnapshotMetadata]:
        snapshot_name, version = self._parse_snapshot_id(snapshot_id)
        table_path = self.snapshots_dir / snapshot_name

        if self.storage and self.storage.exists(str(table_path)):
            delta_version = version - 1
            df = self.storage.read(str(table_path), version=delta_version)
        else:
            parquet_path = self.snapshots_dir / f"{snapshot_id}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")
            df = native_pd.read_parquet(str(parquet_path))

        metadata = self._load_metadata(snapshot_id)

        current_hash = self._compute_hash(df, metadata.cutoff_date)
        if current_hash != metadata.data_hash:
            raise ValueError(f"Snapshot integrity check failed for {snapshot_id}. Cutoff date or data may have changed.")

        return df, metadata

    def list_snapshots(self) -> list[str]:
        snapshots = []
        for p in self.snapshots_dir.glob("*_v*.parquet"):
            snapshots.append(p.stem)
        if self.storage:
            for subdir in self.snapshots_dir.iterdir():
                if subdir.is_dir() and self.storage.exists(str(subdir)):
                    history_len = len(self.storage.history(str(subdir)))
                    for v in range(1, history_len + 1):
                        sid = f"{subdir.name}_v{v}"
                        if sid not in snapshots:
                            snapshots.append(sid)
        return snapshots

    def get_latest_snapshot(self, snapshot_name: str = "training") -> Optional[str]:
        snapshots = [s for s in self.list_snapshots() if s.startswith(f"{snapshot_name}_v")]
        if not snapshots:
            return None
        return sorted(snapshots, key=lambda s: int(s.split("_v")[-1]))[-1]

    def compare_snapshots(self, snapshot_id_1: str, snapshot_id_2: str) -> dict[str, Any]:
        _, meta1 = self.load_snapshot(snapshot_id_1)
        _, meta2 = self.load_snapshot(snapshot_id_2)

        return {
            "snapshot_1": snapshot_id_1,
            "snapshot_2": snapshot_id_2,
            "row_diff": meta2.row_count - meta1.row_count,
            "column_diff": meta2.column_count - meta1.column_count,
            "cutoff_1": meta1.cutoff_date,
            "cutoff_2": meta2.cutoff_date,
            "new_features": set(meta2.feature_columns) - set(meta1.feature_columns),
            "removed_features": set(meta1.feature_columns) - set(meta2.feature_columns),
        }

    def _write_snapshot(self, df: Any, table_path: str, snapshot_id: str) -> None:
        if self.storage and len(df) > 0:
            metadata = {"snapshot_id": snapshot_id, "created_at": datetime.now().isoformat()}
            self.storage.write(df.reset_index(drop=True), table_path, mode="overwrite", metadata=metadata)
        else:
            parquet_path = Path(table_path).parent / f"{snapshot_id}.parquet"
            df.to_parquet(str(parquet_path), index=False)

    def _next_version(self, table_path: Path, snapshot_name: str) -> int:
        if self.storage and self.storage.exists(str(table_path)):
            return len(self.storage.history(str(table_path))) + 1
        parquet_count = len(list(self.snapshots_dir.glob(f"{snapshot_name}_v*.parquet")))
        if self.storage and table_path.is_dir():
            return len(self.storage.history(str(table_path))) + 1
        return parquet_count + 1

    def _parse_snapshot_id(self, snapshot_id: str) -> tuple[str, int]:
        parts = snapshot_id.rsplit("_v", 1)
        return parts[0], int(parts[1])

    def _compute_hash(self, df: Any, cutoff_date: Optional[datetime] = None) -> str:
        df_stable = df.reset_index(drop=True).copy()
        for col in df_stable.columns:
            if is_datetime64_any_dtype(df_stable[col]):
                df_stable[col] = df_stable[col].dt.floor("us").astype(str)
        for col in df_stable.columns:
            if is_extension_array_dtype(df_stable[col]):
                df_stable[col] = df_stable[col].astype(object)
        df_stable = df_stable[sorted(df_stable.columns)]

        data_bytes = native_pd.util.hash_pandas_object(df_stable).values.tobytes()
        if cutoff_date:
            normalized = datetime.fromisoformat(cutoff_date.isoformat())
            data_bytes += normalized.isoformat().encode("utf-8")

        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def _save_metadata(self, metadata: SnapshotMetadata, snapshot_id: str) -> None:
        metadata_path = self.snapshots_dir / f"{snapshot_id}_metadata.json"
        metadata_dict = {
            "snapshot_id": metadata.snapshot_id,
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "cutoff_date": metadata.cutoff_date.isoformat(),
            "label_availability_filter": metadata.label_availability_filter,
            "row_count": metadata.row_count,
            "column_count": metadata.column_count,
            "data_hash": metadata.data_hash,
            "feature_columns": metadata.feature_columns,
            "target_column": metadata.target_column,
            "timestamp_config": metadata.timestamp_config,
        }
        with metadata_path.open("w") as f:
            json.dump(metadata_dict, f, indent=2)

    def _load_metadata(self, snapshot_id: str) -> SnapshotMetadata:
        metadata_path = self.snapshots_dir / f"{snapshot_id}_metadata.json"
        with metadata_path.open("r") as f:
            data = json.load(f)

        return SnapshotMetadata(
            snapshot_id=data["snapshot_id"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            cutoff_date=datetime.fromisoformat(data["cutoff_date"]),
            label_availability_filter=data["label_availability_filter"],
            row_count=data["row_count"],
            column_count=data["column_count"],
            data_hash=data["data_hash"],
            feature_columns=data["feature_columns"],
            target_column=data["target_column"],
            timestamp_config=data["timestamp_config"],
        )


def _get_storage():
    try:
        from customer_retention.integrations.adapters.factory import get_delta
        return get_delta()
    except ImportError:
        return None
