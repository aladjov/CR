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

import pandas as pd


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

    def __init__(self, base_path: Path):
        """Initialize the SnapshotManager.

        Args:
            base_path: Base directory for storing snapshots
        """
        self.base_path = Path(base_path)
        self.snapshots_dir = self.base_path / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self, df: pd.DataFrame, cutoff_date: datetime, target_column: str, snapshot_name: str = "training"
    ) -> SnapshotMetadata:
        snapshot_df = df[
            (df["label_available_flag"] == True) & (df["feature_timestamp"] <= cutoff_date)
        ].copy()

        existing = list(self.snapshots_dir.glob(f"{snapshot_name}_v*.parquet"))
        version = len(existing) + 1
        snapshot_id = f"{snapshot_name}_v{version}"
        data_hash = self._compute_hash(snapshot_df, cutoff_date)

        metadata_cols = ["feature_timestamp", "label_timestamp", "label_available_flag"]
        feature_cols = [c for c in snapshot_df.columns if c not in metadata_cols and c != target_column]

        snapshot_path = self.snapshots_dir / f"{snapshot_id}.parquet"
        snapshot_df.to_parquet(snapshot_path, index=False)

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id, version=version, created_at=datetime.now(),
            cutoff_date=cutoff_date, label_availability_filter="label_available_flag == True AND feature_timestamp <= cutoff",
            row_count=len(snapshot_df), column_count=len(snapshot_df.columns),
            data_hash=data_hash, feature_columns=feature_cols, target_column=target_column,
            timestamp_config={"cutoff_date": cutoff_date.isoformat(), "label_available_only": True},
        )

        self._save_metadata(metadata, snapshot_id)
        return metadata

    def load_snapshot(self, snapshot_id: str) -> tuple[pd.DataFrame, SnapshotMetadata]:
        snapshot_path = self.snapshots_dir / f"{snapshot_id}.parquet"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")

        df = pd.read_parquet(snapshot_path)
        metadata = self._load_metadata(snapshot_id)

        current_hash = self._compute_hash(df, metadata.cutoff_date)
        if current_hash != metadata.data_hash:
            raise ValueError(f"Snapshot integrity check failed for {snapshot_id}. Cutoff date or data may have changed.")

        return df, metadata

    def list_snapshots(self) -> list[str]:
        return [p.stem for p in self.snapshots_dir.glob("*_v*.parquet")]

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

    def _compute_hash(self, df: pd.DataFrame, cutoff_date: Optional[datetime] = None) -> str:
        df_stable = df.reset_index(drop=True).copy()
        for col in df_stable.select_dtypes(include=["datetime64", "datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            df_stable[col] = df_stable[col].astype(str)
        df_stable = df_stable[sorted(df_stable.columns)]

        data_bytes = pd.util.hash_pandas_object(df_stable).values.tobytes()
        if cutoff_date:
            data_bytes += cutoff_date.isoformat().encode("utf-8")

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
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def _load_metadata(self, snapshot_id: str) -> SnapshotMetadata:
        metadata_path = self.snapshots_dir / f"{snapshot_id}_metadata.json"
        with open(metadata_path) as f:
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
