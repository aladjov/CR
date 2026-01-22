"""Unified data preparation for leakage-safe ML pipelines.

This module provides the main entry point for preparing raw data for
ML training with point-in-time correctness. It combines timestamp
management, snapshot creation, and validation into a single workflow.

Example:
    >>> from customer_retention.stages.temporal import (
    ...     ScenarioDetector, UnifiedDataPreparer
    ... )
    >>> from datetime import datetime
    >>>
    >>> # Detect scenario and get config
    >>> detector = ScenarioDetector()
    >>> scenario, config, _ = detector.detect(df, "churn")
    >>>
    >>> # Prepare data
    >>> preparer = UnifiedDataPreparer(output_path, config)
    >>> prepared_df = preparer.prepare_from_raw(df, "churn", "customer_id")
    >>>
    >>> # Create training snapshot
    >>> snapshot_df, meta = preparer.create_training_snapshot(
    ...     prepared_df,
    ...     cutoff_date=datetime(2024, 6, 1)
    ... )
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import pandas as pd

from .timestamp_manager import TimestampManager, TimestampConfig
from .snapshot_manager import SnapshotManager
from .point_in_time_join import PointInTimeJoiner


@dataclass
class PreparedData:
    """Container for prepared data with validation results.

    Attributes:
        unified_df: The prepared DataFrame with timestamps
        snapshot_metadata: Metadata about the training snapshot
        timestamp_strategy: Strategy used for timestamp handling
        validation_report: Report from temporal integrity validation
    """
    unified_df: pd.DataFrame
    snapshot_metadata: dict[str, Any]
    timestamp_strategy: str
    validation_report: dict[str, Any]


class UnifiedDataPreparer:
    """Unified entry point for preparing data with temporal correctness.

    The UnifiedDataPreparer combines timestamp management, data validation,
    and snapshot creation into a single workflow. It ensures all data
    passes through proper point-in-time handling before being used for
    training or inference.

    Example:
        >>> preparer = UnifiedDataPreparer(output_path, config)
        >>> df = preparer.prepare_from_raw(df, "churn", "customer_id")
        >>> snapshot_df, meta = preparer.create_training_snapshot(df, cutoff)
    """

    def __init__(self, output_path: Path, timestamp_config: TimestampConfig):
        """Initialize the UnifiedDataPreparer.

        Args:
            output_path: Directory for output files (unified data, snapshots)
            timestamp_config: Configuration for timestamp handling
        """
        self.output_path = Path(output_path)
        self.timestamp_manager = TimestampManager(timestamp_config)
        self.snapshot_manager = SnapshotManager(output_path)
        self.timestamp_config = timestamp_config
        self.pit_joiner = PointInTimeJoiner()

    def prepare_from_raw(
        self, df: pd.DataFrame, target_column: str, entity_column: str
    ) -> pd.DataFrame:
        df = self.timestamp_manager.ensure_timestamps(df)
        self.timestamp_manager.validate_point_in_time(df)

        df = df.rename(columns={target_column: "target", entity_column: "entity_id"})

        unified_path = self.output_path / "unified" / "unified_dataset.parquet"
        unified_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(unified_path, index=False)

        return df

    def create_training_snapshot(
        self, df: pd.DataFrame, cutoff_date: datetime, snapshot_name: str = "training"
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        metadata = self.snapshot_manager.create_snapshot(
            df=df, cutoff_date=cutoff_date, target_column="target", snapshot_name=snapshot_name
        )
        snapshot_df, _ = self.snapshot_manager.load_snapshot(metadata.snapshot_id)
        return snapshot_df, self._metadata_to_dict(metadata)

    def load_for_eda(self, snapshot_id: str) -> pd.DataFrame:
        df, metadata = self.snapshot_manager.load_snapshot(snapshot_id)
        print(f"Loaded snapshot: {snapshot_id}")
        print(f"  Rows: {metadata.row_count:,}")
        print(f"  Cutoff: {metadata.cutoff_date}")
        print(f"  Hash: {metadata.data_hash}")
        return df

    def load_for_inference(self, df: pd.DataFrame, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        as_of_date = as_of_date or datetime.now()
        df = self.timestamp_manager.ensure_timestamps(df)
        df = df[df["feature_timestamp"] <= as_of_date].copy()
        df["label_available_flag"] = False
        df["label_timestamp"] = as_of_date
        return df

    def prepare_with_validation(
        self, df: pd.DataFrame, target_column: str, entity_column: str, cutoff_date: datetime
    ) -> PreparedData:
        unified_df = self.prepare_from_raw(df, target_column, entity_column)
        validation_report = self.pit_joiner.validate_temporal_integrity(unified_df)
        snapshot_df, snapshot_metadata = self.create_training_snapshot(unified_df, cutoff_date)

        return PreparedData(
            unified_df=snapshot_df,
            snapshot_metadata=snapshot_metadata,
            timestamp_strategy=self.timestamp_config.strategy.value,
            validation_report=validation_report,
        )

    def list_available_snapshots(self) -> list[str]:
        return self.snapshot_manager.list_snapshots()

    def get_snapshot_summary(self, snapshot_id: str) -> dict[str, Any]:
        _, metadata = self.snapshot_manager.load_snapshot(snapshot_id)
        return {
            "snapshot_id": metadata.snapshot_id,
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "cutoff_date": metadata.cutoff_date.isoformat(),
            "row_count": metadata.row_count,
            "feature_count": len(metadata.feature_columns),
            "data_hash": metadata.data_hash,
        }

    def _metadata_to_dict(self, metadata) -> dict[str, Any]:
        return {
            "snapshot_id": metadata.snapshot_id,
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "cutoff_date": metadata.cutoff_date.isoformat(),
            "row_count": metadata.row_count,
            "column_count": metadata.column_count,
            "data_hash": metadata.data_hash,
            "feature_columns": metadata.feature_columns,
            "target_column": metadata.target_column,
        }
