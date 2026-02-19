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
from typing import Any, Optional

from .point_in_time_join import PointInTimeJoiner
from .snapshot_manager import SnapshotManager
from .timestamp_manager import TimestampConfig, TimestampManager


@dataclass
class PreparedData:
    """Container for prepared data with validation results.

    Attributes:
        unified_df: The prepared DataFrame with timestamps
        snapshot_metadata: Metadata about the training snapshot
        timestamp_strategy: Strategy used for timestamp handling
        validation_report: Report from temporal integrity validation
    """
    unified_df: Any
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

    def __init__(self, output_path: Path, timestamp_config: TimestampConfig,
                 storage=None, dataset_name="unified_dataset"):
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        self.timestamp_manager = TimestampManager(timestamp_config)
        self.snapshot_manager = SnapshotManager(output_path, storage=storage, dataset_name=dataset_name)
        self.timestamp_config = timestamp_config
        self.pit_joiner = PointInTimeJoiner()
        self.storage = storage or _get_storage()

    def prepare_from_raw(
        self, df: Any, target_column: str, entity_column: str
    ) -> Any:
        df = self.timestamp_manager.ensure_timestamps(df)
        self.timestamp_manager.validate_point_in_time(df)

        if target_column in df.columns and "label_available_flag" in df.columns:
            df.loc[df[target_column].notna(), "label_available_flag"] = True

        df = df.rename(columns={target_column: "target", entity_column: "entity_id"})

        unified_dir = self.output_path / "unified" / self.dataset_name
        unified_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.storage and len(df) > 0:
            self.storage.write(df, str(unified_dir))
        else:
            parquet_path = self.output_path / "unified" / f"{self.dataset_name}.parquet"
            df.to_parquet(str(parquet_path), index=False)

        return df

    def create_training_snapshot(
        self, df: Any, cutoff_date: datetime, snapshot_name: str = "training",
        timestamp_series: Optional[Any] = None,
    ) -> tuple[Any, dict[str, Any]]:
        metadata = self.snapshot_manager.create_snapshot(
            df=df, cutoff_date=cutoff_date, target_column="target",
            snapshot_name=snapshot_name, timestamp_series=timestamp_series,
        )
        snapshot_df, _ = self.snapshot_manager.load_snapshot(metadata.snapshot_id)
        return snapshot_df, self._metadata_to_dict(metadata)

    def load_for_eda(self, snapshot_id: str) -> Any:
        df, metadata = self.snapshot_manager.load_snapshot(snapshot_id)
        print(f"Loaded snapshot: {snapshot_id}")
        print(f"  Rows: {metadata.row_count:,}")
        print(f"  Cutoff: {metadata.cutoff_date}")
        print(f"  Hash: {metadata.data_hash}")
        return df

    def load_for_inference(self, df: Any, as_of_date: Optional[datetime] = None) -> Any:
        as_of_date = as_of_date or datetime.now()
        df = self.timestamp_manager.ensure_timestamps(df)
        df = df[df["feature_timestamp"] <= as_of_date].copy()
        df["label_available_flag"] = False
        df["label_timestamp"] = as_of_date
        return df

    def prepare_with_validation(
        self, df: Any, target_column: str, entity_column: str, cutoff_date: datetime
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


def _get_storage():
    try:
        from customer_retention.integrations.adapters.factory import get_delta
        return get_delta()
    except ImportError:
        return None
