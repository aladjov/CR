"""Temporal framework for leakage-safe ML pipelines.

This module provides infrastructure for preventing data leakage in ML training
by enforcing point-in-time (PIT) correctness throughout the data preparation
and training pipeline.

Core Components:
    - TimestampManager: Ensures proper timestamp columns exist
    - TimestampDiscoveryEngine: Auto-detects timestamps in datasets
    - ScenarioDetector: Determines appropriate timestamp strategy
    - UnifiedDataPreparer: Single entry point for data preparation
    - SnapshotManager: Versioned training snapshots with integrity hashing
    - DataAccessGuard: Context-based data access control

Quick Start:
    >>> from customer_retention.stages.temporal import (
    ...     ScenarioDetector, UnifiedDataPreparer
    ... )
    >>> from datetime import datetime
    >>>
    >>> # Detect scenario and prepare data
    >>> detector = ScenarioDetector()
    >>> scenario, config, _ = detector.detect(df, target_column="churn")
    >>>
    >>> preparer = UnifiedDataPreparer(output_path="./output", timestamp_config=config)
    >>> prepared_df = preparer.prepare_from_raw(df, "churn", "customer_id")
    >>>
    >>> # Create versioned training snapshot
    >>> snapshot_df, meta = preparer.create_training_snapshot(
    ...     prepared_df,
    ...     cutoff_date=datetime(2024, 6, 1)
    ... )
    >>> print(f"Snapshot: {meta['snapshot_id']}, hash: {meta['data_hash']}")

Timestamp Scenarios:
    - production: Dataset has explicit feature and label timestamps
    - partial: Only feature timestamp found, label derived from window
    - derived: Timestamps can be computed from other columns
    - synthetic: No temporal information, must use synthetic timestamps
"""

# Import canonical temporal metadata columns from central location
from customer_retention.core.utils.leakage import TEMPORAL_METADATA_COLUMNS

from .access_guard import AccessContext, DataAccessGuard
from .data_preparer import PreparedData, UnifiedDataPreparer
from .point_in_time_join import PointInTimeJoiner
from .point_in_time_registry import ConsistencyReport, DatasetSnapshot, PointInTimeRegistry
from .scenario_detector import ScenarioDetector
from .snapshot_manager import (
    SnapshotManager,
    SnapshotMetadata,
    compute_composite_dataset_name,
    require_consistent_cutoffs,
)
from .spark_temporal_merger import SparkTemporalMerger
from .synthetic_coordinator import SyntheticCoordinationParams, SyntheticTimestampCoordinator
from .temporal_merger import (
    DatasetMergeInput,
    MergeConfig,
    MergeReport,
    TemporalMerger,
    classify_silver_columns,
)
from .timestamp_discovery import (
    DatetimeOrderAnalyzer,
    TimestampCandidate,
    TimestampDiscoveryEngine,
    TimestampDiscoveryResult,
    TimestampRole,
)
from .timestamp_manager import TimestampConfig, TimestampManager, TimestampStrategy

# Backwards compatible alias - prefer TEMPORAL_METADATA_COLUMNS
TEMPORAL_METADATA_COLS = TEMPORAL_METADATA_COLUMNS


__all__ = [
    "DatetimeOrderAnalyzer",
    "TimestampStrategy",
    "TimestampConfig",
    "TimestampManager",
    "TimestampRole",
    "TimestampCandidate",
    "TimestampDiscoveryResult",
    "TimestampDiscoveryEngine",
    "SnapshotMetadata",
    "SnapshotManager",
    "PointInTimeJoiner",
    "PreparedData",
    "UnifiedDataPreparer",
    "AccessContext",
    "DataAccessGuard",
    "ScenarioDetector",
    "DatasetSnapshot",
    "ConsistencyReport",
    "PointInTimeRegistry",
    "SyntheticCoordinationParams",
    "SyntheticTimestampCoordinator",
    "compute_composite_dataset_name",
    "require_consistent_cutoffs",
    "TEMPORAL_METADATA_COLS",
    "TemporalMerger",
    "SparkTemporalMerger",
    "MergeConfig",
    "DatasetMergeInput",
    "MergeReport",
    "classify_silver_columns",
]
