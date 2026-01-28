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
from .cutoff_analyzer import CutoffAnalysis, CutoffAnalyzer, SplitResult
from .data_preparer import PreparedData, UnifiedDataPreparer
from .point_in_time_join import PointInTimeJoiner
from .point_in_time_registry import ConsistencyReport, DatasetSnapshot, PointInTimeRegistry
from .scenario_detector import ScenarioDetector
from .snapshot_manager import SnapshotManager, SnapshotMetadata
from .synthetic_coordinator import SyntheticCoordinationParams, SyntheticTimestampCoordinator
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


def _restore_snapshot_columns(df, findings):
    """Reverse the entity_id/target renames applied by UnifiedDataPreparer."""
    renames = {}
    ts_meta = getattr(findings, "time_series_metadata", None)
    entity_col = ts_meta.entity_column if ts_meta else None
    target_col = getattr(findings, "target_column", None)

    if entity_col and "entity_id" in df.columns and entity_col not in df.columns:
        renames["entity_id"] = entity_col
    if target_col and "target" in df.columns and target_col not in df.columns:
        renames["target"] = target_col

    return df.rename(columns=renames) if renames else df


def load_data_with_snapshot_preference(findings, output_dir: str = "../explorations"):
    """Load data preferring snapshots over raw source files.

    This function implements the recommended data loading pattern for exploration
    notebooks. It checks if a training snapshot exists and loads from it if available,
    otherwise falls back to the original source file.

    Parameters
    ----------
    findings : ExplorationFindings
        The findings object loaded from a previous exploration
    output_dir : str
        Directory containing explorations and snapshots

    Returns
    -------
    tuple[pd.DataFrame, str]
        DataFrame and a string indicating the source ("snapshot" or "source")

    Example
    -------
    >>> from customer_retention.stages.temporal import load_data_with_snapshot_preference
    >>> findings = ExplorationFindings.load(FINDINGS_PATH)
    >>> df, source = load_data_with_snapshot_preference(findings)
    >>> print(f"Loaded from: {source}")
    """
    from pathlib import Path

    import pandas as pd

    # Check if snapshot exists in findings
    snapshot_path = getattr(findings, 'snapshot_path', None)

    if snapshot_path and Path(snapshot_path).exists():
        df = pd.read_parquet(snapshot_path)
        return _restore_snapshot_columns(df, findings), "snapshot"

    # Check for snapshots in output directory
    output_path = Path(output_dir) / "snapshots"
    if output_path.exists():
        snapshot_manager = SnapshotManager(Path(output_dir))
        snapshots = snapshot_manager.list_snapshots()
        if snapshots:
            latest = snapshot_manager.get_latest_snapshot()
            if latest:
                df, _ = snapshot_manager.load_snapshot(latest)
                return _restore_snapshot_columns(df, findings), f"snapshot:{latest}"

    # Fall back to source file
    source_path = findings.source_path
    if source_path.endswith('.csv'):
        df = pd.read_csv(source_path)
    else:
        df = pd.read_parquet(source_path)

    return df, "source"


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
    "CutoffAnalysis",
    "CutoffAnalyzer",
    "SplitResult",
    "SyntheticCoordinationParams",
    "SyntheticTimestampCoordinator",
    "load_data_with_snapshot_preference",
    "TEMPORAL_METADATA_COLS",
]
