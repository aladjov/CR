#!/usr/bin/env python3
"""Migration script to convert existing datasets to the leakage-safe temporal format.

This script helps migrate existing datasets that don't have explicit timestamps
to the new temporal framework with feature_timestamp and label_timestamp columns.

Usage:
    python scripts/data/migrate_to_temporal.py --input data.csv --output output/ --target churn
    python scripts/data/migrate_to_temporal.py --input data.parquet --output output/ --target retained --entity customer_id
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

from customer_retention.stages.temporal import (
    ScenarioDetector,
    UnifiedDataPreparer,
    SnapshotManager,
)


def migrate_dataset(
    input_path: str,
    output_dir: str,
    target_column: str,
    entity_column: str | None = None,
    create_snapshot: bool = True,
) -> dict:
    """Migrate a dataset to the temporal format.

    Args:
        input_path: Path to input dataset (CSV or Parquet)
        output_dir: Output directory for temporal data
        target_column: Name of the target column
        entity_column: Name of the entity/customer ID column (auto-detected if None)
        create_snapshot: Whether to create a versioned training snapshot

    Returns:
        Dictionary with migration results including paths and metadata
    """
    input_path = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {input_path}...")
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Detect timestamp scenario
    print("\nDetecting timestamp scenario...")
    detector = ScenarioDetector()
    scenario, ts_config, discovery_result = detector.detect(df, target_column)

    print(f"  Scenario: {scenario}")
    print(f"  Strategy: {ts_config.strategy.value}")
    print(f"  Recommendation: {discovery_result.recommendation}")

    if discovery_result.feature_timestamp:
        ft = discovery_result.feature_timestamp
        print(f"  Feature timestamp: {ft.column_name} (derived={ft.is_derived})")

    if discovery_result.label_timestamp:
        lt = discovery_result.label_timestamp
        print(f"  Label timestamp: {lt.column_name} (derived={lt.is_derived})")

    # Auto-detect entity column if not provided
    if entity_column is None:
        # Look for common ID column patterns
        id_patterns = ["customer_id", "custid", "user_id", "userid", "id", "entity_id"]
        for pattern in id_patterns:
            for col in df.columns:
                if col.lower() == pattern:
                    entity_column = col
                    break
            if entity_column:
                break

        if entity_column is None:
            entity_column = "entity_id"
            print(f"\n  Warning: No entity column detected, using index as '{entity_column}'")
            df[entity_column] = range(len(df))
        else:
            print(f"\n  Auto-detected entity column: {entity_column}")

    # Prepare data with timestamps
    print("\nPreparing data with timestamps...")
    preparer = UnifiedDataPreparer(output_path, ts_config)
    unified_df = preparer.prepare_from_raw(
        df,
        target_column=target_column,
        entity_column=entity_column,
    )

    print(f"  Added columns: feature_timestamp, label_timestamp, label_available_flag")
    print(f"  Output rows: {len(unified_df):,}")

    # Save unified data
    unified_path = output_path / "unified" / f"{input_path.stem}_unified.parquet"
    unified_path.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_parquet(unified_path, index=False)
    print(f"\n  Unified data saved to: {unified_path}")

    result = {
        "input_path": str(input_path),
        "output_dir": str(output_path),
        "unified_path": str(unified_path),
        "scenario": scenario,
        "strategy": ts_config.strategy.value,
        "row_count": len(unified_df),
        "column_count": len(unified_df.columns),
        "entity_column": entity_column,
        "target_column": target_column,
    }

    # Create training snapshot
    if create_snapshot:
        print("\nCreating training snapshot...")
        cutoff_date = datetime.now()
        snapshot_df, snapshot_metadata = preparer.create_training_snapshot(
            unified_df, cutoff_date
        )

        result["snapshot_id"] = snapshot_metadata["snapshot_id"]
        result["snapshot_path"] = str(
            output_path / "snapshots" / f"{snapshot_metadata['snapshot_id']}.parquet"
        )
        result["snapshot_metadata"] = snapshot_metadata

        print(f"  Snapshot ID: {snapshot_metadata['snapshot_id']}")
        print(f"  Data hash: {snapshot_metadata['data_hash'][:16]}...")
        print(f"  Cutoff date: {cutoff_date.date()}")

    print("\n" + "=" * 60)
    print("Migration completed successfully!")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing datasets to the leakage-safe temporal format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input dataset (CSV or Parquet)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for temporal data"
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Name of the target column"
    )
    parser.add_argument(
        "--entity", "-e",
        default=None,
        help="Name of the entity/customer ID column (auto-detected if not provided)"
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip creating a training snapshot"
    )

    args = parser.parse_args()

    result = migrate_dataset(
        input_path=args.input,
        output_dir=args.output,
        target_column=args.target,
        entity_column=args.entity,
        create_snapshot=not args.no_snapshot,
    )

    # Print summary
    print("\nMigration Summary:")
    print(f"  Input: {result['input_path']}")
    print(f"  Output: {result['unified_path']}")
    print(f"  Scenario: {result['scenario']}")
    print(f"  Strategy: {result['strategy']}")
    print(f"  Rows: {result['row_count']:,}")

    if "snapshot_id" in result:
        print(f"  Snapshot: {result['snapshot_id']}")


if __name__ == "__main__":
    main()
