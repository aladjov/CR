#!/usr/bin/env python3
"""Script to create versioned training snapshots from existing data.

This script creates point-in-time training snapshots with integrity hashing
and metadata tracking. Use it to version your training data and ensure
reproducibility.

Usage:
    python scripts/data/create_snapshot.py --input data.parquet --output output/
    python scripts/data/create_snapshot.py --input data.csv --output output/ --cutoff 2024-01-01
    python scripts/data/create_snapshot.py --list --output output/  # List existing snapshots
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

from customer_retention.stages.temporal import SnapshotManager, SnapshotMetadata


def create_snapshot(
    input_path: str,
    output_dir: str,
    cutoff_date: datetime | None = None,
    version: str | None = None,
    description: str | None = None,
) -> SnapshotMetadata:
    """Create a versioned training snapshot.

    Args:
        input_path: Path to input dataset (CSV or Parquet)
        output_dir: Output directory for snapshots
        cutoff_date: Point-in-time cutoff date (defaults to now)
        version: Optional version string
        description: Optional description

    Returns:
        SnapshotMetadata with snapshot details
    """
    input_path = Path(input_path)
    output_path = Path(output_dir)

    # Load data
    print(f"Loading data from {input_path}...")
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Initialize snapshot manager
    snapshot_manager = SnapshotManager(output_path)

    # Use current time if no cutoff provided
    if cutoff_date is None:
        cutoff_date = datetime.now()

    # Create snapshot
    print(f"\nCreating snapshot with cutoff: {cutoff_date}...")
    metadata = snapshot_manager.create_snapshot(
        df=df,
        cutoff_date=cutoff_date,
        version=version,
        description=description,
    )

    print(f"\n  Snapshot created successfully!")
    print(f"  ID: {metadata.snapshot_id}")
    print(f"  Version: {metadata.version}")
    print(f"  Data hash: {metadata.data_hash[:16]}...")
    print(f"  Rows: {metadata.row_count:,}")
    print(f"  Features: {len(metadata.feature_columns)}")
    print(f"  Cutoff: {metadata.cutoff_date}")

    return metadata


def list_snapshots(output_dir: str) -> list[SnapshotMetadata]:
    """List all existing snapshots.

    Args:
        output_dir: Directory containing snapshots

    Returns:
        List of SnapshotMetadata objects
    """
    output_path = Path(output_dir)
    snapshot_manager = SnapshotManager(output_path)

    snapshots = snapshot_manager.list_snapshots()

    if not snapshots:
        print("No snapshots found.")
        return []

    print(f"Found {len(snapshots)} snapshot(s):\n")
    print(f"{'ID':<40} {'Version':<10} {'Rows':<10} {'Cutoff':<12} {'Hash (short)':<16}")
    print("-" * 90)

    for meta in snapshots:
        print(
            f"{meta.snapshot_id:<40} "
            f"{meta.version:<10} "
            f"{meta.row_count:<10,} "
            f"{str(meta.cutoff_date)[:10]:<12} "
            f"{meta.data_hash[:16]}"
        )

    return snapshots


def compare_snapshots(output_dir: str, snapshot_id1: str, snapshot_id2: str) -> dict:
    """Compare two snapshots.

    Args:
        output_dir: Directory containing snapshots
        snapshot_id1: First snapshot ID
        snapshot_id2: Second snapshot ID

    Returns:
        Comparison results dictionary
    """
    output_path = Path(output_dir)
    snapshot_manager = SnapshotManager(output_path)

    comparison = snapshot_manager.compare_snapshots(snapshot_id1, snapshot_id2)

    print(f"\nSnapshot Comparison:")
    print(f"  Snapshot 1: {snapshot_id1}")
    print(f"  Snapshot 2: {snapshot_id2}")
    print(f"\n  Data identical: {comparison['identical']}")
    print(f"  Row count diff: {comparison['row_count_diff']}")
    print(f"  Columns added: {comparison['columns_added']}")
    print(f"  Columns removed: {comparison['columns_removed']}")

    return comparison


def load_snapshot(output_dir: str, snapshot_id: str) -> tuple[pd.DataFrame, SnapshotMetadata]:
    """Load a specific snapshot.

    Args:
        output_dir: Directory containing snapshots
        snapshot_id: Snapshot ID to load

    Returns:
        Tuple of (DataFrame, SnapshotMetadata)
    """
    output_path = Path(output_dir)
    snapshot_manager = SnapshotManager(output_path)

    df, metadata = snapshot_manager.load_snapshot(snapshot_id)

    print(f"\nLoaded snapshot: {snapshot_id}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Cutoff: {metadata.cutoff_date}")
    print(f"  Hash verified: {metadata.data_hash[:16]}...")

    return df, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create and manage versioned training snapshots"
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to input dataset (CSV or Parquet)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for snapshots"
    )
    parser.add_argument(
        "--cutoff", "-c",
        help="Cutoff date for snapshot (YYYY-MM-DD format, defaults to now)"
    )
    parser.add_argument(
        "--version", "-v",
        help="Version string for the snapshot"
    )
    parser.add_argument(
        "--description", "-d",
        help="Description for the snapshot"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all existing snapshots"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("ID1", "ID2"),
        help="Compare two snapshots"
    )
    parser.add_argument(
        "--load",
        metavar="ID",
        help="Load and display info about a specific snapshot"
    )

    args = parser.parse_args()

    # Handle different modes
    if args.list:
        list_snapshots(args.output)
    elif args.compare:
        compare_snapshots(args.output, args.compare[0], args.compare[1])
    elif args.load:
        load_snapshot(args.output, args.load)
    elif args.input:
        cutoff_date = None
        if args.cutoff:
            cutoff_date = datetime.fromisoformat(args.cutoff)

        metadata = create_snapshot(
            input_path=args.input,
            output_dir=args.output,
            cutoff_date=cutoff_date,
            version=args.version,
            description=args.description,
        )

        print(f"\nSnapshot saved to: {args.output}/snapshots/{metadata.snapshot_id}.parquet")
    else:
        parser.print_help()
        print("\nError: Either --input or --list/--compare/--load is required")
        sys.exit(1)


if __name__ == "__main__":
    main()
