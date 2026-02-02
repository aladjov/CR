#!/usr/bin/env python3
"""Migrate existing parquet files to Delta Lake tables.

Scans a directory tree for .parquet files and converts them to delta tables.
Snapshot files matching {name}_v{N}.parquet are consolidated into a single
delta table with sequential versions.

Usage:
    python scripts/data/migrate_parquet_to_delta.py --dir experiments/
    python scripts/data/migrate_parquet_to_delta.py --dir experiments/ --remove-originals
    python scripts/data/migrate_parquet_to_delta.py --dir experiments/ --dry-run
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

SNAPSHOT_PATTERN = re.compile(r"^(.+)_v(\d+)\.parquet$")


def find_parquet_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.parquet"))


def group_snapshot_versions(
    files: list[Path],
) -> tuple[dict[Path, list[tuple[int, Path]]], list[Path]]:
    snapshot_groups: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
    standalone: list[Path] = []

    for f in files:
        match = SNAPSHOT_PATTERN.match(f.name)
        if match:
            name = match.group(1)
            version = int(match.group(2))
            table_path = f.parent / name
            snapshot_groups[table_path].append((version, f))
        else:
            standalone.append(f)

    for table_path in snapshot_groups:
        snapshot_groups[table_path].sort(key=lambda x: x[0])

    return dict(snapshot_groups), standalone


def migrate_standalone(
    storage, file_path: Path, remove_original: bool, dry_run: bool
) -> bool:
    table_path = file_path.parent / file_path.stem
    if table_path.is_dir() and (table_path / "_delta_log").is_dir():
        print(f"  SKIP {file_path} (delta table already exists)")
        return False

    if dry_run:
        print(f"  WOULD migrate {file_path} -> {table_path}/")
        return True

    df = pd.read_parquet(file_path)
    storage.write(df, str(table_path), metadata={"migrated_from": str(file_path)})
    print(f"  MIGRATED {file_path} -> {table_path}/ ({len(df):,} rows)")

    if remove_original:
        file_path.unlink()
        print(f"  REMOVED {file_path}")

    return True


def migrate_snapshot_group(
    storage,
    table_path: Path,
    versions: list[tuple[int, Path]],
    remove_originals: bool,
    dry_run: bool,
) -> int:
    if table_path.is_dir() and (table_path / "_delta_log").is_dir():
        print(f"  SKIP {table_path} (delta table already exists)")
        return 0

    if dry_run:
        print(
            f"  WOULD migrate {len(versions)} versions -> {table_path}/:"
        )
        for ver, path in versions:
            print(f"    v{ver}: {path.name}")
        return len(versions)

    migrated = 0
    for ver, path in versions:
        df = pd.read_parquet(path)
        mode = "overwrite" if migrated == 0 else "overwrite"
        metadata = {"snapshot_version": str(ver), "migrated_from": str(path)}
        storage.write(df, str(table_path), mode=mode, metadata=metadata)
        print(f"  MIGRATED {path.name} -> {table_path}/ v{migrated} ({len(df):,} rows)")
        migrated += 1

        if remove_originals:
            path.unlink()
            print(f"  REMOVED {path}")

    return migrated


def migrate_directory(
    directory: str, remove_originals: bool = False, dry_run: bool = False
) -> dict:
    from customer_retention.integrations.adapters.factory import get_delta

    storage = get_delta(force_local=True)
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    files = find_parquet_files(dir_path)
    if not files:
        print("No parquet files found.")
        return {"standalone_migrated": 0, "snapshot_versions_migrated": 0}

    print(f"Found {len(files)} parquet file(s) in {directory}\n")

    snapshot_groups, standalone = group_snapshot_versions(files)

    standalone_migrated = 0
    snapshot_versions_migrated = 0

    if standalone:
        print(f"Standalone files ({len(standalone)}):")
        for f in standalone:
            if migrate_standalone(storage, f, remove_originals, dry_run):
                standalone_migrated += 1

    if snapshot_groups:
        print(f"\nSnapshot groups ({len(snapshot_groups)}):")
        for table_path, versions in snapshot_groups.items():
            count = migrate_snapshot_group(
                storage, table_path, versions, remove_originals, dry_run
            )
            snapshot_versions_migrated += count

    action = "Would migrate" if dry_run else "Migrated"
    print(f"\n{action}: {standalone_migrated} standalone, "
          f"{snapshot_versions_migrated} snapshot versions")

    return {
        "standalone_migrated": standalone_migrated,
        "snapshot_versions_migrated": snapshot_versions_migrated,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate parquet files to Delta Lake tables"
    )
    parser.add_argument(
        "--dir",
        "-d",
        required=True,
        help="Directory to scan for parquet files",
    )
    parser.add_argument(
        "--remove-originals",
        action="store_true",
        help="Remove original parquet files after migration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )

    args = parser.parse_args()
    migrate_directory(args.dir, args.remove_originals, args.dry_run)


if __name__ == "__main__":
    main()
