#!/usr/bin/env python3
"""
Capture baseline packages from a Databricks Runtime cluster.

Run this script on a fresh DBR cluster (no additional packages installed)
to capture the baseline package versions as constraints.

Usage:
    python capture_runtime.py --output /Workspace/Repos/user/project/constraints/dbr-14.3-lts.txt
    python capture_runtime.py --output /dbfs/constraints/dbr-14.3-lts.txt --dbr-version 14.3-LTS
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def detect_dbr_version() -> str:
    """Detect the current Databricks Runtime version from environment."""
    # Try environment variable first
    dbr_version = os.environ.get("DATABRICKS_RUNTIME_VERSION")
    if dbr_version:
        return dbr_version

    # Try reading from spark conf (if running in Databricks)
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark:
            version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", None)
            if version:
                return version
    except ImportError:
        pass
    except Exception:
        pass

    # Check for Databricks Connect
    dbr_version = os.environ.get("DB_RUNTIME_VERSION")
    if dbr_version:
        return dbr_version

    return "unknown"


def get_python_version() -> str:
    """Get the current Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def capture_installed_packages() -> list[str]:
    """Capture all installed packages using pip freeze."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    packages = result.stdout.strip().split("\n")
    # Filter out editable installs and empty lines
    packages = [p for p in packages if p and not p.startswith("-e ")]
    return sorted(packages, key=lambda x: x.lower())


def generate_constraints_content(
    packages: list[str],
    dbr_version: str,
    python_version: str,
) -> str:
    """Generate constraints file content with metadata header."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# Databricks Runtime: {dbr_version}",
        f"# Captured: {timestamp}",
        f"# Python: {python_version}",
        "# DO NOT EDIT - Auto-generated baseline constraints",
        "#",
        "# This file captures the baseline package versions from a fresh DBR cluster.",
        "# Use this as constraints when installing additional packages to avoid conflicts.",
        "",
    ]
    lines.extend(packages)
    lines.append("")  # Trailing newline

    return "\n".join(lines)


def write_output(content: str, output_path: str) -> None:
    """Write content to the specified output path."""
    path = Path(output_path)

    # Handle DBFS paths
    if output_path.startswith("/dbfs"):
        # Direct file write works on DBFS mounted paths
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    elif output_path.startswith("/Workspace"):
        # Workspace paths work directly
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    else:
        # Local path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print(f"Constraints written to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capture baseline packages from Databricks Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Capture to workspace path
    python capture_runtime.py --output /Workspace/Repos/user/project/constraints/dbr-14.3-lts.txt

    # Capture to DBFS with explicit version
    python capture_runtime.py --output /dbfs/constraints/dbr-14.3-lts.txt --dbr-version 14.3-LTS

    # Capture to local path (for testing)
    python capture_runtime.py --output ./constraints/dbr-local.txt
        """,
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for the constraints file",
    )
    parser.add_argument(
        "--dbr-version",
        default=None,
        help="DBR version string (auto-detected if not provided)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational output",
    )

    args = parser.parse_args()

    # Detect versions
    dbr_version = args.dbr_version or detect_dbr_version()
    python_version = get_python_version()

    if not args.quiet:
        print(f"Databricks Runtime: {dbr_version}")
        print(f"Python Version: {python_version}")
        print("Capturing installed packages...")

    # Capture packages
    try:
        packages = capture_installed_packages()
    except subprocess.CalledProcessError as e:
        print(f"Error capturing packages: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Found {len(packages)} packages")

    # Generate content
    content = generate_constraints_content(packages, dbr_version, python_version)

    # Write output
    try:
        write_output(content, args.output)
    except OSError as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
