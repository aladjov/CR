#!/usr/bin/env python3
"""
Generate a filtered constraints file for uv from DBR baseline.

This script reads the captured runtime constraints and filters them
to only include packages listed in [tool.uv].no-install from pyproject.toml.

Usage:
    python generate_constraints.py \
        --runtime-constraints constraints/dbr-14.3-lts.txt \
        --pyproject pyproject.toml \
        --output constraints/filtered-14.3-lts.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]


def parse_constraints_file(path: Path) -> dict[str, str]:
    """Parse a constraints file and return package -> version mapping."""
    constraints: dict[str, str] = {}
    content = path.read_text(encoding="utf-8")

    for line in content.splitlines():
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Parse package==version format
        match = re.match(r"^([a-zA-Z0-9_-]+)\s*==\s*(.+)$", line)
        if match:
            package_name = match.group(1).lower().replace("-", "_")
            version = match.group(2)
            constraints[package_name] = version

    return constraints


def parse_pyproject(path: Path) -> list[str]:
    """Parse pyproject.toml and return the no-install package list."""
    content = path.read_bytes()
    data = tomllib.loads(content.decode("utf-8"))

    # Get [tool.uv].no-install list
    no_install = data.get("tool", {}).get("uv", {}).get("no-install", [])

    # Normalize package names
    return [pkg.lower().replace("-", "_") for pkg in no_install]


def filter_constraints(
    runtime_constraints: dict[str, str],
    protected_packages: list[str],
) -> dict[str, str]:
    """Filter runtime constraints to only include protected packages."""
    filtered: dict[str, str] = {}

    for pkg in protected_packages:
        normalized = pkg.lower().replace("-", "_")
        if normalized in runtime_constraints:
            filtered[pkg] = runtime_constraints[normalized]

    return filtered


def validate_packages(
    protected_packages: list[str],
    runtime_constraints: dict[str, str],
) -> list[str]:
    """Check which protected packages are missing from runtime."""
    missing = []
    for pkg in protected_packages:
        normalized = pkg.lower().replace("-", "_")
        if normalized not in runtime_constraints:
            missing.append(pkg)
    return missing


def generate_output(
    filtered_constraints: dict[str, str],
    runtime_path: str,
    pyproject_path: str,
) -> str:
    """Generate the filtered constraints file content."""
    lines = [
        "# Filtered constraints for uv",
        f"# Source: {runtime_path}",
        f"# Protected packages from: {pyproject_path}",
        "#",
        "# These packages will be pinned to DBR versions during uv pip install",
        "",
    ]

    # Sort by original package name
    for pkg, version in sorted(filtered_constraints.items()):
        lines.append(f"{pkg}=={version}")

    lines.append("")  # Trailing newline
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate filtered constraints file from DBR baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_constraints.py \\
        --runtime-constraints constraints/dbr-14.3-lts.txt \\
        --pyproject pyproject.toml \\
        --output constraints/filtered.txt

    python generate_constraints.py \\
        --runtime-constraints /dbfs/constraints/dbr-14.3-lts.txt \\
        --pyproject /Workspace/Repos/user/project/pyproject.toml \\
        --output /Workspace/Repos/user/project/constraints/filtered.txt
        """,
    )
    parser.add_argument(
        "--runtime-constraints", "-r",
        required=True,
        help="Path to captured runtime constraints file",
    )
    parser.add_argument(
        "--pyproject", "-p",
        required=True,
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for filtered constraints",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any protected package is missing from runtime",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational output",
    )

    args = parser.parse_args()

    runtime_path = Path(args.runtime_constraints)
    pyproject_path = Path(args.pyproject)
    output_path = Path(args.output)

    # Validate input files exist
    if not runtime_path.exists():
        print(f"Error: Runtime constraints file not found: {runtime_path}", file=sys.stderr)
        return 1

    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found: {pyproject_path}", file=sys.stderr)
        return 1

    # Parse inputs
    if not args.quiet:
        print(f"Reading runtime constraints from: {runtime_path}")
    runtime_constraints = parse_constraints_file(runtime_path)

    if not args.quiet:
        print(f"Reading protected packages from: {pyproject_path}")
    protected_packages = parse_pyproject(pyproject_path)

    if not protected_packages:
        print("Warning: No packages found in [tool.uv].no-install", file=sys.stderr)
        print("Add packages to protect to pyproject.toml under [tool.uv].no-install")

    # Validate packages
    missing = validate_packages(protected_packages, runtime_constraints)
    if missing:
        print(f"Warning: Protected packages not found in runtime: {', '.join(missing)}", file=sys.stderr)
        if args.strict:
            return 1

    # Filter constraints
    filtered = filter_constraints(runtime_constraints, protected_packages)

    if not args.quiet:
        print(f"Filtered to {len(filtered)} protected packages")

    # Generate output
    content = generate_output(
        filtered,
        str(runtime_path),
        str(pyproject_path),
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    if not args.quiet:
        print(f"Filtered constraints written to: {output_path}")
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
