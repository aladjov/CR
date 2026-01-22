"""
CLI commands for customer-retention package.
"""

from __future__ import annotations

import sys
from pathlib import Path


def init_project() -> int:
    """CLI entry point for project initialization."""
    import argparse

    from customer_retention.generators.notebook_generator import (
        ProjectInitializer,
        Platform,
    )

    parser = argparse.ArgumentParser(
        description="Bootstrap a new customer retention project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create project in current directory
    customer-retention-init

    # Create in specific directory
    customer-retention-init --output ./my_churn_analysis

    # With customization
    customer-retention-init --output ./my_project --name "Customer Churn Analysis"
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path.cwd(),
        help="Output directory (default: current)",
    )
    parser.add_argument("--name", "-n", help="Project name")
    parser.add_argument(
        "--platform",
        choices=["local", "databricks", "both"],
        default="both",
        help="Target platform (default: both)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("Customer Retention Project Bootstrap")
    print("=" * 50 + "\n")

    output_dir = args.output.resolve()
    print(f"Output: {output_dir}\n")

    # Determine platforms
    if args.platform == "both":
        platforms = [Platform.LOCAL, Platform.DATABRICKS]
    elif args.platform == "local":
        platforms = [Platform.LOCAL]
    else:
        platforms = [Platform.DATABRICKS]

    # Initialize project using library
    try:
        initializer = ProjectInitializer(output_dir)
        result = initializer.initialize(
            project_name=args.name,
            platforms=platforms,
        )

        print("\n" + "=" * 50)
        print("Done!")
        print("=" * 50)
        print(f"\nNext steps:")
        print(f"  1. cd {output_dir}")
        print(f"  2. Add your data to experiments/data/")
        print(f"  3. Open exploration_notebooks/01_data_discovery.ipynb")
        print(f"  4. Set DATA_PATH to your data file")
        print(f"  5. Run all cells - auto-discovery will do the rest!")
        print()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(init_project())
