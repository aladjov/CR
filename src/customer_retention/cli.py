"""
CLI commands for churnkit package.
"""

from __future__ import annotations

import sys
from pathlib import Path


def init_project() -> int:
    """CLI entry point for project initialization."""
    import argparse

    from customer_retention.generators.notebook_generator import (
        Platform,
        ProjectInitializer,
    )

    parser = argparse.ArgumentParser(
        description="Bootstrap a new customer retention project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create project in current directory
    churnkit-init

    # Create in specific directory
    churnkit-init --output ./my_churn_analysis

    # With customization
    churnkit-init --output ./my_project --name "Customer Churn Analysis"
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
    parser.add_argument(
        "--exploration-notebooks-path",
        default="exploration_notebooks",
        help="Path (folder or nested subpath) for exploration notebooks (default: exploration_notebooks)",
    )
    parser.add_argument(
        "--causal-notebooks-path",
        default="causal_notebooks",
        help="Path (folder or nested subpath) for causal-track notebooks (default: causal_notebooks)",
    )
    parser.add_argument(
        "--experiments-path",
        default="experiments",
        help="Path (folder name) for the experiments volume — set per cluster to enable parallel runs (default: experiments)",
    )
    parser.add_argument(
        "--playbooks-path",
        default="playbooks",
        help="Path (folder name) for the playbooks volume holding causal-track YAMLs (default: playbooks)",
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
        initializer = ProjectInitializer(
            project_name=args.name or output_dir.name,
            platforms=platforms,
            exploration_notebooks_path=args.exploration_notebooks_path,
            causal_notebooks_path=args.causal_notebooks_path,
            experiments_path=args.experiments_path,
            playbooks_path=args.playbooks_path,
        )
        initializer.initialize(str(output_dir))

        print("\n" + "=" * 50)
        print("Done!")
        print("=" * 50)
        print("\nNext steps:")
        print(f"  1. cd {output_dir}")
        print(f"  2. Add your data to {args.experiments_path}/data/")
        print(f"  3. Open {args.exploration_notebooks_path}/01_data_discovery.ipynb")
        print("  4. Set DATA_PATH to your data file")
        print("  5. Run all cells - auto-discovery will do the rest!")
        print(f"  6. (Optional, causal track) Edit {args.playbooks_path}/policies/")
        print(f"     and add per-playbook YAMLs to {args.playbooks_path}/")
        print(f"  7. (Causal track) Run {args.causal_notebooks_path}/c01_publish_definitions.ipynb")
        print()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(init_project())
