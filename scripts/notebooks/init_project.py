#!/usr/bin/env python3
"""
Bootstrap a new customer retention project from templates.

This script copies the template notebooks to your project directory
and optionally customizes them with your project-specific settings.

Usage:
    # Basic - copy to current directory
    python -m customer_retention.init

    # Specify output directory
    python -m customer_retention.init --output ./my_project

    # With customization
    python -m customer_retention.init --output ./my_project --name "Churn Analysis" --target churned

    # From command line
    python scripts/notebooks/init_project.py --output /path/to/my_project
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime


# Source directory relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def find_source_dir() -> Path:
    """Find the source directory containing exploration_notebooks."""
    # Check relative to script
    if (PROJECT_ROOT / "exploration_notebooks").exists():
        return PROJECT_ROOT

    # Check if installed as package
    try:
        import customer_retention
        pkg_dir = Path(customer_retention.__file__).parent.parent.parent
        if (pkg_dir / "exploration_notebooks").exists():
            return pkg_dir
    except ImportError:
        pass

    raise FileNotFoundError(
        "Could not find exploration_notebooks directory. "
        "Make sure you're running from the customer-retention project."
    )


def copy_notebooks(
    output_dir: Path,
    source_dir: Path,
    overwrite: bool = False,
    notebook_type: str = "all",
) -> list[Path]:
    """Copy notebooks to output directory.

    Args:
        output_dir: Where to copy notebooks
        source_dir: Source directory containing exploration_notebooks
        overwrite: Whether to overwrite existing files
        notebook_type: Which notebook set to copy ("exploration" or "all")

    Returns:
        List of copied notebook paths
    """
    copied = []

    # Define notebook sources and destinations
    notebook_dirs = {
        "exploration": ("exploration_notebooks", "exploration_notebooks"),
    }

    # Determine which directories to copy
    dirs_to_copy = ["exploration"] if notebook_type == "all" else [notebook_type]

    for dir_key in dirs_to_copy:
        if dir_key not in notebook_dirs:
            continue

        src_subdir, dst_subdir = notebook_dirs[dir_key]
        notebooks_src = source_dir / src_subdir
        notebooks_dst = output_dir / dst_subdir

        if not notebooks_src.exists():
            print(f"  Skipping {dir_key} (directory not found)")
            continue

        # Create output directory
        notebooks_dst.mkdir(parents=True, exist_ok=True)

        for notebook in sorted(notebooks_src.glob("*.ipynb")):
            dst = notebooks_dst / notebook.name

            if dst.exists() and not overwrite:
                print(f"  Skipping {notebook.name} (already exists)")
                continue

            shutil.copy2(notebook, dst)
            copied.append(dst)
            print(f"  Copied {notebook.name} → {dst_subdir}/")

    return copied


def customize_notebooks(
    notebooks: list[Path],
    project_name: str | None = None,
    target_column: str | None = None,
    id_column: str | None = None,
    data_path: str | None = None,
) -> None:
    """Customize notebook contents with project-specific values."""
    replacements = {}

    if target_column:
        replacements['"churned"'] = f'"{target_column}"'
        replacements["'churned'"] = f"'{target_column}'"
        replacements['TARGET_COLUMN = "churned"'] = f'TARGET_COLUMN = "{target_column}"'

    if id_column:
        replacements['"customer_id"'] = f'"{id_column}"'
        replacements["'customer_id'"] = f"'{id_column}'"
        replacements['ID_COLUMN = "customer_id"'] = f'ID_COLUMN = "{id_column}"'

    if data_path:
        replacements['DATA_PATH = "../data/your_customers.csv"'] = f'DATA_PATH = "{data_path}"'
        replacements['DATA_PATH = "../data/your_features.csv"'] = f'DATA_PATH = "{data_path}"'

    if not replacements:
        return

    for notebook_path in notebooks:
        content = notebook_path.read_text(encoding="utf-8")

        for old, new in replacements.items():
            content = content.replace(old, new)

        notebook_path.write_text(content, encoding="utf-8")

    print(f"  Customized {len(notebooks)} notebooks")


def create_project_structure(output_dir: Path) -> None:
    """Create basic project structure."""
    dirs = [
        "exploration_notebooks",
        "generated_pipelines/local",
        "generated_pipelines/databricks",
        "experiments/findings",
        "experiments/data/bronze",
        "experiments/data/silver",
        "experiments/data/gold",
        "experiments/data/models",
        "experiments/data/predictions",
        "experiments/mlruns",
        "experiments/feature_store",
    ]

    for d in dirs:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    # Create .gitkeep files for top-level directories
    for d in ["exploration_notebooks", "generated_pipelines", "experiments"]:
        gitkeep = output_dir / d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print(f"  Created project directories")


def create_config_template(
    output_dir: Path,
    project_name: str | None = None,
    target_column: str | None = None,
    id_column: str | None = None,
) -> None:
    """Create a project config file."""
    config = {
        "project": {
            "name": project_name or "My Retention Project",
            "created": datetime.now().isoformat(),
        },
        "data": {
            "target_column": target_column or "churned",
            "id_column": id_column or "customer_id",
            "reference_date": None,  # Set when running
        },
        "columns": {
            # Template - user fills in
            "identifiers": [],
            "datetime": [],
            "numeric_continuous": [],
            "numeric_discrete": [],
            "categorical_nominal": [],
            "categorical_ordinal": [],
            "binary": [],
        },
        "databricks": {
            "runtime": "17.3.x-cpu-ml-scala2.12",
            "hf_repo": None,
        }
    }

    config_path = output_dir / "configs" / "project_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"  Created config: configs/project_config.json")


def create_gitignore(output_dir: Path) -> None:
    """Create .gitignore for the project."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/
build/

# Virtual environments
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# Experiments (all generated data - gitignored)
experiments/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Secrets
.env
*.pem
credentials.json
"""

    gitignore_path = output_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content, encoding="utf-8")
        print("  Created .gitignore")


def create_readme(
    output_dir: Path,
    project_name: str | None = None,
) -> None:
    """Create a README for the new project."""
    name = project_name or "My Retention Project"

    readme_content = f"""# {name}

Customer retention analysis project bootstrapped from [customer-retention](https://github.com/your-org/customer-retention).

## Setup

```bash
# Install the customer-retention library
pip install customer-retention

# Or from source
pip install -e /path/to/CustomerRetention
```

## Workflow

This project follows a **dual-track workflow**: Exploration first, then Production.

### Step 1: Exploration (Learning & Discovery)

Start with exploration notebooks - they require minimal configuration:

```python
# In exploration_notebooks/01_data_discovery.ipynb
DATA_PATH = "experiments/data/your_customers.csv"  # ← Just provide this!
# Everything else is automatic
```

### Step 2: Production (Generated Pipelines)

Once exploration is complete, generate production pipelines:

```python
from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform

results = generate_orchestration_notebooks(
    findings_path="experiments/findings/your_data_findings.yaml",
    output_dir="generated_pipelines",
    platforms=[Platform.LOCAL, Platform.DATABRICKS]
)
```

## Project Structure

```
{name}/
├── exploration_notebooks/     # Interactive exploration (version controlled)
├── generated_pipelines/       # Auto-generated pipelines (version controlled)
│   ├── local/                 # Local platform notebooks
│   └── databricks/            # Databricks platform notebooks
└── experiments/               # All experiment outputs (gitignored)
    ├── findings/              # Exploration findings (YAML)
    ├── data/                  # Pipeline outputs (bronze/silver/gold)
    ├── mlruns/                # MLflow experiment tracking
    └── feature_store/         # Feast feature store
```

## Quick Start

1. Put your data in `experiments/data/`
2. Open `exploration_notebooks/01_data_discovery.ipynb`
3. Set `DATA_PATH` to your data file
4. Run all cells - the system will auto-discover everything!
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print("  Created README.md")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap a new customer retention project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create project in current directory
    python scripts/notebooks/init_project.py

    # Create in specific directory
    python scripts/notebooks/init_project.py --output ./my_churn_analysis

    # With customization
    python scripts/notebooks/init_project.py \\
        --output ./my_project \\
        --name "Customer Churn Analysis" \\
        --target is_churned \\
        --id user_id
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path.cwd(),
        help="Output directory for the new project (default: current directory)",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Project name",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default=None,
        help="Target column name (default: churned)",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="ID column name (default: customer_id)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to your data file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--notebooks",
        type=str,
        choices=["all", "exploration"],
        default="all",
        help="Which notebook set to copy (default: all)",
    )

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("Customer Retention Project Bootstrap")
    print(f"{'='*50}\n")

    try:
        source_dir = find_source_dir()
        print(f"Source: {source_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    output_dir = args.output.resolve()
    print(f"Output: {output_dir}\n")

    # Create project structure
    print("Creating project structure...")
    create_project_structure(output_dir)

    # Copy notebooks
    print("\nCopying notebooks...")
    notebooks = copy_notebooks(output_dir, source_dir, args.overwrite, args.notebooks)

    # Customize if options provided
    if args.target or args.id or args.data_path:
        print("\nCustomizing notebooks...")
        customize_notebooks(
            notebooks,
            project_name=args.name,
            target_column=args.target,
            id_column=args.id,
            data_path=args.data_path,
        )

    # Create config
    print("\nCreating configuration...")
    create_config_template(
        output_dir,
        project_name=args.name,
        target_column=args.target,
        id_column=args.id,
    )

    # Create .gitignore
    create_gitignore(output_dir)

    # Create README
    create_readme(output_dir, args.name)

    print(f"\n{'='*50}")
    print("Project bootstrapped successfully!")
    print(f"{'='*50}")
    print(f"\nNext steps:")
    print(f"  1. cd {output_dir}")
    print(f"  2. Add your data to experiments/data/")
    print(f"  3. Open exploration_notebooks/01_data_discovery.ipynb")
    print(f"  4. Set DATA_PATH to your data file")
    print(f"  5. Run all cells - auto-discovery will do the rest!")
    print()
    print("Tip: Exploration notebooks are visual & interactive.")
    print("     Generate production pipelines after exploration is complete.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
