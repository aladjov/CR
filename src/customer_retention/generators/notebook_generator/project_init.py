import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProjectInitializer:
    project_name: str
    generate_orchestration: bool = False
    platforms: Optional[List[str]] = None

    def initialize(self, output_dir: str) -> Dict[str, any]:
        project_path = Path(output_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        self._create_directories(project_path)
        readme_path = self._create_readme(project_path)
        gitignore_path = self._create_gitignore(project_path)
        pyproject_path = self._create_pyproject(project_path)
        exploration_notebooks = self._copy_exploration_notebooks(project_path)
        if self.generate_orchestration:
            self._generate_orchestration(project_path)
        return {
            "readme_path": str(readme_path),
            "gitignore_path": str(gitignore_path),
            "pyproject_path": str(pyproject_path),
            "exploration_notebooks": exploration_notebooks,
        }

    def _create_directories(self, project_path: Path) -> None:
        directories = [
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
        for directory in directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)

    def _create_readme(self, project_path: Path) -> Path:
        readme_path = project_path / "README.md"
        readme_path.write_text(self._readme_content())
        return readme_path

    def _readme_content(self) -> str:
        return f"""# {self.project_name}

Customer retention analysis project using the customer-retention framework.

## Structure

### Code (version controlled)
- `exploration_notebooks/` - Interactive exploration notebooks
- `generated_pipelines/` - Auto-generated pipeline notebooks/scripts
  - `local/` - Local platform notebooks
  - `databricks/` - Databricks platform notebooks

### Data (gitignored)
- `experiments/` - All experiment outputs
  - `findings/` - Exploration findings (YAML files)
  - `data/` - Pipeline outputs (bronze/silver/gold layers)
  - `mlruns/` - MLflow experiment tracking
  - `feature_store/` - Feast feature store

## Getting Started

1. Place your data in `experiments/data/` or configure a data source
2. Run exploration notebooks to understand your data
3. Generate orchestration pipelines for production

## Usage

```python
from customer_retention.generators.notebook_generator import generate_orchestration_notebooks, Platform

results = generate_orchestration_notebooks(
    findings_path="experiments/findings/your_data_findings.yaml",
    output_dir="generated_pipelines",
    platforms=[Platform.LOCAL, Platform.DATABRICKS]
)
```
"""

    def _create_gitignore(self, project_path: Path) -> Path:
        gitignore_path = project_path / ".gitignore"
        gitignore_path.write_text(self._gitignore_content())
        return gitignore_path

    def _gitignore_content(self) -> str:
        return """.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
experiments/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
"""

    def _create_pyproject(self, project_path: Path) -> Path:
        pyproject_path = project_path / "pyproject.toml"
        pyproject_path.write_text(self._pyproject_content())
        return pyproject_path

    def _pyproject_content(self) -> str:
        return f"""[project]
name = "{self.project_name}"
version = "0.1.0"
description = "Customer retention analysis using customer-retention framework"
requires-python = ">=3.9"

dependencies = [
    "customer-retention",
    "pandas>=2.0",
    "jupyter>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1",
]
"""

    def _copy_exploration_notebooks(self, project_path: Path) -> List[str]:
        source_dir = self._get_exploration_source_dir()
        dest_dir = project_path / "exploration_notebooks"
        copied = []
        if source_dir and source_dir.exists():
            for notebook in source_dir.glob("*.ipynb"):
                dest_path = dest_dir / notebook.name
                shutil.copy2(notebook, dest_path)
                copied.append(str(dest_path))
        return copied

    def _get_exploration_source_dir(self) -> Optional[Path]:
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "exploration_notebooks",
            Path("exploration_notebooks"),
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return None

    def _generate_orchestration(self, project_path: Path) -> None:
        from . import Platform, generate_orchestration_notebooks
        platforms = [Platform(p) for p in (self.platforms or ["local", "databricks"])]
        output_dir = project_path / "generated_pipelines"
        generate_orchestration_notebooks(
            output_dir=str(output_dir),
            platforms=platforms,
        )


def initialize_project(
    output_dir: str,
    project_name: str,
    generate_orchestration: bool = False,
) -> Dict[str, any]:
    initializer = ProjectInitializer(
        project_name=project_name,
        generate_orchestration=generate_orchestration,
    )
    return initializer.initialize(output_dir)
