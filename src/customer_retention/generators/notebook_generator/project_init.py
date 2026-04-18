import shutil
import site
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProjectInitializer:
    project_name: str
    generate_orchestration: bool = False
    platforms: Optional[List[str]] = None
    exploration_notebooks_path: str = "exploration_notebooks"
    causal_notebooks_path: str = "causal_notebooks"
    experiments_path: str = "experiments"
    playbooks_path: str = "playbooks"

    def initialize(self, output_dir: str) -> Dict[str, any]:
        project_path = Path(output_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        self._create_directories(project_path)
        readme_path = self._create_readme(project_path)
        gitignore_path = self._create_gitignore(project_path)
        pyproject_path = self._create_pyproject(project_path)
        exploration_notebooks = self._copy_exploration_notebooks(project_path)
        causal_notebooks = self._copy_causal_notebooks(project_path)
        seed_yamls_copied = self._copy_seed_yamls(project_path)
        if self.generate_orchestration:
            self._generate_orchestration(project_path)
        return {
            "readme_path": str(readme_path),
            "gitignore_path": str(gitignore_path),
            "pyproject_path": str(pyproject_path),
            "exploration_notebooks": exploration_notebooks,
            "causal_notebooks": causal_notebooks,
            "seed_yamls_copied": seed_yamls_copied,
        }

    def _create_directories(self, project_path: Path) -> None:
        exp = self.experiments_path
        pb = self.playbooks_path
        directories = [
            self.exploration_notebooks_path,
            self.causal_notebooks_path,
            "generated_pipelines/local",
            "generated_pipelines/databricks",
            f"{exp}/findings",
            f"{exp}/data/bronze",
            f"{exp}/data/silver",
            f"{exp}/data/gold",
            f"{exp}/data/models",
            f"{exp}/data/predictions",
            f"{exp}/mlruns",
            f"{exp}/feature_store",
            f"{pb}/policies",
        ]
        for directory in directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)

    def _create_readme(self, project_path: Path) -> Path:
        readme_path = project_path / "README.md"
        readme_path.write_text(self._readme_content())
        return readme_path

    def _readme_content(self) -> str:
        return f"""# {self.project_name}

Customer retention analysis project using the churnkit framework.

## Structure

### Code (version controlled)
- `{self.exploration_notebooks_path}/` - Interactive exploration notebooks
- `{self.causal_notebooks_path}/` - Causal-track orchestration notebooks
  (`c01_publish_definitions`, `c02_archetype_derivation`,
  `c03_approval_gate`, `c04_batch_inference`,
  `c05_snapshot_and_dashboard`)
- `generated_pipelines/` - Auto-generated pipeline notebooks/scripts
  - `local/` - Local platform notebooks
  - `databricks/` - Databricks platform notebooks

### Data (gitignored)
- `experiments/` - All experiment outputs
  - `findings/` - Exploration findings (YAML files)
  - `data/` - Pipeline outputs (bronze/silver/gold layers)
  - `mlruns/` - MLflow experiment tracking
  - `feature_store/` - Feast feature store
- `playbooks/` - Playbook YAML definitions for the causal track (gitignored
  intentionally; each deployment customizes its own private playbooks)
  - `*.yaml` - Per-playbook intervention design files
  - `policies/` - Decision policy, response schemas, controlled vocabularies
    (seeded from framework templates on first init)

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
        return f""".venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
{self.experiments_path}/
{self.playbooks_path}/
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
description = "Customer retention analysis using churnkit framework"
requires-python = ">=3.10"

dependencies = [
    "churnkit",
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
        dest_dir = project_path / self.exploration_notebooks_path
        return self._copy_notebook_dir(source_dir, dest_dir)

    def _copy_causal_notebooks(self, project_path: Path) -> List[str]:
        source_dir = self._get_causal_source_dir()
        dest_dir = project_path / self.causal_notebooks_path
        return self._copy_notebook_dir(source_dir, dest_dir)

    @staticmethod
    def _copy_notebook_dir(source_dir: Optional[Path], dest_dir: Path) -> List[str]:
        copied: List[str] = []
        if source_dir is None or not source_dir.exists():
            return copied
        dest_dir.mkdir(parents=True, exist_ok=True)
        for notebook in source_dir.glob("*.ipynb"):
            dest_path = dest_dir / notebook.name
            shutil.copy2(notebook, dest_path)
            copied.append(str(dest_path))
        return copied

    def _copy_seed_yamls(self, project_path: Path) -> List[str]:
        """Copy framework seed YAMLs into ``{playbooks_path}/policies/``.

        Source files live inside the framework package at
        ``customer_retention/stages/causal/seed_yamls/``. Existing destination
        files are NOT overwritten — once a deployment customizes its policies,
        re-running ``churnkit-init`` will not clobber the changes.
        """
        source_dir = self._get_seed_yamls_source_dir()
        if source_dir is None or not source_dir.exists():
            return []
        dest_dir = project_path / self.playbooks_path / "policies"
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied: List[str] = []
        for seed_file in source_dir.glob("*.yaml"):
            dest_path = dest_dir / seed_file.name
            if dest_path.exists():
                continue
            shutil.copy2(seed_file, dest_path)
            copied.append(str(dest_path))
        return copied

    def _get_seed_yamls_source_dir(self) -> Optional[Path]:
        # Development / editable install: framework package is in src/
        pkg_root = Path(__file__).parent.parent.parent
        candidate = pkg_root / "stages" / "causal" / "seed_yamls"
        if candidate.is_dir():
            return candidate
        # Installed package: resolve via importlib.resources
        try:
            from importlib.resources import files
            resolved = files("customer_retention.stages.causal") / "seed_yamls"
            resolved_path = Path(str(resolved))
            if resolved_path.is_dir():
                return resolved_path
        except (ImportError, ModuleNotFoundError, FileNotFoundError):
            pass
        return None

    def _get_exploration_source_dir(self) -> Optional[Path]:
        return self._resolve_packaged_notebook_dir(
            dir_name="exploration_notebooks",
            sentinel_basename="00_start_here.ipynb",
        )

    def _get_causal_source_dir(self) -> Optional[Path]:
        return self._resolve_packaged_notebook_dir(
            dir_name="causal_notebooks",
            sentinel_basename="c01_publish_definitions.ipynb",
        )

    def _resolve_packaged_notebook_dir(
        self, *, dir_name: str, sentinel_basename: str,
    ) -> Optional[Path]:
        # 1. Development / editable install: notebooks live in the source tree
        pkg_root = Path(__file__).parent.parent.parent.parent
        for candidate in (pkg_root, pkg_root.parent):
            dev_path = candidate / dir_name
            if dev_path.is_dir():
                return dev_path

        # 2. Installed package: resolve from package metadata RECORD.
        #    Works across pip, conda, uv, pipx — any PEP 376 compliant installer.
        sentinel_suffix = f"{dir_name}/{sentinel_basename}"
        try:
            from importlib.metadata import PackageNotFoundError, distribution
            dist = distribution("churnkit")
            for f in dist.files or []:
                if str(f).endswith(sentinel_suffix):
                    resolved = Path(dist.locate_file(f)).resolve()
                    if resolved.parent.is_dir():
                        return resolved.parent
        except (PackageNotFoundError, FileNotFoundError, TypeError):
            pass

        # 3. Fallback: shared-data location. Hatch's ``shared-data`` is installed
        #    by pip/uv to the wheel's ``data`` scheme — which is ``sysconfig.get_path('data')``,
        #    NOT always ``sys.prefix``. On Databricks ephemeral envs and ``--user``
        #    installs these diverge. Check every plausible data root.
        for data_root in _iter_shared_data_roots():
            candidate = data_root / "share" / "churnkit" / dir_name
            if candidate.is_dir():
                return candidate

        return None

    def _generate_orchestration(self, project_path: Path) -> None:
        from . import Platform, generate_orchestration_notebooks
        platforms = [Platform(p) for p in (self.platforms or ["local", "databricks"])]
        output_dir = project_path / "generated_pipelines"
        generate_orchestration_notebooks(
            output_dir=str(output_dir),
            platforms=platforms,
        )


def _iter_shared_data_roots() -> List[Path]:
    """Return every plausible ``data`` root where hatch ``shared-data`` may live.

    Hatch's ``shared-data`` entries are installed into the wheel's ``data``
    scheme. Which directory that is depends on the installer, the interpreter,
    and whether the install is system-wide, venv, ``--user``, or an ephemeral
    Databricks env — ``sys.prefix`` is only *sometimes* the right answer.
    """
    seen: set[str] = set()
    roots: List[Path] = []

    def _add(path: Optional[str]) -> None:
        if not path:
            return
        resolved = str(Path(path))
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(Path(path))

    _add(sysconfig.get_path("data"))
    for scheme in sysconfig.get_scheme_names():
        try:
            _add(sysconfig.get_paths(scheme).get("data"))
        except KeyError:
            continue
    _add(sys.prefix)
    _add(getattr(sys, "base_prefix", None))
    try:
        _add(site.getuserbase())
    except AttributeError:
        pass
    _add("/databricks/python")
    _add("/databricks/driver")
    return roots


def initialize_project(
    output_dir: str,
    project_name: str,
    generate_orchestration: bool = False,
    exploration_notebooks_path: str = "exploration_notebooks",
    causal_notebooks_path: str = "causal_notebooks",
    experiments_path: str = "experiments",
    playbooks_path: str = "playbooks",
) -> Dict[str, any]:
    initializer = ProjectInitializer(
        project_name=project_name,
        generate_orchestration=generate_orchestration,
        exploration_notebooks_path=exploration_notebooks_path,
        causal_notebooks_path=causal_notebooks_path,
        experiments_path=experiments_path,
        playbooks_path=playbooks_path,
    )
    return initializer.initialize(output_dir)
