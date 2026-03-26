"""Generate requirements.txt and requirements-databricks.txt from pyproject.toml."""
from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

DATABRICKS_RUNTIME_PACKAGES = frozenset({
    "pyspark", "delta-spark", "databricks-sdk", "py4j",
    "mlflow", "mlflow-skinny",
    "numpy", "pandas", "pyarrow", "scipy",
    "scikit-learn", "xgboost", "lightgbm", "catboost", "hyperopt", "shap",
    "torch", "torchvision", "tensorflow",
    "matplotlib", "seaborn", "plotly",
    "ipykernel", "ipywidgets",
})

DATABRICKS_INCOMPATIBLE_PACKAGES = frozenset({
    "deltalake",
})

_DEP_NAME_RE = re.compile(r"^([a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)")


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _extract_package_name(dep: str) -> str | None:
    dep = dep.strip()
    if not dep or dep.startswith("#"):
        return None
    match = _DEP_NAME_RE.match(dep)
    return match.group(1).lower() if match else None


def _is_self_referential(dep: str, project_name: str) -> bool:
    name = _extract_package_name(dep)
    return name is not None and _normalize_name(name) == _normalize_name(project_name)


def _is_databricks_excluded(dep: str) -> bool:
    name = _extract_package_name(dep)
    if name is None:
        return False
    norm = _normalize_name(name)
    return norm in DATABRICKS_RUNTIME_PACKAGES or norm in DATABRICKS_INCOMPATIBLE_PACKAGES


def _collect_unique_deps(all_deps: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for dep in all_deps:
        name = _extract_package_name(dep)
        if name is None:
            continue
        norm = _normalize_name(name)
        if norm not in seen:
            seen.add(norm)
            unique.append(dep.strip())
    return unique


def generate_requirements(pyproject_path: Path) -> tuple[str, str]:
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    project_name = data.get("project", {}).get("name", "")
    core_deps = data.get("project", {}).get("dependencies", [])
    optional = data.get("project", {}).get("optional-dependencies", {})
    ml_deps = optional.get("ml", [])
    delta_deps = optional.get("delta", [])

    all_deps = list(core_deps) + list(delta_deps) + [
        d for d in ml_deps if not _is_self_referential(d, project_name)
    ]
    unique_deps = _collect_unique_deps(all_deps)

    header = "# Auto-generated from pyproject.toml — do not edit manually\n# Regenerate: python scripts/generate_requirements.py\n"
    req_all = header + "\n".join(unique_deps) + "\n"

    dbr_deps = [d for d in unique_deps if not _is_databricks_excluded(d)]
    dbr_header = (
        "# Auto-generated from pyproject.toml — do not edit manually\n"
        "# Excludes packages pre-installed in Databricks ML Runtime\n"
        "# Regenerate: python scripts/generate_requirements.py\n"
    )
    req_dbr = dbr_header + "\n".join(dbr_deps) + "\n"
    return req_all, req_dbr


def write_requirements(project_root: Path) -> tuple[Path, Path]:
    req_all, req_dbr = generate_requirements(project_root / "pyproject.toml")
    req_path = project_root / "requirements.txt"
    dbr_path = project_root / "requirements-databricks.txt"
    req_path.write_text(req_all)
    dbr_path.write_text(req_dbr)
    return req_path, dbr_path
