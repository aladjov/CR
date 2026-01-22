"""
Notebook-scoped dependency setup helper for Databricks.

This module provides a simple interface for installing the customer_retention
package from within a Databricks notebook, supporting both wheel and repo modes.

Usage in notebook:
    # Method 1: Run as script
    %run ./scripts/databricks/notebook_setup

    # Install from wheel (production)
    setup_from_wheel("/Volumes/catalog/schema/packages/")

    # Install from repo (development)
    setup_from_repo()

    # Auto-detect mode
    setup_dependencies()
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def _is_uv_installed() -> bool:
    """Check if uv is available."""
    try:
        subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _install_uv(quiet: bool = False) -> bool:
    """Install uv using pip."""
    if not quiet:
        print("Installing uv...")

    try:
        cmd = [sys.executable, "-m", "pip", "install", "uv"]
        if quiet:
            cmd.append("--quiet")

        subprocess.run(cmd, check=True, capture_output=quiet)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install uv: {e}", file=sys.stderr)
        return False


def _get_notebook_dir() -> Path | None:
    """Get the directory containing the current notebook."""
    try:
        from dbruntime.databricks_repl_context import get_context  # type: ignore[import-not-found]
        ctx = get_context()
        if ctx.notebookPath:
            # Notebook path is like /Workspace/Repos/user/project/notebooks/foo
            return Path(ctx.notebookPath).parent
    except (ImportError, Exception):
        pass
    return None


def _find_project_root(start_path: Path | None = None) -> Path | None:
    """Find project root by looking for pyproject.toml."""
    if start_path is None:
        start_path = _get_notebook_dir() or Path.cwd()

    current = start_path
    for _ in range(10):  # Max depth
        pyproject = current / "pyproject.toml"
        if pyproject.exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def _get_constraints_path(project_path: Path) -> Path | None:
    """Get constraints path from pyproject.toml or auto-detect."""
    if tomllib is None:
        return None

    pyproject_path = project_path / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    try:
        content = pyproject_path.read_bytes()
        data = tomllib.loads(content.decode("utf-8"))

        # Check for explicit path in [tool.databricks]
        constraints_rel = data.get("tool", {}).get("databricks", {}).get("constraints-path")
        if constraints_rel:
            constraints_path = project_path / constraints_rel
            if constraints_path.exists():
                return constraints_path

        # Auto-detect from constraints directory
        constraints_dir = project_path / "constraints"
        if constraints_dir.exists():
            constraint_files = list(constraints_dir.glob("dbr-*.txt"))
            if constraint_files:
                return max(constraint_files, key=lambda p: p.stat().st_mtime)

    except Exception:
        pass

    return None


def _find_wheel(wheel_path: str | Path) -> Path | None:
    """Find wheel file from path or directory."""
    path = Path(wheel_path)

    # Direct wheel file
    if path.suffix == ".whl" and path.exists():
        return path

    # Directory - find latest customer_retention wheel
    if path.is_dir():
        wheels = sorted(
            path.glob("customer_retention-*.whl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if wheels:
            return wheels[0]

    return None


def _run_uv_install(
    target: str,
    constraints_path: Path | None = None,
    editable: bool = False,
    quiet: bool = False,
) -> tuple[bool, list[str]]:
    """Run uv pip install with given target."""
    cmd = ["uv", "pip", "install", "--system"]

    if constraints_path and constraints_path.exists():
        cmd.extend(["-c", str(constraints_path)])

    if editable:
        cmd.extend(["-e", target])
    else:
        cmd.append(target)

    if quiet:
        cmd.append("--quiet")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if not quiet and result.stdout:
            print(result.stdout)
        return True, []
    except subprocess.CalledProcessError as e:
        errors = e.stderr.splitlines() if e.stderr else [str(e)]
        return False, errors


def _verify_import() -> tuple[bool, str]:
    """Verify customer_retention can be imported."""
    try:
        # Force reimport
        if "customer_retention" in sys.modules:
            del sys.modules["customer_retention"]

        import customer_retention
        return True, getattr(customer_retention, "__version__", "unknown")
    except ImportError as e:
        return False, str(e)


def setup_from_wheel(
    wheel_path: str | Path,
    constraints_path: str | Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Install customer_retention from a wheel file.

    Args:
        wheel_path: Path to wheel file or directory containing wheels.
        constraints_path: Path to constraints file (optional).
        quiet: Suppress output if True.

    Returns:
        Dictionary with success status and details.
    """
    result: dict[str, Any] = {
        "success": False,
        "mode": "wheel",
        "wheel_path": None,
        "constraints_path": None,
        "version": None,
        "errors": [],
    }

    # Ensure uv
    if not _is_uv_installed():
        if not _install_uv(quiet):
            result["errors"].append("Failed to install uv")
            return result

    # Find wheel
    wheel = _find_wheel(wheel_path)
    if wheel is None:
        result["errors"].append(f"No wheel found at: {wheel_path}")
        if not quiet:
            print(f"Error: No wheel found at {wheel_path}", file=sys.stderr)
        return result

    result["wheel_path"] = str(wheel)
    if not quiet:
        print(f"Installing from wheel: {wheel}")

    # Get constraints
    constr_path = Path(constraints_path) if constraints_path else None
    if constr_path and constr_path.exists():
        result["constraints_path"] = str(constr_path)

    # Install
    success, errors = _run_uv_install(
        str(wheel),
        constraints_path=constr_path,
        quiet=quiet,
    )

    result["success"] = success
    result["errors"].extend(errors)

    if success:
        ok, version = _verify_import()
        result["version"] = version if ok else None
        if not quiet:
            print(f"Installed customer_retention version: {version}")

    return result


def setup_from_repo(
    project_path: str | Path | None = None,
    constraints_path: str | Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Install customer_retention from a workspace repo (editable mode).

    Args:
        project_path: Path to project root. Auto-detected if None.
        constraints_path: Path to constraints file. Auto-detected if None.
        quiet: Suppress output if True.

    Returns:
        Dictionary with success status and details.
    """
    result: dict[str, Any] = {
        "success": False,
        "mode": "repo",
        "project_path": None,
        "constraints_path": None,
        "version": None,
        "errors": [],
    }

    # Ensure uv
    if not _is_uv_installed():
        if not _install_uv(quiet):
            result["errors"].append("Failed to install uv")
            return result

    # Find project root
    if project_path is not None:
        proj_path = Path(project_path)
    else:
        proj_path = _find_project_root()

    if proj_path is None:
        result["errors"].append("Could not find project root (pyproject.toml not found)")
        if not quiet:
            print("Error: Could not find project root", file=sys.stderr)
        return result

    result["project_path"] = str(proj_path)
    if not quiet:
        print(f"Project root: {proj_path}")

    # Get constraints
    if constraints_path is not None:
        constr_path: Path | None = Path(constraints_path)
    else:
        constr_path = _get_constraints_path(proj_path)

    if constr_path:
        result["constraints_path"] = str(constr_path)
        if not quiet:
            print(f"Using constraints: {constr_path}")

    # Install editable
    if not quiet:
        print("Installing in editable mode...")

    success, errors = _run_uv_install(
        str(proj_path),
        constraints_path=constr_path,
        editable=True,
        quiet=quiet,
    )

    result["success"] = success
    result["errors"].extend(errors)

    if success:
        ok, version = _verify_import()
        result["version"] = version if ok else None
        if not quiet:
            print(f"Installed customer_retention version: {version}")

    return result


def setup_dependencies(
    wheel_path: str | Path | None = None,
    project_path: str | Path | None = None,
    constraints_path: str | Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Auto-detect and install customer_retention.

    Priority:
    1. If wheel_path provided, use wheel mode
    2. If project_path provided or can be detected, use repo mode
    3. Check DBR_WHEEL_PATH env var
    4. Check DBR_PROJECT_PATH env var

    Args:
        wheel_path: Path to wheel file or directory (optional).
        project_path: Path to project root (optional).
        constraints_path: Path to constraints file (optional).
        quiet: Suppress output if True.

    Returns:
        Dictionary with success status and details.
    """
    # Check for explicit wheel path
    if wheel_path is not None:
        return setup_from_wheel(wheel_path, constraints_path, quiet)

    # Check for explicit project path
    if project_path is not None:
        return setup_from_repo(project_path, constraints_path, quiet)

    # Check environment variables
    env_wheel = os.environ.get("DBR_WHEEL_PATH")
    if env_wheel:
        if not quiet:
            print(f"Using wheel from DBR_WHEEL_PATH: {env_wheel}")
        return setup_from_wheel(env_wheel, constraints_path, quiet)

    env_project = os.environ.get("DBR_PROJECT_PATH")
    if env_project:
        if not quiet:
            print(f"Using repo from DBR_PROJECT_PATH: {env_project}")
        return setup_from_repo(env_project, constraints_path, quiet)

    # Try to auto-detect project root
    proj_path = _find_project_root()
    if proj_path:
        return setup_from_repo(proj_path, constraints_path, quiet)

    # Nothing found
    return {
        "success": False,
        "mode": None,
        "errors": [
            "Could not detect installation mode.",
            "Provide wheel_path, project_path, or set DBR_WHEEL_PATH/DBR_PROJECT_PATH.",
        ],
    }


# Aliases for convenience
install_wheel = setup_from_wheel
install_repo = setup_from_repo
setup = setup_dependencies

__all__ = [
    "setup_dependencies",
    "setup_from_wheel",
    "setup_from_repo",
    "install_wheel",
    "install_repo",
    "setup",
]
