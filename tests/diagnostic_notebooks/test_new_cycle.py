from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "diagnostic_notebooks" / "new_cycle.py"
_spec = importlib.util.spec_from_file_location("new_cycle", _MODULE_PATH)
new_cycle = importlib.util.module_from_spec(_spec)
sys.modules["new_cycle"] = new_cycle
_spec.loader.exec_module(new_cycle)


def _setup_engagement(tmp_path: Path, framework_root: str | None = "/Workspace/Repos/x/y") -> Path:
    eng = tmp_path / "debug" / "engagement_test"
    eng.mkdir(parents=True)
    if framework_root is not None:
        (eng / ".engagement.yaml").write_text(f"framework_repo_root: {framework_root}\n")
    return eng


def _setup_templates(tmp_path: Path) -> None:
    diag = tmp_path / "diagnostic_notebooks"
    diag.mkdir(parents=True)
    nb = {
        "cells": [
            {"cell_type": "markdown", "id": "m1", "metadata": {}, "source": ["# cycle intro\n"]},
            {"cell_type": "code", "id": "c1", "metadata": {}, "execution_count": None,
             "outputs": [], "source": ["# @cr:code name='init' id=c1\nprint('init')\n"]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (diag / "cycle_template.ipynb").write_text(json.dumps(nb))
    (diag / "cycle_template.md").write_text("# Cycle NNN — <slug>\n")


def test_build_code_system_cell_has_correct_tag():
    cell = new_cycle.build_code_system_cell("/Workspace/Repos/x/y")
    src = "".join(cell["source"])
    assert src.startswith("# @cr:code_system name='framework_path' id=cr-syspath")
    assert '"/Workspace/Repos/x/y"' in src
    assert "sys.path.insert" in src


def test_instantiate_prepends_code_system_cell(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(new_cycle, "REPO_ROOT", tmp_path)
    _setup_engagement(tmp_path)
    _setup_templates(tmp_path)

    rc = new_cycle.main(["--engagement", "engagement_test", "--cycle", "5", "--slug", "foo"])
    assert rc == 0

    nb_path = tmp_path / "debug" / "engagement_test" / "cycles" / "005_foo.ipynb"
    md_path = tmp_path / "debug" / "engagement_test" / "fix_cycles" / "005_foo.md"
    assert nb_path.exists()
    assert md_path.exists()

    nb = json.loads(nb_path.read_text())
    assert nb["cells"][0]["id"] == "cr-syspath"
    assert "FRAMEWORK_REPO_ROOT" in "".join(nb["cells"][0]["source"])
    assert len(nb["cells"]) == 3


def test_instantiate_does_not_duplicate_code_system(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(new_cycle, "REPO_ROOT", tmp_path)
    _setup_engagement(tmp_path)
    _setup_templates(tmp_path)
    # Pre-populate template with a code_system cell already present
    nb_path = tmp_path / "diagnostic_notebooks" / "cycle_template.ipynb"
    nb = json.loads(nb_path.read_text())
    nb["cells"].insert(0, {
        "cell_type": "code", "id": "cr-syspath", "metadata": {},
        "execution_count": None, "outputs": [],
        "source": ["# @cr:code_system name='framework_path' id=cr-syspath\n"],
    })
    nb_path.write_text(json.dumps(nb))

    new_cycle.main(["--engagement", "engagement_test", "--cycle", "1", "--slug", "bar"])
    out = json.loads((tmp_path / "debug" / "engagement_test" / "cycles" / "001_bar.ipynb").read_text())
    assert sum(1 for c in out["cells"] if c["id"] == "cr-syspath") == 1


def test_missing_engagement_yaml_errors(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(new_cycle, "REPO_ROOT", tmp_path)
    _setup_engagement(tmp_path, framework_root=None)
    _setup_templates(tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        new_cycle.main(["--engagement", "engagement_test", "--cycle", "1", "--slug", "x"])


def test_existing_cycle_refuses_without_force(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(new_cycle, "REPO_ROOT", tmp_path)
    _setup_engagement(tmp_path)
    _setup_templates(tmp_path)
    new_cycle.main(["--engagement", "engagement_test", "--cycle", "1", "--slug", "x"])
    with pytest.raises(SystemExit, match="exists"):
        new_cycle.main(["--engagement", "engagement_test", "--cycle", "1", "--slug", "x"])
    new_cycle.main(["--engagement", "engagement_test", "--cycle", "1", "--slug", "x", "--force"])
