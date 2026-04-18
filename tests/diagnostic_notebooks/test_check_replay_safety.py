from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "diagnostic_notebooks" / "check_replay_safety.py"
_spec = importlib.util.spec_from_file_location("check_replay_safety", _MODULE_PATH)
check_replay_safety = importlib.util.module_from_spec(_spec)
sys.modules["check_replay_safety"] = check_replay_safety
_spec.loader.exec_module(check_replay_safety)


def test_module_path_for_src_file() -> None:
    p = Path("src/customer_retention/core/helpers.py")
    assert check_replay_safety.module_path_for(p) == "customer_retention.core.helpers"


def test_module_path_for_non_src_returns_none() -> None:
    assert check_replay_safety.module_path_for(Path("tests/x.py")) is None
    assert check_replay_safety.module_path_for(Path("notes.ipynb")) is None


def test_search_needles_for_py(tmp_path: Path) -> None:
    p = Path("src/customer_retention/core/helpers.py")
    needles = check_replay_safety.search_needles(p)
    assert "customer_retention.core.helpers" in needles
    assert "helpers" in needles


def test_search_needles_for_notebook() -> None:
    needles = check_replay_safety.search_needles(Path("exploration_notebooks/05_x.ipynb"))
    assert "05_x" in needles


def test_file_contains_any(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("from customer_retention.core.helpers import foo\n")
    assert check_replay_safety.file_contains_any(p, ["customer_retention.core.helpers"])
    assert not check_replay_safety.file_contains_any(p, ["unrelated"])


def test_build_reach_chains_finds_reference(tmp_path: Path) -> None:
    src = tmp_path / "src" / "customer_retention"
    src.mkdir(parents=True)
    target = src / "target.py"
    target.write_text("def foo():\n    return 1\n")
    referrer = src / "referrer.py"
    referrer.write_text("from customer_retention.target import foo\n")

    chains = check_replay_safety.build_reach_chains(
        changed_files=[Path("src/customer_retention/target.py")],
        scan_files=[target, referrer],
        repo_root=tmp_path,
    )
    assert len(chains) == 1
    assert Path("src/customer_retention/referrer.py") in chains[0].referenced_by


def test_build_reach_chains_excludes_self(tmp_path: Path) -> None:
    src = tmp_path / "src" / "customer_retention"
    src.mkdir(parents=True)
    target = src / "target.py"
    target.write_text("import customer_retention.target\n")  # self-reference
    chains = check_replay_safety.build_reach_chains(
        changed_files=[Path("src/customer_retention/target.py")],
        scan_files=[target],
        repo_root=tmp_path,
    )
    assert chains[0].referenced_by == []
