from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "diagnostic_notebooks" / "check_privacy.py"
_spec = importlib.util.spec_from_file_location("check_privacy", _MODULE_PATH)
check_privacy = importlib.util.module_from_spec(_spec)
sys.modules["check_privacy"] = check_privacy
_spec.loader.exec_module(check_privacy)


@pytest.fixture
def denylist_file(tmp_path: Path) -> Path:
    path = tmp_path / "deny.yaml"
    path.write_text("terms:\n  - AcmeCorp\n  - internal_cat_xyz\nid_regexes: []\n")
    return path


def test_clean_file_exits_zero(tmp_path: Path, denylist_file: Path, capsys) -> None:
    clean = tmp_path / "a.py"
    clean.write_text("def foo():\n    return 42\n")
    rc = check_privacy.main(["--paths", str(clean), "--denylist", str(denylist_file)])
    assert rc == 0
    assert "clean" in capsys.readouterr().out


def test_term_hit_exits_one(tmp_path: Path, denylist_file: Path, capsys) -> None:
    dirty = tmp_path / "b.md"
    dirty.write_text("introduction\nacmecorp ships widgets\ntrailing\n")
    rc = check_privacy.main(["--paths", str(dirty), "--denylist", str(denylist_file)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "term:AcmeCorp" in out
    assert "b.md:2" in out


def test_default_crm_regex_hits(tmp_path: Path, capsys) -> None:
    dirty = tmp_path / "c.txt"
    dirty.write_text("reference 001Rn000010swgLIAQ here\n")
    rc = check_privacy.main(["--paths", str(dirty)])
    assert rc == 1
    assert "id:" in capsys.readouterr().out


def test_debug_paths_are_skipped(tmp_path: Path, monkeypatch, capsys) -> None:
    debug_file = tmp_path / "debug" / "engagement_a" / "note.md"
    debug_file.parent.mkdir(parents=True)
    debug_file.write_text("AcmeCorp internal\n")
    monkeypatch.chdir(tmp_path)
    rc = check_privacy.main(["--paths", "debug/engagement_a/note.md"])
    assert rc == 0
    assert "clean" in capsys.readouterr().out


def test_missing_args_errors(capsys) -> None:
    with pytest.raises(SystemExit):
        check_privacy.main([])


def test_load_denylist_handles_no_path() -> None:
    terms, regexes = check_privacy.load_denylist(None)
    assert terms == []
    assert regexes
