"""File-based discovery fallback for ``CR_PROFILE_TEMPLATE_PATH``.

The Databricks App runs in a runtime where env vars are not always set
(notably on multi-task jobs and clusters where operators don't manage env).
``load_config`` therefore falls back to scanning the conventional volume path
``/Volumes/{catalog}/{schema}/dashboard_templates/`` for the newest ``.html``
file when the env var is unset.

These tests pin the contract: env var wins, file-scan picks the newest, and a
missing/empty directory degrades gracefully to "" (so the app keeps using its
bundled default template).
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import pytest

from src import config as cfgmod  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("CR_PROFILE_TEMPLATE_PATH", raising=False)
    monkeypatch.delenv("CR_CATALOG", raising=False)
    monkeypatch.delenv("CR_SCHEMA", raising=False)
    monkeypatch.delenv("CR_WAREHOUSE_ID", raising=False)


def test_env_var_wins_over_file_scan(monkeypatch):
    monkeypatch.setenv("CR_PROFILE_TEMPLATE_PATH", "/explicit/template.html")
    with mock.patch.object(cfgmod, "_discover_profile_template") as discover:
        cfg = cfgmod.load_config()
    discover.assert_not_called()
    assert cfg.profile_template_path == "/explicit/template.html"


def test_empty_env_var_triggers_file_scan(monkeypatch):
    monkeypatch.setenv("CR_PROFILE_TEMPLATE_PATH", "")
    with mock.patch.object(
        cfgmod, "_discover_profile_template", return_value="/scanned/template.html"
    ) as discover:
        cfg = cfgmod.load_config()
    discover.assert_called_once_with("churnkit", "analysis")
    assert cfg.profile_template_path == "/scanned/template.html"


def test_unset_env_var_triggers_file_scan(monkeypatch):
    with mock.patch.object(
        cfgmod, "_discover_profile_template", return_value="/scanned/x.html"
    ) as discover:
        cfg = cfgmod.load_config()
    discover.assert_called_once_with("churnkit", "analysis")
    assert cfg.profile_template_path == "/scanned/x.html"


def test_discover_returns_newest_html(tmp_path, monkeypatch):
    cat, sch = "cat42", "sch42"
    tdir = tmp_path / "Volumes" / cat / sch / "dashboard_templates"
    tdir.mkdir(parents=True)
    older = tdir / "older.html"
    older.write_text("<html>old</html>", encoding="utf-8")
    time.sleep(0.01)
    newer = tdir / "newer.html"
    newer.write_text("<html>new</html>", encoding="utf-8")

    def fake_path(arg):
        # Re-root any /Volumes/... lookup under tmp_path
        if str(arg).startswith("/Volumes/"):
            return tmp_path / str(arg).lstrip("/")
        return Path(arg)

    with mock.patch.object(cfgmod, "Path", side_effect=fake_path):
        result = cfgmod._discover_profile_template(cat, sch)
    assert result.endswith("newer.html")


def test_discover_handles_missing_volume(tmp_path, monkeypatch):
    def fake_path(arg):
        if str(arg).startswith("/Volumes/"):
            return tmp_path / "does_not_exist"
        return Path(arg)

    with mock.patch.object(cfgmod, "Path", side_effect=fake_path):
        result = cfgmod._discover_profile_template("cat", "sch")
    assert result == ""


def test_discover_handles_empty_dir(tmp_path):
    cat, sch = "c", "s"
    tdir = tmp_path / "Volumes" / cat / sch / "dashboard_templates"
    tdir.mkdir(parents=True)

    def fake_path(arg):
        if str(arg).startswith("/Volumes/"):
            return tmp_path / str(arg).lstrip("/")
        return Path(arg)

    with mock.patch.object(cfgmod, "Path", side_effect=fake_path):
        result = cfgmod._discover_profile_template(cat, sch)
    assert result == ""


def test_discover_ignores_non_html(tmp_path):
    cat, sch = "c", "s"
    tdir = tmp_path / "Volumes" / cat / sch / "dashboard_templates"
    tdir.mkdir(parents=True)
    (tdir / "readme.md").write_text("notes")
    (tdir / "schema.sql").write_text("select 1")
    (tdir / "real.html").write_text("<html></html>")

    def fake_path(arg):
        if str(arg).startswith("/Volumes/"):
            return tmp_path / str(arg).lstrip("/")
        return Path(arg)

    with mock.patch.object(cfgmod, "Path", side_effect=fake_path):
        result = cfgmod._discover_profile_template(cat, sch)
    assert result.endswith("real.html")


def test_discover_uses_resolved_catalog_schema(monkeypatch):
    monkeypatch.setenv("CR_CATALOG", "alt_cat")
    monkeypatch.setenv("CR_SCHEMA", "alt_sch")
    with mock.patch.object(cfgmod, "_discover_profile_template", return_value="") as discover:
        cfgmod.load_config()
    discover.assert_called_once_with("alt_cat", "alt_sch")
