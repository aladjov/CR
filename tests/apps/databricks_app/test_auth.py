"""Tests for ``apps/databricks_app/src/auth.py``.

The auth module is deliberately importable without streamlit installed
(its lazy imports live inside try/except blocks). That means the env-var
fallback path is fully exercisable in this venv. The header / Streamlit-
native paths are exercised by source-text assertions because the test
venv intentionally omits streamlit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import auth  # type: ignore[import-not-found]

_AUTH_PY = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "databricks_app"
    / "src"
    / "auth.py"
)


class TestEnvVarFallback:
    def test_email_returns_lowercased_env_var(self, monkeypatch):
        monkeypatch.setenv("CR_USER", "Jane.Doe@CHURNKIT.com")
        assert auth.current_user_email() == "jane.doe@churnkit.com"

    def test_email_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("CR_USER", "  jane@churnkit.com  ")
        assert auth.current_user_email() == "jane@churnkit.com"

    def test_email_returns_none_when_no_source(self, monkeypatch):
        monkeypatch.delenv("CR_USER", raising=False)
        assert auth.current_user_email() is None

    def test_handle_extracts_local_part(self, monkeypatch):
        monkeypatch.setenv("CR_USER", "jane.doe@churnkit.com")
        assert auth.current_user_handle() == "jane.doe"

    def test_handle_falls_back_to_os_username(self, monkeypatch):
        monkeypatch.delenv("CR_USER", raising=False)
        # Whatever getpass returns -- non-empty string, not crashing
        handle = auth.current_user_handle()
        assert isinstance(handle, str)
        assert handle  # non-empty

    def test_handle_passes_through_email_without_at(self, monkeypatch):
        monkeypatch.setenv("CR_USER", "no-at-sign-string")
        assert auth.current_user_handle() == "no-at-sign-string"


class TestSourceShape:
    """Pin the resolution chain in source so refactors don't silently drop a step."""

    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _AUTH_PY.read_text(encoding="utf-8")

    def test_reads_x_forwarded_email_header(self, src):
        assert "X-Forwarded-Email" in src

    def test_reads_streamlit_experimental_user(self, src):
        assert "experimental_user" in src

    def test_reads_cr_user_env_var(self, src):
        assert "CR_USER" in src

    def test_streamlit_imports_are_lazy(self, src):
        # The module must load even when streamlit is absent. Top-level
        # (column-0) ``import streamlit`` would break that contract;
        # indented imports inside functions are the lazy pattern we want.
        for line in src.splitlines():
            if line.startswith("import streamlit") or line.startswith("from streamlit"):
                assert False, f"top-level streamlit import found: {line!r}"

    def test_resolution_order_documented(self, src):
        # The docstring pins the chain order so future edits keep it.
        idx_header = src.find("X-Forwarded-Email")
        idx_native = src.find("experimental_user")
        idx_env    = src.find("CR_USER")
        assert idx_header < idx_native < idx_env, (
            "resolution order in docstring must be header → streamlit → env"
        )
