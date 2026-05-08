"""Tests for the In-scope table's ``Assigned to`` column.

We exercise the pure label helper directly (no streamlit needed) and
pin the surrounding source contract via text assertions: the column
must be merged in, rendered, and configured.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

_ACCOUNTS_VIEW_PY = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "databricks_app"
    / "src"
    / "accounts_view.py"
)


def _load_assignment_label():
    """Load ``_assignment_label`` without dragging in streamlit."""
    src = _ACCOUNTS_VIEW_PY.read_text(encoding="utf-8")
    for name in ("streamlit",):
        sys.modules.setdefault(name, types.ModuleType(name))
    sanitised = []
    for line in src.splitlines(keepends=True):
        if line.startswith("import streamlit"):
            continue
        if line.startswith("from . import "):
            continue
        sanitised.append(line)
    code = "".join(sanitised)
    spec = importlib.util.spec_from_loader("accounts_view_under_test", loader=None)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    mod.__dict__["pd"] = pd
    exec(compile(code, str(_ACCOUNTS_VIEW_PY), "exec"), mod.__dict__)
    return mod._assignment_label


_assignment_label = _load_assignment_label()


class TestAssignmentLabel:
    def test_unassigned_renders_blank(self):
        assert _assignment_label(None, "jane@churnkit.com") == ""

    def test_nan_renders_blank(self):
        assert _assignment_label(float("nan"), "jane@churnkit.com") == ""

    def test_empty_string_renders_blank(self):
        assert _assignment_label("", "jane@churnkit.com") == ""
        assert _assignment_label("   ", "jane@churnkit.com") == ""

    def test_self_renders_as_mine(self):
        assert _assignment_label("jane@churnkit.com", "jane@churnkit.com") == "Mine"

    def test_self_match_is_case_insensitive(self):
        assert _assignment_label("JANE@CHURNKIT.com", "jane@churnkit.com") == "Mine"

    def test_other_renders_as_handle(self):
        assert _assignment_label("alex@churnkit.com", "jane@churnkit.com") == "alex"

    def test_other_without_at_passes_through(self):
        assert _assignment_label("alex", "jane@churnkit.com") == "alex"

    def test_no_current_user_still_renders_other_as_handle(self):
        # When auth fails we still want to show who has the account.
        assert _assignment_label("alex@churnkit.com", None) == "alex"


class TestSourceShape:
    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _ACCOUNTS_VIEW_PY.read_text(encoding="utf-8")

    def test_imports_auth_and_data(self, src):
        assert "from . import auth, data, state" in src

    def test_left_merges_assignments_on_entity_id(self, src):
        assert "data.assignments()" in src
        assert 'on="entity_id"' in src
        assert 'how="left"' in src

    def test_assigned_to_is_in_display_columns(self, src):
        # The new column must be projected into ``display`` so the
        # dataframe actually receives it.
        assert '"assigned_to"' in src

    def test_assigned_to_column_is_renamed(self, src):
        assert '"assigned_to":' in src and '"Assigned to"' in src

    def test_assigned_to_column_has_config(self, src):
        assert '"Assigned to":' in src

    def test_handles_empty_assignments_gracefully(self, src):
        # An empty assignments frame must not break the merge.
        assert "assigns is None or assigns.empty" in src
