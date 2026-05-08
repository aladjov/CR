"""Source-shape tests for the self-assignment data layer.

The live ``data.py`` runtime depends on streamlit and the Databricks SQL
connector, which are intentionally absent from this venv. Following the
pattern set by ``test_portfolio_totals.py``, we pin the SQL contract by
reading the source as text and asserting the load-bearing snippets are
present. This catches refactors that drop the lazy DDL, swap the MERGE
for a plain INSERT (race-prone), or rename one of the sentinel return
strings the UI branches on.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

_DATA_PY = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "databricks_app"
    / "src"
    / "data.py"
)


def _read() -> str:
    return _DATA_PY.read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    src = _read()
    match = re.search(
        rf"def {re.escape(name)}\([^)]*\)[^:]*:\n(.*?)(?:\n@|\ndef |\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert match, f"{name}() not found in data.py"
    return textwrap.dedent(match.group(1))


class TestTableDDL:
    def test_table_name_is_account_assignments(self):
        assert '_ASSIGNMENTS_TABLE = "account_assignments"' in _read()

    def test_create_table_uses_if_not_exists(self):
        body = _function_body("_ensure_assignments_table")
        assert "CREATE TABLE IF NOT EXISTS" in body

    def test_create_table_uses_delta(self):
        body = _function_body("_ensure_assignments_table")
        assert "USING DELTA" in body

    def test_schema_has_three_required_columns(self):
        body = _function_body("_ensure_assignments_table")
        for col in ("entity_id", "assigned_to", "assigned_at"):
            assert col in body, f"column {col!r} missing from CREATE TABLE"
        assert "STRING NOT NULL" in body
        assert "TIMESTAMP NOT NULL" in body

    def test_create_runs_at_most_once_per_session(self):
        # The session_state guard prevents firing DDL on every rerun --
        # without it, a busy dashboard issues hundreds of DDL statements
        # against the warehouse for no reason.
        body = _function_body("_ensure_assignments_table")
        assert "session_state" in body


class TestAssignmentsReader:
    def test_assignments_returns_two_columns(self):
        body = _function_body("assignments")
        assert "SELECT entity_id, assigned_to" in body

    def test_assignments_is_cached(self):
        # 10s TTL is a deliberate balance between freshness (CSMs see
        # each other's claims promptly) and warehouse load.
        src = _read()
        assert "@st.cache_data(ttl=10" in src

    def test_assignments_lazy_creates_table(self):
        body = _function_body("assignments")
        assert "_ensure_assignments_table" in body


class TestToggleAssignment:
    def test_returns_three_sentinel_strings(self):
        # The L4 button branches on these three exact return values.
        # Renaming any of them silently breaks the UI.
        src = _read()
        assert '_TOGGLE_ASSIGNED = "assigned"' in src
        assert '_TOGGLE_UNASSIGNED = "unassigned"' in src
        assert '_TOGGLE_CLAIMED_BY_OTHER = "claimed_by_other"' in src

    def test_rejects_empty_args(self):
        body = _function_body("toggle_assignment")
        assert "raise ValueError" in body

    def test_uses_merge_not_plain_insert_on_create_path(self):
        # Plain INSERT under a read-then-write pattern lets two CSMs
        # land duplicate rows when they click within the same second.
        # MERGE WHEN NOT MATCHED collapses one of the writes into a
        # no-op, which is why we surface ``claimed_by_other`` after
        # verifying the winner.
        body = _function_body("toggle_assignment")
        assert "MERGE INTO" in body
        assert "WHEN NOT MATCHED" in body

    def test_unassign_path_restricts_to_owner(self):
        # The DELETE must filter on ``assigned_to = :user`` so a stale
        # cache read can never let one CSM clear another's claim.
        body = _function_body("toggle_assignment")
        assert "DELETE FROM" in body
        assert "assigned_to = :user" in body

    def test_normalises_user_email(self):
        body = _function_body("toggle_assignment")
        assert ".strip().lower()" in body

    def test_verifies_winner_after_merge(self):
        # The post-MERGE SELECT is what lets us return ``claimed_by_other``
        # when a concurrent claim landed first.
        body = _function_body("toggle_assignment")
        # Two SELECTs in the body: pre-state read and post-MERGE verify.
        assert body.count("SELECT assigned_to FROM") >= 2


class TestAssignmentForLookup:
    def test_reads_through_cached_assignments(self):
        body = _function_body("assignment_for")
        assert "assignments()" in body

    def test_returns_none_for_missing_entity(self):
        body = _function_body("assignment_for")
        assert "return None" in body


class TestAccountIsHoldout:
    def test_reads_through_cached_account_explanation(self):
        # Piggy-backing on ``account_explanation`` (already cached + called
        # by customer_profile.render) means the button check fires zero
        # extra SQL while the L4 panel is open.
        body = _function_body("account_is_holdout")
        assert "account_explanation(" in body

    def test_returns_false_for_empty_or_missing_field(self):
        body = _function_body("account_is_holdout")
        assert "is_holdout" in body
        assert "return False" in body

    def test_handles_nan_safely(self):
        # The view can return NULL is_holdout for legacy snapshots; we must
        # not let pd.isna raise on non-NA-aware values.
        body = _function_body("account_is_holdout")
        assert "pd.isna" in body
