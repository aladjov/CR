"""Tests for FW-11 (§2.11): NB10's `run_pipeline` cell must iterate
`bronze_event_*` notebooks before `bronze_entity_*` so consumers do not
run before producers.

Default `sorted()` puts `bronze_entity_*` first (i < v in ASCII), causing
[TABLE_OR_VIEW_NOT_FOUND] on the first aggregated bronze when the event
notebook hasn't materialised the upstream table yet.

These tests pin the post-FW-11 invariant by reading the NB10 notebook
JSON and asserting:

* The bronze branch uses an explicit ordering key putting events first.
* Other stages keep plain alphabetical ordering.
* Dropping the ordering branch (regression) would be detected by the
  post-key sort being absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

NB_PATH = Path(__file__).resolve().parents[2] / (
    "exploration_notebooks/10_spec_generation.ipynb"
)


def _run_pipeline_cell_source() -> str:
    nb = json.loads(NB_PATH.read_text())
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "name='run_pipeline' id=8b659505" in src:
            return src
    raise AssertionError("NB10 cell `run_pipeline` (id=8b659505) not found")


class TestBronzeEventFirstOrdering:
    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _run_pipeline_cell_source()

    def test_bronze_branch_uses_event_first_key(self, src: str):
        """The bronze ordering key must put `bronze_event_*` first
        (priority 0) and `bronze_entity_*` after (priority 1)."""
        assert 'if _stage == "bronze":' in src
        assert 'startswith("bronze_event_")' in src
        # The ordering tuple `(0 if event else 1, n)` is the canonical key.
        assert "0 if n.startswith(\"bronze_event_\") else 1" in src

    def test_bronze_branch_emits_diagnostic_print(self, src: str):
        """The `[bronze order]` print is the operator-facing diagnostic
        proving the key landed before any `dbutils.notebook.run` fires."""
        assert "[bronze order]" in src

    def test_non_bronze_stages_keep_alphabetical_ordering(self, src: str):
        """Landing/silver/gold/training keep plain `sorted()` so that any
        regression that flips them to event-first ordering is caught."""
        # The else branch must contain the unchanged sorted() shape.
        assert "_notebooks = sorted(f.stem for f in _stage_dir.iterdir() if f.suffix == \".py\")" in src

    def test_ordering_simulation_puts_events_first(self):
        """End-to-end shape check: the documented ordering key must
        produce `[event_a, event_b, entity_a, entity_b]` when given a
        mixed list — not the alphabetical default."""
        names = [
            "bronze_entity_account",
            "bronze_event_orders",
            "bronze_entity_subscription",
            "bronze_event_cases",
        ]
        ordered = sorted(
            names,
            key=lambda n: (0 if n.startswith("bronze_event_") else 1, n),
        )
        assert ordered == [
            "bronze_event_cases",
            "bronze_event_orders",
            "bronze_entity_account",
            "bronze_entity_subscription",
        ]
