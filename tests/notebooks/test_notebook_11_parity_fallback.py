"""AST-level plumbing assertions for NB11's parity-report spec discovery.

The original `load_parity_inputs` cell raised FileNotFoundError whenever
`_parity_namespace.feature_spec_path` was missing — which happens whenever
production training ran in a different run_id than NB08 (the common case
after a user re-runs NB00 but keeps earlier exploration artifacts).

The current cell must try, in order:
  1. the current run's `feature_spec_path`
  2. `production_diagnostics.feature_spec_source_path` (populated by the
     regenerated Databricks training template)
  3. `production_diagnostics.exploration_run_id` → that run's spec

and only raise if all three fail. Keeping this as an AST test rather than a
full notebook execution avoids the papermill overhead while still catching
regressions where someone accidentally drops the fallback.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NB_PATH = (
    Path(__file__).parent.parent.parent
    / "exploration_notebooks"
    / "11_scoring_validation.ipynb"
)

CELL_ID = "3e25cce6"


def _load_cell_source(cell_id: str) -> str:
    nb = json.loads(NB_PATH.read_text())
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            src = cell["source"]
            return "".join(src) if isinstance(src, list) else src
    raise AssertionError(f"cell id={cell_id!r} not found in {NB_PATH.name}")


@pytest.fixture(scope="module")
def cell_source() -> str:
    return _load_cell_source(CELL_ID)


@pytest.fixture(scope="module")
def cell_ast(cell_source: str) -> ast.Module:
    return ast.parse(cell_source)


class TestParityCellFallbackChain:
    def test_cell_parses(self, cell_ast: ast.Module):
        assert isinstance(cell_ast, ast.Module)

    def test_reads_prod_diag_when_spec_missing(self, cell_source: str):
        assert "_prod_path.exists()" in cell_source
        assert "_parity_json.loads(_prod_path.read_text())" in cell_source

    def test_prefers_feature_spec_source_path_hint(self, cell_source: str):
        """The regenerated training template records the baked spec location in
        `production_diagnostics.feature_spec_source_path`. NB11 reads it first
        because it's the direct pointer the training used."""
        assert '"feature_spec_source_path"' in cell_source
        assert '_prod_diag.get("feature_spec_source_path")' in cell_source

    def test_falls_back_to_exploration_run_id(self, cell_source: str):
        """When the baked path is gone (deleted, cross-volume copy), use the
        exploration_run_id to construct a `RunNamespace` under the same root."""
        assert '_prod_diag.get("exploration_run_id")' in cell_source
        assert "_ParityRunNamespace(root=_parity_namespace.root" in cell_source
        assert "_expl_ns.feature_spec_path.exists()" in cell_source

    def test_error_message_names_all_remedies(self, cell_source: str):
        """Fail-closed error must point at both remedies (run NB08 vs.
        regenerate+re-run training) so the user isn't guessing."""
        assert "NB08" in cell_source
        assert "NB10" in cell_source
        assert "feature_spec_source_path" in cell_source

    def test_success_path_sets_spec_path_not_mutated_when_current_exists(self, cell_source: str):
        """The prod_diag fallback block must only run when the current-run spec
        is missing — no risk of accidentally pointing at an older run when the
        current one is self-consistent."""
        assert "not _spec_path.exists() and _prod_path.exists()" in cell_source

    def test_scans_sibling_runs_when_prod_diag_missing(self, cell_source: str):
        """The user's repeated failure mode: current run has neither spec NOR
        production_diagnostics.json (training failed before writing it). The
        scan fallback must run independently of prod_diag, globbing
        `runs/*/merged/feature_spec.yaml`."""
        assert '_runs_root = _parity_namespace.root / "runs"' in cell_source
        assert '_runs_root.glob("*/merged/feature_spec.yaml")' in cell_source

    def test_scan_picks_most_recent_by_mtime(self, cell_source: str):
        """When multiple sibling runs have specs, pick the most recently
        modified — matches `RunNamespace.from_latest` heuristic so the
        fallback aligns with user mental model ('the run I just finished')."""
        assert "sort(key=lambda p: p.stat().st_mtime, reverse=True)" in cell_source

    def test_error_lists_all_searched_paths(self, cell_source: str):
        """On fail-closed, the error must enumerate every path the cell tried,
        so the user can see exactly where the lookup went."""
        assert "_searched" in cell_source
        assert "Searched (in order)" in cell_source
