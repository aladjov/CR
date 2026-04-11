"""AST-level plumbing assertions for NB05 time-slice routing.

Covers docs/nb05_time_slice_relationship_plan.md sections 6.2–6.6. Each test
loads exploration_notebooks/05_relationship_analysis.ipynb, finds a target cell
by its cr:code ``id=`` marker, parses the cell source, and asserts which
DataFrame variable (``df`` vs ``df_selection``) flows into each bulk-stat
call. AST scanning is deliberate: it avoids the papermill overhead of
tests/test_notebooks.py (which currently points at a stale template path) and
makes the invariants trivially diff-reviewable.

Sections skipped pending runnable fixture infrastructure:
  - 6.6 job-count delta baseline (requires .cr_cell_metrics sidecar)
  - 6.7 parity local ↔ distributed (requires synthetic snapshot-grid fixture)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NB_PATH = (
    Path(__file__).parent.parent.parent
    / "exploration_notebooks"
    / "05_relationship_analysis.ipynb"
)

TARGET_CELLS = {
    "cache_analysis_df": "d2b9dcce",
    "build_selection_slice": None,  # resolved dynamically — ID minted at insert time
    "precompute_stats": "a7c3e91f",
    "check_leaking_features": "ecad955f",
    "relationship_config": "a3923181",
    "compute_correlations": "1fa22c14",
    "plot_feature_distributions": "ae2926c9",
    "compute_feature_importance": "f6986cb3",
    "analyze_interactions": "81ec4d57",
    "analyze_categorical_target": "3e26270f",
    "generate_recommendations": "ce9806ee",
    "run_statistical_feature_selection": "2e6e3e63",
    "release_stage_memory": "d16b656b",
}


def _load_cells() -> dict[str, str]:
    nb = json.loads(NB_PATH.read_text())
    by_id: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        cid = c.get("id", "")
        if cid:
            by_id[cid] = src
        first = src.splitlines()[0] if src.strip() else ""
        if "name='" in first:
            name = first.split("name='", 1)[1].split("'", 1)[0]
            by_name[name] = src
    return {"by_id": by_id, "by_name": by_name}  # type: ignore[return-value]


@pytest.fixture(scope="module")
def cells() -> dict[str, dict[str, str]]:
    return _load_cells()  # type: ignore[return-value]


def _cell_source(cells: dict, name: str) -> str:
    by_name = cells["by_name"]
    if name in by_name:
        return by_name[name]
    pytest.fail(f"NB05 cell not found: name={name!r}")


def _first_call_arg_name(src: str, func: str) -> str | None:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id == func:
            pass
        elif isinstance(callee, ast.Attribute) and callee.attr == func:
            pass
        else:
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            return node.args[0].id
        return None
    return None


def _all_call_arg_names(src: str, func: str) -> list[str | None]:
    tree = ast.parse(src)
    hits: list[str | None] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id == func:
            pass
        elif isinstance(callee, ast.Attribute) and callee.attr == func:
            pass
        else:
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            hits.append(node.args[0].id)
        else:
            hits.append(None)
    return hits


def _count_calls(src: str, func: str) -> int:
    tree = ast.parse(src)
    n = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id == func:
            n += 1
        elif isinstance(callee, ast.Attribute) and callee.attr == func:
            n += 1
    return n


class TestNotebook05SlicePlumbing:
    """§6.2 — Four Spark-heavy stats and interactions cell use df_selection."""

    def test_target_corrs_computed_on_slice(self, cells):
        src = _cell_source(cells, "precompute_stats")
        assert _first_call_arg_name(src, "bulk_corr_with_target") == "df_selection"

    def test_null_counts_computed_on_full_panel(self, cells):
        src = _cell_source(cells, "precompute_stats")
        assert _first_call_arg_name(src, "bulk_null_counts") == "df"

    def test_row_count_reports_full_panel(self, cells):
        src = _cell_source(cells, "precompute_stats")
        assert _first_call_arg_name(src, "safe_len") == "df"

    def test_correlation_matrix_computed_on_slice(self, cells):
        src = _cell_source(cells, "compute_correlations")
        assert _first_call_arg_name(src, "batched_corr_matrix") == "df_selection"

    def test_effect_sizes_computed_on_slice(self, cells):
        src = _cell_source(cells, "plot_feature_distributions")
        assert _first_call_arg_name(src, "bulk_effect_sizes") == "df_selection"

    def test_cramers_v_computed_on_slice(self, cells):
        src = _cell_source(cells, "analyze_categorical_target")
        assert _first_call_arg_name(src, "analyze") == "df_selection"

    def test_interactions_cell_missing_fallback_uses_slice(self, cells):
        src = _cell_source(cells, "analyze_interactions")
        names = _all_call_arg_names(src, "bulk_corr_with_target")
        assert names, "analyze_interactions should have the missing-corr fallback call"
        assert all(n == "df_selection" for n in names), (
            f"bulk_corr_with_target first-arg should be df_selection, saw {names}"
        )


class TestLeakageGateSlice:
    """§6.3 — Leakage gate operates on slice df; drop applied to both frames."""

    def test_leakage_gate_receives_slice_df(self, cells):
        src = _cell_source(cells, "check_leaking_features")
        assert _first_call_arg_name(src, "detect_leaking_features") == "df_selection"

    def test_leakage_drop_applied_to_df(self, cells):
        src = _cell_source(cells, "check_leaking_features")
        tree = ast.parse(src)
        targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        if isinstance(node.value, ast.Call):
                            f = node.value.func
                            if isinstance(f, ast.Attribute) and f.attr == "drop":
                                targets.add(t.id)
        assert "df" in targets, "df.drop(columns=...) missing"

    def test_leakage_drop_applied_to_df_selection(self, cells):
        src = _cell_source(cells, "check_leaking_features")
        assert "df_selection = df_selection.drop" in src or (
            "df_selection.drop(" in src and "df_selection =" in src
        ), "df_selection must receive the leakage drop so it stays column-aligned with df"


class TestSlicePlumbingEdgeCases:
    """§6.4 — Config knobs, auto-disable, slice cell presence."""

    def test_config_knobs_present(self, cells):
        src = _cell_source(cells, "relationship_config")
        assert "SELECTION_SLICE_ENABLED" in src
        assert "SELECTOR_MIN_POSITIVE_RATE" in src
        assert "BOX_PLOT_SLICE_DATE" in src

    def test_build_selection_slice_cell_exists(self, cells):
        by_name = cells["by_name"]
        assert "build_selection_slice" in by_name, (
            "New cell build_selection_slice must be present between "
            "cache_analysis_df and precompute_stats"
        )

    def test_build_selection_slice_uses_resolve_time_slice(self, cells):
        src = _cell_source(cells, "build_selection_slice")
        assert "resolve_time_slice(" in src
        assert 'findings.time_column' in src

    def test_build_selection_slice_has_time_column_fallback(self, cells):
        src = _cell_source(cells, "build_selection_slice")
        assert "findings.time_column" in src
        assert "df_selection = df" in src  # fallback assignment for Kaggle/no-time path

    def test_build_selection_slice_tracks_object(self, cells):
        src = _cell_source(cells, "build_selection_slice")
        assert "track_stage_object(df_selection)" in src

    def test_release_stage_memory_deletes_df_selection(self, cells):
        src = _cell_source(cells, "release_stage_memory")
        assert "df_selection" in src
        tree = ast.parse(src)
        deleted = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Delete):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        deleted.add(t.id)
        assert "df_selection" in deleted, "df_selection must be in del list"


class TestBoxPlotOverride:
    """§6.2 — Box-plot default + override behavior."""

    def test_box_plot_default_sources_from_df_selection(self, cells):
        src = _cell_source(cells, "compute_feature_importance")
        # Default path assigns _box_df = df_selection (no override active)
        assert "_box_df = df_selection" in src

    def test_box_plot_override_uses_slice_df_at_date(self, cells):
        src = _cell_source(cells, "compute_feature_importance")
        assert "BOX_PLOT_SLICE_DATE" in src
        assert "slice_df_at_date" in src

    def test_box_plot_override_prints_warning(self, cells):
        src = _cell_source(cells, "compute_feature_importance")
        assert "WARNING" in src, (
            "override branch must print a WARNING: prefix per §5.4"
        )


class TestVarianceNotSliced:
    """§6.5 — Variance path keeps the full panel; no slice metadata on variance drops."""

    def test_run_selection_pipeline_receives_full_panel_df(self, cells):
        src = _cell_source(cells, "run_statistical_feature_selection")
        assert _first_call_arg_name(src, "run_selection_pipeline") == "df"

    def test_variance_branch_does_not_thread_slice_metadata(self, cells):
        src = _cell_source(cells, "run_statistical_feature_selection")
        # Cheap textual guard: ensure the variance branch does NOT contain
        # slice_date= on an add_gold_drop_weak call.  Implementation uses two
        # separate if/elif branches in 2e6e3e63; the variance branch should
        # remain kwarg-free.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            # heuristic: is this the variance branch?
            test_src = ast.unparse(node.test)
            if "variance" not in test_src.lower():
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                f = child.func
                name = f.attr if isinstance(f, ast.Attribute) else (
                    f.id if isinstance(f, ast.Name) else ""
                )
                if name == "add_gold_drop_weak":
                    for kw in child.keywords:
                        assert kw.arg not in {
                            "slice_date", "slice_strategy", "slice_row_count",
                        }, f"variance branch must not thread slice metadata, saw {kw.arg}"

    def test_multicollinear_branch_threads_slice_metadata(self, cells):
        src = _cell_source(cells, "run_statistical_feature_selection")
        tree = ast.parse(src)
        found_slice_kwarg = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.unparse(node.test)
            if "correlation" not in test_src.lower():
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                f = child.func
                name = f.attr if isinstance(f, ast.Attribute) else (
                    f.id if isinstance(f, ast.Name) else ""
                )
                if name != "add_gold_drop_multicollinear":
                    continue
                kw_names = {kw.arg for kw in child.keywords}
                # Explicit kwargs path
                if {"slice_date", "slice_strategy", "slice_row_count"} <= kw_names:
                    found_slice_kwarg = True
                    continue
                # **kwargs spread path: a keyword with arg=None whose value
                # is a Name that mentions "slice_kwargs" — the notebook
                # builds this dict conditionally on _slice_date being set.
                for kw in child.keywords:
                    if kw.arg is None and isinstance(kw.value, ast.Name) and "slice_kwargs" in kw.value.id:
                        found_slice_kwarg = True
                        break
        assert found_slice_kwarg, (
            "correlation branch must thread slice_date/slice_strategy/slice_row_count "
            "into add_gold_drop_multicollinear (explicitly or via **slice_kwargs spread)"
        )


class TestNoRecomputeInvariantStatic:
    """§6.6 — Static invariants (dynamic job-count baseline tracked separately)."""

    def test_bulk_corr_with_target_appears_at_most_twice(self, cells):
        """Once in precompute_stats, once (optional) in the interactions fallback."""
        total = sum(
            _count_calls(src, "bulk_corr_with_target")
            for src in cells["by_id"].values()
        )
        assert total <= 2, f"expected ≤2 bulk_corr_with_target calls, saw {total}"

    def test_bulk_effect_sizes_called_exactly_once(self, cells):
        total = sum(
            _count_calls(src, "bulk_effect_sizes")
            for src in cells["by_id"].values()
        )
        assert total == 1, f"expected exactly 1 bulk_effect_sizes call, saw {total}"

    def test_batched_corr_matrix_called_exactly_once(self, cells):
        total = sum(
            _count_calls(src, "batched_corr_matrix")
            for src in cells["by_id"].values()
        )
        assert total == 1, f"expected exactly 1 batched_corr_matrix call, saw {total}"


@pytest.mark.skip(reason="§6.6 dynamic job-count baseline requires runnable notebook fixture — follow-up")
class TestNoRecomputeDynamicBaseline:
    def test_full_notebook_spark_job_delta_locked_to_baseline(self):
        raise AssertionError("pending runnable fixture")


@pytest.mark.skip(reason="§6.7 parity test requires synthetic snapshot-grid fixture — follow-up")
class TestNb05SliceParity:
    def test_nb05_slice_parity_local_vs_distributed(self):
        raise AssertionError("pending runnable fixture")
