"""Pre-transform drops: NB08 inserts `apply_early_drops` (id `e1d70a01`)
immediately before `apply_gold_transforms` (id `f8a2c4e1`) so the gold
executor never fits transforms on columns NB05 flagged as weak or
multicollinear or columns matching a per-dataset leakage prefix.

These tests pin the structural contract that the optimisation depends on:

- The new cell exists, sits between `9e09a801` (doc) and `f8a2c4e1`.
- It drops from `df` (not from `X_train`/`X_test`, which don't exist yet)
  and updates `feature_cols`.
- It reads NB05 verdicts from the `RecommendationRegistry` and leakage
  prefixes from `MultiDatasetFindings`, gated by `APPLY_NB05_DROPS`.
- The downstream safety-net cells (`a7c3e1f0` leakage prefix exclusion on
  `df`, `b5d70002` NB05 drops on `X_train`/`X_test`) remain in place so
  anything that slips past the early pass is still caught.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPLORATION_NB08 = REPO_ROOT / "exploration_notebooks/08_baseline_experiments.ipynb"


def _load_nb() -> dict:
    return json.loads(EXPLORATION_NB08.read_text())


def _cell_source(nb: dict, cell_id: str) -> str:
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell.get("source", []))
    raise AssertionError(f"cell id={cell_id!r} not found")


def _cell_index(nb: dict, cell_id: str) -> int:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("id") == cell_id:
            return i
    raise AssertionError(f"cell id={cell_id!r} not found")


class TestApplyEarlyDropsCell:
    @pytest.fixture(scope="class")
    def nb(self) -> dict:
        return _load_nb()

    @pytest.fixture(scope="class")
    def src(self, nb: dict) -> str:
        return _cell_source(nb, "e1d70a01")

    def test_cell_exists(self, nb: dict):
        assert any(c.get("id") == "e1d70a01" for c in nb["cells"]), (
            "apply_early_drops cell (id=e1d70a01) must exist. This cell is the "
            "pre-transform filter that removes NB05-weak/multicollinear and "
            "leakage-prefix columns before apply_gold_transforms runs."
        )

    def test_cell_tag_matches_id(self, src: str):
        assert src.startswith("# @cr:code name='apply_early_drops' id=e1d70a01\n"), (
            "Cell e1d70a01 must carry the canonical # @cr:code tag on line 1 with "
            "matching id; the notebook-sync protocol relies on the tag id matching "
            "the JSON cell id."
        )

    def test_cell_runs_before_apply_gold_transforms(self, nb: dict):
        idx_early = _cell_index(nb, "e1d70a01")
        idx_gold = _cell_index(nb, "f8a2c4e1")
        assert idx_early < idx_gold, (
            f"apply_early_drops (idx={idx_early}) must run before apply_gold_transforms "
            f"(idx={idx_gold}) so build_gold_steps/build_silver_derived_steps filter "
            "the doomed columns out of the step list via their pipeline_columns checks."
        )

    def test_cell_runs_immediately_before_apply_gold_transforms(self, nb: dict):
        idx_early = _cell_index(nb, "e1d70a01")
        idx_gold = _cell_index(nb, "f8a2c4e1")
        assert idx_early + 1 == idx_gold, (
            "apply_early_drops should sit immediately before apply_gold_transforms — "
            "no other code cell should mutate df between them or the drop semantics "
            "become ambiguous."
        )

    def test_guarded_by_skip_modeling_and_namespace(self, src: str):
        assert "if not _skip_modeling and _namespace:" in src, (
            "Cell must early-out when modeling is skipped or when namespace is "
            "absent (no recommendations file to consult)."
        )

    def test_reads_nb05_drops_from_recommendations_registry(self, src: str):
        assert "RecommendationRegistry" in src and "feature_selection" in src, (
            "Cell must load the RecommendationRegistry and inspect "
            "gold.feature_selection to find drop_weak/drop_multicollinear "
            "verdicts."
        )
        assert "'drop_multicollinear'" in src and "'drop_weak'" in src, (
            "Cell must include both drop_multicollinear and drop_weak actions; "
            "those are the only NB05 verdicts that translate to a pre-transform drop."
        )

    def test_reads_leakage_prefixes_from_multi_dataset_findings(self, src: str):
        assert "MultiDatasetFindings" in src and "excluded_leaking_features" in src, (
            "Cell must load MultiDatasetFindings and walk every dataset's "
            "excluded_leaking_features to build the prefix list."
        )
        assert "FindingsParser.find_leakage_excluded_columns" in src, (
            "Cell must use the canonical FindingsParser.find_leakage_excluded_columns "
            "matcher (handles lag/velocity variants identically to the post-transform "
            "safety net)."
        )

    def test_drops_from_df_not_x_train(self, src: str):
        assert "df = df.drop(columns=" in src, (
            "Cell must drop from df (the pre-train/test-split DataFrame), not from "
            "X_train/X_test which do not exist at this point in the notebook."
        )
        assert "X_train" not in src and "X_test" not in src, (
            "Cell must not reference X_train/X_test — those variables are only "
            "introduced after split_train_test."
        )

    def test_updates_feature_cols(self, src: str):
        assert "feature_cols = [" in src and "_early_to_drop" in src, (
            "Cell must rebuild feature_cols without the dropped columns so the "
            "downstream pipeline does not retain stale feature names."
        )

    def test_uses_apply_nb05_drops_flag_defensively(self, src: str):
        assert "globals().get('APPLY_NB05_DROPS', True)" in src, (
            "Cell must use globals().get for APPLY_NB05_DROPS — the config cell "
            "that defines this flag (b5d70001) sits AFTER this cell in the "
            "notebook, so a direct reference would NameError on first execution."
        )

    def test_emits_summary_print(self, src: str):
        assert "Pre-transform drops:" in src, (
            "Cell must print a one-line summary so operators see how many "
            "columns the pre-filter removed and from which source."
        )


class TestPostTransformSafetyNetsRetained:
    """Once the early drop runs, the existing post-transform/post-split cells
    must remain in place as idempotent safety nets — they typically no-op
    after the early pass, but catch anything that re-enters (e.g. a future
    user-code cell adding columns or a registry that gets refreshed)."""

    @pytest.fixture(scope="class")
    def nb(self) -> dict:
        return _load_nb()

    def test_apply_leakage_prefix_exclusion_still_present(self, nb: dict):
        idx = _cell_index(nb, "a7c3e1f0")
        idx_gold = _cell_index(nb, "f8a2c4e1")
        assert idx > idx_gold, (
            "apply_leakage_prefix_exclusion (a7c3e1f0) must remain AFTER "
            "apply_gold_transforms so it can sweep any post-transform columns "
            "that happen to match a leakage prefix."
        )

    def test_apply_nb05_drops_still_present(self, nb: dict):
        idx = _cell_index(nb, "b5d70002")
        idx_gold = _cell_index(nb, "f8a2c4e1")
        assert idx > idx_gold, (
            "apply_nb05_drops (b5d70002) must remain AFTER apply_gold_transforms "
            "(and split_train_test) so it sweeps X_train/X_test/scaled variants "
            "as a final safety net."
        )

    def test_apply_nb05_drops_still_targets_x_train(self, nb: dict):
        src = _cell_source(nb, "b5d70002")
        assert "X_train" in src and "X_test" in src, (
            "Safety-net apply_nb05_drops cell must still operate on X_train/X_test "
            "so it can sweep any column that slipped past the early pass (e.g. via "
            "user-code cells added after apply_early_drops)."
        )
