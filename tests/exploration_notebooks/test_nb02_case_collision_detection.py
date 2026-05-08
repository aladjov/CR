"""PA-3 wiring lock: NB02 must surface case-fold column collisions
detected by ``stages.profiling.case_collision`` so the operator can
register the drop_columns recs explicitly. Auto-register is intentionally
not used because the upstream column order determines which variant
``register_case_collision_drops`` would keep — and the operator's intent
(e.g. "keep ``OPPORTUNITY_ID``, drop ``opportunity_id``") may not match
first-seen-wins. The detector reports the collision, the operator
decides. Replaces the engagement-side ``LANDING_DROP_COLUMNS_OVERRIDES``
discovery (which forced the operator to manually find collisions).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPLORATION_NB02 = REPO_ROOT / "exploration_notebooks/02_source_integrity.ipynb"


def _all_code_sources(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return [
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    ]


def _concatenated_code(nb_path: Path) -> str:
    return "\n".join(_all_code_sources(nb_path))


class TestNb02CaseCollisionDetection:
    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _concatenated_code(EXPLORATION_NB02)

    def test_imports_detector(self, src: str):
        assert "detect_case_collisions" in src, (
            "exploration_notebooks/02_source_integrity.ipynb must import "
            "detect_case_collisions from "
            "customer_retention.stages.profiling.case_collision so "
            "case-fold column collisions surface automatically at NB02 time "
            "instead of requiring the operator to manually find them."
        )

    def test_detector_actually_called(self, src: str):
        assert "detect_case_collisions(" in src, (
            "exploration NB02 imports detect_case_collisions but does not "
            "call it — the import alone does not detect anything."
        )

    def test_detector_called_against_dataset_columns(self, src: str):
        """The detector must run on the DataFrame's column list (not on
        ``findings.columns`` which is the schema metadata, since both
        case-variants get folded into one entry there). Spark's
        ``df.columns`` preserves both variants — that's the pre-saveAsTable
        list.
        """
        called_block = src[src.find("detect_case_collisions(") :]
        called_block = called_block[: 400]
        assert "df.columns" in called_block, (
            "detect_case_collisions must be invoked with df.columns — the "
            "column list of the loaded landing-time DataFrame — so both "
            "case-variants are visible to the detector before Spark folds "
            "them at saveAsTable time."
        )

    def test_no_implicit_auto_register(self, src: str):
        """Safety guard: NB02 must not call ``register_case_collision_drops``
        without an opt-in flag. The auto-register helper keeps the first-
        seen variant, which is wrong when the upstream column order puts
        the operator-undesired variant first (e.g. SPS engagement wants to
        keep ``OPPORTUNITY_ID`` but Snowflake may emit ``opportunity_id``
        first). Surfacing the warning is enough — let the operator
        register the drop explicitly.
        """
        assert "register_case_collision_drops(" not in src, (
            "exploration NB02 must not auto-register case-collision drops "
            "(register_case_collision_drops is order-dependent and may pick "
            "the wrong variant). Print the detection result instead and "
            "instruct the operator to add an explicit add_landing_drop_columns "
            "rec for the variant they want to drop."
        )
