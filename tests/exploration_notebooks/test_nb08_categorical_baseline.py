"""PA-2 wiring lock: NB08 must pass ``categorical_columns`` and
``target_column`` to ``build_gold_steps`` so the applicator emits the
parser-baseline one_hot for every nominal/ordinal/cyclical column.
Without this NB08 can label-encode (binary) a categorical that
production codegen one-hots — producing different feature column
names in NB08-vs-production gold (engagement spschurn-fa23ccd0
``REVENUE_MARKET_SEGMENT_1`` parity gap, residual half).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPLORATION_NB08 = REPO_ROOT / "exploration_notebooks/08_baseline_experiments.ipynb"


def _apply_gold_transforms_cell_source() -> str:
    nb = json.loads(EXPLORATION_NB08.read_text())
    for cell in nb["cells"]:
        if cell.get("id") == "f8a2c4e1":
            return "".join(cell.get("source", []))
    raise AssertionError("NB08 cell apply_gold_transforms (id=f8a2c4e1) not found")


class TestNb08CategoricalBaselineWiring:
    @pytest.fixture(scope="class")
    def src(self) -> str:
        return _apply_gold_transforms_cell_source()

    def test_passes_categorical_columns_to_build_gold_steps(self, src: str):
        assert "categorical_columns=" in src, (
            "NB08 cell apply_gold_transforms must pass categorical_columns "
            "to build_gold_steps so the applicator mirrors the parser's "
            "baseline one_hot rule for nominal/ordinal/cyclical columns. "
            "Without this NB08 may label-encode a categorical column "
            "while production codegen one-hots it — different feature "
            "names land in NB08 vs production gold."
        )

    def test_passes_target_column_to_build_gold_steps(self, src: str):
        assert "target_column=target" in src, (
            "NB08 cell apply_gold_transforms must pass target_column=target "
            "so the categorical baseline one_hot does not encode the "
            "target column as a feature."
        )

    def test_categorical_set_built_from_findings(self, src: str):
        assert "findings.columns" in src and "inferred_type" in src, (
            "NB08 must derive categorical_columns from findings.columns "
            "(filtering by inferred_type) — that's the same source the "
            "parser uses in _build_gold_config."
        )

    def test_uses_three_categorical_types(self, src: str):
        for t in (
            "CATEGORICAL_NOMINAL",
            "CATEGORICAL_ORDINAL",
            "CATEGORICAL_CYCLICAL",
        ):
            assert t in src, (
                f"NB08 categorical filter must include {t} to match the "
                "parser's _CATEGORICAL_TYPES set in findings_parser.py."
            )
