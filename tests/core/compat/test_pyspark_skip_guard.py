"""Static guard: test classes that import pyspark must have skip-without-pyspark fixtures.

Without the skip fixture, tests pass locally (pyspark installed) but fail in CI
(pyspark not in dev/ml-shap extras). This catches the omission before it reaches CI.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[2]

_PYSPARK_IMPORT_RE = re.compile(
    r"""(?:from\s+(?:pyspark|customer_retention\.integrations\.adapters\.storage\.databricks"""
    r"""|customer_retention\.core\.compat\.bulk_profiling)\s+import"""
    r"""|import\s+pyspark)""",
)

_SKIP_GUARD_RE = re.compile(r"""importorskip\(\s*['"]pyspark['"]\s*\)""")


def _collect_test_files() -> list[Path]:
    return sorted(
        p for p in _TESTS_ROOT.rglob("test_*.py")
        if not p.name.startswith("__") and p.name != "test_pyspark_skip_guard.py"
    )


def _classes_needing_guard(source: str) -> list[str]:
    """Return class names that import pyspark but lack the skip fixture."""
    violations = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_start = node.lineno
        class_end = node.end_lineno or class_start
        class_source = "\n".join(source.splitlines()[class_start - 1:class_end])

        if _PYSPARK_IMPORT_RE.search(class_source) and not _SKIP_GUARD_RE.search(class_source):
            violations.append(node.name)
    return violations


@pytest.mark.parametrize("test_file", _collect_test_files(), ids=lambda p: str(p.relative_to(_TESTS_ROOT)))
def test_pyspark_test_classes_have_skip_guard(test_file: Path):
    source = test_file.read_text()
    if not _PYSPARK_IMPORT_RE.search(source):
        return
    violations = _classes_needing_guard(source)
    assert not violations, (
        f"\n{test_file.relative_to(_TESTS_ROOT)} has test classes that import pyspark "
        f"without pytest.importorskip('pyspark'):\n"
        + "\n".join(f"  - {name}" for name in violations)
        + "\n\nAdd this fixture to each class:\n"
        "    @pytest.fixture(autouse=True)\n"
        "    def _skip_without_pyspark(self):\n"
        "        pytest.importorskip('pyspark')\n"
    )
