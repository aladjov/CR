"""Static analysis guard for pyspark.pandas-incompatible patterns.

Scans all profiling source files and flags API calls that fail at runtime
on Databricks where data stays distributed as pyspark.pandas DataFrames.
This catches regressions in CI before they reach the cluster.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROFILING_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "customer_retention"
    / "stages"
    / "profiling"
)

ALLOWLISTED_FILES = {"window_recommendation.py"}

_STRING_LITERAL = re.compile(r'''("""[\s\S]*?"""|'''  r"""'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")

DANGEROUS_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"\.dt\.to_period\("),
        ".dt.to_period() is not supported in pyspark.pandas",
        "Use period_start_time() from core.compat",
    ),
    (
        re.compile(r"\.values\b(?!\s*\()"),
        ".values may fail on pyspark.pandas DataFrames",
        "Use .to_numpy() instead",
    ),
    (
        re.compile(r"\.iloc\[-"),
        "Negative .iloc indexing fails in pyspark.pandas",
        "Use .min()/.max() or numpy indexing on collected data",
    ),
    (
        re.compile(r"^import pandas as pd", re.MULTILINE),
        "Bare 'import pandas as pd' bypasses pyspark.pandas compat layer",
        "Import from customer_retention.core.compat instead",
    ),
    (
        re.compile(r"\.agg\(\["),
        ".agg([list]) is not supported in pyspark.pandas",
        "Use groupby_multi_agg() from core.compat",
    ),
]


def _strip_strings(source: str) -> str:
    return _STRING_LITERAL.sub('""', source)


def _collect_profiling_files() -> list[Path]:
    return sorted(
        p for p in PROFILING_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


@pytest.mark.parametrize("source_file", _collect_profiling_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns(source_file: Path):
    raw = source_file.read_text()
    cleaned = _strip_strings(raw)

    violations: list[str] = []
    for pattern, problem, fix in DANGEROUS_PATTERNS:
        matches = list(pattern.finditer(cleaned))
        if matches:
            lines = cleaned[: matches[0].start()].count("\n") + 1
            violations.append(f"  Line ~{lines}: {problem}. Fix: {fix}")

    assert not violations, (
        f"\n{source_file.name} has pyspark.pandas-incompatible patterns:\n"
        + "\n".join(violations)
    )
