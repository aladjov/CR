"""Static analysis guard for pyspark.pandas-incompatible patterns.

Scans all profiling source files and flags API calls that fail at runtime
on Databricks where data stays distributed as pyspark.pandas DataFrames.
This catches regressions in CI before they reach the cluster.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "customer_retention"
)
_STAGES_ROOT = _SRC_ROOT / "stages"

PROFILING_DIR = _STAGES_ROOT / "profiling"
FEATURES_DIR = _STAGES_ROOT / "features"
TEMPORAL_DIR = _STAGES_ROOT / "temporal"
MODELING_DIR = _STAGES_ROOT / "modeling"
VALIDATION_DIR = _STAGES_ROOT / "validation"
AUTO_EXPLORER_DIR = _SRC_ROOT / "analysis" / "auto_explorer"

ALLOWLISTED_FILES = {"window_recommendation.py", "spark_segment_analyzer.py", "feature_manifest.py", "snapshot_manager.py", "analysis_context.py"}

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
    (
        re.compile(r"\.sample\(n="),
        ".sample(n=) is not supported in pyspark.pandas",
        "Use safe_sample() from core.compat",
    ),
    (
        re.compile(r"np\.select\("),
        "np.select() calls __iter__() on pyspark.pandas Series",
        "Use safe_select() from core.compat",
    ),
    (
        re.compile(r"\.drop_duplicates\([^)]*keep\s*="),
        ".drop_duplicates(keep=) is not supported in pyspark.pandas",
        "Use safe_drop_duplicates() from core.compat",
    ),
    (
        re.compile(r"""hasattr\([^,]+,\s*['"]rdd['"]\)"""),
        "hasattr(df, 'rdd') triggers RDD access on shared Databricks clusters",
        "Use _is_native_spark_df() from core.compat",
    ),
    (
        re.compile(r"\bpd\.Timestamp\b"),
        "pd.Timestamp fails when pd is pyspark.pandas (scalar not reimplemented)",
        "Use datetime.datetime (pandas Timestamp is a subclass) or import pandas as _pandas",
    ),
    (
        re.compile(r"native_pd\.to_datetime\(df\b"),
        "native_pd.to_datetime(df[col]) triggers __iter__() on pyspark.pandas Series",
        "Use safe_to_datetime(df[col]) from core.compat",
    ),
]


def _strip_strings(source: str) -> str:
    return _STRING_LITERAL.sub('""', source)


def _collect_profiling_files() -> list[Path]:
    return sorted(
        p for p in PROFILING_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


def _collect_features_files() -> list[Path]:
    return sorted(
        p for p in FEATURES_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


def _collect_temporal_files() -> list[Path]:
    return sorted(
        p for p in TEMPORAL_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


def _collect_modeling_files() -> list[Path]:
    return sorted(
        p for p in MODELING_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


def _collect_validation_files() -> list[Path]:
    return sorted(
        p for p in VALIDATION_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


def _collect_auto_explorer_files() -> list[Path]:
    return sorted(
        p for p in AUTO_EXPLORER_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )




def _check_file(source_file: Path) -> None:
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


@pytest.mark.parametrize("source_file", _collect_profiling_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns(source_file: Path):
    _check_file(source_file)


@pytest.mark.parametrize("source_file", _collect_features_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns_features(source_file: Path):
    _check_file(source_file)


@pytest.mark.parametrize("source_file", _collect_temporal_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns_temporal(source_file: Path):
    _check_file(source_file)


@pytest.mark.parametrize("source_file", _collect_modeling_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns_modeling(source_file: Path):
    _check_file(source_file)


@pytest.mark.parametrize("source_file", _collect_validation_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns_validation(source_file: Path):
    _check_file(source_file)


@pytest.mark.parametrize("source_file", _collect_auto_explorer_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns_auto_explorer(source_file: Path):
    _check_file(source_file)
