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
LIFECYCLE_DIR = _STAGES_ROOT / "lifecycle"
SCD_HISTORY_DIR = _STAGES_ROOT / "scd_history"
AUTO_EXPLORER_DIR = _SRC_ROOT / "analysis" / "auto_explorer"
RECOMMENDATIONS_DIR = _SRC_ROOT / "analysis" / "recommendations"
DISCOVERY_DIR = _SRC_ROOT / "analysis" / "discovery"
DIAGNOSTICS_DIR = _SRC_ROOT / "analysis" / "diagnostics"
CONFIG_DIR = _SRC_ROOT / "core" / "config"
FEATURE_STORE_DIR = _SRC_ROOT / "integrations" / "adapters" / "feature_store"
STORAGE_DIR = _SRC_ROOT / "integrations" / "adapters" / "storage"

SCORING_DIR = _STAGES_ROOT / "scoring"
DEPLOYMENT_DIR = _STAGES_ROOT / "deployment"
CLEANING_DIR = _STAGES_ROOT / "cleaning"
INGESTION_DIR = _STAGES_ROOT / "ingestion"
PREPROCESSING_DIR = _STAGES_ROOT / "preprocessing"
MONITORING_DIR = _STAGES_ROOT / "monitoring"
TRANSFORMATION_DIR = _STAGES_ROOT / "transformation"
VISUALIZATION_DIR = _SRC_ROOT / "analysis" / "visualization"
BUSINESS_DIR = _SRC_ROOT / "analysis" / "business"
INTERPRETABILITY_DIR = _SRC_ROOT / "analysis" / "interpretability"
TRANSFORMS_DIR = _SRC_ROOT / "transforms"
STREAMING_DIR = _SRC_ROOT / "integrations" / "streaming"
ITERATION_DIR = _SRC_ROOT / "integrations" / "iteration"
TOP_FEATURE_STORE_DIR = _SRC_ROOT / "integrations" / "feature_store"

ALLOWLISTED_FILES = {
    "window_recommendation.py", "spark_segment_analyzer.py",
    "feature_manifest.py", "snapshot_manager.py",
    # Visualization always operates on pre-collected local data for plotting
    "chart_builder.py", "column_paginator.py", "console.py", "display.py",
    # Interpretability modules require numpy arrays for sklearn/SHAP
    "shap_explainer.py", "pdp_generator.py", "individual_explainer.py",
    "cohort_analyzer.py", "counterfactual.py",
    # Business analysis operates on aggregated summary data
    "ab_test_designer.py", "risk_profile.py", "report_generator.py",
    # Deployment scoring runs on collected batches
    "batch_scorer.py", "champion_challenger.py",
    # Streaming operates within Spark structured streaming context
    "window_aggregator.py", "online_store_writer.py",
    # Compat detection layer — sparkContext fallback for local mode only
    "detection.py",
}

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
        re.compile(r"\.rdd[.\)]"),
        ".rdd access is banned on Databricks shared clusters (Unity Catalog)",
        "Use F.spark_partition_id() for partition info, or Spark SQL APIs",
    ),
    (
        re.compile(r"\.sparkContext\."),
        ".sparkContext access is banned on Databricks shared clusters (Unity Catalog)",
        "Use get_default_parallelism() from core.compat.detection, or spark.conf.get()",
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
    (
        re.compile(r"(?<!\w)pd\.to_datetime\(df\["),
        "pd.to_datetime(df[col]) may not support format/errors kwargs on pyspark.pandas",
        "Use safe_to_datetime(df[col]) from core.compat",
    ),
    (
        re.compile(r"\.query\("),
        ".query() with Python list syntax fails on pyspark.pandas (Spark SQL parse error)",
        "Use safe_query(df, expr) from core.compat",
    ),
    (
        re.compile(r"(?<!\w)pd\.api\.types\."),
        "pd.api.types does not exist in pyspark.pandas",
        "Use is_numeric_dtype() / is_datetime64_any_dtype() from core.compat",
    ),
    (
        re.compile(r"(?<!\w)pd\.qcut\("),
        "pd.qcut() does not exist in pyspark.pandas",
        "Use qcut() from core.compat",
    ),
    (
        re.compile(r"\.unstack\(fill_value="),
        ".unstack(fill_value=) is not supported in pyspark.pandas",
        "Use .unstack().fillna(0) instead",
    ),
    (
        re.compile(r"np\.where\("),
        "np.where() triggers __iter__() on pyspark.pandas Series",
        "Use series.where(cond, other) or F.when().otherwise()",
    ),
    (
        re.compile(r"\.iterrows\(\)"),
        ".iterrows() collects distributed data row-by-row — OOM on large datasets",
        "Use vectorized operations or iterate on small collected aggregates only",
    ),
    (
        re.compile(r"\.itertuples\(\)"),
        ".itertuples() collects distributed data row-by-row — OOM on large datasets",
        "Use vectorized operations or iterate on small collected aggregates only",
    ),
    (
        re.compile(r"""\.drop\([^)]*errors\s*="""),
        ".drop(errors=) is not supported in pyspark.pandas",
        "Filter columns before dropping: df.drop(columns=[c for c in cols if c in df.columns])",
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


def _collect_lifecycle_files() -> list[Path]:
    return sorted(
        p for p in LIFECYCLE_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


def _collect_scd_history_files() -> list[Path]:
    return sorted(
        p for p in SCD_HISTORY_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


def _collect_auto_explorer_files() -> list[Path]:
    return sorted(
        p for p in AUTO_EXPLORER_DIR.glob("*.py")
        if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
    )


def _collect_guarded_analysis_and_adapter_files() -> list[Path]:
    dirs = [
        RECOMMENDATIONS_DIR, DISCOVERY_DIR, DIAGNOSTICS_DIR,
        CONFIG_DIR, FEATURE_STORE_DIR, STORAGE_DIR,
    ]
    files = []
    for d in dirs:
        files.extend(
            p for p in d.glob("*.py")
            if p.name not in ALLOWLISTED_FILES and not p.name.startswith("__")
        )
    return sorted(files)


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


@pytest.mark.parametrize("source_file", _collect_lifecycle_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns_lifecycle(source_file: Path):
    _check_file(source_file)


@pytest.mark.parametrize("source_file", _collect_scd_history_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns_scd_history(source_file: Path):
    _check_file(source_file)


@pytest.mark.parametrize("source_file", _collect_auto_explorer_files(), ids=lambda p: p.name)
def test_no_dangerous_spark_pandas_patterns_auto_explorer(source_file: Path):
    _check_file(source_file)


@pytest.mark.parametrize("source_file", _collect_guarded_analysis_and_adapter_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_dangerous_spark_pandas_patterns_analysis_and_adapters(source_file: Path):
    _check_file(source_file)


# ---------------------------------------------------------------------------
# Catch-all: scan ALL source files not already covered by targeted tests above
# ---------------------------------------------------------------------------

_EXCLUDED_DIRS = {
    _SRC_ROOT / "core" / "compat",  # compat layer itself
    _SRC_ROOT / "generators",        # code generation templates
}


def _already_covered_files() -> set[Path]:
    covered: set[Path] = set()
    for collector in [
        _collect_profiling_files, _collect_features_files, _collect_temporal_files,
        _collect_modeling_files, _collect_validation_files, _collect_auto_explorer_files,
        _collect_guarded_analysis_and_adapter_files, _collect_lifecycle_files,
        _collect_scd_history_files,
    ]:
        covered.update(collector())
    return covered


def _collect_all_remaining_source_files() -> list[Path]:
    covered = _already_covered_files()
    remaining = []
    for p in sorted(_SRC_ROOT.rglob("*.py")):
        if p in covered or p.name.startswith("__") or p.name in ALLOWLISTED_FILES:
            continue
        if any(str(p).startswith(str(d)) for d in _EXCLUDED_DIRS):
            continue
        remaining.append(p)
    return remaining


@pytest.mark.parametrize(
    "source_file", _collect_all_remaining_source_files(),
    ids=lambda p: str(p.relative_to(_SRC_ROOT)),
)
def test_no_dangerous_spark_pandas_patterns_all_remaining(source_file: Path):
    _check_file(source_file)
