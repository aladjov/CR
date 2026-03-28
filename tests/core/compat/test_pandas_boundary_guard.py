"""Boundary guard: pandas collection operations only at designated system boundaries.

Ensures that operations which pull distributed data to the driver (.toPandas(),
.to_pandas(), .collect()) only appear in files that sit at explicit boundaries
between distributed and local computation (storage adapters, sklearn interface,
compat dispatch layer). Any new file using these operations must be consciously
added to the allowlist with a justification.

Also enforces that all data-handling modules import pd from the compat layer,
not directly from pandas — preventing silent local-only code from reaching
Databricks clusters.

See docs/Coding_Practices.md § "pyspark.pandas Incompatible Patterns" and
docs/wiki/Architecture.md § "Timestamp Sanitization at System Boundaries".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "customer_retention"
)

_STRING_LITERAL = re.compile(
    r'("""[\s\S]*?"""|'
    r"'''[\s\S]*?'''|"
    r'"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*')"
)

# ── Directories excluded from boundary enforcement ──────────────────────
# Compat layer IS the boundary dispatcher — it must call .toPandas() etc.
# Generators produce code template strings, not runtime collection calls.
_EXCLUDED_DIRS = {
    _SRC_ROOT / "core" / "compat",
    _SRC_ROOT / "generators",
}

# ── .toPandas() / .to_pandas() allowlist ────────────────────────────────
# Each entry documents WHY the file sits at a boundary.
TOPANDAS_BOUNDARY_FILES = {
    # Storage adapters — read boundary (Delta → pandas/pyspark.pandas)
    "integrations/adapters/storage/databricks.py",
    "integrations/adapters/storage/local.py",
    # Feature store adapters — read boundary
    "integrations/adapters/feature_store/databricks.py",
    "integrations/feature_store/manager.py",
    # Data loading — system ingress boundary
    "stages/ingestion/loaders.py",
    "stages/scoring/data_loader.py",
    # sklearn interface — needs pandas/numpy arrays
    "stages/modeling/spark_classifier_wrapper.py",
    "stages/modeling/data_splitter.py",  # non-temporal strategies fall back to sklearn (pandas)
    "stages/features/feature_selector.py",  # L1 selection collects to pandas for sklearn LogisticRegression
    # Batch fitting — bounded sample (≤50K rows) for sklearn PowerTransformer
    "transforms/executor.py",
    # Bounded sampling for type detection / fingerprinting
    "stages/temporal/timestamp_discovery.py",
    "stages/profiling/temporal_coverage.py",
    "analysis/auto_explorer/explorer.py",
    "analysis/auto_explorer/analysis_context.py",
    "analysis/auto_explorer/dataset_fingerprinter.py",
}

# ── .collect() allowlist ────────────────────────────────────────────────
# .collect() on aggregated (small) Spark results only.
COLLECT_BOUNDARY_FILES = {
    # Bounded sampling / fingerprinting
    "analysis/auto_explorer/active_dataset_store.py",
    "analysis/auto_explorer/sampling.py",
    # sklearn interface — prediction collection
    "stages/modeling/spark_classifier_wrapper.py",
    # Aggregated bulk stats (always after .agg() — small result)
    "stages/profiling/time_series_profiler.py",
    # Timeseries validation — aggregated stats
    "stages/validation/timeseries_detector.py",
    # Timestamp discovery — bounded .head() + collect
    "stages/temporal/timestamp_discovery.py",
    # Native Spark transform ops — collect only for small aggregates (quantiles, categories)
    "transforms/spark_ops.py",
    # Batch fitting — .agg().collect() for scaler stats (1 row), bounded sample for PowerTransform
    "transforms/executor.py",
    # Training prep — groupBy().agg().collect() for unpivoted null counts + stddev
    "stages/modeling/training_preparator.py",
    # Feature scaler — groupBy().agg().collect() for unpivoted scaler params (1 row per feature)
    "stages/modeling/spark_feature_scaler.py",
    # Field availability audit — .agg().collect() for batched population-rate stats (1 row)
    "analysis/auto_explorer/field_availability_audit.py",
    # CV folds — gc.collect() for memory reclamation between folds (not Spark .collect())
    "stages/modeling/cross_validator.py",
    # L1 thinning — .distinct().collect() for snapshot dates (small: ~100 dates max)
    "stages/features/feature_selector.py",
}

# ── bare 'import pandas as pd' allowlist ────────────────────────────────
# Files that legitimately import pandas directly (not via compat).
BARE_PANDAS_IMPORT_ALLOWLIST = {
    # Compat internals — implement the dispatch
    "core/compat/__init__.py",
    "core/compat/ops.py",
    "core/compat/pandas_backend.py",
    "core/compat/detection.py",
    # Window recommendation — tiny summary DataFrames
    "stages/profiling/window_recommendation.py",
    # Visualization — always operates on collected local data
    "analysis/visualization/chart_builder.py",
    "analysis/visualization/column_paginator.py",
    "analysis/visualization/console.py",
    "analysis/visualization/display.py",
    "analysis/visualization/number_formatter.py",
    "analysis/visualization/attention_scorer.py",
    # Business / interpretability — operates on aggregated data
    "analysis/business/ab_test_designer.py",
    "analysis/business/fairness_analyzer.py",
    "analysis/business/intervention_matcher.py",
    "analysis/business/report_generator.py",
    "analysis/business/risk_profile.py",
    "analysis/business/roi_analyzer.py",
    "analysis/interpretability/shap_explainer.py",
    "analysis/interpretability/pdp_generator.py",
    "analysis/interpretability/individual_explainer.py",
    "analysis/interpretability/cohort_analyzer.py",
    "analysis/interpretability/counterfactual.py",
    # Deployment — collected batch output
    "stages/deployment/batch_scorer.py",
    "stages/deployment/champion_challenger.py",
    "stages/deployment/model_registry.py",
    "stages/deployment/retraining_trigger.py",
    # Monitoring — aggregated metrics
    "stages/monitoring/performance_monitor.py",
    "stages/monitoring/drift_detector.py",
    "stages/monitoring/alert_manager.py",
    # Streaming — Spark structured streaming context
    "integrations/streaming/window_aggregator.py",
    "integrations/streaming/online_store_writer.py",
    "integrations/streaming/batch_integration.py",
    "integrations/streaming/realtime_scorer.py",
    "integrations/streaming/early_warning_model.py",
    "integrations/streaming/trigger_engine.py",
    # Iteration / LLM — metadata only
    "integrations/iteration/context.py",
    "integrations/iteration/feedback_collector.py",
    "integrations/iteration/orchestrator.py",
    "integrations/iteration/recommendation_tracker.py",
    "integrations/llm_context/context_builder.py",
}


def _strip_strings(source: str) -> str:
    return _STRING_LITERAL.sub('""', source)


def _all_source_files() -> list[Path]:
    return sorted(
        p for p in _SRC_ROOT.rglob("*.py")
        if not p.name.startswith("__")
        and "__pycache__" not in str(p)
    )


def _non_excluded_source_files() -> list[Path]:
    return [
        p for p in _all_source_files()
        if not any(str(p).startswith(str(d)) for d in _EXCLUDED_DIRS)
    ]


def _rel(p: Path) -> str:
    return str(p.relative_to(_SRC_ROOT))


# ── Test: .toPandas() / .to_pandas() only at boundaries ────────────────

_TO_PANDAS_PATTERN = re.compile(r"\.toPandas\(\)|\.to_pandas\(\)")


def _files_with_to_pandas() -> list[Path]:
    result = []
    for p in _non_excluded_source_files():
        cleaned = _strip_strings(p.read_text())
        if _TO_PANDAS_PATTERN.search(cleaned):
            result.append(p)
    return result


@pytest.mark.parametrize("source_file", _files_with_to_pandas(), ids=_rel)
def test_to_pandas_only_at_boundaries(source_file: Path):
    rel = _rel(source_file)
    assert rel in TOPANDAS_BOUNDARY_FILES, (
        f"\n{rel} calls .toPandas() or .to_pandas() but is NOT in the boundary allowlist.\n"
        f"If this file is at a legitimate system boundary (storage adapter, sklearn\n"
        f"interface, bounded sampling), add it to TOPANDAS_BOUNDARY_FILES with a comment\n"
        f"explaining why. Otherwise, use the compat layer to keep data distributed."
    )


# ── Test: .collect() only at boundaries ─────────────────────────────────

_COLLECT_PATTERN = re.compile(r"\.collect\(\)")


def _files_with_collect() -> list[Path]:
    result = []
    for p in _non_excluded_source_files():
        cleaned = _strip_strings(p.read_text())
        if _COLLECT_PATTERN.search(cleaned):
            result.append(p)
    return result


@pytest.mark.parametrize("source_file", _files_with_collect(), ids=_rel)
def test_collect_only_at_boundaries(source_file: Path):
    rel = _rel(source_file)
    assert rel in COLLECT_BOUNDARY_FILES, (
        f"\n{rel} calls .collect() but is NOT in the boundary allowlist.\n"
        f"If this file collects aggregated (small) Spark results at a system boundary,\n"
        f"add it to COLLECT_BOUNDARY_FILES with a comment. Otherwise, keep data\n"
        f"distributed and use compat helpers."
    )


# ── Test: bare 'import pandas as pd' only in allowlisted files ─────────

_BARE_PANDAS_IMPORT = re.compile(r"^import pandas as pd", re.MULTILINE)


def _files_with_bare_pandas_import() -> list[Path]:
    result = []
    for p in _all_source_files():
        cleaned = _strip_strings(p.read_text())
        if _BARE_PANDAS_IMPORT.search(cleaned):
            result.append(p)
    return result


@pytest.mark.parametrize("source_file", _files_with_bare_pandas_import(), ids=_rel)
def test_bare_pandas_import_only_in_allowlisted(source_file: Path):
    rel = _rel(source_file)
    assert rel in BARE_PANDAS_IMPORT_ALLOWLIST, (
        f"\n{rel} uses 'import pandas as pd' directly.\n"
        f"Data-handling modules must import pd from customer_retention.core.compat\n"
        f"so the code works on both pandas and pyspark.pandas backends.\n"
        f"If this file only ever operates on small local data, add it to\n"
        f"BARE_PANDAS_IMPORT_ALLOWLIST with a justification."
    )


# ── Test: no unguarded .tolist() on potentially unbounded Series ────────
# .tolist() on column names / index names / scaler params is fine.
# .tolist() on data series without .head() / slicing is dangerous.

_UNGUARDED_TOLIST = re.compile(
    r"(?:\.unique\(\)|\.dropna\(\))\s*\.tolist\(\)"
)

TOLIST_ALLOWLISTED_FILES = {
    # Compat layer — dispatch functions
    "core/compat/__init__.py",
    "core/compat/bulk_profiling.py",
    # Pattern analysis on small pre-aggregated series
    "stages/profiling/pattern_analysis_config.py",
    # Transforms — fitted encoder classes list (small)
    "transforms/fitted.py",
    # CV folds — groups_pd already collected via collect_for_sklearn
    "stages/modeling/cross_validator.py",
}


def _files_with_unguarded_tolist() -> list[Path]:
    result = []
    for p in _non_excluded_source_files():
        cleaned = _strip_strings(p.read_text())
        if _UNGUARDED_TOLIST.search(cleaned):
            result.append(p)
    return result


@pytest.mark.parametrize(
    "source_file", _files_with_unguarded_tolist(), ids=_rel
)
def test_no_unguarded_tolist_on_series(source_file: Path):
    rel = _rel(source_file)
    assert rel in TOLIST_ALLOWLISTED_FILES, (
        f"\n{rel} calls .unique().tolist() or .dropna().tolist() without bounds.\n"
        f"On pyspark.pandas this collects ALL values to the driver — OOM risk.\n"
        f"Use head_as_list(series.unique(), N) from core.compat to set an explicit bound.\n"
        f"If the data is guaranteed small, add to TOLIST_ALLOWLISTED_FILES with reason."
    )


# ── Test: no native_pd.DataFrame(distributed_series) wrapping ──────────
# Wrapping a pyspark.pandas object in native pandas triggers __iter__().

_NATIVE_PD_WRAPPING = re.compile(
    r"native_pd\.\w+\((?:df|series|data)\["
)


def _files_with_native_pd_wrapping() -> list[Path]:
    result = []
    for p in _non_excluded_source_files():
        cleaned = _strip_strings(p.read_text())
        if _NATIVE_PD_WRAPPING.search(cleaned):
            result.append(p)
    return result


@pytest.mark.parametrize(
    "source_file", _files_with_native_pd_wrapping(), ids=_rel
)
def test_no_native_pd_wrapping_distributed_data(source_file: Path):
    rel = _rel(source_file)
    pytest.fail(
        f"\n{rel} wraps df[col]/series[col] in native_pd.* constructor.\n"
        f"This triggers __iter__() on pyspark.pandas — use .to_pandas() first\n"
        f"or use compat helpers that handle both backends."
    )


# ── Test: compat layer exports are dual-backend dispatchers ─────────────
# Per Coding_Practices.md: no public export of single-environment functions.

def test_compat_no_pandas_only_public_exports():
    init_path = _SRC_ROOT / "core" / "compat" / "__init__.py"
    source = init_path.read_text()
    all_match = re.search(r"__all__\s*=\s*\[([^\]]+)\]", source, re.DOTALL)
    if not all_match:
        pytest.skip("No __all__ found in compat __init__.py")

    exported_names = re.findall(r'"(\w+)"|\'(\w+)\'', all_match.group(1))
    exported = {a or b for a, b in exported_names}

    private_leaked = [n for n in exported if n.startswith("_") and not n.startswith("__")]
    # _is_spark_pandas, _is_native_spark_df, _extract_dtype are intentionally
    # semi-private exports for use in stages that need dispatch detection
    known_semi_private = {
        "_is_spark_pandas", "_is_native_spark_df", "_extract_dtype",
        "_normalize_timestamp_columns",
    }
    unexpected_private = [n for n in private_leaked if n not in known_semi_private]

    assert not unexpected_private, (
        f"Compat __init__.py exports these private names: {unexpected_private}\n"
        f"Per Coding_Practices.md, only dual-backend dispatchers should be public.\n"
        f"Keep single-environment implementations private (prefix _)."
    )


# ── Test: test files importing pyspark must have importorskip guard ───
# CI runs without PySpark. Tests that import it (directly or via module-level
# imports in source) must call pytest.importorskip("pyspark") so they skip
# gracefully instead of raising ModuleNotFoundError.

_TESTS_ROOT = Path(__file__).resolve().parents[2]

_PYSPARK_IMPORT_IN_TEST = re.compile(
    r"(?:from\s+pyspark|import\s+pyspark)"
)

_IMPORTORSKIP_GUARD = re.compile(
    r'pytest\.importorskip\(\s*["\']pyspark["\']'
)

_TRY_EXCEPT_IMPORT_GUARD = re.compile(
    r"try:\s*\n\s*(?:from\s+pyspark|import\s+pyspark)[^\n]*\n(?:[^\n]*\n)*?\s*except\s+ImportError"
)

_PYTESTMARK_SPARK = re.compile(
    r"pytestmark\s*=.*pytest\.mark\.spark"
)

# Test files that are entirely guarded at module level (importorskip at top
# or pytestmark = pytest.mark.spark) — no per-method check needed.
_MODULE_LEVEL_GUARDED_TEST_FILES = {
    "transforms/test_spark_ops.py",
    "stages/modeling/test_spark_baseline_trainer.py",
    "stages/modeling/test_spark_classifier_wrapper.py",
    "stages/profiling/test_spark_segment_analyzer.py",
    "stages/profiling/test_spark_temporal_feature_analyzer.py",
    "stages/profiling/test_spark_temporal_feature_engineer.py",
    "stages/profiling/test_spark_time_window_aggregator.py",
    "stages/temporal/test_spark_temporal_merger.py",
    "generators/pipeline_generator/test_databricks_renderer.py",
    "generators/pipeline_generator/test_databricks_renderer_optimize.py",
    "generators/pipeline_generator/test_databricks_generator.py",
    "generators/pipeline_generator/test_pipeline_parity.py",
    "generators/orchestration/test_databricks_exporter.py",
    "integrations/test_databricks_init.py",
    "core/compat/test_spark_backend.py",
}


def _test_rel(p: Path) -> str:
    return str(p.relative_to(_TESTS_ROOT))


def _test_files_with_pyspark_import() -> list[Path]:
    result = []
    for p in sorted(_TESTS_ROOT.rglob("test_*.py")):
        if "__pycache__" in str(p):
            continue
        rel = _test_rel(p)
        if rel in _MODULE_LEVEL_GUARDED_TEST_FILES:
            continue
        source = p.read_text()
        cleaned = _strip_strings(source)
        if _PYSPARK_IMPORT_IN_TEST.search(cleaned):
            if not _IMPORTORSKIP_GUARD.search(source) and not _PYTESTMARK_SPARK.search(source) and not _TRY_EXCEPT_IMPORT_GUARD.search(source):
                result.append(p)
    return result


@pytest.mark.parametrize(
    "test_file", _test_files_with_pyspark_import(), ids=_test_rel
)
def test_pyspark_imports_guarded_in_tests(test_file: Path):
    rel = _test_rel(test_file)
    pytest.fail(
        f"\n{rel} imports pyspark but has no pytest.importorskip('pyspark') guard.\n"
        f"CI runs without PySpark — unguarded imports cause ModuleNotFoundError.\n"
        f"Add pytest.importorskip('pyspark') at the start of each test method that\n"
        f"needs pyspark, or add the file to _MODULE_LEVEL_GUARDED_TEST_FILES if it\n"
        f"has a module-level guard (pytestmark = pytest.mark.spark or importorskip at top)."
    )
