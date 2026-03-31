from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Optional, Set, Tuple

from customer_retention.core.compat import (
    DataFrame,
    Series,
    bulk_null_corr_with_target,
    leakage_corr_combined,
)

from ..components.enums import Severity

TEMPORAL_METADATA_COLUMNS: FrozenSet[str] = frozenset(
    {
        "feature_timestamp",
        "label_timestamp",
        "label_available_flag",
    }
)


SOURCE_TIMESTAMP_COLUMNS: FrozenSet[str] = frozenset({"event_timestamp"})


def _build_exclusion_set(
    entity_column: Optional[str], target_column: Optional[str], additional_exclude: Optional[Set[str]]
) -> Set[str]:
    exclude = set(TEMPORAL_METADATA_COLUMNS) | set(SOURCE_TIMESTAMP_COLUMNS)
    if entity_column:
        exclude.add(entity_column)
    if target_column:
        exclude.add(target_column)
    if additional_exclude:
        exclude.update(additional_exclude)
    return exclude


def get_valid_feature_columns(
    df: DataFrame,
    entity_column: Optional[str] = None,
    target_column: Optional[str] = None,
    additional_exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Filter DataFrame columns to those valid as model features."""
    exclude = _build_exclusion_set(entity_column, target_column, additional_exclude)
    exclude.update(c for c in df.columns if c.startswith("original_"))
    return [c for c in df.columns if c not in exclude]


@dataclass
class LeakageThresholds:
    correlation_critical: float = 0.90
    correlation_high: float = 0.70
    correlation_medium: float = 0.50
    separation_critical: float = 0.0
    separation_high: float = 1.0
    separation_medium: float = 5.0
    auc_critical: float = 0.90
    auc_high: float = 0.80


DEFAULT_THRESHOLDS = LeakageThresholds()


def classify_correlation(corr: float, thresholds: LeakageThresholds = DEFAULT_THRESHOLDS) -> Tuple[Severity, str]:
    abs_corr = abs(corr)
    if abs_corr >= thresholds.correlation_critical:
        return Severity.CRITICAL, "high_correlation"
    if abs_corr >= thresholds.correlation_high:
        return Severity.HIGH, "suspicious_correlation"
    if abs_corr >= thresholds.correlation_medium:
        return Severity.MEDIUM, "elevated_correlation"
    return Severity.INFO, "normal"


def classify_separation(overlap_pct: float, thresholds: LeakageThresholds = DEFAULT_THRESHOLDS) -> Tuple[Severity, str]:
    if overlap_pct <= thresholds.separation_critical:
        return Severity.CRITICAL, "perfect_separation"
    if overlap_pct < thresholds.separation_high:
        return Severity.HIGH, "near_perfect_separation"
    if overlap_pct < thresholds.separation_medium:
        return Severity.MEDIUM, "high_separation"
    return Severity.INFO, "normal"


def _null_correlated_columns(
    df: "DataFrame",
    columns: List[str],
    target_column: str,
    threshold: float,
    precomputed_null_counts: Optional[dict] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> List[str]:
    if not columns:
        return []
    corrs = bulk_null_corr_with_target(
        df, columns, target_column,
        progress_fn=progress_fn, precomputed_null_counts=precomputed_null_counts,
    )
    return [c for c in columns if abs(corrs.get(c, 0.0)) >= threshold]


def detect_target_leaking_datetime_columns(
    df: "DataFrame",
    datetime_columns: List[str],
    target_column: Optional[str],
    null_correlation_threshold: float = 0.8,
) -> List[str]:
    if not target_column or not datetime_columns:
        return []
    return _null_correlated_columns(df, datetime_columns, target_column, null_correlation_threshold)


def detect_leaking_features(
    df: "DataFrame",
    feature_columns: List[str],
    target_column: Optional[str],
    null_correlation_threshold: float = 0.8,
    value_correlation_threshold: float = 0.95,
    progress_fn: Optional[Callable[[str], None]] = None,
    precomputed_value_corrs: Optional[dict] = None,
    precomputed_null_counts: Optional[dict] = None,
) -> List[str]:
    import time
    log = progress_fn or (lambda msg: None)
    if not target_column or not feature_columns:
        return []
    if target_column not in df.columns:
        return []
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        return []
    log(f"Leakage check: {len(available)} columns")
    t0 = time.monotonic()
    if precomputed_value_corrs is not None:
        null_leakers = set(
            _null_correlated_columns(
                df, available, target_column, null_correlation_threshold,
                precomputed_null_counts=precomputed_null_counts, progress_fn=progress_fn,
            ),
        )
        log(f"  [1/2] Null-pattern check: {len(null_leakers)} leaker(s) found ({time.monotonic() - t0:.0f}s)")
        value_candidates = [c for c in available if c not in null_leakers]
        if not value_candidates:
            return [col for col in feature_columns if col in null_leakers]
        t1 = time.monotonic()
        corrs = {c: precomputed_value_corrs[c] for c in value_candidates if c in precomputed_value_corrs}
        value_leakers = {col for col, val in corrs.items() if abs(val) >= value_correlation_threshold}
        log(f"  [2/2] Value correlation: {len(value_leakers)} leaker(s) found ({time.monotonic() - t1:.0f}s)")
    else:
        null_corrs, value_corrs = leakage_corr_combined(df, available, target_column, progress_fn)
        log(f"  [1/2+2/2] Combined correlation pass ({time.monotonic() - t0:.0f}s)")
        null_leakers = {c for c in available if abs(null_corrs.get(c, 0.0)) >= null_correlation_threshold}
        value_candidates = [c for c in available if c not in null_leakers]
        value_leakers = {
            c for c in value_candidates
            if abs(value_corrs.get(c, 0.0)) >= value_correlation_threshold
        }
        log(f"    null-pattern: {len(null_leakers)}, value: {len(value_leakers)}")
    combined = null_leakers | value_leakers
    log(f"  Done: {len(combined)} total leaker(s) in {time.monotonic() - t0:.0f}s")
    return [col for col in feature_columns if col in combined]


def calculate_class_overlap(feature: Series, target: Series) -> float:
    class_0, class_1 = feature[target == 0].dropna(), feature[target == 1].dropna()
    if len(class_0) == 0 or len(class_1) == 0:
        return 100.0
    min_0, max_0 = class_0.min(), class_0.max()
    min_1, max_1 = class_1.min(), class_1.max()
    total_range = max(max_0, max_1) - min(min_0, min_1)
    if total_range == 0:
        return 100.0
    overlap = max(0, min(max_0, max_1) - max(min_0, min_1))
    return (overlap / total_range) * 100
