from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple

from customer_retention.core.compat import DataFrame, Series

from ..components.enums import Severity

TEMPORAL_METADATA_COLUMNS: FrozenSet[str] = frozenset({
    "feature_timestamp",
    "label_timestamp",
    "label_available_flag",
    "event_timestamp",
})


def _build_exclusion_set(entity_column: Optional[str], target_column: Optional[str], additional_exclude: Optional[Set[str]]) -> Set[str]:
    exclude = set(TEMPORAL_METADATA_COLUMNS)
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
