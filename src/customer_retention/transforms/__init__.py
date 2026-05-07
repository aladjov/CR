"""Lean runtime module for DataFrame transformations.

This module is safe to import from generated pipelines — it depends only on
``core.compat``, ``numpy``, ``sklearn``, ``joblib``, and the lightweight
``models`` dataclasses.  It does **not** import from ``analysis/``,
``generators/`` (except ``models``), ``visualization/``, or ``stages/``.
"""

from .artifact_store import ArtifactStore
from .executor import (
    TransformExecutor,
    format_batch_progress,
    format_step_progress,
    print_batch_progress,
    print_step_progress,
)
from .ops import (
    apply_cap_outlier,
    apply_cap_then_log,
    apply_derived_composite,
    apply_derived_cyclical,
    apply_derived_duration,
    apply_derived_extraction_is_weekend,
    apply_derived_interaction,
    apply_derived_ratio,
    apply_derived_recency,
    apply_derived_tenure,
    apply_drop_column,
    apply_feature_select,
    apply_impute_null,
    apply_log_transform,
    apply_one_hot_encode,
    apply_segment_aware_cap,
    apply_sqrt_transform,
    apply_type_cast,
    apply_winsorize,
    apply_zero_inflation_handling,
)

__all__ = [
    "TransformExecutor",
    "ArtifactStore",
    "format_step_progress",
    "format_batch_progress",
    "print_step_progress",
    "print_batch_progress",
    "apply_impute_null",
    "apply_cap_outlier",
    "apply_type_cast",
    "apply_drop_column",
    "apply_winsorize",
    "apply_segment_aware_cap",
    "apply_log_transform",
    "apply_sqrt_transform",
    "apply_zero_inflation_handling",
    "apply_cap_then_log",
    "apply_one_hot_encode",
    "apply_feature_select",
    "apply_derived_ratio",
    "apply_derived_interaction",
    "apply_derived_composite",
    "apply_derived_recency",
    "apply_derived_duration",
    "apply_derived_cyclical",
    "apply_derived_tenure",
    "apply_derived_extraction_is_weekend",
]
