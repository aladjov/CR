from __future__ import annotations

from typing import List, Set

from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry
from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
    TransformationStep,
)

_GOLD_TYPE_MAP = {
    "log": PipelineTransformationType.LOG_TRANSFORM,
    "log_transform": PipelineTransformationType.LOG_TRANSFORM,
    "sqrt": PipelineTransformationType.SQRT_TRANSFORM,
    "sqrt_transform": PipelineTransformationType.SQRT_TRANSFORM,
    "yeo_johnson": PipelineTransformationType.YEO_JOHNSON,
    "zero_inflation_handling": PipelineTransformationType.ZERO_INFLATION_HANDLING,
    "cap_then_log": PipelineTransformationType.CAP_THEN_LOG,
}


def build_gold_steps(
    registry: RecommendationRegistry,
    pipeline_columns: Set[str],
) -> List[TransformationStep]:
    if not hasattr(registry, "gold") or registry.gold is None:
        return []
    gold = registry.gold
    return (
        _build_transformation_steps(gold, pipeline_columns)
        + _build_encoding_steps(gold, pipeline_columns)
        + _build_scaling_steps(gold, pipeline_columns)
    )


def _build_transformation_steps(gold, pipeline_columns: Set[str]) -> List[TransformationStep]:
    steps: List[TransformationStep] = []
    for rec in getattr(gold, "transformations", []):
        if rec.target_column not in pipeline_columns:
            continue
        trans_type = _GOLD_TYPE_MAP.get(rec.action)
        if trans_type is None:
            continue
        steps.append(TransformationStep(
            type=trans_type,
            column=rec.target_column,
            parameters=dict(rec.parameters) if rec.parameters else {},
            rationale=rec.rationale,
            source_notebook=rec.source_notebook,
        ))
    return steps


def _build_encoding_steps(gold, pipeline_columns: Set[str]) -> List[TransformationStep]:
    """Emit one ENCODE step per column with one_hot preference (PA-2).

    Mirrors ``FindingsParser._apply_gold_recommendations`` (findings_parser.py
    seeds ``seen_encoding_columns`` with the baseline ``one_hot`` step from
    ``_build_gold_config`` — always one_hot for every categorical column —
    and skips registry recs whose target_column is already in the seen
    set). Without de-dup here, NB08's ``apply_gold_transforms`` runs both
    a stale ``binary`` rec (label-encodes the string column to integer
    codes) AND a ``one_hot`` rec (which then one-hots those codes into
    positional ``_0``/``_1`` suffixes). Production codegen only ever
    runs one_hot — value-based ``_Emerging``/``_Enterprise`` suffixes —
    so positionally-named features NB08 selects (e.g.
    ``REVENUE_MARKET_SEGMENT_1``) never round-trip through gold.
    """
    chosen_by_column: dict = {}
    order: List[str] = []
    for rec in getattr(gold, "encoding", []):
        if rec.target_column not in pipeline_columns:
            continue
        col = rec.target_column
        method = _normalise_encoding_method(rec)
        existing = chosen_by_column.get(col)
        if existing is None:
            chosen_by_column[col] = rec
            order.append(col)
            continue
        if _normalise_encoding_method(existing) != "one_hot" and method == "one_hot":
            chosen_by_column[col] = rec
    steps: List[TransformationStep] = []
    for col in order:
        rec = chosen_by_column[col]
        steps.append(TransformationStep(
            type=PipelineTransformationType.ENCODE,
            column=col,
            parameters={"method": _normalise_encoding_method(rec)},
            rationale=rec.rationale,
            source_notebook=rec.source_notebook,
        ))
    return steps


def _build_scaling_steps(gold, pipeline_columns: Set[str]) -> List[TransformationStep]:
    steps: List[TransformationStep] = []
    for rec in getattr(gold, "scaling", []):
        if rec.target_column not in pipeline_columns:
            continue
        steps.append(TransformationStep(
            type=PipelineTransformationType.SCALE,
            column=rec.target_column,
            parameters={"method": _extract_method(rec, default="standard")},
            rationale=rec.rationale,
            source_notebook=rec.source_notebook,
        ))
    return steps


def _normalise_encoding_method(rec) -> str:
    method = _extract_method(rec, default="one_hot")
    if method in ("onehot", "one_hot"):
        return "one_hot"
    return method


def _extract_method(rec, *, default: str) -> str:
    params = rec.parameters or {}
    return params.get("method") or rec.action or default


def build_silver_derived_steps(
    registry: RecommendationRegistry,
    pipeline_columns: Set[str],
) -> List[TransformationStep]:
    if not hasattr(registry, "silver") or registry.silver is None:
        return []
    steps: List[TransformationStep] = []
    for rec in getattr(registry.silver, "derived_columns", []):
        action = rec.action
        if action not in ("ratio", "interaction", "composite"):
            continue
        params = dict(rec.parameters) if rec.parameters else {}
        sources = _derived_source_columns(action, params)
        if not sources.issubset(pipeline_columns):
            continue
        steps.append(TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column=rec.target_column,
            parameters={"action": action, **params},
            rationale=rec.rationale,
            source_notebook=rec.source_notebook,
        ))
    return steps


def _derived_source_columns(action: str, params: dict) -> Set[str]:
    if action == "ratio":
        return {params.get("numerator", ""), params.get("denominator", "")} - {""}
    if action == "interaction":
        return {params.get("col_a", ""), params.get("col_b", "")} - {""}
    if action == "composite":
        return set(params.get("columns", []))
    return set()
