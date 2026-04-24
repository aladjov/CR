"""Render JSON predicate trees as business prose.

Sibling to ``predicate_compiler.predicate_to_sql`` with the same recursion
shape. The SQL output stays the source of truth for Spark execution; this
module is the dashboard / LLM-prompt counterpart that lifts column names
and numeric cutoffs into business language.

Mapping rules:

- ``{"op": ">=", "feature": "nps_score", "value": 4}``
  with a ``PopulationStats(q05=3, q25=5, q50=7, q75=8, q95=9)`` and
  ``polarity="high_is_bad"`` → ``"NPS score is low (≥ 4)"``.
- When ``feature_meta`` / ``column_descriptions`` / ``population_stats``
  are missing for a feature, the renderer falls back to the raw column
  name plus the numeric cutoff — readers never see a crash.
- AND / OR / NOT nesting mirrors ``predicate_to_sql`` exactly so the
  two outputs read the same structurally.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from customer_retention.stages.causal.interpretation.archetype_context import (
    _resolve_business_phrase,
    _resolve_polarity,
)
from customer_retention.stages.causal.interpretation.quantile_phrasing import (
    PopulationStats,
    quantile_phrase,
)

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
    from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow

_COMPARISON_WORDS = {
    ">=": "at or above",
    ">": "above",
    "<=": "at or below",
    "<": "below",
    "==": "equal to",
    "!=": "not equal to",
}


def compile_predicate_prose(
    predicate: Dict[str, Any],
    *,
    feature_meta: Optional[Mapping[str, FeatureMetaRow]] = None,
    population_stats: Optional[Mapping[str, PopulationStats]] = None,
    column_descriptions: Optional[Mapping[str, ColumnDescriptionRow]] = None,
) -> str:
    """Render ``predicate`` as business prose.

    Always returns a string — missing metadata degrades to the raw column
    name and literal value. Empty predicate → ``"always"``; ``{"op": "false"}``
    → ``"never"``.
    """
    meta = feature_meta or {}
    stats = population_stats or {}
    descriptions = column_descriptions or {}
    return _render(predicate or {}, meta, stats, descriptions)


def _render(
    predicate: Dict[str, Any],
    feature_meta: Mapping[str, FeatureMetaRow],
    population_stats: Mapping[str, PopulationStats],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> str:
    op = predicate.get("op")
    if not op:
        return "always"
    if op == "true":
        return "always"
    if op == "false":
        return "never"
    if op == "and":
        return _render_join(predicate, " AND ", feature_meta, population_stats, descriptions)
    if op == "or":
        return _render_join(predicate, " OR ", feature_meta, population_stats, descriptions)
    if op == "not":
        clause = predicate.get("clause") or {}
        return f"NOT ({_render(clause, feature_meta, population_stats, descriptions)})"
    if op in _COMPARISON_WORDS:
        return _render_comparison(predicate, op, feature_meta, population_stats, descriptions)
    if op == "in":
        return _render_membership(predicate, "one of", feature_meta, descriptions)
    if op == "not_in":
        return _render_membership(predicate, "not one of", feature_meta, descriptions)
    if op == "is_null":
        return f"{_phrase_for(predicate.get('feature', ''), feature_meta, descriptions)} is missing"
    if op == "not_null":
        return f"{_phrase_for(predicate.get('feature', ''), feature_meta, descriptions)} is present"
    return f"{_phrase_for(predicate.get('feature', ''), feature_meta, descriptions)} ({op})"


def _render_join(
    predicate: Dict[str, Any],
    joiner: str,
    feature_meta: Mapping[str, FeatureMetaRow],
    population_stats: Mapping[str, PopulationStats],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> str:
    clauses = predicate.get("clauses") or []
    if not clauses:
        return "always" if predicate.get("op") == "and" else "never"
    rendered = [_render(c, feature_meta, population_stats, descriptions) for c in clauses]
    if len(rendered) == 1:
        return rendered[0]
    return "(" + joiner.join(rendered) + ")"


def _render_comparison(
    predicate: Dict[str, Any],
    op: str,
    feature_meta: Mapping[str, FeatureMetaRow],
    population_stats: Mapping[str, PopulationStats],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> str:
    feature_name = str(predicate.get("feature", ""))
    value = predicate.get("value")
    phrase = _phrase_for(feature_name, feature_meta, descriptions)
    comparator = _COMPARISON_WORDS[op]
    ordinal = _ordinal_for(feature_name, value, feature_meta, population_stats, descriptions)
    literal = _render_literal(value)
    if ordinal == "unknown":
        return f"{phrase} is {comparator} {literal}"
    return f"{phrase} is {ordinal} ({comparator} {literal})"


def _render_membership(
    predicate: Dict[str, Any],
    connector: str,
    feature_meta: Mapping[str, FeatureMetaRow],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> str:
    feature_name = str(predicate.get("feature", ""))
    values = predicate.get("values") or []
    phrase = _phrase_for(feature_name, feature_meta, descriptions)
    rendered_values = ", ".join(_render_literal(v) for v in values)
    return f"{phrase} is {connector} {{{rendered_values}}}"


def _phrase_for(
    feature_name: str,
    feature_meta: Mapping[str, FeatureMetaRow],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> str:
    return _resolve_business_phrase(feature_name, feature_meta.get(feature_name), descriptions)


def _ordinal_for(
    feature_name: str,
    value: Any,
    feature_meta: Mapping[str, FeatureMetaRow],
    population_stats: Mapping[str, PopulationStats],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> str:
    stats = population_stats.get(feature_name)
    if stats is None:
        return "unknown"
    polarity = _resolve_polarity(feature_name, feature_meta.get(feature_name), descriptions) or "neutral"
    numeric = _as_float(value)
    return quantile_phrase(numeric, stats, polarity)


def _render_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    return repr(value)


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
