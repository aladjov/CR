"""Build ``FeatureMetaRow`` lists from gold-layer lineage.

Two public helpers:

- ``parse_aggregation_feature_name`` — reverse-parses the canonical bronze
  aggregation naming convention (``{col}_{func}_{window}`` with optional
  ``event_count_{window}`` / ``days_since_*`` specials) into a
  ``FeatureLineage``. Returns ``None`` when the name does not match any
  known pattern — the caller then emits a minimal row with just the
  feature name.
- ``build_feature_meta_rows`` — converts a list of ``FeatureLineage``
  inputs into ``FeatureMetaRow`` objects, filling ``business_phrase``
  deterministically from an optional ``column_descriptions`` lookup.

Keeping the two helpers decoupled lets the caller drive the lineage list
from whatever source is available at generation time (``PipelineConfig``
walk, findings.yaml parse, or a manual list for tests) without entangling
the builder with ``FindingsParser`` internals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
    from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow

_AGG_FUNC_ALIASES = {
    "sum": "sum",
    "mean": "avg",
    "avg": "avg",
    "min": "min",
    "max": "max",
    "count": "count",
    "count_distinct": "count_distinct",
    "std": "avg",
    "stddev": "avg",
}

_WINDOW_SUFFIX_RE = re.compile(r"^(?P<body>.+?)_(?P<window>\d+)d$")
_EVENT_COUNT_RE = re.compile(r"^event_count_(?P<window>\d+)d$")
_EVENT_COUNT_ALL_TIME_RE = re.compile(r"^event_count_all_time$")
_DAYS_SINCE_RE = re.compile(r"^days_since_(last|first)_event$")
_INTER_EVENT_GAP_RE = re.compile(r"^inter_event_gap_(?P<stat>mean|std|min|max|median)$")

# Engagement / lifecycle framework-derived features: fixed vocabulary with
# known semantics. Maps name → (aggregation_kind, source_columns).
# These are emitted by the temporal-feature-engineering pipeline and don't
# follow the ``{col}_{func}_{window}d`` shape. Keyed by exact feature name.
_DERIVED_FEATURE_TAXONOMY: dict[str, tuple[str, list[str]]] = {
    "active_span_days": ("derived_datetime", ["event"]),
    "event_frequency": ("ratio", ["event"]),
    "regularity_score": ("passthrough", ["event"]),
    "event_count_all_time": ("count", ["event"]),
}

# Prefix-based matches for lifecycle / recency one-hots produced by the
# aggregator (``lifecycle_quadrant_intense_brief_lifecycle`` etc.).
_DERIVED_PREFIX_TAXONOMY: tuple[tuple[str, str, list[str]], ...] = (
    ("lifecycle_quadrant_", "passthrough", ["lifecycle"]),
    ("recency_bucket_", "passthrough", ["recency"]),
)


@dataclass
class FeatureLineage:
    """Minimal lineage shape a builder needs to emit a ``FeatureMetaRow``.

    All non-identifier fields stay optional so the caller can hand over
    partial lineage and let the builder fill the deterministic ones.
    """

    feature_name: str
    source_columns: Optional[List[str]] = None
    source_table: Optional[str] = None
    aggregation_kind: Optional[str] = None
    window_days: Optional[int] = None
    mask_future: Optional[bool] = None
    target_dependency: Optional[bool] = None
    polarity: Optional[str] = None


def parse_aggregation_feature_name(
    feature_name: str,
    *,
    value_columns: Iterable[str] = (),
    agg_funcs: Iterable[str] = (),
    source_table: Optional[str] = None,
) -> Optional[FeatureLineage]:
    """Reverse-parse the canonical ``{col}_{func}_{window}d`` naming.

    ``event_count_{w}d`` and ``days_since_{last|first}_event`` are recognized
    as specials. Returns ``None`` on any unrecognized shape — the caller
    is then free to emit a minimal row or a row built from another source.
    """
    event_count = _EVENT_COUNT_RE.match(feature_name)
    if event_count:
        return FeatureLineage(
            feature_name=feature_name,
            source_columns=["event"],
            source_table=source_table,
            aggregation_kind="count",
            window_days=int(event_count.group("window")),
        )
    days_since = _DAYS_SINCE_RE.match(feature_name)
    if days_since:
        return FeatureLineage(
            feature_name=feature_name,
            source_columns=["event"],
            source_table=source_table,
            aggregation_kind="recency_days",
        )
    gap = _INTER_EVENT_GAP_RE.match(feature_name)
    if gap:
        return FeatureLineage(
            feature_name=feature_name,
            source_columns=["event_gap"],
            source_table=source_table,
            aggregation_kind=_AGG_FUNC_ALIASES.get(gap.group("stat"), gap.group("stat")),
        )
    fixed = _DERIVED_FEATURE_TAXONOMY.get(feature_name)
    if fixed is not None:
        kind, source_columns = fixed
        return FeatureLineage(
            feature_name=feature_name,
            source_columns=list(source_columns),
            source_table=source_table,
            aggregation_kind=kind,
        )
    for prefix, kind, source_columns in _DERIVED_PREFIX_TAXONOMY:
        if feature_name.startswith(prefix):
            return FeatureLineage(
                feature_name=feature_name,
                source_columns=list(source_columns),
                source_table=source_table,
                aggregation_kind=kind,
            )
    windowed = _WINDOW_SUFFIX_RE.match(feature_name)
    if not windowed:
        return None
    body = windowed.group("body")
    window_days = int(windowed.group("window"))
    col, func = _split_col_and_func(body, value_columns, agg_funcs)
    if col is None or func is None:
        return None
    return FeatureLineage(
        feature_name=feature_name,
        source_columns=[col],
        source_table=source_table,
        aggregation_kind=_AGG_FUNC_ALIASES.get(func, func),
        window_days=window_days,
    )


def _split_col_and_func(
    body: str,
    value_columns: Iterable[str],
    agg_funcs: Iterable[str],
) -> tuple[Optional[str], Optional[str]]:
    """Given ``body = '{col}_{func}'`` resolve the boundary using the known
    column / function vocabulary. Ambiguity is resolved by longest-column
    match, then longest-function match."""
    cols_sorted = sorted((c for c in value_columns if c), key=len, reverse=True)
    funcs_sorted = sorted((f for f in agg_funcs if f), key=len, reverse=True)
    for col in cols_sorted:
        prefix = f"{col}_"
        if body.startswith(prefix):
            tail = body[len(prefix):]
            if tail in _AGG_FUNC_ALIASES or tail in funcs_sorted:
                return col, tail
    for func in funcs_sorted:
        suffix = f"_{func}"
        if body.endswith(suffix):
            return body[: -len(suffix)], func
    return None, None


def build_feature_meta_rows(
    composite_name: str,
    lineages: Iterable[FeatureLineage],
    *,
    column_descriptions: Optional[Mapping[str, "ColumnDescriptionRow"]] = None,
) -> List["FeatureMetaRow"]:
    """Convert lineages into ``FeatureMetaRow``s with rendered phrases."""
    from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow

    rows: List[FeatureMetaRow] = []
    for lineage in lineages:
        polarity = lineage.polarity or _resolve_polarity(lineage, column_descriptions)
        # Backfill: when the caller couldn't determine real lineage, fall back
        # to the feature name itself so source_columns is never empty. This
        # guarantees ``compile_predicate_prose`` always has at least one
        # column to look up in ``column_descriptions``. Real lineage from
        # ``parse_aggregation_feature_name`` or upstream metadata wins.
        source_columns = lineage.source_columns
        if not source_columns:
            source_columns = [lineage.feature_name]
        aggregation_kind = lineage.aggregation_kind or "passthrough"
        base = FeatureMetaRow(
            composite_name=composite_name,
            feature_name=lineage.feature_name,
            source_columns=source_columns,
            source_table=lineage.source_table,
            aggregation_kind=aggregation_kind,
            window_days=lineage.window_days,
            target_dependency=lineage.target_dependency,
            mask_future=lineage.mask_future,
            polarity=polarity,
        )
        source_business_name = _resolve_business_name(lineage, column_descriptions)
        rows.append(base.with_rendered_phrases(source_business_name))
    return rows


def _resolve_business_name(
    lineage: FeatureLineage,
    column_descriptions: Optional[Mapping[str, "ColumnDescriptionRow"]],
) -> Optional[str]:
    if not column_descriptions or not lineage.source_columns:
        return None
    for candidate in lineage.source_columns:
        entry = column_descriptions.get(candidate)
        if entry and entry.business_name:
            return entry.business_name
    return None


def _resolve_polarity(
    lineage: FeatureLineage,
    column_descriptions: Optional[Mapping[str, "ColumnDescriptionRow"]],
) -> Optional[str]:
    """Inherit polarity from the dominant source column per plan §1.2."""
    if not column_descriptions or not lineage.source_columns:
        return None
    for candidate in lineage.source_columns:
        entry = column_descriptions.get(candidate)
        if entry and entry.polarity:
            return entry.polarity
    return None
