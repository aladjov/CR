"""Deterministic business-phrase rendering for gold features.

Pure, IO-free helpers used both at pipeline-generation time (when building
``feature_meta`` rows) and at render time (dashboard, LLM prompt).
Unknown aggregation kinds degrade to the bare ``source_business_name``
rather than raising — lineage is not always complete.
"""
from __future__ import annotations

from typing import Optional

_AGGREGATION_VERB: dict[str, str] = {
    "count": "count of",
    "count_distinct": "distinct count of",
    "sum": "sum of",
    "avg": "average",
    "mean": "average",
    "max": "maximum",
    "min": "minimum",
    "last": "most recent",
    "first": "first observed",
    "ratio": "share of",
    "recency_days": "days since most recent",
}

_WINDOWED_KINDS = frozenset(
    {"count", "count_distinct", "sum", "avg", "mean", "max", "min", "ratio"}
)


def render_business_phrase(
    aggregation_kind: Optional[str],
    source_business_name: str,
    window_phrase: Optional[str] = None,
) -> str:
    """Render a single feature's business phrase."""
    name = (source_business_name or "").strip()
    if not name:
        return ""
    if aggregation_kind is None:
        return name
    verb = _AGGREGATION_VERB.get(aggregation_kind)
    if verb is None:
        return name
    if window_phrase and aggregation_kind in _WINDOWED_KINDS:
        return f"{verb} {name} over {window_phrase}"
    return f"{verb} {name}"


def render_window_phrase(window_days: Optional[int]) -> str:
    """Humanize an aggregation window in days.

    ``None`` / non-positive → ``"lifetime"``; positive integers → ``"last N days"``.
    """
    if window_days is None or window_days <= 0:
        return "lifetime"
    return f"last {window_days} days"
