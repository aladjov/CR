"""Resolve a derived feature name back to its source dataset and base column.

The exploration pipeline produces feature names like
``lag2_NET_PRICE_sum_180d_is_zero`` from a single raw column
``NET_PRICE`` in the ``subscription`` source. NB09 needs the inverse map
so the post-training feature set can be grouped by source dataset and
joined against NB05's cached target correlations — both to validate that
the model's top features actually originate from datasets the user expects,
and to surface features that NB05 never saw (gold transforms / silver
derivations) and therefore could not vet for leakage.

The grammar implicit in `findings_parser._event_aggregated_columns` and
`stages/profiling/time_window_aggregator.py` is:

    {lag\\d+_|velocity_}? {base} (_{agg_func} (_{window})? (_is_zero|_log)? )?
                                 | _delta_hours | _hour | _dow | _is_weekend
                                 | _vs_cohort_pct | _cohort_zscore | …

Rather than maintain a brittle suffix table, this module uses a
**longest-base-prefix-match** strategy: the parser walks each known raw
column name and asks "is the (lag-stripped) feature name equal to this
column or does it start with `column_`?". The longest matching column
wins, which correctly handles cases where one raw column name is a prefix
of another (e.g. ``NET_PRICE`` vs ``NET_PRICE_TIER``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

_LAG_PREFIX_RE = re.compile(r"^(lag\d+|velocity|momentum|acceleration)_")

_INTERACTION_TOKEN_RE = re.compile(r"(^|_)(interaction|x|ratio)(_|$)")


def _is_interaction_feature(feature_name: str) -> bool:
    """Return True if the feature name looks like an interaction / ratio.

    Interaction features combine two or more raw columns via multiply,
    divide, ratio, or a named ``interaction_`` prefix. Their source
    column dtype is ambiguous (which of the N inputs?), so NB09 shows
    no dtype for them.
    """
    return bool(_INTERACTION_TOKEN_RE.search(feature_name.lower()))


@dataclass(frozen=True)
class ParsedFeature:
    """A derived feature decomposed into its origin metadata."""

    feature: str
    source: Optional[str]
    base_column: Optional[str]
    family: str
    lag_prefix: Optional[str]

    @property
    def is_resolved(self) -> bool:
        return self.source is not None and self.base_column is not None


def parse_feature_provenance(
    feature_name: str,
    source_columns: Dict[str, Iterable[str]],
) -> ParsedFeature:
    """Resolve ``feature_name`` to ``(source, base_column, family)``.

    Args:
        feature_name: A derived feature column name (e.g.
            ``"lag2_NET_PRICE_sum_180d_is_zero"``).
        source_columns: ``{source_name: iterable_of_raw_column_names}`` —
            typically built from `MultiDatasetFindings` per-dataset
            ``ExplorationFindings.columns.keys()``.

    Returns:
        ``ParsedFeature``. ``source`` and ``base_column`` will be ``None``
        if no source's raw columns prefix-match the (lag-stripped) name.
    """
    name = feature_name
    lag_prefix: Optional[str] = None
    m = _LAG_PREFIX_RE.match(name)
    if m:
        lag_prefix = m.group(1)
        name = name[m.end():]

    best_source: Optional[str] = None
    best_base: Optional[str] = None
    best_family = ""
    for source, cols in source_columns.items():
        for col in cols:
            if not col:
                continue
            if name == col:
                if best_base is None or len(col) > len(best_base):
                    best_source = source
                    best_base = col
                    best_family = ""
            elif name.startswith(col + "_"):
                if best_base is None or len(col) > len(best_base):
                    best_source = source
                    best_base = col
                    best_family = name[len(col) + 1:]

    return ParsedFeature(
        feature=feature_name,
        source=best_source,
        base_column=best_base,
        family=best_family,
        lag_prefix=lag_prefix,
    )


def build_source_column_map(
    multi_dataset_path,
    findings_loader=None,
) -> Dict[str, Set[str]]:
    """Walk a multi_dataset_findings.yaml and build {source: {raw_col, …}}.

    Args:
        multi_dataset_path: Path to ``multi_dataset_findings.yaml``.
        findings_loader: Optional callable ``(path) -> ExplorationFindings``
            for tests. Defaults to
            ``ExplorationFindings.load`` at runtime.
    """
    import yaml

    if findings_loader is None:
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        findings_loader = ExplorationFindings.load

    from pathlib import Path

    multi_path = Path(str(multi_dataset_path))
    if not multi_path.exists():
        return {}
    with multi_path.open("r") as f:
        multi = yaml.safe_load(f) or {}

    result: Dict[str, Set[str]] = {}
    for name, info in (multi.get("datasets") or {}).items():
        # Strip the "_aggregated" suffix the parser uses for event sources.
        clean = name[: -len("_aggregated")] if name.endswith("_aggregated") else name
        path = info.get("findings_path")
        if not path:
            continue
        try:
            findings = findings_loader(path)
        except Exception:
            continue
        cols = set(getattr(findings, "columns", {}).keys()) if findings else set()
        if cols:
            result[clean] = cols
    return result


def cached_target_correlations(recommendations) -> Dict[str, float]:
    """Extract NB05's cached ``prioritize`` correlations as a dict.

    The relationship recommender writes one ``feature_selection``
    recommendation per analyzed feature with action ``prioritize`` and
    ``parameters['correlation']`` set. Returns ``{feature: |correlation|}``.
    Returns an empty dict for any malformed / missing input.
    """
    if recommendations is None:
        return {}
    gold = getattr(recommendations, "gold", None)
    if gold is None:
        return {}
    out: Dict[str, float] = {}
    for rec in getattr(gold, "feature_selection", []) or []:
        if getattr(rec, "action", "") != "prioritize":
            continue
        params = getattr(rec, "parameters", None) or {}
        corr = params.get("correlation")
        if corr is None:
            continue
        try:
            out[rec.target_column] = abs(float(corr))
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class FeatureProvenanceRow:
    feature: str
    source: str
    base_column: str
    family: str
    lag_prefix: str
    importance: float
    nb05_correlation: Optional[float]
    nb05_status: str  # "confirmed" / "new" / "missing"
    original_dtype: Optional[str] = None


def build_source_column_types(
    multi_dataset_path,
    findings_loader=None,
) -> Dict[str, Dict[str, str]]:
    """Walk a multi_dataset_findings.yaml and build {source: {raw_col: dtype_name}}.

    Mirrors :func:`build_source_column_map` but carries the per-column
    ``inferred_type`` string for each source. Callers pass the result to
    :func:`build_provenance_table` so the NB09 provenance view can show
    the original column type next to each derived feature.
    """
    import yaml

    if findings_loader is None:
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        findings_loader = ExplorationFindings.load

    from pathlib import Path

    multi_path = Path(str(multi_dataset_path))
    if not multi_path.exists():
        return {}
    with multi_path.open("r") as f:
        multi = yaml.safe_load(f) or {}

    result: Dict[str, Dict[str, str]] = {}
    for name, info in (multi.get("datasets") or {}).items():
        clean = name[: -len("_aggregated")] if name.endswith("_aggregated") else name
        path = info.get("findings_path")
        if not path:
            continue
        try:
            findings = findings_loader(path)
        except Exception:
            continue
        cols = getattr(findings, "columns", {}) or {}
        types = {col: getattr(c.inferred_type, "value", str(c.inferred_type)) for col, c in cols.items()}
        if types:
            result[clean] = types
    return result


def build_provenance_table(
    feature_importances: Dict[str, float],
    source_columns: Dict[str, Iterable[str]],
    cached_correlations: Dict[str, float],
    *,
    top_n: int = 20,
    source_column_types: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[FeatureProvenanceRow]:
    """Join model feature importances with provenance + NB05 correlations.

    Returns a list of rows ordered by importance (descending), capped at
    ``top_n``. Each row carries one of three statuses:

    - ``confirmed``: feature was in NB05's analyzed set with a non-trivial
      cached correlation (>= 0.1).
    - ``new``: feature did not exist when NB05 ran (gold transform or
      silver derivation added afterwards). NB05 cannot have vetted it.
    - ``missing``: feature existed at NB05 time but had no cached
      correlation (NB05 chose not to prioritize it / dropped it earlier).
    """
    if not feature_importances:
        return []
    sorted_feats = sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rows: List[FeatureProvenanceRow] = []
    for feat, importance in sorted_feats:
        parsed = parse_feature_provenance(feat, source_columns)
        corr = cached_correlations.get(feat)
        if corr is None:
            # Try the resolved base column too — NB05 sometimes prioritizes
            # the raw column rather than the derived form.
            if parsed.base_column is not None:
                corr = cached_correlations.get(parsed.base_column)
        if corr is None:
            status = "new"
        elif corr >= 0.1:
            status = "confirmed"
        else:
            status = "missing"
        dtype = None
        if (
            source_column_types
            and parsed.source
            and parsed.base_column
            and not _is_interaction_feature(feat)
        ):
            dtype = (source_column_types.get(parsed.source) or {}).get(parsed.base_column)
        rows.append(
            FeatureProvenanceRow(
                feature=feat,
                source=parsed.source or "(unresolved)",
                base_column=parsed.base_column or feat,
                family=parsed.family,
                lag_prefix=parsed.lag_prefix or "",
                importance=float(importance),
                nb05_correlation=corr,
                nb05_status=status,
                original_dtype=dtype,
            )
        )
    return rows


def source_histogram(rows: List[FeatureProvenanceRow]) -> List[tuple]:
    """Aggregate provenance rows by source. Returns list of (source, count, mean_importance)."""
    if not rows:
        return []
    counts: Counter = Counter()
    importances: Dict[str, float] = {}
    for row in rows:
        counts[row.source] += 1
        importances[row.source] = importances.get(row.source, 0.0) + row.importance
    return [
        (src, counts[src], importances[src] / counts[src])
        for src in sorted(counts, key=counts.get, reverse=True)
    ]


def build_overall_source_distribution(
    model_importances: Dict[str, Dict[str, float]],
    source_columns: Dict[str, Iterable[str]],
) -> List[tuple]:
    """Count unique features per source across every trained model.

    Walks the union of feature names used by any model (deduped) and
    resolves each one to its source dataset via
    :func:`parse_feature_provenance`. Returns ``[(source, count), ...]``
    sorted by count descending — this powers the NB09 overall pie chart
    showing which datasets the trained models lean on the most.

    Unresolved features (interactions, composites that don't prefix-match
    any raw column) are grouped under ``"(unresolved)"``.
    """
    unique_features: Set[str] = set()
    for imps in (model_importances or {}).values():
        if not imps:
            continue
        unique_features.update(imps.keys())
    if not unique_features:
        return []
    counts: Counter = Counter()
    for feat in unique_features:
        parsed = parse_feature_provenance(feat, source_columns)
        counts[parsed.source or "(unresolved)"] += 1
    return [(src, counts[src]) for src in sorted(counts, key=counts.get, reverse=True)]
