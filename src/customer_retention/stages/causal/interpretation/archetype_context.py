"""Phase 3 — deterministic archetype context enrichment.

Transforms the raw ``ArchetypeContext`` that the mapper hands to the
namer (cluster size, churn probability, raw driver names + mean SHAP +
mean value, playbook prose) into an ``EnrichedArchetypeContext`` where
every quantity is already narrated in business terms:

- raw column names → ``business_phrase`` via ``feature_meta`` + ``column_descriptions``
- raw numeric ``mean_value`` → ordinal band via ``quantile_phrase`` + ``PopulationStats``
- pairwise cluster contrast → ``ContrastFeature`` with value phrases
- policy + expected-effect prose lifted from playbook records for the LLM

The LLM namer (Phase 4) consumes this structure and only chooses a name /
narrates — it cannot fabricate quantities because none are left to compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from customer_retention.stages.causal.interpretation.quantile_phrasing import (
    PopulationStats,
    quantile_phrase,
)

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
    from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow


@dataclass
class EnrichedDriver:
    """One cluster-level SHAP driver with business narration attached."""

    feature_name: str
    business_phrase: str
    value: Optional[float]
    value_phrase: str
    polarity: Optional[str]
    mean_shap: Optional[float] = None
    stability: Optional[float] = None


@dataclass
class ContrastFeature:
    """One feature where this archetype differs from its nearest sibling."""

    feature_name: str
    business_phrase: str
    self_value: Optional[float]
    sibling_value: Optional[float]
    self_phrase: str
    sibling_phrase: str


@dataclass
class EnrichedPlaybook:
    """Playbook record lifted into prose the LLM can cite verbatim."""

    playbook_id: str
    name: str
    when_applicable: Optional[str] = None
    description: Optional[str] = None
    policy_summary: Optional[str] = None
    expected_effect: Optional[str] = None
    fit_score: Optional[float] = None


@dataclass
class EnrichedArchetypeContext:
    """Full enrichment passed into the LLM prompt (Phase 4)."""

    cluster_index: int
    cluster_size: int
    cluster_mean_churn_probability: float
    share_of_book: Optional[float] = None
    lift_vs_population: Optional[float] = None
    arr_exposure: Optional[float] = None
    top_positive_drivers: List[EnrichedDriver] = field(default_factory=list)
    top_negative_drivers: List[EnrichedDriver] = field(default_factory=list)
    sibling_contrast: List[ContrastFeature] = field(default_factory=list)
    representative_examples: List[Dict[str, Any]] = field(default_factory=list)
    eligibility_rule_prose: Optional[str] = None
    candidate_playbooks: List[EnrichedPlaybook] = field(default_factory=list)


def build_enriched_context(
    raw: Any,
    *,
    feature_meta: Mapping[str, FeatureMetaRow],
    population_stats: Mapping[str, PopulationStats],
    column_descriptions: Optional[Mapping[str, ColumnDescriptionRow]] = None,
    raw_candidate_playbooks: Optional[Sequence[Mapping[str, Any]]] = None,
    sibling_contrast: Optional[Sequence[Mapping[str, Any]]] = None,
    representative_examples: Optional[Sequence[Dict[str, Any]]] = None,
    eligibility_rule_prose: Optional[str] = None,
    driver_stability: Optional[Mapping[str, float]] = None,
    population_mean_churn: Optional[float] = None,
    total_book_size: Optional[int] = None,
    arr_exposure: Optional[float] = None,
) -> EnrichedArchetypeContext:
    """Compose the enriched context from the mapper's raw ``ArchetypeContext``.

    ``raw`` is any object exposing ``cluster_index``, ``cluster_size``,
    ``cluster_mean_churn_probability``, ``top_positive_drivers``,
    ``top_negative_drivers``, and (optionally) ``candidate_playbooks`` —
    matches ``llm_namer.ArchetypeContext`` without taking a hard import
    dependency.
    """
    descriptions = column_descriptions or {}
    stability = driver_stability or {}
    playbooks_source = (
        raw_candidate_playbooks
        if raw_candidate_playbooks is not None
        else getattr(raw, "candidate_playbooks", []) or []
    )
    positives = [
        _build_driver(d, feature_meta, population_stats, descriptions, stability)
        for d in getattr(raw, "top_positive_drivers", []) or []
    ]
    negatives = [
        _build_driver(d, feature_meta, population_stats, descriptions, stability)
        for d in getattr(raw, "top_negative_drivers", []) or []
    ]
    contrasts = [
        _build_contrast(c, feature_meta, population_stats, descriptions)
        for c in (sibling_contrast or [])
    ]
    return EnrichedArchetypeContext(
        cluster_index=int(getattr(raw, "cluster_index", 0)),
        cluster_size=int(getattr(raw, "cluster_size", 0)),
        cluster_mean_churn_probability=float(
            getattr(raw, "cluster_mean_churn_probability", 0.0)
        ),
        share_of_book=_share_of_book(getattr(raw, "cluster_size", 0), total_book_size),
        lift_vs_population=_lift_vs_population(
            getattr(raw, "cluster_mean_churn_probability", None),
            population_mean_churn,
        ),
        arr_exposure=arr_exposure,
        top_positive_drivers=positives,
        top_negative_drivers=negatives,
        sibling_contrast=contrasts,
        representative_examples=list(representative_examples or []),
        eligibility_rule_prose=eligibility_rule_prose,
        candidate_playbooks=[_build_playbook(p) for p in playbooks_source],
    )


def _build_driver(
    raw: Mapping[str, Any],
    feature_meta: Mapping[str, FeatureMetaRow],
    population_stats: Mapping[str, PopulationStats],
    descriptions: Mapping[str, ColumnDescriptionRow],
    stability: Mapping[str, float],
) -> EnrichedDriver:
    feature_name = str(raw.get("feature", ""))
    meta = feature_meta.get(feature_name)
    phrase = _resolve_business_phrase(feature_name, meta, descriptions)
    polarity = _resolve_polarity(feature_name, meta, descriptions)
    value = _as_float(raw.get("mean_value"))
    value_phrase = quantile_phrase(
        value, population_stats.get(feature_name) or PopulationStats(), polarity or "neutral",
    )
    return EnrichedDriver(
        feature_name=feature_name,
        business_phrase=phrase,
        value=value,
        value_phrase=value_phrase,
        polarity=polarity,
        mean_shap=_as_float(raw.get("mean_shap")),
        stability=stability.get(feature_name),
    )


def _build_contrast(
    raw: Mapping[str, Any],
    feature_meta: Mapping[str, FeatureMetaRow],
    population_stats: Mapping[str, PopulationStats],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> ContrastFeature:
    feature_name = str(raw.get("feature", ""))
    meta = feature_meta.get(feature_name)
    phrase = _resolve_business_phrase(feature_name, meta, descriptions)
    polarity = _resolve_polarity(feature_name, meta, descriptions) or "neutral"
    stats = population_stats.get(feature_name) or PopulationStats()
    self_value = _as_float(raw.get("self_value"))
    sibling_value = _as_float(raw.get("sibling_value"))
    return ContrastFeature(
        feature_name=feature_name,
        business_phrase=phrase,
        self_value=self_value,
        sibling_value=sibling_value,
        self_phrase=quantile_phrase(self_value, stats, polarity),
        sibling_phrase=quantile_phrase(sibling_value, stats, polarity),
    )


def _build_playbook(raw: Mapping[str, Any]) -> EnrichedPlaybook:
    return EnrichedPlaybook(
        playbook_id=str(raw.get("playbook_id") or raw.get("id") or ""),
        name=str(raw.get("name") or raw.get("playbook_id") or ""),
        when_applicable=raw.get("when_applicable"),
        description=raw.get("description"),
        policy_summary=raw.get("policy_summary"),
        expected_effect=raw.get("expected_effect"),
        fit_score=_as_float(raw.get("fit_score")),
    )


def _resolve_business_phrase(
    feature_name: str,
    meta: Optional[FeatureMetaRow],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> str:
    if meta and meta.business_phrase:
        return meta.business_phrase
    if meta and meta.source_columns:
        for source_col in meta.source_columns:
            entry = descriptions.get(source_col)
            if entry and entry.business_name:
                return entry.business_name
    entry = descriptions.get(feature_name)
    if entry and entry.business_name:
        return entry.business_name
    return feature_name


def _resolve_polarity(
    feature_name: str,
    meta: Optional[FeatureMetaRow],
    descriptions: Mapping[str, ColumnDescriptionRow],
) -> Optional[str]:
    if meta and meta.polarity:
        return meta.polarity
    if meta and meta.source_columns:
        for source_col in meta.source_columns:
            entry = descriptions.get(source_col)
            if entry and entry.polarity:
                return entry.polarity
    entry = descriptions.get(feature_name)
    return entry.polarity if entry else None


def _share_of_book(cluster_size: Any, total: Optional[int]) -> Optional[float]:
    if not total or total <= 0:
        return None
    return float(cluster_size) / float(total)


def _lift_vs_population(cluster_mean: Any, population_mean: Optional[float]) -> Optional[float]:
    cluster_f = _as_float(cluster_mean)
    if cluster_f is None or population_mean is None or population_mean == 0:
        return None
    return cluster_f / float(population_mean)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ContrastFeature",
    "EnrichedArchetypeContext",
    "EnrichedDriver",
    "EnrichedPlaybook",
    "build_enriched_context",
]
