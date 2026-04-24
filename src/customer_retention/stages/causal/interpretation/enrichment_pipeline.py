"""Single-call entrypoint composing Phases 1-3 against a ``RunNamespace``.

``enrich_archetype_from_namespace(raw_context, namespace, composite_name, ...)``
loads the three sidecars (feature_meta, feature_population_stats,
column_descriptions), optionally renders the eligibility predicate as prose
via ``compile_predicate_prose``, and returns a fully populated
``EnrichedArchetypeContext`` ready to hand to ``LLMNamer.name_archetype``.

This is the one call the causal c02 notebook makes per archetype — keeps
all the sidecar / phrasing / enrichment plumbing behind a single name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

from customer_retention.stages.causal.interpretation.archetype_context import (
    EnrichedArchetypeContext,
    build_enriched_context,
)
from customer_retention.stages.causal.interpretation.predicate_prose import (
    compile_predicate_prose,
)
from customer_retention.stages.causal.interpretation.sidecars import (
    load_column_descriptions_sidecar,
    load_feature_meta_sidecar,
    load_population_stats_sidecar,
)

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace


def enrich_archetype_from_namespace(
    raw_context: Any,
    namespace: "RunNamespace",
    composite_name: str,
    *,
    eligibility_predicate: Optional[Dict[str, Any]] = None,
    siblings: Optional[Sequence[Any]] = None,
    sibling_contrast: Optional[Sequence[Mapping[str, Any]]] = None,
    representative_examples: Optional[Sequence[Dict[str, Any]]] = None,
    raw_candidate_playbooks: Optional[Sequence[Mapping[str, Any]]] = None,
    driver_stability: Optional[Mapping[str, float]] = None,
    population_mean_churn: Optional[float] = None,
    total_book_size: Optional[int] = None,
    arr_exposure: Optional[float] = None,
) -> EnrichedArchetypeContext:
    """Compose a fully-narrated archetype context from sidecar metadata.

    All sidecars degrade to empty mappings when missing — the returned
    ``EnrichedArchetypeContext`` is always usable; callers fall through
    to raw column names / "unknown" band phrases when lineage is absent.
    """
    feature_meta = load_feature_meta_sidecar(namespace, composite_name=composite_name)
    population_stats = load_population_stats_sidecar(namespace)
    column_descriptions = load_column_descriptions_sidecar(namespace)

    eligibility_rule_prose = None
    if eligibility_predicate:
        eligibility_rule_prose = compile_predicate_prose(
            eligibility_predicate,
            feature_meta=feature_meta,
            population_stats=population_stats,
            column_descriptions=column_descriptions,
        )

    return build_enriched_context(
        raw_context,
        feature_meta=feature_meta,
        population_stats=population_stats,
        column_descriptions=column_descriptions,
        raw_candidate_playbooks=raw_candidate_playbooks,
        sibling_contrast=sibling_contrast,
        representative_examples=representative_examples,
        eligibility_rule_prose=eligibility_rule_prose,
        driver_stability=driver_stability,
        population_mean_churn=population_mean_churn,
        total_book_size=total_book_size,
        arr_exposure=arr_exposure,
    )


__all__ = ["enrich_archetype_from_namespace"]
