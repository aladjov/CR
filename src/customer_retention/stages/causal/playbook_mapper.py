"""Map model-derived archetypes to retention playbooks.

Implements **Mapping B** from the implementation plan
(``docs/causal_track_implementation_plan.md`` §"SHAP → Playbook mapping"):

1. **Step 1 — feature-overlap baseline (always runs).** For each playbook in
   ``playbook_catalog``, extract feature names referenced in the
   description (regex over the gold column list). Score each archetype
   against each playbook by the fraction of the playbook's features that
   appear among the archetype's top SHAP drivers. Match if the score is
   ≥ ``MIN_OVERLAP_SCORE`` *or* the archetype's #1 positive driver is in
   the playbook's referenced feature set.

2. **Step 2 — LLM refinement (optional).** For each archetype, call the
   ``LLMNamer`` to refine the candidate set, name the archetype, and write
   per-playbook rationale strings. Empty/template namer ⇒ deterministic
   pass-through.

The mapping is many-to-many: one archetype can target multiple playbooks
and one playbook can be targeted by multiple archetypes. The output is a
list of ``ArchetypeMapping`` objects, one per archetype, that the
derivation orchestrator turns into ``archetype_catalog`` and
``eligibility_policy`` rows.

The module is pure Python — no Spark imports — so it can run on the driver
once the SHAP cluster summaries (small, one row per cluster) have been
collected from the distributed pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .llm_namer import (
    ArchetypeContext,
    ArchetypeNaming,
    LLMNamer,
    PlaybookFitDecision,
    TemplateNamer,
    is_llm_fallback_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArchetypeSummary:
    """Compact per-cluster summary the mapper consumes.

    Built by ``derivation.py`` from the SHAP-clustering output once each
    cluster's mean SHAP and mean feature values have been collected from
    the distributed pipeline. Holds only the small payload needed for
    mapping — no per-row data.
    """

    cluster_index: int
    cluster_size: int
    cluster_mean_churn_probability: float
    top_positive_drivers: List[Dict[str, float]] = field(default_factory=list)
    top_negative_drivers: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ArchetypeMapping:
    """One archetype's full mapping result, ready for catalog/policy writes.

    Holds both the deterministic baseline candidates and the (optionally
    LLM-refined) final fit decisions, so the orchestrator can write the
    chosen playbooks to ``eligibility_policy`` and surface the rationale
    strings for human review.
    """

    cluster_index: int
    archetype_name: str
    archetype_description: str
    rationale: str
    confidence: float
    candidate_playbook_ids: List[str]
    fit_decisions: List[PlaybookFitDecision]
    llm_model_id: Optional[str] = None
    derivation_method: str = "feature_overlap"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


MIN_OVERLAP_SCORE: float = 0.3
DEFAULT_TOP_N_DRIVERS: int = 10


def map_archetypes_to_playbooks(
    archetypes: Sequence[ArchetypeSummary],
    playbooks: Sequence[Dict[str, Any]],
    gold_feature_names: Sequence[str],
    llm_namer: Optional[LLMNamer] = None,
    min_overlap_score: float = MIN_OVERLAP_SCORE,
) -> List[ArchetypeMapping]:
    """Run the hybrid feature-overlap + LLM refinement mapping.

    ``playbooks`` is a list of ``playbook_catalog`` row dicts (the loader's
    output). ``gold_feature_names`` is the list of column names in the gold
    feature table — the regex matches feature tokens against this set so
    arbitrary words in playbook descriptions don't get mistaken for column
    names.

    When ``llm_namer`` is None, ``TemplateNamer`` is used so the result is
    always deterministic and self-contained.
    """
    namer = llm_namer or TemplateNamer()
    playbook_features = _extract_playbook_features(playbooks, gold_feature_names)
    mappings: List[ArchetypeMapping] = []
    for archetype in archetypes:
        candidates = _candidate_set(archetype, playbooks, playbook_features, min_overlap_score)
        context = _build_context(archetype, candidates)
        naming = namer.name_archetype(context)
        mappings.append(_assemble_mapping(archetype, candidates, naming))
    return mappings


def extract_features_from_text(
    text: str, gold_feature_names: Iterable[str]
) -> List[str]:
    """Public helper: extract gold feature column names from a free-text playbook description.

    Used directly by ``_extract_playbook_features`` and exposed for unit
    tests so the regex behavior can be exercised in isolation.
    """
    if not text:
        return []
    feature_set = set(gold_feature_names)
    if not feature_set:
        return []
    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    matches = pattern.findall(text)
    seen: set[str] = set()
    out: List[str] = []
    for token in matches:
        if token in feature_set and token not in seen:
            seen.add(token)
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# Internals — feature overlap baseline
# ---------------------------------------------------------------------------


def _extract_playbook_features(
    playbooks: Sequence[Dict[str, Any]], gold_feature_names: Sequence[str]
) -> Dict[str, List[str]]:
    feature_set = list(gold_feature_names)
    out: Dict[str, List[str]] = {}
    for playbook in playbooks:
        text_parts = [
            str(playbook.get("description") or ""),
            str(playbook.get("name") or ""),
            str(playbook.get("analysis_population_rule") or ""),
        ]
        target_features = playbook.get("target_features")
        if isinstance(target_features, list):
            text_parts.extend(str(f) for f in target_features)
        out[str(playbook["playbook_id"])] = extract_features_from_text(
            " ".join(text_parts), feature_set
        )
    return out


def _candidate_set(
    archetype: ArchetypeSummary,
    playbooks: Sequence[Dict[str, Any]],
    playbook_features: Dict[str, List[str]],
    min_overlap_score: float,
) -> List[Dict[str, Any]]:
    top_features = _archetype_feature_set(archetype)
    primary_driver = (
        archetype.top_positive_drivers[0]["feature"]
        if archetype.top_positive_drivers
        else None
    )
    candidates: List[Dict[str, Any]] = []
    for playbook in playbooks:
        playbook_id = str(playbook["playbook_id"])
        ref_features = playbook_features.get(playbook_id, [])
        score = _overlap_score(ref_features, top_features)
        primary_match = bool(primary_driver and primary_driver in ref_features)
        if score >= min_overlap_score or primary_match:
            candidates.append(
                {
                    "playbook_id": playbook_id,
                    "playbook_version": str(playbook.get("version", "1.0.0")),
                    "description": str(playbook.get("description") or ""),
                    "overlap_score": float(score),
                    "primary_driver_match": primary_match,
                    "matched_features": [f for f in ref_features if f in top_features],
                }
            )
    candidates.sort(key=lambda c: (-c["overlap_score"], c["playbook_id"]))
    return candidates


def _archetype_feature_set(archetype: ArchetypeSummary) -> set[str]:
    features: set[str] = set()
    for driver in archetype.top_positive_drivers[:DEFAULT_TOP_N_DRIVERS]:
        features.add(str(driver["feature"]))
    for driver in archetype.top_negative_drivers[:DEFAULT_TOP_N_DRIVERS]:
        features.add(str(driver["feature"]))
    return features


def _overlap_score(playbook_features: Sequence[str], archetype_features: set[str]) -> float:
    if not playbook_features:
        return 0.0
    matched = sum(1 for f in playbook_features if f in archetype_features)
    return matched / len(playbook_features)


# ---------------------------------------------------------------------------
# Internals — LLM context + assembly
# ---------------------------------------------------------------------------


def _build_context(archetype: ArchetypeSummary, candidates: List[Dict[str, Any]]) -> ArchetypeContext:
    return ArchetypeContext(
        cluster_index=archetype.cluster_index,
        cluster_size=archetype.cluster_size,
        cluster_mean_churn_probability=archetype.cluster_mean_churn_probability,
        top_positive_drivers=list(archetype.top_positive_drivers),
        top_negative_drivers=list(archetype.top_negative_drivers),
        candidate_playbooks=candidates,
    )


def _assemble_mapping(
    archetype: ArchetypeSummary,
    candidates: List[Dict[str, Any]],
    naming: ArchetypeNaming,
) -> ArchetypeMapping:
    candidate_ids = [str(c["playbook_id"]) for c in candidates]
    decisions = naming.playbooks or [
        PlaybookFitDecision(
            playbook_id=cand_id,
            fit_score=float(cand["overlap_score"]),
            rationale="Selected by feature-overlap baseline",
        )
        for cand_id, cand in zip(candidate_ids, candidates)
    ]
    rationale = "; ".join(f"{d.playbook_id}: {d.rationale}" for d in decisions[:3])
    derivation_method = (
        "feature_overlap"
        if is_llm_fallback_id(naming.llm_model_id)
        else "feature_overlap+llm"
    )
    return ArchetypeMapping(
        cluster_index=archetype.cluster_index,
        archetype_name=naming.archetype_name,
        archetype_description=naming.archetype_description,
        rationale=rationale,
        confidence=naming.confidence,
        candidate_playbook_ids=candidate_ids,
        fit_decisions=decisions,
        llm_model_id=naming.llm_model_id,
        derivation_method=derivation_method,
    )
