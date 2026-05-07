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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .llm_namer import (
    ArchetypeContext,
    ArchetypeNaming,
    LLMNamer,
    PlaybookFitDecision,
    ProseOverlapMatcher,
    is_llm_fallback_id,
)

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.stages.causal.interpretation.archetype_context import (
        EnrichedArchetypeContext,
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


DEFAULT_TOP_N_DRIVERS: int = 10
PROSE_MATCH_MIN_SCORE: float = 0.1

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_STOPWORDS: frozenset = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
        "this", "to", "was", "with", "who", "which", "their", "they", "these",
        "those", "than", "then", "not", "but", "can", "customer", "customers",
        "account", "accounts", "playbook", "playbooks", "active", "goal", "data",
        "intended", "population", "target", "trial", "sense", "case", "cases",
        "run", "runs", "applicable", "use", "used", "when", "where", "what",
        "one", "two", "three",
    }
)


def map_archetypes_to_playbooks(
    archetypes: Sequence[ArchetypeSummary],
    playbooks: Sequence[Dict[str, Any]],
    gold_feature_names: Sequence[str],  # retained for signature stability; unused
    llm_namer: Optional[LLMNamer] = None,
    enrichment_builder: Optional[
        Callable[[ArchetypeSummary, ArchetypeContext], Optional["EnrichedArchetypeContext"]]
    ] = None,
) -> List[ArchetypeMapping]:
    """Match archetypes to playbooks by prose (never by feature columns).

    Every playbook is passed to the namer as a candidate for every
    archetype — the matcher decides fit by comparing the playbook's
    ``description`` + ``when_applicable`` + ``name`` against the
    archetype's top SHAP driver names + description. The feature-column
    schema is not used anywhere in this function; that would couple
    business-owned playbook prose to the model's feature layout and
    break the quarterly-vs-per-retrain cadence split.

    ``gold_feature_names`` is retained in the signature for caller
    stability but is no longer consumed. It will be removed once
    downstream callers drop it.

    Two namer implementations drive the scoring:

    - ``ProseOverlapMatcher`` (deterministic, always available) scores each
      playbook by lexical overlap between the playbook's prose tokens
      and the archetype's driver feature names + archetype prose.
      Always non-empty — every (archetype, playbook) pair produces a
      ``PlaybookFitDecision`` with a transparent rationale.

    - ``DatabricksFoundationModelNamer`` (LLM-refined) calls the
      configured Foundation Model endpoint and returns the LLM's
      per-playbook fit scores + rationale. When the LLM is unavailable,
      it falls back to ``ProseOverlapMatcher`` — the result is still
      non-empty.
    """
    del gold_feature_names  # intentionally unused; see docstring
    namer = llm_namer or ProseOverlapMatcher()
    mappings: List[ArchetypeMapping] = []
    for archetype in archetypes:
        candidates = _all_playbooks_as_candidates(playbooks)
        for cand in candidates:
            score, _ = prose_overlap_score(cand, archetype)
            cand["overlap_score"] = float(score)
        context = _build_context(archetype, candidates)
        enriched = _safely_enrich(enrichment_builder, archetype, context)
        naming = namer.name_archetype(context, enriched=enriched) if enriched is not None \
            else namer.name_archetype(context)
        mappings.append(_assemble_mapping(archetype, candidates, naming))
    return mappings


def _safely_enrich(
    builder: Optional[Callable],
    archetype: ArchetypeSummary,
    context: ArchetypeContext,
) -> Optional["EnrichedArchetypeContext"]:
    """Call ``builder`` defensively — enrichment failure must not block mapping."""
    if builder is None:
        return None
    try:
        return builder(archetype, context)
    except Exception as exc:  # noqa: BLE001 — best-effort; fall through to raw path
        logger.warning("enrichment_builder failed; continuing without enrichment: %s", exc)
        return None


def prose_overlap_score(
    playbook: Dict[str, Any], archetype: ArchetypeSummary
) -> Tuple[float, List[str]]:
    """Score a (playbook, archetype) prose match. Returns (score, matched_tokens).

    Matching surfaces (both sides):

    - Playbook tokens: ``description`` + ``when_applicable`` + ``name`` +
      ``playbook_id`` (tokenized on word boundaries, lowercased,
      stopwords removed).
    - Archetype tokens: driver feature names (e.g. ``nps_score`` →
      ``nps``, ``score``) from top positive + top negative SHAP drivers.

    Score is ``|matched| / |archetype_tokens|`` — the fraction of
    archetype driver concepts this playbook's prose references. Range
    [0, 1]. A zero score means the playbook mentions none of the
    archetype's driving concepts; this is surfaced (not filtered) so
    humans see the full grid at review time.
    """
    pb_tokens = _playbook_prose_tokens(playbook)
    arch_tokens = _archetype_driver_tokens(archetype)
    if not arch_tokens:
        return 0.0, []
    matched = sorted(t for t in arch_tokens if t in pb_tokens)
    score = len(matched) / len(arch_tokens)
    return float(score), matched


def extract_features_from_text(
    text: str, gold_feature_names: Iterable[str]
) -> List[str]:
    """Extract gold feature column names from a free-text description.

    Retained for backwards compatibility with existing tests. No longer
    used by the mapper — the mapper matches prose tokens directly (see
    ``prose_overlap_score``) rather than against the gold feature
    schema.
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
# Internals — prose tokenization + candidate assembly
# ---------------------------------------------------------------------------


def _all_playbooks_as_candidates(
    playbooks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return every playbook annotated with the minimum prose needed for matching.

    The namer (ProseOverlapMatcher or LLM) receives every playbook — no
    pre-filter. The namer produces one fit decision per candidate, so
    every (archetype, playbook) pair becomes a policy row at write
    time, with fit_score + rationale visible for human review.
    """
    out: List[Dict[str, Any]] = []
    for playbook in playbooks:
        out.append(
            {
                "playbook_id": str(playbook.get("playbook_id", "")),
                "playbook_version": str(playbook.get("version", "1.0.0")),
                "name": str(playbook.get("name") or ""),
                "description": str(playbook.get("description") or ""),
                "when_applicable": str(playbook.get("when_applicable") or ""),
            }
        )
    return out


def _playbook_prose_tokens(playbook: Dict[str, Any]) -> set[str]:
    """Tokenize a playbook's authoring prose. Lowercased, stopwords removed."""
    parts = [
        str(playbook.get("description") or ""),
        str(playbook.get("when_applicable") or ""),
        str(playbook.get("name") or ""),
        str(playbook.get("playbook_id") or ""),
    ]
    return _tokenize(" ".join(parts))


def _archetype_driver_tokens(archetype: ArchetypeSummary) -> set[str]:
    """Extract business-meaningful tokens from the archetype's top SHAP drivers.

    Driver feature names like ``nps_score``, ``tenure_days``, and
    ``missed_payment_count_90d`` carry business semantics. We split on
    underscore/digit boundaries and drop stopwords, yielding tokens like
    ``nps``, ``tenure``, ``missed``, ``payment`` that business prose is
    likely to use.
    """
    out: set[str] = set()
    for driver in archetype.top_positive_drivers[:DEFAULT_TOP_N_DRIVERS]:
        out.update(_tokenize_feature_name(str(driver.get("feature", ""))))
    for driver in archetype.top_negative_drivers[:DEFAULT_TOP_N_DRIVERS]:
        out.update(_tokenize_feature_name(str(driver.get("feature", ""))))
    return out


def _tokenize_feature_name(feature_name: str) -> set[str]:
    """Split a feature column name into business-meaningful word tokens.

    Examples:
    - ``nps_score`` → {"nps", "score"}
    - ``missed_payment_count_90d`` → {"missed", "payment", "count"}
    - ``emails_opens_30d`` → {"emails", "opens"}
    - ``tenureDaysV2`` → {"tenure", "days"}
    """
    if not feature_name:
        return set()
    snake = re.sub(r"([a-z])([A-Z])", r"\1_\2", feature_name)
    return _tokenize(snake.replace("_", " "))


def _tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for match in _TOKEN_RE.findall(text.lower()):
        if len(match) <= 2:
            continue
        if match.isdigit():
            continue
        if match in _STOPWORDS:
            continue
        out.add(match)
    return out


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
    decisions = list(naming.playbooks)
    rationale = "; ".join(
        f"{d.playbook_id}: {d.rationale}" for d in decisions[:3]
    )
    derivation_method = (
        "prose_overlap"
        if is_llm_fallback_id(naming.llm_model_id)
        else "prose_overlap+llm"
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
