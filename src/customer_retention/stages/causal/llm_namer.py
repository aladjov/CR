"""Archetype → playbook matching: deterministic baseline and LLM refinement.

The mapper (``playbook_mapper.py``) calls into an ``LLMNamer`` instance
for each archetype. The instance is responsible for two outputs in one
shot: (a) the archetype's human-readable name and description, and
(b) a per-playbook fit decision with a rationale. Matching happens in
prose space — the playbook's ``description`` / ``when_applicable``
text against the archetype's top SHAP driver names (tokenized) and
archetype description. Feature column layout is never consulted; that
would couple business-owned playbook prose to the model's feature
schema and break the cadence split.

Two implementations ship out of the box:

- **``ProseOverlapMatcher``** — deterministic, no network calls. Scores
  each playbook by the fraction of archetype driver tokens that appear
  in the playbook's prose. Always available and always non-empty: every
  (archetype, playbook) pair yields a fit decision so every match shows
  up in the review queue with a transparent rationale.
- **``DatabricksFoundationModelNamer``** — calls a Databricks-hosted
  Mosaic AI Foundation Model endpoint via the OpenAI-compatible client.
  Default endpoint is ``databricks-claude-sonnet-4-6``. On any failure
  (network, auth, JSON parse, missing ``openai`` client) it falls back
  to ``ProseOverlapMatcher`` so the derivation pipeline never blocks,
  with ``llm_model_id`` suffixed by ``:fallback`` for audit.

Factory: ``build_llm_namer(endpoint_name)``. Empty string ⇒
``ProseOverlapMatcher``; otherwise the Databricks endpoint client.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.stages.causal.interpretation.archetype_context import (
        EnrichedArchetypeContext,
    )


FALLBACK_SUFFIX: str = ":fallback"


def is_llm_fallback_id(model_id: Optional[str]) -> bool:
    """Return True when ``model_id`` came from a fallback path.

    Used by ``playbook_mapper`` so it can correctly tag a mapping as
    deterministic-only when the LLM call failed and
    ``ProseOverlapMatcher`` produced the result. The
    ``DatabricksFoundationModelNamer`` appends ``FALLBACK_SUFFIX`` to
    its endpoint name on every fallback path.
    """
    if not model_id:
        return False
    return model_id == ProseOverlapMatcher.model_id or model_id.endswith(FALLBACK_SUFFIX)


# ---------------------------------------------------------------------------
# Public dataclasses passed between mapper and namer
# ---------------------------------------------------------------------------


@dataclass
class ArchetypeContext:
    """Per-cluster context the mapper hands to the namer.

    The mapper builds one of these per cluster from the SHAP-space centroid
    plus the per-feature mean values, and passes a list of all archetypes
    plus all candidate playbooks to ``refine_mapping`` in a single call so
    the namer can produce stable names across the full set.
    """

    cluster_index: int
    cluster_size: int
    cluster_mean_churn_probability: float
    top_positive_drivers: List[Dict[str, float]]
    top_negative_drivers: List[Dict[str, float]]
    candidate_playbooks: List[Dict[str, Any]]


@dataclass
class PlaybookFitDecision:
    """One playbook's refined fit for one archetype."""

    playbook_id: str
    fit_score: float
    rationale: str


@dataclass
class ArchetypeNaming:
    """The namer's output for one archetype."""

    archetype_name: str
    archetype_description: str
    playbooks: List[PlaybookFitDecision] = field(default_factory=list)
    confidence: float = 0.0
    llm_model_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMNamer(Protocol):
    """Stable contract for archetype naming + mapping refinement.

    Implementations either call an LLM endpoint or fall back to deterministic
    template strings. ``model_id`` exposes the underlying endpoint name for
    audit (written to ``archetype_catalog.llm_model_id``).
    """

    @property
    def model_id(self) -> str: ...  # pragma: no cover

    def name_archetype(
        self,
        context: ArchetypeContext,
        enriched: Optional["EnrichedArchetypeContext"] = None,
    ) -> ArchetypeNaming: ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_llm_namer(
    endpoint_name: Optional[str],
    workspace_url: Optional[str] = None,
    workspace_token: Optional[str] = None,
) -> LLMNamer:
    """Return the appropriate ``LLMNamer`` for ``endpoint_name``.

    Empty / ``None`` selects ``ProseOverlapMatcher``. Otherwise constructs a
    ``DatabricksFoundationModelNamer``. The Databricks namer reads
    workspace credentials from the env (``DATABRICKS_HOST``, ``DATABRICKS_TOKEN``)
    when explicit values are not provided.
    """
    if not endpoint_name:
        return ProseOverlapMatcher()
    return DatabricksFoundationModelNamer(
        endpoint_name=endpoint_name,
        workspace_url=workspace_url,
        workspace_token=workspace_token,
    )


# ---------------------------------------------------------------------------
# Template (deterministic, always available)
# ---------------------------------------------------------------------------


class ProseOverlapMatcher:
    """Deterministic prose-overlap namer and always-available baseline.

    Role: when no LLM endpoint is configured, or when the Databricks
    Foundation Model endpoint fails, this namer produces the matching
    result. It also runs as a transparency signal alongside the LLM so
    reviewers can compare deterministic overlap against LLM judgment.

    What it matches on:

    - Archetype side — driver feature names from ``top_positive_drivers``
      and ``top_negative_drivers``, split on underscore/case boundaries
      into business-meaningful tokens (``nps_score`` → {``nps``,
      ``score``}; ``missed_payment_count_90d`` → {``missed``, ``payment``,
      ``count``}).
    - Playbook side — authoring prose from ``description`` +
      ``when_applicable`` + ``name`` + ``playbook_id``, tokenized with
      stopwords removed.

    Feature column layout is never consulted. Matching happens entirely
    in prose/token space so playbook YAMLs stay coupled to the business
    vocabulary, not the feature schema.

    Every candidate playbook produces exactly one ``PlaybookFitDecision``
    so every (archetype, playbook) pair appears in the review queue with
    a transparent rationale — never silently dropped.
    """

    model_id: str = "prose_overlap"

    def name_archetype(
        self,
        context: ArchetypeContext,
        enriched: Optional["EnrichedArchetypeContext"] = None,
    ) -> ArchetypeNaming:
        from .playbook_mapper import ArchetypeSummary, prose_overlap_score

        archetype = ArchetypeSummary(
            cluster_index=context.cluster_index,
            cluster_size=context.cluster_size,
            cluster_mean_churn_probability=context.cluster_mean_churn_probability,
            top_positive_drivers=list(context.top_positive_drivers),
            top_negative_drivers=list(context.top_negative_drivers),
        )

        top_positive = context.top_positive_drivers[0] if context.top_positive_drivers else None
        top_negative = context.top_negative_drivers[0] if context.top_negative_drivers else None
        label = self._template_label(context, top_positive, top_negative)
        description = (
            self._template_description_enriched(context, enriched)
            if enriched is not None
            else self._template_description(context, top_positive, top_negative)
        )

        decisions: List[PlaybookFitDecision] = []
        for cand in context.candidate_playbooks:
            score, matched_tokens = prose_overlap_score(cand, archetype)
            rationale = self._template_rationale(cand, score, matched_tokens, top_positive)
            decisions.append(
                PlaybookFitDecision(
                    playbook_id=str(cand.get("playbook_id", "")),
                    fit_score=float(score),
                    rationale=rationale,
                )
            )

        decisions.sort(key=lambda d: (-d.fit_score, d.playbook_id))
        confidence = max((d.fit_score for d in decisions), default=0.0)
        return ArchetypeNaming(
            archetype_name=label,
            archetype_description=description,
            playbooks=decisions,
            confidence=confidence,
            llm_model_id=self.model_id,
        )

    @staticmethod
    def _template_label(
        context: ArchetypeContext,
        top_positive: Optional[Dict[str, Any]],
        top_negative: Optional[Dict[str, Any]],
    ) -> str:
        if top_positive:
            return f"Archetype {context.cluster_index}: high {top_positive['feature']}"
        if top_negative:
            return f"Archetype {context.cluster_index}: low {top_negative['feature']}"
        return f"Archetype {context.cluster_index}"

    @staticmethod
    def _template_description(
        context: ArchetypeContext,
        top_positive: Optional[Dict[str, Any]],
        top_negative: Optional[Dict[str, Any]],
    ) -> str:
        parts = [
            f"Cluster of {context.cluster_size} customers with mean churn "
            f"probability {context.cluster_mean_churn_probability:.3f}."
        ]
        if top_positive:
            parts.append(f"Top risk driver: {top_positive['feature']}.")
        if top_negative:
            parts.append(f"Top protective driver: {top_negative['feature']}.")
        return " ".join(parts)

    @staticmethod
    def _template_description_enriched(
        context: ArchetypeContext,
        enriched: "EnrichedArchetypeContext",
    ) -> str:
        """Phase 3/4 fallback — use enriched ``business_phrase``/``value_phrase``
        so the deterministic path reads the same as the LLM path."""
        parts = [
            f"Cluster of {context.cluster_size} customers with mean churn "
            f"probability {context.cluster_mean_churn_probability:.3f}."
        ]
        if enriched.top_positive_drivers:
            driver = enriched.top_positive_drivers[0]
            parts.append(
                f"Top risk driver: {driver.business_phrase} "
                f"({driver.value_phrase})."
            )
        if enriched.top_negative_drivers:
            driver = enriched.top_negative_drivers[0]
            parts.append(
                f"Top protective driver: {driver.business_phrase} "
                f"({driver.value_phrase})."
            )
        if enriched.eligibility_rule_prose:
            parts.append(f"Eligible when: {enriched.eligibility_rule_prose}.")
        return " ".join(parts)

    @staticmethod
    def _template_rationale(
        candidate: Dict[str, Any],
        score: float,
        matched_tokens: List[str],
        top_positive: Optional[Dict[str, Any]],
    ) -> str:
        if matched_tokens:
            tokens_str = ", ".join(matched_tokens)
            return (
                f"prose-overlap score {score:.2f}; "
                f"archetype driver tokens matched in playbook prose: [{tokens_str}]"
            )
        if top_positive:
            return (
                f"prose-overlap score {score:.2f}; "
                f"no archetype driver tokens (e.g. from '{top_positive['feature']}') "
                "appear in this playbook's description or when_applicable prose"
            )
        return f"prose-overlap score {score:.2f}; archetype has no driver tokens to match"


# ---------------------------------------------------------------------------
# Databricks Foundation Model
# ---------------------------------------------------------------------------


class DatabricksFoundationModelNamer:
    """Calls a Databricks Mosaic AI Foundation Model serving endpoint.

    Uses ``mlflow.deployments.get_deploy_client("databricks")`` — the same
    route that ``column_describer`` and the notebook-0.15 auto-describer
    use. The deployments client picks up ambient cluster auth, so no
    ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` env vars are required on
    serverless / shared clusters.

    On any failure (endpoint unreachable, malformed JSON, missing SDK),
    the call falls back to ``ProseOverlapMatcher`` so the derivation
    pipeline never blocks.
    """

    def __init__(
        self,
        endpoint_name: str,
        workspace_url: Optional[str] = None,  # accepted for back-compat; unused
        workspace_token: Optional[str] = None,  # accepted for back-compat; unused
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.workspace_url = workspace_url or os.environ.get("DATABRICKS_HOST", "")
        self.workspace_token = workspace_token or os.environ.get("DATABRICKS_TOKEN", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._fallback = ProseOverlapMatcher()
        self._client: Any = None

    @property
    def model_id(self) -> str:
        return self.endpoint_name

    def name_archetype(
        self,
        context: ArchetypeContext,
        enriched: Optional["EnrichedArchetypeContext"] = None,
    ) -> ArchetypeNaming:
        client = self._get_client()
        if client is None:
            return self._fallback_with_log("deployments client unavailable", context, enriched)
        try:
            messages = self._build_messages(context, enriched)
            response = client.predict(
                endpoint=self.endpoint_name,
                inputs={
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            content = _extract_message_content(response)
            parsed = self._parse_response(content)
        except Exception as exc:  # noqa: BLE001 — fallback for any LLM error
            return self._fallback_with_log(f"LLM call failed: {exc}", context, enriched)
        if parsed is None:
            return self._fallback_with_log("LLM returned unparseable JSON", context, enriched)
        decisions = [
            PlaybookFitDecision(
                playbook_id=str(item.get("playbook_id")),
                fit_score=float(item.get("fit_score", 0.0)),
                rationale=str(item.get("rationale", "")),
            )
            for item in parsed.get("playbooks", [])
            if item.get("playbook_id")
        ]
        return ArchetypeNaming(
            archetype_name=str(parsed.get("archetype_name", f"Archetype {context.cluster_index}")),
            archetype_description=str(parsed.get("archetype_description", "")),
            playbooks=decisions,
            confidence=float(parsed.get("confidence", 0.0)),
            llm_model_id=self.endpoint_name,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import mlflow.deployments  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            self._client = mlflow.deployments.get_deploy_client("databricks")
        except Exception:  # noqa: BLE001 — no ambient auth / off-cluster
            return None
        return self._client

    def _fallback_with_log(
        self,
        reason: str,
        context: ArchetypeContext,
        enriched: Optional["EnrichedArchetypeContext"] = None,
    ) -> ArchetypeNaming:
        logger.warning(
            "DatabricksFoundationModelNamer fallback (%s); using ProseOverlapMatcher", reason
        )
        result = self._fallback.name_archetype(context, enriched=enriched)
        result.llm_model_id = self.endpoint_name + FALLBACK_SUFFIX
        return result

    def _build_messages(
        self,
        context: ArchetypeContext,
        enriched: Optional["EnrichedArchetypeContext"],
    ) -> List[Dict[str, str]]:
        if enriched is not None:
            from customer_retention.stages.causal.interpretation.llm_prompt import (
                build_enriched_prompt_messages,
            )
            return build_enriched_prompt_messages(enriched)
        return [{"role": "user", "content": self._build_prompt(context)}]

    def _build_prompt(self, context: ArchetypeContext) -> str:
        positive_block = "\n".join(
            f"  - {d['feature']}: mean SHAP {d.get('mean_shap', 0.0):+.3f}, "
            f"mean value {d.get('mean_value', 0.0):.2f}"
            for d in context.top_positive_drivers
        ) or "  (none)"
        negative_block = "\n".join(
            f"  - {d['feature']}: mean SHAP {d.get('mean_shap', 0.0):+.3f}, "
            f"mean value {d.get('mean_value', 0.0):.2f}"
            for d in context.top_negative_drivers
        ) or "  (none)"
        # Inject FULL description + when_applicable (no truncation) and the
        # populated overlap_score so the LLM can use it as a sanity floor.
        # The candidate-assembly path in playbook_mapper.map_archetypes_to_playbooks
        # writes overlap_score onto each candidate before this prompt is built.
        candidate_block = "\n".join(
            f"  - {c['playbook_id']} v{c.get('playbook_version', '1.0.0')}: {c.get('name', '')}\n"
            f"      description: {c.get('description', '')}\n"
            f"      when_applicable: {c.get('when_applicable', '')}\n"
            f"      deterministic_overlap_with_archetype: {float(c.get('overlap_score', 0.0)):.2f}"
            for c in context.candidate_playbooks
        ) or "  (none)"
        n_playbooks = len(context.candidate_playbooks)
        return (
            "You are mapping a customer churn archetype onto a fixed catalog of retention "
            "playbooks. Your job is RANKING — score every playbook's fit to this archetype's "
            "risk pattern. Do not drop, omit, or filter playbooks; downstream policy generation "
            "requires every candidate to receive a score.\n\n"

            "═══ ARCHETYPE PROFILE ═══\n"
            f"archetype_index: {context.cluster_index}\n"
            f"cluster_size: {context.cluster_size}\n"
            f"cluster_mean_churn_probability: {context.cluster_mean_churn_probability:.3f}  "
            "(INFORMATIONAL ONLY — do not let this drive fit_score; even healthy clusters need "
            "a best-fit retention play recommendation in case their risk profile shifts)\n"
            f"top positive drivers (raise risk):\n{positive_block}\n"
            f"top negative drivers (lower risk):\n{negative_block}\n\n"

            "═══ CANDIDATE PLAYBOOKS (score every one) ═══\n"
            f"{candidate_block}\n\n"

            "═══ SCORING RUBRIC ═══\n"
            "fit_score is a 0.0-1.0 measure of how well THIS PLAYBOOK ADDRESSES THIS ARCHETYPE'S "
            "RISK PATTERN. It is NOT a probability of churn, NOT a predicted uplift, NOT a "
            "recommendation strength. Use these brackets:\n\n"
            "  0.85-1.00  Perfect fit: the playbook is explicitly designed for this exact risk\n"
            "             pattern (e.g., archetype dominated by 'low recent engagement' + a\n"
            "             playbook named 'Re-Engagement Win-Back').\n"
            "  0.50-0.84  Strong fit: playbook addresses 2+ of the archetype's top drivers, even\n"
            "             if its primary use-case is a slightly different segment.\n"
            "  0.20-0.49  Partial fit: playbook addresses one of the archetype's drivers, or a\n"
            "             closely related concept.\n"
            "  0.05-0.19  Weak fit: playbook is in the retention space but addresses a different\n"
            "             pattern. Always preferred over dropping a playbook.\n"
            "  0.00-0.04  No fit: playbook is unrelated even at the vocabulary level.\n\n"

            "═══ ANCHORING RULES ═══\n"
            "1. The 'deterministic_overlap_with_archetype' value is a token-level vocabulary\n"
            "   match between the playbook's prose and the archetype's driver feature names.\n"
            "   Use it as a FLOOR signal: if it is 0.40+, your fit_score should rarely be below\n"
            "   0.20. Disagreement is allowed but must be justified in the rationale.\n"
            f"2. Always return ALL {n_playbooks} input playbooks in the 'playbooks' array, sorted\n"
            "   by fit_score descending. Omitting a playbook is a hard error.\n"
            "3. cluster_mean_churn_probability is provided for context but is NOT a scoring input.\n"
            "   A cluster with low predicted churn still gets fit_score reflecting which playbook\n"
            "   would apply IF its members showed risk.\n"
            "4. Return JSON only — no commentary, no markdown fences.\n\n"

            "═══ OUTPUT SCHEMA (strict) ═══\n"
            '{\n'
            '  "archetype_name": "<2-4 words capturing the dominant risk pattern>",\n'
            '  "archetype_description": "<exactly 2 sentences>",\n'
            '  "playbooks": [\n'
            '    {"playbook_id": "...", "fit_score": 0.0, "rationale": "<one sentence per the rubric>"}\n'
            f'    /* exactly {n_playbooks} entries, sorted by fit_score desc */\n'
            '  ],\n'
            '  "confidence": 0.0\n'
            '}'
        )

    @staticmethod
    def _parse_response(content: str) -> Optional[Dict[str, Any]]:  # noqa: D401
        return _parse_json_response(content)


def _extract_message_content(response: Any) -> str:
    """Pull the assistant text from an `mlflow.deployments.DatabricksDeploymentClient.predict` response.

    The serving-endpoints API returns an OpenAI-style dict:
    ``{"choices": [{"message": {"role": "assistant", "content": "..."}, ...}], ...}``.
    The deploy client wraps it in a mapping-like object that supports both
    dict access and attribute access depending on SDK version.
    """
    if response is None:
        return ""
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    if message is None:
        return ""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    return content or ""


def _parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
