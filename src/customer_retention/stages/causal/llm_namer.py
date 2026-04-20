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
    pass


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

    def name_archetype(self, context: ArchetypeContext) -> ArchetypeNaming: ...


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

    def name_archetype(self, context: ArchetypeContext) -> ArchetypeNaming:
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
        description = self._template_description(context, top_positive, top_negative)

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

    Uses the OpenAI-compatible client pointed at the workspace's serving
    endpoints URL — no API key management because Databricks authenticates
    via the workspace credentials passed in (or inherited from environment
    variables ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN``).

    On any failure (network, auth, malformed JSON), the call falls back to
    ``ProseOverlapMatcher`` so the derivation pipeline never blocks.
    """

    def __init__(
        self,
        endpoint_name: str,
        workspace_url: Optional[str] = None,
        workspace_token: Optional[str] = None,
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

    def name_archetype(self, context: ArchetypeContext) -> ArchetypeNaming:
        client = self._get_client()
        if client is None:
            return self._fallback_with_log("OpenAI client unavailable", context)
        try:
            response = client.chat.completions.create(
                model=self.endpoint_name,
                messages=[{"role": "user", "content": self._build_prompt(context)}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content if response and response.choices else ""
            parsed = self._parse_response(content)
        except Exception as exc:  # noqa: BLE001 — fallback for any LLM error
            return self._fallback_with_log(f"LLM call failed: {exc}", context)
        if parsed is None:
            return self._fallback_with_log("LLM returned unparseable JSON", context)
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
        if not self.workspace_url:
            return None
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError:
            return None
        base_url = self.workspace_url.rstrip("/") + "/serving-endpoints"
        self._client = OpenAI(base_url=base_url, api_key=self.workspace_token or "dummy")
        return self._client

    def _fallback_with_log(self, reason: str, context: ArchetypeContext) -> ArchetypeNaming:
        logger.warning(
            "DatabricksFoundationModelNamer fallback (%s); using ProseOverlapMatcher", reason
        )
        result = self._fallback.name_archetype(context)
        result.llm_model_id = self.endpoint_name + FALLBACK_SUFFIX
        return result

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
        candidate_block = "\n".join(
            f"  - {c['playbook_id']} v{c.get('playbook_version', '1.0.0')}: "
            f"{c.get('description', '')[:200]} (overlap {c.get('overlap_score', 0.0):.2f})"
            for c in context.candidate_playbooks
        ) or "  (none — return at least one suggestion)"
        return (
            "You are mapping a customer churn archetype to applicable retention playbooks.\n\n"
            f"Archetype #{context.cluster_index}:\n"
            f"- Cluster size: {context.cluster_size}\n"
            f"- Cluster mean churn probability: {context.cluster_mean_churn_probability:.3f}\n"
            f"- Top positive drivers (raise risk):\n{positive_block}\n"
            f"- Top negative drivers (lower risk):\n{negative_block}\n\n"
            f"Candidate playbooks (deterministic feature-overlap pre-filter):\n{candidate_block}\n\n"
            "Your task:\n"
            "1. Refine the candidate set: keep, drop, or add playbooks based on semantic fit.\n"
            "2. For each kept/added playbook, give a fit_score 0.0-1.0 and a one-sentence rationale.\n"
            "3. Propose a 2-4 word archetype_name and a 2-sentence archetype_description.\n"
            "4. Return JSON only:\n"
            '{"archetype_name": "...", "archetype_description": "...", '
            '"playbooks": [{"playbook_id": "...", "fit_score": 0.0, "rationale": "..."}], '
            '"confidence": 0.0}'
        )

    @staticmethod
    def _parse_response(content: str) -> Optional[Dict[str, Any]]:
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
