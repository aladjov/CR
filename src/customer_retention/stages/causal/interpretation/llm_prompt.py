"""Phase 4 — enriched LLM prompt builder.

Consumes an ``EnrichedArchetypeContext`` (Phase 3) and emits an OpenAI-
compatible message list where the user payload is structured JSON: every
quantity in the prompt is already narrated (``business_phrase`` +
``value_phrase``, ``policy_summary``, ``expected_effect``, sibling contrast
phrases, eligibility rule prose).

The LLM's only job is narration: choose a 2-4 word name, write a 2-sentence
description that references the eligibility rule prose verbatim, emit a
one-sentence sibling contrast, and cite ≥1 driver business_phrase per
playbook rationale. The response schema is encoded both in the system
message and in a trailing ``response_format`` block so the model cannot
fabricate numeric quantities — none are left for it to compute.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.stages.causal.interpretation.archetype_context import (
        EnrichedArchetypeContext,
    )


_SYSTEM_MESSAGE = (
    "You are naming customer-churn archetypes and rating retention-playbook fit. "
    "Every numeric fact in the user payload is already narrated — you MUST NOT "
    "invent thresholds, percentages, or counts. Reference provided business "
    "phrases verbatim. Output JSON only, no prose commentary."
)

_RESPONSE_SCHEMA = {
    "archetype_name": "2-4 word label, Title Case",
    "archetype_description": (
        "2 sentences. The second sentence MUST quote the eligibility_rule_prose "
        "verbatim (or say 'no eligibility rule' when absent)."
    ),
    "contrast_with_sibling": (
        "1 sentence using the sibling_contrast phrases. Empty string when "
        "sibling_contrast is missing."
    ),
    "playbooks": [
        {
            "playbook_id": "string (one of the provided candidate ids)",
            "fit_score": "0.0 to 1.0",
            "rationale": (
                "1 sentence citing at least one top_positive_driver or "
                "top_negative_driver business_phrase verbatim."
            ),
        }
    ],
    "confidence": "0.0 to 1.0, your overall confidence in this mapping",
}


def build_enriched_prompt_messages(
    enriched: "EnrichedArchetypeContext",
    *,
    siblings: Optional[Sequence["EnrichedArchetypeContext"]] = None,
) -> List[Dict[str, str]]:
    """Return ``[system, user]`` messages for the Databricks Foundation endpoint."""
    payload = _build_user_payload(enriched, siblings)
    return [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": json.dumps(
                {"archetype": payload, "response_schema": _RESPONSE_SCHEMA},
                indent=2, sort_keys=True,
            ),
        },
    ]


def _build_user_payload(
    enriched: "EnrichedArchetypeContext",
    siblings: Optional[Sequence["EnrichedArchetypeContext"]],
) -> Dict[str, Any]:
    return {
        "cluster_index": enriched.cluster_index,
        "cluster_size": enriched.cluster_size,
        "cluster_mean_churn_probability": _round(enriched.cluster_mean_churn_probability, 4),
        "share_of_book": _round(enriched.share_of_book, 4),
        "lift_vs_population": _round(enriched.lift_vs_population, 3),
        "arr_exposure": _round(enriched.arr_exposure, 2),
        "top_positive_drivers": [_driver_payload(d) for d in enriched.top_positive_drivers],
        "top_negative_drivers": [_driver_payload(d) for d in enriched.top_negative_drivers],
        "sibling_contrast": [_contrast_payload(c) for c in enriched.sibling_contrast],
        "siblings": [_sibling_summary(s) for s in (siblings or [])],
        "representative_examples": list(enriched.representative_examples),
        "eligibility_rule_prose": enriched.eligibility_rule_prose,
        "candidate_playbooks": [_playbook_payload(p) for p in enriched.candidate_playbooks],
    }


def _driver_payload(driver: Any) -> Dict[str, Any]:
    return {
        "feature_name": driver.feature_name,
        "business_phrase": driver.business_phrase,
        "value_phrase": driver.value_phrase,
        "polarity": driver.polarity,
        "mean_shap": _round(driver.mean_shap, 4),
        "stability": _round(driver.stability, 3),
    }


def _contrast_payload(contrast: Any) -> Dict[str, Any]:
    return {
        "feature_name": contrast.feature_name,
        "business_phrase": contrast.business_phrase,
        "self_phrase": contrast.self_phrase,
        "sibling_phrase": contrast.sibling_phrase,
    }


def _sibling_summary(sibling: "EnrichedArchetypeContext") -> Dict[str, Any]:
    return {
        "cluster_index": sibling.cluster_index,
        "cluster_size": sibling.cluster_size,
        "cluster_mean_churn_probability": _round(sibling.cluster_mean_churn_probability, 4),
        "top_positive_drivers": [
            {"business_phrase": d.business_phrase, "value_phrase": d.value_phrase}
            for d in sibling.top_positive_drivers
        ],
    }


def _playbook_payload(playbook: Any) -> Dict[str, Any]:
    return {
        "playbook_id": playbook.playbook_id,
        "name": playbook.name,
        "when_applicable": playbook.when_applicable,
        "description": playbook.description,
        "policy_summary": playbook.policy_summary,
        "expected_effect": playbook.expected_effect,
        "fit_score_prefilter": _round(playbook.fit_score, 3),
    }


def _round(value: Optional[float], ndigits: int) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


__all__ = ["build_enriched_prompt_messages"]
