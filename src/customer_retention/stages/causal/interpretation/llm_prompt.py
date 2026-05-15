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


_ARCHETYPE_NAMING_RULES = (
    "═══ ARCHETYPE NAMING RULES ═══\n"
    "The archetype_name is the customer-facing label that appears in CSM\n"
    "dashboards, intervention queues, and weekly reports. Humans read it\n"
    "every day — quality matters more than the JSON envelope.\n\n"

    "REQUIREMENTS:\n"
    "  • 2-4 words, Title Case (\"Pre-Renewal Power Users\")\n"
    "  • Names a CUSTOMER SEGMENT, not the data behind it — describe what\n"
    "    these accounts ARE, not which features explain them\n"
    "  • Uses CSM/retention vocabulary a non-technical operator recognises:\n"
    "    Onboarding, Renewal, Win-Back, Expansion, At-Risk, Stable,\n"
    "    Disengaged, High-Touch, Self-Serve, Power Users, Drop-Off,\n"
    "    Long-Tenure, New, Pre-Cancel, Re-Engaging\n"
    "  • Pattern: <CUSTOMER STATE adjective> + <CUSTOMER NOUN>\n\n"

    "GOOD EXAMPLES:\n"
    "  • \"New-Onboarding Adopters\"      (ramping engagement, recent start)\n"
    "  • \"Pre-Renewal Power Users\"      (high engagement, renewal window)\n"
    "  • \"Disengaged Long-Tenure\"       (low recent activity, established)\n"
    "  • \"Cancellation Risk\"            (cancel-precursor patterns)\n"
    "  • \"Stable Subscribers\"           (steady-state, low risk)\n"
    "  • \"Re-Engaging Returners\"        (activity rebound after gap)\n"
    "  • \"Single-Partner At-Risk\"       (concentration + cancel signal)\n\n"

    "FORBIDDEN — these will be rejected at review:\n"
    "  • \"Archetype 0\", \"Cluster 3\", or any reference to a numeric index\n"
    "  • Raw feature names: \"High Recent Vs Overall Ratio\",\n"
    "    \"Subscription Terminate Ratio Group\"\n"
    "  • Statistical / modelling jargon: \"shap\", \"centroid\", \"cluster\",\n"
    "    \"segment\" (without a state adjective), \"cohort\"\n"
    "  • Bloated descriptions: \"High Engagement Existing Customers With\n"
    "    Subscription Activity\" — keep it ≤ 4 words\n"
    "  • Pure feature mentions: \"Engagement Ratio Shift\" (it's a feature\n"
    "    name; say what the customer IS, e.g. \"Re-Engaging Returners\")\n"
)


_SYSTEM_MESSAGE = (
    "You are naming customer-churn archetypes and rating retention-playbook fit. "
    "Every numeric fact in the user payload is already narrated — you MUST NOT "
    "invent thresholds, percentages, or counts. Reference provided business "
    "phrases verbatim. Output JSON only, no prose commentary.\n\n"

    + _ARCHETYPE_NAMING_RULES + "\n"

    "═══ SCORING RUBRIC ═══\n"
    "fit_score is a 0.0-1.0 measure of how well THIS PLAYBOOK ADDRESSES THIS "
    "ARCHETYPE'S RISK PATTERN. It is NOT a probability of churn, NOT a predicted "
    "uplift, NOT a recommendation strength. Use these brackets:\n"
    "  0.85-1.00  Perfect fit: playbook is explicitly designed for this exact "
    "risk pattern.\n"
    "  0.50-0.84  Strong fit: addresses 2+ of the archetype's top drivers.\n"
    "  0.20-0.49  Partial fit: addresses one driver or a closely related concept.\n"
    "  0.05-0.19  Weak fit: in the retention space but a different pattern. "
    "Always preferred over dropping a playbook.\n"
    "  0.00-0.04  No fit: unrelated even at the vocabulary level.\n\n"

    "═══ ANCHORING RULES ═══\n"
    "1. Each playbook arrives with a ``fit_score_prefilter`` — a deterministic "
    "vocabulary-overlap score between the playbook's prose and the archetype's "
    "driver feature names. Use it as a FLOOR signal: when it is 0.40 or higher, "
    "your fit_score should rarely fall below 0.20. Disagreement is allowed but "
    "must be justified in the rationale.\n"
    "2. Score EVERY candidate playbook in the response array, sorted by "
    "fit_score descending. Omitting a playbook is a hard error — downstream "
    "policy generation requires every candidate to receive a score.\n"
    "3. ``cluster_mean_churn_probability``, ``lift_vs_population``, "
    "``arr_exposure``, and ``share_of_book`` are INFORMATIONAL ONLY — do not "
    "let predicted churn volume drive fit_score. A cluster with low predicted "
    "churn still gets a fit_score reflecting which playbook would apply IF its "
    "members showed risk.\n"
    "4. Return JSON only — no markdown fences, no commentary."
)

_RESPONSE_SCHEMA = {
    "archetype_name": (
        "2-4 word Title-Case label following ARCHETYPE NAMING RULES. "
        "Customer-segment language only — never feature names, cluster numbers, "
        "or statistical jargon. See system message for examples."
    ),
    "archetype_description": (
        "2 sentences. First sentence describes the customer segment in business "
        "terms (no feature names). Second sentence MUST quote the "
        "eligibility_rule_prose verbatim (or say 'no eligibility rule' when absent)."
    ),
    "contrast_with_sibling": (
        "1 sentence using the sibling_contrast phrases. Empty string when "
        "sibling_contrast is missing."
    ),
    "playbooks": [
        {
            "playbook_id": "string (one of the provided candidate ids)",
            "fit_score": "0.0 to 1.0 per the rubric in the system message",
            "rationale": (
                "1 sentence citing at least one top_positive_driver or "
                "top_negative_driver business_phrase verbatim, and explaining "
                "the score relative to fit_score_prefilter when they diverge."
            ),
        }
    ],
    "confidence": "0.0 to 1.0, your overall confidence in this mapping",
    "_response_rules": (
        "MUST contain exactly one entry per candidate_playbook. Sort by "
        "fit_score descending. Omitting a candidate is a hard error."
    ),
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
