"""Tests for the Phase 4 enriched prompt builder."""
from __future__ import annotations

import json

import pytest

from customer_retention.stages.causal.interpretation.archetype_context import (
    ContrastFeature,
    EnrichedArchetypeContext,
    EnrichedDriver,
    EnrichedPlaybook,
)
from customer_retention.stages.causal.interpretation.llm_prompt import (
    build_enriched_prompt_messages,
)


def _ctx(**overrides):
    defaults = dict(
        cluster_index=0,
        cluster_size=250,
        cluster_mean_churn_probability=0.42,
        share_of_book=0.25,
        lift_vs_population=1.4,
        arr_exposure=1_200_000.5,
        top_positive_drivers=[
            EnrichedDriver(
                feature_name="nps_mean_90d",
                business_phrase="average NPS score over last 90 days",
                value=8.5,
                value_phrase="elevated",
                polarity="high_is_good",
                mean_shap=0.18,
                stability=0.88,
            )
        ],
        top_negative_drivers=[
            EnrichedDriver(
                feature_name="tenure_days",
                business_phrase="tenure in days",
                value=30.0,
                value_phrase="low",
                polarity="high_is_good",
                mean_shap=-0.12,
                stability=0.7,
            )
        ],
        sibling_contrast=[
            ContrastFeature(
                feature_name="nps_mean_90d",
                business_phrase="average NPS score over last 90 days",
                self_value=2.0,
                sibling_value=8.0,
                self_phrase="low",
                sibling_phrase="elevated",
            )
        ],
        representative_examples=[{"entity_id": "abc", "churn_probability": 0.71}],
        eligibility_rule_prose=(
            "average NPS score over last 90 days is low (below 4) AND tenure in days is low"
        ),
        candidate_playbooks=[
            EnrichedPlaybook(
                playbook_id="nps_followup",
                name="NPS Follow-up",
                when_applicable="low NPS accounts",
                description="CSM reaches out to unhappy NPS accounts.",
                policy_summary="14-day cooldown, 50 slots/week.",
                expected_effect="~22% churn reduction historically.",
                fit_score=0.81,
            )
        ],
    )
    defaults.update(overrides)
    return EnrichedArchetypeContext(**defaults)


class TestBuildEnrichedPromptMessages:
    def test_returns_system_and_user_messages(self):
        msgs = build_enriched_prompt_messages(_ctx())
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_message_forbids_invention(self):
        msgs = build_enriched_prompt_messages(_ctx())
        system = msgs[0]["content"].lower()
        assert "not invent" in system or "must not" in system
        assert "verbatim" in system

    def test_user_payload_is_valid_json(self):
        msgs = build_enriched_prompt_messages(_ctx())
        parsed = json.loads(msgs[1]["content"])
        assert "archetype" in parsed
        assert "response_schema" in parsed

    def test_user_payload_narrates_every_driver(self):
        payload = json.loads(build_enriched_prompt_messages(_ctx())[1]["content"])
        pos = payload["archetype"]["top_positive_drivers"][0]
        assert pos["business_phrase"] == "average NPS score over last 90 days"
        assert pos["value_phrase"] == "elevated"
        assert pos["polarity"] == "high_is_good"
        assert "mean_value" not in pos  # raw numeric value is intentionally omitted

    def test_eligibility_rule_prose_appears_verbatim(self):
        payload = json.loads(build_enriched_prompt_messages(_ctx())[1]["content"])
        assert (
            "average NPS score over last 90 days is low"
            in payload["archetype"]["eligibility_rule_prose"]
        )

    def test_playbook_carries_policy_and_expected_effect(self):
        payload = json.loads(build_enriched_prompt_messages(_ctx())[1]["content"])
        pb = payload["archetype"]["candidate_playbooks"][0]
        assert pb["policy_summary"] == "14-day cooldown, 50 slots/week."
        assert pb["expected_effect"].startswith("~22%")

    def test_sibling_contrast_phrases_rendered(self):
        payload = json.loads(build_enriched_prompt_messages(_ctx())[1]["content"])
        contrast = payload["archetype"]["sibling_contrast"][0]
        assert contrast["self_phrase"] == "low"
        assert contrast["sibling_phrase"] == "elevated"

    def test_response_schema_contract_required_fields(self):
        payload = json.loads(build_enriched_prompt_messages(_ctx())[1]["content"])
        schema = payload["response_schema"]
        assert "archetype_name" in schema
        assert "archetype_description" in schema
        assert "contrast_with_sibling" in schema
        assert "playbooks" in schema
        assert "confidence" in schema
        rationale_hint = schema["playbooks"][0]["rationale"].lower()
        assert "business_phrase" in rationale_hint
        assert "verbatim" in rationale_hint

    def test_empty_contrast_and_playbooks_do_not_crash(self):
        payload = json.loads(
            build_enriched_prompt_messages(
                _ctx(sibling_contrast=[], candidate_playbooks=[])
            )[1]["content"]
        )
        assert payload["archetype"]["sibling_contrast"] == []
        assert payload["archetype"]["candidate_playbooks"] == []

    def test_none_share_and_lift_serialize_as_null(self):
        payload = json.loads(
            build_enriched_prompt_messages(
                _ctx(share_of_book=None, lift_vs_population=None, arr_exposure=None)
            )[1]["content"]
        )
        assert payload["archetype"]["share_of_book"] is None
        assert payload["archetype"]["lift_vs_population"] is None
        assert payload["archetype"]["arr_exposure"] is None

    def test_numeric_rounding_keeps_context_compact(self):
        payload = json.loads(build_enriched_prompt_messages(_ctx())[1]["content"])
        assert payload["archetype"]["cluster_mean_churn_probability"] == pytest.approx(0.42)
        assert payload["archetype"]["share_of_book"] == pytest.approx(0.25)

    def test_siblings_summary_includes_top_positive(self):
        sibling = _ctx(cluster_index=1)
        msgs = build_enriched_prompt_messages(_ctx(), siblings=[sibling])
        payload = json.loads(msgs[1]["content"])
        assert len(payload["archetype"]["siblings"]) == 1
        sib = payload["archetype"]["siblings"][0]
        assert sib["cluster_index"] == 1
        assert sib["top_positive_drivers"][0]["business_phrase"].startswith("average NPS")
