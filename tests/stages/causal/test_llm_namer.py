"""Unit tests for ``llm_namer``.

``ProseOverlapMatcher`` is exercised end-to-end with no mocks — each
test drives a real (archetype, playbooks) pair through the matcher and
inspects the returned naming + fit decisions.

The Databricks Foundation Model namer is exercised with a fake OpenAI
client substituted into the module via monkeypatch — the test verifies
prompt construction, JSON parsing, and the fallback path. No real
network calls.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from customer_retention.stages.causal.llm_namer import (
    ArchetypeContext,
    DatabricksFoundationModelNamer,
    ProseOverlapMatcher,
    build_llm_namer,
)


def _make_context(**overrides: Any) -> ArchetypeContext:
    defaults = dict(
        cluster_index=1,
        cluster_size=120,
        cluster_mean_churn_probability=0.42,
        top_positive_drivers=[
            {"feature": "tenure_days", "mean_shap": 0.18, "mean_value": 100.0},
            {"feature": "support_tickets_30d", "mean_shap": 0.07, "mean_value": 4.0},
        ],
        top_negative_drivers=[
            {"feature": "nps_score", "mean_shap": -0.12, "mean_value": 8.0},
        ],
        candidate_playbooks=[
            {
                "playbook_id": "low_nps",
                "playbook_version": "1.0.0",
                "name": "Low NPS",
                "description": "NPS recovery outreach for low-score survey responses",
                "when_applicable": "",
            },
            {
                "playbook_id": "relationship",
                "playbook_version": "1.0.0",
                "name": "Relationship",
                "description": "Quarterly relationship review — tenure and support history",
                "when_applicable": "",
            },
        ],
    )
    defaults.update(overrides)
    return ArchetypeContext(**defaults)


# ---------------------------------------------------------------------------
# build_llm_namer factory
# ---------------------------------------------------------------------------


class TestBuildLLMNamer:
    def test_empty_endpoint_returns_template_namer(self):
        namer = build_llm_namer("")
        assert isinstance(namer, ProseOverlapMatcher)

    def test_none_returns_template_namer(self):
        namer = build_llm_namer(None)
        assert isinstance(namer, ProseOverlapMatcher)

    def test_non_empty_endpoint_returns_databricks_namer(self):
        namer = build_llm_namer("databricks-claude-sonnet-4-6")
        assert isinstance(namer, DatabricksFoundationModelNamer)
        assert namer.endpoint_name == "databricks-claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# ProseOverlapMatcher
# ---------------------------------------------------------------------------


class TestProseOverlapMatcher:
    def test_model_id_is_prose_overlap(self):
        assert ProseOverlapMatcher().model_id == "prose_overlap"

    def test_name_uses_top_positive_driver(self):
        result = ProseOverlapMatcher().name_archetype(_make_context())
        assert "tenure_days" in result.archetype_name
        assert result.llm_model_id == "prose_overlap"

    def test_name_falls_back_to_negative_driver(self):
        ctx = _make_context(top_positive_drivers=[])
        result = ProseOverlapMatcher().name_archetype(ctx)
        assert "nps_score" in result.archetype_name

    def test_name_falls_back_to_index_only_when_no_drivers(self):
        ctx = _make_context(top_positive_drivers=[], top_negative_drivers=[])
        result = ProseOverlapMatcher().name_archetype(ctx)
        assert result.archetype_name == "Archetype 1"

    def test_description_includes_size_and_mean_prob(self):
        result = ProseOverlapMatcher().name_archetype(_make_context())
        assert "120" in result.archetype_description
        assert "0.42" in result.archetype_description

    def test_one_decision_per_candidate(self):
        result = ProseOverlapMatcher().name_archetype(_make_context())
        assert len(result.playbooks) == 2
        assert {d.playbook_id for d in result.playbooks} == {"low_nps", "relationship"}

    def test_fit_score_reflects_prose_overlap(self):
        """low_nps's prose mentions 'NPS'; that driver token overlaps. Relationship's mentions 'tenure'."""
        result = ProseOverlapMatcher().name_archetype(_make_context())
        scores = {d.playbook_id: d.fit_score for d in result.playbooks}
        # Both playbooks should get a non-zero score because archetype tokens
        # from nps_score/tenure_days/support_tickets match their prose.
        assert scores["low_nps"] > 0.0
        assert scores["relationship"] > 0.0

    def test_rationale_lists_matched_tokens(self):
        result = ProseOverlapMatcher().name_archetype(_make_context())
        low_nps = next(d for d in result.playbooks if d.playbook_id == "low_nps")
        assert "nps" in low_nps.rationale.lower()

    def test_enriched_description_uses_business_phrases(self):
        from customer_retention.stages.causal.interpretation.archetype_context import (
            EnrichedArchetypeContext,
            EnrichedDriver,
        )
        enriched = EnrichedArchetypeContext(
            cluster_index=1, cluster_size=120, cluster_mean_churn_probability=0.42,
            eligibility_rule_prose="tenure in days is low AND NPS score is low",
            top_positive_drivers=[EnrichedDriver(
                feature_name="tenure_days", business_phrase="tenure in days",
                value=30.0, value_phrase="low", polarity=None,
            )],
            top_negative_drivers=[EnrichedDriver(
                feature_name="nps_score", business_phrase="NPS score",
                value=8.0, value_phrase="elevated", polarity="high_is_good",
            )],
        )
        result = ProseOverlapMatcher().name_archetype(_make_context(), enriched=enriched)
        assert "tenure in days" in result.archetype_description
        assert "NPS score" in result.archetype_description
        assert "tenure in days is low AND NPS score is low" in result.archetype_description

    def test_zero_candidates_yields_zero_confidence(self):
        ctx = _make_context(candidate_playbooks=[])
        result = ProseOverlapMatcher().name_archetype(ctx)
        assert result.confidence == 0.0
        assert result.playbooks == []


# ---------------------------------------------------------------------------
# DatabricksFoundationModelNamer
# ---------------------------------------------------------------------------


class _FakeDeployClient:
    """Stand-in for ``mlflow.deployments.get_deploy_client('databricks')``.

    Its ``predict(endpoint, inputs)`` signature + ``{"choices": [{"message":
    {"content": ...}}]}`` response shape matches the real Databricks
    Mosaic AI serving-endpoint response, so the namer's response parsing
    is exercised end-to-end without network.
    """

    def __init__(self, content: str = "", raise_exc: Exception | None = None) -> None:
        self.content = content
        self.raise_exc = raise_exc
        self.last_endpoint: str | None = None
        self.last_inputs: dict | None = None
        # Tests that predate the deploy-client switch still read
        # ``last_kwargs["messages"]``; keep the compat alias.
        self.last_kwargs: dict | None = None

    def predict(self, endpoint: str, inputs: dict):
        self.last_endpoint = endpoint
        self.last_inputs = inputs
        self.last_kwargs = inputs
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"choices": [{"message": {"content": self.content}}]}


class TestDatabricksFoundationModelNamer:
    def test_model_id_is_endpoint_name(self):
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        assert namer.model_id == "databricks-claude-sonnet-4-6"

    def test_client_unavailable_falls_back_to_template(self, monkeypatch):
        """When `mlflow.deployments` can't create a deploy client (off-cluster,
        missing SDK), the namer falls back to `ProseOverlapMatcher` with the
        `:fallback` suffix on the model id."""
        import sys
        monkeypatch.setitem(sys.modules, "mlflow.deployments", None)
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
        )
        result = namer.name_archetype(_make_context())
        assert result.llm_model_id == "databricks-claude-sonnet-4-6:fallback"
        assert "tenure_days" in result.archetype_name

    def test_successful_response_parses_json(self):
        client = _FakeDeployClient(
            content=json.dumps(
                {
                    "archetype_name": "High-Risk Tenured",
                    "archetype_description": "Long-tenured accounts showing churn signals.",
                    "playbooks": [
                        {
                            "playbook_id": "low_nps",
                            "fit_score": 0.92,
                            "rationale": "Targets the NPS pattern this cluster shows.",
                        }
                    ],
                    "confidence": 0.88,
                }
            )
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        result = namer.name_archetype(_make_context())
        assert result.archetype_name == "High-Risk Tenured"
        assert result.playbooks[0].fit_score == pytest.approx(0.92)
        assert result.confidence == pytest.approx(0.88)
        assert result.llm_model_id == "databricks-claude-sonnet-4-6"

    def test_malformed_json_falls_back(self):
        client = _FakeDeployClient(content="not even close to json")
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        result = namer.name_archetype(_make_context())
        assert result.llm_model_id == "databricks-claude-sonnet-4-6:fallback"
        assert "tenure_days" in result.archetype_name

    def test_network_exception_falls_back(self):
        client = _FakeDeployClient(raise_exc=ConnectionError("network down"))
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        result = namer.name_archetype(_make_context())
        assert result.llm_model_id == "databricks-claude-sonnet-4-6:fallback"

    def test_json_in_code_fence_is_extracted(self):
        client = _FakeDeployClient(
            content="```json\n"
            + json.dumps(
                {
                    "archetype_name": "Wrapped",
                    "archetype_description": "Wrapped result.",
                    "playbooks": [],
                    "confidence": 0.5,
                }
            )
            + "\n```"
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        result = namer.name_archetype(_make_context())
        assert result.archetype_name == "Wrapped"

    def test_json_with_leading_text_is_extracted(self):
        client = _FakeDeployClient(
            content="here you go: "
            + json.dumps(
                {
                    "archetype_name": "Extracted",
                    "archetype_description": "Extracted from text.",
                    "playbooks": [],
                    "confidence": 0.3,
                }
            )
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        result = namer.name_archetype(_make_context())
        assert result.archetype_name == "Extracted"

    def test_prompt_includes_archetype_context(self):
        client = _FakeDeployClient(
            content=json.dumps(
                {"archetype_name": "X", "archetype_description": "x", "playbooks": []}
            )
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        namer.name_archetype(_make_context())
        prompt = client.last_kwargs["messages"][0]["content"]
        assert "tenure_days" in prompt
        assert "low_nps" in prompt
        assert "0.420" in prompt or "0.42" in prompt

    def test_prompt_injects_when_applicable_and_overlap_score(self):
        long_description = (
            "This is a deliberately long playbook description that exceeds the legacy "
            "200-character truncation boundary so we can assert the full text reaches the "
            "LLM rather than a truncated prefix. " * 3
        )
        when_applicable_text = "Apply when subscriber has missed 3 consecutive sends and lifecycle quadrant is steady_loyal"
        client = _FakeDeployClient(
            content=json.dumps(
                {"archetype_name": "X", "archetype_description": "x", "playbooks": []}
            )
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        ctx = _make_context(
            candidate_playbooks=[
                {
                    "playbook_id": "frequency_adjustment",
                    "playbook_version": "1.0.0",
                    "name": "Frequency Adjustment",
                    "description": long_description,
                    "when_applicable": when_applicable_text,
                    "overlap_score": 0.71,
                },
            ],
        )
        namer.name_archetype(ctx)
        prompt = client.last_kwargs["messages"][0]["content"]
        assert when_applicable_text in prompt, (
            "when_applicable must be injected into the prompt so the LLM sees the "
            "playbook's applicability rules"
        )
        assert long_description.strip() in prompt, (
            "description must be injected in full (no [:200] truncation)"
        )
        assert "0.71" in prompt, (
            "overlap_score from the candidate dict must render in the prompt as a "
            "deterministic floor signal"
        )
        assert "INFORMATIONAL ONLY" in prompt, (
            "the rubric should explicitly decouple cluster_mean_churn_probability from fit_score"
        )
        assert "Omitting a playbook is a hard error" in prompt, (
            "the prompt must instruct the LLM to score every candidate"
        )

    def test_enriched_context_routes_through_phase4_builder(self):
        from customer_retention.stages.causal.interpretation.archetype_context import (
            EnrichedArchetypeContext,
            EnrichedDriver,
            EnrichedPlaybook,
        )
        client = _FakeDeployClient(
            content=json.dumps(
                {"archetype_name": "Low Tenure Risk", "archetype_description": "x",
                 "playbooks": [{"playbook_id": "nps_followup", "fit_score": 0.9, "rationale": "r"}],
                 "confidence": 0.9}
            )
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        enriched = EnrichedArchetypeContext(
            cluster_index=0, cluster_size=100, cluster_mean_churn_probability=0.5,
            eligibility_rule_prose="tenure is low AND NPS is low",
            top_positive_drivers=[EnrichedDriver(
                feature_name="tenure_days", business_phrase="tenure in days",
                value=30.0, value_phrase="low", polarity="high_is_good",
            )],
            candidate_playbooks=[EnrichedPlaybook(
                playbook_id="nps_followup", name="NPS Follow-up",
            )],
        )
        result = namer.name_archetype(_make_context(), enriched=enriched)
        assert result.archetype_name == "Low Tenure Risk"
        messages = client.last_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        user_content = messages[1]["content"]
        assert "tenure in days" in user_content
        assert "tenure is low AND NPS is low" in user_content

    def test_legacy_path_uses_raw_prompt_when_no_enriched(self):
        client = _FakeDeployClient(
            content=json.dumps(
                {"archetype_name": "X", "archetype_description": "x", "playbooks": []}
            )
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        namer.name_archetype(_make_context())
        messages = client.last_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "tenure_days" in messages[0]["content"]

    def test_enriched_path_fallback_preserves_enriched_description(self):
        from customer_retention.stages.causal.interpretation.archetype_context import (
            EnrichedArchetypeContext,
            EnrichedDriver,
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="",
            workspace_token="",
        )
        enriched = EnrichedArchetypeContext(
            cluster_index=0, cluster_size=100, cluster_mean_churn_probability=0.5,
            eligibility_rule_prose="tenure is low",
            top_positive_drivers=[EnrichedDriver(
                feature_name="tenure_days", business_phrase="tenure in days",
                value=30.0, value_phrase="low", polarity=None,
            )],
        )
        result = namer.name_archetype(_make_context(), enriched=enriched)
        assert result.llm_model_id.endswith(":fallback")
        assert "tenure in days" in result.archetype_description
        assert "tenure is low" in result.archetype_description

    def test_get_client_returns_none_without_mlflow_deployments(self, monkeypatch):
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
        )
        import sys
        monkeypatch.setitem(sys.modules, "mlflow.deployments", None)
        client = namer._get_client()
        assert client is None

    def test_predict_payload_matches_deploy_client_contract(self):
        """Namer calls `client.predict(endpoint=..., inputs={...})` with
        messages, max_tokens, temperature packed into ``inputs`` — matches
        the real `DatabricksDeploymentClient` signature."""
        client = _FakeDeployClient(
            content=json.dumps(
                {"archetype_name": "X", "archetype_description": "x", "playbooks": [],
                 "confidence": 0.0}
            )
        )
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            max_tokens=1234, temperature=0.25,
        )
        namer._client = client
        namer.name_archetype(_make_context())
        assert client.last_endpoint == "databricks-claude-sonnet-4-6"
        assert "messages" in client.last_inputs
        assert client.last_inputs["max_tokens"] == 1234
        assert client.last_inputs["temperature"] == 0.25
