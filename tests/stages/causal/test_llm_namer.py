"""Unit tests for ``llm_namer``.

The ``TemplateNamer`` is exercised end-to-end with no mocks. The Databricks
Foundation Model namer is exercised with a fake OpenAI client substituted
into the module via monkeypatch — the test verifies prompt construction,
JSON parsing, and the fallback path. No real network calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal.llm_namer import (
    ArchetypeContext,
    DatabricksFoundationModelNamer,
    TemplateNamer,
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
                "description": "NPS recovery outreach",
                "overlap_score": 0.5,
            },
            {
                "playbook_id": "relationship",
                "playbook_version": "1.0.0",
                "description": "Quarterly relationship review",
                "overlap_score": 0.33,
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
        assert isinstance(namer, TemplateNamer)

    def test_none_returns_template_namer(self):
        namer = build_llm_namer(None)
        assert isinstance(namer, TemplateNamer)

    def test_non_empty_endpoint_returns_databricks_namer(self):
        namer = build_llm_namer("databricks-claude-sonnet-4-6")
        assert isinstance(namer, DatabricksFoundationModelNamer)
        assert namer.endpoint_name == "databricks-claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# TemplateNamer
# ---------------------------------------------------------------------------


class TestTemplateNamer:
    def test_model_id_is_template(self):
        assert TemplateNamer().model_id == "template"

    def test_name_uses_top_positive_driver(self):
        result = TemplateNamer().name_archetype(_make_context())
        assert "tenure_days" in result.archetype_name
        assert result.llm_model_id == "template"

    def test_name_falls_back_to_negative_driver(self):
        ctx = _make_context(top_positive_drivers=[])
        result = TemplateNamer().name_archetype(ctx)
        assert "nps_score" in result.archetype_name

    def test_name_falls_back_to_index_only_when_no_drivers(self):
        ctx = _make_context(top_positive_drivers=[], top_negative_drivers=[])
        result = TemplateNamer().name_archetype(ctx)
        assert result.archetype_name == "Archetype 1"

    def test_description_includes_size_and_mean_prob(self):
        result = TemplateNamer().name_archetype(_make_context())
        assert "120" in result.archetype_description
        assert "0.42" in result.archetype_description

    def test_one_decision_per_candidate(self):
        result = TemplateNamer().name_archetype(_make_context())
        assert len(result.playbooks) == 2
        assert {d.playbook_id for d in result.playbooks} == {"low_nps", "relationship"}

    def test_decision_fit_score_uses_overlap(self):
        result = TemplateNamer().name_archetype(_make_context())
        scores = {d.playbook_id: d.fit_score for d in result.playbooks}
        assert scores["low_nps"] == pytest.approx(0.5)
        assert scores["relationship"] == pytest.approx(0.33)

    def test_no_candidates_yields_zero_confidence(self):
        ctx = _make_context(candidate_playbooks=[])
        result = TemplateNamer().name_archetype(ctx)
        assert result.confidence == 0.0
        assert result.playbooks == []


# ---------------------------------------------------------------------------
# DatabricksFoundationModelNamer
# ---------------------------------------------------------------------------


class _FakeOpenAIResponse:
    def __init__(self, content: str) -> None:
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        self.choices = [choice]


class _FakeOpenAIClient:
    def __init__(self, content: str = "", raise_exc: Exception | None = None) -> None:
        self.content = content
        self.raise_exc = raise_exc
        self.last_kwargs: dict | None = None
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.raise_exc is not None:
            raise self.raise_exc
        return _FakeOpenAIResponse(self.content)


class TestDatabricksFoundationModelNamer:
    def test_model_id_is_endpoint_name(self):
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        assert namer.model_id == "databricks-claude-sonnet-4-6"

    def test_no_workspace_url_falls_back_to_template(self):
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="",
            workspace_token="",
        )
        result = namer.name_archetype(_make_context())
        assert result.llm_model_id == "databricks-claude-sonnet-4-6:fallback"
        assert "tenure_days" in result.archetype_name

    def test_successful_response_parses_json(self):
        client = _FakeOpenAIClient(
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
        client = _FakeOpenAIClient(content="not even close to json")
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
        client = _FakeOpenAIClient(raise_exc=ConnectionError("network down"))
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        namer._client = client
        result = namer.name_archetype(_make_context())
        assert result.llm_model_id == "databricks-claude-sonnet-4-6:fallback"

    def test_json_in_code_fence_is_extracted(self):
        client = _FakeOpenAIClient(
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
        client = _FakeOpenAIClient(
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
        client = _FakeOpenAIClient(
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

    def test_get_client_returns_none_without_openai(self, monkeypatch):
        namer = DatabricksFoundationModelNamer(
            endpoint_name="databricks-claude-sonnet-4-6",
            workspace_url="https://example.com",
            workspace_token="token",
        )
        # Force the import path to fail
        import sys

        monkeypatch.setitem(sys.modules, "openai", None)
        client = namer._get_client()
        assert client is None
