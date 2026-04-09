"""Tests for the policy YAML → row loader."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from customer_retention.stages.causal.policy_loader import (
    load_decision_policy,
    load_policies_from_dir,
    load_response_schemas,
    load_vocabularies,
)
from customer_retention.stages.causal.schemas import (
    decision_policy_schema,
    response_schemas_schema,
    vocabularies_schema,
)


@pytest.fixture
def policies_dir(tmp_path: Path) -> Path:
    """A populated playbooks_dir/policies/ that mirrors the framework seeds."""
    pdir = tmp_path / "policies"
    pdir.mkdir()

    (pdir / "decision_policy.yaml").write_text(
        textwrap.dedent(
            """
            decision_policy_id: test_policy_v1
            version: "1.0.0"
            valid_from: "2026-04-01T00:00:00Z"
            valid_to: null
            eligibility_logic_version: "1.0"
            holdout_stratification_fields:
              - risk_decile
              - archetype_id
            holdout_seed_recipe: "sha256(account_id)"
            holdout_fractions_by_playbook:
              alpha: 0.10
              beta: 0.05
            capacity_by_playbook:
              alpha: 100
              beta: 25
            cooldown_days_by_playbook:
              alpha: 14
              beta: 30
            suppression_rules:
              alpha:
                - field: arr_usd
                  op: lt
                  value: 10000
            channel_availability_rules: {}
            risk_tier_high_threshold: 0.65
            risk_tier_medium_threshold: 0.35
            description: "Test policy"
            changed_by: "test_user"
            """
        ).strip()
    )

    (pdir / "response_schemas.yaml").write_text(
        textwrap.dedent(
            """
            - schema_id: test_email_v1
              version: "1.0.0"
              name: "Test Email Schema"
              description: "Just for tests"
              fields:
                - name: opened
                  type: bool
                - name: clicked
                  type: bool
                - name: reply_sentiment
                  type: enum
                  vocabulary: sentiment
            - schema_id: test_call_v1
              version: "1.0.0"
              fields:
                - name: contact_made
                  type: bool
            """
        ).strip()
    )

    (pdir / "vocabularies.yaml").write_text(
        textwrap.dedent(
            """
            sentiment:
              - positive
              - negative
              - neutral
            engagement_level:
              - engaged
              - disengaged
            """
        ).strip()
    )
    return tmp_path  # parent of policies/ — the loader appends it


class TestLoadDecisionPolicy:
    def test_parses_single_policy(self, policies_dir):
        rows = load_decision_policy(policies_dir / "policies" / "decision_policy.yaml")
        assert len(rows) == 1

    def test_field_coercion(self, policies_dir):
        rows = load_decision_policy(policies_dir / "policies" / "decision_policy.yaml")
        row = rows[0]
        assert row["decision_policy_id"] == "test_policy_v1"
        assert row["version"] == "1.0.0"
        assert isinstance(row["valid_from"], datetime)
        assert row["valid_to"] is None
        assert row["holdout_stratification_fields"] == ["risk_decile", "archetype_id"]
        assert row["holdout_fractions_by_playbook"] == {"alpha": 0.10, "beta": 0.05}
        assert row["capacity_by_playbook"] == {"alpha": 100, "beta": 25}
        assert row["cooldown_days_by_playbook"] == {"alpha": 14, "beta": 30}
        assert row["risk_tier_high_threshold"] == 0.65
        assert row["risk_tier_medium_threshold"] == 0.35

    def test_suppression_rules_serialized_to_json(self, policies_dir):
        import json

        rows = load_decision_policy(policies_dir / "policies" / "decision_policy.yaml")
        # JSON-string with sorted keys for stability
        rules = json.loads(rows[0]["suppression_rules"])
        assert "alpha" in rules
        assert rules["alpha"][0]["field"] == "arr_usd"

    def test_missing_file_returns_empty(self, tmp_path):
        rows = load_decision_policy(tmp_path / "doesnt_exist.yaml")
        assert rows == []

    def test_row_keys_match_schema(self, policies_dir):
        rows = load_decision_policy(policies_dir / "policies" / "decision_policy.yaml")
        schema_fields = {f.name for f in decision_policy_schema().fields}
        for row in rows:
            assert set(row.keys()) == schema_fields


class TestLoadResponseSchemas:
    def test_parses_list_of_schemas(self, policies_dir):
        rows = load_response_schemas(policies_dir / "policies" / "response_schemas.yaml")
        assert len(rows) == 2
        ids = {r["schema_id"] for r in rows}
        assert ids == {"test_email_v1", "test_call_v1"}

    def test_fields_serialized_to_json(self, policies_dir):
        import json

        rows = load_response_schemas(policies_dir / "policies" / "response_schemas.yaml")
        email_row = next(r for r in rows if r["schema_id"] == "test_email_v1")
        fields = json.loads(email_row["fields"])
        assert any(f["name"] == "opened" and f["type"] == "bool" for f in fields)
        assert any(
            f["name"] == "reply_sentiment" and f.get("vocabulary") == "sentiment"
            for f in fields
        )

    def test_default_version_when_missing(self, policies_dir):
        rows = load_response_schemas(policies_dir / "policies" / "response_schemas.yaml")
        call_row = next(r for r in rows if r["schema_id"] == "test_call_v1")
        # YAML omitted version → loader defaults to "1.0.0"
        assert call_row["version"] == "1.0.0"

    def test_row_keys_match_schema(self, policies_dir):
        rows = load_response_schemas(policies_dir / "policies" / "response_schemas.yaml")
        schema_fields = {f.name for f in response_schemas_schema().fields}
        for row in rows:
            assert set(row.keys()) == schema_fields


class TestLoadVocabularies:
    def test_parses_dict_shape(self, policies_dir):
        rows = load_vocabularies(policies_dir / "policies" / "vocabularies.yaml")
        # 3 sentiment values + 2 engagement_level values
        assert len(rows) == 5

    def test_value_per_row(self, policies_dir):
        rows = load_vocabularies(policies_dir / "policies" / "vocabularies.yaml")
        sentiments = sorted(r["value"] for r in rows if r["vocabulary_name"] == "sentiment")
        assert sentiments == ["negative", "neutral", "positive"]

    def test_valid_from_defaults_to_now(self, policies_dir):
        rows = load_vocabularies(policies_dir / "policies" / "vocabularies.yaml")
        for row in rows:
            assert isinstance(row["valid_from"], datetime)
            assert row["valid_to"] is None

    def test_explicit_list_shape(self, tmp_path):
        # Shape B from the docstring — explicit list with descriptions
        vocab_file = tmp_path / "vocabularies.yaml"
        vocab_file.write_text(
            textwrap.dedent(
                """
                - vocabulary_name: priority
                  value: high
                  description: "Highest urgency"
                - vocabulary_name: priority
                  value: low
                """
            ).strip()
        )
        rows = load_vocabularies(vocab_file)
        assert len(rows) == 2
        high_row = next(r for r in rows if r["value"] == "high")
        assert high_row["description"] == "Highest urgency"

    def test_row_keys_match_schema(self, policies_dir):
        rows = load_vocabularies(policies_dir / "policies" / "vocabularies.yaml")
        schema_fields = {f.name for f in vocabularies_schema().fields}
        for row in rows:
            assert set(row.keys()) == schema_fields


class TestLoadPoliciesFromDir:
    def test_top_level_loader_returns_all_three(self, policies_dir):
        result = load_policies_from_dir(policies_dir)
        assert set(result.keys()) == {"decision_policy", "response_schemas", "vocabularies"}
        assert len(result["decision_policy"]) == 1
        assert len(result["response_schemas"]) == 2
        assert len(result["vocabularies"]) == 5

    def test_missing_policies_dir_returns_empties_not_errors(self, tmp_path):
        result = load_policies_from_dir(tmp_path)
        assert result["decision_policy"] == []
        assert result["response_schemas"] == []
        assert result["vocabularies"] == []
