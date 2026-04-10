"""Tests for the playbook YAML → row loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pyspark", reason="PySpark required for playbook loader tests")

from customer_retention.stages.causal.playbook_loader import load_playbooks_from_dir
from customer_retention.stages.causal.schemas import playbook_catalog_schema, playbook_steps_schema


@pytest.fixture
def playbooks_dir(tmp_path: Path) -> Path:
    """Two synthetic playbook YAMLs covering the happy path + a malformed file."""
    (tmp_path / "alpha.yaml").write_text(
        textwrap.dedent(
            """
            catalog:
              playbook_id: alpha
              version: "1.0.0"
              name: "Alpha Playbook"
              description: "Test playbook for unit tests"
              cost_per_customer_default: 10.5
              expected_uplift_pct_default: 0.25
              outcome_windows_days: [30, 60, 90]
              outcome_definition_version: cancellation_v1
              time_zero_definition: recommendation_created
              grace_period_days: 5
              followup_start_rule: at_time_zero
              followup_end_rule: fixed_from_time_zero
              default_estimand: intention_to_treat
              analysis_population_rule: all_assigned
              active_from: "2026-01-01T00:00:00Z"
              active_to: null
            steps:
              - step_id: a_step1
                step_sequence: 1
                step_name: "First step"
                action_type: csm_call
                automation_level: csm_executed
                cadence_trigger: immediate
                timeout_days: 3
                response_schema_id: structured_call_log_v1
                skip_conditions: {}
                stop_conditions:
                  operator: OR
                  conditions:
                    - field: response_payload.call_response.contact_made
                      op: eq
                      value: true
              - step_id: a_step2
                step_sequence: 2
                step_name: "Second step"
                action_type: csm_email
                automation_level: csm_executed
                cadence_trigger: relative_to_step
                cadence_relative_to: a_step1
                cadence_offset_days: 3
                timeout_days: 5
                response_schema_id: email_engagement_v1
            """
        ).strip()
    )
    (tmp_path / "beta.yaml").write_text(
        textwrap.dedent(
            """
            catalog:
              playbook_id: beta
              version: "2.1.0"
              name: "Beta Playbook"
            steps:
              - step_id: b_step1
                action_type: assessment
                automation_level: csm_executed
            """
        ).strip()
    )
    # A file we want the loader to skip rather than crash on
    (tmp_path / "broken.yaml").write_text("not: a valid playbook structure: [")
    # A file inside a subdirectory must NOT be picked up — that's policy_loader's job
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "decision_policy.yaml").write_text(
        "decision_policy_id: ignore_me\n"
    )
    return tmp_path


class TestLoadPlaybooksFromDir:
    def test_loads_all_top_level_playbooks(self, playbooks_dir):
        catalog_rows, step_rows = load_playbooks_from_dir(playbooks_dir)
        # alpha + beta; broken.yaml is silently skipped
        ids = {row["playbook_id"] for row in catalog_rows}
        assert ids == {"alpha", "beta"}

    def test_subdirectory_yamls_are_excluded(self, playbooks_dir):
        catalog_rows, _ = load_playbooks_from_dir(playbooks_dir)
        ids = {row["playbook_id"] for row in catalog_rows}
        assert "ignore_me" not in ids

    def test_step_count_per_playbook(self, playbooks_dir):
        _, step_rows = load_playbooks_from_dir(playbooks_dir)
        per_pb = {}
        for s in step_rows:
            per_pb.setdefault(s["playbook_id"], []).append(s)
        assert len(per_pb["alpha"]) == 2
        assert len(per_pb["beta"]) == 1

    def test_catalog_field_coercion(self, playbooks_dir):
        catalog_rows, _ = load_playbooks_from_dir(playbooks_dir)
        alpha = next(r for r in catalog_rows if r["playbook_id"] == "alpha")
        assert alpha["version"] == "1.0.0"
        assert alpha["cost_per_customer_default"] == 10.5
        assert alpha["expected_uplift_pct_default"] == 0.25
        assert alpha["outcome_windows_days"] == [30, 60, 90]
        assert alpha["grace_period_days"] == 5

    def test_active_from_parses_to_datetime(self, playbooks_dir):
        from datetime import datetime

        catalog_rows, _ = load_playbooks_from_dir(playbooks_dir)
        alpha = next(r for r in catalog_rows if r["playbook_id"] == "alpha")
        assert isinstance(alpha["active_from"], datetime)
        assert alpha["active_to"] is None

    def test_step_inherits_playbook_id_from_parent_when_missing(self, playbooks_dir):
        _, step_rows = load_playbooks_from_dir(playbooks_dir)
        beta_step = next(s for s in step_rows if s["playbook_id"] == "beta")
        # The step YAML didn't include playbook_id explicitly; the loader
        # should fill it from the parent catalog
        assert beta_step["playbook_id"] == "beta"
        assert beta_step["playbook_version"] == "2.1.0"
        assert beta_step["step_sequence"] == 1  # default to YAML order

    def test_step_skip_and_stop_conditions_serialized_to_json(self, playbooks_dir):
        import json

        _, step_rows = load_playbooks_from_dir(playbooks_dir)
        a_step1 = next(s for s in step_rows if s["step_id"] == "a_step1")
        # skip_conditions={} → "{}" (JSON serialized)
        assert a_step1["skip_conditions"] == "{}"
        # stop_conditions had structured operator/conditions tree → JSON-serialized
        stop = json.loads(a_step1["stop_conditions"])
        assert stop["operator"] == "OR"
        assert stop["conditions"][0]["field"] == "response_payload.call_response.contact_made"

    def test_loaded_rows_match_schema_fields(self, playbooks_dir):
        catalog_rows, step_rows = load_playbooks_from_dir(playbooks_dir)
        catalog_fields = {f.name for f in playbook_catalog_schema().fields}
        step_fields = {f.name for f in playbook_steps_schema().fields}
        for row in catalog_rows:
            assert set(row.keys()) <= catalog_fields, (
                f"playbook_catalog row has unknown fields: {set(row.keys()) - catalog_fields}"
            )
            assert catalog_fields <= set(row.keys()), (
                f"playbook_catalog row missing fields: {catalog_fields - set(row.keys())}"
            )
        for row in step_rows:
            assert set(row.keys()) <= step_fields
            assert step_fields <= set(row.keys())

    def test_empty_directory_returns_empty_lists(self, tmp_path):
        catalog_rows, step_rows = load_playbooks_from_dir(tmp_path)
        assert catalog_rows == []
        assert step_rows == []

    def test_nonexistent_directory_returns_empty_lists(self, tmp_path):
        catalog_rows, step_rows = load_playbooks_from_dir(tmp_path / "does_not_exist")
        assert catalog_rows == []
        assert step_rows == []
