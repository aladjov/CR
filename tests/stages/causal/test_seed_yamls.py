"""Validation tests for the framework seed YAMLs.

These guard against drift between the seed templates that ship with the
framework and the loaders that read them. Any change to the seed YAMLs or
the loader contract will surface here before it ships.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_retention.stages.causal.policy_loader import (
    load_decision_policy,
    load_response_schemas,
    load_vocabularies,
)

SEED_DIR = (
    Path(__file__).resolve().parents[3]
    / "src" / "customer_retention" / "stages" / "causal" / "seed_yamls"
)


class TestSeedYamlsExist:
    def test_seed_directory_present(self):
        assert SEED_DIR.is_dir(), f"Seed directory not found at {SEED_DIR}"

    @pytest.mark.parametrize(
        "filename",
        ["decision_policy.yaml", "response_schemas.yaml", "vocabularies.yaml"],
    )
    def test_each_seed_file_exists(self, filename):
        assert (SEED_DIR / filename).is_file()


class TestSeedDecisionPolicy:
    def test_loads_to_a_single_row(self):
        rows = load_decision_policy(SEED_DIR / "decision_policy.yaml")
        assert len(rows) == 1

    def test_holdout_default_is_ten_percent(self):
        # Plan §resolved-decisions: holdout 0.10 across the board
        rows = load_decision_policy(SEED_DIR / "decision_policy.yaml")
        fractions = rows[0]["holdout_fractions_by_playbook"]
        assert fractions, "decision_policy.yaml must seed holdout fractions"
        assert all(value == 0.10 for value in fractions.values()), (
            f"All seed holdout fractions must be 0.10, got {fractions}"
        )

    def test_risk_tier_thresholds_seeded(self):
        rows = load_decision_policy(SEED_DIR / "decision_policy.yaml")
        row = rows[0]
        assert row["risk_tier_high_threshold"] == 0.6
        assert row["risk_tier_medium_threshold"] == 0.3

    def test_holdout_fraction_keys_match_existing_playbooks(self):
        # The holdout fractions must cover every playbook the user already
        # has (otherwise we'd silently get no holdout for new playbooks).
        # We assert against the seed coverage, not the runtime user playbooks
        # (those are gitignored), so this test will surface drift inside the
        # seed itself.
        from customer_retention.stages.causal.playbook_loader import load_playbooks_from_dir

        # Use the repo's existing playbooks/ directory if present (gitignored)
        repo_root = Path(__file__).resolve().parents[3]
        playbooks_dir = repo_root / "playbooks"
        if not playbooks_dir.exists():
            pytest.skip("No playbooks directory in this checkout")
        catalog_rows, _ = load_playbooks_from_dir(playbooks_dir)
        if not catalog_rows:
            pytest.skip("Playbooks directory is empty")
        seed_dp = load_decision_policy(SEED_DIR / "decision_policy.yaml")
        seed_keys = set(seed_dp[0]["holdout_fractions_by_playbook"].keys())
        actual_pb_ids = {row["playbook_id"] for row in catalog_rows}
        missing = actual_pb_ids - seed_keys
        assert not missing, (
            f"Seed decision_policy.yaml is missing holdout fractions for "
            f"existing playbooks: {sorted(missing)}"
        )


class TestSeedResponseSchemasReferentialIntegrity:
    def test_every_referenced_schema_is_defined(self):
        # The seed response_schemas.yaml must define every schema_id that
        # the existing playbook YAMLs reference. If a playbook references a
        # schema that doesn't exist in the seed, the framework can't
        # interpret its responses.
        from customer_retention.stages.causal.playbook_loader import load_playbooks_from_dir

        repo_root = Path(__file__).resolve().parents[3]
        playbooks_dir = repo_root / "playbooks"
        if not playbooks_dir.exists():
            pytest.skip("No playbooks directory in this checkout")
        _, step_rows = load_playbooks_from_dir(playbooks_dir)
        if not step_rows:
            pytest.skip("Playbooks directory is empty")

        referenced = {s["response_schema_id"] for s in step_rows if s["response_schema_id"]}
        seed_schemas = load_response_schemas(SEED_DIR / "response_schemas.yaml")
        defined = {r["schema_id"] for r in seed_schemas}

        missing = referenced - defined
        assert not missing, (
            f"Playbook steps reference response schemas that are not defined "
            f"in the seed: {sorted(missing)}"
        )


class TestSeedVocabulariesReferentialIntegrity:
    def test_every_referenced_vocabulary_is_defined(self):
        # Every "vocabulary": "name" reference inside response_schemas.yaml
        # field definitions must resolve to a vocabulary in the seed.
        seed_schemas = load_response_schemas(SEED_DIR / "response_schemas.yaml")
        seed_vocabs = load_vocabularies(SEED_DIR / "vocabularies.yaml")
        defined_vocab_names = {r["vocabulary_name"] for r in seed_vocabs}

        referenced = set()
        for schema_row in seed_schemas:
            fields = json.loads(schema_row["fields"])
            for field in fields:
                if isinstance(field, dict) and "vocabulary" in field:
                    referenced.add(field["vocabulary"])

        missing = referenced - defined_vocab_names
        assert not missing, (
            f"Response schema fields reference vocabularies that are not "
            f"defined in vocabularies.yaml: {sorted(missing)}"
        )

    def test_seed_includes_doc_required_vocabularies(self):
        # Doc §1.3 enumerates the initial vocabulary set the framework should
        # ship with. Anything missing here breaks compatibility with the
        # data model spec.
        rows = load_vocabularies(SEED_DIR / "vocabularies.yaml")
        names = {r["vocabulary_name"] for r in rows}
        required = {
            "sentiment",
            "engagement_level",
            "topics_raised",
            "objections_raised",
            "commitments_obtained",
            "commitment_strength",
        }
        missing = required - names
        assert not missing, f"Seed vocabularies missing doc-required entries: {sorted(missing)}"
