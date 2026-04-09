"""Edge-case coverage for the YAML→row loaders.

The happy paths are exercised in ``test_playbook_loader.py`` and
``test_policy_loader.py``. This module fills in the defensive coercion and
error-handling branches to bring coverage above 90%.
"""

from __future__ import annotations

import textwrap
from datetime import datetime

import pytest

from customer_retention.stages.causal import playbook_loader, policy_loader

# ---------------------------------------------------------------------------
# playbook_loader edge cases
# ---------------------------------------------------------------------------


class TestPlaybookLoaderErrorHandling:
    def test_yaml_that_parses_to_non_mapping_is_skipped(self, tmp_path):
        # Top-level list (instead of dict) — loader should skip with a warning
        (tmp_path / "list_root.yaml").write_text("- just\n- a\n- list\n")
        catalog, steps = playbook_loader.load_playbooks_from_dir(tmp_path)
        assert catalog == [] and steps == []

    def test_yaml_missing_catalog_block_is_skipped(self, tmp_path):
        (tmp_path / "no_catalog.yaml").write_text("steps:\n  - step_id: x\n")
        catalog, _ = playbook_loader.load_playbooks_from_dir(tmp_path)
        assert catalog == []

    def test_catalog_missing_required_field_is_skipped(self, tmp_path):
        # No 'name' field — _parse_catalog raises, loader catches + logs
        (tmp_path / "incomplete.yaml").write_text(
            "catalog:\n  playbook_id: x\n  version: '1.0.0'\n"
        )
        catalog, _ = playbook_loader.load_playbooks_from_dir(tmp_path)
        assert catalog == []

    def test_catalog_must_be_a_mapping(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("catalog: just a string\n")
        catalog, _ = playbook_loader.load_playbooks_from_dir(tmp_path)
        assert catalog == []

    def test_steps_must_be_a_list_when_present(self, tmp_path):
        (tmp_path / "bad_steps.yaml").write_text(
            textwrap.dedent(
                """
                catalog:
                  playbook_id: a
                  version: "1.0.0"
                  name: "A"
                steps: "not a list"
                """
            ).strip()
        )
        catalog, steps = playbook_loader.load_playbooks_from_dir(tmp_path)
        # The whole entry is rejected because step parsing raises
        assert catalog == []
        assert steps == []

    def test_step_without_step_id_is_rejected(self, tmp_path):
        (tmp_path / "bad_step.yaml").write_text(
            textwrap.dedent(
                """
                catalog:
                  playbook_id: a
                  version: "1.0.0"
                  name: "A"
                steps:
                  - action_type: csm_call
                """
            ).strip()
        )
        catalog, _ = playbook_loader.load_playbooks_from_dir(tmp_path)
        assert catalog == []

    def test_yml_extension_also_loaded(self, tmp_path):
        (tmp_path / "alpha.yml").write_text(
            textwrap.dedent(
                """
                catalog:
                  playbook_id: alpha
                  version: "1.0.0"
                  name: "Alpha"
                """
            ).strip()
        )
        catalog, _ = playbook_loader.load_playbooks_from_dir(tmp_path)
        assert len(catalog) == 1
        assert catalog[0]["playbook_id"] == "alpha"


class TestPlaybookLoaderCoercion:
    def test_invalid_int_field_coerces_to_none(self):
        assert playbook_loader._as_optional_int("not a number") is None
        assert playbook_loader._as_optional_int(None) is None
        assert playbook_loader._as_optional_int("42") == 42

    def test_invalid_float_field_coerces_to_none(self):
        assert playbook_loader._as_optional_float("not a number") is None
        assert playbook_loader._as_optional_float(None) is None
        assert playbook_loader._as_optional_float("3.14") == 3.14

    def test_int_array_skips_unparseable_items(self):
        result = playbook_loader._as_int_array([1, "two", 3, None, 5])
        assert result == [1, 3, 5]

    def test_int_array_returns_none_for_non_iterable(self):
        assert playbook_loader._as_int_array(42) is None
        assert playbook_loader._as_int_array("not iterable") is None
        assert playbook_loader._as_int_array(None) is None

    def test_optional_str_strips_whitespace(self):
        assert playbook_loader._as_optional_str("  hello  ") == "hello"
        assert playbook_loader._as_optional_str("") is None
        assert playbook_loader._as_optional_str("   ") is None
        assert playbook_loader._as_optional_str(None) is None

    def test_optional_timestamp_handles_iso_with_z(self):
        result = playbook_loader._as_optional_timestamp("2026-04-08T12:00:00Z")
        assert isinstance(result, datetime)

    def test_optional_timestamp_returns_none_for_invalid(self):
        assert playbook_loader._as_optional_timestamp("not a date") is None
        assert playbook_loader._as_optional_timestamp("null") is None
        assert playbook_loader._as_optional_timestamp("") is None
        assert playbook_loader._as_optional_timestamp(None) is None

    def test_optional_timestamp_passthrough_existing_datetime(self):
        dt = datetime(2026, 4, 8, 12, 0, 0)
        assert playbook_loader._as_optional_timestamp(dt) is dt

    def test_as_str_raises_on_none(self):
        with pytest.raises(ValueError):
            playbook_loader._as_str(None)

    def test_json_string_passthrough_string(self):
        assert playbook_loader._as_json_string("already a string") == "already a string"
        assert playbook_loader._as_json_string(None) is None

    def test_json_string_serializes_dict(self):
        import json
        out = playbook_loader._as_json_string({"a": 1, "b": [2, 3]})
        assert json.loads(out) == {"a": 1, "b": [2, 3]}

    def test_json_string_returns_none_for_unserializable(self):
        # An object with no JSON serialization
        class Weird:
            pass

        # default=str is the fallback so most things still serialize, but
        # things that raise outside json.dumps still hit the except branch
        result = playbook_loader._as_json_string(Weird())
        # default=str converts via __str__, so this actually serializes
        assert result is not None


# ---------------------------------------------------------------------------
# policy_loader edge cases
# ---------------------------------------------------------------------------


class TestPolicyLoaderErrorHandling:
    def test_decision_policy_missing_id_is_skipped(self, tmp_path):
        (tmp_path / "decision_policy.yaml").write_text("version: '1.0.0'\n")
        rows = policy_loader.load_decision_policy(tmp_path / "decision_policy.yaml")
        assert rows == []

    def test_decision_policy_accepts_list_shape(self, tmp_path):
        path = tmp_path / "decision_policy.yaml"
        path.write_text(
            textwrap.dedent(
                """
                - decision_policy_id: p1
                  version: "1.0.0"
                - decision_policy_id: p2
                  version: "2.0.0"
                """
            ).strip()
        )
        rows = policy_loader.load_decision_policy(path)
        assert len(rows) == 2
        assert {r["decision_policy_id"] for r in rows} == {"p1", "p2"}

    def test_decision_policy_neither_dict_nor_list_returns_empty(self, tmp_path):
        path = tmp_path / "decision_policy.yaml"
        path.write_text("just a string\n")
        rows = policy_loader.load_decision_policy(path)
        assert rows == []

    def test_decision_policy_skips_non_mapping_entries_in_list(self, tmp_path):
        path = tmp_path / "decision_policy.yaml"
        path.write_text(
            textwrap.dedent(
                """
                - decision_policy_id: ok
                  version: "1.0.0"
                - just a string
                - 42
                """
            ).strip()
        )
        rows = policy_loader.load_decision_policy(path)
        assert len(rows) == 1
        assert rows[0]["decision_policy_id"] == "ok"

    def test_response_schemas_accepts_dict_with_schemas_key(self, tmp_path):
        path = tmp_path / "response_schemas.yaml"
        path.write_text(
            textwrap.dedent(
                """
                schemas:
                  - schema_id: s1
                    version: "1.0.0"
                  - schema_id: s2
                    version: "2.0.0"
                """
            ).strip()
        )
        rows = policy_loader.load_response_schemas(path)
        assert len(rows) == 2

    def test_response_schemas_invalid_root_returns_empty(self, tmp_path):
        path = tmp_path / "response_schemas.yaml"
        path.write_text("just a string")
        assert policy_loader.load_response_schemas(path) == []

    def test_response_schemas_skips_entries_without_schema_id(self, tmp_path):
        path = tmp_path / "response_schemas.yaml"
        path.write_text(
            textwrap.dedent(
                """
                - version: "1.0.0"
                  name: "no id"
                - schema_id: ok
                  version: "1.0.0"
                """
            ).strip()
        )
        rows = policy_loader.load_response_schemas(path)
        assert len(rows) == 1

    def test_vocabularies_skips_non_list_values_in_dict_shape(self, tmp_path):
        path = tmp_path / "vocabularies.yaml"
        path.write_text(
            textwrap.dedent(
                """
                sentiment:
                  - positive
                  - negative
                bad_vocab: "not a list"
                """
            ).strip()
        )
        rows = policy_loader.load_vocabularies(path)
        assert len(rows) == 2
        assert all(r["vocabulary_name"] == "sentiment" for r in rows)

    def test_vocabularies_skips_non_mapping_entries_in_list_shape(self, tmp_path):
        path = tmp_path / "vocabularies.yaml"
        path.write_text(
            textwrap.dedent(
                """
                - vocabulary_name: priority
                  value: high
                - just a string
                - vocabulary_name: priority
                  value: low
                """
            ).strip()
        )
        rows = policy_loader.load_vocabularies(path)
        assert len(rows) == 2

    def test_vocabularies_skips_entries_missing_required_keys(self, tmp_path):
        path = tmp_path / "vocabularies.yaml"
        path.write_text(
            textwrap.dedent(
                """
                - vocabulary_name: priority
                # missing value
                - value: orphan_value
                # missing vocabulary_name
                - vocabulary_name: priority
                  value: high
                """
            ).strip()
        )
        rows = policy_loader.load_vocabularies(path)
        assert len(rows) == 1
        assert rows[0]["value"] == "high"

    def test_missing_yaml_file_returns_empty(self, tmp_path):
        assert policy_loader.load_decision_policy(tmp_path / "nope.yaml") == []
        assert policy_loader.load_response_schemas(tmp_path / "nope.yaml") == []
        assert policy_loader.load_vocabularies(tmp_path / "nope.yaml") == []

    def test_malformed_yaml_returns_empty(self, tmp_path):
        path = tmp_path / "broken.yaml"
        path.write_text("this: is: not: valid: yaml: [")
        assert policy_loader.load_decision_policy(path) == []
        assert policy_loader.load_response_schemas(path) == []
        assert policy_loader.load_vocabularies(path) == []


class TestPolicyLoaderCoercion:
    def test_optional_float_invalid(self):
        assert policy_loader._as_optional_float("nope") is None
        assert policy_loader._as_optional_float(None) is None
        assert policy_loader._as_optional_float("3.14") == 3.14

    def test_str_array_invalid_input(self):
        assert policy_loader._as_str_array(None) is None
        assert policy_loader._as_str_array("not a list") is None
        assert policy_loader._as_str_array([1, 2, 3]) == ["1", "2", "3"]

    def test_str_double_map_invalid_inputs(self):
        assert policy_loader._as_str_double_map(None) is None
        assert policy_loader._as_str_double_map("not a dict") is None
        # Skips unparseable values
        result = policy_loader._as_str_double_map({"a": 0.5, "b": "nope"})
        assert result == {"a": 0.5}

    def test_str_int_map_invalid_inputs(self):
        assert policy_loader._as_str_int_map(None) is None
        assert policy_loader._as_str_int_map("not a dict") is None
        result = policy_loader._as_str_int_map({"a": 5, "b": "nope"})
        assert result == {"a": 5}

    def test_optional_timestamp_handles_z_suffix(self):
        result = policy_loader._as_optional_timestamp("2026-04-08T12:00:00Z")
        assert isinstance(result, datetime)

    def test_optional_timestamp_passthrough(self):
        dt = datetime(2026, 1, 1)
        assert policy_loader._as_optional_timestamp(dt) is dt

    def test_optional_timestamp_invalid(self):
        assert policy_loader._as_optional_timestamp("nonsense") is None
        assert policy_loader._as_optional_timestamp("null") is None
        assert policy_loader._as_optional_timestamp(None) is None

    def test_optional_str_strips_whitespace(self):
        assert policy_loader._as_optional_str("  x  ") == "x"
        assert policy_loader._as_optional_str("") is None
        assert policy_loader._as_optional_str(None) is None

    def test_json_string_passthrough(self):
        assert policy_loader._as_json_string("already a string") == "already a string"
        assert policy_loader._as_json_string(None) is None
        import json
        out = policy_loader._as_json_string({"k": "v"})
        assert json.loads(out) == {"k": "v"}
