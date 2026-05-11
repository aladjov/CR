from __future__ import annotations

import pytest

from customer_retention.parity.kinds import ApplyOpKind


class TestApplyOpKindClosedSet:
    def test_is_string_enum(self):
        assert isinstance(ApplyOpKind.LIFECYCLE_ENRICH.value, str)
        assert ApplyOpKind.LIFECYCLE_ENRICH == "landing.lifecycle_enrich"

    def test_all_landing_kinds_present(self):
        landing = {k for k in ApplyOpKind if k.value.startswith("landing.")}
        names = {k.name for k in landing}
        expected = {
            "LIFECYCLE_ENRICH",
            "SAMPLE_FILTER",
            "LANDING_FILTER",
            "KEY_RESOLUTION",
            "FEATURE_TIMESTAMP_DERIVE",
            "LABEL_TIMESTAMP_DERIVE",
            "LABEL_AVAILABLE_FLAG",
            "DATETIME_DERIVE",
            "TEMPORAL_LOOKBACK",
            "TIMESTAMP_NORMALIZE",
        }
        assert expected <= names

    def test_bronze_silver_gold_training_kinds(self):
        present = {k.name for k in ApplyOpKind}
        required = {
            "TARGET_DERIVE",
            "BRONZE_AGGREGATE",
            "BRONZE_VALUE_COUNTS",
            "SILVER_TEMPORAL_MERGE",
            "SILVER_DERIVED_FEATURE",
            "SILVER_HOLDOUT_MASK",
            "SILVER_TARGET_LABEL_MAP",
            "GOLD_TRANSFORMATION",
            "GOLD_ENCODING",
            "GOLD_FEATURE_SPEC_GATE",
            "TRAINING_SPLIT",
            "TRAINING_FIT",
            "TRAINING_EVALUATE",
        }
        assert required <= present

    def test_values_are_dotted_stage_prefixed(self):
        for kind in ApplyOpKind:
            assert "." in kind.value, f"{kind.name}={kind.value!r} missing stage prefix"
            stage = kind.value.split(".", 1)[0]
            assert stage in {
                "landing",
                "target_derive",
                "bronze",
                "silver",
                "gold",
                "training",
            }, f"unknown stage prefix in {kind.name}={kind.value!r}"

    def test_enum_is_closed_23_entries(self):
        assert len(list(ApplyOpKind)) == 23

    def test_lookup_by_value_round_trips(self):
        for kind in ApplyOpKind:
            assert ApplyOpKind(kind.value) is kind

    def test_stage_helper_groups_kinds_by_prefix(self):
        landing_kinds = ApplyOpKind.kinds_for_stage("landing")
        assert ApplyOpKind.LIFECYCLE_ENRICH in landing_kinds
        assert ApplyOpKind.BRONZE_AGGREGATE not in landing_kinds
        assert ApplyOpKind.TARGET_DERIVE not in landing_kinds

    def test_stage_helper_unknown_stage_returns_empty(self):
        assert ApplyOpKind.kinds_for_stage("nonexistent") == frozenset()

    def test_stage_property_extracts_prefix(self):
        assert ApplyOpKind.LIFECYCLE_ENRICH.stage == "landing"
        assert ApplyOpKind.BRONZE_AGGREGATE.stage == "bronze"
        assert ApplyOpKind.TARGET_DERIVE.stage == "target_derive"

    @pytest.mark.parametrize(
        "expected_stage,kind",
        [
            ("target_derive", ApplyOpKind.TARGET_DERIVE),
            ("landing_post", ApplyOpKind.LIFECYCLE_ENRICH),
            ("silver", ApplyOpKind.SILVER_DERIVED_FEATURE),
            ("silver_post", ApplyOpKind.SILVER_DERIVED_FEATURE),
        ],
    )
    def test_from_expected_stage_known_mappings(self, expected_stage, kind):
        assert ApplyOpKind.from_expected_stage(expected_stage) is kind

    def test_from_expected_stage_unknown_returns_none(self):
        assert ApplyOpKind.from_expected_stage("not_a_stage") is None
