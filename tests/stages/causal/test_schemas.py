"""Tests for the causal-track Delta schemas."""

from __future__ import annotations

import pytest

# Schemas use lazy pyspark imports, but the assertions check the resolved
# StructType — skip the whole module if pyspark isn't installed (CI runners
# without Spark).
pytest.importorskip("pyspark")

from customer_retention.stages.causal import schemas  # noqa: E402

# Every table the data model doc enumerates plus our two derived caches
EXPECTED_TABLES = {
    "playbook_catalog",
    "playbook_steps",
    "response_schemas",
    "vocabularies",
    "archetype_catalog",
    "eligibility_policy",
    "decision_policy",
    "eligibility_snapshot",
    "assignments",
    "actions",
    "outcomes",
    "shap_background",
    "top_shap_drivers",
}


class TestRegistry:
    def test_all_expected_tables_registered(self):
        assert set(schemas.ALL_SCHEMAS) == EXPECTED_TABLES

    def test_get_schema_returns_struct_for_each_table(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import StructType

        for name in EXPECTED_TABLES:
            struct = schemas.get_schema(name)
            assert isinstance(struct, StructType), f"{name} did not return StructType"
            assert len(struct.fields) > 0

    def test_get_schema_unknown_table_raises_keyerror(self):
        with pytest.raises(KeyError) as exc_info:
            schemas.get_schema("nonexistent_table_name")
        # Error message should list the known tables for discoverability
        assert "playbook_catalog" in str(exc_info.value)


class TestPlaybookCatalogSchema:
    def test_required_keys_present(self):
        struct = schemas.playbook_catalog_schema()
        names = {f.name for f in struct.fields}
        # Doc §1.1 mandatory primary identity
        assert {"playbook_id", "version", "name"} <= names
        # Target-trial fields the doc requires for §2.2 alignment
        assert {
            "time_zero_definition",
            "grace_period_days",
            "followup_start_rule",
            "followup_end_rule",
            "default_estimand",
            "analysis_population_rule",
        } <= names

    def test_id_and_version_are_non_nullable(self):
        struct = schemas.playbook_catalog_schema()
        nullability = {f.name: f.nullable for f in struct.fields}
        assert nullability["playbook_id"] is False
        assert nullability["version"] is False
        assert nullability["name"] is False


class TestArchetypeCatalogSchema:
    def test_dual_centroid_storage(self):
        # Plan §key-design-choices: archetype_catalog stores BOTH SHAP-space
        # centroid (for stability tracking) and raw-feature centroid (for
        # runtime nearest-neighbor archetype assignment in cell 5)
        struct = schemas.archetype_catalog_schema()
        names = {f.name for f in struct.fields}
        assert "centroid_vector" in names  # SHAP-space
        assert "centroid_vector_raw" in names  # raw-feature
        assert "centroid_feature_order" in names  # column order for the raw centroid

    def test_status_lifecycle_field_required(self):
        struct = schemas.archetype_catalog_schema()
        nullability = {f.name: f.nullable for f in struct.fields}
        assert nullability["status"] is False


class TestEligibilityPolicySchema:
    def test_archetype_ids_is_array(self):
        # Doc §1.5: many-to-many — one policy can target multiple archetypes
        pytest.importorskip("pyspark")
        from pyspark.sql.types import ArrayType, StringType

        struct = schemas.eligibility_policy_schema()
        archetype_field = next(f for f in struct.fields if f.name == "archetype_ids")
        assert isinstance(archetype_field.dataType, ArrayType)
        assert isinstance(archetype_field.dataType.elementType, StringType)

    def test_rules_have_both_json_and_sql_renderings(self):
        struct = schemas.eligibility_policy_schema()
        names = {f.name for f in struct.fields}
        assert "eligibility_rules" in names  # JSON predicate tree
        assert "eligibility_rules_sql" in names  # rendered for the dashboard


class TestDecisionPolicySchema:
    def test_per_playbook_maps(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import DoubleType, IntegerType, MapType

        struct = schemas.decision_policy_schema()
        by_name = {f.name: f for f in struct.fields}
        assert isinstance(by_name["holdout_fractions_by_playbook"].dataType, MapType)
        assert isinstance(by_name["holdout_fractions_by_playbook"].dataType.valueType, DoubleType)
        assert isinstance(by_name["capacity_by_playbook"].dataType, MapType)
        assert isinstance(by_name["capacity_by_playbook"].dataType.valueType, IntegerType)
        assert isinstance(by_name["cooldown_days_by_playbook"].dataType, MapType)

    def test_risk_tier_thresholds_versioned_with_policy(self):
        struct = schemas.decision_policy_schema()
        names = {f.name for f in struct.fields}
        assert "risk_tier_high_threshold" in names
        assert "risk_tier_medium_threshold" in names


class TestEligibilitySnapshotSchema:
    def test_four_way_anchor_present(self):
        # Plan §design-choices + doc §4 load-bearing fields:
        # every snapshot row must carry the four-way definition anchor tuple
        struct = schemas.eligibility_snapshot_schema()
        names = {f.name for f in struct.fields}
        anchors = {
            "playbook_id",
            "playbook_version",
            "archetype_id",
            "archetype_version",
            "eligibility_policy_id",
            "eligibility_policy_version",
            "decision_policy_id",
        }
        assert anchors <= names

    def test_id_and_run_id_non_nullable(self):
        struct = schemas.eligibility_snapshot_schema()
        nullability = {f.name: f.nullable for f in struct.fields}
        assert nullability["eligibility_id"] is False
        assert nullability["scoring_run_id"] is False
        assert nullability["entity_id"] is False
        # churn_probability is the frozen outcome risk — must be present
        assert nullability["churn_probability"] is False
        assert nullability["recommended"] is False
