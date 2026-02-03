"""Tests for the feature store registry module."""


import pytest

from customer_retention.integrations.feature_store.definitions import (
    FeatureComputationType,
    FeatureGroup,
    TemporalAggregation,
    TemporalFeatureDefinition,
)
from customer_retention.integrations.feature_store.registry import (
    FeatureRegistry,
    create_standard_churn_features,
)


def _make_feature(name="feat", entity_key="customer_id", computation_type=None,
                  leakage_risk="low", source_columns=None, **kwargs):
    """Helper to create a TemporalFeatureDefinition with sensible defaults."""
    return TemporalFeatureDefinition(
        name=name,
        description=f"Test feature: {name}",
        entity_key=entity_key,
        computation_type=computation_type or FeatureComputationType.PASSTHROUGH,
        source_columns=source_columns or [],
        leakage_risk=leakage_risk,
        **kwargs,
    )


class TestFeatureRegistry:
    @pytest.fixture
    def registry(self):
        return FeatureRegistry()

    # --- __init__ ---
    def test_init_empty(self, registry):
        assert len(registry) == 0
        assert registry.list_features() == []
        assert registry.list_groups() == []

    def test_init_metadata(self, registry):
        assert "created_at" in registry._metadata
        assert registry._metadata["version"] == "1.0.0"

    # --- register ---
    def test_register_feature(self, registry):
        feature = _make_feature("tenure_months")
        registry.register(feature)
        assert "tenure_months" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self, registry):
        feature = _make_feature("feat1")
        registry.register(feature)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(feature)

    def test_register_duplicate_with_overwrite(self, registry):
        f1 = _make_feature("feat1", data_type="float64")
        f2 = _make_feature("feat1", data_type="int64")
        registry.register(f1)
        registry.register(f2, overwrite=True)
        assert registry.get("feat1").data_type == "int64"

    def test_register_with_group_name_creates_group(self, registry):
        feature = _make_feature("feat1")
        registry.register(feature, group_name="my_group")
        assert "my_group" in registry.list_groups()
        group = registry.get_group("my_group")
        assert group is not None
        assert "feat1" in group.list_feature_names()

    def test_register_with_existing_group(self, registry):
        f1 = _make_feature("feat1")
        f2 = _make_feature("feat2")
        registry.register(f1, group_name="grp")
        registry.register(f2, group_name="grp")
        group = registry.get_group("grp")
        assert len(group.features) == 2

    # --- register_group ---
    def test_register_group(self, registry):
        group = FeatureGroup(
            name="demo",
            description="Demographic features",
            entity_key="customer_id",
        )
        f1 = _make_feature("age")
        f2 = _make_feature("tenure")
        group.add_feature(f1)
        group.add_feature(f2)

        registry.register_group(group)
        assert "demo" in registry.list_groups()
        assert "age" in registry
        assert "tenure" in registry

    def test_register_group_overwrite(self, registry):
        registry.register(_make_feature("existing_feat"))
        group = FeatureGroup(name="grp", description="Group", entity_key="cid")
        group.add_feature(_make_feature("existing_feat"))
        # Without overwrite, this would fail
        registry.register_group(group, overwrite=True)
        assert "existing_feat" in registry

    # --- get ---
    def test_get_existing(self, registry):
        registry.register(_make_feature("f1"))
        assert registry.get("f1") is not None
        assert registry.get("f1").name == "f1"

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    # --- get_group ---
    def test_get_group_existing(self, registry):
        registry.register(_make_feature("f1"), group_name="grp")
        assert registry.get_group("grp") is not None

    def test_get_group_nonexistent(self, registry):
        assert registry.get_group("nonexistent") is None

    # --- remove ---
    def test_remove_existing(self, registry):
        registry.register(_make_feature("f1"))
        assert registry.remove("f1") is True
        assert "f1" not in registry

    def test_remove_nonexistent(self, registry):
        assert registry.remove("nonexistent") is False

    def test_remove_also_removes_from_groups(self, registry):
        registry.register(_make_feature("f1"), group_name="grp")
        registry.remove("f1")
        group = registry.get_group("grp")
        assert "f1" not in group.list_feature_names()

    # --- list_features ---
    def test_list_features(self, registry):
        registry.register(_make_feature("a"))
        registry.register(_make_feature("b"))
        names = registry.list_features()
        assert "a" in names
        assert "b" in names

    # --- list_groups ---
    def test_list_groups(self, registry):
        registry.register(_make_feature("f1"), group_name="grp1")
        registry.register(_make_feature("f2"), group_name="grp2")
        groups = registry.list_groups()
        assert "grp1" in groups
        assert "grp2" in groups

    # --- list_by_computation_type ---
    def test_list_by_computation_type(self, registry):
        registry.register(_make_feature("pass_feat", computation_type=FeatureComputationType.PASSTHROUGH))
        registry.register(_make_feature(
            "window_feat",
            computation_type=FeatureComputationType.WINDOW,
            aggregation=TemporalAggregation.SUM,
            window_days=30,
        ))
        passthrough = registry.list_by_computation_type(FeatureComputationType.PASSTHROUGH)
        assert len(passthrough) == 1
        assert passthrough[0].name == "pass_feat"

    def test_list_by_computation_type_empty(self, registry):
        registry.register(_make_feature("f1"))
        assert registry.list_by_computation_type(FeatureComputationType.RATIO) == []

    # --- list_by_entity ---
    def test_list_by_entity(self, registry):
        registry.register(_make_feature("f1", entity_key="customer_id"))
        registry.register(_make_feature("f2", entity_key="account_id"))
        customer_feats = registry.list_by_entity("customer_id")
        assert len(customer_feats) == 1
        assert customer_feats[0].name == "f1"

    def test_list_by_entity_empty(self, registry):
        registry.register(_make_feature("f1", entity_key="customer_id"))
        assert registry.list_by_entity("nonexistent_key") == []

    # --- list_high_leakage_risk ---
    def test_list_high_leakage_risk(self, registry):
        registry.register(_make_feature("safe", leakage_risk="low"))
        registry.register(_make_feature("risky", leakage_risk="high"))
        high_risk = registry.list_high_leakage_risk()
        assert len(high_risk) == 1
        assert high_risk[0].name == "risky"

    def test_list_high_leakage_risk_none(self, registry):
        registry.register(_make_feature("safe", leakage_risk="low"))
        assert registry.list_high_leakage_risk() == []

    # --- validate_features ---
    def test_validate_features_all_present(self, registry):
        registry.register(_make_feature("f1", source_columns=["col_a", "col_b"]))
        issues = registry.validate_features(["col_a", "col_b", "col_c"])
        assert issues == {}

    def test_validate_features_missing_columns(self, registry):
        registry.register(_make_feature("f1", source_columns=["col_a", "col_missing"]))
        issues = registry.validate_features(["col_a", "col_b"])
        assert "f1" in issues
        assert "col_missing" in issues["f1"]

    def test_validate_features_no_source_columns(self, registry):
        registry.register(_make_feature("f1", source_columns=[]))
        issues = registry.validate_features(["col_a"])
        assert issues == {}

    # --- get_feature_refs ---
    def test_get_feature_refs(self, registry):
        registry.register(_make_feature("feat_a"))
        registry.register(_make_feature("feat_b"))
        refs = registry.get_feature_refs("my_view")
        assert "my_view:feat_a" in refs
        assert "my_view:feat_b" in refs

    def test_get_feature_refs_specific_names(self, registry):
        registry.register(_make_feature("feat_a"))
        registry.register(_make_feature("feat_b"))
        refs = registry.get_feature_refs("view", feature_names=["feat_a"])
        assert len(refs) == 1
        assert refs[0] == "view:feat_a"

    def test_get_feature_refs_nonexistent_name(self, registry):
        registry.register(_make_feature("feat_a"))
        refs = registry.get_feature_refs("view", feature_names=["nonexistent"])
        assert refs == []

    # --- save / load ---
    def test_save_and_load(self, registry, tmp_path):
        registry.register(_make_feature("f1", source_columns=["col"]))
        registry.register(_make_feature("f2"))
        save_path = tmp_path / "registry.json"
        registry.save(save_path)

        loaded = FeatureRegistry.load(save_path)
        assert len(loaded) == 2
        assert "f1" in loaded
        assert "f2" in loaded

    def test_save_and_load_with_groups(self, registry, tmp_path):
        group = FeatureGroup(
            name="grp", description="Group", entity_key="cid",
            source_table="my_table", tags={"env": "prod"},
        )
        f1 = _make_feature("feat1")
        group.add_feature(f1)
        registry.register_group(group)

        save_path = tmp_path / "registry.json"
        registry.save(save_path)

        loaded = FeatureRegistry.load(save_path)
        assert "grp" in loaded.list_groups()
        loaded_group = loaded.get_group("grp")
        assert loaded_group is not None
        assert "feat1" in [f.name for f in loaded_group.features]

    def test_save_creates_parent_dirs(self, tmp_path):
        reg = FeatureRegistry()
        reg.register(_make_feature("f1"))
        deep_path = tmp_path / "a" / "b" / "c" / "registry.json"
        reg.save(deep_path)
        assert deep_path.exists()

    def test_load_preserves_metadata(self, registry, tmp_path):
        save_path = tmp_path / "registry.json"
        registry.save(save_path)
        loaded = FeatureRegistry.load(save_path)
        assert "created_at" in loaded._metadata

    def test_load_groups_link_features(self, tmp_path):
        """Load correctly links group features to registered features."""
        registry = FeatureRegistry()
        f1 = _make_feature("linked_feat")
        registry.register(f1, group_name="linked_grp")

        save_path = tmp_path / "reg.json"
        registry.save(save_path)

        loaded = FeatureRegistry.load(save_path)
        grp = loaded.get_group("linked_grp")
        assert len(grp.features) == 1
        assert grp.features[0].name == "linked_feat"

    # --- to_dict ---
    def test_to_dict(self, registry):
        registry.register(_make_feature("f1"), group_name="grp")
        d = registry.to_dict()
        assert "metadata" in d
        assert "features" in d
        assert "groups" in d
        assert "f1" in d["features"]
        assert "grp" in d["groups"]

    # --- __len__ ---
    def test_len(self, registry):
        assert len(registry) == 0
        registry.register(_make_feature("f1"))
        assert len(registry) == 1

    # --- __contains__ ---
    def test_contains(self, registry):
        assert "f1" not in registry
        registry.register(_make_feature("f1"))
        assert "f1" in registry


class TestCreateStandardChurnFeatures:
    def test_returns_registry(self):
        registry = create_standard_churn_features()
        assert isinstance(registry, FeatureRegistry)

    def test_has_demographic_features(self):
        registry = create_standard_churn_features()
        assert "tenure_months" in registry
        assert "age" in registry

    def test_has_behavioral_features(self):
        registry = create_standard_churn_features()
        assert "total_spend_30d" in registry
        assert "transaction_count_30d" in registry
        assert "avg_transaction_amount" in registry

    def test_has_engagement_features(self):
        registry = create_standard_churn_features()
        assert "days_since_last_activity" in registry

    def test_has_groups(self):
        registry = create_standard_churn_features()
        groups = registry.list_groups()
        assert "demographic" in groups
        assert "behavioral" in groups
        assert "engagement" in groups

    def test_feature_count(self):
        registry = create_standard_churn_features()
        assert len(registry) == 6  # 2 demographic + 3 behavioral + 1 engagement

    def test_save_and_load_standard_features(self, tmp_path):
        """Standard features can be round-tripped through save/load."""
        registry = create_standard_churn_features()
        save_path = tmp_path / "standard.json"
        registry.save(save_path)
        loaded = FeatureRegistry.load(save_path)
        assert len(loaded) == len(registry)
