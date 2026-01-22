from datetime import datetime

import pandas as pd
import pytest

from customer_retention.stages.features import FeatureManifest, FeatureSet, FeatureSetRegistry


class TestFeatureManifest:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "custid": ["C001", "C002", "C003"],
            "tenure_days": [365, 180, 730],
            "order_frequency": [2.0, 1.0, 3.0],
            "email_engagement": [0.5, 0.3, 0.7]
        })

    def test_create_manifest_from_dataframe(self, sample_df):
        manifest = FeatureManifest.from_dataframe(
            df=sample_df,
            feature_columns=["tenure_days", "order_frequency", "email_engagement"],
            entity_column="custid"
        )

        assert manifest is not None
        assert manifest.row_count == 3
        assert manifest.column_count == 3

    def test_manifest_has_id(self, sample_df):
        manifest = FeatureManifest.from_dataframe(
            df=sample_df,
            feature_columns=["tenure_days"],
            entity_column="custid"
        )

        assert manifest.manifest_id is not None
        assert len(manifest.manifest_id) > 0

    def test_manifest_has_timestamp(self, sample_df):
        manifest = FeatureManifest.from_dataframe(
            df=sample_df,
            feature_columns=["tenure_days"],
            entity_column="custid"
        )

        assert manifest.created_at is not None
        assert isinstance(manifest.created_at, datetime)

    def test_manifest_contains_features_list(self, sample_df):
        features = ["tenure_days", "order_frequency"]
        manifest = FeatureManifest.from_dataframe(
            df=sample_df,
            feature_columns=features,
            entity_column="custid"
        )

        assert manifest.features_included == features

    def test_manifest_contains_checksum(self, sample_df):
        manifest = FeatureManifest.from_dataframe(
            df=sample_df,
            feature_columns=["tenure_days"],
            entity_column="custid"
        )

        assert manifest.checksum is not None

    def test_different_data_produces_different_checksum(self, sample_df):
        manifest1 = FeatureManifest.from_dataframe(
            df=sample_df,
            feature_columns=["tenure_days"],
            entity_column="custid"
        )

        sample_df2 = sample_df.copy()
        sample_df2["tenure_days"] = sample_df2["tenure_days"] + 1

        manifest2 = FeatureManifest.from_dataframe(
            df=sample_df2,
            feature_columns=["tenure_days"],
            entity_column="custid"
        )

        assert manifest1.checksum != manifest2.checksum


class TestFeatureSet:
    def test_create_feature_set(self):
        feature_set = FeatureSet(
            name="baseline_features_v1",
            version="1.0.0",
            description="Baseline feature set for model training",
            features_included=["tenure_days", "order_frequency"],
            created_by="test_user"
        )

        assert feature_set.name == "baseline_features_v1"
        assert feature_set.version == "1.0.0"

    def test_feature_set_has_created_at(self):
        feature_set = FeatureSet(
            name="test",
            version="1.0.0",
            description="Test",
            features_included=["f1"]
        )

        assert feature_set.created_at is not None
        assert isinstance(feature_set.created_at, datetime)

    def test_feature_set_with_excluded_features(self):
        feature_set = FeatureSet(
            name="test",
            version="1.0.0",
            description="Test",
            features_included=["f1", "f2"],
            features_excluded=["f3"],
            exclusion_reasons={"f3": "high correlation with f1"}
        )

        assert "f3" in feature_set.features_excluded
        assert "f3" in feature_set.exclusion_reasons

    def test_feature_set_with_parent(self):
        feature_set = FeatureSet(
            name="derived_features",
            version="1.0.0",
            description="Derived from baseline",
            features_included=["f1"],
            parent_feature_set="baseline_features_v1"
        )

        assert feature_set.parent_feature_set == "baseline_features_v1"

    def test_feature_set_to_dict(self):
        feature_set = FeatureSet(
            name="test",
            version="1.0.0",
            description="Test",
            features_included=["f1", "f2"]
        )

        data = feature_set.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["version"] == "1.0.0"
        assert data["features_included"] == ["f1", "f2"]


class TestFeatureSetRegistry:
    @pytest.fixture
    def sample_registry(self):
        registry = FeatureSetRegistry()
        registry.register(FeatureSet(
            name="baseline",
            version="1.0.0",
            description="Baseline features",
            features_included=["f1", "f2"]
        ))
        registry.register(FeatureSet(
            name="baseline",
            version="1.1.0",
            description="Updated baseline features",
            features_included=["f1", "f2", "f3"]
        ))
        return registry

    def test_register_feature_set(self):
        registry = FeatureSetRegistry()
        feature_set = FeatureSet(
            name="test",
            version="1.0.0",
            description="Test",
            features_included=["f1"]
        )

        registry.register(feature_set)

        assert registry.get("test", "1.0.0") is not None

    def test_get_feature_set(self, sample_registry):
        feature_set = sample_registry.get("baseline", "1.0.0")

        assert feature_set is not None
        assert feature_set.version == "1.0.0"

    def test_get_nonexistent_returns_none(self, sample_registry):
        feature_set = sample_registry.get("nonexistent", "1.0.0")
        assert feature_set is None

    def test_list_all_feature_sets(self, sample_registry):
        all_sets = sample_registry.list_all()

        assert len(all_sets) == 2

    def test_list_versions(self, sample_registry):
        versions = sample_registry.list_versions("baseline")

        assert "1.0.0" in versions
        assert "1.1.0" in versions

    def test_get_latest_version(self, sample_registry):
        latest = sample_registry.get_latest("baseline")

        assert latest is not None
        assert latest.version == "1.1.0"

    def test_duplicate_registration_raises_error(self, sample_registry):
        with pytest.raises(ValueError, match="already registered"):
            sample_registry.register(FeatureSet(
                name="baseline",
                version="1.0.0",
                description="Duplicate",
                features_included=["f1"]
            ))


class TestFeatureSetComparison:
    def test_compare_feature_sets(self):
        set1 = FeatureSet(
            name="test",
            version="1.0.0",
            description="Version 1",
            features_included=["f1", "f2", "f3"]
        )
        set2 = FeatureSet(
            name="test",
            version="1.1.0",
            description="Version 2",
            features_included=["f1", "f2", "f4", "f5"]
        )

        registry = FeatureSetRegistry()
        diff = registry.compare(set1, set2)

        assert "added" in diff
        assert "removed" in diff
        assert "f4" in diff["added"] or "f5" in diff["added"]
        assert "f3" in diff["removed"]


class TestManifestToDict:
    def test_manifest_to_dict(self):
        df = pd.DataFrame({
            "custid": ["C001"],
            "f1": [1.0]
        })
        manifest = FeatureManifest.from_dataframe(
            df=df,
            feature_columns=["f1"],
            entity_column="custid"
        )

        data = manifest.to_dict()

        assert isinstance(data, dict)
        assert "manifest_id" in data
        assert "created_at" in data
        assert "row_count" in data
        assert "column_count" in data
        assert "features_included" in data
        assert "checksum" in data
