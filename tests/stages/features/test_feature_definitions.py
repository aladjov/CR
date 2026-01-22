import pytest
import pandas as pd
from datetime import datetime
from customer_retention.stages.features import (
    FeatureDefinition, FeatureCategory, LeakageRisk, FeatureCatalog
)


class TestFeatureDefinition:
    def test_create_basic_definition(self):
        feature = FeatureDefinition(
            name="tenure_days",
            description="Customer lifetime in days",
            category=FeatureCategory.TEMPORAL,
            derivation="reference_date - created_date",
            source_columns=["created"],
            data_type="float",
            business_meaning="How long customer has been with us"
        )

        assert feature.name == "tenure_days"
        assert feature.category == FeatureCategory.TEMPORAL
        assert feature.leakage_risk == LeakageRisk.LOW  # default

    def test_create_definition_with_all_fields(self):
        feature = FeatureDefinition(
            name="days_since_last_order",
            display_name="Days Since Last Order",
            description="Number of days between reference date and last order",
            category=FeatureCategory.TEMPORAL,
            derivation="reference_date - last_order_date",
            source_columns=["lastorder"],
            data_type="integer",
            value_range=(0, float("inf")),
            business_meaning="Customer recency - higher values indicate dormant customers",
            leakage_risk=LeakageRisk.MEDIUM,
            created_by="test_user"
        )

        assert feature.display_name == "Days Since Last Order"
        assert feature.value_range == (0, float("inf"))
        assert feature.leakage_risk == LeakageRisk.MEDIUM

    def test_definition_has_created_date(self):
        feature = FeatureDefinition(
            name="test_feature",
            description="Test",
            category=FeatureCategory.TEMPORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test meaning"
        )

        assert feature.created_date is not None
        assert isinstance(feature.created_date, datetime)


class TestFeatureCategory:
    def test_all_categories_exist(self):
        expected_categories = [
            "TEMPORAL", "BEHAVIORAL", "MONETARY", "ENGAGEMENT",
            "ADOPTION", "DEMOGRAPHIC", "AGGREGATE", "RATIO",
            "TREND", "INTERACTION"
        ]
        for cat in expected_categories:
            assert hasattr(FeatureCategory, cat)


class TestLeakageRisk:
    def test_all_risk_levels_exist(self):
        assert hasattr(LeakageRisk, "LOW")
        assert hasattr(LeakageRisk, "MEDIUM")
        assert hasattr(LeakageRisk, "HIGH")


class TestFeatureCatalog:
    @pytest.fixture
    def sample_catalog(self):
        catalog = FeatureCatalog()
        catalog.add(FeatureDefinition(
            name="tenure_days",
            description="Customer lifetime",
            category=FeatureCategory.TEMPORAL,
            derivation="ref - created",
            source_columns=["created"],
            data_type="float",
            business_meaning="Tenure"
        ))
        catalog.add(FeatureDefinition(
            name="order_frequency",
            description="Orders per month",
            category=FeatureCategory.BEHAVIORAL,
            derivation="orders / months",
            source_columns=["total_orders", "tenure_months"],
            data_type="float",
            business_meaning="How often customer orders"
        ))
        return catalog

    def test_add_feature(self):
        catalog = FeatureCatalog()
        feature = FeatureDefinition(
            name="test_feature",
            description="Test",
            category=FeatureCategory.TEMPORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test"
        )
        catalog.add(feature)

        assert "test_feature" in catalog.list_names()

    def test_get_feature(self, sample_catalog):
        feature = sample_catalog.get("tenure_days")

        assert feature is not None
        assert feature.name == "tenure_days"
        assert feature.category == FeatureCategory.TEMPORAL

    def test_get_nonexistent_feature(self, sample_catalog):
        feature = sample_catalog.get("nonexistent")
        assert feature is None

    def test_list_by_category(self, sample_catalog):
        temporal = sample_catalog.list_by_category(FeatureCategory.TEMPORAL)
        behavioral = sample_catalog.list_by_category(FeatureCategory.BEHAVIORAL)

        assert len(temporal) == 1
        assert len(behavioral) == 1
        assert temporal[0].name == "tenure_days"
        assert behavioral[0].name == "order_frequency"

    def test_list_names(self, sample_catalog):
        names = sample_catalog.list_names()

        assert "tenure_days" in names
        assert "order_frequency" in names

    def test_remove_feature(self, sample_catalog):
        sample_catalog.remove("tenure_days")

        assert "tenure_days" not in sample_catalog.list_names()

    def test_to_dataframe(self, sample_catalog):
        df = sample_catalog.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "name" in df.columns
        assert "category" in df.columns
        assert "description" in df.columns
        assert "leakage_risk" in df.columns


class TestFeatureCatalogValidation:
    def test_duplicate_name_raises_error(self):
        catalog = FeatureCatalog()
        feature1 = FeatureDefinition(
            name="test_feature",
            description="First",
            category=FeatureCategory.TEMPORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test"
        )
        feature2 = FeatureDefinition(
            name="test_feature",
            description="Second",
            category=FeatureCategory.BEHAVIORAL,
            derivation="test2",
            source_columns=["col2"],
            data_type="float",
            business_meaning="Test2"
        )

        catalog.add(feature1)
        with pytest.raises(ValueError, match="already exists"):
            catalog.add(feature2)

    def test_allow_overwrite(self):
        catalog = FeatureCatalog()
        feature1 = FeatureDefinition(
            name="test_feature",
            description="First",
            category=FeatureCategory.TEMPORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test"
        )
        feature2 = FeatureDefinition(
            name="test_feature",
            description="Second",
            category=FeatureCategory.BEHAVIORAL,
            derivation="test2",
            source_columns=["col2"],
            data_type="float",
            business_meaning="Test2"
        )

        catalog.add(feature1)
        catalog.add(feature2, overwrite=True)

        assert catalog.get("test_feature").description == "Second"


class TestFeatureCatalogFiltering:
    @pytest.fixture
    def catalog_with_risk_levels(self):
        catalog = FeatureCatalog()
        catalog.add(FeatureDefinition(
            name="low_risk",
            description="Low risk feature",
            category=FeatureCategory.TEMPORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test",
            leakage_risk=LeakageRisk.LOW
        ))
        catalog.add(FeatureDefinition(
            name="medium_risk",
            description="Medium risk feature",
            category=FeatureCategory.TEMPORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test",
            leakage_risk=LeakageRisk.MEDIUM
        ))
        catalog.add(FeatureDefinition(
            name="high_risk",
            description="High risk feature",
            category=FeatureCategory.TEMPORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test",
            leakage_risk=LeakageRisk.HIGH
        ))
        return catalog

    def test_list_by_leakage_risk(self, catalog_with_risk_levels):
        high_risk = catalog_with_risk_levels.list_by_leakage_risk(LeakageRisk.HIGH)
        low_risk = catalog_with_risk_levels.list_by_leakage_risk(LeakageRisk.LOW)

        assert len(high_risk) == 1
        assert high_risk[0].name == "high_risk"
        assert len(low_risk) == 1
        assert low_risk[0].name == "low_risk"


class TestFeatureCatalogExport:
    @pytest.fixture
    def sample_catalog(self):
        catalog = FeatureCatalog()
        catalog.add(FeatureDefinition(
            name="tenure_days",
            description="Customer lifetime",
            category=FeatureCategory.TEMPORAL,
            derivation="ref - created",
            source_columns=["created"],
            data_type="float",
            business_meaning="Tenure"
        ))
        return catalog

    def test_to_dict(self, sample_catalog):
        data = sample_catalog.to_dict()

        assert isinstance(data, dict)
        assert "tenure_days" in data
        assert data["tenure_days"]["category"] == "TEMPORAL"

    def test_catalog_size(self, sample_catalog):
        assert len(sample_catalog) == 1

        sample_catalog.add(FeatureDefinition(
            name="new_feature",
            description="New",
            category=FeatureCategory.BEHAVIORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test"
        ))
        assert len(sample_catalog) == 2
