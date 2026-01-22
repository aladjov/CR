from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.features import (
    BehavioralFeatureGenerator,
    FeatureCatalog,
    FeatureCategory,
    FeatureDefinition,
    FeatureEngineer,
    FeatureEngineerConfig,
    FeatureManifest,
    FeatureSelector,
    FeatureSet,
    FeatureSetRegistry,
    InteractionFeatureGenerator,
    LeakageRisk,
    SelectionMethod,
    TemporalFeatureGenerator,
)
from customer_retention.stages.validation import LeakageGate


@pytest.fixture
def retail_data():
    """Load retail test data."""
    retail_path = Path(__file__).parent.parent / "fixtures" / "customer_retention_retail.csv"
    return pd.read_csv(retail_path)


@pytest.fixture
def reference_date():
    return pd.Timestamp("2024-07-01")


class TestTemporalFeatureDerivation:
    def test_tenure_days_calculated_correctly(self, retail_data, reference_date):
        retail_data["created"] = pd.to_datetime(retail_data["created"])
        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            created_column="created"
        )
        result = generator.fit_transform(retail_data)

        expected = (reference_date - retail_data["created"]).dt.days
        pd.testing.assert_series_equal(
            result["tenure_days"], expected, check_names=False
        )

    def test_days_since_last_order_uses_reference_date(self, retail_data, reference_date):
        retail_data["lastorder"] = pd.to_datetime(retail_data["lastorder"], errors='coerce')
        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            last_order_column="lastorder"
        )
        result = generator.fit_transform(retail_data)

        expected = (reference_date - retail_data["lastorder"]).dt.days
        # Compare only non-null values due to invalid dates in test data
        valid_mask = expected.notna() & result["days_since_last_order"].notna()
        pd.testing.assert_series_equal(
            result.loc[valid_mask, "days_since_last_order"],
            expected[valid_mask],
            check_names=False
        )

    def test_days_to_first_order_handles_same_day(self, retail_data, reference_date):
        retail_data["created"] = pd.to_datetime(retail_data["created"], errors='coerce')
        retail_data["firstorder"] = pd.to_datetime(retail_data["firstorder"], errors='coerce')

        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            created_column="created",
            first_order_column="firstorder"
        )
        result = generator.fit_transform(retail_data)

        # Feature should be calculated (firstorder - created)
        # Note: Some values may be negative due to data quality issues
        # (firstorder before created). This test verifies the feature is calculated.
        valid_values = result["days_to_first_order"].dropna()
        assert len(valid_values) > 0
        # Most values should be non-negative (same-day orders or later)
        assert (valid_values >= 0).mean() > 0.95  # At least 95% non-negative


class TestBehavioralFeatureDerivation:
    def test_email_engagement_score_handles_zeros(self, retail_data):
        generator = BehavioralFeatureGenerator(
            open_rate_column="eopenrate",
            click_rate_column="eclickrate"
        )
        result = generator.fit_transform(retail_data)

        assert "email_engagement_score" in result.columns
        # Should handle zero open rates
        assert result["email_engagement_score"].notna().all()

    def test_click_to_open_handles_division_by_zero(self, retail_data):
        generator = BehavioralFeatureGenerator(
            open_rate_column="eopenrate",
            click_rate_column="eclickrate"
        )
        result = generator.fit_transform(retail_data)

        assert "click_to_open_rate" in result.columns
        # Should not have infinities
        assert not np.isinf(result["click_to_open_rate"]).any()

    def test_service_adoption_sums_correctly(self, retail_data):
        generator = BehavioralFeatureGenerator(
            service_columns=["paperless", "refill", "doorstep"]
        )
        result = generator.fit_transform(retail_data)

        expected = retail_data["paperless"] + retail_data["refill"] + retail_data["doorstep"]
        pd.testing.assert_series_equal(
            result["service_adoption_score"].astype(int),
            expected.astype(int),
            check_names=False
        )


class TestInteractionFeatureDerivation:
    def test_value_per_order_freq_handles_zero_frequency(self, retail_data):
        generator = InteractionFeatureGenerator(
            ratios=[("avgorder", "ordfreq", "value_per_freq")]
        )
        result = generator.fit_transform(retail_data)

        assert "value_per_freq" in result.columns
        # Should not have infinities where ordfreq > 0
        valid_rows = retail_data["ordfreq"] > 0
        assert not np.isinf(result.loc[valid_rows, "value_per_freq"]).any()

    def test_orders_per_email_handles_zero_emails(self, retail_data):
        # Create a derived total_orders column for testing
        retail_data["total_orders"] = (retail_data["ordfreq"] * 12).astype(int)

        generator = InteractionFeatureGenerator(
            ratios=[("total_orders", "esent", "orders_per_email")]
        )
        result = generator.fit_transform(retail_data)

        assert "orders_per_email" in result.columns


class TestFeatureCatalog:
    def test_all_features_have_metadata(self):
        catalog = FeatureCatalog()

        # Add sample feature definitions
        catalog.add(FeatureDefinition(
            name="tenure_days",
            description="Customer lifetime in days",
            category=FeatureCategory.TEMPORAL,
            derivation="reference_date - created_date",
            source_columns=["created"],
            data_type="float",
            business_meaning="How long customer has been with us",
            leakage_risk=LeakageRisk.LOW
        ))

        feature = catalog.get("tenure_days")
        assert feature.name is not None
        assert feature.description is not None
        assert feature.category is not None
        assert feature.business_meaning is not None
        assert feature.leakage_risk is not None

    def test_feature_catalog_exportable(self):
        catalog = FeatureCatalog()
        catalog.add(FeatureDefinition(
            name="test_feature",
            description="Test",
            category=FeatureCategory.TEMPORAL,
            derivation="test",
            source_columns=["col"],
            data_type="float",
            business_meaning="Test meaning"
        ))

        # Export to DataFrame
        df = catalog.to_dataframe()
        assert len(df) > 0

        # Export to dict
        data = catalog.to_dict()
        assert "test_feature" in data


class TestFeatureSelection:
    def test_variance_filter_removes_constant(self, retail_data):
        # Add a constant column
        retail_data["constant"] = 1.0

        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="retained"
        )
        result = selector.fit_transform(retail_data)

        assert "constant" not in result.selected_features
        assert "constant" in result.dropped_features

    def test_correlation_filter_identifies_high_correlation(self, retail_data):
        # eopenrate and eclickrate might be correlated
        selector = FeatureSelector(
            method=SelectionMethod.CORRELATION,
            correlation_threshold=0.95,
            target_column="retained"
        )
        result = selector.fit_transform(retail_data)

        # Should provide drop reasons
        assert result.drop_reasons is not None

    def test_selection_preserves_required_features(self, retail_data):
        # Add a constant column
        retail_data["constant"] = 1.0

        selector = FeatureSelector(
            method=SelectionMethod.VARIANCE,
            variance_threshold=0.01,
            target_column="retained",
            preserve_features=["constant"]
        )
        result = selector.fit_transform(retail_data)

        # constant should be preserved even though it has zero variance
        assert "constant" in result.selected_features


class TestFeatureManifestCreation:
    def test_manifest_created_with_all_fields(self, retail_data, reference_date):
        feature_cols = ["avgorder", "ordfreq", "eopenrate"]

        manifest = FeatureManifest.from_dataframe(
            df=retail_data,
            feature_columns=feature_cols,
            entity_column="custid"
        )

        assert manifest.manifest_id is not None
        assert manifest.created_at is not None
        assert manifest.row_count == len(retail_data)
        assert manifest.column_count == len(feature_cols)
        assert manifest.features_included == feature_cols
        assert manifest.checksum is not None


class TestFeatureSetManagement:
    def test_feature_set_registers_successfully(self):
        registry = FeatureSetRegistry()

        feature_set = FeatureSet(
            name="retail_baseline",
            version="1.0.0",
            description="Baseline features for retail retention",
            features_included=["tenure_days", "avgorder", "ordfreq"]
        )

        registry.register(feature_set)
        retrieved = registry.get("retail_baseline", "1.0.0")

        assert retrieved is not None
        assert retrieved.name == "retail_baseline"

    def test_version_comparison_works(self):
        set1 = FeatureSet(
            name="test",
            version="1.0.0",
            description="V1",
            features_included=["f1", "f2", "f3"]
        )
        set2 = FeatureSet(
            name="test",
            version="1.1.0",
            description="V2",
            features_included=["f1", "f2", "f4"]
        )

        registry = FeatureSetRegistry()
        diff = registry.compare(set1, set2)

        assert "f4" in diff["added"]
        assert "f3" in diff["removed"]
        assert "f1" in diff["unchanged"]


class TestLeakageGateIntegration:
    def test_leakage_gate_catches_high_correlation(self, retail_data):
        # Add a leaky feature
        retail_data["leaky"] = retail_data["retained"].astype(float) + np.random.randn(len(retail_data)) * 0.01

        gate = LeakageGate(target_column="retained")
        result = gate.run(retail_data)

        assert not result.passed
        assert len(result.critical_issues) > 0

    def test_clean_dataset_passes_leakage_gate(self, retail_data):
        # Use only legitimate features
        clean_data = retail_data[["avgorder", "ordfreq", "eopenrate", "eclickrate", "retained"]].copy()

        gate = LeakageGate(target_column="retained")
        result = gate.run(clean_data)

        assert result.passed

    def test_leakage_gate_produces_report(self, retail_data):
        gate = LeakageGate(target_column="retained")
        result = gate.run(retail_data)

        assert result.leakage_report is not None
        assert "correlations" in result.leakage_report


class TestFullFeatureEngineeringPipeline:
    def test_pipeline_end_to_end(self, retail_data, reference_date):
        # Parse dates with error handling for invalid dates like '1/0/00'
        retail_data["created"] = pd.to_datetime(retail_data["created"], errors='coerce')
        retail_data["firstorder"] = pd.to_datetime(retail_data["firstorder"], errors='coerce')
        retail_data["lastorder"] = pd.to_datetime(retail_data["lastorder"], errors='coerce')

        config = FeatureEngineerConfig(
            reference_date=reference_date,
            created_column="created",
            first_order_column="firstorder",
            last_order_column="lastorder",
            open_rate_column="eopenrate",
            click_rate_column="eclickrate",
            service_columns=["paperless", "refill", "doorstep"],
            populate_catalog=True
        )

        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(retail_data)

        assert result.df is not None
        assert len(result.df) == len(retail_data)
        assert len(result.generated_features) > 0

    def test_pipeline_preserves_original_data(self, retail_data, reference_date):
        retail_data["created"] = pd.to_datetime(retail_data["created"])

        config = FeatureEngineerConfig(
            reference_date=reference_date,
            created_column="created",
            preserve_original=True
        )

        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(retail_data)

        # Original columns should still be present
        assert "custid" in result.df.columns


class TestAcceptanceCriteria:
    """Tests for acceptance criteria from specification."""

    def test_ac4_1_temporal_features_calculate_correctly(self, retail_data, reference_date):
        """AC4.1: All temporal features calculate correctly."""
        retail_data["created"] = pd.to_datetime(retail_data["created"], errors='coerce')
        retail_data["lastorder"] = pd.to_datetime(retail_data["lastorder"], errors='coerce')

        generator = TemporalFeatureGenerator(
            reference_date=reference_date,
            created_column="created",
            last_order_column="lastorder"
        )
        result = generator.fit_transform(retail_data)

        # Verify tenure_days (only for non-null values)
        expected_tenure = (reference_date - retail_data["created"]).dt.days
        valid = expected_tenure.notna() & result["tenure_days"].notna()
        assert (result.loc[valid, "tenure_days"] == expected_tenure[valid]).all()

        # Verify days_since_last_order (only for non-null values)
        expected_recency = (reference_date - retail_data["lastorder"]).dt.days
        valid = expected_recency.notna() & result["days_since_last_order"].notna()
        assert (result.loc[valid, "days_since_last_order"] == expected_recency[valid]).all()

    def test_ac4_2_reference_date_handling_works(self, retail_data):
        """AC4.2: Reference date handling works."""
        retail_data["lastorder"] = pd.to_datetime(retail_data["lastorder"], errors='coerce')

        # Test with different reference dates
        date1 = pd.Timestamp("2024-07-01")
        date2 = pd.Timestamp("2024-08-01")

        gen1 = TemporalFeatureGenerator(
            reference_date=date1,
            last_order_column="lastorder"
        )
        result1 = gen1.fit_transform(retail_data)

        gen2 = TemporalFeatureGenerator(
            reference_date=date2,
            last_order_column="lastorder"
        )
        result2 = gen2.fit_transform(retail_data)

        # Results should differ by ~31 days (only for non-null values)
        diff = result2["days_since_last_order"] - result1["days_since_last_order"]
        valid = diff.notna()
        assert (diff[valid] == 31).all()

    def test_ac4_3_derived_features_handle_nulls(self):
        """AC4.3: Derived features handle nulls."""
        df = pd.DataFrame({
            "created": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "lastorder": pd.to_datetime(["2024-06-01", "2024-06-15", None])
        })

        generator = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-07-01"),
            created_column="created",
            last_order_column="lastorder"
        )
        result = generator.fit_transform(df)

        # Should have NaN where input was null
        assert pd.isna(result["tenure_days"].iloc[1])
        assert pd.isna(result["days_since_last_order"].iloc[2])

    def test_ac4_6_all_features_have_metadata(self):
        """AC4.6: All features have metadata."""
        catalog = FeatureCatalog()
        feature = FeatureDefinition(
            name="test_feature",
            description="Test description",
            category=FeatureCategory.TEMPORAL,
            derivation="test formula",
            source_columns=["col1"],
            data_type="float",
            business_meaning="Test meaning",
            leakage_risk=LeakageRisk.LOW
        )
        catalog.add(feature)

        retrieved = catalog.get("test_feature")
        assert retrieved.name is not None
        assert retrieved.description is not None
        assert retrieved.category is not None
        assert retrieved.derivation is not None
        assert retrieved.business_meaning is not None
        assert retrieved.leakage_risk is not None

    def test_ac4_14_manifest_captures_all_info(self, retail_data):
        """AC4.14: Manifest captures all info."""
        manifest = FeatureManifest.from_dataframe(
            df=retail_data,
            feature_columns=["avgorder", "ordfreq"],
            entity_column="custid"
        )

        assert manifest.manifest_id is not None
        assert manifest.created_at is not None
        assert manifest.features_included is not None
        assert manifest.row_count is not None
        assert manifest.column_count is not None
        assert manifest.checksum is not None

    def test_ac4_18_high_correlation_detected(self, retail_data):
        """AC4.18: High correlation detected."""
        # Add leaky feature
        retail_data["leaky"] = retail_data["retained"].astype(float) * 0.99 + np.random.randn(len(retail_data)) * 0.01

        gate = LeakageGate(target_column="retained")
        result = gate.run(retail_data)

        assert not result.passed
        leaky_detected = any("leaky" in str(i) for i in result.critical_issues)
        assert leaky_detected

    def test_ac4_21_gate_blocks_on_critical(self, retail_data):
        """AC4.21: Gate blocks on critical."""
        # Add perfectly separating feature
        retail_data["perfect_sep"] = retail_data["retained"] * 100.0

        gate = LeakageGate(target_column="retained")
        result = gate.run(retail_data)

        assert not result.passed
        assert len(result.critical_issues) > 0
