"""End-to-end tests for the feature store module.

These tests verify the complete feature store workflow including:
- Feature definition and registry
- Publishing features to the store
- Retrieving point-in-time correct features
- Integration with the temporal framework
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from customer_retention.integrations.feature_store import (
    TemporalFeatureDefinition,
    FeatureComputationType,
    TemporalAggregation,
    FeatureRegistry,
    FeatureStoreManager,
)
from customer_retention.integrations.feature_store.definitions import FeatureGroup
from customer_retention.integrations.feature_store.registry import create_standard_churn_features
from customer_retention.stages.temporal import (
    ScenarioDetector,
    UnifiedDataPreparer,
    TimestampConfig,
    TimestampStrategy,
)


class TestTemporalFeatureDefinition:
    """Tests for TemporalFeatureDefinition."""

    def test_passthrough_feature_creation(self):
        """Test creating a simple passthrough feature."""
        feature = TemporalFeatureDefinition(
            name="tenure_months",
            description="Customer tenure in months",
            entity_key="customer_id",
            timestamp_column="feature_timestamp",
            source_columns=["tenure"],
            computation_type=FeatureComputationType.PASSTHROUGH,
        )

        assert feature.name == "tenure_months"
        assert feature.entity_key == "customer_id"
        assert feature.computation_type == FeatureComputationType.PASSTHROUGH

    def test_window_feature_requires_window_days(self):
        """Test that window features require window_days."""
        with pytest.raises(ValueError, match="window_days required"):
            TemporalFeatureDefinition(
                name="total_spend_30d",
                description="Total spend in 30 days",
                entity_key="customer_id",
                computation_type=FeatureComputationType.WINDOW,
                aggregation=TemporalAggregation.SUM,
                # Missing window_days
            )

    def test_window_feature_requires_aggregation(self):
        """Test that window features require aggregation."""
        with pytest.raises(ValueError, match="aggregation required"):
            TemporalFeatureDefinition(
                name="total_spend_30d",
                description="Total spend in 30 days",
                entity_key="customer_id",
                computation_type=FeatureComputationType.WINDOW,
                window_days=30,
                # Missing aggregation
            )

    def test_derived_feature_requires_formula(self):
        """Test that derived features require derivation_formula."""
        with pytest.raises(ValueError, match="derivation_formula required"):
            TemporalFeatureDefinition(
                name="days_since_signup",
                description="Days since signup",
                entity_key="customer_id",
                computation_type=FeatureComputationType.DERIVED,
                # Missing derivation_formula
            )

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        feature = TemporalFeatureDefinition(
            name="total_spend_30d",
            description="Total spend in 30 days",
            entity_key="customer_id",
            source_columns=["amount"],
            computation_type=FeatureComputationType.WINDOW,
            aggregation=TemporalAggregation.SUM,
            window_days=30,
            data_type="float64",
            fill_value=0.0,
            leakage_risk="low",
            tags={"category": "monetary"},
        )

        data = feature.to_dict()
        restored = TemporalFeatureDefinition.from_dict(data)

        assert restored.name == feature.name
        assert restored.window_days == feature.window_days
        assert restored.aggregation == feature.aggregation
        assert restored.tags == feature.tags

    def test_get_feature_ref(self):
        """Test Feast-style feature reference generation."""
        feature = TemporalFeatureDefinition(
            name="tenure_months",
            description="Tenure",
            entity_key="customer_id",
        )

        ref = feature.get_feature_ref("customer_features")
        assert ref == "customer_features:tenure_months"


class TestFeatureRegistry:
    """Tests for FeatureRegistry."""

    def test_register_and_get_feature(self):
        """Test registering and retrieving a feature."""
        registry = FeatureRegistry()
        feature = TemporalFeatureDefinition(
            name="tenure_months",
            description="Tenure in months",
            entity_key="customer_id",
        )

        registry.register(feature)

        retrieved = registry.get("tenure_months")
        assert retrieved is not None
        assert retrieved.name == "tenure_months"

    def test_register_duplicate_raises_error(self):
        """Test that duplicate registration raises error."""
        registry = FeatureRegistry()
        feature = TemporalFeatureDefinition(
            name="tenure_months",
            description="Tenure",
            entity_key="customer_id",
        )

        registry.register(feature)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(feature)

    def test_register_duplicate_with_overwrite(self):
        """Test that overwrite allows duplicate registration."""
        registry = FeatureRegistry()
        feature1 = TemporalFeatureDefinition(
            name="tenure_months",
            description="Original",
            entity_key="customer_id",
        )
        feature2 = TemporalFeatureDefinition(
            name="tenure_months",
            description="Updated",
            entity_key="customer_id",
        )

        registry.register(feature1)
        registry.register(feature2, overwrite=True)

        assert registry.get("tenure_months").description == "Updated"

    def test_list_features(self):
        """Test listing all features."""
        registry = FeatureRegistry()
        for name in ["feature_a", "feature_b", "feature_c"]:
            registry.register(TemporalFeatureDefinition(
                name=name,
                description=f"Feature {name}",
                entity_key="customer_id",
            ))

        names = registry.list_features()
        assert set(names) == {"feature_a", "feature_b", "feature_c"}

    def test_list_by_computation_type(self):
        """Test filtering by computation type."""
        registry = FeatureRegistry()
        registry.register(TemporalFeatureDefinition(
            name="passthrough_feature",
            description="Passthrough",
            entity_key="customer_id",
            computation_type=FeatureComputationType.PASSTHROUGH,
        ))
        registry.register(TemporalFeatureDefinition(
            name="derived_feature",
            description="Derived",
            entity_key="customer_id",
            computation_type=FeatureComputationType.DERIVED,
            derivation_formula="a + b",
        ))

        passthrough = registry.list_by_computation_type(FeatureComputationType.PASSTHROUGH)
        assert len(passthrough) == 1
        assert passthrough[0].name == "passthrough_feature"

    def test_save_and_load(self):
        """Test saving and loading registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FeatureRegistry()
            registry.register(TemporalFeatureDefinition(
                name="test_feature",
                description="Test",
                entity_key="customer_id",
                leakage_risk="medium",
            ))

            path = Path(tmpdir) / "registry.json"
            registry.save(path)

            loaded = FeatureRegistry.load(path)
            assert "test_feature" in loaded
            assert loaded.get("test_feature").leakage_risk == "medium"

    def test_feature_group(self):
        """Test feature groups."""
        registry = FeatureRegistry()
        group = FeatureGroup(
            name="demographic",
            description="Demographic features",
            entity_key="customer_id",
        )

        group.add_feature(TemporalFeatureDefinition(
            name="age",
            description="Customer age",
            entity_key="customer_id",
        ))
        group.add_feature(TemporalFeatureDefinition(
            name="income",
            description="Customer income",
            entity_key="customer_id",
        ))

        registry.register_group(group)

        assert len(registry) == 2
        assert "demographic" in registry.list_groups()

    def test_standard_churn_features(self):
        """Test creating standard churn features."""
        registry = create_standard_churn_features()

        assert len(registry) > 0
        assert "tenure_months" in registry
        assert "demographic" in registry.list_groups()


class TestFeatureStoreManager:
    """Tests for FeatureStoreManager with Feast backend."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100

        base_date = datetime(2024, 1, 1)
        df = pd.DataFrame({
            "entity_id": [f"customer_{i}" for i in range(n)],
            "feature_timestamp": [base_date + timedelta(days=i % 30) for i in range(n)],
            "tenure_months": np.random.randint(1, 60, n),
            "total_spend": np.random.uniform(100, 10000, n),
            "transaction_count": np.random.randint(1, 100, n),
            "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        })

        return df

    @pytest.fixture
    def sample_registry(self):
        """Create sample feature registry."""
        registry = FeatureRegistry()
        registry.register(TemporalFeatureDefinition(
            name="tenure_months",
            description="Tenure in months",
            entity_key="entity_id",
            timestamp_column="feature_timestamp",
            source_columns=["tenure_months"],
            computation_type=FeatureComputationType.PASSTHROUGH,
            data_type="int64",
        ))
        registry.register(TemporalFeatureDefinition(
            name="total_spend",
            description="Total spend",
            entity_key="entity_id",
            timestamp_column="feature_timestamp",
            source_columns=["total_spend"],
            computation_type=FeatureComputationType.PASSTHROUGH,
            data_type="float64",
        ))
        registry.register(TemporalFeatureDefinition(
            name="transaction_count",
            description="Transaction count",
            entity_key="entity_id",
            timestamp_column="feature_timestamp",
            source_columns=["transaction_count"],
            computation_type=FeatureComputationType.PASSTHROUGH,
            data_type="int64",
        ))
        return registry

    def test_create_feast_manager(self):
        """Test creating a Feast-based manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FeatureStoreManager.create(
                backend="feast",
                repo_path=f"{tmpdir}/feature_repo",
                output_path=tmpdir,
            )

            assert manager is not None
            assert isinstance(manager.backend, type(manager.backend))

    def test_publish_features(self, sample_data, sample_registry):
        """Test publishing features to the store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FeatureStoreManager.create(
                backend="feast",
                repo_path=f"{tmpdir}/feature_repo",
                output_path=tmpdir,
            )

            table_name = manager.publish_features(
                df=sample_data,
                registry=sample_registry,
                table_name="customer_features",
                entity_key="entity_id",
                timestamp_column="feature_timestamp",
            )

            assert table_name == "customer_features"
            assert "customer_features" in manager.list_tables()

    def test_get_training_features(self, sample_data, sample_registry):
        """Test retrieving point-in-time training features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FeatureStoreManager.create(
                backend="feast",
                repo_path=f"{tmpdir}/feature_repo",
                output_path=tmpdir,
            )

            # Publish features
            manager.publish_features(
                df=sample_data,
                registry=sample_registry,
                table_name="customer_features",
                entity_key="entity_id",
                timestamp_column="feature_timestamp",
            )

            # Create entity DataFrame
            entity_df = sample_data[["entity_id", "feature_timestamp"]].copy()
            entity_df = entity_df.rename(columns={"feature_timestamp": "event_timestamp"})

            # Get training features
            training_df = manager.get_training_features(
                entity_df=entity_df,
                registry=sample_registry,
                table_name="customer_features",
            )

            assert len(training_df) > 0
            assert "entity_id" in training_df.columns


class TestFeatureStoreWithTemporalFramework:
    """Integration tests for feature store with temporal framework."""

    @pytest.fixture
    def kaggle_style_data(self):
        """Create Kaggle-style data without timestamps."""
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            "customer_id": range(n),
            "tenure": np.random.randint(1, 72, n),
            "monthly_charges": np.random.uniform(20, 120, n),
            "total_charges": np.random.uniform(100, 8000, n),
            "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "churn": np.random.choice([0, 1], n, p=[0.73, 0.27]),
        })

        return df

    def test_full_pipeline_kaggle_to_feature_store(self, kaggle_style_data):
        """Test complete pipeline from Kaggle data to feature store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Step 1: Detect scenario
            detector = ScenarioDetector()
            scenario, config, _ = detector.detect(kaggle_style_data, "churn")

            assert scenario in ["synthetic", "derived"]

            # Step 2: Prepare data with timestamps
            preparer = UnifiedDataPreparer(output_path, config)
            prepared_df = preparer.prepare_from_raw(
                kaggle_style_data,
                target_column="churn",
                entity_column="customer_id",
            )

            assert "feature_timestamp" in prepared_df.columns
            assert "label_timestamp" in prepared_df.columns
            assert "label_available_flag" in prepared_df.columns

            # Step 3: Create feature registry
            registry = FeatureRegistry()
            numeric_cols = ["tenure", "monthly_charges", "total_charges"]
            for col in numeric_cols:
                if col in prepared_df.columns:
                    registry.register(TemporalFeatureDefinition(
                        name=col,
                        description=f"Feature: {col}",
                        entity_key="entity_id",
                        timestamp_column="feature_timestamp",
                        source_columns=[col],
                        computation_type=FeatureComputationType.PASSTHROUGH,
                    ))

            # Step 4: Publish to feature store
            manager = FeatureStoreManager.create(
                backend="feast",
                repo_path=f"{tmpdir}/feature_repo",
                output_path=tmpdir,
            )

            table_name = manager.publish_features(
                df=prepared_df,
                registry=registry,
                table_name="churn_features",
                entity_key="entity_id",
                timestamp_column="feature_timestamp",
            )

            assert table_name == "churn_features"

            # Step 5: Retrieve training features
            entity_df = prepared_df[["entity_id", "feature_timestamp"]].copy()
            entity_df = entity_df.rename(columns={"feature_timestamp": "event_timestamp"})

            training_df = manager.get_training_features(
                entity_df=entity_df.head(50),
                registry=registry,
                table_name="churn_features",
            )

            assert len(training_df) > 0

    def test_production_data_to_feature_store(self):
        """Test pipeline with production data that has timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Create production-style data with timestamps
            np.random.seed(42)
            n = 100
            base_date = datetime(2024, 1, 1)

            df = pd.DataFrame({
                "customer_id": range(n),
                "last_activity_date": [base_date + timedelta(days=np.random.randint(0, 90)) for _ in range(n)],
                "monthly_charges": np.random.uniform(20, 120, n),
                "churn_date": [base_date + timedelta(days=90 + np.random.randint(0, 30)) if np.random.random() > 0.7 else None for _ in range(n)],
                "churned": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            })

            # Detect scenario
            detector = ScenarioDetector()
            scenario, config, _ = detector.detect(df, "churned")

            # Prepare data
            preparer = UnifiedDataPreparer(output_path, config)
            prepared_df = preparer.prepare_from_raw(
                df,
                target_column="churned",
                entity_column="customer_id",
            )

            # Create registry
            registry = FeatureRegistry()
            registry.register(TemporalFeatureDefinition(
                name="monthly_charges",
                description="Monthly charges",
                entity_key="entity_id",
                timestamp_column="feature_timestamp",
                computation_type=FeatureComputationType.PASSTHROUGH,
            ))

            # Publish and retrieve
            manager = FeatureStoreManager.create(
                backend="feast",
                repo_path=f"{tmpdir}/feature_repo",
                output_path=tmpdir,
            )

            manager.publish_features(
                df=prepared_df,
                registry=registry,
                table_name="production_features",
                entity_key="entity_id",
                timestamp_column="feature_timestamp",
            )

            assert "production_features" in manager.list_tables()


class TestFeatureEngineerIntegration:
    """Test integration with FeatureEngineer."""

    def test_feature_engineer_to_registry(self):
        """Test converting FeatureEngineer output to registry."""
        from customer_retention.stages.features.feature_engineer import (
            FeatureEngineer,
            FeatureEngineerConfig,
        )

        # Create sample data
        np.random.seed(42)
        n = 50
        base_date = datetime(2024, 1, 1)

        df = pd.DataFrame({
            "entity_id": range(n),
            "feature_timestamp": [base_date + timedelta(days=i) for i in range(n)],
            "tenure_months": np.random.randint(1, 60, n),
            "total_orders": np.random.randint(1, 100, n),
            "target": np.random.choice([0, 1], n),
        })

        # Configure and run feature engineer
        config = FeatureEngineerConfig(
            generate_temporal=False,
            generate_behavioral=True,
            generate_interaction=False,
            tenure_months_column="tenure_months",
            total_orders_column="total_orders",
            populate_catalog=True,
            id_column="entity_id",
            feature_timestamp_column="feature_timestamp",
        )

        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(df)

        # Convert to feature registry
        registry = engineer.to_feature_registry()

        assert len(registry) > 0
        # Check that generated features are in registry
        for feature_name in result.generated_features:
            assert feature_name in registry
