"""Tests for AdversarialScoringValidator.

This validator compares features computed during training vs scoring
for the same holdout entities to catch pipeline inconsistencies.
"""
import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.validation.adversarial_scoring_validator import (
    AdversarialScoringValidator,
    AdversarialValidationResult,
    DriftSeverity,
    FeatureDrift,
)


@pytest.fixture
def sample_gold_features():
    """Gold features with holdout mask."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n) * 10 + 50,
        "feature_c": np.random.choice(["cat1", "cat2", "cat3"], n),
        "target": np.random.randint(0, 2, n),
    })
    holdout_mask = np.random.rand(n) < 0.1
    df["original_target"] = pd.NA
    df.loc[holdout_mask, "original_target"] = df.loc[holdout_mask, "target"]
    df.loc[holdout_mask, "target"] = pd.NA
    return df


@pytest.fixture
def sample_silver_data():
    """Raw silver data before gold transformations."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "raw_a": np.random.randn(n) * 2,
        "raw_b": np.random.randn(n) * 20 + 100,
        "category": np.random.choice(["cat1", "cat2", "cat3"], n),
        "target": np.random.randint(0, 2, n),
    })


class TestAdversarialValidationResult:
    def test_creates_passed_result(self):
        result = AdversarialValidationResult(passed=True, entities_validated=50)
        assert result.passed
        assert result.entities_validated == 50
        assert len(result.feature_drifts) == 0

    def test_creates_failed_result_with_drifts(self):
        drift = FeatureDrift(
            feature_name="feature_a",
            max_absolute_diff=0.5,
            mean_absolute_diff=0.1,
            affected_entities=10,
            severity=DriftSeverity.HIGH,
        )
        result = AdversarialValidationResult(
            passed=False, entities_validated=50, feature_drifts=[drift]
        )
        assert not result.passed
        assert len(result.feature_drifts) == 1

    def test_summary_property(self):
        result = AdversarialValidationResult(passed=True, entities_validated=100)
        assert "PASSED" in result.summary
        assert "100" in result.summary


class TestFeatureDrift:
    def test_creates_with_required_fields(self):
        drift = FeatureDrift(
            feature_name="test_feature",
            max_absolute_diff=1.5,
            mean_absolute_diff=0.3,
            affected_entities=5,
            severity=DriftSeverity.MEDIUM,
        )
        assert drift.feature_name == "test_feature"
        assert drift.severity == DriftSeverity.MEDIUM

    def test_severity_levels(self):
        assert DriftSeverity.LOW.value < DriftSeverity.MEDIUM.value
        assert DriftSeverity.MEDIUM.value < DriftSeverity.HIGH.value
        assert DriftSeverity.HIGH.value < DriftSeverity.CRITICAL.value


class TestAdversarialScoringValidator:
    def test_creates_with_dataframes(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        assert validator.entity_column == "customer_id"

    def test_identifies_holdout_entities(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        holdout_ids = validator.get_holdout_entity_ids()
        assert len(holdout_ids) > 0
        assert len(holdout_ids) < len(sample_gold_features)

    def test_validate_identical_features_passes(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        recomputed = sample_gold_features.copy()
        result = validator.validate_features(recomputed)
        assert result.passed

    def test_validate_detects_numeric_drift(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        recomputed = sample_gold_features.copy()
        recomputed["feature_a"] = recomputed["feature_a"] + 1.0
        result = validator.validate_features(recomputed)
        assert not result.passed
        assert any(d.feature_name == "feature_a" for d in result.feature_drifts)

    def test_validate_detects_categorical_drift(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        recomputed = sample_gold_features.copy()
        recomputed["feature_c"] = "different_value"
        result = validator.validate_features(recomputed)
        assert not result.passed

    def test_validate_ignores_target_columns(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        recomputed = sample_gold_features.copy()
        recomputed["target"] = 999
        recomputed["original_target"] = 999
        result = validator.validate_features(recomputed)
        assert result.passed

    def test_validate_with_tolerance(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
            tolerance=0.1,
        )
        recomputed = sample_gold_features.copy()
        recomputed["feature_a"] = recomputed["feature_a"] + 0.05
        result = validator.validate_features(recomputed)
        assert result.passed

    def test_validate_with_transform_function(self, sample_gold_features, sample_silver_data):
        def mock_transform(df):
            result = df.copy()
            result["feature_a"] = result["raw_a"] / 2
            result["feature_b"] = result["raw_b"] / 2
            result["feature_c"] = result["category"]
            return result

        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        result = validator.validate_with_transform(
            silver_data=sample_silver_data,
            transform_fn=mock_transform,
        )
        assert isinstance(result, AdversarialValidationResult)

    def test_computes_drift_severity_correctly(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        recomputed = sample_gold_features.copy()
        recomputed["feature_a"] = recomputed["feature_a"] + 10.0
        result = validator.validate_features(recomputed)
        drift = next(d for d in result.feature_drifts if d.feature_name == "feature_a")
        assert drift.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)

    def test_handles_missing_entities(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        recomputed = sample_gold_features.iloc[:50].copy()
        result = validator.validate_features(recomputed)
        assert result.entities_validated <= 50

    def test_to_dataframe_returns_drift_details(self, sample_gold_features):
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        recomputed = sample_gold_features.copy()
        recomputed["feature_a"] = recomputed["feature_a"] + 1.0
        result = validator.validate_features(recomputed)
        df = result.to_dataframe()
        assert "feature_name" in df.columns
        assert "severity" in df.columns


class TestAdversarialScoringValidatorEdgeCases:
    def test_handles_empty_holdout(self):
        df = pd.DataFrame({
            "customer_id": ["c1", "c2"],
            "feature_a": [1.0, 2.0],
            "target": [0, 1],
        })
        validator = AdversarialScoringValidator(
            gold_features=df,
            entity_column="customer_id",
            target_column="target",
        )
        holdout_ids = validator.get_holdout_entity_ids()
        assert len(holdout_ids) == 0

    def test_handles_all_holdout(self):
        df = pd.DataFrame({
            "customer_id": ["c1", "c2"],
            "feature_a": [1.0, 2.0],
            "target": [pd.NA, pd.NA],
            "original_target": [0, 1],
        })
        validator = AdversarialScoringValidator(
            gold_features=df,
            entity_column="customer_id",
            target_column="target",
        )
        holdout_ids = validator.get_holdout_entity_ids()
        assert len(holdout_ids) == 2

    def test_handles_nan_values_in_features(self, sample_gold_features):
        sample_gold_features.loc[0, "feature_a"] = np.nan
        validator = AdversarialScoringValidator(
            gold_features=sample_gold_features,
            entity_column="customer_id",
            target_column="target",
        )
        recomputed = sample_gold_features.copy()
        result = validator.validate_features(recomputed)
        assert result.passed
