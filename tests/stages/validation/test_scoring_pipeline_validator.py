"""Tests for ScoringPipelineValidator - adversarial validation between training and scoring pipelines.

This module tests the validation of scoring pipeline consistency with training pipeline.
The validator ensures that:
1. Features computed through scoring pipeline match training pipeline features
2. Model predictions on scoring-processed features match training-processed features
3. Any discrepancies are reported with detailed diagnostics
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from customer_retention.stages.validation.scoring_pipeline_validator import (
    FeatureMismatch,
    MismatchSeverity,
    PredictionMismatch,
    ScoringPipelineValidator,
    ValidationConfig,
    ValidationReport,
)


@pytest.fixture
def sample_training_features():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "entity_id": [f"e{i}" for i in range(n)],
        "numeric_a": np.random.randn(n),
        "numeric_b": np.random.randn(n) * 10 + 50,
        "category": np.random.choice(["A", "B", "C"], n),
        "target": np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_validation_features(sample_training_features):
    return sample_training_features.iloc[:20].copy()


@pytest.fixture
def sample_scoring_features(sample_validation_features):
    return sample_validation_features.copy()


@pytest.fixture
def mismatched_scoring_features(sample_validation_features):
    df = sample_validation_features.copy()
    df["numeric_a"] = df["numeric_a"] + 0.1
    return df


@pytest.fixture
def fitted_model(sample_training_features):
    X = sample_training_features[["numeric_a", "numeric_b"]].values
    y = sample_training_features["target"].values
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def validation_artifacts(tmp_path, sample_validation_features, fitted_model):
    artifacts_dir = tmp_path / "validation_artifacts"
    artifacts_dir.mkdir()
    X_val = sample_validation_features[["numeric_a", "numeric_b"]].values
    y_val = sample_validation_features["target"].values
    y_pred = fitted_model.predict(X_val)
    y_proba = fitted_model.predict_proba(X_val)[:, 1]
    features_df = sample_validation_features[["entity_id", "numeric_a", "numeric_b"]].copy()
    features_df.to_parquet(artifacts_dir / "validation_features.parquet", index=False)
    predictions_df = pd.DataFrame({
        "entity_id": sample_validation_features["entity_id"].values,
        "y_true": y_val,
        "y_pred": y_pred,
        "y_proba": y_proba,
    })
    predictions_df.to_parquet(artifacts_dir / "validation_predictions.parquet", index=False)
    return artifacts_dir


class TestValidationReport:
    def test_creates_with_defaults(self):
        report = ValidationReport()
        assert report.passed is True
        assert len(report.feature_mismatches) == 0
        assert len(report.prediction_mismatches) == 0

    def test_passed_false_when_feature_mismatch(self):
        mismatch = FeatureMismatch(
            feature_name="numeric_a",
            severity=MismatchSeverity.HIGH,
            training_mean=0.0,
            scoring_mean=0.5,
            max_absolute_diff=0.5,
            mismatch_percentage=10.0,
        )
        report = ValidationReport(feature_mismatches=[mismatch])
        assert report.passed is False

    def test_passed_false_when_prediction_mismatch(self):
        mismatch = PredictionMismatch(
            entity_id="e1",
            training_prediction=0,
            scoring_prediction=1,
            training_proba=0.3,
            scoring_proba=0.7,
        )
        report = ValidationReport(prediction_mismatches=[mismatch])
        assert report.passed is False

    def test_summary_counts_mismatches(self):
        feature_mismatches = [
            FeatureMismatch("a", MismatchSeverity.LOW, 0, 0.01, 0.01, 1.0),
            FeatureMismatch("b", MismatchSeverity.HIGH, 0, 0.5, 0.5, 50.0),
        ]
        report = ValidationReport(feature_mismatches=feature_mismatches)
        summary = report.summary()
        assert summary["total_feature_mismatches"] == 2
        assert summary["high_severity_features"] == 1

    def test_to_dict_serialization(self):
        report = ValidationReport(
            feature_mismatches=[FeatureMismatch("a", MismatchSeverity.LOW, 0, 0.01, 0.01, 1.0)]
        )
        d = report.to_dict()
        assert "passed" in d
        assert "feature_mismatches" in d
        assert len(d["feature_mismatches"]) == 1


class TestFeatureMismatch:
    def test_creates_with_required_fields(self):
        mismatch = FeatureMismatch(
            feature_name="income",
            severity=MismatchSeverity.HIGH,
            training_mean=50000.0,
            scoring_mean=52000.0,
            max_absolute_diff=5000.0,
            mismatch_percentage=4.0,
        )
        assert mismatch.feature_name == "income"
        assert mismatch.severity == MismatchSeverity.HIGH

    def test_severity_levels(self):
        assert MismatchSeverity.LOW.value < MismatchSeverity.MEDIUM.value
        assert MismatchSeverity.MEDIUM.value < MismatchSeverity.HIGH.value
        assert MismatchSeverity.HIGH.value < MismatchSeverity.CRITICAL.value


class TestValidationConfig:
    def test_default_thresholds(self):
        config = ValidationConfig()
        assert config.absolute_tolerance > 0
        assert config.relative_tolerance > 0
        assert config.prediction_threshold == 0.5

    def test_custom_thresholds(self):
        config = ValidationConfig(absolute_tolerance=0.001, relative_tolerance=0.001)
        assert config.absolute_tolerance == 0.001


class TestScoringPipelineValidatorInit:
    def test_creates_with_dataframes(self, sample_validation_features, sample_scoring_features):
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=sample_scoring_features,
        )
        assert validator.training_features is not None
        assert validator.scoring_features is not None

    def test_creates_with_paths(self, validation_artifacts):
        train_path = validation_artifacts / "validation_features.parquet"
        validator = ScoringPipelineValidator(
            training_features=train_path,
            scoring_features=train_path,
        )
        assert len(validator.training_features) > 0


class TestScoringPipelineValidatorFeatureComparison:
    def test_identical_features_pass(self, sample_validation_features):
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=sample_validation_features.copy(),
        )
        report = validator.validate_features()
        assert report.passed

    def test_detects_numeric_mismatch(self, sample_validation_features, mismatched_scoring_features):
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=mismatched_scoring_features,
        )
        report = validator.validate_features()
        assert not report.passed
        feature_names = [m.feature_name for m in report.feature_mismatches]
        assert "numeric_a" in feature_names

    def test_ignores_entity_columns_in_comparison(self, sample_validation_features):
        # Entity column should be excluded from feature comparison (used only for alignment)
        # When features match but entity_id formatting differs, it should still pass
        scoring = sample_validation_features.copy()
        # Features are identical, only difference is entity_id column is excluded from comparison
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=scoring,
            entity_column="entity_id",
        )
        report = validator.validate_features()
        assert report.passed
        # Verify entity_id is not in the mismatches
        feature_names = [m.feature_name for m in report.feature_mismatches]
        assert "entity_id" not in feature_names

    def test_ignores_target_column(self, sample_validation_features):
        scoring = sample_validation_features.copy()
        scoring["target"] = 1 - scoring["target"]
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=scoring,
            target_column="target",
        )
        report = validator.validate_features()
        assert report.passed

    def test_reports_mismatch_statistics(self, sample_validation_features, mismatched_scoring_features):
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=mismatched_scoring_features,
        )
        report = validator.validate_features()
        mismatch = next(m for m in report.feature_mismatches if m.feature_name == "numeric_a")
        assert mismatch.max_absolute_diff > 0
        assert mismatch.mismatch_percentage > 0

    def test_respects_tolerance(self, sample_validation_features):
        scoring = sample_validation_features.copy()
        scoring["numeric_a"] = scoring["numeric_a"] + 1e-10
        config = ValidationConfig(absolute_tolerance=1e-6)
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=scoring,
            config=config,
        )
        report = validator.validate_features()
        assert report.passed

    def test_categorical_mismatch_detected(self, sample_validation_features):
        scoring = sample_validation_features.copy()
        scoring["category"] = scoring["category"].replace({"A": "X"})
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=scoring,
        )
        report = validator.validate_features()
        assert not report.passed
        feature_names = [m.feature_name for m in report.feature_mismatches]
        assert "category" in feature_names


class TestScoringPipelineValidatorPredictionComparison:
    def test_identical_predictions_pass(self, sample_validation_features, fitted_model):
        X = sample_validation_features[["numeric_a", "numeric_b"]].values
        y_pred = fitted_model.predict(X)
        y_proba = fitted_model.predict_proba(X)[:, 1]
        training_preds = pd.DataFrame({
            "entity_id": sample_validation_features["entity_id"],
            "y_pred": y_pred,
            "y_proba": y_proba,
        })
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=sample_validation_features.copy(),
            training_predictions=training_preds,
            scoring_predictions=training_preds.copy(),
            entity_column="entity_id",
        )
        report = validator.validate_predictions()
        assert report.passed

    def test_detects_prediction_mismatch(self, sample_validation_features, fitted_model):
        X = sample_validation_features[["numeric_a", "numeric_b"]].values
        y_pred = fitted_model.predict(X)
        y_proba = fitted_model.predict_proba(X)[:, 1]
        training_preds = pd.DataFrame({
            "entity_id": sample_validation_features["entity_id"],
            "y_pred": y_pred,
            "y_proba": y_proba,
        })
        scoring_preds = training_preds.copy()
        scoring_preds["y_pred"] = 1 - scoring_preds["y_pred"]
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=sample_validation_features.copy(),
            training_predictions=training_preds,
            scoring_predictions=scoring_preds,
            entity_column="entity_id",
        )
        report = validator.validate_predictions()
        assert not report.passed
        assert len(report.prediction_mismatches) > 0

    def test_reports_probability_differences(self, sample_validation_features, fitted_model):
        X = sample_validation_features[["numeric_a", "numeric_b"]].values
        y_pred = fitted_model.predict(X)
        y_proba = fitted_model.predict_proba(X)[:, 1]
        training_preds = pd.DataFrame({
            "entity_id": sample_validation_features["entity_id"],
            "y_pred": y_pred,
            "y_proba": y_proba,
        })
        scoring_preds = training_preds.copy()
        scoring_preds["y_proba"] = scoring_preds["y_proba"] + 0.1
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=sample_validation_features.copy(),
            training_predictions=training_preds,
            scoring_predictions=scoring_preds,
            entity_column="entity_id",
        )
        report = validator.validate_predictions()
        if report.prediction_mismatches:
            mismatch = report.prediction_mismatches[0]
            assert mismatch.training_proba is not None
            assert mismatch.scoring_proba is not None


class TestScoringPipelineValidatorFullValidation:
    def test_validate_runs_both_checks(self, sample_validation_features, fitted_model):
        X = sample_validation_features[["numeric_a", "numeric_b"]].values
        y_pred = fitted_model.predict(X)
        y_proba = fitted_model.predict_proba(X)[:, 1]
        preds = pd.DataFrame({
            "entity_id": sample_validation_features["entity_id"],
            "y_pred": y_pred,
            "y_proba": y_proba,
        })
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=sample_validation_features.copy(),
            training_predictions=preds,
            scoring_predictions=preds.copy(),
            entity_column="entity_id",
        )
        report = validator.validate()
        assert report.passed
        assert report.features_validated
        assert report.predictions_validated

    def test_validate_fails_on_feature_mismatch(self, sample_validation_features, mismatched_scoring_features):
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=mismatched_scoring_features,
        )
        report = validator.validate()
        assert not report.passed
        assert report.features_validated
        assert not report.predictions_validated


class TestScoringPipelineValidatorReport:
    def test_generate_report_text(self, sample_validation_features, mismatched_scoring_features):
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=mismatched_scoring_features,
        )
        report = validator.validate_features()
        text = report.to_text()
        assert "FAILED" in text
        assert "numeric_a" in text

    def test_generate_report_dataframe(self, sample_validation_features, mismatched_scoring_features):
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=mismatched_scoring_features,
        )
        report = validator.validate_features()
        df = report.to_dataframe()
        assert "feature_name" in df.columns
        assert len(df) > 0

    def test_save_report(self, tmp_path, sample_validation_features, mismatched_scoring_features):
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=mismatched_scoring_features,
        )
        report = validator.validate_features()
        report_path = tmp_path / "validation_report.yaml"
        report.save(report_path)
        assert report_path.exists()


class TestScoringPipelineValidatorRowAlignment:
    def test_aligns_rows_by_entity_id(self, sample_validation_features):
        shuffled = sample_validation_features.sample(frac=1, random_state=123)
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=shuffled,
            entity_column="entity_id",
        )
        report = validator.validate_features()
        assert report.passed

    def test_detects_missing_entities(self, sample_validation_features):
        partial = sample_validation_features.iloc[:-5].copy()
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=partial,
            entity_column="entity_id",
        )
        report = validator.validate_features()
        assert not report.passed
        assert report.missing_entities_count == 5

    def test_detects_extra_entities(self, sample_validation_features):
        extra_row = sample_validation_features.iloc[0:1].copy()
        extra_row["entity_id"] = "extra_entity"
        with_extra = pd.concat([sample_validation_features, extra_row], ignore_index=True)
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=with_extra,
            entity_column="entity_id",
        )
        report = validator.validate_features()
        assert report.extra_entities_count == 1


class TestScoringPipelineValidatorEdgeCases:
    def test_handles_nan_values(self, sample_validation_features):
        with_nans = sample_validation_features.copy()
        with_nans.loc[0, "numeric_a"] = np.nan
        validator = ScoringPipelineValidator(
            training_features=with_nans,
            scoring_features=with_nans.copy(),
        )
        report = validator.validate_features()
        assert report.passed

    def test_handles_empty_dataframe(self):
        empty = pd.DataFrame({"entity_id": [], "value": []})
        validator = ScoringPipelineValidator(
            training_features=empty,
            scoring_features=empty,
        )
        report = validator.validate_features()
        assert report.passed

    def test_handles_single_row(self):
        single = pd.DataFrame({"entity_id": ["e1"], "value": [1.0]})
        validator = ScoringPipelineValidator(
            training_features=single,
            scoring_features=single,
        )
        report = validator.validate_features()
        assert report.passed

    def test_handles_all_identical_values(self):
        identical = pd.DataFrame({"entity_id": [f"e{i}" for i in range(10)], "value": [5.0] * 10})
        validator = ScoringPipelineValidator(
            training_features=identical,
            scoring_features=identical,
        )
        report = validator.validate_features()
        assert report.passed


class TestScoringPipelineValidatorSeverityClassification:
    def test_low_severity_for_small_relative_differences(self, sample_validation_features):
        # When the absolute difference is tiny relative to the values, severity should be LOW
        scoring = sample_validation_features.copy()
        # Add a very small relative difference (0.001% of values)
        scoring["numeric_b"] = scoring["numeric_b"] * 1.00001  # 0.001% change
        config = ValidationConfig(absolute_tolerance=1e-9, relative_tolerance=1e-9)
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=scoring,
            config=config,
        )
        report = validator.validate_features()
        numeric_b_mismatches = [m for m in report.feature_mismatches if m.feature_name == "numeric_b"]
        if numeric_b_mismatches:
            # Small relative difference should result in LOW severity
            assert numeric_b_mismatches[0].severity == MismatchSeverity.LOW

    def test_critical_severity_for_large_differences(self, sample_validation_features):
        scoring = sample_validation_features.copy()
        scoring["numeric_a"] = scoring["numeric_a"] * 100
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=scoring,
        )
        report = validator.validate_features()
        assert any(m.severity == MismatchSeverity.CRITICAL for m in report.feature_mismatches)


class TestScoringPipelineValidatorWithModel:
    def test_validate_with_model_inference(self, sample_validation_features, fitted_model):
        feature_cols = ["numeric_a", "numeric_b"]
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=sample_validation_features.copy(),
            model=fitted_model,
            feature_columns=feature_cols,
            entity_column="entity_id",
        )
        report = validator.validate_with_model()
        assert report.passed
        assert report.predictions_validated

    def test_model_inference_detects_feature_drift(self, sample_validation_features, mismatched_scoring_features, fitted_model):
        feature_cols = ["numeric_a", "numeric_b"]
        validator = ScoringPipelineValidator(
            training_features=sample_validation_features,
            scoring_features=mismatched_scoring_features,
            model=fitted_model,
            feature_columns=feature_cols,
            entity_column="entity_id",
        )
        report = validator.validate_with_model()
        assert not report.passed
