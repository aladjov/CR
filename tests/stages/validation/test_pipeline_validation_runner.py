"""Tests for PipelineValidationRunner and helper functions."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from customer_retention.stages.validation.pipeline_validation_runner import (
    PipelineValidationConfig,
    PipelineValidationRunner,
    compare_pipeline_outputs,
    run_pipeline_validation,
    validate_feature_transformation,
)


@pytest.fixture
def sample_gold_features():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "event_timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n) * 10 + 50,
        "target": np.random.randint(0, 2, n),
    })
    holdout_mask = np.random.rand(n) < 0.1
    df["original_target"] = pd.NA
    df.loc[holdout_mask, "original_target"] = df.loc[holdout_mask, "target"]
    df.loc[holdout_mask, "target"] = pd.NA
    return df


@pytest.fixture
def sample_features_parquet(tmp_path, sample_gold_features):
    path = tmp_path / "gold_features.parquet"
    sample_gold_features.to_parquet(path, index=False)
    return path


@pytest.fixture
def sample_model(sample_gold_features):
    df = sample_gold_features[sample_gold_features["target"].notna()].copy()
    X = df[["feature_a", "feature_b"]].values
    y = df["target"].astype(int).values
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


class TestPipelineValidationConfig:
    def test_default_values(self):
        config = PipelineValidationConfig()
        assert config.entity_column == "customer_id"
        assert config.target_column == "target"
        assert config.validation_fraction == 0.2

    def test_custom_values(self):
        config = PipelineValidationConfig(entity_column="user_id", target_column="churn")
        assert config.entity_column == "user_id"
        assert config.target_column == "churn"


class TestPipelineValidationRunner:
    def test_creates_with_default_config(self):
        runner = PipelineValidationRunner()
        assert runner.config is not None

    def test_creates_with_custom_config(self):
        config = PipelineValidationConfig(entity_column="user_id")
        runner = PipelineValidationRunner(config=config)
        assert runner.config.entity_column == "user_id"

    def test_load_training_artifacts(self, sample_features_parquet):
        runner = PipelineValidationRunner()
        runner.load_training_artifacts(sample_features_parquet)
        assert runner._training_features is not None
        assert len(runner._training_features) > 0

    def test_load_scoring_artifacts(self, sample_features_parquet):
        runner = PipelineValidationRunner()
        runner.load_scoring_artifacts(sample_features_parquet)
        assert runner._scoring_features is not None

    def test_fluent_interface(self, sample_features_parquet):
        runner = PipelineValidationRunner()
        result = runner.load_training_artifacts(sample_features_parquet).load_scoring_artifacts(sample_features_parquet)
        assert result is runner

    def test_validate_requires_features(self):
        runner = PipelineValidationRunner()
        with pytest.raises(ValueError, match="Must load both"):
            runner.validate()

    def test_validate_identical_features(self, sample_features_parquet):
        runner = PipelineValidationRunner()
        runner.load_training_artifacts(sample_features_parquet)
        runner.load_scoring_artifacts(sample_features_parquet)
        report = runner.validate()
        assert report.passed

    def test_extract_validation_set_with_holdout(self, sample_gold_features):
        runner = PipelineValidationRunner()
        training, holdout = runner.extract_validation_set(sample_gold_features, holdout_column="original_target")
        assert len(training) + len(holdout) == len(sample_gold_features)
        assert holdout["original_target"].notna().all()

    def test_extract_validation_set_random_split(self, sample_gold_features):
        df = sample_gold_features.drop(columns=["original_target"])
        runner = PipelineValidationRunner()
        training, validation = runner.extract_validation_set(df)
        assert len(training) + len(validation) == len(df)


class TestRunPipelineValidation:
    def test_runs_validation(self, sample_features_parquet):
        report = run_pipeline_validation(
            gold_features_path=sample_features_parquet,
            entity_column="customer_id",
            target_column="target",
            verbose=False,
        )
        assert report is not None
        assert report.features_validated

    def test_uses_holdout_column(self, sample_features_parquet):
        report = run_pipeline_validation(
            gold_features_path=sample_features_parquet,
            entity_column="customer_id",
            target_column="target",
            holdout_column="original_target",
            verbose=False,
        )
        assert report is not None

    def test_with_prepare_features_fn(self, tmp_path):
        # Create data without holdout split for simpler testing
        np.random.seed(42)
        df = pd.DataFrame({
            "customer_id": [f"c{i}" for i in range(100)],
            "feature_a": np.random.randn(100),
            "feature_b": np.random.randn(100) * 10,
            "target": np.random.randint(0, 2, 100),
        })
        path = tmp_path / "features.parquet"
        df.to_parquet(path, index=False)
        def simple_transform(data):
            result = data.copy()
            result["feature_a_squared"] = result["feature_a"] ** 2
            return result
        report = run_pipeline_validation(
            gold_features_path=path,
            entity_column="customer_id",
            target_column="target",
            prepare_features_fn=simple_transform,
            verbose=False,
        )
        # When no holdout, random split is used - features_validated should still work
        assert report.features_validated

    def test_with_model_on_holdout_data(self, tmp_path):
        # Create data with proper holdout split that has overlapping entities
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "customer_id": [f"c{i}" for i in range(n)],
            "feature_a": np.random.randn(n),
            "feature_b": np.random.randn(n) * 10,
            "target": np.random.randint(0, 2, n),
        })
        # Create holdout by masking some targets
        holdout_mask = np.random.rand(n) < 0.1
        df["original_target"] = pd.NA
        df.loc[holdout_mask, "original_target"] = df.loc[holdout_mask, "target"]
        df.loc[holdout_mask, "target"] = pd.NA
        path = tmp_path / "features.parquet"
        df.to_parquet(path, index=False)
        # Train on non-holdout data
        train_df = df[df["target"].notna()]
        X = train_df[["feature_a", "feature_b"]].values
        y = train_df["target"].astype(int).values
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        # Validation will use holdout vs training comparison
        report = run_pipeline_validation(
            gold_features_path=path,
            entity_column="customer_id",
            target_column="target",
            holdout_column="original_target",
            model=model,
            feature_columns=["feature_a", "feature_b"],
            verbose=False,
        )
        # With holdout, training and validation have different entities
        # So predictions won't be validated across them (no overlap)
        # But the function should run without errors
        assert report.features_validated

    def test_verbose_output(self, sample_features_parquet, capsys):
        run_pipeline_validation(
            gold_features_path=sample_features_parquet,
            entity_column="customer_id",
            target_column="target",
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "ADVERSARIAL PIPELINE VALIDATION" in captured.out


class TestValidateFeatureTransformation:
    def test_validates_consistent_transform(self):
        df = pd.DataFrame({
            "customer_id": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
        })
        def identity(x): return x.copy()
        report = validate_feature_transformation(df, df, identity, verbose=False)
        assert report.passed

    def test_detects_inconsistent_transform(self):
        df = pd.DataFrame({
            "customer_id": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
        })
        call_count = [0]
        def inconsistent_transform(x):
            result = x.copy()
            if call_count[0] > 0:
                result["value"] = result["value"] + 10
            call_count[0] += 1
            return result
        report = validate_feature_transformation(df, df.copy(), inconsistent_transform, verbose=False)
        assert not report.passed


class TestComparePipelineOutputs:
    def test_compares_identical_outputs(self, tmp_path, sample_gold_features):
        path1 = tmp_path / "output1.parquet"
        path2 = tmp_path / "output2.parquet"
        sample_gold_features.to_parquet(path1, index=False)
        sample_gold_features.to_parquet(path2, index=False)
        report = compare_pipeline_outputs(path1, path2, verbose=False)
        assert report.passed

    def test_detects_different_outputs(self, tmp_path, sample_gold_features):
        path1 = tmp_path / "output1.parquet"
        path2 = tmp_path / "output2.parquet"
        sample_gold_features.to_parquet(path1, index=False)
        modified = sample_gold_features.copy()
        modified["feature_a"] = modified["feature_a"] + 100
        modified.to_parquet(path2, index=False)
        report = compare_pipeline_outputs(path1, path2, verbose=False)
        assert not report.passed

    def test_saves_report(self, tmp_path, sample_gold_features):
        path1 = tmp_path / "output1.parquet"
        path2 = tmp_path / "output2.parquet"
        report_path = tmp_path / "report.yaml"
        sample_gold_features.to_parquet(path1, index=False)
        sample_gold_features.to_parquet(path2, index=False)
        compare_pipeline_outputs(path1, path2, output_report_path=report_path, verbose=False)
        assert report_path.exists()
