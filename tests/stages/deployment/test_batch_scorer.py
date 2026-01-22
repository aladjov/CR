from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from customer_retention.stages.deployment import BatchScorer, RiskSegment, ScoringConfig


@pytest.fixture
def sample_features():
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": [f"CUST{i:04d}" for i in range(100)],
        "avgorder": np.random.uniform(20, 200, 100),
        "ordfreq": np.random.uniform(1, 30, 100),
        "eopenrate": np.random.uniform(0, 1, 100),
        "eclickrate": np.random.uniform(0, 0.5, 100),
        "paperless": np.random.choice([0, 1], 100),
        "refill": np.random.choice([0, 1], 100),
    })


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = np.random.rand(100, 6)
    y = np.random.choice([0, 1], 100)
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    return model


class TestScoringConfig:
    def test_config_has_default_values(self):
        config = ScoringConfig(model_name="churn_model")
        assert config.model_name == "churn_model"
        assert config.model_stage == "Production"
        assert config.batch_size == 10000

    def test_config_accepts_custom_values(self):
        config = ScoringConfig(
            model_name="churn_model",
            model_stage="Staging",
            batch_size=5000,
            parallelism=4
        )
        assert config.model_stage == "Staging"
        assert config.batch_size == 5000
        assert config.parallelism == 4


class TestBatchScorerPipeline:
    def test_pipeline_runs_without_error(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        assert result is not None

    def test_output_has_correct_schema(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        assert "customer_id" in result.predictions.columns
        assert "churn_probability" in result.predictions.columns
        assert "risk_segment" in result.predictions.columns
        assert "predicted_churn" in result.predictions.columns
        assert "score_timestamp" in result.predictions.columns

    def test_all_customers_scored(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        assert len(result.predictions) == len(sample_features)

    def test_probabilities_in_valid_range(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        assert result.predictions["churn_probability"].min() >= 0
        assert result.predictions["churn_probability"].max() <= 1


class TestRiskSegmentation:
    def test_risk_segment_enum_values(self):
        assert RiskSegment.CRITICAL.value == "Critical"
        assert RiskSegment.HIGH.value == "High"
        assert RiskSegment.MEDIUM.value == "Medium"
        assert RiskSegment.LOW.value == "Low"

    def test_assigns_risk_segments(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        segments = result.predictions["risk_segment"].unique()
        valid_segments = [s.value for s in RiskSegment]
        for seg in segments:
            assert seg in valid_segments

    def test_critical_segment_for_high_probability(self, trained_model):
        scorer = BatchScorer(model=trained_model, threshold=0.5)
        segment = scorer._assign_risk_segment(0.85)
        assert segment == RiskSegment.CRITICAL.value

    def test_low_segment_for_low_probability(self, trained_model):
        scorer = BatchScorer(model=trained_model, threshold=0.5)
        segment = scorer._assign_risk_segment(0.15)
        assert segment == RiskSegment.LOW.value


class TestScoringWithScaler:
    def test_applies_scaler_when_provided(self, sample_features, trained_model):
        mock_scaler = MagicMock()
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        mock_scaler.transform.return_value = sample_features[feature_cols].values
        scorer = BatchScorer(model=trained_model, scaler=mock_scaler)
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        mock_scaler.transform.assert_called()

    def test_works_without_scaler(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model, scaler=None)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        assert result is not None


class TestScoringMetadata:
    def test_includes_model_version(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model, model_version="3")
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        assert "model_version" in result.predictions.columns or result.model_version == "3"

    def test_includes_timestamp(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        assert "score_timestamp" in result.predictions.columns

    def test_scoring_result_contains_stats(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        assert result.total_scored > 0
        assert result.scoring_duration_seconds >= 0


class TestBatchProcessing:
    def test_processes_in_batches(self, trained_model):
        np.random.seed(42)
        large_features = pd.DataFrame({
            "customer_id": [f"CUST{i:06d}" for i in range(25000)],
            "avgorder": np.random.uniform(20, 200, 25000),
            "ordfreq": np.random.uniform(1, 30, 25000),
            "eopenrate": np.random.uniform(0, 1, 25000),
            "eclickrate": np.random.uniform(0, 0.5, 25000),
            "paperless": np.random.choice([0, 1], 25000),
            "refill": np.random.choice([0, 1], 25000),
        })
        scorer = BatchScorer(model=trained_model, batch_size=5000)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(large_features, feature_columns=feature_cols, id_column="customer_id")
        assert len(result.predictions) == 25000


class TestErrorHandling:
    def test_handles_missing_features_gracefully(self, trained_model):
        incomplete_features = pd.DataFrame({
            "customer_id": ["CUST001", "CUST002"],
            "avgorder": [50.0, 75.0],
        })
        scorer = BatchScorer(model=trained_model)
        with pytest.raises(ValueError):
            scorer.score(incomplete_features, feature_columns=["avgorder", "ordfreq"], id_column="customer_id")

    def test_handles_null_values(self, trained_model):
        features_with_nulls = pd.DataFrame({
            "customer_id": ["CUST001", "CUST002", "CUST003"],
            "avgorder": [50.0, None, 75.0],
            "ordfreq": [5, 10, None],
            "eopenrate": [0.5, 0.3, 0.6],
            "eclickrate": [0.1, 0.2, 0.15],
            "paperless": [1, 0, 1],
            "refill": [0, 1, 0],
        })
        scorer = BatchScorer(model=trained_model, handle_nulls="fill_zero")
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(features_with_nulls, feature_columns=feature_cols, id_column="customer_id")
        assert len(result.predictions) == 3


class TestThresholdApplication:
    def test_applies_custom_threshold(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model, threshold=0.3)
        feature_cols = ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill"]
        result = scorer.score(sample_features, feature_columns=feature_cols, id_column="customer_id")
        for idx, row in result.predictions.iterrows():
            if row["churn_probability"] >= 0.3:
                assert row["predicted_churn"] == 1
            else:
                assert row["predicted_churn"] == 0

    def test_default_threshold_is_half(self, sample_features, trained_model):
        scorer = BatchScorer(model=trained_model)
        assert scorer.threshold == 0.5
