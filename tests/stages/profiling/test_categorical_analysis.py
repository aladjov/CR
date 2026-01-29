import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.categorical_target_analyzer import (
    CategoricalAnalysisResult,
    analyze_categorical_features,
    filter_categorical_columns,
)


@pytest.fixture
def sample_entity_data():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "entity_id": [f"E{i}" for i in range(n)],
        "email_id": [f"email_{i}@test.com" for i in range(n)],  # High cardinality - should filter
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),  # Date - should filter
        "plan_type": np.random.choice(["free", "basic", "premium"], n, p=[0.5, 0.3, 0.2]),
        "region": np.random.choice(["US", "EU", "APAC", "LATAM"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "device": np.random.choice(["mobile", "desktop", "tablet"], n, p=[0.5, 0.35, 0.15]),
        "target": np.random.choice([0, 1], n, p=[0.6, 0.4])
    })


@pytest.fixture
def data_with_predictive_feature():
    np.random.seed(42)
    n = 500
    plan_type = np.random.choice(["free", "basic", "premium"], n, p=[0.5, 0.3, 0.2])
    # Premium has 80% retention, free has 20%
    target = np.array([
        1 if (p == "premium" and np.random.random() < 0.8) or
             (p == "basic" and np.random.random() < 0.5) or
             (p == "free" and np.random.random() < 0.2) else 0
        for p in plan_type
    ])
    return pd.DataFrame({
        "entity_id": [f"E{i}" for i in range(n)],
        "plan_type": plan_type,
        "region": np.random.choice(["US", "EU"], n),  # Random - no signal
        "target": target
    })


class TestFilterCategoricalColumns:
    def test_filters_identifier_columns(self, sample_entity_data):
        filtered = filter_categorical_columns(
            sample_entity_data, "entity_id", "target", cardinality_threshold=0.5
        )
        assert "email_id" not in filtered
        assert "entity_id" not in filtered

    def test_filters_date_columns(self, sample_entity_data):
        filtered = filter_categorical_columns(
            sample_entity_data, "entity_id", "target", cardinality_threshold=0.5
        )
        assert "signup_date" not in filtered

    def test_keeps_valid_categorical_columns(self, sample_entity_data):
        filtered = filter_categorical_columns(
            sample_entity_data, "entity_id", "target", cardinality_threshold=0.5
        )
        assert "plan_type" in filtered
        assert "region" in filtered
        assert "device" in filtered

    def test_excludes_target_column(self, sample_entity_data):
        filtered = filter_categorical_columns(
            sample_entity_data, "entity_id", "target", cardinality_threshold=0.5
        )
        assert "target" not in filtered


class TestAnalyzeCategoricalFeatures:
    def test_returns_analysis_result(self, sample_entity_data):
        result = analyze_categorical_features(
            sample_entity_data, "entity_id", "target"
        )
        assert isinstance(result, CategoricalAnalysisResult)

    def test_excludes_filtered_columns_from_results(self, sample_entity_data):
        result = analyze_categorical_features(
            sample_entity_data, "entity_id", "target"
        )
        analyzed_features = [f.feature_name for f in result.feature_insights]
        assert "email_id" not in analyzed_features
        assert "signup_date" not in analyzed_features

    def test_identifies_filtered_columns(self, sample_entity_data):
        result = analyze_categorical_features(
            sample_entity_data, "entity_id", "target"
        )
        assert "email_id" in result.filtered_columns
        assert "identifier" in result.filter_reasons.get("email_id", "").lower()

    def test_detects_strong_predictor(self, data_with_predictive_feature):
        result = analyze_categorical_features(
            data_with_predictive_feature, "entity_id", "target"
        )
        plan_insight = next((f for f in result.feature_insights if f.feature_name == "plan_type"), None)
        assert plan_insight is not None
        assert plan_insight.cramers_v > 0.2  # Should show moderate+ association

    def test_generates_recommendations(self, data_with_predictive_feature):
        result = analyze_categorical_features(
            data_with_predictive_feature, "entity_id", "target"
        )
        assert len(result.recommendations) > 0

    def test_recommendations_reference_strong_features(self, data_with_predictive_feature):
        result = analyze_categorical_features(
            data_with_predictive_feature, "entity_id", "target"
        )
        rec_text = " ".join(str(r) for r in result.recommendations)
        assert "plan_type" in rec_text.lower()


class TestCategoricalFeatureInsight:
    def test_insight_has_interpretation(self, data_with_predictive_feature):
        result = analyze_categorical_features(
            data_with_predictive_feature, "entity_id", "target"
        )
        for insight in result.feature_insights:
            assert insight.interpretation is not None
            assert len(insight.interpretation) > 0

    def test_high_risk_categories_identified(self, data_with_predictive_feature):
        result = analyze_categorical_features(
            data_with_predictive_feature, "entity_id", "target"
        )
        plan_insight = next((f for f in result.feature_insights if f.feature_name == "plan_type"), None)
        assert plan_insight is not None
        # "free" should be high risk (low retention)
        assert len(plan_insight.high_risk_categories) > 0


class TestEdgeCases:
    def test_no_categorical_columns(self):
        df = pd.DataFrame({
            "entity_id": ["E1", "E2", "E3"],
            "value": [1.0, 2.0, 3.0],
            "target": [0, 1, 0]
        })
        result = analyze_categorical_features(df, "entity_id", "target")
        assert len(result.feature_insights) == 0

    def test_all_columns_filtered(self):
        df = pd.DataFrame({
            "entity_id": [f"E{i}" for i in range(100)],
            "unique_col": [f"val_{i}" for i in range(100)],  # All unique
            "target": [0, 1] * 50
        })
        result = analyze_categorical_features(df, "entity_id", "target")
        assert len(result.feature_insights) == 0
        assert "unique_col" in result.filtered_columns
