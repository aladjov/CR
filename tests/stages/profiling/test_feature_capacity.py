"""Tests for FeatureCapacityAnalyzer - estimates favorable feature-to-data ratios."""

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.feature_capacity import (
    EffectiveFeaturesResult,
    FeatureCapacityAnalyzer,
    FeatureCapacityResult,
    ModelComplexityGuidance,
    SegmentCapacityResult,
)


class TestFeatureCapacityResult:
    def test_result_has_required_fields(self):
        result = FeatureCapacityResult(
            total_samples=1000,
            minority_class_samples=200,
            total_features=20,
            effective_features=15,
            recommended_features_conservative=10,
            recommended_features_moderate=15,
            recommended_features_aggressive=25,
            events_per_variable=10.0,
            samples_per_feature=50.0,
            capacity_status="adequate",
            recommendations=[],
        )
        assert result.total_samples == 1000
        assert result.effective_features == 15
        assert result.capacity_status == "adequate"

    def test_result_to_dict(self):
        result = FeatureCapacityResult(
            total_samples=1000,
            minority_class_samples=200,
            total_features=20,
            effective_features=15,
            recommended_features_conservative=10,
            recommended_features_moderate=15,
            recommended_features_aggressive=25,
            events_per_variable=10.0,
            samples_per_feature=50.0,
            capacity_status="adequate",
            recommendations=["Consider regularization"],
        )
        d = result.to_dict()
        assert "total_samples" in d
        assert "capacity_status" in d
        assert d["events_per_variable"] == 10.0


class TestFeatureCapacityAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return FeatureCapacityAnalyzer()

    @pytest.fixture
    def sample_df(self):
        """Standard dataset with 1000 samples."""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "feature_1": np.random.normal(0, 1, n),
            "feature_2": np.random.normal(0, 1, n),
            "feature_3": np.random.normal(0, 1, n),
            "feature_4": np.random.normal(0, 1, n),
            "feature_5": np.random.normal(0, 1, n),
            "target": np.random.binomial(1, 0.3, n),  # 30% minority class
        })

    @pytest.fixture
    def correlated_df(self):
        """Dataset with highly correlated features."""
        np.random.seed(42)
        n = 1000
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 0.95 + np.random.normal(0, 0.1, n)  # ~0.95 correlation with x1
        x3 = x1 * 0.9 + np.random.normal(0, 0.15, n)  # ~0.9 correlation with x1
        x4 = np.random.normal(0, 1, n)  # Independent
        return pd.DataFrame({
            "feature_1": x1,
            "feature_2": x2,
            "feature_3": x3,
            "feature_4": x4,
            "target": np.random.binomial(1, 0.3, n),
        })

    @pytest.fixture
    def small_df(self):
        """Small dataset with only 100 samples."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "feature_1": np.random.normal(0, 1, n),
            "feature_2": np.random.normal(0, 1, n),
            "feature_3": np.random.normal(0, 1, n),
            "target": np.random.binomial(1, 0.2, n),  # 20% minority (~20 events)
        })

    def test_analyze_returns_result(self, analyzer, sample_df):
        result = analyzer.analyze(
            sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            target_col="target",
        )
        assert isinstance(result, FeatureCapacityResult)
        assert result.total_samples == 1000
        assert result.total_features == 5

    def test_calculates_minority_class_samples(self, analyzer, sample_df):
        result = analyzer.analyze(
            sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="target",
        )
        # With 30% minority rate and 1000 samples, expect ~300 minority samples
        assert 250 <= result.minority_class_samples <= 350

    def test_calculates_events_per_variable(self, analyzer, sample_df):
        result = analyzer.analyze(
            sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="target",
        )
        # EPV = minority_samples / n_features ≈ 300 / 3 ≈ 100
        assert result.events_per_variable > 50

    def test_detects_inadequate_capacity_small_data(self, analyzer, small_df):
        result = analyzer.analyze(
            small_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="target",
        )
        # 100 samples, 20% minority = 20 events, 3 features
        # EPV = 20/3 ≈ 6.7 - below recommended minimum
        assert result.capacity_status in ["limited", "inadequate"]

    def test_provides_feature_recommendations(self, analyzer, small_df):
        result = analyzer.analyze(
            small_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="target",
        )
        assert len(result.recommendations) > 0

    def test_conservative_recommendation_uses_epv_20(self, analyzer, sample_df):
        result = analyzer.analyze(
            sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="target",
        )
        # Conservative: minority_samples / 20
        # ~300 / 20 = 15
        assert result.recommended_features_conservative <= result.minority_class_samples / 20 + 1

    def test_moderate_recommendation_uses_epv_10(self, analyzer, sample_df):
        result = analyzer.analyze(
            sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="target",
        )
        # Moderate: minority_samples / 10
        assert result.recommended_features_moderate <= result.minority_class_samples / 10 + 1


class TestEffectiveFeatures:
    @pytest.fixture
    def analyzer(self):
        return FeatureCapacityAnalyzer()

    @pytest.fixture
    def correlated_df(self):
        np.random.seed(42)
        n = 1000
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 0.95 + np.random.normal(0, 0.1, n)
        x3 = x1 * 0.9 + np.random.normal(0, 0.15, n)
        x4 = np.random.normal(0, 1, n)
        return pd.DataFrame({
            "feature_1": x1,
            "feature_2": x2,
            "feature_3": x3,
            "feature_4": x4,
            "target": np.random.binomial(1, 0.3, n),
        })

    def test_calculates_effective_features(self, analyzer, correlated_df):
        result = analyzer.calculate_effective_features(
            correlated_df,
            feature_cols=["feature_1", "feature_2", "feature_3", "feature_4"],
        )
        assert isinstance(result, EffectiveFeaturesResult)
        # 4 features but highly correlated, so effective < 4
        assert result.effective_count < 4
        assert result.total_count == 4

    def test_identifies_redundant_features(self, analyzer, correlated_df):
        result = analyzer.calculate_effective_features(
            correlated_df,
            feature_cols=["feature_1", "feature_2", "feature_3", "feature_4"],
        )
        # feature_2 and feature_3 are highly correlated with feature_1
        assert len(result.redundant_features) >= 1

    def test_identifies_feature_clusters(self, analyzer, correlated_df):
        result = analyzer.calculate_effective_features(
            correlated_df,
            feature_cols=["feature_1", "feature_2", "feature_3", "feature_4"],
        )
        # Should identify a cluster of correlated features
        assert len(result.feature_clusters) >= 1

    def test_returns_representative_features(self, analyzer, correlated_df):
        result = analyzer.calculate_effective_features(
            correlated_df,
            feature_cols=["feature_1", "feature_2", "feature_3", "feature_4"],
        )
        # Should suggest keeping fewer features
        assert len(result.representative_features) < 4


class TestSegmentCapacityAnalysis:
    @pytest.fixture
    def analyzer(self):
        return FeatureCapacityAnalyzer()

    @pytest.fixture
    def segmented_df(self):
        np.random.seed(42)
        # Create segments with different sizes
        segment_a = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 500),
            "feature_2": np.random.normal(0, 1, 500),
            "segment": "A",
            "target": np.random.binomial(1, 0.3, 500),
        })
        segment_b = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 300),
            "feature_2": np.random.normal(0, 1, 300),
            "segment": "B",
            "target": np.random.binomial(1, 0.2, 300),
        })
        segment_c = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 50),
            "feature_2": np.random.normal(0, 1, 50),
            "segment": "C",
            "target": np.random.binomial(1, 0.25, 50),
        })
        return pd.concat([segment_a, segment_b, segment_c], ignore_index=True)

    def test_analyze_segments_returns_results(self, analyzer, segmented_df):
        result = analyzer.analyze_segment_capacity(
            segmented_df,
            feature_cols=["feature_1", "feature_2"],
            target_col="target",
            segment_col="segment",
        )
        assert isinstance(result, SegmentCapacityResult)
        assert len(result.segment_capacities) == 3

    def test_identifies_segments_with_insufficient_capacity(self, analyzer, segmented_df):
        result = analyzer.analyze_segment_capacity(
            segmented_df,
            feature_cols=["feature_1", "feature_2"],
            target_col="target",
            segment_col="segment",
        )
        # Segment C has only 50 samples, ~12 minority events
        insufficient = [s for s, cap in result.segment_capacities.items()
                       if cap.capacity_status in ["limited", "inadequate"]]
        assert "C" in insufficient

    def test_recommends_segment_strategy(self, analyzer, segmented_df):
        result = analyzer.analyze_segment_capacity(
            segmented_df,
            feature_cols=["feature_1", "feature_2"],
            target_col="target",
            segment_col="segment",
        )
        assert result.recommended_strategy in ["single_model", "segment_models", "hybrid"]


class TestModelComplexityGuidance:
    @pytest.fixture
    def analyzer(self):
        return FeatureCapacityAnalyzer()

    def test_guidance_for_small_data(self, analyzer):
        guidance = analyzer.get_complexity_guidance(
            n_samples=100,
            n_minority=20,
            n_features=10,
        )
        assert isinstance(guidance, ModelComplexityGuidance)
        assert guidance.max_features_linear <= 2  # EPV 10 rule
        assert "regularization" in guidance.recommendations[0].lower() or "features" in guidance.recommendations[0].lower()

    def test_guidance_for_large_data(self, analyzer):
        guidance = analyzer.get_complexity_guidance(
            n_samples=10000,
            n_minority=3000,
            n_features=50,
        )
        # With 3000 events, EPV=60 for 50 features - very healthy
        assert guidance.max_features_linear >= 50

    def test_tree_models_allow_more_features(self, analyzer):
        guidance = analyzer.get_complexity_guidance(
            n_samples=500,
            n_minority=100,
            n_features=20,
        )
        # Trees are more flexible with feature count
        assert guidance.max_features_tree >= guidance.max_features_linear

    def test_guidance_includes_model_recommendations(self, analyzer):
        guidance = analyzer.get_complexity_guidance(
            n_samples=200,
            n_minority=40,
            n_features=15,
        )
        assert len(guidance.model_recommendations) > 0
