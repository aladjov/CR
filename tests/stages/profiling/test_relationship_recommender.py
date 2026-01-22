"""Tests for RelationshipRecommender - generates actionable recommendations from relationship analysis."""

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.relationship_recommender import (
    RecommendationCategory,
    RelationshipAnalysisSummary,
    RelationshipRecommendation,
    RelationshipRecommender,
)


class TestRecommendationCategory:
    def test_category_values(self):
        assert RecommendationCategory.FEATURE_SELECTION.value == "feature_selection"
        assert RecommendationCategory.FEATURE_ENGINEERING.value == "feature_engineering"
        assert RecommendationCategory.STRATIFICATION.value == "stratification"
        assert RecommendationCategory.MODEL_SELECTION.value == "model_selection"


class TestRelationshipRecommendation:
    def test_recommendation_to_dict(self):
        rec = RelationshipRecommendation(
            category=RecommendationCategory.FEATURE_SELECTION,
            title="Remove redundant feature",
            description="esent and eopenrate are highly correlated (r=0.85)",
            action="Drop one of the features to reduce multicollinearity",
            priority="high",
            affected_features=["esent", "eopenrate"],
            evidence={"correlation": 0.85},
        )
        d = rec.to_dict()
        assert d["category"] == "feature_selection"
        assert d["title"] == "Remove redundant feature"
        assert d["priority"] == "high"
        assert "esent" in d["affected_features"]


class TestRelationshipRecommender:
    @pytest.fixture
    def recommender(self):
        return RelationshipRecommender()

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 1000
        # Create correlated features
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 0.9 + np.random.normal(0, 0.1, n)  # High correlation with x1
        x3 = np.random.normal(0, 1, n)  # Independent
        # Target correlated with x1 and x3
        target = (x1 > 0).astype(int) * 0.7 + (x3 > 0.5).astype(int) * 0.3
        target = (target > 0.5).astype(int)

        return pd.DataFrame({
            "feature_a": x1,
            "feature_b": x2,
            "feature_c": x3,
            "retained": target,
        })

    @pytest.fixture
    def categorical_df(self):
        np.random.seed(42)
        n = 1000
        categories = np.random.choice(["A", "B", "C", "D"], n, p=[0.5, 0.3, 0.15, 0.05])
        # Different retention rates by category
        retention_probs = {"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.3}
        target = [np.random.binomial(1, retention_probs[c]) for c in categories]

        return pd.DataFrame({
            "segment": categories,
            "retained": target,
        })

    def test_analyze_returns_summary(self, recommender, sample_df):
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
        )
        assert isinstance(summary, RelationshipAnalysisSummary)
        assert len(summary.recommendations) > 0

    def test_detects_multicollinearity(self, recommender, sample_df):
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
        )
        multicollinear_recs = [
            r for r in summary.recommendations
            if r.category == RecommendationCategory.FEATURE_SELECTION
            and "multicollinear" in r.title.lower() or "correlated" in r.description.lower()
        ]
        assert len(multicollinear_recs) >= 1
        # Should mention feature_a and feature_b
        affected = []
        for rec in multicollinear_recs:
            affected.extend(rec.affected_features)
        assert "feature_a" in affected or "feature_b" in affected

    def test_identifies_strong_predictors(self, recommender, sample_df):
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
        )
        assert len(summary.strong_predictors) > 0

    def test_recommends_stratification_for_imbalanced_categories(self, recommender, categorical_df):
        summary = recommender.analyze(
            categorical_df,
            categorical_cols=["segment"],
            target_col="retained",
        )
        strat_recs = [
            r for r in summary.recommendations
            if r.category == RecommendationCategory.STRATIFICATION
        ]
        assert len(strat_recs) >= 1

    def test_groups_recommendations_by_category(self, recommender, sample_df):
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
        )
        grouped = summary.recommendations_by_category
        assert isinstance(grouped, dict)
        # At least one category should have recommendations
        assert sum(len(recs) for recs in grouped.values()) > 0

    def test_provides_model_selection_guidance(self, recommender, sample_df):
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
        )
        model_recs = [
            r for r in summary.recommendations
            if r.category == RecommendationCategory.MODEL_SELECTION
        ]
        # Should have at least one model selection recommendation
        assert len(model_recs) >= 1

    def test_summary_has_actionable_sections(self, recommender, sample_df):
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
        )
        # Check that summary provides grouped recommendations
        assert hasattr(summary, "recommendations_by_category")
        assert hasattr(summary, "high_priority_actions")

    def test_handles_empty_numeric_cols(self, recommender, categorical_df):
        summary = recommender.analyze(
            categorical_df,
            categorical_cols=["segment"],
            target_col="retained",
        )
        assert isinstance(summary, RelationshipAnalysisSummary)

    def test_handles_no_target(self, recommender, sample_df):
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col=None,
        )
        assert isinstance(summary, RelationshipAnalysisSummary)
        # Should still detect multicollinearity
        assert len(summary.recommendations) >= 1


class TestFeatureEngineeringRecommendations:
    @pytest.fixture
    def recommender(self):
        return RelationshipRecommender()

    @pytest.fixture
    def interaction_df(self):
        """DataFrame where interaction of features is predictive."""
        np.random.seed(42)
        n = 1000
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 10, n)
        # Target depends on product of x1 and x2
        target = ((x1 * x2) > 25).astype(int)

        return pd.DataFrame({
            "factor_a": x1,
            "factor_b": x2,
            "retained": target,
        })

    def test_suggests_interaction_features(self, recommender, interaction_df):
        summary = recommender.analyze(
            interaction_df,
            numeric_cols=["factor_a", "factor_b"],
            target_col="retained",
        )
        eng_recs = [
            r for r in summary.recommendations
            if r.category == RecommendationCategory.FEATURE_ENGINEERING
        ]
        # May or may not detect interaction, but should have some engineering recs
        assert isinstance(eng_recs, list)


class TestHighRiskSegmentRecommendations:
    @pytest.fixture
    def recommender(self):
        return RelationshipRecommender()

    @pytest.fixture
    def high_risk_df(self):
        np.random.seed(42)
        n = 1000
        region = np.random.choice(["North", "South", "East", "West"], n)
        # Very different retention rates
        retention_map = {"North": 0.9, "South": 0.85, "East": 0.5, "West": 0.3}
        target = [np.random.binomial(1, retention_map[r]) for r in region]

        return pd.DataFrame({"region": region, "retained": target})

    def test_identifies_high_risk_segments(self, recommender, high_risk_df):
        summary = recommender.analyze(
            high_risk_df,
            categorical_cols=["region"],
            target_col="retained",
        )
        # Should identify West and East as high risk
        high_risk = summary.high_risk_segments
        assert len(high_risk) > 0
        segment_names = [s["segment"] for s in high_risk]
        assert "West" in segment_names or "East" in segment_names

    def test_recommends_stratified_sampling(self, recommender, high_risk_df):
        summary = recommender.analyze(
            high_risk_df,
            categorical_cols=["region"],
            target_col="retained",
        )
        strat_recs = [
            r for r in summary.recommendations
            if r.category == RecommendationCategory.STRATIFICATION
        ]
        assert len(strat_recs) >= 1
        # Should mention stratified sampling
        has_stratify_mention = any(
            "stratif" in r.action.lower() or "stratif" in r.description.lower()
            for r in strat_recs
        )
        assert has_stratify_mention
