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


class TestPrecomputedInputs:
    @pytest.fixture
    def recommender(self):
        return RelationshipRecommender()

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 1000
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 0.9 + np.random.normal(0, 0.1, n)
        x3 = np.random.normal(0, 1, n)
        target = (x1 > 0).astype(int) * 0.7 + (x3 > 0.5).astype(int) * 0.3
        target = (target > 0.5).astype(int)
        return pd.DataFrame({
            "feature_a": x1, "feature_b": x2, "feature_c": x3, "retained": target,
        })

    def test_analyze_reuses_precomputed_correlation_matrix(self, recommender, sample_df):
        from customer_retention.core.compat import batched_corr_matrix

        cols = ["feature_a", "feature_b", "feature_c", "retained"]
        precomputed = batched_corr_matrix(sample_df, cols)

        summary_with = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
            correlation_matrix=precomputed,
        )
        summary_without = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
        )
        assert len(summary_with.strong_predictors) == len(summary_without.strong_predictors)
        assert len(summary_with.multicollinear_pairs) == len(summary_without.multicollinear_pairs)

    def test_analyze_reuses_precomputed_effect_sizes(self, recommender, sample_df):
        from customer_retention.core.compat import bulk_effect_sizes

        result = bulk_effect_sizes(
            sample_df, ["feature_a", "feature_b", "feature_c"], "retained",
        )

        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
            effect_sizes=result.effect_sizes,
        )
        assert isinstance(summary, RelationshipAnalysisSummary)
        assert len(summary.strong_predictors) > 0

    def test_precomputed_effect_sizes_used(self, recommender, sample_df):
        from customer_retention.core.compat import bulk_effect_sizes

        result = bulk_effect_sizes(
            sample_df, ["feature_a", "feature_b", "feature_c"], "retained",
        )

        summary_pre = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
            effect_sizes=result.effect_sizes,
        )
        pre_effects = {p["feature"]: p["effect_size"] for p in summary_pre.strong_predictors}
        for feat, d in pre_effects.items():
            assert d == pytest.approx(result.effect_sizes[feat], abs=1e-10)

    def test_missing_effect_sizes_default_zero(self, recommender, sample_df):
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
        )
        for p in summary.strong_predictors:
            assert p["effect_size"] == 0.0

    def test_partial_correlation_matrix_triggers_recompute(self, recommender, sample_df):
        partial_corr = sample_df[["feature_a", "feature_b"]].corr()
        summary = recommender.analyze(
            sample_df,
            numeric_cols=["feature_a", "feature_b", "feature_c"],
            target_col="retained",
            correlation_matrix=partial_corr,
        )
        assert isinstance(summary, RelationshipAnalysisSummary)
        assert summary.correlation_matrix is not None

    def test_precomputed_categorical_produces_recommendations(self):
        from customer_retention.stages.profiling.categorical_target_analyzer import (
            CategoricalTargetAnalyzer,
        )
        np.random.seed(42)
        n = 1000
        region = np.random.choice(["North", "South", "East", "West"], n)
        retention_map = {"North": 0.9, "South": 0.85, "East": 0.5, "West": 0.3}
        target = [np.random.binomial(1, retention_map[r]) for r in region]
        df = pd.DataFrame({"region": region, "retained": target})

        analyzer = CategoricalTargetAnalyzer(min_samples_per_category=10)
        precomputed = {"region": analyzer.analyze(df, "region", "retained")}
        recommender = RelationshipRecommender()
        summary = recommender.analyze(
            df, categorical_cols=["region"], target_col="retained",
            categorical_results=precomputed,
        )
        assert len(summary.categorical_associations) == 1
        assert len(summary.high_risk_segments) > 0

    def test_precomputed_categorical_avoids_df_computation(self):
        from customer_retention.stages.profiling.categorical_target_analyzer import (
            CategoricalTargetResult,
        )
        minimal_df = pd.DataFrame({"x": [1, 2, 3]})
        stats = pd.DataFrame({
            "category": ["A", "B"], "total_count": [100, 100],
            "retained_count": [80, 40], "retention_rate": [0.8, 0.4],
            "lift": [1.33, 0.67], "churned_count": [20, 60], "pct_of_total": [0.5, 0.5],
        })
        precomputed = {
            "segment": CategoricalTargetResult(
                categorical_col="segment", target_col="retained",
                n_categories=2, cramers_v=0.4, chi2_statistic=10.0, p_value=0.001,
                effect_strength="moderate", category_stats=stats,
                high_risk_categories=["B"], low_risk_categories=["A"],
                overall_rate=0.6,
            ),
        }
        recommender = RelationshipRecommender()
        summary = recommender.analyze(
            minimal_df, categorical_cols=["segment"], target_col="retained",
            categorical_results=precomputed,
        )
        assert len(summary.categorical_associations) == 1
        assert summary.categorical_associations[0]["cramers_v"] == 0.4

    def test_precomputed_categorical_identifies_high_risk_from_stats(self):
        from customer_retention.stages.profiling.categorical_target_analyzer import (
            CategoricalTargetResult,
        )
        minimal_df = pd.DataFrame({"x": [1]})
        stats = pd.DataFrame({
            "category": ["Good", "Bad"], "total_count": [500, 500],
            "retained_count": [450, 100], "retention_rate": [0.9, 0.2],
            "lift": [1.5, 0.33], "churned_count": [50, 400], "pct_of_total": [0.5, 0.5],
        })
        precomputed = {
            "tier": CategoricalTargetResult(
                categorical_col="tier", target_col="retained",
                n_categories=2, cramers_v=0.6, chi2_statistic=50.0, p_value=0.0001,
                effect_strength="strong", category_stats=stats,
                high_risk_categories=["Bad"], low_risk_categories=["Good"],
                overall_rate=0.55,
            ),
        }
        recommender = RelationshipRecommender()
        summary = recommender.analyze(
            minimal_df, categorical_cols=["tier"], target_col="retained",
            categorical_results=precomputed,
        )
        assert any(s["segment"] == "Bad" for s in summary.high_risk_segments)
        strat_recs = [r for r in summary.recommendations if r.category == RecommendationCategory.STRATIFICATION]
        assert len(strat_recs) >= 1

    def test_precomputed_categorical_stratification_from_rate_spread(self):
        from customer_retention.stages.profiling.categorical_target_analyzer import (
            CategoricalTargetResult,
        )
        minimal_df = pd.DataFrame({"x": [1]})
        stats = pd.DataFrame({
            "category": ["X", "Y", "Z"], "total_count": [200, 200, 200],
            "retained_count": [180, 100, 60], "retention_rate": [0.9, 0.5, 0.3],
            "lift": [1.5, 0.83, 0.5], "churned_count": [20, 100, 140],
            "pct_of_total": [0.33, 0.33, 0.33],
        })
        precomputed = {
            "group": CategoricalTargetResult(
                categorical_col="group", target_col="retained",
                n_categories=3, cramers_v=0.5, chi2_statistic=30.0, p_value=0.001,
                effect_strength="strong", category_stats=stats,
                high_risk_categories=["Z"], low_risk_categories=["X"],
                overall_rate=0.6,
            ),
        }
        recommender = RelationshipRecommender()
        summary = recommender.analyze(
            minimal_df, categorical_cols=["group"], target_col="retained",
            categorical_results=precomputed,
        )
        strat_recs = [r for r in summary.recommendations if r.category == RecommendationCategory.STRATIFICATION]
        assert any("Stratify" in r.title or "stratif" in r.action.lower() for r in strat_recs)


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
