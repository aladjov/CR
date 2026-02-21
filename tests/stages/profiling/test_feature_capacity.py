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


class TestAnalyzeWithAllNaNTarget:
    @pytest.fixture
    def analyzer(self):
        return FeatureCapacityAnalyzer()

    def test_analyze_all_nan_target_returns_inadequate(self, analyzer):
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
            "target": [np.nan, np.nan, np.nan],
        })
        result = analyzer.analyze(df, ["feature_1", "feature_2"], "target")
        assert result.minority_class_samples == 0
        assert result.capacity_status == "inadequate"

    def test_segment_capacity_skips_all_nan_segments(self, analyzer):
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "segment": ["A", "A", "A", "B", "B", "B"],
            "target": [1.0, 0.0, 0.0, np.nan, np.nan, np.nan],
        })
        result = analyzer.analyze_segment_capacity(
            df, ["feature_1", "feature_2"], "target", "segment"
        )
        assert "A" in result.segment_capacities
        assert "B" not in result.segment_capacities


class TestBulkSegmentCapacity:
    """Tests that segment capacity uses bulk groupby instead of per-segment loops."""

    @pytest.fixture
    def analyzer(self):
        return FeatureCapacityAnalyzer()

    @pytest.fixture
    def many_segments_df(self):
        np.random.seed(42)
        segments = [f"seg_{i}" for i in range(20)]
        rows = []
        for seg in segments:
            n = np.random.randint(30, 200)
            for _ in range(n):
                rows.append({
                    "f1": np.random.normal(), "f2": np.random.normal(),
                    "f3": np.random.normal(), "segment": seg,
                    "target": np.random.binomial(1, 0.3),
                })
        return pd.DataFrame(rows)

    def test_bulk_segment_matches_per_segment_results(self, analyzer, many_segments_df):
        result = analyzer.analyze_segment_capacity(
            many_segments_df, ["f1", "f2", "f3"], "target", "segment"
        )
        for seg_name, cap in result.segment_capacities.items():
            seg_df = many_segments_df[many_segments_df["segment"] == seg_name]
            assert cap.total_samples == len(seg_df)
            expected_minority = int(seg_df["target"].value_counts().min())
            assert cap.minority_class_samples == expected_minority

    def test_bulk_segment_shares_effective_features(self, analyzer, many_segments_df):
        result = analyzer.analyze_segment_capacity(
            many_segments_df, ["f1", "f2", "f3"], "target", "segment"
        )
        effective_counts = {cap.effective_features for cap in result.segment_capacities.values()}
        assert len(effective_counts) == 1

    def test_bulk_segment_epv_uses_segment_minority(self, analyzer):
        df = pd.DataFrame({
            "f1": np.random.normal(0, 1, 600),
            "f2": np.random.normal(0, 1, 600),
            "segment": ["big"] * 500 + ["small"] * 100,
            "target": [1] * 250 + [0] * 250 + [1] * 10 + [0] * 90,
        })
        result = analyzer.analyze_segment_capacity(df, ["f1", "f2"], "target", "segment")
        assert result.segment_capacities["big"].minority_class_samples == 250
        assert result.segment_capacities["small"].minority_class_samples == 10
        assert result.segment_capacities["big"].events_per_variable == 125.0
        assert result.segment_capacities["small"].events_per_variable == 5.0

    def test_bulk_segment_single_class_segment_skipped(self, analyzer):
        df = pd.DataFrame({
            "f1": [1.0] * 10 + [2.0] * 10,
            "f2": [3.0] * 10 + [4.0] * 10,
            "segment": ["has_both"] * 10 + ["only_zeros"] * 10,
            "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] + [0] * 10,
        })
        result = analyzer.analyze_segment_capacity(df, ["f1", "f2"], "target", "segment")
        assert "has_both" in result.segment_capacities
        assert result.segment_capacities["has_both"].minority_class_samples == 5

    def test_bulk_segment_correlation_computed_once(self, analyzer, many_segments_df, monkeypatch):
        call_count = {"n": 0}
        original = analyzer.calculate_effective_features

        def counting_wrapper(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(analyzer, "calculate_effective_features", counting_wrapper)
        analyzer.analyze_segment_capacity(
            many_segments_df, ["f1", "f2", "f3"], "target", "segment"
        )
        assert call_count["n"] == 1


class TestEffectiveFeaturesUsesBatchedCorr:
    @pytest.fixture
    def analyzer(self):
        return FeatureCapacityAnalyzer()

    def test_calculate_effective_features_uses_batched_corr(self, analyzer, monkeypatch):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "f1": np.random.normal(0, 1, n),
            "f2": np.random.normal(0, 1, n),
            "f3": np.random.normal(0, 1, n),
        })
        calls = []
        original = None

        import customer_retention.stages.profiling.feature_capacity as mod
        from customer_retention.core.compat import batched_corr_matrix as _orig

        def tracking_batched(df_arg, cols):
            calls.append(cols)
            return _orig(df_arg, cols)

        monkeypatch.setattr(mod, "batched_corr_matrix", tracking_batched)
        result = analyzer.calculate_effective_features(df, ["f1", "f2", "f3"])
        assert len(calls) == 1
        assert set(calls[0]) == {"f1", "f2", "f3"}
        assert result.total_count == 3

    def test_effective_features_result_unchanged_after_batched_corr(self, analyzer):
        np.random.seed(42)
        n = 500
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 0.95 + np.random.normal(0, 0.1, n)
        x3 = np.random.normal(0, 1, n)
        df = pd.DataFrame({"a": x1, "b": x2, "c": x3})
        result = analyzer.calculate_effective_features(df, ["a", "b", "c"])
        assert result.effective_count < 3
        assert len(result.redundant_features) >= 1
        assert "b" in result.redundant_features

    def test_effective_features_handles_zero_variance_column(self, analyzer):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "constant": [5.0] * n,
            "varying": np.random.normal(0, 1, n),
            "another": np.random.normal(0, 1, n),
        })
        result = analyzer.calculate_effective_features(df, ["constant", "varying", "another"])
        assert result.total_count == 3
        assert result.effective_count >= 1


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
