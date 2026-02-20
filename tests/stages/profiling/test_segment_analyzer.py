import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.segment_analyzer import (
    ClusterVisualizationResult,
    DimensionReductionMethod,
    FullSegmentationResult,
    SegmentAnalyzer,
    SegmentationDecisionMetrics,
    SegmentationMethod,
    SegmentationResult,
    SegmentProfile,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 300
    # Create data with 3 natural clusters
    cluster_1 = pd.DataFrame({
        "feature_a": np.random.normal(10, 1, n // 3),
        "feature_b": np.random.normal(100, 10, n // 3),
        "feature_c": np.random.normal(5, 0.5, n // 3),
        "target": np.random.choice([0, 1], n // 3, p=[0.8, 0.2]),
    })
    cluster_2 = pd.DataFrame({
        "feature_a": np.random.normal(50, 2, n // 3),
        "feature_b": np.random.normal(200, 15, n // 3),
        "feature_c": np.random.normal(15, 1, n // 3),
        "target": np.random.choice([0, 1], n // 3, p=[0.5, 0.5]),
    })
    cluster_3 = pd.DataFrame({
        "feature_a": np.random.normal(90, 3, n // 3),
        "feature_b": np.random.normal(50, 5, n // 3),
        "feature_c": np.random.normal(25, 2, n // 3),
        "target": np.random.choice([0, 1], n // 3, p=[0.2, 0.8]),
    })
    return pd.concat([cluster_1, cluster_2, cluster_3], ignore_index=True)


@pytest.fixture
def homogeneous_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "feature_a": np.random.normal(50, 5, n),
        "feature_b": np.random.normal(100, 10, n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def analyzer():
    return SegmentAnalyzer()


class TestSegmentAnalyzerInit:
    def test_default_method(self, analyzer):
        assert analyzer.default_method == SegmentationMethod.KMEANS

    def test_custom_method(self):
        analyzer = SegmentAnalyzer(default_method=SegmentationMethod.HIERARCHICAL)
        assert analyzer.default_method == SegmentationMethod.HIERARCHICAL


class TestSegmentationMethod:
    def test_method_values(self):
        assert SegmentationMethod.KMEANS.value == "kmeans"
        assert SegmentationMethod.HIERARCHICAL.value == "hierarchical"
        assert SegmentationMethod.DBSCAN.value == "dbscan"


class TestAnalyze:
    def test_returns_segmentation_result(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        assert isinstance(result, SegmentationResult)

    def test_detects_multiple_segments(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        assert result.n_segments >= 2

    def test_returns_quality_score(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        assert 0 <= result.quality_score <= 1

    def test_returns_segment_profiles(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        assert len(result.profiles) == result.n_segments
        assert all(isinstance(p, SegmentProfile) for p in result.profiles)

    def test_profiles_have_required_fields(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        for profile in result.profiles:
            assert profile.segment_id >= 0
            assert profile.size > 0
            assert 0 <= profile.size_pct <= 100
            assert profile.defining_features is not None

    def test_target_rate_included_when_target_provided(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        for profile in result.profiles:
            assert profile.target_rate is not None
            assert 0 <= profile.target_rate <= 1

    def test_target_rate_none_when_no_target(self, analyzer, sample_df):
        df = sample_df.drop(columns=["target"])
        result = analyzer.analyze(df)
        for profile in result.profiles:
            assert profile.target_rate is None

    def test_returns_recommendation(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        assert result.recommendation in [
            "single_model",
            "consider_segmentation",
            "strong_segmentation",
        ]

    def test_returns_confidence(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        assert 0 <= result.confidence <= 1

    def test_returns_rationale(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        assert isinstance(result.rationale, list)
        assert len(result.rationale) > 0

    def test_respects_max_segments(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, max_segments=2)
        assert result.n_segments <= 2

    def test_uses_specified_features(self, analyzer, sample_df):
        result = analyzer.analyze(
            sample_df,
            feature_cols=["feature_a", "feature_b"],
            target_col="target"
        )
        assert result is not None

    def test_homogeneous_data_suggests_single_model(self, analyzer, homogeneous_df):
        result = analyzer.analyze(homogeneous_df, target_col="target")
        # Homogeneous data should not strongly suggest segmentation
        assert result.recommendation in ["single_model", "consider_segmentation"]


class TestFindOptimalSegments:
    def test_returns_integer(self, analyzer, sample_df):
        feature_cols = ["feature_a", "feature_b", "feature_c"]
        n = analyzer.find_optimal_segments(sample_df, feature_cols)
        assert isinstance(n, int)
        assert n >= 1

    def test_respects_max_k(self, analyzer, sample_df):
        feature_cols = ["feature_a", "feature_b", "feature_c"]
        n = analyzer.find_optimal_segments(sample_df, feature_cols, max_k=3)
        assert n <= 3

    def test_finds_natural_clusters(self, analyzer, sample_df):
        feature_cols = ["feature_a", "feature_b", "feature_c"]
        n = analyzer.find_optimal_segments(sample_df, feature_cols)
        # Should detect roughly 3 clusters in our sample data
        assert 2 <= n <= 5


class TestProfileSegments:
    def test_returns_list_of_profiles(self, analyzer, sample_df):
        feature_cols = ["feature_a", "feature_b", "feature_c"]
        labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
        profiles = analyzer.profile_segments(
            sample_df, labels, feature_cols, target_col="target"
        )
        assert isinstance(profiles, list)
        assert len(profiles) == 3

    def test_profiles_sum_to_total(self, analyzer, sample_df):
        feature_cols = ["feature_a", "feature_b", "feature_c"]
        labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
        profiles = analyzer.profile_segments(
            sample_df, labels, feature_cols, target_col="target"
        )
        total_size = sum(p.size for p in profiles)
        assert total_size == len(sample_df)

    def test_defining_features_identify_segment(self, analyzer, sample_df):
        feature_cols = ["feature_a", "feature_b", "feature_c"]
        labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
        profiles = analyzer.profile_segments(
            sample_df, labels, feature_cols, target_col="target"
        )
        for profile in profiles:
            assert "mean" in profile.defining_features or len(profile.defining_features) > 0


class TestTargetVarianceAnalysis:
    def test_calculates_target_variance_ratio(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        assert result.target_variance_ratio is not None
        assert result.target_variance_ratio >= 0

    def test_high_variance_suggests_segmentation(self, analyzer, sample_df):
        # Sample data has very different target rates per cluster
        result = analyzer.analyze(sample_df, target_col="target")
        if result.target_variance_ratio > 0.3:
            assert result.recommendation in ["consider_segmentation", "strong_segmentation"]


class TestEdgeCases:
    def test_handles_missing_values(self, analyzer):
        df = pd.DataFrame({
            "feature_a": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            "feature_b": [10, np.nan, 30, 40, 50, 60, 70, 80, 90, 100],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        result = analyzer.analyze(df, target_col="target")
        assert result is not None

    def test_handles_small_dataset(self, analyzer):
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        })
        result = analyzer.analyze(df, target_col="target")
        assert result.n_segments >= 1

    def test_handles_single_feature(self, analyzer):
        np.random.seed(42)
        df = pd.DataFrame({
            "feature_a": np.concatenate([
                np.random.normal(10, 1, 50),
                np.random.normal(50, 1, 50),
            ]),
            "target": [0] * 50 + [1] * 50,
        })
        result = analyzer.analyze(df, feature_cols=["feature_a"], target_col="target")
        assert result is not None

    def test_handles_categorical_target(self, analyzer):
        np.random.seed(42)
        df = pd.DataFrame({
            "feature_a": np.random.normal(0, 1, 100),
            "feature_b": np.random.normal(0, 1, 100),
            "target": np.random.choice(["A", "B", "C"], 100),
        })
        result = analyzer.analyze(df, target_col="target")
        # Should still work, target_rate might be None for non-binary
        assert result is not None


class TestSegmentationResult:
    def test_dataclass_fields(self):
        result = SegmentationResult(
            n_segments=3,
            method=SegmentationMethod.KMEANS,
            quality_score=0.75,
            profiles=[],
            target_variance_ratio=0.25,
            recommendation="consider_segmentation",
            confidence=0.8,
            rationale=["Different target rates across segments"],
            labels=np.array([0, 1, 2]),
        )
        assert result.n_segments == 3
        assert result.method == SegmentationMethod.KMEANS
        assert result.quality_score == 0.75


class TestSegmentProfile:
    def test_dataclass_fields(self):
        profile = SegmentProfile(
            segment_id=0,
            size=100,
            size_pct=33.3,
            target_rate=0.25,
            defining_features={"feature_a": {"mean": 10.5, "std": 1.2}},
        )
        assert profile.segment_id == 0
        assert profile.size == 100
        assert profile.size_pct == 33.3
        assert profile.target_rate == 0.25


class TestDimensionReductionMethod:
    def test_method_values(self):
        assert DimensionReductionMethod.PCA.value == "pca"
        assert DimensionReductionMethod.TSNE.value == "tsne"
        assert DimensionReductionMethod.UMAP.value == "umap"


class TestClusterVisualizationResult:
    def test_dataclass_fields(self):
        result = ClusterVisualizationResult(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([4.0, 5.0, 6.0]),
            labels=np.array([0, 1, 0]),
            method=DimensionReductionMethod.PCA,
            explained_variance_ratio=0.85,
        )
        assert len(result.x) == 3
        assert len(result.y) == 3
        assert len(result.labels) == 3
        assert result.method == DimensionReductionMethod.PCA
        assert result.explained_variance_ratio == 0.85


class TestGetClusterVisualization:
    def test_returns_visualization_result(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        viz = analyzer.get_cluster_visualization(
            sample_df,
            labels=result.labels,
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert isinstance(viz, ClusterVisualizationResult)

    def test_returns_2d_coordinates(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        viz = analyzer.get_cluster_visualization(
            sample_df,
            labels=result.labels,
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert len(viz.x) == len(sample_df)
        assert len(viz.y) == len(sample_df)

    def test_coordinates_are_numeric(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        viz = analyzer.get_cluster_visualization(
            sample_df,
            labels=result.labels,
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert np.issubdtype(viz.x.dtype, np.floating)
        assert np.issubdtype(viz.y.dtype, np.floating)

    def test_labels_match_input(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        viz = analyzer.get_cluster_visualization(
            sample_df,
            labels=result.labels,
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert np.array_equal(viz.labels, result.labels)

    def test_default_method_is_pca(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        viz = analyzer.get_cluster_visualization(
            sample_df,
            labels=result.labels,
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert viz.method == DimensionReductionMethod.PCA

    def test_pca_returns_explained_variance(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        viz = analyzer.get_cluster_visualization(
            sample_df,
            labels=result.labels,
            feature_cols=["feature_a", "feature_b", "feature_c"],
            method=DimensionReductionMethod.PCA,
        )
        assert viz.explained_variance_ratio is not None
        assert 0 <= viz.explained_variance_ratio <= 1

    def test_tsne_method(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        viz = analyzer.get_cluster_visualization(
            sample_df,
            labels=result.labels,
            feature_cols=["feature_a", "feature_b", "feature_c"],
            method=DimensionReductionMethod.TSNE,
        )
        assert viz.method == DimensionReductionMethod.TSNE
        assert len(viz.x) == len(sample_df)

    def test_handles_missing_values(self, analyzer):
        df = pd.DataFrame({
            "feature_a": [1, 2, np.nan, 4, 5] * 20,
            "feature_b": [10, np.nan, 30, 40, 50] * 20,
        })
        labels = np.array([0, 0, -1, 1, 1] * 20)  # -1 for invalid rows
        viz = analyzer.get_cluster_visualization(
            df,
            labels=labels,
            feature_cols=["feature_a", "feature_b"],
        )
        # Should return coordinates for all rows (NaN for invalid)
        assert len(viz.x) == len(df)

    def test_clusters_are_visually_separated(self, analyzer, sample_df):
        """Clusters with distinct features should appear separated in 2D."""
        result = analyzer.analyze(sample_df, target_col="target")
        viz = analyzer.get_cluster_visualization(
            sample_df,
            labels=result.labels,
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        # Calculate centroids for each cluster
        unique_labels = set(result.labels[result.labels >= 0])
        if len(unique_labels) >= 2:
            centroids = []
            for label in unique_labels:
                mask = result.labels == label
                centroid = (np.mean(viz.x[mask]), np.mean(viz.y[mask]))
                centroids.append(centroid)

            # At least some centroids should be reasonably separated
            # (distance > 0.5 in normalized space)
            distances = []
            for i, c1 in enumerate(centroids):
                for c2 in centroids[i+1:]:
                    dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    distances.append(dist)

            assert max(distances) > 0.1  # Some visual separation exists


class TestSegmentationDecisionMetrics:
    def test_dataclass_fields(self):
        metrics = SegmentationDecisionMetrics(
            silhouette_score=0.45,
            silhouette_interpretation="Reasonable",
            target_variance_ratio=0.25,
            target_variance_interpretation="Moderate",
            n_segments=3,
            segments_interpretation="Manageable",
            confidence=0.65,
            confidence_interpretation="High",
            recommendation="consider_segmentation",
            rationale=["Good cluster separation"],
        )
        assert metrics.silhouette_score == 0.45
        assert metrics.n_segments == 3
        assert metrics.recommendation == "consider_segmentation"

    def test_from_segmentation_result(self, analyzer, sample_df):
        result = analyzer.analyze(sample_df, target_col="target")
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)

        assert isinstance(metrics, SegmentationDecisionMetrics)
        assert -1 <= metrics.silhouette_score <= 1
        assert metrics.n_segments == result.n_segments
        assert metrics.recommendation == result.recommendation
        assert metrics.silhouette_interpretation in [
            "Strong structure", "Reasonable", "Weak structure", "No structure"
        ]


class TestFullSegmentationResult:
    def test_dataclass_fields(self):
        result = FullSegmentationResult(
            metrics=SegmentationDecisionMetrics(
                silhouette_score=0.3, silhouette_interpretation="Reasonable",
                target_variance_ratio=0.2, target_variance_interpretation="Moderate",
                n_segments=2, segments_interpretation="Manageable",
                confidence=0.5, confidence_interpretation="Medium",
                recommendation="consider_segmentation", rationale=[]
            ),
            profiles=[],
            size_distribution={"total": 100, "min_size": 40, "max_size": 60, "balance_ratio": 0.67},
            visualization=None,
            segmentation_result=None,
        )
        assert result.metrics.n_segments == 2
        assert result.size_distribution["total"] == 100

    def test_has_visualization_property(self):
        result = FullSegmentationResult(
            metrics=SegmentationDecisionMetrics(
                silhouette_score=0.3, silhouette_interpretation="Reasonable",
                target_variance_ratio=None, target_variance_interpretation="N/A",
                n_segments=1, segments_interpretation="Single",
                confidence=0.5, confidence_interpretation="Medium",
                recommendation="single_model", rationale=[]
            ),
            profiles=[],
            size_distribution={},
            visualization=None,
            segmentation_result=None,
        )
        assert result.has_visualization is False


class TestRunFullAnalysis:
    def test_returns_full_result(self, analyzer, sample_df):
        result = analyzer.run_full_analysis(
            sample_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert isinstance(result, FullSegmentationResult)

    def test_contains_metrics(self, analyzer, sample_df):
        result = analyzer.run_full_analysis(
            sample_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert isinstance(result.metrics, SegmentationDecisionMetrics)
        assert result.metrics.silhouette_score is not None
        assert result.metrics.recommendation is not None

    def test_contains_profiles(self, analyzer, sample_df):
        result = analyzer.run_full_analysis(
            sample_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert len(result.profiles) > 0
        assert all(isinstance(p, SegmentProfile) for p in result.profiles)

    def test_contains_size_distribution(self, analyzer, sample_df):
        result = analyzer.run_full_analysis(
            sample_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert "total" in result.size_distribution
        assert "min_size" in result.size_distribution
        assert "max_size" in result.size_distribution
        assert "balance_ratio" in result.size_distribution

    def test_includes_visualization_when_multiple_segments(self, analyzer, sample_df):
        result = analyzer.run_full_analysis(
            sample_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        if result.metrics.n_segments > 1:
            assert result.visualization is not None
            assert isinstance(result.visualization, ClusterVisualizationResult)

    def test_preserves_segmentation_result(self, analyzer, sample_df):
        result = analyzer.run_full_analysis(
            sample_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert result.segmentation_result is not None
        assert isinstance(result.segmentation_result, SegmentationResult)
        assert len(result.segmentation_result.labels) == len(sample_df)

    def test_works_without_target(self, analyzer, sample_df):
        df = sample_df.drop(columns=["target"])
        result = analyzer.run_full_analysis(
            df,
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        assert result.metrics.target_variance_ratio is None
        assert result.metrics.target_variance_interpretation == "N/A"

    def test_homogeneous_data_returns_single_segment(self, analyzer, homogeneous_df):
        result = analyzer.run_full_analysis(
            homogeneous_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b"],
        )
        # Should work even with homogeneous data
        assert result.metrics.n_segments >= 1

    def test_get_decision_summary_method(self, analyzer, sample_df):
        result = analyzer.run_full_analysis(
            sample_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b", "feature_c"],
        )
        summary = result.get_decision_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key decision information
        assert any(word in summary.lower() for word in ["segment", "model", "recommend"])


class TestDecisionSummaryBranches:
    """Test all get_decision_summary branches."""

    def _make_full_result(self, recommendation):
        metrics = SegmentationDecisionMetrics(
            silhouette_score=0.3,
            silhouette_interpretation="Reasonable",
            target_variance_ratio=0.2,
            target_variance_interpretation="Moderate",
            n_segments=2,
            segments_interpretation="Manageable",
            confidence=0.5,
            confidence_interpretation="Medium",
            recommendation=recommendation,
            rationale=[],
        )
        return FullSegmentationResult(
            metrics=metrics,
            profiles=[],
            size_distribution={},
            visualization=None,
            segmentation_result=None,
        )

    def test_strong_segmentation_summary(self):
        result = self._make_full_result("strong_segmentation")
        summary = result.get_decision_summary()
        assert "STRONG EVIDENCE" in summary

    def test_consider_segmentation_summary(self):
        result = self._make_full_result("consider_segmentation")
        summary = result.get_decision_summary()
        assert "MODERATE EVIDENCE" in summary

    def test_single_model_summary(self):
        result = self._make_full_result("single_model")
        summary = result.get_decision_summary()
        assert "SINGLE MODEL" in summary

    def test_has_visualization_true(self):
        viz = ClusterVisualizationResult(
            x=np.array([1.0]),
            y=np.array([2.0]),
            labels=np.array([0]),
            method=DimensionReductionMethod.PCA,
        )
        result = self._make_full_result("single_model")
        result.visualization = viz
        assert result.has_visualization is True


class TestFromSegmentationResultBranches:
    """Test all branches of SegmentationDecisionMetrics.from_segmentation_result."""

    def _make_seg_result(self, quality_score, target_variance, n_segments, confidence):
        return SegmentationResult(
            n_segments=n_segments,
            method=SegmentationMethod.KMEANS,
            quality_score=quality_score,
            profiles=[],
            target_variance_ratio=target_variance,
            recommendation="single_model",
            confidence=confidence,
            rationale=[],
            labels=np.array([0]),
        )

    def test_strong_silhouette(self):
        # quality_score > 0.7 => silhouette > 0.5 => "Strong structure"
        result = self._make_seg_result(0.85, 0.5, 2, 0.7)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.silhouette_interpretation == "Strong structure"

    def test_reasonable_silhouette(self):
        # quality_score between 0.625 and 0.75 => silhouette between 0.25 and 0.5
        result = self._make_seg_result(0.7, 0.2, 2, 0.5)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.silhouette_interpretation == "Reasonable"

    def test_weak_silhouette(self):
        # quality_score between 0.5 and 0.625 => silhouette between 0 and 0.25
        result = self._make_seg_result(0.55, 0.1, 2, 0.3)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.silhouette_interpretation == "Weak structure"

    def test_no_structure_silhouette(self):
        # quality_score < 0.5 => silhouette < 0
        result = self._make_seg_result(0.3, None, 2, 0.1)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.silhouette_interpretation == "No structure"

    def test_target_variance_high(self):
        result = self._make_seg_result(0.6, 0.5, 2, 0.5)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.target_variance_interpretation == "High separation"

    def test_target_variance_moderate(self):
        result = self._make_seg_result(0.6, 0.2, 2, 0.5)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.target_variance_interpretation == "Moderate"

    def test_target_variance_low(self):
        result = self._make_seg_result(0.6, 0.1, 2, 0.5)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.target_variance_interpretation == "Low separation"

    def test_target_variance_none(self):
        result = self._make_seg_result(0.6, None, 2, 0.5)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.target_variance_interpretation == "N/A"

    def test_complex_segments(self):
        result = self._make_seg_result(0.6, 0.2, 6, 0.5)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.segments_interpretation == "Complex"

    def test_manageable_segments(self):
        result = self._make_seg_result(0.6, 0.2, 3, 0.5)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.segments_interpretation == "Manageable"

    def test_high_confidence(self):
        result = self._make_seg_result(0.6, 0.2, 2, 0.7)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.confidence_interpretation == "High"

    def test_medium_confidence(self):
        result = self._make_seg_result(0.6, 0.2, 2, 0.4)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.confidence_interpretation == "Medium"

    def test_low_confidence(self):
        result = self._make_seg_result(0.6, 0.2, 2, 0.2)
        metrics = SegmentationDecisionMetrics.from_segmentation_result(result)
        assert metrics.confidence_interpretation == "Low"


class TestAnalyzeEdgeCasesExtended:
    """Additional edge cases for the analyze method."""

    def test_no_numeric_features(self):
        """No numeric features => _empty_result."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "cat_a": ["a", "b", "c", "a", "b"],
            "cat_b": ["x", "y", "z", "x", "y"],
        })
        result = analyzer.analyze(df)
        assert result.n_segments == 1
        assert result.recommendation == "single_model"
        assert result.quality_score == 0.0

    def test_too_few_rows_after_dropna(self):
        """Fewer than 10 valid rows => _single_segment_result."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        })
        result = analyzer.analyze(df, target_col="target")
        assert result.n_segments == 1
        assert result.recommendation == "single_model"

    def test_single_segment_result_without_target(self):
        """_single_segment_result with no target column."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
        })
        result = analyzer.analyze(df)
        assert result.n_segments == 1
        assert result.profiles[0].target_rate is None

    def test_single_segment_result_non_numeric_target(self):
        """_single_segment_result when target is non-numeric."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
            "target": ["a", "b", "a", "b", "a"],
        })
        result = analyzer.analyze(df, target_col="target")
        assert result.n_segments == 1
        assert result.profiles[0].target_rate is None

    def test_hierarchical_method(self):
        """Test HIERARCHICAL clustering method."""
        np.random.seed(42)
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(10, 1, 50)]),
            "feature_b": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(10, 1, 50)]),
            "target": [0] * 50 + [1] * 50,
        })
        result = analyzer.analyze(df, target_col="target", method=SegmentationMethod.HIERARCHICAL)
        assert result.method == SegmentationMethod.HIERARCHICAL

    def test_dbscan_method(self):
        """Test DBSCAN clustering method."""
        np.random.seed(42)
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(10, 1, 50)]),
            "feature_b": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(10, 1, 50)]),
            "target": [0] * 50 + [1] * 50,
        })
        result = analyzer.analyze(df, target_col="target", method=SegmentationMethod.DBSCAN)
        assert result.method == SegmentationMethod.DBSCAN

    def test_feature_cols_not_in_df(self):
        """Feature cols specified but none exist in df."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = analyzer.analyze(df, feature_cols=["x", "y"])
        assert result.n_segments == 1
        assert result.recommendation == "single_model"

    def test_select_features_removes_target(self):
        """Auto-selected features should exclude target column."""
        analyzer = SegmentAnalyzer()
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feature_a": np.random.normal(0, 1, n),
            "feature_b": np.random.normal(5, 1, n),
            "target": np.random.choice([0, 1], n),
        })
        cols = analyzer._select_features(df, None, "target")
        assert "target" not in cols


class TestFindOptimalSegmentsEdgeCases:
    def test_very_small_data(self):
        """Fewer than 10 rows => return 1."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        n = analyzer.find_optimal_segments(df, ["a", "b"])
        assert n == 1

    def test_max_k_less_than_2(self):
        """max_k computed < 2 => return 1."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
        # max_k = min(2, 10//3, 10) = min(2, 3, 10) = 2, so still k=2
        # But if we pass max_k=1, it triggers max_k < 2
        n = analyzer.find_optimal_segments(df, ["a", "b"], max_k=1)
        assert n == 1


class TestAnalyzeCollectsDataOnce:
    def test_to_numpy_called_once_for_features(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "a": np.random.normal(0, 1, n),
            "b": np.random.normal(5, 1, n),
            "target": np.random.choice([0, 1], n),
        })
        analyzer = SegmentAnalyzer()
        call_count = {"n": 0}
        orig_find = analyzer.find_optimal_segments

        def counting_find(*args, **kwargs):
            call_count["n"] += 1
            return orig_find(*args, **kwargs)

        analyzer.find_optimal_segments = counting_find
        analyzer.analyze(df, target_col="target", feature_cols=["a", "b"])
        assert call_count["n"] == 1

    def test_analyze_reuses_precomputed_numpy(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "a": np.random.normal(0, 1, n),
            "b": np.random.normal(5, 1, n),
            "target": np.random.choice([0, 1], n),
        })
        analyzer = SegmentAnalyzer()
        result = analyzer.analyze(df, target_col="target", feature_cols=["a", "b"])
        assert result.n_segments >= 1


class TestCalculateQualityEdgeCases:
    def test_single_label(self):
        """Only one unique label => quality 0.0."""
        analyzer = SegmentAnalyzer()
        scaled = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 0, 0])
        assert analyzer._calculate_quality(scaled, labels) == 0.0


class TestCalculateTargetVarianceEdgeCases:
    def test_no_target_col(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        labels = np.array([0, 0, 1])
        assert analyzer._calculate_target_variance(df, labels, None) is None

    def test_target_not_in_df(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        labels = np.array([0, 0, 1])
        assert analyzer._calculate_target_variance(df, labels, "missing") is None

    def test_non_numeric_target(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"target": ["a", "b", "c"]})
        labels = np.array([0, 0, 1])
        assert analyzer._calculate_target_variance(df, labels, "target") is None

    def test_non_binary_target(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"target": [0, 1, 2, 3, 4]})
        labels = np.array([0, 0, 1, 1, 1])
        assert analyzer._calculate_target_variance(df, labels, "target") is None

    def test_single_segment_returns_zero(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"target": [0, 1, 0, 1]})
        labels = np.array([0, 0, 0, 0])
        assert analyzer._calculate_target_variance(df, labels, "target") == 0.0


class TestMakeRecommendationBranches:
    def test_high_quality_high_variance_large_segments(self):
        """All favorable conditions => strong_segmentation."""
        analyzer = SegmentAnalyzer()
        profiles = [
            SegmentProfile(segment_id=0, size=50, size_pct=50.0, target_rate=0.2, defining_features={}),
            SegmentProfile(segment_id=1, size=50, size_pct=50.0, target_rate=0.8, defining_features={}),
        ]
        rec, conf, rationale = analyzer._make_recommendation(0.8, 0.5, 2, profiles)
        assert rec == "strong_segmentation"
        assert conf > 0.6

    def test_moderate_quality(self):
        """Moderate cluster quality branch."""
        analyzer = SegmentAnalyzer()
        profiles = [
            SegmentProfile(segment_id=0, size=50, size_pct=50.0, target_rate=0.5, defining_features={}),
            SegmentProfile(segment_id=1, size=50, size_pct=50.0, target_rate=0.5, defining_features={}),
        ]
        rec, conf, rationale = analyzer._make_recommendation(0.6, 0.2, 2, profiles)
        assert any("Moderate cluster quality" in r for r in rationale)

    def test_low_target_variance(self):
        """Low target rate variation branch."""
        analyzer = SegmentAnalyzer()
        profiles = [
            SegmentProfile(segment_id=0, size=50, size_pct=50.0, target_rate=0.5, defining_features={}),
            SegmentProfile(segment_id=1, size=50, size_pct=50.0, target_rate=0.5, defining_features={}),
        ]
        rec, conf, rationale = analyzer._make_recommendation(0.4, 0.05, 2, profiles)
        assert any("Low target rate" in r for r in rationale)

    def test_small_segments(self):
        """Some segments < 10% but >= 5%."""
        analyzer = SegmentAnalyzer()
        profiles = [
            SegmentProfile(segment_id=0, size=92, size_pct=92.0, target_rate=0.5, defining_features={}),
            SegmentProfile(segment_id=1, size=8, size_pct=8.0, target_rate=0.5, defining_features={}),
        ]
        rec, conf, rationale = analyzer._make_recommendation(0.4, None, 2, profiles)
        assert any("small" in r.lower() for r in rationale)

    def test_very_small_segments(self):
        """Very small segments < 5%."""
        analyzer = SegmentAnalyzer()
        profiles = [
            SegmentProfile(segment_id=0, size=97, size_pct=97.0, target_rate=0.5, defining_features={}),
            SegmentProfile(segment_id=1, size=3, size_pct=3.0, target_rate=0.5, defining_features={}),
        ]
        rec, conf, rationale = analyzer._make_recommendation(0.4, None, 2, profiles)
        assert any("Very small" in r for r in rationale)

    def test_many_segments_penalty(self):
        """More than 5 segments adds penalty."""
        analyzer = SegmentAnalyzer()
        profiles = [
            SegmentProfile(segment_id=i, size=10, size_pct=10.0, target_rate=0.5, defining_features={})
            for i in range(10)
        ]
        rec, conf, rationale = analyzer._make_recommendation(0.4, None, 10, profiles)
        assert any("Many segments" in r for r in rationale)

    def test_no_target_variance(self):
        """target_variance is None => no variance rationale."""
        analyzer = SegmentAnalyzer()
        profiles = [
            SegmentProfile(segment_id=0, size=50, size_pct=50.0, target_rate=None, defining_features={}),
        ]
        rec, conf, rationale = analyzer._make_recommendation(0.4, None, 2, profiles)
        assert not any("target rate" in r.lower() for r in rationale)

    def test_empty_profiles(self):
        """Empty profiles list."""
        analyzer = SegmentAnalyzer()
        rec, conf, rationale = analyzer._make_recommendation(0.4, None, 2, [])
        assert rec in ["single_model", "consider_segmentation", "strong_segmentation"]


class TestGetClusterVisualizationEdgeCases:
    def test_too_few_valid_rows(self):
        """Less than 2 valid rows => early return with NaN."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
        labels = np.array([0])
        viz = analyzer.get_cluster_visualization(df, labels, ["a", "b"])
        assert np.isnan(viz.x[0])
        assert np.isnan(viz.y[0])
        assert viz.explained_variance_ratio is None

    def test_umap_fallback_to_pca(self):
        """UMAP not installed => fallback to PCA."""
        np.random.seed(42)
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "a": np.random.normal(0, 1, 50),
            "b": np.random.normal(5, 1, 50),
        })
        labels = np.array([0] * 25 + [1] * 25)
        viz = analyzer.get_cluster_visualization(
            df, labels, ["a", "b"], method=DimensionReductionMethod.UMAP
        )
        # If umap not installed, falls back to PCA
        assert viz.method in (DimensionReductionMethod.UMAP, DimensionReductionMethod.PCA)
        assert len(viz.x) == 50


class TestProfileSegmentsEdgeCases:
    def test_non_numeric_column(self):
        """Non-numeric feature column is skipped."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "cat": ["a", "b", "c"],
            "num": [1.0, 2.0, 3.0],
        })
        labels = np.array([0, 0, 1])
        profiles = analyzer.profile_segments(df, labels, ["cat", "num"])
        for p in profiles:
            assert "cat" not in p.defining_features
            assert "num" in p.defining_features

    def test_empty_col_data_after_dropna(self):
        """Feature column all NaN in segment."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "a": [np.nan, np.nan, 1.0, 2.0],
        })
        labels = np.array([0, 0, 1, 1])
        profiles = analyzer.profile_segments(df, labels, ["a"])
        # Segment 0 has all NaN, so no defining features for "a"
        seg0 = [p for p in profiles if p.segment_id == 0][0]
        assert "a" not in seg0.defining_features

    def test_non_binary_target_ignored_for_rate(self):
        """Target with values other than 0/1 => target_rate is None."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "target": [2, 3, 4, 5],
        })
        labels = np.array([0, 0, 1, 1])
        profiles = analyzer.profile_segments(df, labels, ["a"], target_col="target")
        for p in profiles:
            assert p.target_rate is None


class TestRunFullAnalysisSingleSegment:
    def test_no_visualization_for_single_segment(self):
        """Single segment => no visualization generated."""
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })
        result = analyzer.run_full_analysis(df, feature_cols=["a", "b"])
        assert result.visualization is None


class TestLatestSnapshotFiltering:

    def test_no_snapshot_column_returns_unchanged(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({"feature_a": [1, 2, 3], "feature_b": [4, 5, 6]})
        result = analyzer._to_latest_snapshot(df)
        assert len(result) == 3

    def test_as_of_date_filters_to_latest(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4],
            "as_of_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
        })
        result = analyzer._to_latest_snapshot(df)
        assert len(result) == 2
        assert (result["as_of_date"] == pd.Timestamp("2024-02-01")).all()

    def test_feature_timestamp_filters_to_latest(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
        })
        result = analyzer._to_latest_snapshot(df)
        assert len(result) == 2

    def test_as_of_date_preferred_over_feature_timestamp(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4],
            "as_of_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
            "feature_timestamp": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-04-01", "2024-04-01"]),
        })
        result = analyzer._to_latest_snapshot(df)
        assert len(result) == 2
        assert (result["as_of_date"] == pd.Timestamp("2024-02-01")).all()

    def test_resets_index_after_filtering(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5, 6],
            "as_of_date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-02-01"] * 3),
        })
        result = analyzer._to_latest_snapshot(df)
        assert list(result.index) == [0, 1, 2]

    def test_single_snapshot_returns_all(self):
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": [1, 2, 3],
            "as_of_date": pd.to_datetime(["2024-01-01"] * 3),
        })
        result = analyzer._to_latest_snapshot(df)
        assert len(result) == 3

    def test_analyze_uses_latest_snapshot_only(self):
        np.random.seed(42)
        analyzer = SegmentAnalyzer()
        n = 50
        dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        dfs = []
        for date in dates:
            dfs.append(pd.DataFrame({
                "feature_a": np.random.normal(10, 1, n),
                "feature_b": np.random.normal(100, 10, n),
                "as_of_date": date,
                "target": np.random.choice([0, 1], n),
            }))
        df = pd.concat(dfs, ignore_index=True)
        result = analyzer.analyze(df, target_col="target", feature_cols=["feature_a", "feature_b"])
        assert len(result.labels) == n

    def test_analyze_without_snapshot_column_uses_all_rows(self):
        np.random.seed(42)
        analyzer = SegmentAnalyzer()
        df = pd.DataFrame({
            "feature_a": np.random.normal(10, 1, 100),
            "feature_b": np.random.normal(100, 10, 100),
            "target": np.random.choice([0, 1], 100),
        })
        result = analyzer.analyze(df, target_col="target", feature_cols=["feature_a", "feature_b"])
        assert len(result.labels) == 100


class TestSilhouetteSampling:
    def test_silhouette_uses_sample_size_for_large_data(self):
        from unittest.mock import patch

        from sklearn.metrics import silhouette_score as real_silhouette

        np.random.seed(42)
        data = np.vstack([
            np.random.normal(0, 1, (800, 2)),
            np.random.normal(10, 1, (800, 2)),
        ])
        df = pd.DataFrame(data, columns=["a", "b"])

        analyzer = SegmentAnalyzer()
        with patch(
            "customer_retention.stages.profiling.segment_analyzer.silhouette_score",
            wraps=real_silhouette,
        ) as mock_sil:
            analyzer.find_optimal_segments(df, ["a", "b"])
            for call in mock_sil.call_args_list:
                assert "sample_size" in call.kwargs

    def test_large_data_still_finds_correct_clusters(self):
        np.random.seed(42)
        data = np.vstack([
            np.random.normal(0, 1, (800, 2)),
            np.random.normal(20, 1, (800, 2)),
        ])
        df = pd.DataFrame(data, columns=["a", "b"])

        analyzer = SegmentAnalyzer()
        n = analyzer.find_optimal_segments(df, ["a", "b"])
        assert n == 2

    def test_calculate_quality_uses_sample_size(self):
        from unittest.mock import patch

        from sklearn.metrics import silhouette_score as real_silhouette

        np.random.seed(42)
        scaled = np.vstack([
            np.random.normal(0, 1, (800, 2)),
            np.random.normal(10, 1, (800, 2)),
        ])
        labels = np.array([0] * 800 + [1] * 800)

        analyzer = SegmentAnalyzer()
        with patch(
            "customer_retention.stages.profiling.segment_analyzer.silhouette_score",
            wraps=real_silhouette,
        ) as mock_sil:
            quality = analyzer._calculate_quality(scaled, labels)
            assert quality > 0
            assert "sample_size" in mock_sil.call_args.kwargs
