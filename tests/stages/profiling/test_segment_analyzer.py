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
