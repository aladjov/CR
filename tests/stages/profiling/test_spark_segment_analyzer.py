import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.segment_analyzer import (
    FullSegmentationResult,
    SegmentAnalyzer,
    SegmentationMethod,
    SegmentationResult,
    SegmentProfile,
)
from customer_retention.stages.profiling.spark_segment_analyzer import (
    SparkSegmentAnalyzer,
    _propagate_labels,
    _stratified_sample,
)


@pytest.fixture
def clustered_entity_data():
    np.random.seed(42)
    n = 300
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
def large_clustered_data():
    np.random.seed(42)
    n = 600
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


FEATURE_COLS = ["feature_a", "feature_b", "feature_c"]


class TestBehavioralEquivalence:

    @pytest.mark.parametrize("analyzer_cls", [SegmentAnalyzer, SparkSegmentAnalyzer])
    def test_analyze_returns_segmentation_result(self, analyzer_cls, clustered_entity_data):
        analyzer = analyzer_cls()
        result = analyzer.analyze(clustered_entity_data, target_col="target")
        assert isinstance(result, SegmentationResult)

    @pytest.mark.parametrize("analyzer_cls", [SegmentAnalyzer, SparkSegmentAnalyzer])
    def test_detects_multiple_segments(self, analyzer_cls, clustered_entity_data):
        analyzer = analyzer_cls()
        result = analyzer.analyze(
            clustered_entity_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        assert result.n_segments >= 2

    @pytest.mark.parametrize("analyzer_cls", [SegmentAnalyzer, SparkSegmentAnalyzer])
    def test_quality_in_valid_range(self, analyzer_cls, clustered_entity_data):
        analyzer = analyzer_cls()
        result = analyzer.analyze(
            clustered_entity_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        assert 0 <= result.quality_score <= 1

    @pytest.mark.parametrize("analyzer_cls", [SegmentAnalyzer, SparkSegmentAnalyzer])
    def test_profiles_present(self, analyzer_cls, clustered_entity_data):
        analyzer = analyzer_cls()
        result = analyzer.analyze(
            clustered_entity_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        assert len(result.profiles) == result.n_segments
        assert all(isinstance(p, SegmentProfile) for p in result.profiles)

    @pytest.mark.parametrize("analyzer_cls", [SegmentAnalyzer, SparkSegmentAnalyzer])
    def test_recommendation_valid(self, analyzer_cls, clustered_entity_data):
        analyzer = analyzer_cls()
        result = analyzer.analyze(
            clustered_entity_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        assert result.recommendation in [
            "single_model", "consider_segmentation", "strong_segmentation",
        ]

    @pytest.mark.parametrize("analyzer_cls", [SegmentAnalyzer, SparkSegmentAnalyzer])
    def test_works_without_target(self, analyzer_cls, clustered_entity_data):
        analyzer = analyzer_cls()
        df = clustered_entity_data.drop(columns=["target"])
        result = analyzer.analyze(df, feature_cols=FEATURE_COLS)
        assert isinstance(result, SegmentationResult)
        for p in result.profiles:
            assert p.target_rate is None


class TestSamplingLogic:

    def test_below_threshold_identical_to_parent(self, clustered_entity_data):
        parent = SegmentAnalyzer()
        spark = SparkSegmentAnalyzer(max_sample_size=500)
        parent_result = parent.analyze(
            clustered_entity_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        spark_result = spark.analyze(
            clustered_entity_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        assert parent_result.n_segments == spark_result.n_segments
        assert np.array_equal(parent_result.labels, spark_result.labels)

    def test_above_threshold_still_detects_clusters(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.analyze(
            large_clustered_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        assert result.n_segments >= 2

    def test_labels_array_full_length(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.analyze(
            large_clustered_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        assert len(result.labels) == len(large_clustered_data)

    def test_all_non_nan_rows_get_labels(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.analyze(
            large_clustered_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        features_valid = large_clustered_data[FEATURE_COLS].dropna()
        valid_indices = features_valid.index.to_numpy()
        assert all(result.labels[i] >= 0 for i in valid_indices)

    def test_cluster_count_preserved(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.analyze(
            large_clustered_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        unique_labels = set(result.labels[result.labels >= 0])
        assert len(unique_labels) == result.n_segments

    def test_max_sample_size_per_call_override(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=10000)
        result = analyzer.analyze(
            large_clustered_data,
            target_col="target",
            feature_cols=FEATURE_COLS,
            max_sample_size=100,
        )
        assert len(result.labels) == len(large_clustered_data)
        assert result.n_segments >= 2

    def test_deterministic_output(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        r1 = analyzer.analyze(
            large_clustered_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        r2 = analyzer.analyze(
            large_clustered_data, target_col="target", feature_cols=FEATURE_COLS,
        )
        assert np.array_equal(r1.labels, r2.labels)
        assert r1.n_segments == r2.n_segments


class TestStratifiedSample:

    def test_target_distribution_preserved(self, large_clustered_data):
        original_rate = large_clustered_data["target"].mean()
        sample = _stratified_sample(large_clustered_data, "target", max_size=100)
        sample_rate = sample["target"].mean()
        assert abs(original_rate - sample_rate) < 0.05

    def test_no_target_random_sampling(self, large_clustered_data):
        sample = _stratified_sample(large_clustered_data, None, max_size=100)
        assert len(sample) == 100

    def test_never_exceeds_max_size(self, large_clustered_data):
        sample = _stratified_sample(large_clustered_data, "target", max_size=50)
        assert len(sample) <= 50

    def test_small_df_returns_full(self):
        df = pd.DataFrame({
            "feature_a": [1, 2, 3],
            "target": [0, 1, 0],
        })
        sample = _stratified_sample(df, "target", max_size=100)
        assert len(sample) == len(df)

    def test_missing_target_column_falls_back(self, large_clustered_data):
        sample = _stratified_sample(large_clustered_data, "nonexistent", max_size=100)
        assert len(sample) == 100


class TestPropagateLabels:

    def test_sampled_rows_keep_original_labels(self):
        np.random.seed(42)
        full_df = pd.DataFrame({
            "f1": np.random.normal(0, 1, 50),
            "f2": np.random.normal(0, 1, 50),
        })
        sample_indices = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
        sample_df = full_df.iloc[sample_indices].copy()
        sample_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(full_df[["f1", "f2"]].to_numpy())

        propagated = _propagate_labels(
            full_df, sample_df, sample_labels, ["f1", "f2"], scaler,
        )
        for i, idx in enumerate(sample_indices):
            assert propagated[idx] == sample_labels[i]

    def test_unsampled_rows_get_valid_cluster_ids(self):
        np.random.seed(42)
        full_df = pd.DataFrame({
            "f1": np.concatenate([np.random.normal(0, 0.1, 25), np.random.normal(10, 0.1, 25)]),
            "f2": np.concatenate([np.random.normal(0, 0.1, 25), np.random.normal(10, 0.1, 25)]),
        })
        sample_indices = [0, 1, 2, 25, 26, 27]
        sample_df = full_df.iloc[sample_indices].copy()
        sample_labels = np.array([0, 0, 0, 1, 1, 1])

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(full_df[["f1", "f2"]].to_numpy())

        propagated = _propagate_labels(
            full_df, sample_df, sample_labels, ["f1", "f2"], scaler,
        )
        valid_labels = set(sample_labels)
        for i in range(len(full_df)):
            assert propagated[i] in valid_labels

    def test_nan_feature_rows_get_minus_one(self):
        full_df = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0],
            "f2": [1.0, 2.0, np.nan, 4.0],
        })
        sample_df = full_df.iloc[[0, 3]].copy()
        sample_labels = np.array([0, 1])

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        valid = full_df[["f1", "f2"]].dropna()
        scaler.fit(valid.to_numpy())

        propagated = _propagate_labels(
            full_df, sample_df, sample_labels, ["f1", "f2"], scaler,
        )
        assert propagated[1] == -1
        assert propagated[2] == -1
        assert propagated[0] >= 0
        assert propagated[3] >= 0


class TestRunFullAnalysisWithSampling:

    def test_returns_full_segmentation_result(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.run_full_analysis(
            large_clustered_data,
            target_col="target",
            feature_cols=FEATURE_COLS,
        )
        assert isinstance(result, FullSegmentationResult)

    def test_profiles_reflect_full_dataset_sizes(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.run_full_analysis(
            large_clustered_data,
            target_col="target",
            feature_cols=FEATURE_COLS,
        )
        total_in_profiles = sum(p.size for p in result.profiles)
        assert total_in_profiles == len(large_clustered_data)

    def test_size_distribution_total_equals_full_row_count(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.run_full_analysis(
            large_clustered_data,
            target_col="target",
            feature_cols=FEATURE_COLS,
        )
        assert result.size_distribution["total"] == len(large_clustered_data)

    def test_max_sample_size_passthrough(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=10000)
        result = analyzer.run_full_analysis(
            large_clustered_data,
            target_col="target",
            feature_cols=FEATURE_COLS,
            max_sample_size=100,
        )
        assert isinstance(result, FullSegmentationResult)
        assert result.size_distribution["total"] == len(large_clustered_data)


class TestEdgeCases:

    def test_empty_df(self):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        df = pd.DataFrame({"feature_a": pd.Series(dtype=float), "target": pd.Series(dtype=int)})
        result = analyzer.analyze(df, target_col="target", feature_cols=["feature_a"])
        assert isinstance(result, SegmentationResult)

    def test_single_feature(self):
        np.random.seed(42)
        analyzer = SparkSegmentAnalyzer(max_sample_size=50)
        df = pd.DataFrame({
            "feature_a": np.concatenate([
                np.random.normal(0, 1, 50),
                np.random.normal(20, 1, 50),
            ]),
            "target": [0] * 50 + [1] * 50,
        })
        result = analyzer.analyze(
            df, target_col="target", feature_cols=["feature_a"], max_sample_size=40,
        )
        assert isinstance(result, SegmentationResult)
        assert len(result.labels) == 100

    def test_all_nan_features(self):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        df = pd.DataFrame({
            "feature_a": [np.nan] * 20,
            "feature_b": [np.nan] * 20,
            "target": [0] * 10 + [1] * 10,
        })
        result = analyzer.analyze(
            df, target_col="target", feature_cols=["feature_a", "feature_b"],
        )
        assert isinstance(result, SegmentationResult)

    def test_homogeneous_data(self):
        np.random.seed(42)
        analyzer = SparkSegmentAnalyzer(max_sample_size=50)
        df = pd.DataFrame({
            "feature_a": np.random.normal(50, 0.01, 100),
            "feature_b": np.random.normal(50, 0.01, 100),
            "target": np.random.choice([0, 1], 100),
        })
        result = analyzer.analyze(
            df, target_col="target", feature_cols=["feature_a", "feature_b"],
            max_sample_size=40,
        )
        assert isinstance(result, SegmentationResult)

    def test_hierarchical_with_sampling(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.analyze(
            large_clustered_data,
            target_col="target",
            feature_cols=FEATURE_COLS,
            method=SegmentationMethod.HIERARCHICAL,
        )
        assert result.method == SegmentationMethod.HIERARCHICAL
        assert len(result.labels) == len(large_clustered_data)

    def test_dbscan_with_sampling(self, large_clustered_data):
        analyzer = SparkSegmentAnalyzer(max_sample_size=100)
        result = analyzer.analyze(
            large_clustered_data,
            target_col="target",
            feature_cols=FEATURE_COLS,
            method=SegmentationMethod.DBSCAN,
        )
        assert result.method == SegmentationMethod.DBSCAN
        assert len(result.labels) == len(large_clustered_data)


class TestInit:

    def test_default_max_sample_size(self):
        analyzer = SparkSegmentAnalyzer()
        assert analyzer.max_sample_size == 50_000

    def test_custom_max_sample_size(self):
        analyzer = SparkSegmentAnalyzer(max_sample_size=10_000)
        assert analyzer.max_sample_size == 10_000

    def test_inherits_from_segment_analyzer(self):
        analyzer = SparkSegmentAnalyzer()
        assert isinstance(analyzer, SegmentAnalyzer)

    def test_custom_default_method(self):
        analyzer = SparkSegmentAnalyzer(
            default_method=SegmentationMethod.HIERARCHICAL, max_sample_size=1000,
        )
        assert analyzer.default_method == SegmentationMethod.HIERARCHICAL
        assert analyzer.max_sample_size == 1000
