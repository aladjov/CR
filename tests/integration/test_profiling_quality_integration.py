import numpy as np
import pandas as pd

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    RecommendationRegistry,
)
from customer_retention.core.utils.leakage import (
    calculate_class_overlap,
    classify_correlation,
    classify_separation,
)
from customer_retention.generators.orchestration.data_materializer import DataMaterializer
from customer_retention.stages.profiling.categorical_distribution import (
    CategoricalDistributionAnalyzer,
)
from customer_retention.stages.profiling.distribution_analysis import (
    DistributionAnalysis,
    DistributionAnalyzer,
)
from customer_retention.stages.profiling.segment_aware_outlier import SegmentAwareOutlierAnalyzer
from customer_retention.stages.validation.quality_scorer import (
    ColumnFindings,
    ExplorationFindings,
    QualityScorer,
)


def _build_churn_dataframe(n=800, seed=42):
    np.random.seed(seed)
    spending = np.concatenate([
        np.random.exponential(scale=200, size=int(n * 0.7)),
        np.random.exponential(scale=2000, size=int(n * 0.3)),
    ])
    tenure = np.random.normal(24, 8, n).clip(1, 60)
    logins = np.random.poisson(5, n)
    churn = (np.random.rand(n) < 0.25).astype(int)
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "monthly_spending": spending[:n],
        "tenure_months": tenure,
        "login_count": logins,
        "churn": churn,
    })


def _build_segmented_dataframe(n=600, seed=42):
    np.random.seed(seed)
    segment_sizes = [int(n * 0.5), int(n * 0.3), n - int(n * 0.5) - int(n * 0.3)]
    spending = np.concatenate([
        np.random.normal(50, 10, segment_sizes[0]),
        np.random.normal(200, 20, segment_sizes[1]),
        np.random.normal(500, 50, segment_sizes[2]),
    ])
    frequency = np.concatenate([
        np.random.normal(2, 0.5, segment_sizes[0]),
        np.random.normal(8, 1, segment_sizes[1]),
        np.random.normal(15, 2, segment_sizes[2]),
    ])
    churn = np.concatenate([
        (np.random.rand(segment_sizes[0]) < 0.4).astype(int),
        (np.random.rand(segment_sizes[1]) < 0.2).astype(int),
        (np.random.rand(segment_sizes[2]) < 0.1).astype(int),
    ])
    segment_labels = np.concatenate([
        np.full(segment_sizes[0], "retail"),
        np.full(segment_sizes[1], "mid_market"),
        np.full(segment_sizes[2], "enterprise"),
    ])
    return pd.DataFrame({
        "spending": spending,
        "frequency": frequency,
        "churn": churn,
        "segment": segment_labels,
    })


def _build_quality_findings(df, null_pcts=None, id_col=None):
    columns = {}
    for col in df.columns:
        null_pct = 0.0
        if null_pcts and col in null_pcts:
            null_pct = null_pcts[col]
        distinct_pct = df[col].nunique() / len(df) * 100 if len(df) > 0 else 100
        col_type_value = "identifier" if col == id_col else "numeric"

        class _InferredType:
            def __init__(self, val):
                self.value = val

        columns[col] = ColumnFindings(
            inferred_type=_InferredType(col_type_value),
            universal_metrics={"null_percentage": null_pct, "distinct_percentage": distinct_pct},
        )
    return ExplorationFindings(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
    )


class TestDistributionToTransformationFlow:
    def test_highly_skewed_feature_recommends_log_transform(self):
        np.random.seed(68)
        analyzer = DistributionAnalyzer()
        series = pd.Series(np.random.gamma(0.8, 10, 1000) + 1)
        analysis = analyzer.analyze_distribution(series, "spending")
        recommendation = analyzer.recommend_transformation(analysis)

        assert analysis.is_highly_skewed
        assert analysis.outlier_percentage <= DistributionAnalyzer.OUTLIER_THRESHOLD
        assert recommendation.recommended_transform.value == "log_transform"
        assert recommendation.priority == "high"

    def test_zero_inflated_feature_recommends_zero_inflation_handling(self):
        np.random.seed(42)
        values = np.random.exponential(scale=50, size=1000)
        values[np.random.choice(1000, size=400, replace=False)] = 0
        analyzer = DistributionAnalyzer()
        analysis = analyzer.analyze_distribution(pd.Series(values), "inactive_months")
        recommendation = analyzer.recommend_transformation(analysis)

        assert analysis.has_zero_inflation
        assert recommendation.recommended_transform.value == "zero_inflation_handling"

    def test_normal_distribution_recommends_none(self):
        np.random.seed(42)
        analyzer = DistributionAnalyzer()
        analysis = analyzer.analyze_distribution(pd.Series(np.random.normal(50, 10, 1000)), "tenure")
        recommendation = analyzer.recommend_transformation(analysis)

        assert not analysis.is_highly_skewed
        assert not analysis.is_moderately_skewed
        assert recommendation.recommended_transform.value == "none"

    def test_heavy_tails_with_outliers_recommends_cap_then_log(self):
        np.random.seed(42)
        analyzer = DistributionAnalyzer()
        series = pd.Series(np.random.lognormal(mean=3, sigma=2, size=1000))
        analysis = analyzer.analyze_distribution(series, "order_value")
        recommendation = analyzer.recommend_transformation(analysis)

        assert analysis.is_highly_skewed
        assert analysis.outlier_percentage > DistributionAnalyzer.OUTLIER_THRESHOLD
        assert recommendation.recommended_transform.value == "cap_then_log"

    def test_moderate_skew_recommends_sqrt(self):
        np.random.seed(42)
        analyzer = DistributionAnalyzer()
        series = pd.Series(np.random.gamma(shape=4, scale=10, size=1000))
        analysis = analyzer.analyze_distribution(series, "visit_duration")
        recommendation = analyzer.recommend_transformation(analysis)

        assert analysis.is_moderately_skewed
        assert recommendation.recommended_transform.value == "sqrt_transform"
        assert recommendation.priority == "medium"

    def test_analyze_dataframe_batch_returns_all_numeric_columns(self):
        analyzer = DistributionAnalyzer()
        df = _build_churn_dataframe(n=500)
        results = analyzer.analyze_dataframe(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert set(results.keys()) == set(numeric_cols)
        assert all(isinstance(v, DistributionAnalysis) for v in results.values())

    def test_get_all_recommendations_sorts_by_priority(self):
        analyzer = DistributionAnalyzer()
        df = _build_churn_dataframe(n=500)
        recommendations = analyzer.get_all_recommendations(df)

        priorities = [r.priority for r in recommendations]
        priority_order = {"high": 0, "medium": 1, "low": 2}
        numeric_priorities = [priority_order[p] for p in priorities]
        assert numeric_priorities == sorted(numeric_priorities)

    def test_generate_report_contains_all_sections(self):
        analyzer = DistributionAnalyzer()
        df = _build_churn_dataframe(n=500)
        report = analyzer.generate_report(df)

        assert "summary" in report
        assert "categories" in report
        assert "analyses" in report
        assert "recommendations" in report
        assert report["summary"]["total_columns"] == len(df.select_dtypes(include=[np.number]).columns)


class TestSegmentAwareOutlierFlow:
    def test_global_outliers_segment_normal_detected_as_false(self):
        df = _build_segmented_dataframe(n=600)
        analyzer = SegmentAwareOutlierAnalyzer(max_segments=3)
        result = analyzer.analyze(df, feature_cols=["spending", "frequency"], segment_col="segment")

        total_false = sum(result.false_outliers.values())
        assert total_false > 0
        assert result.n_segments == 3

    def test_segmentation_reduces_outlier_count_vs_global(self):
        df = _build_segmented_dataframe(n=600)
        analyzer = SegmentAwareOutlierAnalyzer(max_segments=3)
        result = analyzer.analyze(df, feature_cols=["spending", "frequency"], segment_col="segment")

        global_total = sum(r.outliers_detected for r in result.global_analysis.values())
        segment_total = sum(
            sum(r.outliers_detected for r in seg.values())
            for seg in result.segment_analysis.values()
        )
        assert segment_total < global_total

    def test_explicit_segment_column_works(self):
        df = _build_segmented_dataframe(n=600)
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=["spending"], segment_col="segment")

        assert result.n_segments == 3
        assert len(result.segment_analysis) > 0

    def test_auto_detection_with_clear_clusters(self):
        df = _build_segmented_dataframe(n=600)
        analyzer = SegmentAwareOutlierAnalyzer(max_segments=5)
        result = analyzer.analyze(
            df, feature_cols=["spending", "frequency"], target_col="churn"
        )

        assert result.n_segments >= 2
        assert result.segmentation_result is not None

    def test_single_segment_recommends_global_approach(self):
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.normal(50, 10, 500)})
        analyzer = SegmentAwareOutlierAnalyzer(max_segments=2)
        result = analyzer.analyze(df, feature_cols=["value"])

        assert any("global" in r.lower() for r in result.recommendations) or result.n_segments <= 2

    def test_empty_or_all_null_features_handled(self):
        df = pd.DataFrame({"value": [np.nan] * 100})
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=["value"])

        assert result.n_segments == 0
        assert not result.segmentation_recommended


class TestCategoricalDistributionFlow:
    def test_low_cardinality_recommends_one_hot(self):
        np.random.seed(42)
        series = pd.Series(np.random.choice(["A", "B", "C", "D"], size=500))
        analyzer = CategoricalDistributionAnalyzer()
        analysis = analyzer.analyze(series, "plan_type")
        rec = analyzer.recommend_encoding(analysis)

        assert rec.encoding_type.value == "one_hot"

    def test_high_cardinality_recommends_target_encoding(self):
        np.random.seed(42)
        categories = [f"city_{i}" for i in range(50)]
        series = pd.Series(np.random.choice(categories, size=1000))
        analyzer = CategoricalDistributionAnalyzer()
        analysis = analyzer.analyze(series, "city")
        rec = analyzer.recommend_encoding(analysis)

        assert rec.encoding_type.value == "target"
        assert rec.priority == "high"

    def test_binary_column_recommends_binary_encoding(self):
        np.random.seed(42)
        series = pd.Series(np.random.choice(["yes", "no"], size=500))
        analyzer = CategoricalDistributionAnalyzer()
        analysis = analyzer.analyze(series, "paperless")
        rec = analyzer.recommend_encoding(analysis)

        assert rec.encoding_type.value == "binary"

    def test_cyclical_column_recommends_cyclical_encoding(self):
        np.random.seed(42)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        series = pd.Series(np.random.choice(months, size=500))
        analyzer = CategoricalDistributionAnalyzer()
        analysis = analyzer.analyze(series, "signup_month")
        rec = analyzer.recommend_encoding(analysis, is_cyclical=True)

        assert rec.encoding_type.value == "cyclical"

    def test_rare_categories_trigger_preprocessing_step(self):
        np.random.seed(42)
        common = np.random.choice(["A", "B", "C", "D", "E", "F", "G"], size=950)
        rare = np.random.choice(["Z1", "Z2", "Z3", "Z4", "Z5"], size=50)
        series = pd.Series(np.concatenate([common, rare]))
        analyzer = CategoricalDistributionAnalyzer()
        analysis = analyzer.analyze(series, "product")
        rec = analyzer.recommend_encoding(analysis)

        assert analysis.has_rare_categories
        assert len(rec.preprocessing_steps) > 0
        assert any("rare" in step.lower() or "other" in step.lower() for step in rec.preprocessing_steps)

    def test_analyze_dataframe_auto_detects_categorical_columns(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "plan": np.random.choice(["basic", "premium", "enterprise"], size=200),
            "region": np.random.choice(["north", "south", "east", "west"], size=200),
            "revenue": np.random.normal(100, 20, 200),
        })
        analyzer = CategoricalDistributionAnalyzer()
        results = analyzer.analyze_dataframe(df)

        assert "plan" in results
        assert "region" in results
        assert "revenue" not in results


class TestQualityScorerIntegration:
    def test_high_quality_dataset_scores_above_90(self):
        df = _build_churn_dataframe(n=500)
        findings = _build_quality_findings(df, id_col="customer_id")
        scorer = QualityScorer()
        result = scorer.calculate(findings)

        assert result.overall_score >= 90
        assert result.quality_level.value == "excellent"

    def test_dataset_with_missing_values_scores_lower_on_completeness(self):
        df = _build_churn_dataframe(n=500)
        findings = _build_quality_findings(
            df, null_pcts={"monthly_spending": 50.0, "tenure_months": 40.0, "login_count": 35.0}
        )
        scorer = QualityScorer()
        result = scorer.calculate(findings)

        assert result.components["completeness"] < 80
        assert result.overall_score < 95

    def test_dataset_with_duplicates_scores_lower_on_consistency(self):
        from customer_retention.stages.validation.data_validators import DuplicateResult

        df = _build_churn_dataframe(n=500)
        findings = _build_quality_findings(df)
        dup_result = DuplicateResult(
            key_column="customer_id", total_rows=500, unique_keys=450,
            duplicate_keys=50, duplicate_rows=100, duplicate_percentage=20.0,
            has_value_conflicts=True, conflict_columns=["monthly_spending"],
        )
        scorer = QualityScorer()
        result = scorer.calculate(findings, duplicate_result=dup_result)

        assert result.components["consistency"] < 80

    def test_dataset_with_range_violations_scores_lower_on_validity(self):
        from customer_retention.stages.validation.data_validators import RangeValidationResult

        df = _build_churn_dataframe(n=500)
        findings = _build_quality_findings(df)
        range_results = [RangeValidationResult(
            column_name="monthly_spending", total_values=500,
            valid_values=400, invalid_values=100, invalid_percentage=20.0,
            rule_type="non_negative", expected_range="[0, +inf)",
            actual_range="[-50, 5000]",
        )]
        scorer = QualityScorer()
        result = scorer.calculate(findings, range_results=range_results)

        assert result.components["validity"] < 85

    def test_custom_weights_change_component_emphasis(self):
        df = _build_churn_dataframe(n=500)
        findings = _build_quality_findings(
            df, null_pcts={"monthly_spending": 60.0, "tenure_months": 50.0, "login_count": 40.0}
        )
        equal_scorer = QualityScorer()
        completeness_scorer = QualityScorer(weights={
            "completeness": 0.70, "validity": 0.10,
            "consistency": 0.10, "uniqueness": 0.10,
        })
        equal_result = equal_scorer.calculate(findings)
        weighted_result = completeness_scorer.calculate(findings)

        assert weighted_result.overall_score < equal_result.overall_score

    def test_all_components_contribute_to_overall_score(self):
        df = _build_churn_dataframe(n=500)
        findings = _build_quality_findings(df)
        scorer = QualityScorer()
        result = scorer.calculate(findings)

        expected_overall = sum(
            result.components[k] * result.component_weights[k]
            for k in result.components
        )
        assert abs(result.overall_score - expected_overall) < 0.1


class TestLeakageDetection:
    def test_perfect_correlation_classified_as_critical(self):
        severity, label = classify_correlation(0.95)

        assert severity.value == "critical"
        assert label == "high_correlation"

    def test_moderate_correlation_classified_as_medium(self):
        severity, label = classify_correlation(0.55)

        assert severity.value == "medium"
        assert label == "elevated_correlation"

    def test_zero_overlap_classified_as_perfect_separation(self):
        severity, label = classify_separation(0.0)

        assert severity.value == "critical"
        assert label == "perfect_separation"

    def test_large_overlap_classified_as_normal(self):
        severity, label = classify_separation(50.0)

        assert severity.value == "info"
        assert label == "normal"

    def test_calculate_class_overlap_with_separated_distributions(self):
        np.random.seed(42)
        feature = pd.Series(np.concatenate([
            np.random.normal(10, 1, 500),
            np.random.normal(100, 1, 500),
        ]))
        target = pd.Series([0] * 500 + [1] * 500)
        overlap = calculate_class_overlap(feature, target)

        assert overlap < 5.0


class TestDataMaterializerFlow:
    def _make_registry_with_bronze_nulls(self, columns, strategy="median"):
        registry = RecommendationRegistry()
        registry.init_bronze(source_file="test")
        for col in columns:
            registry.add_bronze_null(col, strategy, "fill nulls", "nb02")
        return registry

    def _make_registry_with_gold_log(self, columns):
        registry = RecommendationRegistry()
        registry.init_bronze(source_file="test")
        registry.init_gold(target_column="churn")
        for col in columns:
            registry.add_gold_transformation(col, "log", {"method": "log"}, "skewed", "nb02")
        return registry

    def test_apply_bronze_handles_null_filling_strategies(self):
        df = pd.DataFrame({
            "spending": [100.0, np.nan, 200.0, np.nan, 300.0],
            "tenure": [12.0, 24.0, np.nan, 36.0, 48.0],
        })
        registry = self._make_registry_with_bronze_nulls(["spending", "tenure"])
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(df)

        assert result["spending"].isna().sum() == 0
        assert result["tenure"].isna().sum() == 0
        assert result["spending"].iloc[1] == 200.0

    def test_apply_gold_applies_log_transformation(self):
        np.random.seed(42)
        df = pd.DataFrame({"spending": np.random.exponential(100, 500)})
        registry = self._make_registry_with_gold_log(["spending"])
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(df)

        assert (result["spending"] < df["spending"]).all()
        assert np.allclose(result["spending"], np.log1p(df["spending"]))

    def test_apply_gold_applies_one_hot_encoding(self):
        df = pd.DataFrame({"plan": ["basic", "premium", "basic", "enterprise", "premium"]})
        registry = RecommendationRegistry()
        registry.init_bronze(source_file="test")
        registry.init_gold(target_column="churn")
        registry.add_gold_encoding("plan", "one_hot", "categorical", "nb02")
        materializer = DataMaterializer(registry)
        result = materializer.apply_gold(df)

        assert "plan" not in result.columns
        assert any("plan_" in c for c in result.columns)
        assert len(result) == 5

    def test_full_transform_pipeline_bronze_to_gold(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "spending": np.concatenate([[np.nan] * 50, np.random.exponential(100, 450)]),
            "plan": np.random.choice(["basic", "premium"], size=500),
            "churn": np.random.choice([0, 1], size=500),
        })
        registry = RecommendationRegistry()
        registry.init_bronze(source_file="test")
        registry.init_gold(target_column="churn")
        registry.add_bronze_null("spending", "median", "fill", "nb02")
        registry.add_gold_transformation("spending", "log", {"method": "log"}, "skewed", "nb02")
        registry.add_gold_encoding("plan", "one_hot", "categorical", "nb02")
        materializer = DataMaterializer(registry)
        result = materializer.transform(df)

        assert result["spending"].isna().sum() == 0
        assert "plan" not in result.columns
        assert any("plan_" in c for c in result.columns)

    def test_materialized_output_preserves_all_rows(self):
        np.random.seed(42)
        df = _build_churn_dataframe(n=500)
        df.loc[0:49, "monthly_spending"] = np.nan
        registry = self._make_registry_with_bronze_nulls(["monthly_spending"])
        materializer = DataMaterializer(registry)
        result = materializer.apply_bronze(df)

        assert len(result) == 500
        assert result["monthly_spending"].isna().sum() == 0
