"""Tests for distribution analysis module."""

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling import (
    DistributionAnalysis,
    DistributionAnalyzer,
    DistributionTransformationType,
    TransformationRecommendation,
)


class TestDistributionAnalyzer:
    """Tests for DistributionAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a DistributionAnalyzer instance."""
        return DistributionAnalyzer()

    @pytest.fixture
    def normal_series(self):
        """Create a normally distributed series."""
        np.random.seed(42)
        return pd.Series(np.random.normal(100, 15, 1000), name="normal")

    @pytest.fixture
    def skewed_series(self):
        """Create a highly skewed series (log-normal)."""
        np.random.seed(42)
        return pd.Series(np.random.lognormal(3, 1, 1000), name="skewed")

    @pytest.fixture
    def zero_inflated_series(self):
        """Create a zero-inflated series."""
        np.random.seed(42)
        values = np.random.exponential(10, 1000)
        # Make 40% of values zero
        mask = np.random.random(1000) < 0.4
        values[mask] = 0
        return pd.Series(values, name="zero_inflated")


class TestAnalyzeDistribution(TestDistributionAnalyzer):
    """Tests for analyze_distribution method."""

    def test_normal_distribution(self, analyzer, normal_series):
        """Test analysis of normally distributed data."""
        result = analyzer.analyze_distribution(normal_series, "normal")

        assert result.column_name == "normal"
        assert result.count == 1000
        assert abs(result.skewness) < 1.0  # Should be close to 0
        assert not result.is_highly_skewed
        assert not result.is_moderately_skewed
        assert not result.has_zero_inflation

    def test_skewed_distribution(self, analyzer, skewed_series):
        """Test analysis of skewed data."""
        result = analyzer.analyze_distribution(skewed_series, "skewed")

        assert result.is_highly_skewed or result.is_moderately_skewed
        assert result.skewness > 0  # Log-normal is right-skewed

    def test_zero_inflated_distribution(self, analyzer, zero_inflated_series):
        """Test analysis of zero-inflated data."""
        result = analyzer.analyze_distribution(zero_inflated_series, "zero_inflated")

        assert result.has_zero_inflation
        assert result.zero_percentage > 30.0
        assert result.zero_count > 0

    def test_empty_series(self, analyzer):
        """Test analysis of empty series."""
        empty_series = pd.Series([], dtype=float, name="empty")
        result = analyzer.analyze_distribution(empty_series, "empty")

        assert result.count == 0
        assert result.mean == 0.0
        assert result.skewness == 0.0

    def test_series_with_nulls(self, analyzer):
        """Test that null values are handled correctly."""
        series = pd.Series([1, 2, 3, None, None, 5, 6], name="with_nulls")
        result = analyzer.analyze_distribution(series, "with_nulls")

        assert result.count == 5  # Only non-null values

    def test_percentiles(self, analyzer, normal_series):
        """Test that percentiles are calculated correctly."""
        result = analyzer.analyze_distribution(normal_series, "normal")

        assert "p1" in result.percentiles
        assert "p50" in result.percentiles
        assert "p99" in result.percentiles
        assert result.percentiles["p50"] == result.median

    def test_outlier_detection(self, analyzer):
        """Test outlier detection using IQR method."""
        # Create data with known outliers
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], name="outliers")
        result = analyzer.analyze_distribution(series, "outliers")

        assert result.outlier_count_iqr > 0
        assert result.outlier_percentage > 0


class TestRecommendTransformation(TestDistributionAnalyzer):
    """Tests for recommend_transformation method."""

    def test_no_transform_for_normal(self, analyzer, normal_series):
        """Test that normal distribution gets no transformation."""
        analysis = analyzer.analyze_distribution(normal_series, "normal")
        rec = analyzer.recommend_transformation(analysis)

        assert rec.recommended_transform == DistributionTransformationType.NONE
        assert rec.priority == "low"

    def test_log_transform_for_skewed(self, analyzer, skewed_series):
        """Test that skewed positive data gets log transform recommendation."""
        analysis = analyzer.analyze_distribution(skewed_series, "skewed")
        rec = analyzer.recommend_transformation(analysis)

        # Should recommend log transform or cap_then_log
        assert rec.recommended_transform in [
            DistributionTransformationType.LOG_TRANSFORM,
            DistributionTransformationType.CAP_THEN_LOG,
            DistributionTransformationType.SQRT_TRANSFORM
        ]
        assert rec.priority in ["high", "medium"]

    def test_zero_inflation_handling(self, analyzer, zero_inflated_series):
        """Test recommendation for zero-inflated data."""
        analysis = analyzer.analyze_distribution(zero_inflated_series, "zero_inflated")
        rec = analyzer.recommend_transformation(analysis)

        assert rec.recommended_transform == DistributionTransformationType.ZERO_INFLATION_HANDLING
        assert len(rec.warnings) > 0

    def test_yeo_johnson_for_negative(self, analyzer):
        """Test that data with negatives gets Yeo-Johnson recommendation."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 10, 1000) - 50, name="negative")
        # Make it skewed
        series = series ** 3 / 10000

        analysis = analyzer.analyze_distribution(series, "negative")
        if analysis.is_highly_skewed:
            rec = analyzer.recommend_transformation(analysis)
            assert rec.recommended_transform == DistributionTransformationType.YERO_JOHNSON

    def test_cap_outliers_recommendation(self, analyzer):
        """Test outlier capping recommendation."""
        # Create data with many outliers but low skewness in main body
        np.random.seed(42)
        main_data = np.random.normal(50, 5, 900)
        outliers = np.random.uniform(100, 200, 100)  # 10% extreme outliers
        series = pd.Series(np.concatenate([main_data, outliers]), name="outlier_heavy")

        analysis = analyzer.analyze_distribution(series, "outlier_heavy")
        if analysis.outlier_percentage > 5.0:
            rec = analyzer.recommend_transformation(analysis)
            assert rec.recommended_transform in [
                DistributionTransformationType.CAP_OUTLIERS,
                DistributionTransformationType.CAP_THEN_LOG
            ]


class TestAnalyzeDataframe(TestDistributionAnalyzer):
    """Tests for analyze_dataframe method."""

    def test_analyze_all_numeric(self, analyzer):
        """Test analysis of all numeric columns in DataFrame."""
        np.random.seed(42)
        df = pd.DataFrame({
            "normal": np.random.normal(100, 15, 100),
            "skewed": np.random.lognormal(3, 1, 100),
            "categorical": ["A", "B", "C"] * 33 + ["A"]  # Non-numeric, should be skipped
        })
        results = analyzer.analyze_dataframe(df)

        assert "normal" in results
        assert "skewed" in results
        assert "categorical" not in results

    def test_analyze_specific_columns(self, analyzer):
        """Test analysis of specific columns only."""
        np.random.seed(42)
        df = pd.DataFrame({
            "col1": np.random.normal(0, 1, 100),
            "col2": np.random.normal(0, 1, 100),
            "col3": np.random.normal(0, 1, 100)
        })
        results = analyzer.analyze_dataframe(df, numeric_columns=["col1", "col2"])

        assert "col1" in results
        assert "col2" in results
        assert "col3" not in results


class TestGetAllRecommendations(TestDistributionAnalyzer):
    """Tests for get_all_recommendations method."""

    def test_recommendations_sorted_by_priority(self, analyzer):
        """Test that recommendations are sorted by priority."""
        np.random.seed(42)
        df = pd.DataFrame({
            "normal": np.random.normal(100, 15, 1000),  # Low priority
            "skewed": np.random.lognormal(5, 2, 1000),  # High priority
        })
        recs = analyzer.get_all_recommendations(df)

        # High priority should come first
        if len(recs) > 1:
            priorities = [r.priority for r in recs]
            priority_order = {"high": 0, "medium": 1, "low": 2}
            priority_indices = [priority_order.get(p, 3) for p in priorities]
            assert priority_indices == sorted(priority_indices)

    def test_only_actionable_recommendations(self, analyzer):
        """Test that only columns needing transformation are included."""
        np.random.seed(42)
        df = pd.DataFrame({
            "normal": np.random.normal(100, 15, 1000)  # Should be NONE
        })
        recs = analyzer.get_all_recommendations(df)

        # Normal distribution should not be in recommendations (NONE is filtered)
        for rec in recs:
            assert rec.recommended_transform != DistributionTransformationType.NONE


class TestGenerateReport(TestDistributionAnalyzer):
    """Tests for generate_report method."""

    def test_report_structure(self, analyzer):
        """Test that report has correct structure."""
        np.random.seed(42)
        df = pd.DataFrame({
            "normal": np.random.normal(100, 15, 100),
            "skewed": np.random.lognormal(3, 1, 100),
        })
        report = analyzer.generate_report(df)

        assert "summary" in report
        assert "categories" in report
        assert "analyses" in report
        assert "recommendations" in report

        assert "total_columns" in report["summary"]
        assert "highly_skewed" in report["categories"]

    def test_report_categorization(self, analyzer):
        """Test column categorization in report."""
        np.random.seed(42)
        df = pd.DataFrame({
            "normal": np.random.normal(100, 15, 1000),
            "skewed": np.random.lognormal(5, 2, 1000),  # Highly skewed
        })
        report = analyzer.generate_report(df)

        # Check that columns are categorized
        all_categorized = (
            report["categories"]["highly_skewed"] +
            report["categories"]["moderately_skewed"] +
            report["categories"]["approximately_normal"]
        )
        assert len(all_categorized) == 2


class TestDistributionAnalysisDataclass:
    """Tests for DistributionAnalysis dataclass."""

    def test_is_highly_skewed_property(self):
        """Test is_highly_skewed property."""
        analysis = DistributionAnalysis(
            column_name="test",
            count=100,
            mean=50.0,
            std=10.0,
            min_value=0.0,
            max_value=100.0,
            median=50.0,
            q1=40.0,
            q3=60.0,
            iqr=20.0,
            skewness=2.5,  # > 2.0
            kurtosis=3.0,
            zero_count=0,
            zero_percentage=0.0,
            negative_count=0,
            negative_percentage=0.0,
            outlier_count_iqr=5,
            outlier_percentage=5.0
        )

        assert analysis.is_highly_skewed is True
        assert analysis.is_moderately_skewed is False

    def test_has_zero_inflation_property(self):
        """Test has_zero_inflation property."""
        analysis = DistributionAnalysis(
            column_name="test",
            count=100,
            mean=50.0,
            std=10.0,
            min_value=0.0,
            max_value=100.0,
            median=50.0,
            q1=40.0,
            q3=60.0,
            iqr=20.0,
            skewness=0.5,
            kurtosis=3.0,
            zero_count=40,
            zero_percentage=40.0,  # > 30%
            negative_count=0,
            negative_percentage=0.0,
            outlier_count_iqr=5,
            outlier_percentage=5.0
        )

        assert analysis.has_zero_inflation is True

    def test_to_dict_method(self):
        """Test to_dict method."""
        analysis = DistributionAnalysis(
            column_name="test",
            count=100,
            mean=50.0,
            std=10.0,
            min_value=0.0,
            max_value=100.0,
            median=50.0,
            q1=40.0,
            q3=60.0,
            iqr=20.0,
            skewness=0.5,
            kurtosis=3.0,
            zero_count=5,
            zero_percentage=5.0,
            negative_count=0,
            negative_percentage=0.0,
            outlier_count_iqr=5,
            outlier_percentage=5.0
        )
        d = analysis.to_dict()

        assert d["column"] == "test"
        assert d["count"] == 100
        assert "skewness" in d
        assert "is_highly_skewed" in d


class TestTransformationRecommendationDataclass:
    """Tests for TransformationRecommendation dataclass."""

    def test_to_dict_method(self):
        """Test to_dict method."""
        rec = TransformationRecommendation(
            column_name="test",
            recommended_transform=DistributionTransformationType.LOG_TRANSFORM,
            reason="High skewness",
            priority="high",
            parameters={"base": "natural"},
            alternative_transforms=[DistributionTransformationType.SQRT_TRANSFORM],
            warnings=["Consider handling zeros first"]
        )
        d = rec.to_dict()

        assert d["column"] == "test"
        assert d["transform"] == "log_transform"
        assert d["priority"] == "high"
        assert "sqrt_transform" in d["alternatives"]
        assert len(d["warnings"]) == 1
