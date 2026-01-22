import pytest
import pandas as pd
import numpy as np

from customer_retention.analysis.visualization.chart_builder import ChartBuilder


@pytest.fixture
def chart_builder():
    return ChartBuilder()


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        "numeric_col": np.random.randn(100),
        "categorical_col": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100)
    })


class TestChartBuilderInit:
    def test_default_theme(self, chart_builder):
        assert chart_builder.theme == "plotly_white"

    def test_custom_theme(self):
        builder = ChartBuilder(theme="plotly_dark")
        assert builder.theme == "plotly_dark"

    def test_has_color_palette(self, chart_builder):
        assert "primary" in chart_builder.colors
        assert "success" in chart_builder.colors
        assert "warning" in chart_builder.colors
        assert "danger" in chart_builder.colors


class TestColumnTypeDistribution:
    def test_creates_figure(self, chart_builder):
        type_counts = {"numeric_continuous": 5, "categorical_nominal": 3, "target": 1}
        fig = chart_builder.column_type_distribution(type_counts)
        assert fig is not None
        assert hasattr(fig, "update_layout")

    def test_empty_input(self, chart_builder):
        fig = chart_builder.column_type_distribution({})
        assert fig is not None


class TestDataQualityScorecard:
    def test_creates_figure(self, chart_builder):
        scores = {"col1": 95.0, "col2": 75.0, "col3": 45.0}
        fig = chart_builder.data_quality_scorecard(scores)
        assert fig is not None

    def test_colors_by_score(self, chart_builder):
        scores = {"good": 90.0, "medium": 70.0, "bad": 40.0}
        fig = chart_builder.data_quality_scorecard(scores)
        assert fig is not None


class TestMissingValueBars:
    def test_creates_figure(self, chart_builder):
        null_pcts = {"col1": 0.0, "col2": 10.0, "col3": 50.0}
        fig = chart_builder.missing_value_bars(null_pcts)
        assert fig is not None


class TestHistogramWithStats:
    def test_creates_figure(self, chart_builder, sample_dataframe):
        fig = chart_builder.histogram_with_stats(sample_dataframe["numeric_col"])
        assert fig is not None

    def test_with_custom_title(self, chart_builder, sample_dataframe):
        fig = chart_builder.histogram_with_stats(sample_dataframe["numeric_col"], title="Custom Title")
        assert fig is not None


class TestBoxPlot:
    def test_creates_figure(self, chart_builder, sample_dataframe):
        fig = chart_builder.box_plot(sample_dataframe["numeric_col"])
        assert fig is not None


class TestOutlierVisualization:
    def test_creates_figure(self, chart_builder, sample_dataframe):
        fig = chart_builder.outlier_visualization(sample_dataframe["numeric_col"])
        assert fig is not None


class TestCategoryBarChart:
    def test_creates_figure(self, chart_builder, sample_dataframe):
        fig = chart_builder.category_bar_chart(sample_dataframe["categorical_col"])
        assert fig is not None

    def test_top_n_parameter(self, chart_builder, sample_dataframe):
        fig = chart_builder.category_bar_chart(sample_dataframe["categorical_col"], top_n=2)
        assert fig is not None


class TestCorrelationHeatmap:
    def test_creates_figure(self, chart_builder, sample_dataframe):
        numeric_df = sample_dataframe[["numeric_col", "target"]]
        fig = chart_builder.correlation_heatmap(numeric_df)
        assert fig is not None


class TestTargetCorrelationBars:
    def test_creates_figure(self, chart_builder):
        correlations = {"feature1": 0.5, "feature2": -0.3, "feature3": 0.1}
        fig = chart_builder.target_correlation_bars(correlations, "target")
        assert fig is not None


class TestROCCurve:
    def test_creates_figure(self, chart_builder):
        fpr = [0.0, 0.1, 0.2, 0.5, 1.0]
        tpr = [0.0, 0.4, 0.6, 0.8, 1.0]
        fig = chart_builder.roc_curve(fpr, tpr, auc_score=0.85)
        assert fig is not None


class TestConfusionMatrixHeatmap:
    def test_creates_figure(self, chart_builder):
        cm = [[50, 10], [5, 35]]
        fig = chart_builder.confusion_matrix_heatmap(cm)
        assert fig is not None

    def test_with_labels(self, chart_builder):
        cm = [[50, 10], [5, 35]]
        fig = chart_builder.confusion_matrix_heatmap(cm, labels=["Negative", "Positive"])
        assert fig is not None


class TestFeatureImportancePlot:
    def test_creates_figure(self, chart_builder):
        importance_df = pd.DataFrame({
            "feature": ["feat1", "feat2", "feat3"],
            "importance": [0.5, 0.3, 0.2]
        })
        fig = chart_builder.feature_importance_plot(importance_df)
        assert fig is not None


class TestLiftCurve:
    def test_creates_figure(self, chart_builder):
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        lift_values = [3.0, 2.5, 2.0, 1.8, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8]
        fig = chart_builder.lift_curve(percentiles, lift_values)
        assert fig is not None


class TestTimeSeriesPlot:
    def test_creates_figure(self, chart_builder):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "value": np.random.randn(10)
        })
        fig = chart_builder.time_series_plot(df, "date", "value")
        assert fig is not None


class TestCohortRetentionHeatmap:
    def test_creates_figure(self, chart_builder):
        retention_matrix = pd.DataFrame({
            "Month 0": [1.0, 1.0, 1.0],
            "Month 1": [0.8, 0.7, 0.9],
            "Month 2": [0.6, 0.5, 0.7]
        }, index=["Cohort A", "Cohort B", "Cohort C"])
        fig = chart_builder.cohort_retention_heatmap(retention_matrix)
        assert fig is not None


class TestBarChart:
    def test_creates_figure(self, chart_builder):
        fig = chart_builder.bar_chart(["A", "B", "C"], [10, 20, 30], title="Test Bar")
        assert fig is not None

    def test_horizontal_bar(self, chart_builder):
        fig = chart_builder.bar_chart(["A", "B"], [10, 20], horizontal=True)
        assert fig is not None

    def test_with_labels(self, chart_builder):
        fig = chart_builder.bar_chart(["X", "Y"], [5, 15], x_label="Category", y_label="Count")
        assert fig is not None


class TestHistogram:
    def test_creates_figure(self, chart_builder, sample_dataframe):
        fig = chart_builder.histogram(sample_dataframe["numeric_col"])
        assert fig is not None

    def test_with_title(self, chart_builder, sample_dataframe):
        fig = chart_builder.histogram(sample_dataframe["numeric_col"], title="Custom Histogram")
        assert fig is not None

    def test_with_nbins(self, chart_builder, sample_dataframe):
        fig = chart_builder.histogram(sample_dataframe["numeric_col"], nbins=10)
        assert fig is not None


class TestHeatmap:
    def test_creates_figure(self, chart_builder):
        z = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        fig = chart_builder.heatmap(z, ["A", "B", "C"], ["X", "Y", "Z"])
        assert fig is not None

    def test_with_title(self, chart_builder):
        z = [[0.5, -0.3], [-0.3, 0.8]]
        fig = chart_builder.heatmap(z, ["a", "b"], ["a", "b"], title="Correlation")
        assert fig is not None

    def test_with_custom_colorscale(self, chart_builder):
        z = [[1, 2], [3, 4]]
        fig = chart_builder.heatmap(z, ["A", "B"], ["X", "Y"], colorscale="Blues")
        assert fig is not None


class TestScatterMatrix:
    def test_creates_figure(self, chart_builder, sample_dataframe):
        numeric_df = sample_dataframe[["numeric_col", "target"]]
        fig = chart_builder.scatter_matrix(numeric_df)
        assert fig is not None

    def test_with_title(self, chart_builder, sample_dataframe):
        numeric_df = sample_dataframe[["numeric_col", "target"]]
        fig = chart_builder.scatter_matrix(numeric_df, title="Scatter Matrix")
        assert fig is not None


class TestMultiLineChart:
    def test_creates_figure(self, chart_builder):
        data = [
            {"fpr": [0, 0.5, 1], "tpr": [0, 0.8, 1], "name": "Model A"},
            {"fpr": [0, 0.5, 1], "tpr": [0, 0.6, 1], "name": "Model B"}
        ]
        fig = chart_builder.multi_line_chart(data, "fpr", "tpr", "name", title="ROC Comparison")
        assert fig is not None

    def test_with_axis_titles(self, chart_builder):
        data = [{"x": [1, 2, 3], "y": [1, 4, 9], "label": "Quadratic"}]
        fig = chart_builder.multi_line_chart(data, "x", "y", "label", x_title="X", y_title="Y")
        assert fig is not None


class TestTemporalDistribution:
    @pytest.fixture
    def temporal_analysis(self):
        from customer_retention.stages.profiling import TemporalAnalyzer, TemporalGranularity
        analyzer = TemporalAnalyzer()
        dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31", freq="D"))
        return analyzer.analyze(dates)

    def test_creates_figure(self, chart_builder, temporal_analysis):
        fig = chart_builder.temporal_distribution(temporal_analysis)
        assert fig is not None

    def test_bar_chart_type(self, chart_builder, temporal_analysis):
        fig = chart_builder.temporal_distribution(temporal_analysis, chart_type="bar")
        assert fig is not None

    def test_line_chart_type(self, chart_builder, temporal_analysis):
        fig = chart_builder.temporal_distribution(temporal_analysis, chart_type="line")
        assert fig is not None

    def test_with_title(self, chart_builder, temporal_analysis):
        fig = chart_builder.temporal_distribution(temporal_analysis, title="Custom Title")
        assert fig is not None


class TestTemporalTrend:
    @pytest.fixture
    def temporal_analysis(self):
        from customer_retention.stages.profiling import TemporalAnalyzer
        analyzer = TemporalAnalyzer()
        dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31", freq="D"))
        return analyzer.analyze(dates)

    def test_creates_figure(self, chart_builder, temporal_analysis):
        fig = chart_builder.temporal_trend(temporal_analysis)
        assert fig is not None

    def test_with_trend_line(self, chart_builder, temporal_analysis):
        fig = chart_builder.temporal_trend(temporal_analysis, show_trend=True)
        assert fig is not None

    def test_without_trend_line(self, chart_builder, temporal_analysis):
        fig = chart_builder.temporal_trend(temporal_analysis, show_trend=False)
        assert fig is not None


class TestTemporalHeatmap:
    def test_creates_figure(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31", freq="D"))
        fig = chart_builder.temporal_heatmap(dates)
        assert fig is not None

    def test_with_title(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", "2023-03-31", freq="D"))
        fig = chart_builder.temporal_heatmap(dates, title="Day of Week")
        assert fig is not None

    def test_handles_empty_series(self, chart_builder):
        fig = chart_builder.temporal_heatmap(pd.Series([], dtype="datetime64[ns]"))
        assert fig is not None


class TestYearMonthHeatmap:
    def test_creates_figure(self, chart_builder):
        pivot_df = pd.DataFrame(
            [[100, 110, 120], [90, 100, 110]],
            index=[2022, 2023],
            columns=["Jan", "Feb", "Mar"]
        )
        fig = chart_builder.year_month_heatmap(pivot_df)
        assert fig is not None

    def test_handles_empty_dataframe(self, chart_builder):
        fig = chart_builder.year_month_heatmap(pd.DataFrame())
        assert fig is not None


class TestCumulativeGrowthChart:
    def test_creates_figure(self, chart_builder):
        cumulative = pd.Series([100, 250, 450, 700], index=["Q1", "Q2", "Q3", "Q4"])
        fig = chart_builder.cumulative_growth_chart(cumulative)
        assert fig is not None

    def test_handles_empty_series(self, chart_builder):
        fig = chart_builder.cumulative_growth_chart(pd.Series([], dtype=float))
        assert fig is not None


class TestYearOverYearLines:
    def test_creates_figure(self, chart_builder):
        pivot_df = pd.DataFrame(
            [[100, 110, 120], [90, 100, 110]],
            index=[2022, 2023],
            columns=["Jan", "Feb", "Mar"]
        )
        fig = chart_builder.year_over_year_lines(pivot_df)
        assert fig is not None

    def test_handles_empty_dataframe(self, chart_builder):
        fig = chart_builder.year_over_year_lines(pd.DataFrame())
        assert fig is not None


class TestGrowthSummaryIndicators:
    def test_creates_figure(self, chart_builder):
        growth_data = {
            "has_data": True,
            "overall_growth_pct": 25.5,
            "avg_monthly_growth": 2.1,
            "trend_direction": "growing",
            "trend_slope": 15.3,
        }
        fig = chart_builder.growth_summary_indicators(growth_data)
        assert fig is not None

    def test_handles_no_data(self, chart_builder):
        fig = chart_builder.growth_summary_indicators({"has_data": False})
        assert fig is not None

    def test_declining_trend(self, chart_builder):
        growth_data = {
            "has_data": True,
            "overall_growth_pct": -15.0,
            "avg_monthly_growth": -1.5,
            "trend_direction": "declining",
            "trend_slope": -10.2,
        }
        fig = chart_builder.growth_summary_indicators(growth_data)
        assert fig is not None


class TestSegmentOverview:
    @pytest.fixture
    def segmentation_result(self):
        from customer_retention.stages.profiling import (
            SegmentationResult, SegmentProfile, SegmentationMethod
        )
        return SegmentationResult(
            n_segments=3,
            method=SegmentationMethod.KMEANS,
            quality_score=0.72,
            profiles=[
                SegmentProfile(segment_id=0, size=100, size_pct=33.3, target_rate=0.2,
                              defining_features={"feature_a": {"mean": 10}}),
                SegmentProfile(segment_id=1, size=100, size_pct=33.3, target_rate=0.5,
                              defining_features={"feature_a": {"mean": 50}}),
                SegmentProfile(segment_id=2, size=100, size_pct=33.4, target_rate=0.8,
                              defining_features={"feature_a": {"mean": 90}}),
            ],
            target_variance_ratio=0.35,
            recommendation="consider_segmentation",
            confidence=0.7,
            rationale=["High target variance"],
            labels=np.array([0] * 100 + [1] * 100 + [2] * 100),
        )

    def test_creates_figure(self, chart_builder, segmentation_result):
        fig = chart_builder.segment_overview(segmentation_result)
        assert fig is not None

    def test_with_title(self, chart_builder, segmentation_result):
        fig = chart_builder.segment_overview(segmentation_result, title="Custom Title")
        assert fig is not None


class TestSegmentFeatureComparison:
    @pytest.fixture
    def segmentation_result(self):
        from customer_retention.stages.profiling import (
            SegmentationResult, SegmentProfile, SegmentationMethod
        )
        return SegmentationResult(
            n_segments=3,
            method=SegmentationMethod.KMEANS,
            quality_score=0.72,
            profiles=[
                SegmentProfile(segment_id=0, size=100, size_pct=33.3, target_rate=0.2,
                              defining_features={
                                  "feature_a": {"mean": 10, "std": 2},
                                  "feature_b": {"mean": 100, "std": 10},
                              }),
                SegmentProfile(segment_id=1, size=100, size_pct=33.3, target_rate=0.5,
                              defining_features={
                                  "feature_a": {"mean": 50, "std": 5},
                                  "feature_b": {"mean": 200, "std": 15},
                              }),
                SegmentProfile(segment_id=2, size=100, size_pct=33.4, target_rate=0.8,
                              defining_features={
                                  "feature_a": {"mean": 90, "std": 8},
                                  "feature_b": {"mean": 50, "std": 5},
                              }),
            ],
            target_variance_ratio=0.35,
            recommendation="consider_segmentation",
            confidence=0.7,
            rationale=["High target variance"],
            labels=np.array([0] * 100 + [1] * 100 + [2] * 100),
        )

    def test_creates_figure(self, chart_builder, segmentation_result):
        fig = chart_builder.segment_feature_comparison(segmentation_result)
        assert fig is not None

    def test_with_selected_features(self, chart_builder, segmentation_result):
        fig = chart_builder.segment_feature_comparison(
            segmentation_result, features=["feature_a"]
        )
        assert fig is not None


class TestSegmentRecommendationCard:
    @pytest.fixture
    def segmentation_result(self):
        from customer_retention.stages.profiling import (
            SegmentationResult, SegmentProfile, SegmentationMethod
        )
        return SegmentationResult(
            n_segments=3,
            method=SegmentationMethod.KMEANS,
            quality_score=0.72,
            profiles=[
                SegmentProfile(segment_id=0, size=100, size_pct=33.3, target_rate=0.2,
                              defining_features={}),
            ],
            target_variance_ratio=0.35,
            recommendation="consider_segmentation",
            confidence=0.7,
            rationale=["High target variance", "Good cluster separation"],
            labels=np.array([0] * 100),
        )

    def test_creates_figure(self, chart_builder, segmentation_result):
        fig = chart_builder.segment_recommendation_card(segmentation_result)
        assert fig is not None

    def test_single_model_recommendation(self, chart_builder):
        from customer_retention.stages.profiling import (
            SegmentationResult, SegmentProfile, SegmentationMethod
        )
        result = SegmentationResult(
            n_segments=1,
            method=SegmentationMethod.KMEANS,
            quality_score=0.3,
            profiles=[SegmentProfile(0, 100, 100.0, 0.3, {})],
            target_variance_ratio=0.05,
            recommendation="single_model",
            confidence=0.8,
            rationale=["Low target variance"],
            labels=np.array([0] * 100),
        )
        fig = chart_builder.segment_recommendation_card(result)
        assert fig is not None

    def test_strong_segmentation_recommendation(self, chart_builder):
        from customer_retention.stages.profiling import (
            SegmentationResult, SegmentProfile, SegmentationMethod
        )
        result = SegmentationResult(
            n_segments=3,
            method=SegmentationMethod.KMEANS,
            quality_score=0.85,
            profiles=[
                SegmentProfile(0, 100, 33.3, 0.1, {}),
                SegmentProfile(1, 100, 33.3, 0.5, {}),
                SegmentProfile(2, 100, 33.4, 0.9, {}),
            ],
            target_variance_ratio=0.5,
            recommendation="strong_segmentation",
            confidence=0.9,
            rationale=["Very high target variance", "Excellent cluster quality"],
            labels=np.array([0] * 100 + [1] * 100 + [2] * 100),
        )
        fig = chart_builder.segment_recommendation_card(result)
        assert fig is not None


class TestPrecisionRecallCurve:
    def test_creates_figure(self, chart_builder):
        precision = [1.0, 0.8, 0.6, 0.4, 0.2]
        recall = [0.0, 0.3, 0.6, 0.8, 1.0]
        fig = chart_builder.precision_recall_curve(precision, recall, pr_auc=0.75)
        assert fig is not None

    def test_with_baseline(self, chart_builder):
        precision = [1.0, 0.9, 0.7, 0.5]
        recall = [0.0, 0.4, 0.7, 1.0]
        fig = chart_builder.precision_recall_curve(precision, recall, pr_auc=0.80, baseline=0.3)
        assert fig is not None

    def test_without_baseline(self, chart_builder):
        precision = [0.9, 0.8, 0.6]
        recall = [0.2, 0.5, 0.9]
        fig = chart_builder.precision_recall_curve(precision, recall, pr_auc=0.65)
        assert fig is not None


class TestModelComparisonGrid:
    @pytest.fixture
    def model_results(self):
        """Create sample model results for testing."""
        np.random.seed(42)
        y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        return {
            "Logistic Regression": {
                "y_pred": np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1]),
                "y_pred_proba": np.array([0.2, 0.3, 0.6, 0.4, 0.3, 0.7, 0.8, 0.4, 0.9, 0.85]),
            },
            "Random Forest": {
                "y_pred": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                "y_pred_proba": np.array([0.1, 0.2, 0.3, 0.25, 0.55, 0.8, 0.9, 0.7, 0.95, 0.88]),
            },
        }

    def test_creates_figure(self, chart_builder, model_results):
        y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        fig = chart_builder.model_comparison_grid(model_results, y_test)
        assert fig is not None

    def test_with_class_labels(self, chart_builder, model_results):
        y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        fig = chart_builder.model_comparison_grid(
            model_results, y_test, class_labels=["Retained", "Churned"]
        )
        assert fig is not None

    def test_with_title(self, chart_builder, model_results):
        y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        fig = chart_builder.model_comparison_grid(model_results, y_test, title="Model Comparison")
        assert fig is not None

    def test_single_model(self, chart_builder):
        y_test = np.array([0, 0, 0, 1, 1, 1])
        model_results = {
            "Single Model": {
                "y_pred": np.array([0, 0, 1, 1, 0, 1]),
                "y_pred_proba": np.array([0.2, 0.3, 0.6, 0.8, 0.4, 0.9]),
            }
        }
        fig = chart_builder.model_comparison_grid(model_results, y_test)
        assert fig is not None

    def test_three_models(self, chart_builder, model_results):
        y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        model_results["Gradient Boosting"] = {
            "y_pred": np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1]),
            "y_pred_proba": np.array([0.15, 0.25, 0.35, 0.52, 0.45, 0.85, 0.92, 0.75, 0.97, 0.91]),
        }
        fig = chart_builder.model_comparison_grid(model_results, y_test)
        assert fig is not None
