import numpy as np
import pandas as pd
import pytest

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

    def test_with_color_column(self, chart_builder):
        df = pd.DataFrame({
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "cohort": ["Retained"] * 25 + ["Churned"] * 25,
        })
        fig = chart_builder.scatter_matrix(df[["feature1", "feature2"]], color_column=df["cohort"])
        assert fig is not None
        assert len(fig.data) > 0

    def test_color_column_with_custom_colors(self, chart_builder):
        df = pd.DataFrame({
            "f1": np.random.randn(40),
            "f2": np.random.randn(40),
            "group": ["Retained"] * 20 + ["Churned"] * 20,
        })
        color_map = {"Retained": "#2ECC71", "Churned": "#E74C3C"}
        fig = chart_builder.scatter_matrix(
            df[["f1", "f2"]], color_column=df["group"], color_map=color_map
        )
        assert fig is not None

    def test_color_column_uses_default_cohort_colors(self, chart_builder):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [4, 3, 2, 1],
            "cohort": ["Retained", "Retained", "Churned", "Churned"],
        })
        fig = chart_builder.scatter_matrix(df[["x", "y"]], color_column=df["cohort"])
        assert fig is not None

    def test_markers_have_transparency(self, chart_builder):
        df = pd.DataFrame({
            "x": np.random.randn(20),
            "y": np.random.randn(20),
            "cohort": ["Retained"] * 10 + ["Churned"] * 10,
        })
        fig = chart_builder.scatter_matrix(df[["x", "y"]], color_column=df["cohort"])
        assert fig is not None
        for trace in fig.data:
            if hasattr(trace, "marker") and trace.marker is not None:
                assert trace.marker.opacity is not None
                assert trace.marker.opacity < 1.0


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
        from customer_retention.stages.profiling import TemporalAnalyzer
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
        from customer_retention.stages.profiling import SegmentationMethod, SegmentationResult, SegmentProfile
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
        from customer_retention.stages.profiling import SegmentationMethod, SegmentationResult, SegmentProfile
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
        from customer_retention.stages.profiling import SegmentationMethod, SegmentationResult, SegmentProfile
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
        from customer_retention.stages.profiling import SegmentationMethod, SegmentationResult, SegmentProfile
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
        from customer_retention.stages.profiling import SegmentationMethod, SegmentationResult, SegmentProfile
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


class TestSparkline:
    def test_creates_figure(self, chart_builder):
        fig = chart_builder.sparkline([1, 2, 3, 4, 5])
        assert fig is not None
        assert fig.layout.height == 60
        assert fig.layout.width == 200

    def test_with_title(self, chart_builder):
        fig = chart_builder.sparkline([10, 20, 15, 25], title="Trend")
        assert fig.layout.title.text == "Trend"

    def test_shows_endpoints(self, chart_builder):
        fig = chart_builder.sparkline([1, 5, 3, 7], show_endpoints=True)
        assert len(fig.data) >= 2

    def test_no_endpoints(self, chart_builder):
        fig = chart_builder.sparkline([1, 5, 3, 7], show_endpoints=False, show_min_max=False)
        assert len(fig.data) == 1

    def test_shows_min_max_markers(self, chart_builder):
        fig = chart_builder.sparkline([3, 1, 5, 2], show_min_max=True)
        assert len(fig.data) >= 3

    def test_custom_dimensions(self, chart_builder):
        fig = chart_builder.sparkline([1, 2, 3], height=100, width=300)
        assert fig.layout.height == 100
        assert fig.layout.width == 300

    def test_single_value(self, chart_builder):
        fig = chart_builder.sparkline([42])
        assert fig is not None


class TestSparklineGrid:
    def test_creates_figure(self, chart_builder):
        data = {"Series A": [1, 2, 3, 4], "Series B": [4, 3, 2, 1]}
        fig = chart_builder.sparkline_grid(data)
        assert fig is not None

    def test_multiple_series(self, chart_builder):
        data = {f"S{i}": list(range(10)) for i in range(6)}
        fig = chart_builder.sparkline_grid(data, columns=3)
        assert fig is not None

    def test_trend_coloring(self, chart_builder):
        data = {"Up": [1, 2, 3, 4], "Down": [4, 3, 2, 1]}
        fig = chart_builder.sparkline_grid(data)
        assert len(fig.data) >= 4

    def test_single_series(self, chart_builder):
        fig = chart_builder.sparkline_grid({"Only": [5, 10, 15]})
        assert fig is not None


class TestCalendarHeatmap:
    def test_creates_figure(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", "2023-06-30", freq="D"))
        fig = chart_builder.calendar_heatmap(dates)
        assert fig is not None

    def test_with_values(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", "2023-03-31", freq="D"))
        values = pd.Series(np.random.rand(len(dates)))
        fig = chart_builder.calendar_heatmap(dates, values=values)
        assert fig is not None

    def test_with_title(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-06-01", "2023-06-30", freq="D"))
        fig = chart_builder.calendar_heatmap(dates, title="June Activity")
        assert fig.layout.title.text == "June Activity"

    def test_empty_dates(self, chart_builder):
        dates = pd.Series([], dtype="datetime64[ns]")
        fig = chart_builder.calendar_heatmap(dates)
        assert fig is not None

    def test_custom_colorscale(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", "2023-01-31", freq="D"))
        fig = chart_builder.calendar_heatmap(dates, colorscale="Reds")
        assert fig is not None


class TestMonthlyCalendarHeatmap:
    def test_creates_figure(self, chart_builder):
        dates = pd.Series(pd.date_range("2022-01-01", "2023-12-31", freq="D"))
        fig = chart_builder.monthly_calendar_heatmap(dates)
        assert fig is not None

    def test_with_values(self, chart_builder):
        dates = pd.Series(pd.date_range("2022-01-01", "2022-12-31", freq="D"))
        values = pd.Series(np.random.rand(len(dates)))
        fig = chart_builder.monthly_calendar_heatmap(dates, values=values)
        assert fig is not None

    def test_with_title(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", "2023-06-30", freq="D"))
        fig = chart_builder.monthly_calendar_heatmap(dates, title="Patterns")
        assert fig.layout.title.text == "Patterns"


class TestTimeSeriesWithAnomalies:
    def test_creates_figure(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", periods=100, freq="D"))
        np.random.seed(42)
        values = pd.Series(np.random.randn(100).cumsum())
        fig = chart_builder.time_series_with_anomalies(dates, values)
        assert fig is not None

    def test_detects_anomalies(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", periods=50, freq="D"))
        values = pd.Series([1.0] * 50)
        values.iloc[25] = 100.0
        fig = chart_builder.time_series_with_anomalies(dates, values, window=5, n_std=2.0)
        assert len(fig.data) >= 3

    def test_no_anomalies(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", periods=30, freq="D"))
        values = pd.Series([5.0] * 30)
        fig = chart_builder.time_series_with_anomalies(dates, values)
        assert fig is not None

    def test_custom_window_and_std(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", periods=50, freq="D"))
        values = pd.Series(np.random.randn(50))
        fig = chart_builder.time_series_with_anomalies(dates, values, window=14, n_std=3.0)
        assert fig is not None

    def test_with_title(self, chart_builder):
        dates = pd.Series(pd.date_range("2023-01-01", periods=20, freq="D"))
        values = pd.Series(np.random.randn(20))
        fig = chart_builder.time_series_with_anomalies(dates, values, title="My Anomalies")
        assert "My Anomalies" in fig.layout.title.text


class TestWaterfallChart:
    def test_creates_figure(self, chart_builder):
        fig = chart_builder.waterfall_chart(["A", "B", "C"], [10, -5, 15])
        assert fig is not None

    def test_positive_and_negative(self, chart_builder):
        fig = chart_builder.waterfall_chart(["Gain", "Loss", "Gain2"], [20, -8, 12])
        assert fig is not None
        assert len(fig.data) == 1

    def test_custom_labels(self, chart_builder):
        fig = chart_builder.waterfall_chart(
            ["Step1"], [100],
            initial_label="Begin", final_label="Total"
        )
        assert fig is not None

    def test_with_title(self, chart_builder):
        fig = chart_builder.waterfall_chart(["X"], [5], title="Score")
        assert fig.layout.title.text == "Score"

    def test_all_negative(self, chart_builder):
        fig = chart_builder.waterfall_chart(["A", "B"], [-10, -20])
        assert fig is not None


class TestQualityWaterfall:
    def test_creates_figure(self, chart_builder):
        checks = [
            {"name": "Nulls", "passed": True, "weight": 20},
            {"name": "Types", "passed": False, "weight": 30},
            {"name": "Range", "passed": True, "weight": 50},
        ]
        fig = chart_builder.quality_waterfall(checks)
        assert fig is not None

    def test_all_passed(self, chart_builder):
        checks = [
            {"name": "A", "passed": True, "weight": 50},
            {"name": "B", "passed": True, "weight": 50},
        ]
        fig = chart_builder.quality_waterfall(checks)
        assert fig is not None

    def test_all_failed(self, chart_builder):
        checks = [
            {"name": "A", "passed": False, "weight": 30},
            {"name": "B", "passed": False, "weight": 70},
        ]
        fig = chart_builder.quality_waterfall(checks)
        assert fig is not None

    def test_custom_max_score(self, chart_builder):
        checks = [{"name": "Check", "passed": False, "weight": 10}]
        fig = chart_builder.quality_waterfall(checks, max_score=50)
        assert fig is not None


class TestVelocityAccelerationChart:
    def test_creates_figure(self, chart_builder):
        data = {
            "amount": {
                "retained": [1, 2, 3, 4, 5],
                "churned": [5, 4, 3, 2, 1],
                "velocity_retained": [1, 1, 1, 1, 1],
                "velocity_churned": [-1, -1, -1, -1, -1],
                "accel_retained": [0, 0, 0, 0, 0],
                "accel_churned": [0, 0, 0, 0, 0],
            }
        }
        fig = chart_builder.velocity_acceleration_chart(data)
        assert fig is not None

    def test_multiple_columns(self, chart_builder):
        data = {
            "col1": {"retained": [1, 2, 3], "churned": [3, 2, 1]},
            "col2": {"retained": [10, 20, 30], "churned": [30, 20, 10]},
        }
        fig = chart_builder.velocity_acceleration_chart(data)
        assert fig is not None

    def test_with_title(self, chart_builder):
        data = {"x": {"retained": [1, 2], "churned": [2, 1]}}
        fig = chart_builder.velocity_acceleration_chart(data, title="Custom")
        assert "Custom" in fig.layout.title.text


class TestLagCorrelationHeatmap:
    def test_creates_figure(self, chart_builder):
        data = {
            "feature_a": [0.9, 0.8, 0.7, 0.6, 0.5],
            "feature_b": [0.3, 0.2, 0.1, 0.0, -0.1],
        }
        fig = chart_builder.lag_correlation_heatmap(data)
        assert fig is not None

    def test_custom_max_lag(self, chart_builder):
        data = {"col": [0.5] * 20}
        fig = chart_builder.lag_correlation_heatmap(data, max_lag=10)
        assert fig is not None

    def test_with_title(self, chart_builder):
        data = {"x": [0.1, 0.2, 0.3]}
        fig = chart_builder.lag_correlation_heatmap(data, title="Lags")
        assert fig.layout.title.text == "Lags"

    def test_single_variable(self, chart_builder):
        data = {"only": [0.9, 0.7, 0.5, 0.3, 0.1]}
        fig = chart_builder.lag_correlation_heatmap(data)
        assert fig is not None


class TestPredictivePowerChart:
    def test_creates_figure(self, chart_builder):
        iv = {"feat1": 0.4, "feat2": 0.15, "feat3": 0.05}
        ks = {"feat1": 0.5, "feat2": 0.25, "feat3": 0.1}
        fig = chart_builder.predictive_power_chart(iv, ks)
        assert fig is not None

    def test_color_thresholds(self, chart_builder):
        iv = {"strong": 0.6, "good": 0.35, "weak": 0.12, "none": 0.03}
        ks = {"strong": 0.5, "good": 0.3, "weak": 0.15, "none": 0.05}
        fig = chart_builder.predictive_power_chart(iv, ks)
        assert fig is not None

    def test_with_title(self, chart_builder):
        iv = {"a": 0.2}
        ks = {"a": 0.3}
        fig = chart_builder.predictive_power_chart(iv, ks, title="Power")
        assert fig.layout.title.text == "Power"

    def test_missing_ks_for_column(self, chart_builder):
        iv = {"a": 0.5, "b": 0.3}
        ks = {"a": 0.4}
        fig = chart_builder.predictive_power_chart(iv, ks)
        assert fig is not None


class TestMomentumComparisonChart:
    def test_creates_figure(self, chart_builder):
        data = {
            "amount": {"retained_7_30": 1.2, "churned_7_30": 0.8,
                      "retained_30_90": 1.1, "churned_30_90": 0.9},
        }
        fig = chart_builder.momentum_comparison_chart(data)
        assert fig is not None

    def test_multiple_columns(self, chart_builder):
        data = {
            "col1": {"retained_7_30": 1.5, "churned_7_30": 0.5},
            "col2": {"retained_7_30": 1.1, "churned_7_30": 0.9},
        }
        fig = chart_builder.momentum_comparison_chart(data)
        assert fig is not None

    def test_with_title(self, chart_builder):
        data = {"x": {"retained_7_30": 1.0, "churned_7_30": 1.0}}
        fig = chart_builder.momentum_comparison_chart(data, title="Momentum")
        assert fig.layout.title.text == "Momentum"


class TestCohortSparklines:
    def test_creates_3x3_grid(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2], "monthly": [2, 3], "yearly": [3]},
            "churned": {"weekly": [2, 1], "monthly": [3, 2], "yearly": [2]},
            "overall": {"weekly": [1.5], "monthly": [2.5], "yearly": [2.5]},
        }
        fig = chart_builder.cohort_sparklines(data, feature_name="amount")
        assert fig is not None
        assert len(fig.data) == 9  # 3 cohorts × 3 periods

    def test_with_title(self, chart_builder):
        data = {"retained": {"weekly": [1]}, "churned": {"weekly": [1]}, "overall": {"weekly": [1]}}
        fig = chart_builder.cohort_sparklines(data, feature_name="test_feature")
        assert "test_feature" in fig.layout.title.text

    def test_row_labels_are_cohorts(self, chart_builder):
        data = {
            "retained": {"weekly": [1], "monthly": [1], "yearly": [1]},
            "churned": {"weekly": [1], "monthly": [1], "yearly": [1]},
            "overall": {"weekly": [1], "monthly": [1], "yearly": [1]},
        }
        fig = chart_builder.cohort_sparklines(data, feature_name="col")
        annotations = [a.text for a in fig.layout.annotations if a.text]
        assert any("Retained" in str(t) for t in annotations)
        assert any("Churned" in str(t) for t in annotations)
        assert any("Overall" in str(t) for t in annotations)

    def test_column_labels_are_periods(self, chart_builder):
        data = {
            "retained": {"weekly": [1], "monthly": [1], "yearly": [1]},
            "churned": {"weekly": [1], "monthly": [1], "yearly": [1]},
            "overall": {"weekly": [1], "monthly": [1], "yearly": [1]},
        }
        fig = chart_builder.cohort_sparklines(data, feature_name="col")
        annotations = [a.text for a in fig.layout.annotations if a.text]
        assert any("Weekly" in str(t) for t in annotations)
        assert any("Monthly" in str(t) for t in annotations)
        assert any("Yearly" in str(t) for t in annotations)

    def test_no_bounding_boxes(self, chart_builder):
        data = {"retained": {"weekly": [1]}, "churned": {"weekly": [1]}, "overall": {"weekly": [1]}}
        fig = chart_builder.cohort_sparklines(data, feature_name="f")
        assert fig.layout.shapes is None or len(fig.layout.shapes) == 0


class TestAnalyzeCohortTrends:
    def test_returns_analysis_dict(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 3, 4, 5], "monthly": [1, 2, 3], "yearly": [1, 2]},
            "churned": {"weekly": [5, 4, 3, 2, 1], "monthly": [3, 2, 1], "yearly": [2, 1]},
        }
        result = chart_builder.analyze_cohort_trends(data, "test_feature")
        assert isinstance(result, dict)
        assert "feature" in result
        assert "periods" in result

    def test_computes_divergence_per_period(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 3], "monthly": [1, 2, 3]},
            "churned": {"weekly": [3, 2, 1], "monthly": [3, 2, 1]},
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert "weekly" in result["periods"]
        assert "monthly" in result["periods"]
        assert "divergence" in result["periods"]["weekly"]

    def test_identifies_opposite_trends(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 3, 4, 5]},  # upward
            "churned": {"weekly": [5, 4, 3, 2, 1]},   # downward
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert result["periods"]["weekly"]["opposite_trends"] is True

    def test_identifies_same_direction_trends(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 3, 4, 5]},  # upward
            "churned": {"weekly": [2, 3, 4, 5, 6]},   # also upward
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert result["periods"]["weekly"]["opposite_trends"] is False

    def test_finds_best_period_for_separation(self, chart_builder):
        data = {
            "retained": {"weekly": [5, 5, 5], "monthly": [1, 1, 1], "yearly": [1, 2]},
            "churned": {"weekly": [5, 5, 5], "monthly": [10, 10, 10], "yearly": [2, 1]},
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert result["best_period"] == "monthly"  # monthly has largest mean difference

    def test_generates_recommendation(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 3, 4, 5], "monthly": [1, 3, 5]},
            "churned": {"weekly": [5, 4, 3, 2, 1], "monthly": [5, 3, 1]},
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert "recommendation" in result
        assert len(result["recommendation"]) > 0

    def test_handles_missing_periods(self, chart_builder):
        data = {"retained": {"weekly": [1, 2]}, "churned": {"weekly": [2, 1]}}
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert "weekly" in result["periods"]
        assert "monthly" not in result["periods"]

    def test_detects_seasonality(self, chart_builder):
        seasonal = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5]  # alternating pattern
        data = {"retained": {"monthly": seasonal}, "churned": {"monthly": [3] * 12}}
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert result["periods"]["monthly"]["seasonality_detected"] is True

    def test_no_seasonality_for_flat(self, chart_builder):
        flat = [5, 5, 5, 5, 5, 5, 5, 5]
        data = {"retained": {"monthly": flat}, "churned": {"monthly": flat}}
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert result["periods"]["monthly"]["seasonality_detected"] is False

    def test_computes_variance_ratio(self, chart_builder):
        high_var = [1, 10, 2, 9, 3, 8]
        low_var = [5, 5, 5, 5, 5, 5]
        data = {"retained": {"weekly": high_var}, "churned": {"weekly": low_var}}
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert "variance_ratio" in result["periods"]["weekly"]
        assert result["periods"]["weekly"]["variance_ratio"] > 1.0

    def test_returns_actionable_recommendations(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 3, 4, 5], "monthly": [1, 3, 5]},
            "churned": {"weekly": [5, 4, 3, 2, 1], "monthly": [5, 3, 1]},
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert "actions" in result
        assert isinstance(result["actions"], list)

    def test_action_has_required_fields(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 3, 4, 5]},
            "churned": {"weekly": [5, 4, 3, 2, 1]},
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        if result["actions"]:
            action = result["actions"][0]
            assert "action_type" in action
            assert "feature" in action
            assert "reason" in action

    def test_returns_overall_effect_size(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 3], "monthly": [2, 3, 4]},
            "churned": {"weekly": [4, 5, 6], "monthly": [5, 6, 7]},
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert "overall_effect_size" in result
        assert isinstance(result["overall_effect_size"], float)

    def test_overall_effect_size_combines_periods(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2, 1], "monthly": [1, 2, 1], "yearly": [1, 2]},
            "churned": {"weekly": [8, 9, 8], "monthly": [8, 9, 8], "yearly": [8, 9]},
        }
        result = chart_builder.analyze_cohort_trends(data, "col")
        assert abs(result["overall_effect_size"]) > 1.0  # Large separation

    def test_sparklines_shows_effect_size_per_period(self, chart_builder):
        data = {
            "retained": {"weekly": [1, 2], "monthly": [1, 2], "yearly": [1, 2]},
            "churned": {"weekly": [5, 6], "monthly": [5, 6], "yearly": [5, 6]},
        }
        period_effects = {"weekly": 0.5, "monthly": 0.8, "yearly": 1.2}
        fig = chart_builder.cohort_sparklines(data, feature_name="test", period_effects=period_effects)
        col_titles = [a.text for a in fig.layout.annotations if a.text and ("Weekly" in a.text or "Monthly" in a.text or "Yearly" in a.text)]
        assert any("0.5" in t or "0.8" in t or "1.2" in t for t in col_titles)

    def test_sparklines_without_period_effects(self, chart_builder):
        data = {"retained": {"weekly": [1, 2]}, "churned": {"weekly": [2, 1]}, "overall": {"weekly": [1.5]}}
        fig = chart_builder.cohort_sparklines(data, feature_name="test")
        assert fig is not None  # Should work without period_effects


class TestDescriptiveStatsTiles:
    @pytest.fixture
    def mock_findings(self):
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
        from customer_retention.core.config.column_config import ColumnType
        return ExplorationFindings(
            source_path="test.csv",
            source_format="csv",
            row_count=100,
            column_count=4,
            columns={
                "num_col": ColumnFinding(
                    name="num_col", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=0.9, evidence=["float"],
                    universal_metrics={"null_count": 0, "null_percentage": 0},
                    type_metrics={"mean": 5.0, "median": 4.5, "std": 2.0}),
                "cat_col": ColumnFinding(
                    name="cat_col", inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                    confidence=0.9, evidence=["string"],
                    universal_metrics={"null_count": 0, "distinct_count": 3}),
                "bin_col": ColumnFinding(
                    name="bin_col", inferred_type=ColumnType.BINARY,
                    confidence=0.9, evidence=["binary"],
                    universal_metrics={"null_count": 0}),
                "id_col": ColumnFinding(
                    name="id_col", inferred_type=ColumnType.IDENTIFIER,
                    confidence=0.9, evidence=["unique"],
                    universal_metrics={"null_count": 0, "distinct_count": 100}),
            }
        )

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "num_col": np.random.randn(100),
            "cat_col": np.random.choice(["A", "B", "C"], 100),
            "bin_col": np.random.choice([0, 1], 100),
            "id_col": [f"ID_{i}" for i in range(100)],
        })

    def test_creates_figure(self, chart_builder, sample_df, mock_findings):
        fig = chart_builder.descriptive_stats_tiles(sample_df, mock_findings)
        assert fig is not None

    def test_max_columns(self, chart_builder, sample_df, mock_findings):
        fig = chart_builder.descriptive_stats_tiles(sample_df, mock_findings, max_columns=2)
        assert fig is not None

    def test_columns_per_row(self, chart_builder, sample_df, mock_findings):
        fig = chart_builder.descriptive_stats_tiles(sample_df, mock_findings, columns_per_row=2)
        assert fig is not None


class TestDatasetAtAGlance:
    @pytest.fixture
    def mock_findings(self):
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
        from customer_retention.core.config.column_config import ColumnType
        return ExplorationFindings(
            source_path="data.parquet",
            source_format="parquet",
            row_count=500,
            column_count=3,
            columns={
                "amount": ColumnFinding(
                    name="amount", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=0.9, evidence=["float"],
                    universal_metrics={"null_count": 5, "null_percentage": 1.0},
                    type_metrics={"mean": 100.0, "median": 90.0, "std": 30.0}),
                "category": ColumnFinding(
                    name="category", inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                    confidence=0.9, evidence=["string"],
                    universal_metrics={"null_count": 0, "null_percentage": 0, "distinct_count": 5}),
                "target": ColumnFinding(
                    name="target", inferred_type=ColumnType.TARGET,
                    confidence=0.9, evidence=["binary target"],
                    universal_metrics={"null_count": 0, "null_percentage": 0}),
            }
        )

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "amount": np.random.rand(500) * 200,
            "category": np.random.choice(["X", "Y", "Z", "W", "V"], 500),
            "target": np.random.choice([0, 1], 500),
        })

    def test_creates_figure(self, chart_builder, sample_df, mock_findings):
        fig = chart_builder.dataset_at_a_glance(sample_df, mock_findings)
        assert fig is not None

    def test_with_source_path(self, chart_builder, sample_df, mock_findings):
        fig = chart_builder.dataset_at_a_glance(
            sample_df, mock_findings, source_path="data.parquet"
        )
        assert fig is not None

    def test_event_granularity(self, chart_builder, sample_df, mock_findings):
        fig = chart_builder.dataset_at_a_glance(
            sample_df, mock_findings, granularity="event"
        )
        assert fig is not None

    def test_max_columns(self, chart_builder, sample_df, mock_findings):
        fig = chart_builder.dataset_at_a_glance(
            sample_df, mock_findings, max_columns=2
        )
        assert fig is not None


class TestColumnTiles:
    def test_numeric_tile(self, chart_builder):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series(np.random.randn(100))
        chart_builder._add_numeric_tile(
            fig, series, {"null_percentage": 5.0},
            {"mean": 0.0, "median": 0.1, "std": 1.0},
            row=1, col=1, n_cols=1, formatter=NumberFormatter()
        )
        assert len(fig.data) >= 1

    def test_categorical_tile(self, chart_builder):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series(np.random.choice(["A", "B", "C", "D"], 100))
        chart_builder._add_categorical_tile(
            fig, series, {"distinct_count": 4, "null_percentage": 0},
            row=1, col=1, n_cols=1, formatter=NumberFormatter()
        )
        assert len(fig.data) >= 1

    def test_binary_tile(self, chart_builder):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series([0] * 70 + [1] * 30)
        chart_builder._add_binary_tile(
            fig, series, {},
            row=1, col=1, n_cols=1, formatter=NumberFormatter()
        )
        assert len(fig.data) >= 1

    def test_binary_tile_empty(self, chart_builder):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series([], dtype=int)
        chart_builder._add_binary_tile(
            fig, series, {},
            row=1, col=1, n_cols=1, formatter=NumberFormatter()
        )
        assert len(fig.data) == 0

    def test_datetime_tile(self, chart_builder):
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series(pd.date_range("2023-01-01", periods=365, freq="D"))
        chart_builder._add_datetime_tile(
            fig, series, {},
            row=1, col=1, n_cols=1
        )
        assert len(fig.data) >= 1

    def test_datetime_tile_empty(self, chart_builder):
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series([], dtype="datetime64[ns]")
        chart_builder._add_datetime_tile(fig, series, {}, row=1, col=1, n_cols=1)
        assert len(fig.data) == 0

    def test_identifier_tile(self, chart_builder):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series([f"ID_{i}" for i in range(100)])
        chart_builder._add_identifier_tile(
            fig, series, {"distinct_count": 100},
            row=1, col=1, n_cols=1, formatter=NumberFormatter()
        )
        assert len(fig.data) >= 2

    def test_target_tile(self, chart_builder):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series([0] * 80 + [1] * 20)
        chart_builder._add_target_tile(
            fig, series, {},
            row=1, col=1, n_cols=1, formatter=NumberFormatter()
        )
        assert len(fig.data) >= 1

    def test_generic_tile(self, chart_builder):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series(["x", "y", "z", "x", "y"])
        chart_builder._add_generic_tile(
            fig, series, {"distinct_count": 3, "null_percentage": 0},
            row=1, col=1, n_cols=1, formatter=NumberFormatter()
        )
        assert len(fig.data) >= 1


class TestGetAxisRef:
    def test_first_subplot(self, chart_builder):
        assert chart_builder._get_axis_ref(1, 1, 4, "x") == "x"
        assert chart_builder._get_axis_ref(1, 1, 4, "y") == "y"

    def test_second_subplot(self, chart_builder):
        assert chart_builder._get_axis_ref(1, 2, 4, "x") == "x2"

    def test_second_row(self, chart_builder):
        assert chart_builder._get_axis_ref(2, 1, 4, "x") == "x5"

    def test_third_position(self, chart_builder):
        assert chart_builder._get_axis_ref(1, 3, 4, "x") == "x3"


class TestCutoffSelectionChart:
    @pytest.fixture
    def cutoff_analysis(self):
        from customer_retention.stages.temporal.cutoff_analyzer import CutoffAnalysis
        dates = pd.date_range("2023-01-01", periods=12, freq="ME").tolist()
        bin_counts = [100] * 12
        train_pcts = [float(i) / 12 * 100 for i in range(1, 13)]
        score_pcts = [100 - t for t in train_pcts]
        return CutoffAnalysis(
            timestamp_column="event_date",
            total_rows=1200,
            bins=dates,
            bin_counts=bin_counts,
            train_percentages=train_pcts,
            score_percentages=score_pcts,
            date_range=(dates[0], dates[-1]),
        )

    def test_creates_figure(self, chart_builder, cutoff_analysis):
        fig = chart_builder.cutoff_selection_chart(cutoff_analysis)
        assert fig is not None

    def test_with_suggested_cutoff(self, chart_builder, cutoff_analysis):
        suggested = pd.Timestamp("2023-06-30")
        fig = chart_builder.cutoff_selection_chart(cutoff_analysis, suggested_cutoff=suggested)
        assert fig is not None
        assert len(fig.data) >= 3

    def test_with_current_cutoff(self, chart_builder, cutoff_analysis):
        current = pd.Timestamp("2023-09-30")
        fig = chart_builder.cutoff_selection_chart(cutoff_analysis, current_cutoff=current)
        assert fig is not None

    def test_with_both_cutoffs(self, chart_builder, cutoff_analysis):
        suggested = pd.Timestamp("2023-06-30")
        current = pd.Timestamp("2023-09-30")
        fig = chart_builder.cutoff_selection_chart(
            cutoff_analysis, suggested_cutoff=suggested, current_cutoff=current
        )
        assert fig is not None

    def test_cutoff_outside_range(self, chart_builder, cutoff_analysis):
        current = pd.Timestamp("2025-01-01")
        fig = chart_builder.cutoff_selection_chart(cutoff_analysis, current_cutoff=current)
        assert fig is not None

    def test_same_suggested_and_current(self, chart_builder, cutoff_analysis):
        cutoff = pd.Timestamp("2023-06-30")
        fig = chart_builder.cutoff_selection_chart(
            cutoff_analysis, suggested_cutoff=cutoff, current_cutoff=cutoff
        )
        assert fig is not None

    def test_empty_analysis(self, chart_builder):
        from customer_retention.stages.temporal.cutoff_analyzer import CutoffAnalysis
        empty = CutoffAnalysis(
            timestamp_column="dt", total_rows=0,
            bins=[], bin_counts=[], train_percentages=[],
            score_percentages=[], date_range=(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")),
        )
        fig = chart_builder.cutoff_selection_chart(empty)
        assert fig is not None


class TestAddColumnTileDispatch:
    @pytest.fixture
    def findings_col(self):
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding
        from customer_retention.core.config.column_config import ColumnType
        return ColumnFinding(
            name="test", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9, evidence=[],
            universal_metrics={"null_count": 0, "null_percentage": 0},
            type_metrics={"mean": 0, "median": 0, "std": 1}
        )

    def test_dispatches_numeric(self, chart_builder, findings_col):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series(np.random.randn(50))
        chart_builder._add_column_tile(
            fig, series, findings_col, "numeric_continuous",
            row=1, col=1, formatter=NumberFormatter(), n_cols=1
        )
        assert len(fig.data) >= 1

    def test_dispatches_datetime(self, chart_builder):
        from plotly.subplots import make_subplots

        from customer_retention.analysis.auto_explorer.findings import ColumnFinding
        from customer_retention.analysis.visualization.number_formatter import NumberFormatter
        from customer_retention.core.config.column_config import ColumnType
        fig = make_subplots(rows=1, cols=1)
        series = pd.Series(pd.date_range("2023-01-01", periods=30, freq="D"))
        col_finding = ColumnFinding(
            name="dt", inferred_type=ColumnType.DATETIME,
            confidence=0.9, evidence=[],
            universal_metrics={}, type_metrics={}
        )
        chart_builder._add_column_tile(
            fig, series, col_finding, "datetime",
            row=1, col=1, formatter=NumberFormatter(), n_cols=1
        )
        assert len(fig.data) >= 1


class TestCategoricalAnalysisPanel:
    @pytest.fixture
    def mock_insights(self):
        from dataclasses import dataclass, field
        @dataclass
        class MockInsight:
            feature_name: str
            cramers_v: float
            effect_strength: str
            p_value: float
            n_categories: int
            high_risk_categories: list = field(default_factory=list)
            low_risk_categories: list = field(default_factory=list)
            category_stats: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
                "category": ["A", "B", "C"], "retention_rate": [0.8, 0.5, 0.3]
            }))
        return [
            MockInsight("plan_type", 0.35, "strong", 0.001, 3, ["free"], ["premium"]),
            MockInsight("region", 0.15, "moderate", 0.01, 4, ["APAC"], ["US"]),
            MockInsight("device", 0.05, "weak", 0.1, 3, [], []),
        ]

    def test_creates_figure(self, chart_builder, mock_insights):
        fig = chart_builder.categorical_analysis_panel(mock_insights, overall_rate=0.4)
        assert fig is not None
        assert hasattr(fig, "update_layout")

    def test_empty_insights(self, chart_builder):
        fig = chart_builder.categorical_analysis_panel([], overall_rate=0.4)
        assert fig is not None

    def test_respects_max_features(self, chart_builder, mock_insights):
        fig = chart_builder.categorical_analysis_panel(mock_insights, overall_rate=0.4, max_features=2)
        assert fig is not None

    def test_has_four_subplots(self, chart_builder, mock_insights):
        fig = chart_builder.categorical_analysis_panel(mock_insights, overall_rate=0.4)
        assert len(fig.data) >= 4


class TestVelocitySignalHeatmap:
    @pytest.fixture
    def effect_size_data(self):
        return {
            "velocity": {
                "metric_a": {"7d": 0.8, "14d": 0.6, "30d": 0.4},
                "metric_b": {"7d": 0.2, "14d": 0.3, "30d": 0.5},
            },
            "acceleration": {
                "metric_a": {"7d": 0.1, "14d": 0.2, "30d": 0.3},
                "metric_b": {"7d": 0.05, "14d": 0.1, "30d": 0.15},
            }
        }

    def test_creates_figure(self, chart_builder, effect_size_data):
        fig = chart_builder.velocity_signal_heatmap(effect_size_data)
        assert fig is not None
        assert hasattr(fig, "update_layout")

    def test_shows_both_velocity_and_acceleration(self, chart_builder, effect_size_data):
        fig = chart_builder.velocity_signal_heatmap(effect_size_data)
        assert len(fig.data) >= 2

    def test_handles_empty_data(self, chart_builder):
        fig = chart_builder.velocity_signal_heatmap({"velocity": {}, "acceleration": {}})
        assert fig is not None

    def test_custom_title(self, chart_builder, effect_size_data):
        fig = chart_builder.velocity_signal_heatmap(effect_size_data, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"


class TestCohortVelocitySparklines:
    @pytest.fixture
    def sparkline_data(self):
        from customer_retention.stages.profiling.temporal_feature_analyzer import CohortVelocityResult
        return [
            CohortVelocityResult(
                column="metric_a", window_days=7,
                retained_velocity=[0.1, 0.2, 0.15, 0.18, 0.22],
                churned_velocity=[-0.1, -0.15, -0.2, -0.18, -0.22],
                overall_velocity=[0.0, 0.05, -0.02, 0.0, 0.01],
                retained_accel=[0.01, 0.02, -0.01, 0.02, 0.01],
                churned_accel=[-0.01, -0.02, -0.01, 0.01, -0.02],
                overall_accel=[0.0, 0.0, -0.01, 0.01, 0.0],
                velocity_effect_size=0.85, velocity_effect_interp="Large effect",
                accel_effect_size=0.3, accel_effect_interp="Small effect",
                period_label="Weekly"
            ),
            CohortVelocityResult(
                column="metric_a", window_days=14,
                retained_velocity=[0.12, 0.22, 0.17],
                churned_velocity=[-0.12, -0.17, -0.22],
                overall_velocity=[0.0, 0.02, -0.02],
                retained_accel=[0.01, 0.02, -0.01],
                churned_accel=[-0.01, -0.02, -0.01],
                overall_accel=[0.0, 0.0, -0.01],
                velocity_effect_size=0.75, velocity_effect_interp="Medium effect",
                accel_effect_size=0.25, accel_effect_interp="Small effect",
                period_label="Bi-weekly"
            )
        ]

    def test_creates_figure(self, chart_builder, sparkline_data):
        fig = chart_builder.cohort_velocity_sparklines(sparkline_data, feature_name="metric_a")
        assert fig is not None
        assert hasattr(fig, "update_layout")

    def test_shows_cohort_traces(self, chart_builder, sparkline_data):
        fig = chart_builder.cohort_velocity_sparklines(sparkline_data, feature_name="metric_a")
        assert len(fig.data) >= 4

    def test_handles_single_window(self, chart_builder, sparkline_data):
        fig = chart_builder.cohort_velocity_sparklines([sparkline_data[0]], feature_name="metric_a")
        assert fig is not None

    def test_includes_period_labels(self, chart_builder, sparkline_data):
        fig = chart_builder.cohort_velocity_sparklines(sparkline_data, feature_name="metric_a")
        annotations = fig.layout.annotations or []
        assert any("Weekly" in str(a.text) for a in annotations)
        assert any("Velocity" in str(a.text) for a in annotations)
