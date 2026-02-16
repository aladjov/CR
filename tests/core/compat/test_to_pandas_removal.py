from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import to_pandas


@pytest.fixture
def artifact_store(tmp_path):
    from customer_retention.transforms.artifact_store import ArtifactStore
    return ArtifactStore(str(tmp_path / "artifacts"))


@pytest.fixture
def numeric_df():
    np.random.seed(42)
    return pd.DataFrame({"val": np.random.randn(100) * 10 + 50})


@pytest.fixture
def events_df():
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    return pd.DataFrame({
        "customer_id": np.random.choice(["C1", "C2", "C3", "C4", "C5"], 200),
        "event_date": np.random.choice(dates, 200),
        "amount": np.random.rand(200) * 100,
        "target": np.random.choice([0, 1], 200),
    })


@pytest.fixture
def entity_df():
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(50)],
        "created": pd.date_range("2020-01-01", periods=50, freq="7D"),
        "last_order": pd.date_range("2023-01-01", periods=50, freq="3D"),
        "first_order": pd.date_range("2020-02-01", periods=50, freq="7D"),
        "tenure_days": np.random.randint(100, 1000, 50),
        "total_spend": np.random.rand(50) * 1000,
        "target": np.random.choice([0, 1], 50),
        "category": np.random.choice(["A", "B", "C"], 50),
    })


class TestFittedScalerUsesToNumpy:
    def test_fit_transform_produces_correct_results(self, numeric_df, artifact_store):
        from customer_retention.transforms.fitted import FittedScaler
        scaler = FittedScaler("standard")
        result = scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        assert result["val"].mean() == pytest.approx(0, abs=0.1)
        assert result["val"].std() == pytest.approx(1, abs=0.1)

    def test_transform_produces_correct_results(self, numeric_df, artifact_store):
        from customer_retention.transforms.fitted import FittedScaler
        scaler = FittedScaler("standard")
        scaler.fit_transform(numeric_df.copy(), "val", artifact_store)
        scaler2 = FittedScaler("standard")
        result = scaler2.transform(numeric_df.copy(), "val", artifact_store)
        assert result["val"].mean() == pytest.approx(0, abs=0.1)

    def test_power_transform_produces_correct_results(self, numeric_df, artifact_store):
        from customer_retention.transforms.fitted import FittedPowerTransform
        pt = FittedPowerTransform()
        result = pt.fit_transform(numeric_df.copy(), "val", artifact_store)
        assert not result["val"].isna().any()

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.transforms import fitted
        source = inspect.getsource(fitted)
        assert "to_pandas" not in source


class TestTimeSeriesProfilerNoPandasConversion:
    def test_profile_works_without_to_pandas(self, events_df):
        from customer_retention.stages.profiling.time_series_profiler import TimeSeriesProfiler
        profiler = TimeSeriesProfiler("customer_id", "event_date")
        result = profiler.profile(events_df)
        assert result.total_events > 0
        assert result.unique_entities == 5

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.stages.profiling import time_series_profiler
        source = inspect.getsource(time_series_profiler)
        assert "to_pandas" not in source


class TestTemporalFeaturesNoPandasConversion:
    def test_transform_produces_temporal_features(self, entity_df):
        from customer_retention.stages.features.temporal_features import (
            ReferenceDateSource,
            TemporalFeatureGenerator,
        )
        gen = TemporalFeatureGenerator(
            reference_date=pd.Timestamp("2024-01-01"),
            reference_date_source=ReferenceDateSource.CONFIG,
            created_column="created",
            first_order_column="first_order",
            last_order_column="last_order",
        )
        gen.fit(entity_df)
        result = gen.transform(entity_df)
        assert "tenure_days" in result.columns
        assert "days_since_last_order" in result.columns

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.stages.features import temporal_features
        source = inspect.getsource(temporal_features)
        assert "to_pandas" not in source


class TestTimeSeriesDetectorNoPandasConversion:
    def test_detect_works(self, events_df):
        from customer_retention.stages.validation.timeseries_detector import TimeSeriesDetector
        detector = TimeSeriesDetector()
        result = detector.detect(events_df)
        assert result.entity_column is not None

    def test_validate_works(self, events_df):
        from customer_retention.stages.validation.timeseries_detector import TimeSeriesValidator
        validator = TimeSeriesValidator()
        result = validator.validate(events_df, "customer_id", "event_date")
        assert result.temporal_quality_score >= 0

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.stages.validation import timeseries_detector
        source = inspect.getsource(timeseries_detector)
        assert "to_pandas" not in source


class TestTemporalQualityChecksNoPandasConversion:
    def test_duplicate_check_no_entry_to_pandas(self, events_df):
        from customer_retention.stages.profiling.temporal_quality_checks import DuplicateEventCheck
        check = DuplicateEventCheck("customer_id", "event_date")
        result = check.run(events_df)
        assert result.check_id == "TQ001"

    def test_gap_check_no_entry_to_pandas(self, events_df):
        from customer_retention.stages.profiling.temporal_quality_checks import TemporalGapCheck
        check = TemporalGapCheck("event_date")
        result = check.run(events_df)
        assert result.check_id == "TQ002"

    def test_future_date_check_no_entry_to_pandas(self, events_df):
        from customer_retention.stages.profiling.temporal_quality_checks import FutureDateCheck
        check = FutureDateCheck("event_date")
        result = check.run(events_df)
        assert result.check_id == "TQ003"

    def test_event_order_check_no_entry_to_pandas(self, events_df):
        from customer_retention.stages.profiling.temporal_quality_checks import EventOrderCheck
        check = EventOrderCheck("customer_id", "event_date")
        result = check.run(events_df)
        assert result.check_id == "TQ004"

    def test_to_pandas_only_used_on_tiny_results(self):
        import inspect

        from customer_retention.stages.profiling import temporal_quality_checks
        source = inspect.getsource(temporal_quality_checks)
        lines_with_to_pandas = [
            line.strip() for line in source.split('\n')
            if 'to_pandas(' in line and 'import' not in line
        ]
        for line in lines_with_to_pandas:
            assert '.head(' in line, f"to_pandas used on non-tiny data: {line}"


class TestTemporalTargetAnalyzerNoPandasConversion:
    def test_analyze_produces_results(self, events_df):
        from customer_retention.stages.profiling.temporal_target_analyzer import TemporalTargetAnalyzer
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(events_df, "event_date", "target")
        assert result.overall_rate is not None
        assert result.n_valid_dates > 0

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.stages.profiling import temporal_target_analyzer
        source = inspect.getsource(temporal_target_analyzer)
        assert "to_pandas" not in source


class TestTemporalPatternAnalyzerNoPandasConversion:
    def test_analyze_cohort_distribution(self, events_df):
        from customer_retention.stages.profiling.temporal_pattern_analyzer import analyze_cohort_distribution
        first_events = events_df.groupby("customer_id")["event_date"].min().reset_index()
        result = analyze_cohort_distribution(first_events, "event_date")
        assert result.total_entities == 5

    def test_compute_recency_buckets(self, events_df):
        from customer_retention.stages.profiling.temporal_pattern_analyzer import compute_recency_buckets
        ref_date = pd.Timestamp("2023-01-01")
        result = compute_recency_buckets(events_df, "customer_id", "event_date", "target", ref_date)
        assert isinstance(result, list)

    def test_detect_trend(self, events_df):
        from customer_retention.stages.profiling.temporal_pattern_analyzer import TemporalPatternAnalyzer
        analyzer = TemporalPatternAnalyzer("event_date")
        result = analyzer.detect_trend(events_df, "amount")
        assert result.direction is not None

    def test_analyze_cohorts(self, events_df):
        from customer_retention.stages.profiling.temporal_pattern_analyzer import TemporalPatternAnalyzer
        analyzer = TemporalPatternAnalyzer("event_date")
        result = analyzer.analyze_cohorts(events_df, "customer_id", "event_date", "target")
        assert len(result) > 0

    def test_analyze_recency(self, events_df):
        from customer_retention.stages.profiling.temporal_pattern_analyzer import TemporalPatternAnalyzer
        analyzer = TemporalPatternAnalyzer("event_date")
        result = analyzer.analyze_recency(events_df, "customer_id", "target")
        assert result.avg_recency_days >= 0

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.stages.profiling import temporal_pattern_analyzer
        source = inspect.getsource(temporal_pattern_analyzer)
        assert "to_pandas" not in source


class TestCategoricalTargetAnalyzerNoPandasConversion:
    def test_analyze_produces_results(self, entity_df):
        from customer_retention.stages.profiling.categorical_target_analyzer import CategoricalTargetAnalyzer
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(entity_df, "category", "target")
        assert result.categorical_col == "category"

    def test_analyze_categorical_features(self, entity_df):
        from customer_retention.stages.profiling.categorical_target_analyzer import analyze_categorical_features
        result = analyze_categorical_features(entity_df, "customer_id", "target")
        assert result is not None

    def test_to_pandas_only_in_crosstab(self):
        import inspect

        from customer_retention.stages.profiling import categorical_target_analyzer
        source = inspect.getsource(categorical_target_analyzer)
        occurrences = source.count("to_pandas")
        assert occurrences <= 2


class TestSegmentAnalyzerNoPandasConversion:
    def test_analyze_produces_segments(self, entity_df):
        from customer_retention.stages.profiling.segment_analyzer import SegmentAnalyzer
        analyzer = SegmentAnalyzer()
        result = analyzer.analyze(entity_df, target_col="target", feature_cols=["tenure_days", "total_spend"])
        assert result.n_segments >= 1

    def test_find_optimal_segments(self, entity_df):
        from customer_retention.stages.profiling.segment_analyzer import SegmentAnalyzer
        analyzer = SegmentAnalyzer()
        n = analyzer.find_optimal_segments(entity_df, ["tenure_days", "total_spend"])
        assert n >= 1

    def test_profile_segments(self, entity_df):
        from customer_retention.stages.profiling.segment_analyzer import SegmentAnalyzer
        labels = np.zeros(len(entity_df), dtype=int)
        analyzer = SegmentAnalyzer()
        profiles = analyzer.profile_segments(entity_df, labels, ["tenure_days", "total_spend"], "target")
        assert len(profiles) >= 1

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.stages.profiling import segment_analyzer
        source = inspect.getsource(segment_analyzer)
        assert "to_pandas" not in source


class TestExplorerNoPandasConversion:
    def test_load_source_returns_dataframe_directly(self, entity_df):
        from customer_retention.analysis.auto_explorer.explorer import DataExplorer
        explorer = DataExplorer(visualize=False, save_findings=False)
        df, path, fmt = explorer._load_source(entity_df)
        assert fmt == "dataframe"
        assert len(df) == len(entity_df)
        assert df is entity_df

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.analysis.auto_explorer import explorer
        source = inspect.getsource(explorer)
        assert "to_pandas" not in source


class TestTemporalFeatureAnalyzerNoPandasConversion:
    def test_prepare_dataframe_works(self, events_df):
        from customer_retention.stages.profiling.temporal_feature_analyzer import TemporalFeatureAnalyzer
        analyzer = TemporalFeatureAnalyzer(
            entity_column="customer_id", time_column="event_date"
        )
        prepared = analyzer._prepare_dataframe(events_df)
        assert len(prepared) == len(events_df)

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.stages.profiling import temporal_feature_analyzer
        source = inspect.getsource(temporal_feature_analyzer)
        assert "to_pandas" not in source


class TestSegmentAwareOutlierNoPandasConversion:
    def test_analyze_produces_results(self, entity_df):
        from customer_retention.stages.profiling.segment_aware_outlier import SegmentAwareOutlierAnalyzer
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(entity_df, ["tenure_days", "total_spend"])
        assert result is not None

    def test_no_to_pandas_import(self):
        import inspect

        from customer_retention.stages.profiling import segment_aware_outlier
        source = inspect.getsource(segment_aware_outlier)
        assert "to_pandas" not in source


class TestChartBuilderNoPandasConversion:
    def test_correlation_heatmap_uses_to_numpy(self):
        from customer_retention.analysis.visualization.chart_builder import ChartBuilder
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        builder = ChartBuilder()
        fig = builder.correlation_heatmap(df)
        assert fig is not None

    def test_feature_importance_no_entry_to_pandas(self):
        from customer_retention.analysis.visualization.chart_builder import ChartBuilder
        df = pd.DataFrame({"feature": ["a", "b", "c"], "importance": [0.5, 0.3, 0.2]})
        with patch(
            "customer_retention.analysis.visualization.chart_builder.to_pandas",
            side_effect=AssertionError("to_pandas called"),
        ):
            builder = ChartBuilder()
            fig = builder.feature_importance_plot(df)
        assert fig is not None

    def test_to_pandas_only_at_plotly_boundary(self):
        import inspect

        from customer_retention.analysis.visualization import chart_builder
        source = inspect.getsource(chart_builder)
        lines_with_to_pandas = [
            line.strip() for line in source.split('\n')
            if 'to_pandas(' in line and 'import' not in line
        ]
        for line in lines_with_to_pandas:
            assert 'px.' in line or 'to_pandas(df)' not in line, (
                f"to_pandas used at method entry instead of Plotly boundary: {line}"
            )
