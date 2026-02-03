import numpy as np
import pandas as pd

from customer_retention.core.config import ColumnType


class TestExplorerUsesCompatPd:
    def test_load_parquet_returns_real_pandas(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)
        from customer_retention.analysis.auto_explorer.explorer import DataExplorer
        explorer = DataExplorer(visualize=False, save_findings=False)
        result_df, _, fmt = explorer._load_source(str(path))
        assert isinstance(result_df, pd.DataFrame)
        assert fmt == "parquet"

    def test_load_csv_returns_real_pandas(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        from customer_retention.analysis.auto_explorer.explorer import DataExplorer
        explorer = DataExplorer(visualize=False, save_findings=False)
        result_df, _, fmt = explorer._load_source(str(path))
        assert isinstance(result_df, pd.DataFrame)
        assert fmt == "csv"


class TestCategoricalEncoderNoPandasConversion:
    def test_fit_works_on_plain_series(self):
        from customer_retention.stages.transformation.categorical_encoder import CategoricalEncoder
        series = pd.Series(["a", "b", "c", "a"])
        encoder = CategoricalEncoder()
        encoder.fit(series)
        assert encoder._is_fitted

    def test_transform_works_on_plain_series(self):
        from customer_retention.stages.transformation.categorical_encoder import CategoricalEncoder
        series = pd.Series(["a", "b", "c", "a"])
        encoder = CategoricalEncoder()
        encoder.fit(series)
        result = encoder.transform(series)
        assert result.series is not None


class TestBinaryHandlerNoPandasConversion:
    def test_fit_works_on_plain_series(self):
        from customer_retention.stages.transformation.binary_handler import BinaryHandler
        series = pd.Series([0, 1, 0, 1])
        handler = BinaryHandler()
        handler.fit(series)
        assert handler._is_fitted

    def test_transform_works_on_plain_series(self):
        from customer_retention.stages.transformation.binary_handler import BinaryHandler
        series = pd.Series([0, 1, 0, 1])
        handler = BinaryHandler()
        handler.fit(series)
        result = handler.transform(series)
        assert result.series is not None


class TestMissingHandlerNoPandasConversion:
    def test_fit_works_on_plain_series(self):
        from customer_retention.stages.cleaning.missing_handler import MissingValueHandler
        series = pd.Series([1.0, 2.0, None, 4.0])
        handler = MissingValueHandler()
        handler.fit(series)
        assert handler._is_fitted

    def test_transform_works_on_plain_series(self):
        from customer_retention.stages.cleaning.missing_handler import MissingValueHandler
        series = pd.Series([1.0, 2.0, None, 4.0])
        handler = MissingValueHandler()
        handler.fit(series)
        result = handler.transform(series)
        assert result.values_imputed == 1


class TestStatisticsNoPandasConversion:
    def test_compute_psi_numeric_works(self):
        from customer_retention.core.utils.statistics import compute_psi_numeric
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        edges = np.linspace(0, 6, 11).tolist()
        counts = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        result = compute_psi_numeric(series, edges, counts)
        assert isinstance(result, float)

    def test_compute_ks_statistic_works(self):
        from customer_retention.core.utils.statistics import compute_ks_statistic
        ref = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        curr = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        stat, pvalue = compute_ks_statistic(ref, curr)
        assert isinstance(stat, float)
        assert isinstance(pvalue, float)

    def test_compute_psi_from_series_works(self):
        from customer_retention.core.utils.statistics import compute_psi_from_series
        ref = pd.Series([1.0, 2.0, 3.0])
        curr = pd.Series([1.5, 2.5, 3.5])
        result = compute_psi_from_series(ref, curr)
        assert isinstance(result, float)


class TestDriftDetectorNoPandasConversion:
    def test_set_baseline_works(self):
        from customer_retention.stages.profiling.drift_detector import BaselineDriftChecker
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        checker = BaselineDriftChecker()
        checker.set_baseline("col1", series, ColumnType.NUMERIC_CONTINUOUS)
        assert "col1" in checker.baseline

    def test_detect_drift_works(self):
        from customer_retention.stages.profiling.drift_detector import BaselineDriftChecker
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        checker = BaselineDriftChecker()
        checker.set_baseline("col1", series, ColumnType.NUMERIC_CONTINUOUS)
        result = checker.detect_drift("col1", series, ColumnType.NUMERIC_CONTINUOUS)
        assert result.column_name == "col1"


class TestFittedEncoderNoPandasConversion:
    def test_fit_transform_works_on_plain_series(self):
        from customer_retention.transforms.artifact_store import ArtifactStore
        from customer_retention.transforms.fitted import FittedEncoder
        store = ArtifactStore("/tmp/test_artifacts_ep")
        df = pd.DataFrame({"cat": ["a", "b", "c", "a"]})
        encoder = FittedEncoder()
        result = encoder.fit_transform(df, "cat", store)
        assert result["cat"].dtype in (np.int64, np.intp, int)

    def test_transform_handles_unknowns(self):
        from customer_retention.transforms.artifact_store import ArtifactStore
        from customer_retention.transforms.fitted import FittedEncoder
        store = ArtifactStore("/tmp/test_artifacts_ep")
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        encoder = FittedEncoder()
        encoder.fit_transform(df, "cat", store)
        df_new = pd.DataFrame({"cat": ["a", "unknown"]})
        encoder2 = FittedEncoder()
        result = encoder2.transform(df_new, "cat", store)
        assert result["cat"].iloc[1] == 0


class TestTemporalAnalyzerNoPandasConversion:
    def test_detect_granularity_works(self):
        from customer_retention.stages.profiling.temporal_analyzer import TemporalAnalyzer
        dates = pd.Series(pd.date_range("2023-01-01", periods=90, freq="D"))
        analyzer = TemporalAnalyzer()
        result = analyzer.detect_granularity(dates)
        assert result is not None

    def test_analyze_works(self):
        from customer_retention.stages.profiling.temporal_analyzer import TemporalAnalyzer
        dates = pd.Series(pd.date_range("2023-01-01", periods=180, freq="D"))
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze(dates)
        assert result.span_days > 0

    def test_analyze_seasonality_works(self):
        from customer_retention.stages.profiling.temporal_analyzer import TemporalAnalyzer
        dates = pd.Series(pd.date_range("2020-01-01", "2023-12-31", freq="D"))
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze_seasonality(dates)
        assert result.weekly_pattern is not None

    def test_year_over_year_works(self):
        from customer_retention.stages.profiling.temporal_analyzer import TemporalAnalyzer
        dates = pd.Series(pd.date_range("2020-01-01", "2023-12-31", freq="D"))
        analyzer = TemporalAnalyzer()
        result = analyzer.year_over_year_comparison(dates)
        assert len(result) == 4

    def test_calculate_growth_rate_works(self):
        from customer_retention.stages.profiling.temporal_analyzer import TemporalAnalyzer
        dates = pd.Series(pd.date_range("2023-01-01", periods=365, freq="D"))
        analyzer = TemporalAnalyzer()
        result = analyzer.calculate_growth_rate(dates)
        assert result["has_data"] is True

    def test_recommend_features_works(self):
        from customer_retention.stages.profiling.temporal_analyzer import TemporalAnalyzer
        dates = pd.Series(pd.date_range("2020-01-01", "2023-12-31", freq="D"))
        analyzer = TemporalAnalyzer()
        recs = analyzer.recommend_features(dates, "signup_date")
        assert len(recs) > 0


class TestDatetimeTransformerNoPandasConversion:
    def test_fit_transform_works(self):
        from customer_retention.stages.transformation import DatetimeTransformer
        series = pd.Series(pd.to_datetime(["2024-01-15", "2024-06-10"]))
        transformer = DatetimeTransformer(extract_features=["year", "month"])
        result = transformer.fit_transform(series)
        assert result.df["year"].iloc[0] == 2024

    def test_string_dates_work(self):
        from customer_retention.stages.transformation import DatetimeTransformer
        series = pd.Series(["2024-01-01", "2024-06-15"])
        transformer = DatetimeTransformer(extract_features=["year"])
        result = transformer.fit_transform(series)
        assert result.df["year"].iloc[0] == 2024
