from unittest.mock import patch

import pandas as pd

from customer_retention.core.config import ColumnType


class TestExplorerLoadSourceUsesNativePandas:
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


class TestCategoricalEncoderConverts:
    @patch('customer_retention.stages.transformation.categorical_encoder.ensure_pandas_series')
    def test_fit_converts_to_pandas(self, mock_ensure):
        series = pd.Series(["a", "b", "c", "a"])
        mock_ensure.return_value = series
        from customer_retention.stages.transformation.categorical_encoder import CategoricalEncoder
        encoder = CategoricalEncoder()
        encoder.fit(series)
        mock_ensure.assert_called_once_with(series)

    @patch('customer_retention.stages.transformation.categorical_encoder.ensure_pandas_series')
    def test_transform_converts_to_pandas(self, mock_ensure):
        series = pd.Series(["a", "b", "c", "a"])
        mock_ensure.return_value = series
        from customer_retention.stages.transformation.categorical_encoder import CategoricalEncoder
        encoder = CategoricalEncoder()
        encoder.fit(series)
        mock_ensure.reset_mock()
        mock_ensure.return_value = series
        encoder.transform(series)
        mock_ensure.assert_called_once_with(series)


class TestBinaryHandlerConverts:
    @patch('customer_retention.stages.transformation.binary_handler.ensure_pandas_series')
    def test_fit_converts_to_pandas(self, mock_ensure):
        series = pd.Series([0, 1, 0, 1])
        mock_ensure.return_value = series
        from customer_retention.stages.transformation.binary_handler import BinaryHandler
        handler = BinaryHandler()
        handler.fit(series)
        mock_ensure.assert_called_once_with(series)


class TestMissingHandlerConverts:
    @patch('customer_retention.stages.cleaning.missing_handler.ensure_pandas_series')
    def test_fit_converts_to_pandas(self, mock_ensure):
        series = pd.Series([1.0, 2.0, None, 4.0])
        mock_ensure.return_value = series
        from customer_retention.stages.cleaning.missing_handler import MissingValueHandler
        handler = MissingValueHandler()
        handler.fit(series)
        mock_ensure.assert_called_once_with(series)


class TestStatisticsConverts:
    @patch('customer_retention.core.utils.statistics.ensure_pandas_series')
    def test_compute_psi_numeric_converts(self, mock_ensure):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_ensure.return_value = series
        import numpy as np

        from customer_retention.core.utils.statistics import compute_psi_numeric
        edges = np.linspace(0, 6, 11).tolist()
        counts = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        compute_psi_numeric(series, edges, counts)
        mock_ensure.assert_called_once_with(series)

    @patch('customer_retention.core.utils.statistics.ensure_pandas_series')
    def test_compute_psi_from_series_converts_both(self, mock_ensure):
        ref = pd.Series([1.0, 2.0, 3.0])
        curr = pd.Series([1.5, 2.5, 3.5])
        mock_ensure.side_effect = lambda s: s
        from customer_retention.core.utils.statistics import compute_psi_from_series
        compute_psi_from_series(ref, curr)
        assert mock_ensure.call_count == 2


class TestDriftDetectorConverts:
    @patch('customer_retention.stages.profiling.drift_detector.ensure_pandas_series')
    def test_set_baseline_converts_to_pandas(self, mock_ensure):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_ensure.return_value = series
        from customer_retention.stages.profiling.drift_detector import BaselineDriftChecker
        checker = BaselineDriftChecker()
        checker.set_baseline("col1", series, ColumnType.NUMERIC_CONTINUOUS)
        mock_ensure.assert_called_once_with(series)

    @patch('customer_retention.stages.profiling.drift_detector.ensure_pandas_series')
    def test_detect_drift_converts_to_pandas(self, mock_ensure):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_ensure.return_value = series
        from customer_retention.stages.profiling.drift_detector import BaselineDriftChecker
        checker = BaselineDriftChecker()
        checker.set_baseline("col1", series, ColumnType.NUMERIC_CONTINUOUS)
        mock_ensure.reset_mock()
        mock_ensure.return_value = series
        checker.detect_drift("col1", series, ColumnType.NUMERIC_CONTINUOUS)
        mock_ensure.assert_called_once_with(series)
