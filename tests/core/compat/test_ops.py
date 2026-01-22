from pathlib import Path

import pandas as pd
import pytest

try:
    import deltalake
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False

requires_delta = pytest.mark.skipif(not DELTA_AVAILABLE, reason="deltalake not installed")


class TestDataOpsInterface:
    def test_ops_singleton_exists(self):
        from customer_retention.core.compat import ops
        assert ops is not None

    def test_ops_is_dataops_instance(self):
        from customer_retention.core.compat import ops
        from customer_retention.core.compat.ops import DataOps
        assert isinstance(ops, DataOps)


class TestDataOpsReadCsv:
    def test_read_csv_returns_dataframe(self, tmp_path):
        from customer_retention.core.compat import ops
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        df = ops.read_csv(str(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_read_csv_with_kwargs(self, tmp_path):
        from customer_retention.core.compat import ops
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        df = ops.read_csv(str(csv_path), usecols=["a"])
        assert list(df.columns) == ["a"]


@requires_delta
class TestDataOpsReadWriteDelta:
    def test_write_delta_creates_table(self, tmp_path):
        from customer_retention.core.compat import ops
        delta_path = str(tmp_path / "test_delta")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ops.write_delta(df, delta_path)
        assert Path(delta_path).exists()

    def test_read_delta_returns_dataframe(self, tmp_path):
        from customer_retention.core.compat import ops
        delta_path = str(tmp_path / "test_delta")
        original = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ops.write_delta(original, delta_path)
        result = ops.read_delta(delta_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_write_delta_append_mode(self, tmp_path):
        from customer_retention.core.compat import ops
        delta_path = str(tmp_path / "test_delta")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ops.write_delta(df, delta_path)
        ops.write_delta(df, delta_path, mode="append")
        result = ops.read_delta(delta_path)
        assert len(result) == 4

    def test_read_delta_with_version(self, tmp_path):
        from customer_retention.core.compat import ops
        delta_path = str(tmp_path / "test_delta")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        ops.write_delta(df1, delta_path)
        ops.write_delta(df2, delta_path, mode="append")
        result_v0 = ops.read_delta(delta_path, version=0)
        assert len(result_v0) == 2


class TestDataOpsMissingStats:
    def test_get_missing_stats_returns_dict(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
        stats = ops.get_missing_stats(df)
        assert isinstance(stats, dict)

    def test_get_missing_stats_calculates_correctly(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
        stats = ops.get_missing_stats(df)
        assert abs(stats["a"] - (1/3)) < 0.01
        assert abs(stats["b"] - (2/3)) < 0.01

    def test_get_missing_stats_handles_no_missing(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": [1, 2, 3]})
        stats = ops.get_missing_stats(df)
        assert stats["a"] == 0.0


class TestDataOpsCorrelationMatrix:
    def test_correlation_matrix_returns_dataframe(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        corr = ops.correlation_matrix(df)
        assert isinstance(corr, pd.DataFrame)

    def test_correlation_matrix_is_symmetric(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        corr = ops.correlation_matrix(df)
        assert corr.loc["a", "b"] == corr.loc["b", "a"]

    def test_correlation_matrix_with_columns_filter(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        corr = ops.correlation_matrix(df, columns=["a", "b"])
        assert list(corr.columns) == ["a", "b"]


class TestDataOpsDtypeInfo:
    def test_get_dtype_info_returns_dict(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        info = ops.get_dtype_info(df)
        assert isinstance(info, dict)

    def test_get_dtype_info_detects_types(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.5, 2.5]})
        info = ops.get_dtype_info(df)
        assert "int" in info["a"].lower()
        assert "object" in info["b"].lower() or "str" in info["b"].lower()
        assert "float" in info["c"].lower()


class TestDataOpsSample:
    def test_sample_returns_correct_size(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": range(100)})
        sample = ops.sample(df, n=10)
        assert len(sample) == 10

    def test_sample_with_random_state_is_reproducible(self):
        from customer_retention.core.compat import ops
        df = pd.DataFrame({"a": range(100)})
        sample1 = ops.sample(df, n=10, random_state=42)
        sample2 = ops.sample(df, n=10, random_state=42)
        pd.testing.assert_frame_equal(sample1, sample2)


class TestDataOpsConcat:
    def test_concat_combines_dataframes(self):
        from customer_retention.core.compat import ops
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        result = ops.concat([df1, df2])
        assert len(result) == 4

    def test_concat_axis_1(self):
        from customer_retention.core.compat import ops
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        result = ops.concat([df1, df2], axis=1)
        assert "a" in result.columns and "b" in result.columns
        assert len(result.columns) == 2


class TestBackwardCompatibility:
    def test_pd_import_works(self):
        from customer_retention.core.compat import pd
        assert pd is not None

    def test_dataframe_type_works(self):
        from customer_retention.core.compat import DataFrame
        assert DataFrame is not None

    def test_is_spark_available_works(self):
        from customer_retention.core.compat import is_spark_available
        result = is_spark_available()
        assert isinstance(result, bool)

    def test_to_pandas_works(self):
        from customer_retention.core.compat import to_pandas
        df = pd.DataFrame({"a": [1, 2]})
        result = to_pandas(df)
        assert isinstance(result, pd.DataFrame)
