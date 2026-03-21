from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import (
    bulk_label_encode,
    bulk_median_impute,
    bulk_zero_variance_cols,
    collect_for_sklearn,
    spark_checkpoint,
)


class TestSparkCheckpoint:

    def test_noop_on_pandas(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = spark_checkpoint(df)
        pd.testing.assert_frame_equal(result, df)

    def test_returns_same_object_on_pandas(self):
        df = pd.DataFrame({"x": [10]})
        assert spark_checkpoint(df) is df


class TestBulkLabelEncode:

    def test_encodes_single_column(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
        result = bulk_label_encode(df, ["color"])
        assert result["color"].dtype in (np.int64, np.int32, int)
        assert set(result["color"].tolist()) == {0, 1, 2}

    def test_alphabetical_ordering(self):
        df = pd.DataFrame({"fruit": ["cherry", "apple", "banana", "apple"]})
        result = bulk_label_encode(df, ["fruit"])
        # apple=0, banana=1, cherry=2
        assert result["fruit"].tolist() == [2, 0, 1, 0]

    def test_multiple_columns(self):
        df = pd.DataFrame({
            "color": ["red", "blue", "green"],
            "size": ["L", "M", "S"],
            "value": [1.0, 2.0, 3.0],
        })
        result = bulk_label_encode(df, ["color", "size"])
        assert result["color"].dtype in (np.int64, np.int32, int)
        assert result["size"].dtype in (np.int64, np.int32, int)
        assert result["value"].dtype == np.float64

    def test_handles_nan(self):
        df = pd.DataFrame({"cat": ["a", None, "b", "a"]})
        result = bulk_label_encode(df, ["cat"])
        assert result["cat"].dtype in (np.int64, np.int32, int)
        assert len(result) == 4

    def test_preserves_non_target_columns(self):
        df = pd.DataFrame({
            "cat": ["x", "y", "z"],
            "num": [10, 20, 30],
        })
        result = bulk_label_encode(df, ["cat"])
        assert result["num"].tolist() == [10, 20, 30]

    def test_empty_columns_list(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = bulk_label_encode(df, [])
        pd.testing.assert_frame_equal(result, df)

    def test_column_not_in_df_skipped(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        result = bulk_label_encode(df, ["a", "nonexistent"])
        assert "a" in result.columns
        assert result["a"].dtype in (np.int64, np.int32, int)


class TestSparkBulkLabelEncode:

    def test_pandas_spark_parity_single_column(self):
        pdf = pd.DataFrame({"fruit": ["cherry", "apple", "banana", "apple"]})
        pandas_result = bulk_label_encode(pdf.copy(), ["fruit"])
        assert pandas_result["fruit"].tolist() == [2, 0, 1, 0]

    def test_pandas_spark_parity_multiple_columns(self):
        pdf = pd.DataFrame({
            "color": ["red", "blue", "green"],
            "size": ["L", "M", "S"],
            "value": [1.0, 2.0, 3.0],
        })
        result = bulk_label_encode(pdf.copy(), ["color", "size"])
        assert result["color"].tolist() == [2, 0, 1]
        assert result["size"].tolist() == [0, 1, 2]
        assert result["value"].dtype == np.float64

    def test_none_becomes_string_none(self):
        pdf = pd.DataFrame({"cat": ["a", None, "b", "a"]})
        result = bulk_label_encode(pdf.copy(), ["cat"])
        # None → "None" via astype(str), sorts before "a"
        assert result["cat"].tolist() == [1, 0, 2, 1]

    def test_no_ml_pipeline_import_in_spark_path(self):
        import inspect

        from customer_retention.core.compat import _spark_bulk_label_encode
        source = inspect.getsource(_spark_bulk_label_encode)
        assert "Pipeline" not in source
        assert "StringIndexer" not in source

    def test_high_cardinality_uses_hash_encoding(self):
        pytest.importorskip("pyspark")
        from unittest.mock import MagicMock, patch

        from customer_retention.core.compat import _MAX_LABEL_CARDINALITY, _spark_bulk_label_encode

        n = 100
        high_card_values = [f"id_{i}" for i in range(n)]
        pdf = pd.DataFrame({
            "high_card": high_card_values,
            "num": range(n),
        })

        mock_spark_df = MagicMock()
        mock_spark_df.columns = ["high_card", "num"]

        card_row = MagicMock()
        card_row.__getitem__ = lambda self, k: n
        mock_spark_df.agg.return_value.collect.return_value = [card_row]

        mock_result = MagicMock()
        mock_as_pandas = MagicMock(return_value=mock_result)

        with patch("customer_retention.core.compat.as_spark_df", return_value=mock_spark_df):
            with patch("customer_retention.core.compat._MAX_LABEL_CARDINALITY", 50):
                with patch("customer_retention.core.compat.spark_backend._as_pandas_api", mock_as_pandas):
                    result = _spark_bulk_label_encode(pdf, ["high_card"])

        select_call = mock_spark_df.select.call_args
        assert select_call is not None

    def test_high_cardinality_threshold_constant_exists(self):
        from customer_retention.core.compat import _MAX_LABEL_CARDINALITY
        assert _MAX_LABEL_CARDINALITY == 10_000


class TestBulkMedianImpute:

    def test_fills_nan_with_median(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0],
        })
        result = bulk_median_impute(df)
        assert not result.isna().any().any()
        assert result["a"].iloc[1] == pytest.approx(3.0)
        assert result["b"].iloc[2] == pytest.approx(30.0)

    def test_all_nan_column_fills_with_zero(self):
        df = pd.DataFrame({
            "good": [1.0, 2.0, 3.0],
            "bad": [np.nan, np.nan, np.nan],
        })
        result = bulk_median_impute(df)
        assert result["bad"].tolist() == [0.0, 0.0, 0.0]
        assert result["good"].tolist() == [1.0, 2.0, 3.0]

    def test_no_nan_returns_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = bulk_median_impute(df)
        pd.testing.assert_frame_equal(result, df)

    def test_specific_columns(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, 20.0, np.nan],
        })
        result = bulk_median_impute(df, columns=["a"])
        assert not result["a"].isna().any()
        assert result["b"].isna().sum() == 2

    def test_non_numeric_columns_ignored(self):
        df = pd.DataFrame({
            "num": [1.0, np.nan, 3.0],
            "cat": ["a", "b", "c"],
        })
        result = bulk_median_impute(df)
        assert result["num"].iloc[1] == pytest.approx(2.0)
        assert result["cat"].tolist() == ["a", "b", "c"]


class TestBulkZeroVarianceCols:

    def test_detects_constant_column(self):
        df = pd.DataFrame({
            "const": [5.0, 5.0, 5.0, 5.0],
            "varies": [1.0, 2.0, 3.0, 4.0],
        })
        result = bulk_zero_variance_cols(df)
        assert "const" in result
        assert "varies" not in result

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = bulk_zero_variance_cols(df)
        assert result == []

    def test_all_constant(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        result = bulk_zero_variance_cols(df)
        assert set(result) == {"a", "b"}

    def test_no_constant(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = bulk_zero_variance_cols(df)
        assert result == []

    def test_all_nan_column_detected(self):
        df = pd.DataFrame({
            "ok": [1.0, 2.0, 3.0],
            "all_nan": [np.nan, np.nan, np.nan],
        })
        result = bulk_zero_variance_cols(df)
        assert "all_nan" in result

    def test_ignores_non_numeric(self):
        df = pd.DataFrame({
            "num_const": [5.0, 5.0, 5.0],
            "cat": ["a", "a", "a"],
        })
        result = bulk_zero_variance_cols(df)
        assert "num_const" in result
        assert "cat" not in result


class TestCollectForSklearn:

    def test_noop_on_pandas(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = collect_for_sklearn(df)
        assert result is df

    def test_noop_on_pandas_series(self):
        s = pd.Series([1, 2, 3])
        result = collect_for_sklearn(s)
        assert result is s

    def test_returns_pandas_type(self):
        df = pd.DataFrame({"x": [10, 20]})
        result = collect_for_sklearn(df)
        assert isinstance(result, pd.DataFrame)

    def test_spark_pandas_series_returns_pandas_series(self):
        expected = pd.DataFrame({"entity_id": [1, 2, 3]})

        class FakeSparkDF:
            def toPandas(self):  # noqa: N802
                return expected

        class FakeFrame:
            def to_spark(self):
                return FakeSparkDF()

        class FakeSparkPandasSeries:
            spark = True
            ndim = 1
            def to_frame(self):
                return FakeFrame()

        result = collect_for_sklearn(FakeSparkPandasSeries())
        assert isinstance(result, pd.Series)
        assert result.tolist() == [1, 2, 3]

    def test_spark_pandas_dataframe_returns_pandas_dataframe(self):
        expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        class FakeSparkDF:
            def toPandas(self):  # noqa: N802
                return expected

        class FakeSparkPandasDF:
            spark = True
            ndim = 2
            def to_spark(self):
                return FakeSparkDF()

        result = collect_for_sklearn(FakeSparkPandasDF())
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected)
