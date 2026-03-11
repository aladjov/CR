from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat.bulk_profiling import (
    BinaryColumnStats,
    BulkStats,
    CategoricalColumnStats,
    DatetimeColumnStats,
    IdentifierColumnStats,
    NumericColumnStats,
    PerColumnStats,
    TextColumnStats,
    TypedBulkStats,
    _pandas_typed_bulk_stats,
    compute_typed_bulk_stats,
)


class TestDatetimeColumnStats:
    def test_defaults(self):
        s = DatetimeColumnStats()
        assert s.min_date is None
        assert s.max_date is None
        assert s.date_range_days is None
        assert s.future_date_count == 0
        assert s.placeholder_count == 0
        assert s.weekend_count == 0

    def test_all_fields(self):
        s = DatetimeColumnStats(
            min_date="2020-01-01",
            max_date="2023-12-31",
            date_range_days=1461,
            future_date_count=5,
            placeholder_count=2,
            weekend_count=10,
        )
        assert s.min_date == "2020-01-01"
        assert s.date_range_days == 1461
        assert s.future_date_count == 5


class TestCategoricalColumnStats:
    def test_defaults(self):
        s = CategoricalColumnStats()
        assert s.cardinality == 0
        assert s.cardinality_ratio == 0.0
        assert s.top_categories == []
        assert s.value_counts == {}
        assert s.rare_category_count == 0
        assert s.rare_category_percentage == 0.0
        assert s.case_variations == []

    def test_all_fields(self):
        s = CategoricalColumnStats(
            cardinality=5,
            cardinality_ratio=0.05,
            top_categories=[("a", 100), ("b", 50)],
            value_counts={"a": 100, "b": 50, "c": 10},
            rare_category_count=1,
            rare_category_percentage=5.0,
            case_variations=["New York vs new york"],
        )
        assert s.cardinality == 5
        assert len(s.top_categories) == 2
        assert s.rare_category_count == 1


class TestIdentifierColumnStats:
    def test_defaults(self):
        s = IdentifierColumnStats()
        assert s.is_unique is False
        assert s.duplicate_count == 0
        assert s.length_min is None
        assert s.length_max is None
        assert s.length_mode is None

    def test_unique(self):
        s = IdentifierColumnStats(is_unique=True, duplicate_count=0, length_min=5, length_max=5, length_mode=5)
        assert s.is_unique is True


class TestBinaryColumnStats:
    def test_defaults(self):
        s = BinaryColumnStats()
        assert s.true_count == 0
        assert s.false_count == 0
        assert s.true_percentage == 0.0
        assert s.balance_ratio == 0.0
        assert s.is_boolean is False

    def test_balanced(self):
        s = BinaryColumnStats(true_count=50, false_count=50, true_percentage=50.0, balance_ratio=1.0, is_boolean=True)
        assert s.balance_ratio == 1.0


class TestTextColumnStats:
    def test_defaults(self):
        s = TextColumnStats()
        assert s.length_min == 0
        assert s.length_max == 0
        assert s.length_mean == 0.0
        assert s.length_median == 0.0
        assert s.empty_count == 0
        assert s.empty_percentage == 0.0
        assert s.word_count_mean == 0.0
        assert s.contains_digits_pct == 0.0
        assert s.contains_special_pct == 0.0
        assert s.pii_email_count == 0
        assert s.pii_phone_count == 0
        assert s.pii_ssn_count == 0
        assert s.pii_cc_count == 0

    def test_with_pii(self):
        s = TextColumnStats(pii_email_count=5, pii_phone_count=3)
        assert s.pii_email_count == 5
        assert s.pii_phone_count == 3


class TestTypedBulkStats:
    def test_empty(self):
        s = TypedBulkStats()
        assert s.datetime == {}
        assert s.categorical == {}
        assert s.identifier == {}
        assert s.binary == {}
        assert s.text == {}

    def test_mixed(self):
        s = TypedBulkStats(
            datetime={"dt_col": DatetimeColumnStats()},
            categorical={"cat_col": CategoricalColumnStats()},
            binary={"bin_col": BinaryColumnStats()},
        )
        assert "dt_col" in s.datetime
        assert "cat_col" in s.categorical
        assert "bin_col" in s.binary


class TestPandasTypedBulkStatsDatetime:
    def test_basic_datetime(self):
        df = pd.DataFrame({
            "dt": pd.date_range("2020-01-01", periods=100, freq="D"),
        })
        bulk = BulkStats(total_count=100, columns={"dt": PerColumnStats(null_count=0, distinct_count=100)})
        result = _pandas_typed_bulk_stats(df, bulk, datetime_cols=["dt"])
        assert "dt" in result.datetime
        ds = result.datetime["dt"]
        assert ds.min_date is not None
        assert ds.max_date is not None
        assert ds.date_range_days > 0
        assert ds.future_date_count == 0
        assert ds.placeholder_count == 0
        assert ds.weekend_count > 0

    def test_datetime_with_future_dates(self):
        future = datetime.now() + timedelta(days=365)
        df = pd.DataFrame({
            "dt": [datetime(2020, 1, 1)] * 95 + [future] * 5,
        })
        bulk = BulkStats(total_count=100, columns={"dt": PerColumnStats(null_count=0, distinct_count=2)})
        result = _pandas_typed_bulk_stats(df, bulk, datetime_cols=["dt"])
        assert result.datetime["dt"].future_date_count == 5

    def test_datetime_with_placeholders(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["1970-01-01", "2020-06-15", "2020-06-16", "1900-01-01", "2020-06-17"]),
        })
        bulk = BulkStats(total_count=5, columns={"dt": PerColumnStats(null_count=0, distinct_count=5)})
        result = _pandas_typed_bulk_stats(df, bulk, datetime_cols=["dt"])
        assert result.datetime["dt"].placeholder_count == 2

    def test_datetime_all_null(self):
        df = pd.DataFrame({"dt": pd.array([pd.NaT, pd.NaT, pd.NaT], dtype="datetime64[ns]")})
        bulk = BulkStats(total_count=3, columns={"dt": PerColumnStats(null_count=3, distinct_count=0)})
        result = _pandas_typed_bulk_stats(df, bulk, datetime_cols=["dt"])
        ds = result.datetime["dt"]
        assert ds.min_date is None
        assert ds.max_date is None

    def test_weekend_count(self):
        dates = pd.date_range("2024-01-06", periods=7, freq="D")  # Sat, Sun, Mon, ...
        df = pd.DataFrame({"dt": dates})
        bulk = BulkStats(total_count=7, columns={"dt": PerColumnStats(null_count=0, distinct_count=7)})
        result = _pandas_typed_bulk_stats(df, bulk, datetime_cols=["dt"])
        assert result.datetime["dt"].weekend_count == 2


class TestPandasTypedBulkStatsCategorical:
    def test_basic_categorical(self):
        df = pd.DataFrame({"cat": ["a", "b", "b", "c", "c", "c"]})
        bulk = BulkStats(total_count=6, columns={"cat": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, categorical_cols=["cat"])
        assert "cat" in result.categorical
        cs = result.categorical["cat"]
        assert cs.cardinality == 3
        assert cs.cardinality_ratio > 0
        assert len(cs.top_categories) == 3
        assert cs.top_categories[0] == ("c", 3)
        assert cs.value_counts == {"a": 1, "b": 2, "c": 3}

    def test_rare_categories(self):
        values = ["common"] * 200 + ["rare_a"] * 1 + ["rare_b"] * 1
        df = pd.DataFrame({"cat": values})
        bulk = BulkStats(total_count=202, columns={"cat": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, categorical_cols=["cat"])
        cs = result.categorical["cat"]
        assert cs.rare_category_count == 2
        assert cs.rare_category_percentage > 0

    def test_case_variations(self):
        df = pd.DataFrame({"cat": ["New York", "new york", "NEW YORK", "Boston"]})
        bulk = BulkStats(total_count=4, columns={"cat": PerColumnStats(null_count=0, distinct_count=4)})
        result = _pandas_typed_bulk_stats(df, bulk, categorical_cols=["cat"])
        assert len(result.categorical["cat"].case_variations) > 0

    def test_empty_categorical(self):
        df = pd.DataFrame({"cat": pd.array([None, None, None], dtype="object")})
        bulk = BulkStats(total_count=3, columns={"cat": PerColumnStats(null_count=3, distinct_count=0)})
        result = _pandas_typed_bulk_stats(df, bulk, categorical_cols=["cat"])
        cs = result.categorical["cat"]
        assert cs.cardinality == 0

    def test_high_cardinality_skipped(self):
        df = pd.DataFrame({"cat": [f"val_{i}" for i in range(200)]})
        bulk = BulkStats(
            total_count=200,
            columns={"cat": PerColumnStats(null_count=0, distinct_count=200)},
        )
        result = _pandas_typed_bulk_stats(
            df, bulk, categorical_cols=["cat"], cardinality_limit=100
        )
        cs = result.categorical["cat"]
        assert cs.cardinality == 200
        assert cs.value_counts == {}


class TestPandasTypedBulkStatsIdentifier:
    def test_unique_identifier(self):
        df = pd.DataFrame({"id": [f"ID-{i:05d}" for i in range(100)]})
        bulk = BulkStats(
            total_count=100,
            columns={"id": PerColumnStats(null_count=0, distinct_count=100)},
        )
        result = _pandas_typed_bulk_stats(df, bulk, identifier_cols=["id"])
        ids = result.identifier["id"]
        assert ids.is_unique is True
        assert ids.duplicate_count == 0
        assert ids.length_min == 8
        assert ids.length_max == 8

    def test_duplicates(self):
        df = pd.DataFrame({"id": ["A", "A", "B", "C", "C"]})
        bulk = BulkStats(total_count=5, columns={"id": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, identifier_cols=["id"])
        ids = result.identifier["id"]
        assert ids.is_unique is False
        assert ids.duplicate_count == 2

    def test_lengths(self):
        df = pd.DataFrame({"id": ["ab", "abc", "abcd"]})
        bulk = BulkStats(total_count=3, columns={"id": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, identifier_cols=["id"])
        ids = result.identifier["id"]
        assert ids.length_min == 2
        assert ids.length_max == 4


class TestPandasTypedBulkStatsBinary:
    def test_boolean_column(self):
        df = pd.DataFrame({"flag": [True, False, True, True, False]})
        bulk = BulkStats(
            total_count=5,
            columns={
                "flag": PerColumnStats(
                    null_count=0, distinct_count=2,
                    most_common_value="True", most_common_frequency=3,
                ),
            },
        )
        result = _pandas_typed_bulk_stats(df, bulk, binary_cols=["flag"])
        bs = result.binary["flag"]
        assert bs.true_count == 3
        assert bs.false_count == 2
        assert bs.is_boolean is True
        assert bs.balance_ratio > 0

    def test_zero_one_column(self):
        df = pd.DataFrame({"flag": [0, 1, 1, 0, 1]})
        bulk = BulkStats(
            total_count=5,
            columns={
                "flag": PerColumnStats(
                    null_count=0, distinct_count=2,
                    most_common_value="1", most_common_frequency=3,
                ),
            },
        )
        result = _pandas_typed_bulk_stats(df, bulk, binary_cols=["flag"])
        bs = result.binary["flag"]
        assert bs.true_count + bs.false_count == 5

    def test_yes_no_column(self):
        df = pd.DataFrame({"flag": ["yes", "no", "yes", "no"]})
        bulk = BulkStats(
            total_count=4,
            columns={
                "flag": PerColumnStats(
                    null_count=0, distinct_count=2,
                    most_common_value="yes", most_common_frequency=2,
                ),
            },
        )
        result = _pandas_typed_bulk_stats(df, bulk, binary_cols=["flag"])
        bs = result.binary["flag"]
        assert bs.true_count + bs.false_count == 4


class TestPandasTypedBulkStatsText:
    def test_basic_text(self):
        df = pd.DataFrame({
            "text": [
                "Hello world",
                "This is a longer sentence with more words",
                "Short",
                "",
                "Another sentence",
            ],
        })
        bulk = BulkStats(total_count=5, columns={"text": PerColumnStats(null_count=0, distinct_count=5)})
        result = _pandas_typed_bulk_stats(df, bulk, text_cols=["text"])
        ts = result.text["text"]
        assert ts.length_min == 0
        assert ts.length_max > 0
        assert ts.length_mean > 0
        assert ts.empty_count == 1
        assert ts.word_count_mean > 0

    def test_text_with_digits(self):
        df = pd.DataFrame({"text": ["abc123", "def456", "pure_text"]})
        bulk = BulkStats(total_count=3, columns={"text": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, text_cols=["text"])
        ts = result.text["text"]
        assert ts.contains_digits_pct > 0

    def test_text_with_special(self):
        df = pd.DataFrame({"text": ["hello!", "world?", "plain"]})
        bulk = BulkStats(total_count=3, columns={"text": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, text_cols=["text"])
        ts = result.text["text"]
        assert ts.contains_special_pct > 0

    def test_pii_email_detection(self):
        df = pd.DataFrame({"text": ["user@example.com", "another@test.org", "no email"]})
        bulk = BulkStats(total_count=3, columns={"text": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, text_cols=["text"])
        ts = result.text["text"]
        assert ts.pii_email_count == 2

    def test_pii_phone_detection(self):
        df = pd.DataFrame({"text": ["Call 555-123-4567", "No phone", "800-555-1234"]})
        bulk = BulkStats(total_count=3, columns={"text": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, text_cols=["text"])
        ts = result.text["text"]
        assert ts.pii_phone_count == 2

    def test_pii_ssn_detection(self):
        df = pd.DataFrame({"text": ["SSN: 123-45-6789", "Normal text", "456-78-9012"]})
        bulk = BulkStats(total_count=3, columns={"text": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk, text_cols=["text"])
        ts = result.text["text"]
        assert ts.pii_ssn_count == 2


class TestComputeTypedBulkStatsDispatch:
    def test_pandas_dispatch(self):
        df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=5)})
        bulk = BulkStats(total_count=5, columns={"dt": PerColumnStats(null_count=0, distinct_count=5)})
        result = compute_typed_bulk_stats(df, bulk, datetime_cols=["dt"])
        assert isinstance(result, TypedBulkStats)
        assert "dt" in result.datetime

    def test_spark_dispatch(self):
        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        bulk = BulkStats(total_count=10, columns={})

        with patch(
            "customer_retention.core.compat.bulk_profiling._spark_typed_bulk_stats"
        ) as mock_spark:
            mock_spark.return_value = TypedBulkStats()
            result = compute_typed_bulk_stats(mock_df, bulk, datetime_cols=["dt"])
            mock_spark.assert_called_once()
            assert isinstance(result, TypedBulkStats)


class TestSparkTypedBulkStatsTypeFiltering:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_non_string_columns_filtered_from_text_cols(self):
        from pyspark.sql.types import DoubleType, StringType, StructField, StructType

        from customer_retention.core.compat.bulk_profiling import _spark_typed_bulk_stats

        schema = StructType([
            StructField("str_col", StringType(), True),
            StructField("num_col", DoubleType(), True),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema
        mock_row = MagicMock()
        mock_row.__getitem__ = MagicMock(return_value=0)
        mock_spark_df.agg.return_value.collect.return_value = [mock_row]
        mock_spark_df.groupBy.return_value.agg.return_value.collect.return_value = [mock_row]

        bulk = BulkStats(
            total_count=10,
            columns={
                "str_col": PerColumnStats(null_count=0, distinct_count=5),
                "num_col": PerColumnStats(null_count=0, distinct_count=5),
            },
        )

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_spark_df):
            result = _spark_typed_bulk_stats(
                mock_spark_df, bulk,
                text_cols=["str_col", "num_col"],
                identifier_cols=["num_col"],
            )

        assert "str_col" in result.text
        assert "num_col" not in result.text
        assert "num_col" not in result.identifier

    def test_all_non_string_text_cols_produces_empty_result(self):
        from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

        from customer_retention.core.compat.bulk_profiling import _spark_typed_bulk_stats

        schema = StructType([
            StructField("a", DoubleType(), True),
            StructField("b", IntegerType(), True),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema
        bulk = BulkStats(total_count=5, columns={})

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_spark_df):
            result = _spark_typed_bulk_stats(mock_spark_df, bulk, text_cols=["a", "b"])

        assert result.text == {}
        assert result.identifier == {}


class TestSparkTypedBulkStatsDatetimeFiltering:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_string_datetime_cols_filtered_out(self):
        from pyspark.sql.types import StringType, StructField, StructType, TimestampNTZType

        from customer_retention.core.compat.bulk_profiling import _spark_typed_bulk_stats

        schema = StructType([
            StructField("real_ts", TimestampNTZType(), True),
            StructField("quarter_str", StringType(), True),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema
        mock_row = MagicMock()
        mock_row.__getitem__ = MagicMock(return_value=None)
        mock_spark_df.agg.return_value.collect.return_value = [mock_row]

        bulk = BulkStats(
            total_count=10,
            columns={
                "real_ts": PerColumnStats(null_count=0, distinct_count=5),
                "quarter_str": PerColumnStats(null_count=0, distinct_count=4),
            },
        )

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_spark_df):
            result = _spark_typed_bulk_stats(
                mock_spark_df, bulk, datetime_cols=["real_ts", "quarter_str"],
            )

        assert "quarter_str" not in result.datetime

    def test_all_string_datetime_cols_skips_batch(self):
        from pyspark.sql.types import StringType, StructField, StructType

        from customer_retention.core.compat.bulk_profiling import _spark_typed_bulk_stats

        schema = StructType([
            StructField("q_col", StringType(), True),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema
        bulk = BulkStats(total_count=5, columns={})

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_spark_df):
            result = _spark_typed_bulk_stats(mock_spark_df, bulk, datetime_cols=["q_col"])

        assert result.datetime == {}
        mock_spark_df.agg.assert_not_called()

    def test_date_type_cols_included(self):
        from pyspark.sql.types import DateType, StructField, StructType

        from customer_retention.core.compat.bulk_profiling import _spark_typed_bulk_stats

        schema = StructType([StructField("d_col", DateType(), True)])
        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema
        mock_row = MagicMock()
        mock_row.__getitem__ = MagicMock(return_value=None)
        mock_spark_df.agg.return_value.collect.return_value = [mock_row]

        bulk = BulkStats(
            total_count=5,
            columns={"d_col": PerColumnStats(null_count=0, distinct_count=3)},
        )

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_spark_df):
            result = _spark_typed_bulk_stats(mock_spark_df, bulk, datetime_cols=["d_col"])

        mock_spark_df.agg.assert_called_once()


class TestPandasTypedBulkStatsNoColumns:
    def test_no_columns_returns_empty(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        bulk = BulkStats(total_count=3, columns={"x": PerColumnStats(null_count=0, distinct_count=3)})
        result = _pandas_typed_bulk_stats(df, bulk)
        assert result.datetime == {}
        assert result.categorical == {}
        assert result.identifier == {}
        assert result.binary == {}
        assert result.text == {}

    def test_multiple_types_at_once(self):
        df = pd.DataFrame({
            "dt": pd.date_range("2020-01-01", periods=5),
            "cat": ["a", "b", "b", "a", "c"],
            "flag": [True, False, True, False, True],
        })
        bulk = BulkStats(
            total_count=5,
            columns={
                "dt": PerColumnStats(null_count=0, distinct_count=5),
                "cat": PerColumnStats(null_count=0, distinct_count=3),
                "flag": PerColumnStats(
                    null_count=0, distinct_count=2,
                    most_common_value="True", most_common_frequency=3,
                ),
            },
        )
        result = _pandas_typed_bulk_stats(
            df, bulk,
            datetime_cols=["dt"],
            categorical_cols=["cat"],
            binary_cols=["flag"],
        )
        assert "dt" in result.datetime
        assert "cat" in result.categorical
        assert "flag" in result.binary
