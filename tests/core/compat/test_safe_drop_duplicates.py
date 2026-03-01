from __future__ import annotations

import pandas as pd
import pytest


class TestSafeDropDuplicates:
    def test_keep_last_removes_earlier_rows(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({"id": ["A", "A", "B"], "val": [1, 2, 3]})
        result = safe_drop_duplicates(df, subset=["id"], keep="last")
        assert len(result) == 2
        a_val = result[result["id"] == "A"]["val"].iloc[0]
        assert a_val == 2

    def test_keep_first_removes_later_rows(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({"id": ["A", "A", "B"], "val": [1, 2, 3]})
        result = safe_drop_duplicates(df, subset=["id"], keep="first")
        assert len(result) == 2
        a_val = result[result["id"] == "A"]["val"].iloc[0]
        assert a_val == 1

    def test_no_duplicates_returns_unchanged(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({"id": ["A", "B", "C"], "val": [1, 2, 3]})
        result = safe_drop_duplicates(df, subset=["id"], keep="last")
        assert len(result) == 3

    def test_multi_column_subset(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({
            "id": ["A", "A", "A"],
            "date": ["2024-01-01", "2024-01-01", "2024-02-01"],
            "val": [10, 20, 30],
        })
        result = safe_drop_duplicates(df, subset=["id", "date"], keep="last")
        assert len(result) == 2

    def test_empty_dataframe(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({"id": [], "val": []})
        result = safe_drop_duplicates(df, subset=["id"], keep="last")
        assert len(result) == 0

    def test_single_row(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({"id": ["A"], "val": [42]})
        result = safe_drop_duplicates(df, subset=["id"], keep="last")
        assert len(result) == 1
        assert result["val"].iloc[0] == 42

    def test_all_duplicates_keeps_one(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({"id": ["A", "A", "A"], "val": [1, 2, 3]})
        result = safe_drop_duplicates(df, subset=["id"], keep="last")
        assert len(result) == 1

    def test_preserves_non_subset_columns(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({"id": ["A", "A"], "x": [1, 2], "y": [3, 4]})
        result = safe_drop_duplicates(df, subset=["id"], keep="last")
        assert list(result.columns) == ["id", "x", "y"]

    def test_default_keep_is_last(self):
        from customer_retention.core.compat import safe_drop_duplicates
        df = pd.DataFrame({"id": ["A", "A"], "val": [1, 2]})
        result = safe_drop_duplicates(df, subset=["id"])
        assert len(result) == 1
        assert result["val"].iloc[0] == 2


class TestIsNativeSparkDf:
    def test_pandas_dataframe_returns_false(self):
        from customer_retention.core.compat import _is_native_spark_df
        assert _is_native_spark_df(pd.DataFrame()) is False

    def test_plain_object_returns_false(self):
        from customer_retention.core.compat import _is_native_spark_df
        assert _is_native_spark_df({"a": 1}) is False

    def test_does_not_access_rdd_attribute(self):
        from customer_retention.core.compat import _is_native_spark_df

        class FakeSparkDF:
            @property
            def rdd(self):
                raise RuntimeError("RDD access not allowed on shared clusters")

        assert _is_native_spark_df(FakeSparkDF()) is False

    def test_safe_drop_duplicates_does_not_access_rdd(self):
        from customer_retention.core.compat import safe_drop_duplicates

        class SharedClusterDF:
            @property
            def rdd(self):
                raise RuntimeError("RDD access not allowed on shared clusters")

            def drop_duplicates(self, subset=None, keep="first"):
                return pd.DataFrame({"id": ["A"], "val": [2]})

        df = SharedClusterDF()
        result = safe_drop_duplicates(df, subset=["id"], keep="last")
        assert len(result) == 1


class TestSafeDropDuplicatesInExports:
    def test_in_compat_all(self):
        from customer_retention.core import compat
        assert "safe_drop_duplicates" in compat.__all__
