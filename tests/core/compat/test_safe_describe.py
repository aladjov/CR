import pandas as pd
import pytest

from customer_retention.core.compat import safe_describe


class TestSafeDescribe:
    def test_numeric_only(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = safe_describe(df)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "mean" in result.index

    def test_mixed_types(self):
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "ts": pd.date_range("2024-01-01", periods=3),
            "cat": ["a", "b", "c"],
        })
        result = safe_describe(df)
        assert len(result.columns) > 0

    def test_timestamp_only_fallback(self):
        df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3)})
        result = safe_describe(df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = safe_describe(df)
        assert isinstance(result, pd.DataFrame)

    def test_describe_failure_falls_back_to_numeric(self):
        class BrokenDescribeDF:
            def __init__(self):
                self._df = pd.DataFrame({"a": [1, 2, 3]})
                self.columns = self._df.columns

            def describe(self):
                raise RuntimeError("avg on TIMESTAMP_NTZ not supported")

            def select_dtypes(self, include=None):
                return self._df.select_dtypes(include=include)

        result = safe_describe(BrokenDescribeDF())
        assert "a" in result.columns
        assert "mean" in result.index
