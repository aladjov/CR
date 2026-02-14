from pathlib import Path
from unittest.mock import patch

import pandas as pd

from customer_retention.core.compat import is_dataframe


class TestIsDataframeLocalPandas:
    def test_pandas_dataframe(self):
        assert is_dataframe(pd.DataFrame({"a": [1]})) is True

    def test_pandas_series_is_not_dataframe(self):
        assert is_dataframe(pd.Series([1, 2])) is False

    def test_string_is_not_dataframe(self):
        assert is_dataframe("some/path.csv") is False

    def test_path_is_not_dataframe(self):
        assert is_dataframe(Path("some/path.csv")) is False

    def test_dict_is_not_dataframe(self):
        assert is_dataframe({"a": 1}) is False

    def test_none_is_not_dataframe(self):
        assert is_dataframe(None) is False

    def test_int_is_not_dataframe(self):
        assert is_dataframe(42) is False


class _FakeSparkDataFrame:
    pass


class TestIsDataframeSparkTypes:
    def test_recognizes_extra_type_in_dataframe_types(self):
        extended = (pd.DataFrame, _FakeSparkDataFrame)
        with patch("customer_retention.core.compat._DATAFRAME_TYPES", extended):
            assert is_dataframe(_FakeSparkDataFrame()) is True

    def test_still_recognizes_pandas_with_extended_types(self):
        extended = (pd.DataFrame, _FakeSparkDataFrame)
        with patch("customer_retention.core.compat._DATAFRAME_TYPES", extended):
            assert is_dataframe(pd.DataFrame({"a": [1]})) is True

    def test_rejects_string_with_extended_types(self):
        extended = (pd.DataFrame, _FakeSparkDataFrame)
        with patch("customer_retention.core.compat._DATAFRAME_TYPES", extended):
            assert is_dataframe("some/path.csv") is False
