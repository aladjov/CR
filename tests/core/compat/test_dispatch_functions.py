"""Tests for compat dispatch functions: to_datetime, to_numeric, cut, qcut.

When pyspark.pandas is active (pd = ps), these must still work on native
pandas inputs AND pyspark.pandas inputs without crashing.
"""
from unittest.mock import MagicMock

import pandas as native_pd
import pytest

try:
    import pyspark.pandas as ps  # noqa: F401
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

requires_pyspark = pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")


class _FakeSparkSeries:
    """Mimics pyspark.pandas Series for dispatch testing without Spark session."""

    def __init__(self, data):
        self._data = native_pd.Series(data)
        self.spark = True

    def to_pandas(self):
        return self._data

    def __len__(self):
        return len(self._data)


class TestToDatetimeDispatch:
    def test_native_pandas_series(self):
        from customer_retention.core.compat import to_datetime

        s = native_pd.Series(["2024-01-01", "2024-02-01"])
        result = to_datetime(s)
        assert native_pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 2

    @requires_pyspark
    def test_native_pandas_when_pd_is_pyspark(self, monkeypatch):
        import customer_retention.core.compat as compat

        monkeypatch.setattr(compat, "_SPARK_PANDAS_AVAILABLE", True)
        s = native_pd.Series(["2024-01-01", "2024-02-01"])
        result = compat.to_datetime(s)
        assert native_pd.api.types.is_datetime64_any_dtype(result)

    @requires_pyspark
    def test_non_native_dispatches_to_pd(self, monkeypatch):
        import customer_retention.core.compat as compat

        mock_pd = MagicMock()
        mock_pd.to_datetime.return_value = "dispatched"
        monkeypatch.setattr(compat, "_SPARK_PANDAS_AVAILABLE", True)
        monkeypatch.setattr(compat, "pd", mock_pd)
        fake = _FakeSparkSeries(["2024-01-01"])
        result = compat.to_datetime(fake)
        mock_pd.to_datetime.assert_called_once()
        assert result == "dispatched"


class TestToNumericDispatch:
    def test_native_pandas_series(self):
        from customer_retention.core.compat import to_numeric

        s = native_pd.Series(["1", "2", "bad"])
        result = to_numeric(s, errors="coerce")
        assert result[0] == 1
        assert native_pd.isna(result[2])

    @requires_pyspark
    def test_native_pandas_when_pd_is_pyspark(self, monkeypatch):
        import customer_retention.core.compat as compat

        monkeypatch.setattr(compat, "_SPARK_PANDAS_AVAILABLE", True)
        s = native_pd.Series(["1", "2", "bad"])
        result = compat.to_numeric(s, errors="coerce")
        assert result[0] == 1
        assert native_pd.isna(result[2])

    @requires_pyspark
    def test_non_native_dispatches_to_pd(self, monkeypatch):
        import customer_retention.core.compat as compat

        mock_pd = MagicMock()
        mock_pd.to_numeric.return_value = "dispatched"
        monkeypatch.setattr(compat, "_SPARK_PANDAS_AVAILABLE", True)
        monkeypatch.setattr(compat, "pd", mock_pd)
        fake = _FakeSparkSeries(["1", "2"])
        result = compat.to_numeric(fake, errors="coerce")
        mock_pd.to_numeric.assert_called_once()
        assert result == "dispatched"


class TestCutDispatch:
    def test_native_pandas_series(self):
        from customer_retention.core.compat import cut

        s = native_pd.Series([1, 5, 10, 15, 20])
        result = cut(s, bins=3)
        assert len(result) == 5
        assert hasattr(result, "cat")

    @requires_pyspark
    def test_native_pandas_when_pd_is_pyspark(self, monkeypatch):
        import customer_retention.core.compat as compat

        monkeypatch.setattr(compat, "_SPARK_PANDAS_AVAILABLE", True)
        s = native_pd.Series([1, 5, 10, 15, 20])
        result = compat.cut(s, bins=3)
        assert len(result) == 5

    @requires_pyspark
    def test_pyspark_pandas_series_collected(self, monkeypatch):
        import customer_retention.core.compat as compat

        monkeypatch.setattr(compat, "_SPARK_PANDAS_AVAILABLE", True)
        fake = _FakeSparkSeries([1, 5, 10, 15, 20])
        result = compat.cut(fake, bins=3)
        assert len(result) == 5
        assert isinstance(result, native_pd.Series)


class TestQcutDispatch:
    def test_native_pandas_series(self):
        from customer_retention.core.compat import qcut

        s = native_pd.Series(range(100))
        result = qcut(s, q=4, labels=False)
        assert len(result) == 100

    @requires_pyspark
    def test_native_pandas_when_pd_is_pyspark(self, monkeypatch):
        import customer_retention.core.compat as compat

        monkeypatch.setattr(compat, "_SPARK_PANDAS_AVAILABLE", True)
        s = native_pd.Series(range(100))
        result = compat.qcut(s, q=4, labels=False)
        assert len(result) == 100

    @requires_pyspark
    def test_pyspark_pandas_series_collected(self, monkeypatch):
        import customer_retention.core.compat as compat

        monkeypatch.setattr(compat, "_SPARK_PANDAS_AVAILABLE", True)
        fake = _FakeSparkSeries(range(100))
        result = compat.qcut(fake, q=4, labels=False)
        assert len(result) == 100
        assert isinstance(result, native_pd.Series)
