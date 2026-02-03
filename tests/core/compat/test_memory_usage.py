from unittest.mock import MagicMock

import pandas as pd

from customer_retention.core.compat import safe_memory_usage_bytes


class TestSafeMemoryUsageBytes:
    def test_dataframe_returns_total_bytes(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = safe_memory_usage_bytes(df)
        assert result > 0
        assert result == df.memory_usage(deep=True).sum()

    def test_series_returns_bytes(self):
        s = pd.Series([1, 2, 3])
        result = safe_memory_usage_bytes(s)
        assert result > 0
        assert result == s.memory_usage(deep=True)

    def test_returns_zero_on_not_implemented(self):
        mock_obj = MagicMock()
        mock_obj.memory_usage.side_effect = NotImplementedError("not supported")
        assert safe_memory_usage_bytes(mock_obj) == 0

    def test_returns_zero_on_any_exception(self):
        mock_obj = MagicMock()
        mock_obj.memory_usage.side_effect = Exception("something broke")
        assert safe_memory_usage_bytes(mock_obj) == 0
