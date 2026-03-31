from __future__ import annotations

import pandas as pd

from customer_retention.core.compat import safe_len


class TestSafeLen:
    def test_native_pandas(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert safe_len(df) == 3

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": []})
        assert safe_len(df) == 0

    def test_returns_int(self):
        df = pd.DataFrame({"a": range(10)})
        result = safe_len(df)
        assert isinstance(result, int)
