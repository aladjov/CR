from __future__ import annotations

import pandas as pd
import pytest


class TestSafeSample:
    def test_returns_requested_size(self):
        from customer_retention.core.compat import safe_sample
        df = pd.DataFrame({"a": range(100)})
        result = safe_sample(df, 10)
        assert len(result) == 10

    def test_clamps_to_dataframe_length(self):
        from customer_retention.core.compat import safe_sample
        df = pd.DataFrame({"a": range(5)})
        result = safe_sample(df, 100)
        assert len(result) == 5

    def test_reproducible_with_random_state(self):
        from customer_retention.core.compat import safe_sample
        df = pd.DataFrame({"a": range(100)})
        r1 = safe_sample(df, 10, random_state=42)
        r2 = safe_sample(df, 10, random_state=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_preserves_columns(self):
        from customer_retention.core.compat import safe_sample
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = safe_sample(df, 2)
        assert list(result.columns) == ["x", "y"]

    def test_empty_dataframe(self):
        from customer_retention.core.compat import safe_sample
        df = pd.DataFrame({"a": []})
        result = safe_sample(df, 10)
        assert len(result) == 0

    def test_n_zero_returns_empty(self):
        from customer_retention.core.compat import safe_sample
        df = pd.DataFrame({"a": range(10)})
        result = safe_sample(df, 0)
        assert len(result) == 0

    def test_single_row(self):
        from customer_retention.core.compat import safe_sample
        df = pd.DataFrame({"a": [42]})
        result = safe_sample(df, 1)
        assert len(result) == 1
        assert result.iloc[0]["a"] == 42


class TestSafeSampleInAllExports:
    def test_safe_sample_in_compat_all(self):
        from customer_retention.core import compat
        assert "safe_sample" in compat.__all__
