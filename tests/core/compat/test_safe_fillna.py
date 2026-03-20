from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSafeFillna:
    def test_fills_nan_with_scalar(self):
        from customer_retention.core.compat import safe_fillna
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        result = safe_fillna(df, 0)
        assert result["a"].tolist() == [1.0, 0.0, 3.0]
        assert result["b"].tolist() == [0.0, 2.0, 0.0]

    def test_preserves_columns(self):
        from customer_retention.core.compat import safe_fillna
        df = pd.DataFrame({"x": [np.nan], "y": [1.0], "z": [np.nan]})
        result = safe_fillna(df, 0)
        assert list(result.columns) == ["x", "y", "z"]

    def test_no_nans_is_noop(self):
        from customer_retention.core.compat import safe_fillna
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = safe_fillna(df, 0)
        pd.testing.assert_frame_equal(result, df)

    def test_preserves_dtypes_for_int_columns(self):
        from customer_retention.core.compat import safe_fillna
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = safe_fillna(df, 0)
        assert result["a"].dtype == np.float64

    def test_empty_dataframe(self):
        from customer_retention.core.compat import safe_fillna
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = safe_fillna(df, 0)
        assert len(result) == 0
        assert list(result.columns) == ["a"]

    def test_all_nan_dataframe(self):
        from customer_retention.core.compat import safe_fillna
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        result = safe_fillna(df, -1)
        assert result["a"].tolist() == [-1.0, -1.0]
        assert result["b"].tolist() == [-1.0, -1.0]

    def test_mixed_types_only_fills_numeric(self):
        from customer_retention.core.compat import safe_fillna
        df = pd.DataFrame({"num": [1.0, np.nan], "txt": ["a", None]})
        result = safe_fillna(df, 0)
        assert result["num"].tolist() == [1.0, 0.0]

    def test_returns_new_dataframe(self):
        from customer_retention.core.compat import safe_fillna
        df = pd.DataFrame({"a": [np.nan]})
        result = safe_fillna(df, 0)
        assert result is not df


class TestLazyFillna:
    def test_fills_nan_without_checkpoint(self):
        from customer_retention.core.compat import lazy_fillna
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        result = lazy_fillna(df, 0)
        assert result["a"].tolist() == [1.0, 0.0, 3.0]
        assert result["b"].tolist() == [0.0, 2.0, 0.0]

    def test_preserves_columns(self):
        from customer_retention.core.compat import lazy_fillna
        df = pd.DataFrame({"x": [np.nan], "y": [1.0]})
        result = lazy_fillna(df, 0)
        assert list(result.columns) == ["x", "y"]

    def test_no_nans_is_noop(self):
        from customer_retention.core.compat import lazy_fillna
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = lazy_fillna(df, 0)
        pd.testing.assert_frame_equal(result, df)

    def test_same_result_as_safe_fillna_on_pandas(self):
        from customer_retention.core.compat import lazy_fillna, safe_fillna
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        result_lazy = lazy_fillna(df, 0)
        result_safe = safe_fillna(df, 0)
        pd.testing.assert_frame_equal(result_lazy, result_safe)


class TestSafeFillnaExport:
    def test_safe_fillna_in_compat(self):
        from customer_retention.core import compat
        assert hasattr(compat, "safe_fillna")

    def test_lazy_fillna_in_compat(self):
        from customer_retention.core import compat
        assert hasattr(compat, "lazy_fillna")
