"""Tests for normalize_decimal_columns."""
import decimal

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import normalize_decimal_columns


class TestNormalizeDecimalPandas:
    def test_converts_decimal_object_column(self):
        df = pd.DataFrame({"amount": [decimal.Decimal("1.5"), decimal.Decimal("2.3")]})
        result = normalize_decimal_columns(df)
        assert result["amount"].dtype == np.float64
        assert list(result["amount"]) == pytest.approx([1.5, 2.3])

    def test_leaves_float_columns_unchanged(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = normalize_decimal_columns(df)
        assert result["value"].dtype == np.float64
        assert list(result["value"]) == [1.0, 2.0, 3.0]

    def test_leaves_non_decimal_object_column(self):
        df = pd.DataFrame({"name": ["alice", "bob"]})
        result = normalize_decimal_columns(df)
        assert result["name"].dtype == object

    def test_handles_nulls_in_decimal_column(self):
        df = pd.DataFrame({"amount": [decimal.Decimal("1.5"), None, decimal.Decimal("3.0")]})
        result = normalize_decimal_columns(df)
        assert result["amount"].dtype == np.float64
        assert pd.isna(result["amount"].iloc[1])

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"amount": [decimal.Decimal("1.5"), decimal.Decimal("2.3")]})
        original_dtype = df["amount"].dtype
        normalize_decimal_columns(df)
        assert df["amount"].dtype == original_dtype

    def test_mixed_decimal_and_numeric_columns(self):
        df = pd.DataFrame({
            "price": [decimal.Decimal("9.99"), decimal.Decimal("19.99")],
            "count": [1, 2],
        })
        result = normalize_decimal_columns(df)
        assert result["price"].dtype == np.float64
        assert result["count"].dtype in (np.int64, np.int_)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"amount": pd.Series([], dtype=object)})
        result = normalize_decimal_columns(df)
        assert len(result) == 0
