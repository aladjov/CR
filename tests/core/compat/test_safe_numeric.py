import numpy as np
import pandas as pd

from customer_retention.core.compat import safe_isfinite, safe_isinf


class TestSafeIsinf:
    def test_detects_positive_inf(self):
        series = pd.Series([1.0, float('inf'), 3.0])
        result = safe_isinf(series)
        assert result.tolist() == [False, True, False]

    def test_detects_negative_inf(self):
        series = pd.Series([1.0, float('-inf'), 3.0])
        result = safe_isinf(series)
        assert result.tolist() == [False, True, False]

    def test_returns_false_for_normal_values(self):
        series = pd.Series([1.0, 2.0, 3.0, -5.0, 0.0])
        result = safe_isinf(series)
        assert not result.any()

    def test_handles_nan(self):
        series = pd.Series([1.0, float('nan'), float('inf')])
        result = safe_isinf(series)
        assert result.tolist() == [False, False, True]

    def test_both_inf_variants(self):
        series = pd.Series([float('-inf'), 0.0, float('inf')])
        result = safe_isinf(series)
        assert result.tolist() == [True, False, True]


class TestSafeIsfinite:
    def test_excludes_inf_and_nan(self):
        series = pd.Series([1.0, float('inf'), float('nan'), float('-inf')])
        result = safe_isfinite(series)
        assert result.tolist() == [True, False, False, False]

    def test_includes_normal_values(self):
        series = pd.Series([1.0, -2.5, 0.0, 100.0])
        result = safe_isfinite(series)
        assert result.all()

    def test_excludes_positive_inf(self):
        series = pd.Series([float('inf'), 1.0])
        result = safe_isfinite(series)
        assert result.tolist() == [False, True]

    def test_excludes_negative_inf(self):
        series = pd.Series([float('-inf'), 1.0])
        result = safe_isfinite(series)
        assert result.tolist() == [False, True]

    def test_excludes_nan(self):
        series = pd.Series([float('nan'), 1.0])
        result = safe_isfinite(series)
        assert result.tolist() == [False, True]

    def test_empty_series(self):
        series = pd.Series([], dtype=float)
        result = safe_isfinite(series)
        assert len(result) == 0
