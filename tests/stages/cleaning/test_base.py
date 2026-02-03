"""Tests for BaseHandler abstract class."""
import pandas as pd
import pytest

from customer_retention.stages.cleaning.base import BaseHandler


class ConcreteHandler(BaseHandler[pd.Series]):
    """Concrete implementation of BaseHandler for testing."""

    def fit(self, series, **kwargs):
        self._is_fitted = True
        self._fit_value = series.mean()
        return self

    def _apply(self, series, **kwargs):
        return series.fillna(self._fit_value)


class TestBaseHandlerInit:
    def test_initial_state_not_fitted(self):
        handler = ConcreteHandler()
        assert handler._is_fitted is False


class TestTransform:
    def test_raises_when_not_fitted(self):
        handler = ConcreteHandler()
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Handler not fitted"):
            handler.transform(series)

    def test_works_after_fit(self):
        handler = ConcreteHandler()
        series = pd.Series([1, 2, None, 4])
        handler.fit(series)
        result = handler.transform(series)
        assert result.isna().sum() == 0

    def test_transform_delegates_to_apply(self):
        handler = ConcreteHandler()
        series = pd.Series([1, 2, None])
        handler.fit(series)
        result = handler.transform(series)
        assert result.iloc[2] == pytest.approx(1.5)


class TestFitTransform:
    def test_fit_transform_fits_and_applies(self):
        handler = ConcreteHandler()
        series = pd.Series([1, 2, None, 4])
        result = handler.fit_transform(series)
        assert result.isna().sum() == 0
        assert handler._is_fitted is True

    def test_fit_transform_result_matches_separate_fit_apply(self):
        handler1 = ConcreteHandler()
        handler2 = ConcreteHandler()
        series = pd.Series([10, 20, None, 40])
        result_combined = handler1.fit_transform(series)
        handler2.fit(series)
        result_separate = handler2.transform(series)
        pd.testing.assert_series_equal(result_combined, result_separate)

    def test_can_transform_after_fit_transform(self):
        handler = ConcreteHandler()
        series = pd.Series([1, 2, 3])
        handler.fit_transform(series)
        # Should be fitted now, so transform should work
        new_series = pd.Series([10, None, 30])
        result = handler.transform(new_series)
        assert result.isna().sum() == 0


class TestFit:
    def test_fit_sets_fitted_flag(self):
        handler = ConcreteHandler()
        series = pd.Series([1, 2, 3])
        handler.fit(series)
        assert handler._is_fitted is True

    def test_fit_returns_self(self):
        handler = ConcreteHandler()
        series = pd.Series([1, 2, 3])
        result = handler.fit(series)
        assert result is handler
