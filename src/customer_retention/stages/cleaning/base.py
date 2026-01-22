from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from customer_retention.core.compat import Series

TResult = TypeVar('TResult')


class BaseHandler(ABC, Generic[TResult]):
    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, series: Series, **kwargs) -> "BaseHandler[TResult]":
        pass

    @abstractmethod
    def _apply(self, series: Series, **kwargs) -> TResult:
        pass

    def transform(self, series: Series, **kwargs) -> TResult:
        if not self._is_fitted:
            raise ValueError("Handler not fitted. Call fit() or fit_transform() first.")
        return self._apply(series, **kwargs)

    def fit_transform(self, series: Series, **kwargs) -> TResult:
        self.fit(series, **kwargs)
        return self._apply(series, **kwargs)
