from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from customer_retention.core.compat import Series


@dataclass
class BinaryTransformResult:
    series: Series
    mapping: dict = field(default_factory=dict)
    original_values: list = field(default_factory=list)
    positive_class: Any = None


class BinaryHandler:
    TRUE_VALUES = {1, 1.0, True, "1", "yes", "Yes", "YES", "true", "True", "TRUE", "y", "Y"}
    FALSE_VALUES = {0, 0.0, False, "0", "no", "No", "NO", "false", "False", "FALSE", "n", "N"}

    def __init__(self, positive_class: Optional[Any] = None):
        self.positive_class = positive_class
        self._mapping: Optional[dict] = None
        self._original_values: Optional[list] = None
        self._positive: Any = None
        self._is_fitted = False

    def fit(self, series: Series) -> "BinaryHandler":
        clean = series.dropna()
        unique_vals = clean.unique().tolist()
        self._original_values = unique_vals

        if self.positive_class is not None:
            self._positive = self.positive_class
            self._mapping = {v: 1 if v == self.positive_class else 0 for v in unique_vals}
        else:
            self._mapping, self._positive = self._infer_mapping(unique_vals)

        self._is_fitted = True
        return self

    def transform(self, series: Series) -> BinaryTransformResult:
        if not self._is_fitted:
            raise ValueError("Handler not fitted. Call fit() or fit_transform() first.")
        return self._apply_transform(series)

    def fit_transform(self, series: Series) -> BinaryTransformResult:
        self.fit(series)
        return self._apply_transform(series)

    def _infer_mapping(self, unique_vals: list) -> tuple[dict, Any]:
        if len(unique_vals) == 1:
            val = unique_vals[0]
            if val in self.TRUE_VALUES or str(val).lower() in {"yes", "y", "true", "1"}:
                return {val: 1}, val
            return {val: 0}, None

        mapping = {}
        positive = None

        for val in unique_vals:
            val_lower = str(val).lower() if isinstance(val, str) else val
            if val in self.TRUE_VALUES or val_lower in {"yes", "y", "true", "1", "active"}:
                mapping[val] = 1
                positive = val
            elif val in self.FALSE_VALUES or val_lower in {"no", "n", "false", "0", "inactive"}:
                mapping[val] = 0

        if len(mapping) == len(unique_vals) and positive is not None:
            return mapping, positive

        if len(unique_vals) == 2:
            sorted_vals = sorted(unique_vals, key=lambda x: (str(x).lower(), x))
            return {sorted_vals[0]: 0, sorted_vals[1]: 1}, sorted_vals[1]

        return {v: i for i, v in enumerate(unique_vals)}, unique_vals[-1] if unique_vals else None

    def _apply_transform(self, series: Series) -> BinaryTransformResult:
        result = series.map(self._mapping)
        result = result.where(series.notna(), np.nan)

        return BinaryTransformResult(
            series=result, mapping=self._mapping or {},
            original_values=self._original_values or [], positive_class=self._positive
        )
