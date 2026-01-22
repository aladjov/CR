from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field
import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series


class EncodingStrategy(str, Enum):
    ONE_HOT = "one_hot"
    LABEL = "label"
    ORDINAL = "ordinal"
    CYCLICAL = "cyclical"
    TARGET = "target"
    FREQUENCY = "frequency"
    BINARY = "binary"
    HASH = "hash"


@dataclass
class CategoricalEncodeResult:
    series: Optional[Series] = None
    df: Optional[DataFrame] = None
    strategy: EncodingStrategy = EncodingStrategy.LABEL
    columns_created: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)
    dropped_categories: list = field(default_factory=list)


class CategoricalEncoder:
    def __init__(
        self,
        strategy: EncodingStrategy = EncodingStrategy.LABEL,
        drop_first: bool = True,
        handle_unknown: str = "error",
        categories: Optional[list] = None,
        period: Optional[int] = None,
        smoothing: float = 1.0,
        min_frequency: Optional[int] = None
    ):
        self.strategy = strategy
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.categories = categories
        self.period = period
        self.smoothing = smoothing
        self.min_frequency = min_frequency
        self._mapping: Optional[dict] = None
        self._categories: Optional[list] = None
        self._target_means: Optional[dict] = None
        self._global_mean: Optional[float] = None
        self._frequencies: Optional[dict] = None
        self._cyclical_mapping: Optional[dict] = None
        self._is_fitted = False

    def fit(self, series: Series, target: Optional[Series] = None) -> "CategoricalEncoder":
        clean = series.dropna()

        if self.strategy == EncodingStrategy.ONE_HOT:
            self._fit_one_hot(clean)
        elif self.strategy == EncodingStrategy.LABEL:
            self._fit_label(clean)
        elif self.strategy == EncodingStrategy.ORDINAL:
            self._fit_ordinal(clean)
        elif self.strategy == EncodingStrategy.CYCLICAL:
            self._fit_cyclical(clean)
        elif self.strategy == EncodingStrategy.FREQUENCY:
            self._fit_frequency(clean)
        elif self.strategy == EncodingStrategy.TARGET:
            self._fit_target(clean, target)

        self._is_fitted = True
        return self

    def transform(self, series: Series, target: Optional[Series] = None) -> CategoricalEncodeResult:
        if not self._is_fitted:
            raise ValueError("Encoder not fitted. Call fit() or fit_transform() first.")
        return self._apply_encoding(series, target)

    def fit_transform(self, series: Series, target: Optional[Series] = None) -> CategoricalEncodeResult:
        self.fit(series, target)
        return self._apply_encoding(series, target)

    def _fit_one_hot(self, clean: Series):
        categories = clean.unique().tolist()
        if self.min_frequency is not None:
            value_counts = clean.value_counts()
            categories = [c for c in categories if value_counts.get(c, 0) >= self.min_frequency]
        self._categories = sorted(categories)
        self._mapping = {cat: i for i, cat in enumerate(self._categories)}

    def _fit_label(self, clean: Series):
        categories = sorted(clean.unique().tolist())
        self._mapping = {cat: i for i, cat in enumerate(categories)}

    def _fit_ordinal(self, clean: Series):
        if self.categories is None:
            raise ValueError("Ordinal encoding requires categories parameter")
        self._mapping = {cat: i for i, cat in enumerate(self.categories)}

    def _fit_cyclical(self, clean: Series):
        if self.period is None:
            raise ValueError("Cyclical encoding requires period parameter")
        # Check if values are strings and need mapping to indices
        if clean.dtype == object:
            unique_values = sorted(clean.unique().tolist())
            # Auto-detect day of week names
            day_names = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6,
                'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6
            }
            month_names = {
                'january': 0, 'february': 1, 'march': 2, 'april': 3, 'may': 4, 'june': 5,
                'july': 6, 'august': 7, 'september': 8, 'october': 9, 'november': 10, 'december': 11,
                'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'jun': 5, 'jul': 6, 'aug': 7,
                'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
            }
            # Try to auto-detect mapping from common patterns
            sample_lower = [str(v).lower() for v in unique_values]
            if all(s in day_names for s in sample_lower):
                self._cyclical_mapping = {v: day_names[str(v).lower()] for v in unique_values}
            elif all(s in month_names for s in sample_lower):
                self._cyclical_mapping = {v: month_names[str(v).lower()] for v in unique_values}
            else:
                # Generic mapping: assign indices based on order
                self._cyclical_mapping = {v: i for i, v in enumerate(unique_values)}
        else:
            self._cyclical_mapping = None

    def _fit_frequency(self, clean: Series):
        total = len(clean)
        value_counts = clean.value_counts()
        self._frequencies = {cat: count / total for cat, count in value_counts.items()}

    def _fit_target(self, clean: Series, target: Optional[Series]):
        if target is None:
            raise ValueError("Target encoding requires target parameter")

        self._global_mean = target.mean()
        self._target_means = {}

        for cat in clean.unique():
            mask = clean == cat
            cat_target = target[mask]
            n = len(cat_target)
            cat_mean = cat_target.mean()

            smoothed = (cat_target.sum() + self.smoothing * self._global_mean) / (n + self.smoothing)
            self._target_means[cat] = smoothed

    def _apply_encoding(self, series: Series, target: Optional[Series] = None) -> CategoricalEncodeResult:
        if self.strategy == EncodingStrategy.ONE_HOT:
            return self._encode_one_hot(series)
        elif self.strategy == EncodingStrategy.LABEL:
            return self._encode_label(series)
        elif self.strategy == EncodingStrategy.ORDINAL:
            return self._encode_ordinal(series)
        elif self.strategy == EncodingStrategy.CYCLICAL:
            return self._encode_cyclical(series)
        elif self.strategy == EncodingStrategy.FREQUENCY:
            return self._encode_frequency(series)
        elif self.strategy == EncodingStrategy.TARGET:
            return self._encode_target(series)

        return CategoricalEncodeResult(series=series, strategy=self.strategy)

    def _encode_one_hot(self, series: Series) -> CategoricalEncodeResult:
        categories = self._categories if self._categories else sorted(series.dropna().unique().tolist())
        if self.drop_first and len(categories) > 0:
            categories = categories[1:]

        cols = {}
        col_names = []
        for cat in categories:
            col_name = f"{series.name or 'col'}_{cat}"
            cols[col_name] = (series == cat).astype(int)
            col_names.append(col_name)

        if self.handle_unknown == "ignore":
            for col in cols:
                known_cats = set(self._categories) if self._categories else set()
                unknown_mask = ~series.isin(known_cats) & series.notna()
                cols[col] = cols[col].where(~unknown_mask, 0)

        df = DataFrame(cols)

        return CategoricalEncodeResult(
            df=df, strategy=self.strategy,
            columns_created=col_names, mapping=self._mapping or {}
        )

    def _encode_label(self, series: Series) -> CategoricalEncodeResult:
        result = series.map(self._mapping)
        return CategoricalEncodeResult(
            series=result, strategy=self.strategy, mapping=self._mapping or {}
        )

    def _encode_ordinal(self, series: Series) -> CategoricalEncodeResult:
        unknown = series[series.notna() & ~series.isin(self._mapping.keys())]
        if len(unknown) > 0 and self.handle_unknown == "error":
            raise ValueError(f"Found unknown categories: {unknown.unique().tolist()}")

        result = series.map(self._mapping)
        return CategoricalEncodeResult(
            series=result, strategy=self.strategy, mapping=self._mapping or {}
        )

    def _encode_cyclical(self, series: Series) -> CategoricalEncodeResult:
        # Map strings to numeric indices if mapping exists
        if hasattr(self, '_cyclical_mapping') and self._cyclical_mapping is not None:
            numeric = series.map(self._cyclical_mapping)
        else:
            numeric = pd.to_numeric(series, errors='coerce')

        sin_vals = np.sin(2 * np.pi * numeric / self.period)
        cos_vals = np.cos(2 * np.pi * numeric / self.period)

        col_name = series.name or "col"
        sin_col = f"{col_name}_sin"
        cos_col = f"{col_name}_cos"

        df = DataFrame({sin_col: sin_vals, cos_col: cos_vals})

        return CategoricalEncodeResult(
            df=df, strategy=self.strategy, columns_created=[sin_col, cos_col],
            mapping=self._cyclical_mapping if hasattr(self, '_cyclical_mapping') else {}
        )

    def _encode_frequency(self, series: Series) -> CategoricalEncodeResult:
        result = series.map(self._frequencies)
        return CategoricalEncodeResult(
            series=result, strategy=self.strategy, mapping=self._frequencies or {}
        )

    def _encode_target(self, series: Series) -> CategoricalEncodeResult:
        result = series.map(self._target_means)
        unknown_mask = result.isna() & series.notna()
        result = result.fillna(self._global_mean)
        result = result.where(series.notna(), np.nan)

        return CategoricalEncodeResult(
            series=result, strategy=self.strategy, mapping=self._target_means or {}
        )
