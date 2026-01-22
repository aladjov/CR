from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from customer_retention.core.compat import DataFrame, Series, pd
from customer_retention.core.config import ColumnType


class ImputationStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    DROP_ROW = "drop_row"
    DROP_COLUMN = "drop_column"
    KNN = "knn"
    ITERATIVE = "iterative"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    ERROR = "error"


@dataclass
class ImputationResult:
    series: Series
    strategy_used: ImputationStrategy
    values_imputed: int
    fill_value: Optional[Any] = None
    indicator_column: Optional[Series] = None
    rows_dropped: int = 0
    drop_mask: Optional[list[bool]] = None


class MissingValueHandler:
    DEFAULT_STRATEGIES = {
        ColumnType.IDENTIFIER: ImputationStrategy.ERROR,
        ColumnType.TARGET: ImputationStrategy.DROP_ROW,
        ColumnType.NUMERIC_CONTINUOUS: ImputationStrategy.MEDIAN,
        ColumnType.NUMERIC_DISCRETE: ImputationStrategy.MODE,
        ColumnType.CATEGORICAL_NOMINAL: ImputationStrategy.MODE,
        ColumnType.CATEGORICAL_ORDINAL: ImputationStrategy.MODE,
        ColumnType.CATEGORICAL_CYCLICAL: ImputationStrategy.MODE,
        ColumnType.DATETIME: ImputationStrategy.DROP_ROW,
        ColumnType.BINARY: ImputationStrategy.MODE,
        ColumnType.TEXT: ImputationStrategy.CONSTANT,
    }

    def __init__(
        self,
        strategy: ImputationStrategy = ImputationStrategy.MEDIAN,
        fill_value: Optional[Any] = None,
        knn_neighbors: int = 5,
        add_indicator: bool = False
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        self.knn_neighbors = knn_neighbors
        self.add_indicator = add_indicator
        self._fitted_value: Optional[Any] = None
        self._is_fitted = False

    @classmethod
    def from_column_type(cls, column_type: ColumnType, **kwargs) -> "MissingValueHandler":
        strategy = cls.DEFAULT_STRATEGIES.get(column_type, ImputationStrategy.MODE)
        fill_value = "" if column_type == ColumnType.TEXT else kwargs.get("fill_value")
        return cls(strategy=strategy, fill_value=fill_value, **kwargs)

    def fit(self, series: Series, reference_df: Optional[DataFrame] = None) -> "MissingValueHandler":
        clean_series = series.dropna()
        if len(clean_series) == 0:
            raise ValueError("Cannot fit imputer: all values are missing")

        self._fitted_value = self._compute_fill_value(clean_series)
        self._is_fitted = True
        return self

    def transform(self, series: Series, reference_df: Optional[DataFrame] = None) -> ImputationResult:
        if not self._is_fitted:
            raise ValueError("Handler not fitted. Call fit() or fit_transform() first.")
        return self._apply_imputation(series, reference_df)

    def fit_transform(self, series: Series, reference_df: Optional[DataFrame] = None) -> ImputationResult:
        self.fit(series, reference_df)
        return self._apply_imputation(series, reference_df)

    def _compute_fill_value(self, clean_series: Series) -> Any:
        if self.strategy == ImputationStrategy.MEAN:
            return clean_series.mean()
        elif self.strategy == ImputationStrategy.MEDIAN:
            return clean_series.median()
        elif self.strategy == ImputationStrategy.MODE:
            modes = clean_series.mode()
            return modes.iloc[0] if len(modes) > 0 else None
        elif self.strategy == ImputationStrategy.CONSTANT:
            return self.fill_value
        return None

    def _apply_imputation(self, series: Series, reference_df: Optional[DataFrame] = None) -> ImputationResult:
        missing_mask = series.isna()
        values_imputed = int(missing_mask.sum())
        indicator = pd.Series(missing_mask.astype(int), index=series.index) if self.add_indicator else None

        if self.strategy == ImputationStrategy.ERROR:
            if values_imputed > 0:
                raise ValueError("Identifier columns should not have missing values")
            return ImputationResult(
                series=series.copy(), strategy_used=self.strategy,
                values_imputed=0, indicator_column=indicator
            )

        if self.strategy == ImputationStrategy.DROP_ROW:
            return ImputationResult(
                series=series.copy(), strategy_used=self.strategy, values_imputed=0,
                rows_dropped=values_imputed, drop_mask=missing_mask.tolist(), indicator_column=indicator
            )

        result_series = series.copy()

        if self.strategy in [ImputationStrategy.MEAN, ImputationStrategy.MEDIAN, ImputationStrategy.MODE, ImputationStrategy.CONSTANT]:
            result_series = result_series.fillna(self._fitted_value)
            return ImputationResult(
                series=result_series, strategy_used=self.strategy, values_imputed=values_imputed,
                fill_value=self._fitted_value, indicator_column=indicator
            )

        if self.strategy == ImputationStrategy.FORWARD_FILL:
            result_series = result_series.ffill()
        elif self.strategy == ImputationStrategy.BACKWARD_FILL:
            result_series = result_series.bfill()
        elif self.strategy == ImputationStrategy.INTERPOLATE:
            result_series = result_series.interpolate(method='linear')
        elif self.strategy == ImputationStrategy.KNN:
            result_series = self._knn_impute(series, reference_df)

        return ImputationResult(
            series=result_series, strategy_used=self.strategy,
            values_imputed=values_imputed, indicator_column=indicator
        )

    def _knn_impute(self, series: Series, reference_df: Optional[DataFrame]) -> Series:
        if reference_df is None:
            return series.fillna(series.median())

        from sklearn.impute import KNNImputer
        col_name = series.name or "_target_col"
        df_copy = reference_df.select_dtypes(include=[np.number]).copy()

        if col_name in df_copy.columns:
            df_copy[col_name] = series.values
        else:
            df_copy.insert(0, col_name, series.values)

        imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        imputed = imputer.fit_transform(df_copy)
        imputed_df = pd.DataFrame(imputed, columns=df_copy.columns, index=df_copy.index)

        return imputed_df[col_name]
