from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from customer_retention.core.compat import (
    DataFrame,
    head_as_list,
    timedelta_to_days,
    to_pandas,
)

from .temporal_feature_analyzer import (
    MomentumResult,
    TemporalFeatureAnalyzer,
)


class SparkTemporalFeatureAnalyzer(TemporalFeatureAnalyzer):

    def _prepare_dataframe(self, df: DataFrame) -> DataFrame:
        df = to_pandas(df)
        return super()._prepare_dataframe(df)

    def _momentum_for_column(
        self, df: DataFrame, col: str, short_window: int, long_window: int, reference_date: Any,
    ) -> MomentumResult:
        entity_momentum = self._vectorized_entity_momentum(df, col, short_window, long_window)
        mean_mom = float(np.mean(entity_momentum)) if entity_momentum else 1.0
        std_mom = float(np.std(entity_momentum)) if entity_momentum else 0.0
        return MomentumResult(
            column=col, short_window=short_window, long_window=long_window,
            mean_momentum=mean_mom, std_momentum=std_mom,
            interpretation=self._classify_momentum(mean_mom),
        )

    def _vectorized_entity_momentum(
        self, df: DataFrame, col: str, short_w: int, long_w: int,
    ) -> List[float]:
        if len(df) == 0 or col not in df.columns:
            return []
        reference_date = df[self.time_column].max()
        df_calc = df[[self.entity_column, self.time_column, col]].copy()
        df_calc["_days_ago"] = timedelta_to_days(reference_date - df_calc[self.time_column])
        short_means = df_calc[df_calc["_days_ago"] <= short_w].groupby(self.entity_column)[col].mean()
        long_means = df_calc[df_calc["_days_ago"] <= long_w].groupby(self.entity_column)[col].mean()
        valid = (long_means > 0) & short_means.notna() & long_means.notna()
        momentum = (short_means[valid] / long_means[valid]).dropna()
        return head_as_list(momentum, 10000)

    def _velocity_accel_series(
        self, df: DataFrame, col: str, window: int,
    ) -> Tuple[List[float], List[float]]:
        if len(df) == 0 or col not in df.columns:
            return [], []
        return super()._velocity_accel_series(df, col, window)
