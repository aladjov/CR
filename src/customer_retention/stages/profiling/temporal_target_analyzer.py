from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from customer_retention.core.compat import DataFrame, Timestamp, _is_spark_pandas, as_spark_df, native_pd, to_datetime
from customer_retention.stages.profiling.stats_helpers import calculate_group_retention_stats


@dataclass
class TemporalTargetResult:
    datetime_col: str
    target_col: str
    min_date: Timestamp
    max_date: Timestamp
    n_valid_dates: int
    overall_rate: float

    yearly_stats: DataFrame
    yearly_trend: str

    monthly_stats: DataFrame
    best_month: Optional[str]
    worst_month: Optional[str]
    seasonal_spread: float

    dow_stats: DataFrame

    quarterly_stats: DataFrame


class TemporalTargetAnalyzer:

    MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self, min_samples_per_period: int = 10):
        self.min_samples_per_period = min_samples_per_period

    def analyze(
        self,
        df: DataFrame,
        datetime_col: str,
        target_col: str
    ) -> TemporalTargetResult:
        if len(df) == 0 or datetime_col not in df.columns or target_col not in df.columns:
            return self._empty_result(datetime_col, target_col)

        df_clean = df[[datetime_col, target_col]].copy()
        df_clean[datetime_col] = to_datetime(df_clean[datetime_col], errors='coerce')
        df_clean = df_clean.dropna()

        if len(df_clean) == 0:
            return self._empty_result(datetime_col, target_col)

        overall_rate = df_clean[target_col].mean()
        df_clean = self._extract_temporal_components(df_clean, datetime_col)

        # Calculate stats by time period
        yearly_stats = self._calculate_period_stats(df_clean, '_year', target_col, overall_rate)
        monthly_stats = self._calculate_monthly_stats(df_clean, target_col, overall_rate)
        quarterly_stats = self._calculate_period_stats(df_clean, '_quarter', target_col, overall_rate)
        dow_stats = self._calculate_dow_stats(df_clean, target_col, overall_rate)

        # Determine yearly trend
        yearly_trend = self._determine_yearly_trend(yearly_stats)

        # Find best/worst months
        best_month, worst_month, seasonal_spread = self._find_seasonal_extremes(monthly_stats)

        return TemporalTargetResult(
            datetime_col=datetime_col,
            target_col=target_col,
            min_date=df_clean[datetime_col].min(),
            max_date=df_clean[datetime_col].max(),
            n_valid_dates=len(df_clean),
            overall_rate=overall_rate,
            yearly_stats=yearly_stats,
            yearly_trend=yearly_trend,
            monthly_stats=monthly_stats,
            best_month=best_month,
            worst_month=worst_month,
            seasonal_spread=seasonal_spread,
            dow_stats=dow_stats,
            quarterly_stats=quarterly_stats
        )

    @staticmethod
    def _extract_temporal_components(df_clean: DataFrame, datetime_col: str) -> DataFrame:
        if _is_spark_pandas(df_clean):
            import pyspark.sql.functions as F  # noqa: N812
            spark_df = as_spark_df(df_clean)
            ts = F.col(datetime_col)
            spark_df = spark_df.withColumns({
                "_year": F.year(ts),
                "_month": F.month(ts),
                "_quarter": F.quarter(ts),
                "_dow": F.dayofweek(ts) - 1,
            })
            from customer_retention.core.compat.spark_backend import _as_pandas_api
            return _as_pandas_api(spark_df)
        return df_clean.assign(
            _year=df_clean[datetime_col].dt.year,
            _month=df_clean[datetime_col].dt.month,
            _quarter=df_clean[datetime_col].dt.quarter,
            _dow=df_clean[datetime_col].dt.dayofweek,
        )

    def _calculate_period_stats(
        self,
        df: DataFrame,
        period_col: str,
        target_col: str,
        overall_rate: float
    ) -> DataFrame:
        stats = calculate_group_retention_stats(
            df, period_col, target_col, overall_rate,
            min_samples=self.min_samples_per_period, sort_by="group",
        )
        return stats.rename(columns={"group": "period"})

    def _calculate_monthly_stats(
        self,
        df: DataFrame,
        target_col: str,
        overall_rate: float
    ) -> DataFrame:
        stats = calculate_group_retention_stats(
            df, "_month", target_col, overall_rate,
            min_samples=self.min_samples_per_period, sort_by="group",
        )
        stats = stats.rename(columns={"group": "month"})
        stats['month_name'] = stats['month'].apply(
            lambda x: self.MONTH_NAMES[int(x) - 1] if 1 <= x <= 12 else 'Unknown'
        )
        return stats

    def _calculate_dow_stats(
        self,
        df: DataFrame,
        target_col: str,
        overall_rate: float
    ) -> DataFrame:
        stats = calculate_group_retention_stats(
            df, "_dow", target_col, overall_rate, sort_by="group",
        )
        stats = stats.rename(columns={"group": "day_of_week"})
        stats['day_name'] = stats['day_of_week'].apply(
            lambda x: self.DOW_NAMES[int(x)] if 0 <= x <= 6 else 'Unknown'
        )
        return stats

    def _determine_yearly_trend(self, yearly_stats: DataFrame) -> str:
        if len(yearly_stats) < 2:
            return 'stable'

        rates = yearly_stats['retention_rate'].to_numpy()

        if len(rates) >= 2:
            slope = np.polyfit(range(len(rates)), rates, 1)[0]

            if slope > 0.02:  # More than 2% improvement per year
                return 'improving'
            elif slope < -0.02:  # More than 2% decline per year
                return 'declining'

        return 'stable'

    @staticmethod
    def _find_seasonal_extremes(monthly_stats: DataFrame) -> tuple:
        if len(monthly_stats) == 0:
            return None, None, 0.0

        rates = monthly_stats['retention_rate'].to_numpy()
        names = monthly_stats['month_name'].to_numpy()
        best_pos = int(rates.argmax())
        worst_pos = int(rates.argmin())
        return str(names[best_pos]), str(names[worst_pos]), float(rates[best_pos] - rates[worst_pos])

    def _empty_result(self, datetime_col: str, target_col: str) -> TemporalTargetResult:
        empty_df = native_pd.DataFrame()

        return TemporalTargetResult(
            datetime_col=datetime_col,
            target_col=target_col,
            min_date=native_pd.NaT,
            max_date=native_pd.NaT,
            n_valid_dates=0,
            overall_rate=0.0,
            yearly_stats=empty_df,
            yearly_trend='stable',
            monthly_stats=empty_df,
            best_month=None,
            worst_month=None,
            seasonal_spread=0.0,
            dow_stats=empty_df,
            quarterly_stats=empty_df
        )

    def analyze_multiple(
        self,
        df: DataFrame,
        datetime_cols: List[str],
        target_col: str
    ) -> native_pd.DataFrame:
        results = []
        for col in datetime_cols:
            result = self.analyze(df, col, target_col)
            results.append({
                'feature': col,
                'n_valid': result.n_valid_dates,
                'yearly_trend': result.yearly_trend,
                'best_month': result.best_month,
                'worst_month': result.worst_month,
                'seasonal_spread': result.seasonal_spread
            })

        return native_pd.DataFrame(results)
