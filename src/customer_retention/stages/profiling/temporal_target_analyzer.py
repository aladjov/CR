"""Temporal feature analysis with respect to a binary target."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from customer_retention.core.compat import DataFrame, to_pandas


@dataclass
class TemporalTargetResult:
    """Results from temporal-target analysis."""
    datetime_col: str
    target_col: str
    min_date: pd.Timestamp
    max_date: pd.Timestamp
    n_valid_dates: int
    overall_rate: float

    # Yearly analysis
    yearly_stats: pd.DataFrame  # year, count, retention_rate, lift
    yearly_trend: str  # 'improving', 'declining', 'stable'

    # Monthly analysis (seasonality)
    monthly_stats: pd.DataFrame  # month, month_name, count, retention_rate, lift
    best_month: Optional[str]
    worst_month: Optional[str]
    seasonal_spread: float  # difference between best and worst

    # Day of week analysis
    dow_stats: pd.DataFrame  # day_of_week, day_name, count, retention_rate, lift

    # Quarterly analysis
    quarterly_stats: pd.DataFrame


class TemporalTargetAnalyzer:
    """Analyzes relationship between datetime features and binary target.

    Computes retention rates by:
    - Year (cohort analysis)
    - Month (seasonality)
    - Day of week (weekly patterns)
    - Quarter
    """

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
        """Analyze relationship between datetime feature and binary target."""
        df = to_pandas(df)

        if len(df) == 0 or datetime_col not in df.columns or target_col not in df.columns:
            return self._empty_result(datetime_col, target_col)

        # Parse dates and prepare data
        df_clean = df[[datetime_col, target_col]].copy()
        df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col], errors='coerce')
        df_clean = df_clean.dropna()

        if len(df_clean) == 0:
            return self._empty_result(datetime_col, target_col)

        # Calculate overall retention rate
        overall_rate = df_clean[target_col].mean()

        # Extract temporal components
        df_clean['_year'] = df_clean[datetime_col].dt.year
        df_clean['_month'] = df_clean[datetime_col].dt.month
        df_clean['_quarter'] = df_clean[datetime_col].dt.quarter
        df_clean['_dow'] = df_clean[datetime_col].dt.dayofweek

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

    def _calculate_period_stats(
        self,
        df: pd.DataFrame,
        period_col: str,
        target_col: str,
        overall_rate: float
    ) -> pd.DataFrame:
        """Calculate retention stats for a time period."""
        stats = df.groupby(period_col)[target_col].agg(['sum', 'count', 'mean']).reset_index()
        stats.columns = ['period', 'retained_count', 'count', 'retention_rate']
        stats['lift'] = stats['retention_rate'] / overall_rate if overall_rate > 0 else 0

        # Filter small samples
        stats = stats[stats['count'] >= self.min_samples_per_period]

        return stats.sort_values('period').reset_index(drop=True)

    def _calculate_monthly_stats(
        self,
        df: pd.DataFrame,
        target_col: str,
        overall_rate: float
    ) -> pd.DataFrame:
        """Calculate monthly retention stats with month names."""
        stats = df.groupby('_month')[target_col].agg(['sum', 'count', 'mean']).reset_index()
        stats.columns = ['month', 'retained_count', 'count', 'retention_rate']
        stats['lift'] = stats['retention_rate'] / overall_rate if overall_rate > 0 else 0
        stats['month_name'] = stats['month'].apply(
            lambda x: self.MONTH_NAMES[int(x) - 1] if 1 <= x <= 12 else 'Unknown'
        )

        # Filter small samples
        stats = stats[stats['count'] >= self.min_samples_per_period]

        return stats.sort_values('month').reset_index(drop=True)

    def _calculate_dow_stats(
        self,
        df: pd.DataFrame,
        target_col: str,
        overall_rate: float
    ) -> pd.DataFrame:
        """Calculate day-of-week retention stats."""
        stats = df.groupby('_dow')[target_col].agg(['sum', 'count', 'mean']).reset_index()
        stats.columns = ['day_of_week', 'retained_count', 'count', 'retention_rate']
        stats['lift'] = stats['retention_rate'] / overall_rate if overall_rate > 0 else 0
        stats['day_name'] = stats['day_of_week'].apply(
            lambda x: self.DOW_NAMES[int(x)] if 0 <= x <= 6 else 'Unknown'
        )

        return stats.sort_values('day_of_week').reset_index(drop=True)

    def _determine_yearly_trend(self, yearly_stats: pd.DataFrame) -> str:
        """Determine if retention is improving, declining, or stable over years."""
        if len(yearly_stats) < 2:
            return 'stable'

        rates = yearly_stats['retention_rate'].values
        yearly_stats['period'].values

        # Simple linear regression
        if len(rates) >= 2:
            slope = np.polyfit(range(len(rates)), rates, 1)[0]

            if slope > 0.02:  # More than 2% improvement per year
                return 'improving'
            elif slope < -0.02:  # More than 2% decline per year
                return 'declining'

        return 'stable'

    def _find_seasonal_extremes(
        self,
        monthly_stats: pd.DataFrame
    ) -> tuple:
        """Find best and worst months for retention."""
        if len(monthly_stats) == 0:
            return None, None, 0.0

        best_idx = monthly_stats['retention_rate'].idxmax()
        worst_idx = monthly_stats['retention_rate'].idxmin()

        best_month = monthly_stats.loc[best_idx, 'month_name']
        worst_month = monthly_stats.loc[worst_idx, 'month_name']
        spread = monthly_stats.loc[best_idx, 'retention_rate'] - monthly_stats.loc[worst_idx, 'retention_rate']

        return best_month, worst_month, float(spread)

    def _empty_result(self, datetime_col: str, target_col: str) -> TemporalTargetResult:
        """Return empty result for edge cases."""
        empty_df = pd.DataFrame()

        return TemporalTargetResult(
            datetime_col=datetime_col,
            target_col=target_col,
            min_date=pd.NaT,
            max_date=pd.NaT,
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
    ) -> pd.DataFrame:
        """Analyze multiple datetime columns and return summary."""
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

        return pd.DataFrame(results)
