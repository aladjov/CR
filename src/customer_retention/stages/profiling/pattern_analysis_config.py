from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from customer_retention.core.compat import DataFrame


@dataclass
class PatternAnalysisConfig:
    entity_column: str
    time_column: str
    target_column: Optional[str] = None
    aggregation_windows: List[str] = field(default_factory=list)
    velocity_window_days: int = 7
    short_momentum_window: int = 7
    long_momentum_window: int = 30
    rolling_window: int = 7
    sparkline_columns: List[str] = field(default_factory=list)
    sparkline_freq: str = "W"
    sparkline_agg: str = "mean"
    has_target: bool = False
    is_event_level: bool = True

    @classmethod
    def from_findings(cls, findings: Any, target_column: Optional[str] = None,
                      window_override: Optional[List[str]] = None) -> "PatternAnalysisConfig":
        ts_meta = findings.time_series_metadata
        if ts_meta is None:
            raise ValueError("Findings do not contain time series metadata. Run notebook 01a first.")

        windows = (window_override or ts_meta.suggested_aggregations or ["7d", "30d", "90d"])
        target_col = target_column or findings.target_column

        config = cls(
            entity_column=ts_meta.entity_column, time_column=ts_meta.time_column,
            target_column=target_col, aggregation_windows=windows,
            has_target=target_col is not None, is_event_level=True)
        config._derive_window_settings()
        return config

    def _derive_window_settings(self):
        if not self.aggregation_windows:
            return
        window_days = sorted([d for w in self.aggregation_windows if (d := self._parse_window_to_days(w))])
        if not window_days:
            return
        self.velocity_window_days = self.rolling_window = self.short_momentum_window = window_days[0]
        self.long_momentum_window = window_days[1] if len(window_days) >= 2 else window_days[0] * 4

    def _parse_window_to_days(self, window: str) -> Optional[int]:
        if not window:
            return None
        w = window.lower().strip()
        multipliers = {"d": 1, "w": 7, "m": 30}
        for suffix, mult in multipliers.items():
            if w.endswith(suffix):
                try:
                    return int(w[:-1]) * mult
                except ValueError:
                    return None
        try:
            return int(w)
        except ValueError:
            return None

    def get_momentum_pairs(self) -> List[Tuple[int, int]]:
        if len(self.aggregation_windows) < 2:
            return [(self.short_momentum_window, self.long_momentum_window)]
        window_days = sorted({d for w in self.aggregation_windows if (d := self._parse_window_to_days(w))})
        pairs = [(window_days[i], window_days[i + 1]) for i in range(len(window_days) - 1)]
        return pairs if pairs else [(self.short_momentum_window, self.long_momentum_window)]

    def print_config(self):
        print("=" * 70)
        print("PATTERN ANALYSIS CONFIGURATION")
        print("=" * 70)
        print(f"\nCore Columns:\n   Entity: {self.entity_column}\n   Time:   {self.time_column}")
        print(f"   Target: {self.target_column or '(none)'}")
        print(f"\nAggregation Windows (from findings):\n   {self.aggregation_windows}")
        print(f"\nDerived Settings:\n   Velocity window:  {self.velocity_window_days} days")
        print(f"   Rolling window:   {self.rolling_window} days\n   Momentum pairs:   {self.get_momentum_pairs()}")
        if self.sparkline_columns:
            print(f"\nSparkline Config:\n   Columns: {self.sparkline_columns}")
            print(f"   Frequency: {self.sparkline_freq}\n   Aggregation: {self.sparkline_agg}")

    def configure_sparklines(self, df: DataFrame, columns: Optional[List[str]] = None, max_columns: int = 5):
        if columns:
            self.sparkline_columns = columns[:max_columns]
            return
        import numpy as np
        exclude = {self.entity_column, self.time_column, self.target_column} - {None}
        candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        self.sparkline_columns = candidates[:max_columns]


@dataclass
class PatternAnalysisResult:
    trend_detected: bool = False
    trend_direction: Optional[str] = None
    trend_strength: float = 0.0
    seasonality_detected: bool = False
    seasonality_periods: List[str] = field(default_factory=list)
    recency_effect: bool = False
    recency_correlation: float = 0.0
    cohort_effect: bool = False
    cohort_trend: Optional[str] = None
    velocity_features_recommended: List[str] = field(default_factory=list)
    momentum_features_recommended: List[str] = field(default_factory=list)

    def print_summary(self):
        print("\n" + "=" * 70 + "\nPATTERN ANALYSIS SUMMARY\n" + "=" * 70)
        patterns = []
        if self.trend_detected:
            patterns.append(f"Trend: {self.trend_direction} (strength: {self.trend_strength:.2f})")
        if self.seasonality_detected:
            patterns.append(f"Seasonality: {', '.join(self.seasonality_periods)}")
        if self.recency_effect:
            patterns.append(f"Recency effect: r={self.recency_correlation:.2f}")
        if self.cohort_effect:
            patterns.append(f"Cohort effect: {self.cohort_trend}")
        if patterns:
            print("\nDetected Patterns:")
            for p in patterns:
                print(f"   - {p}")
        else:
            print("\n   No significant patterns detected")
        if self.velocity_features_recommended:
            print(f"\nRecommended velocity features: {self.velocity_features_recommended}")
        if self.momentum_features_recommended:
            print(f"Recommended momentum features: {self.momentum_features_recommended}")


def get_sparkline_frequency(time_span_days: int) -> str:
    if time_span_days <= 60:
        return "D"
    return "W" if time_span_days <= 365 else "ME"


def get_analysis_frequency(time_span_days: int) -> Tuple[str, str]:
    if time_span_days <= 90:
        return "D", "Daily"
    return ("W", "Weekly") if time_span_days <= 365 else ("ME", "Monthly")


@dataclass
class SparklineData:
    column: str
    weeks: List
    retained_values: List[float]
    churned_values: Optional[List[float]] = None
    has_target_split: bool = False

    @property
    def divergence_score(self) -> float:
        if not self.has_target_split or self.churned_values is None:
            return 0.0
        import numpy as np
        ret_arr = np.array([v for v in self.retained_values if v is not None and not np.isnan(v)])
        churn_arr = np.array([v for v in self.churned_values if v is not None and not np.isnan(v)])
        if len(ret_arr) == 0 or len(churn_arr) == 0:
            return 0.0
        return abs(ret_arr.mean() - churn_arr.mean()) / max(ret_arr.std(), churn_arr.std(), 0.001)


class SparklineDataBuilder:
    def __init__(self, entity_column: str, time_column: str,
                 target_column: Optional[str] = None, freq: str = "W"):
        self.entity_column = entity_column
        self.time_column = time_column
        self.target_column = target_column
        self.freq = freq

    def build(self, df: DataFrame, columns: List[str]) -> Tuple[List[SparklineData], bool]:
        import numpy as np
        import pandas as pd

        if self.target_column and self.target_column in df.columns:
            entity_target = df.groupby(self.entity_column)[self.target_column].first()
            df_work = df.merge(
                entity_target.reset_index().rename(columns={self.target_column: '_target'}),
                on=self.entity_column)
            has_target_split = True
        else:
            df_work = df.copy()
            df_work['_target'] = 1
            has_target_split = False

        df_work['_period'] = pd.to_datetime(df_work[self.time_column]).dt.to_period(self.freq).dt.start_time
        results = []
        for col in columns:
            if col not in df_work.columns:
                continue
            if has_target_split:
                retained = df_work[df_work['_target'] == 1].groupby('_period')[col].mean()
                churned = df_work[df_work['_target'] == 0].groupby('_period')[col].mean()
                all_periods = sorted(set(retained.index) | set(churned.index))
                retained_vals = [retained.get(p, np.nan) for p in all_periods]
                churned_vals = [churned.get(p, np.nan) for p in all_periods]
            else:
                overall = df_work.groupby('_period')[col].mean()
                all_periods = sorted(overall.index)
                retained_vals = overall.tolist()
                churned_vals = None
            results.append(SparklineData(
                column=col, weeks=all_periods, retained_values=retained_vals,
                churned_values=churned_vals, has_target_split=has_target_split))
        return results, has_target_split

    def print_summary(self, sparkline_data: List[SparklineData], has_target: bool):
        print("=" * 70)
        if has_target:
            print("SPARKLINE COMPARISON: Retained vs Churned Trends\n" + "=" * 70)
            print("\n  Retained (target=1) | Churned (target=0)\n")
        else:
            print("SPARKLINE TRENDS: Overall Patterns\n" + "=" * 70)
        for data in sparkline_data:
            if data.has_target_split:
                print(f"  {data.column}: divergence={data.divergence_score:.2f}")
