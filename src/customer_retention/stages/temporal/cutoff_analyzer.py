from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from customer_retention.stages.temporal.timestamp_discovery import DatetimeOrderAnalyzer


@dataclass
class CutoffAnalysis:
    timestamp_column: str
    total_rows: int
    bins: list[datetime]
    bin_counts: list[int]
    train_percentages: list[float]
    score_percentages: list[float]
    date_range: tuple[datetime, datetime]

    def suggest_cutoff(self, train_ratio: float = 0.9) -> datetime:
        target_pct = train_ratio * 100
        for i, train_pct in enumerate(self.train_percentages):
            if train_pct >= target_pct:
                return self.bins[i]
        return self.bins[-1]

    def get_train_percentage(self, cutoff_date: datetime) -> float:
        for i, bin_date in enumerate(self.bins):
            if bin_date >= cutoff_date:
                return self.train_percentages[max(0, i - 1)]
        return self.train_percentages[-1]

    def get_split_at_date(self, cutoff_date: datetime) -> dict:
        train_pct = self.get_train_percentage(cutoff_date)
        train_count = int(self.total_rows * train_pct / 100)
        return {
            "train_count": train_count,
            "score_count": self.total_rows - train_count,
            "train_pct": train_pct,
            "score_pct": 100 - train_pct,
        }

    def to_dataframe(self) -> pd.DataFrame:
        cumulative = np.cumsum(self.bin_counts)
        return pd.DataFrame({
            "date": self.bins,
            "bin_count": self.bin_counts,
            "cumulative_count": cumulative,
            "train_pct": self.train_percentages,
            "score_pct": self.score_percentages,
        })

    def get_percentage_milestones(self, step: int = 5) -> list[dict]:
        milestones = []
        target_pcts = list(range(step, 100, step))
        for target in target_pcts:
            for i, train_pct in enumerate(self.train_percentages):
                if train_pct >= target:
                    milestones.append({
                        "date": self.bins[i],
                        "train_pct": round(train_pct, 1),
                        "score_pct": round(100 - train_pct, 1),
                    })
                    break
        return milestones


class CutoffAnalyzer:
    TIMESTAMP_PATTERNS = ["feature_timestamp", "label_timestamp", "timestamp", "date", "datetime"]

    def __init__(self):
        self._datetime_analyzer = DatetimeOrderAnalyzer()

    def analyze(
        self, df: pd.DataFrame, timestamp_column: Optional[str] = None, n_bins: int = 20
    ) -> CutoffAnalysis:
        ts_col = timestamp_column or self._detect_timestamp_column(df)
        ts_series = self._ensure_datetime_series(df[ts_col])

        if len(ts_series) == 0:
            return self._empty_analysis(ts_col)

        bins, counts = self._compute_bins(ts_series, n_bins)
        train_pcts, score_pcts = self._compute_percentages(counts)

        return CutoffAnalysis(
            timestamp_column=ts_col,
            total_rows=len(ts_series),
            bins=bins,
            bin_counts=counts,
            train_percentages=train_pcts,
            score_percentages=score_pcts,
            date_range=(ts_series.min(), ts_series.max()),
        )

    def _detect_timestamp_column(self, df: pd.DataFrame) -> str:
        datetime_cols = self._datetime_analyzer._get_datetime_columns(df)
        for pattern in self.TIMESTAMP_PATTERNS:
            for col in datetime_cols:
                if pattern in col.lower():
                    return col
        if datetime_cols:
            return datetime_cols[0]
        raise ValueError("No timestamp column found")

    def _ensure_datetime_series(self, series: pd.Series) -> pd.Series:
        series = series.dropna()
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        return pd.to_datetime(series, format="mixed", errors="coerce").dropna()

    def _compute_bins(self, ts_series: pd.Series, n_bins: int) -> tuple[list[datetime], list[int]]:
        if ts_series.nunique() == 1:
            return [ts_series.iloc[0].to_pydatetime()], [len(ts_series)]

        bin_edges = pd.date_range(ts_series.min(), ts_series.max(), periods=n_bins + 1)
        counts, _ = np.histogram(ts_series, bins=bin_edges)
        bin_centers = [edge.to_pydatetime() for edge in bin_edges[:-1]]
        return bin_centers, counts.tolist()

    def _compute_percentages(self, counts: list[int]) -> tuple[list[float], list[float]]:
        total = sum(counts)
        if total == 0:
            return [0.0] * len(counts), [100.0] * len(counts)

        cumulative = np.cumsum(counts)
        train_pcts = (cumulative / total * 100).tolist()
        score_pcts = [100 - p for p in train_pcts]
        return train_pcts, score_pcts

    def _empty_analysis(self, ts_col: str) -> CutoffAnalysis:
        return CutoffAnalysis(
            timestamp_column=ts_col, total_rows=0, bins=[], bin_counts=[],
            train_percentages=[], score_percentages=[],
            date_range=(datetime.now(), datetime.now()),
        )
