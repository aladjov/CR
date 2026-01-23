import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from customer_retention.stages.temporal.timestamp_discovery import DatetimeOrderAnalyzer


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    score_df: pd.DataFrame
    unresolvable_df: pd.DataFrame
    cutoff_date: datetime
    timestamp_source: str
    train_count: int
    score_count: int
    unresolvable_count: int
    original_count: int


@dataclass
class CutoffAnalysis:
    timestamp_column: str
    total_rows: int
    bins: list[datetime]
    bin_counts: list[int]
    train_percentages: list[float]
    score_percentages: list[float]
    date_range: tuple[datetime, datetime]
    source_rows: int = 0
    covered_rows: int = 0
    resolved_timestamp_series: Optional[pd.Series] = field(default=None, repr=False)
    _source_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def coverage_ratio(self) -> float:
        return self.covered_rows / self.source_rows if self.source_rows > 0 else 0.0

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

    def split_at_cutoff(self, cutoff_date: Optional[datetime] = None) -> "SplitResult":
        if self.resolved_timestamp_series is None:
            raise ValueError(
                "No resolved timestamp series available. "
                "Re-run analyze() to populate resolved_timestamp_series."
            )
        if self._source_df is None:
            raise ValueError(
                "No source DataFrame available. "
                "Re-run analyze() to populate the source reference."
            )

        cutoff = cutoff_date or self.suggest_cutoff()
        ts = self.resolved_timestamp_series
        df = self._source_df

        not_null_mask = ts.notna()
        train_mask = not_null_mask & (ts <= cutoff)
        score_mask = not_null_mask & (ts > cutoff)
        unresolvable_mask = ~not_null_mask

        train_df = df.loc[train_mask]
        score_df = df.loc[score_mask]
        unresolvable_df = df.loc[unresolvable_mask]

        assert len(train_df) + len(score_df) + len(unresolvable_df) == len(df), (
            f"Data loss detected: train({len(train_df)}) + score({len(score_df)}) + "
            f"unresolvable({len(unresolvable_df)}) != original({len(df)})"
        )

        return SplitResult(
            train_df=train_df,
            score_df=score_df,
            unresolvable_df=unresolvable_df,
            cutoff_date=cutoff,
            timestamp_source=self.timestamp_column,
            train_count=len(train_df),
            score_count=len(score_df),
            unresolvable_count=len(unresolvable_df),
            original_count=len(df),
        )

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
        self,
        df: pd.DataFrame,
        timestamp_column: Optional[str] = None,
        n_bins: int = 20,
        timestamp_series: Optional[pd.Series] = None,
    ) -> CutoffAnalysis:
        source_rows = len(df)
        ts_col, full_series = self._resolve_timestamp_series_full(df, timestamp_column, timestamp_series)
        ts_series = full_series.dropna()

        if len(ts_series) == 0:
            return self._empty_analysis(ts_col, source_rows=source_rows, df=df, full_series=full_series)

        covered_rows = len(ts_series)
        coverage_ratio = covered_rows / source_rows if source_rows > 0 else 0.0
        if coverage_ratio < 0.5:
            warnings.warn(
                f"Low timestamp coverage: {covered_rows}/{source_rows} rows "
                f"({coverage_ratio:.1%}). Results may not represent the full dataset.",
                stacklevel=2,
            )

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
            source_rows=source_rows,
            covered_rows=covered_rows,
            resolved_timestamp_series=full_series,
            _source_df=df,
        )

    def _resolve_timestamp_series_full(
        self,
        df: pd.DataFrame,
        timestamp_column: Optional[str],
        timestamp_series: Optional[pd.Series],
    ) -> tuple[str, pd.Series]:
        if timestamp_series is not None:
            ts_col = timestamp_series.name or "timestamp_series"
            series = self._ensure_datetime_series_full(timestamp_series)
            return ts_col, series
        ts_col = timestamp_column or self._detect_timestamp_column(df)
        series = self._ensure_datetime_series_full(df[ts_col])
        return ts_col, series

    def _detect_timestamp_column(self, df: pd.DataFrame) -> str:
        datetime_cols = self._datetime_analyzer._get_datetime_columns(df)
        for pattern in self.TIMESTAMP_PATTERNS:
            for col in datetime_cols:
                if pattern in col.lower():
                    return col
        if datetime_cols:
            return datetime_cols[0]
        raise ValueError("No timestamp column found")

    def _ensure_datetime_series_full(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        return pd.to_datetime(series, format="mixed", errors="coerce")

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

    def _empty_analysis(self, ts_col: str, source_rows: int = 0, df: Optional[pd.DataFrame] = None, full_series: Optional[pd.Series] = None) -> CutoffAnalysis:
        return CutoffAnalysis(
            timestamp_column=ts_col, total_rows=0, bins=[], bin_counts=[],
            train_percentages=[], score_percentages=[],
            date_range=(datetime.now(), datetime.now()),
            source_rows=source_rows, covered_rows=0,
            resolved_timestamp_series=full_series,
            _source_df=df,
        )
