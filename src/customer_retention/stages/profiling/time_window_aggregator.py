"""
Time window aggregator for event-level data.

Aggregates time series data into entity-level features using configurable
time windows (24h, 7d, 30d, 90d, all_time).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from customer_retention.core.compat import DataFrame, pd


class AggregationType(str, Enum):
    """Available aggregation types."""
    # Numeric aggregations
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    COUNT = "count"
    STD = "std"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"
    # Categorical aggregations
    MODE = "mode"
    NUNIQUE = "nunique"
    MODE_RATIO = "mode_ratio"
    ENTROPY = "entropy"
    VALUE_COUNTS = "value_counts"


CATEGORICAL_AGG_FUNCS = {"mode", "nunique", "mode_ratio", "entropy", "value_counts"}
NUMERIC_AGG_FUNCS = {"sum", "mean", "max", "min", "count", "std", "median", "first", "last"}


@dataclass
class TimeWindow:
    """Definition of a time window."""
    name: str
    days: Optional[int]  # None means all_time

    @classmethod
    def from_string(cls, window_str: str) -> "TimeWindow":
        """Parse window string like '7d', '24h', 'all_time'."""
        window_str = window_str.lower().strip()

        if window_str == "all_time":
            return cls(name="all_time", days=None)

        if window_str.endswith("d"):
            days = int(window_str[:-1])
            return cls(name=window_str, days=days)

        if window_str.endswith("h"):
            hours = int(window_str[:-1])
            days = hours / 24
            return cls(name=window_str, days=days)

        if window_str.endswith("w"):
            weeks = int(window_str[:-1])
            return cls(name=window_str, days=weeks * 7)

        raise ValueError(f"Unknown window format: {window_str}")


@dataclass
class AggregationPlan:
    """Plan for aggregation showing all features to be created."""
    entity_column: str
    time_column: str
    windows: List[TimeWindow]
    value_columns: List[str]
    agg_funcs: List[str]
    feature_columns: List[str] = field(default_factory=list)
    include_event_count: bool = True
    include_recency: bool = False
    include_tenure: bool = False
    value_counts_categories: Dict[str, List[str]] = field(default_factory=dict)


class TimeWindowAggregator:
    """Aggregates time series data into entity-level features."""

    def __init__(self, entity_column: str, time_column: str):
        self.entity_column = entity_column
        self.time_column = time_column

    def aggregate(
        self,
        df: DataFrame,
        windows: Optional[List[str]] = None,
        value_columns: Optional[List[str]] = None,
        agg_funcs: Optional[List[str]] = None,
        reference_date: Optional[pd.Timestamp] = None,
        include_event_count: bool = False,
        include_recency: bool = False,
        include_tenure: bool = False,
    ) -> DataFrame:
        """
        Aggregate event data into entity-level features.

        Args:
            df: Event-level dataframe
            windows: List of time windows (e.g., ["7d", "30d", "all_time"])
            value_columns: Columns to aggregate
            agg_funcs: Aggregation functions (sum, mean, max, count, etc.)
            reference_date: Reference date for window calculations (default: max date in data)
            include_event_count: Add event_count_{window} columns
            include_recency: Add days_since_last_event column
            include_tenure: Add days_since_first_event column

        Returns:
            Entity-level dataframe with aggregated features
        """
        if len(df) == 0:
            return pd.DataFrame()

        # Parse time column if needed
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        # Set reference date
        if reference_date is None:
            reference_date = df[self.time_column].max()

        # Default windows
        if windows is None:
            windows = ["30d"]

        # Parse windows
        parsed_windows = [TimeWindow.from_string(w) for w in windows]

        # Get unique entities
        entities = df[self.entity_column].unique()
        result_data = {self.entity_column: entities}

        # Add event counts per window
        if include_event_count or (value_columns is None and agg_funcs is None):
            for window in parsed_windows:
                col_name = f"event_count_{window.name}"
                result_data[col_name] = self._compute_event_counts(
                    df, entities, window, reference_date
                )

        # Add value aggregations
        if value_columns and agg_funcs:
            for window in parsed_windows:
                for col in value_columns:
                    for func in agg_funcs:
                        if func == "value_counts":
                            value_counts_cols = self._compute_value_counts(
                                df, entities, col, window, reference_date
                            )
                            result_data.update(value_counts_cols)
                        else:
                            col_name = f"{col}_{func}_{window.name}"
                            result_data[col_name] = self._compute_aggregation(
                                df, entities, col, func, window, reference_date
                            )

        # Add recency features
        if include_recency:
            result_data["days_since_last_event"] = self._compute_recency(
                df, entities, reference_date
            )

        # Add tenure features
        if include_tenure:
            result_data["days_since_first_event"] = self._compute_tenure(
                df, entities, reference_date
            )

        return pd.DataFrame(result_data)

    def generate_plan(
        self, df: DataFrame, windows: List[str], value_columns: List[str],
        agg_funcs: List[str], include_event_count: bool = True,
        include_recency: bool = False, include_tenure: bool = False,
    ) -> AggregationPlan:
        parsed_windows = [TimeWindow.from_string(w) for w in windows]
        feature_columns = []
        value_counts_categories: Dict[str, List[str]] = {}

        if include_event_count:
            for window in parsed_windows:
                feature_columns.append(f"event_count_{window.name}")

        for window in parsed_windows:
            for col in value_columns:
                for func in agg_funcs:
                    if func == "value_counts":
                        unique_vals = list(df[col].dropna().unique())
                        value_counts_categories[col] = unique_vals
                        for val in unique_vals:
                            feature_columns.append(f"{col}_{val}_count_{window.name}")
                    else:
                        feature_columns.append(f"{col}_{func}_{window.name}")

        if include_recency:
            feature_columns.append("days_since_last_event")
        if include_tenure:
            feature_columns.append("days_since_first_event")

        return AggregationPlan(
            entity_column=self.entity_column,
            time_column=self.time_column,
            windows=parsed_windows,
            value_columns=value_columns,
            agg_funcs=agg_funcs,
            feature_columns=feature_columns,
            include_event_count=include_event_count,
            include_recency=include_recency,
            include_tenure=include_tenure,
            value_counts_categories=value_counts_categories,
        )

    def _compute_event_counts(
        self,
        df: DataFrame,
        entities: np.ndarray,
        window: TimeWindow,
        reference_date: pd.Timestamp,
    ) -> np.ndarray:
        """Compute event counts per entity within window."""
        if window.days is None:
            # All time
            counts = df.groupby(self.entity_column).size()
        else:
            cutoff = reference_date - pd.Timedelta(days=window.days)
            mask = df[self.time_column] >= cutoff
            counts = df[mask].groupby(self.entity_column).size()

        return np.array([counts.get(e, 0) for e in entities])

    def _filter_by_window(self, df: DataFrame, window: TimeWindow, reference_date: pd.Timestamp) -> DataFrame:
        if window.days is None:
            return df
        cutoff = reference_date - pd.Timedelta(days=window.days)
        return df[df[self.time_column] >= cutoff]

    def _compute_aggregation(
        self, df: DataFrame, entities: np.ndarray, value_column: str,
        agg_func: str, window: TimeWindow, reference_date: pd.Timestamp,
    ) -> np.ndarray:
        filtered_df = self._filter_by_window(df, window, reference_date)
        if len(filtered_df) == 0:
            default = 0 if agg_func in ["sum", "count", "nunique"] else np.nan
            return np.full(len(entities), default)

        is_numeric = pd.api.types.is_numeric_dtype(df[value_column])
        if agg_func in CATEGORICAL_AGG_FUNCS:
            return self._compute_categorical_agg(filtered_df, entities, value_column, agg_func)
        elif agg_func in NUMERIC_AGG_FUNCS and not is_numeric:
            return np.full(len(entities), np.nan)
        return self._compute_numeric_agg(filtered_df, entities, value_column, agg_func)

    def _compute_numeric_agg(self, filtered_df: DataFrame, entities: np.ndarray, value_column: str, agg_func: str) -> np.ndarray:
        if agg_func == "count":
            agg_result = filtered_df.groupby(self.entity_column)[value_column].count()
        else:
            agg_result = filtered_df.groupby(self.entity_column)[value_column].agg(agg_func)
        default = 0 if agg_func in ["sum", "count"] else np.nan
        return np.array([agg_result.get(e, default) for e in entities])

    def _compute_categorical_agg(self, filtered_df: DataFrame, entities: np.ndarray, value_column: str, agg_func: str) -> np.ndarray:
        if agg_func == "mode":
            return self._agg_mode(filtered_df, entities, value_column)
        elif agg_func == "nunique":
            return self._agg_nunique(filtered_df, entities, value_column)
        elif agg_func == "mode_ratio":
            return self._agg_mode_ratio(filtered_df, entities, value_column)
        elif agg_func == "entropy":
            return self._agg_entropy(filtered_df, entities, value_column)
        return np.full(len(entities), np.nan)

    def _agg_mode(self, df: DataFrame, entities: np.ndarray, col: str) -> np.ndarray:
        def get_mode(x):
            if len(x) == 0:
                return None
            return x.value_counts().idxmax()
        mode_result = df.groupby(self.entity_column)[col].apply(get_mode)
        return np.array([mode_result.get(e, None) for e in entities], dtype=object)

    def _agg_nunique(self, df: DataFrame, entities: np.ndarray, col: str) -> np.ndarray:
        nunique_result = df.groupby(self.entity_column)[col].nunique()
        return np.array([nunique_result.get(e, 0) for e in entities])

    def _agg_mode_ratio(self, df: DataFrame, entities: np.ndarray, col: str) -> np.ndarray:
        def get_mode_ratio(x):
            if len(x) == 0:
                return np.nan
            counts = x.value_counts()
            return counts.iloc[0] / len(x)
        ratio_result = df.groupby(self.entity_column)[col].apply(get_mode_ratio)
        return np.array([ratio_result.get(e, np.nan) for e in entities])

    def _agg_entropy(self, df: DataFrame, entities: np.ndarray, col: str) -> np.ndarray:
        def calc_entropy(x):
            if len(x) == 0:
                return np.nan
            probs = x.value_counts(normalize=True)
            if len(probs) == 1:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        entropy_result = df.groupby(self.entity_column)[col].apply(calc_entropy)
        return np.array([entropy_result.get(e, np.nan) for e in entities])

    def _compute_value_counts(
        self, df: DataFrame, entities: np.ndarray, col: str,
        window: TimeWindow, reference_date: pd.Timestamp
    ) -> Dict[str, np.ndarray]:
        filtered_df = self._filter_by_window(df, window, reference_date)
        unique_values = df[col].dropna().unique()
        result = {}
        for val in unique_values:
            col_name = f"{col}_{val}_count_{window.name}"
            if len(filtered_df) == 0:
                result[col_name] = np.zeros(len(entities))
            else:
                counts = filtered_df[filtered_df[col] == val].groupby(self.entity_column).size()
                result[col_name] = np.array([counts.get(e, 0) for e in entities])
        return result

    def _compute_recency(
        self,
        df: DataFrame,
        entities: np.ndarray,
        reference_date: pd.Timestamp,
    ) -> np.ndarray:
        """Compute days since last event for each entity."""
        last_dates = df.groupby(self.entity_column)[self.time_column].max()
        recency = (reference_date - last_dates).dt.days

        return np.array([recency.get(e, np.nan) for e in entities])

    def _compute_tenure(
        self,
        df: DataFrame,
        entities: np.ndarray,
        reference_date: pd.Timestamp,
    ) -> np.ndarray:
        """Compute days since first event for each entity."""
        first_dates = df.groupby(self.entity_column)[self.time_column].min()
        tenure = (reference_date - first_dates).dt.days

        return np.array([tenure.get(e, np.nan) for e in entities])
