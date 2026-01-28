import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from customer_retention.core.compat import DataFrame, pd


class AggregationType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    COUNT = "count"
    STD = "std"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"
    MODE = "mode"
    NUNIQUE = "nunique"
    MODE_RATIO = "mode_ratio"
    ENTROPY = "entropy"
    VALUE_COUNTS = "value_counts"


CATEGORICAL_AGG_FUNCS = {"mode", "nunique", "mode_ratio", "entropy", "value_counts"}
NUMERIC_AGG_FUNCS = {"sum", "mean", "max", "min", "count", "std", "median", "first", "last"}


@dataclass
class TimeWindow:
    name: str
    days: Optional[int]

    @classmethod
    def from_string(cls, window_str: str) -> "TimeWindow":
        window_str = window_str.lower().strip()
        if window_str == "all_time":
            return cls(name="all_time", days=None)
        if window_str.endswith("d"):
            return cls(name=window_str, days=int(window_str[:-1]))
        if window_str.endswith("h"):
            return cls(name=window_str, days=int(window_str[:-1]) / 24)
        if window_str.endswith("w"):
            return cls(name=window_str, days=int(window_str[:-1]) * 7)
        raise ValueError(f"Unknown window format: {window_str}")


@dataclass
class AggregationPlan:
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
    def __init__(self, entity_column: str, time_column: str):
        self.entity_column = entity_column
        self.time_column = time_column

    def aggregate(
        self, df: DataFrame, windows: Optional[List[str]] = None,
        value_columns: Optional[List[str]] = None, agg_funcs: Optional[List[str]] = None,
        reference_date: Optional[pd.Timestamp] = None, include_event_count: bool = False,
        include_recency: bool = False, include_tenure: bool = False,
        exclude_columns: Optional[List[str]] = None,
    ) -> DataFrame:
        if len(df) == 0:
            return pd.DataFrame()

        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        reference_date = self._validate_reference_date(df, reference_date)
        parsed_windows = [TimeWindow.from_string(w) for w in (windows or ["30d"])]

        exclude_set = set(exclude_columns) if exclude_columns else set()
        if value_columns:
            value_columns = [c for c in value_columns if c not in exclude_set]

        entities = df[self.entity_column].unique()
        result_data = {self.entity_column: entities}

        if include_event_count or (value_columns is None and agg_funcs is None):
            for window in parsed_windows:
                result_data[f"event_count_{window.name}"] = self._compute_event_counts(
                    df, entities, window, reference_date)

        if value_columns and agg_funcs:
            self._add_value_aggregations(
                result_data, df, entities, parsed_windows, value_columns, agg_funcs, reference_date)

        if include_recency:
            result_data["days_since_last_event"] = self._compute_recency(df, entities, reference_date)
        if include_tenure:
            result_data["days_since_first_event"] = self._compute_tenure(df, entities, reference_date)

        result = pd.DataFrame(result_data)
        result.attrs["aggregation_reference_date"] = (
            reference_date.isoformat() if hasattr(reference_date, "isoformat") else str(reference_date))
        result.attrs["aggregation_timestamp"] = pd.Timestamp.now().isoformat()
        return result

    def _add_value_aggregations(
        self, result_data: Dict, df: DataFrame, entities: np.ndarray,
        windows: List[TimeWindow], value_columns: List[str], agg_funcs: List[str],
        reference_date: pd.Timestamp,
    ) -> None:
        for window in windows:
            for col in value_columns:
                for func in agg_funcs:
                    if func == "value_counts":
                        result_data.update(self._compute_value_counts(df, entities, col, window, reference_date))
                    else:
                        result_data[f"{col}_{func}_{window.name}"] = self._compute_aggregation(
                            df, entities, col, func, window, reference_date)

    def generate_plan(
        self, df: DataFrame, windows: List[str], value_columns: List[str], agg_funcs: List[str],
        include_event_count: bool = True, include_recency: bool = False, include_tenure: bool = False,
        exclude_columns: Optional[List[str]] = None,
    ) -> AggregationPlan:
        parsed_windows = [TimeWindow.from_string(w) for w in windows]
        exclude_set = set(exclude_columns) if exclude_columns else set()
        value_columns = [c for c in value_columns if c not in exclude_set]

        feature_columns, value_counts_categories = self._build_feature_column_list(
            df, parsed_windows, value_columns, agg_funcs, include_event_count, include_recency, include_tenure)

        return AggregationPlan(
            entity_column=self.entity_column, time_column=self.time_column, windows=parsed_windows,
            value_columns=value_columns, agg_funcs=agg_funcs, feature_columns=feature_columns,
            include_event_count=include_event_count, include_recency=include_recency,
            include_tenure=include_tenure, value_counts_categories=value_counts_categories)

    def _build_feature_column_list(
        self, df: DataFrame, windows: List[TimeWindow], value_columns: List[str],
        agg_funcs: List[str], include_event_count: bool, include_recency: bool, include_tenure: bool,
    ) -> tuple:
        feature_columns = []
        value_counts_categories: Dict[str, List[str]] = {}

        if include_event_count:
            feature_columns.extend(f"event_count_{w.name}" for w in windows)

        for window in windows:
            for col in value_columns:
                for func in agg_funcs:
                    if func == "value_counts":
                        unique_vals = list(df[col].dropna().unique())
                        value_counts_categories[col] = unique_vals
                        feature_columns.extend(f"{col}_{val}_count_{window.name}" for val in unique_vals)
                    else:
                        feature_columns.append(f"{col}_{func}_{window.name}")

        if include_recency:
            feature_columns.append("days_since_last_event")
        if include_tenure:
            feature_columns.append("days_since_first_event")

        return feature_columns, value_counts_categories

    def _validate_reference_date(self, df: DataFrame, reference_date: Optional[pd.Timestamp]) -> pd.Timestamp:
        data_min, data_max = df[self.time_column].min(), df[self.time_column].max()
        current_date = pd.Timestamp.now()

        if reference_date is None:
            warnings.warn(
                f"reference_date not provided, defaulting to data max ({data_max}). "
                "For production use, provide explicit reference_date for PIT correctness. "
                "This ensures features are computed as-of a specific point in time.",
                UserWarning, stacklevel=3)
            return data_max

        if reference_date > current_date:
            warnings.warn(
                f"reference_date ({reference_date}) is in the future (current: {current_date}). "
                "This may indicate incorrect date handling. Features will use future data.",
                UserWarning, stacklevel=3)

        if reference_date < data_min:
            warnings.warn(
                f"reference_date ({reference_date}) is before all data ({data_min}). "
                "All time-windowed features will be empty or zero.",
                UserWarning, stacklevel=3)

        return reference_date

    def _compute_event_counts(
        self, df: DataFrame, entities: np.ndarray, window: TimeWindow, reference_date: pd.Timestamp,
    ) -> np.ndarray:
        filtered_df = self._filter_by_window(df, window, reference_date)
        counts = filtered_df.groupby(self.entity_column).size()
        return np.array([counts.get(e, 0) for e in entities])

    def _filter_by_window(self, df: DataFrame, window: TimeWindow, reference_date: pd.Timestamp) -> DataFrame:
        if window.days is None:
            return df
        cutoff = reference_date - pd.Timedelta(days=window.days)
        return df[df[self.time_column] >= cutoff]

    def _compute_aggregation(
        self,
        df: DataFrame,
        entities: np.ndarray,
        value_column: str,
        agg_func: str,
        window: TimeWindow,
        reference_date: pd.Timestamp,
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

    def _compute_numeric_agg(
        self, filtered_df: DataFrame, entities: np.ndarray, value_column: str, agg_func: str
    ) -> np.ndarray:
        if agg_func == "count":
            agg_result = filtered_df.groupby(self.entity_column)[value_column].count()
        else:
            agg_result = filtered_df.groupby(self.entity_column)[value_column].agg(agg_func)
        default = 0 if agg_func in ["sum", "count"] else np.nan
        return np.array([agg_result.get(e, default) for e in entities])

    def _compute_categorical_agg(
        self, filtered_df: DataFrame, entities: np.ndarray, value_column: str, agg_func: str
    ) -> np.ndarray:
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
        self, df: DataFrame, entities: np.ndarray, col: str, window: TimeWindow, reference_date: pd.Timestamp
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

    def _compute_recency(self, df: DataFrame, entities: np.ndarray, reference_date: pd.Timestamp) -> np.ndarray:
        last_dates = df.groupby(self.entity_column)[self.time_column].max()
        days_since_last = (reference_date - last_dates).dt.days
        return np.array([days_since_last.get(e, np.nan) for e in entities])

    def _compute_tenure(self, df: DataFrame, entities: np.ndarray, reference_date: pd.Timestamp) -> np.ndarray:
        first_dates = df.groupby(self.entity_column)[self.time_column].min()
        days_since_first = (reference_date - first_dates).dt.days
        return np.array([days_since_first.get(e, np.nan) for e in entities])


def save_aggregated_parquet(df: DataFrame, path: Union[str, Path]) -> Dict[str, str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _extract_temporal_metadata(df)

    df_clean = df.copy()
    df_clean.attrs = {}
    table = pa.Table.from_pandas(df_clean)

    if metadata:
        existing_metadata = table.schema.metadata or {}
        encoded_metadata = {k.encode("utf-8"): v.encode("utf-8") for k, v in metadata.items()}
        encoded_metadata.update(existing_metadata)
        table = table.replace_schema_metadata(encoded_metadata)

    pq.write_table(table, path)
    return metadata


def _extract_temporal_metadata(df: DataFrame) -> Dict[str, str]:
    metadata = {}
    for key in ["aggregation_reference_date", "aggregation_timestamp"]:
        if key in df.attrs:
            value = df.attrs[key]
            metadata[key] = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return metadata
