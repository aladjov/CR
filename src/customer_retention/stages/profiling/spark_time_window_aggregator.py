from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from customer_retention.core.compat import (
    Timestamp,
    as_tz_naive,
    ensure_timestamp,
    head_as_list,
    is_numeric_dtype,
    native_pd,
    timedelta_to_days,
    to_pandas,
)

from .time_window_aggregator import (
    CATEGORICAL_AGG_FUNCS,
    NUMERIC_AGG_FUNCS,
    TimeWindow,
    TimeWindowAggregator,
)


class SparkTimeWindowAggregator(TimeWindowAggregator):
    def aggregate(
        self, df: Any, windows: Optional[List[str]] = None,
        value_columns: Optional[List[str]] = None, agg_funcs: Optional[List[str]] = None,
        reference_date: Optional[Timestamp] = None, include_event_count: bool = False,
        include_recency: bool = False, include_tenure: bool = False,
        exclude_columns: Optional[List[str]] = None,
    ) -> Any:
        df = to_pandas(df)
        if len(df) == 0:
            return native_pd.DataFrame()

        df = df.copy()
        ensure_timestamp(df, self.time_column)
        df[self.time_column] = as_tz_naive(df[self.time_column])
        reference_date = self._validate_reference_date(df, reference_date)
        parsed_windows = [TimeWindow.from_string(w) for w in (windows or ["30d"])]

        exclude_set = set(exclude_columns) if exclude_columns else set()
        if value_columns:
            value_columns = [c for c in value_columns if c not in exclude_set]

        result = df[[self.entity_column]].drop_duplicates().reset_index(drop=True)

        if include_event_count or (value_columns is None and agg_funcs is None):
            result = self._batch_event_counts(df, result, parsed_windows, reference_date)

        if value_columns and agg_funcs:
            result = self._batch_value_aggregations(
                df, result, parsed_windows, value_columns, agg_funcs, reference_date)

        if include_recency:
            result = self._batch_recency(df, result, reference_date)
        if include_tenure:
            result = self._batch_tenure(df, result, reference_date)

        result.attrs["aggregation_reference_date"] = (
            reference_date.isoformat() if hasattr(reference_date, "isoformat") else str(reference_date))
        result.attrs["aggregation_timestamp"] = Timestamp.now().isoformat()
        return result

    def _filter_by_window(self, df: Any, window: TimeWindow, reference_date: Timestamp) -> Any:
        if window.days is None:
            return df[df[self.time_column] <= reference_date]
        from customer_retention.core.compat import Timedelta
        cutoff = reference_date - Timedelta(days=window.days)
        return df[(df[self.time_column] >= cutoff) & (df[self.time_column] <= reference_date)]

    def _batch_event_counts(
        self, df: Any, result: Any, windows: List[TimeWindow], reference_date: Timestamp,
    ) -> Any:
        for window in windows:
            filtered = self._filter_by_window(df, window, reference_date)
            counts = filtered.groupby(self.entity_column).size().reset_index(name=f"event_count_{window.name}")
            result = result.merge(counts, on=self.entity_column, how="left")
            result[f"event_count_{window.name}"] = result[f"event_count_{window.name}"].fillna(0).astype(int)
        return result

    def _batch_value_aggregations(
        self, df: Any, result: Any, windows: List[TimeWindow],
        value_columns: List[str], agg_funcs: List[str], reference_date: Timestamp,
    ) -> Any:
        numeric_funcs = [f for f in agg_funcs if f in NUMERIC_AGG_FUNCS]
        categorical_funcs = [f for f in agg_funcs if f in CATEGORICAL_AGG_FUNCS and f != "value_counts"]
        has_value_counts = "value_counts" in agg_funcs

        for window in windows:
            filtered = self._filter_by_window(df, window, reference_date)
            if numeric_funcs:
                result = self._batch_numeric_aggs(
                    df, filtered, result, window, value_columns, numeric_funcs)
            if categorical_funcs:
                result = self._batch_categorical_aggs(
                    df, filtered, result, window, value_columns, categorical_funcs)
            if has_value_counts:
                result = self._batch_value_counts(
                    df, filtered, result, window, value_columns)
        return result

    def _batch_numeric_aggs(
        self, full_df: Any, filtered: Any, result: Any, window: TimeWindow,
        value_columns: List[str], funcs: List[str],
    ) -> Any:
        numeric_cols = [c for c in value_columns if is_numeric_dtype(full_df[c])]
        non_numeric_cols = [c for c in value_columns if not is_numeric_dtype(full_df[c])]

        for col in non_numeric_cols:
            for func in funcs:
                result[f"{col}_{func}_{window.name}"] = np.nan

        if not numeric_cols or len(filtered) == 0:
            for col in numeric_cols:
                for func in funcs:
                    default = 0 if func in ("sum", "count") else np.nan
                    result[f"{col}_{func}_{window.name}"] = default
            return result

        agg_dict = {}
        for col in numeric_cols:
            for func in funcs:
                agg_dict[f"{col}_{func}_{window.name}"] = (col, func)

        grouped = filtered.groupby(self.entity_column)
        for output_name, (col, func) in agg_dict.items():
            agg_result = grouped[col].agg(func).reset_index(name=output_name)
            result = result.merge(agg_result, on=self.entity_column, how="left")
            if func in ("sum", "count"):
                result[output_name] = result[output_name].fillna(0)

        return result

    def _batch_categorical_aggs(
        self, full_df: Any, filtered: Any, result: Any, window: TimeWindow,
        value_columns: List[str], funcs: List[str],
    ) -> Any:
        for col in value_columns:
            for func in funcs:
                col_name = f"{col}_{func}_{window.name}"
                if len(filtered) == 0:
                    default = 0 if func == "nunique" else np.nan
                    result[col_name] = default
                    continue
                agg_series = self._categorical_groupby(filtered, col, func)
                merged = agg_series.reset_index(name=col_name) if hasattr(agg_series, 'reset_index') else agg_series
                if not isinstance(merged, native_pd.DataFrame):
                    merged = merged.to_frame(name=col_name).reset_index()
                result = result.merge(merged, on=self.entity_column, how="left")
                if func == "nunique":
                    result[col_name] = result[col_name].fillna(0)
        return result

    def _categorical_groupby(self, filtered: Any, col: str, func: str) -> Any:
        grouped = filtered.groupby(self.entity_column)[col]
        if func == "mode":
            return grouped.apply(lambda x: x.value_counts().idxmax() if len(x) > 0 else None)
        if func == "nunique":
            return grouped.nunique()
        if func == "mode_ratio":
            return grouped.apply(
                lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else np.nan)
        if func == "entropy":
            return grouped.apply(self._calc_entropy)
        return grouped.apply(lambda x: np.nan)

    @staticmethod
    def _calc_entropy(x: Any) -> float:
        if len(x) == 0:
            return np.nan
        probs = x.value_counts(normalize=True).to_numpy()
        if len(probs) == 1:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def _batch_value_counts(
        self, full_df: Any, filtered: Any, result: Any,
        window: TimeWindow, value_columns: List[str],
    ) -> Any:
        for col in value_columns:
            unique_values = head_as_list(full_df[col].dropna().unique(), 100)
            for val in unique_values:
                col_name = f"{col}_{val}_count_{window.name}"
                if len(filtered) == 0:
                    result[col_name] = 0
                else:
                    val_df = filtered[filtered[col] == val]
                    counts = val_df.groupby(self.entity_column).size().reset_index(name=col_name)
                    result = result.merge(counts, on=self.entity_column, how="left")
                    result[col_name] = result[col_name].fillna(0).astype(int)
        return result

    def _batch_recency(self, df: Any, result: Any, reference_date: Timestamp) -> Any:
        valid = df[df[self.time_column] <= reference_date]
        last_dates = valid.groupby(self.entity_column)[self.time_column].max().reset_index(
            name="_last_event")
        result = result.merge(last_dates, on=self.entity_column, how="left")
        result["days_since_last_event"] = timedelta_to_days(reference_date - result["_last_event"])
        result = result.drop(columns=["_last_event"])
        return result

    def _batch_tenure(self, df: Any, result: Any, reference_date: Timestamp) -> Any:
        valid = df[df[self.time_column] <= reference_date]
        first_dates = valid.groupby(self.entity_column)[self.time_column].min().reset_index(
            name="_first_event")
        result = result.merge(first_dates, on=self.entity_column, how="left")
        result["days_since_first_event"] = timedelta_to_days(reference_date - result["_first_event"])
        result = result.drop(columns=["_first_event"])
        return result
