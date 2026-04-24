import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from customer_retention.core.compat import (
    DataFrame,
    Timedelta,
    Timestamp,
    _is_native_spark_df,
    _is_spark_pandas,
    as_spark_df,
    as_tz_naive,
    cut,
    ensure_timestamp,
    head_as_list,
    is_numeric_dtype,
    native_pd,
    safe_to_datetime,
    timedelta_to_days,
    timestamp_diff_days,
    timestamp_diff_seconds,
    to_pandas,
)

# Cap on distinct values per value_counts column. Cardinality > this is
# almost certainly a misconfigured override (e.g. value_counts on a free-
# text column); refuse to fan it out into thousands of feature columns.
_VALUE_COUNTS_MAX_CARDINALITY = 200

# Aggregation funcs the Spark dispatch path supports natively. Anything
# outside this set falls back to the pandas path (mode/entropy/mode_ratio/
# nunique/std/median/first/last) so we don't silently change semantics.
# Extending this set requires a parity-test addition.
_SPARK_SUPPORTED_AGG_FUNCS = frozenset({"sum", "mean", "max", "min", "count", "value_counts"})


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
        reference_date: Optional[Timestamp] = None, include_event_count: bool = False,
        include_recency: bool = False, include_tenure: bool = False,
        exclude_columns: Optional[List[str]] = None,
    ) -> DataFrame:
        # Dispatch: keep distributed end-to-end when input is Spark/pyspark.pandas
        # AND every requested agg func has a native-Spark implementation. Otherwise
        # fall through to the legacy pandas path (covers mode/entropy/mode_ratio/
        # nunique/std/median/first/last and all native-pandas inputs). Output type
        # mirrors input type: Spark in → native PySpark DataFrame out;
        # pandas in → native pandas out. NB01d's downstream code already calls
        # `as_spark_df(df_aggregated)` on the result, so returning Spark is a
        # pure improvement (eliminates the round-trip via the driver).
        if _is_spark_pandas(df) or _is_native_spark_df(df):
            funcs_for_dispatch = list(agg_funcs or [])
            if all(f in _SPARK_SUPPORTED_AGG_FUNCS for f in funcs_for_dispatch):
                return self._aggregate_spark(
                    df, windows=windows, value_columns=value_columns,
                    agg_funcs=funcs_for_dispatch, reference_date=reference_date,
                    include_event_count=include_event_count,
                    include_recency=include_recency, include_tenure=include_tenure,
                    exclude_columns=exclude_columns,
                )
            warnings.warn(
                f"TimeWindowAggregator: agg_funcs={funcs_for_dispatch!r} contain "
                f"functions outside the Spark-supported set "
                f"({sorted(_SPARK_SUPPORTED_AGG_FUNCS)}); falling back to pandas "
                f"path which collects the input frame to driver memory. "
                f"Add a Spark implementation or narrow agg_funcs to keep the run "
                f"distributed.",
                stacklevel=2,
            )

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

        result = native_pd.DataFrame(result_data)
        result.attrs["aggregation_reference_date"] = (
            reference_date.isoformat() if hasattr(reference_date, "isoformat") else str(reference_date))
        result.attrs["aggregation_timestamp"] = Timestamp.now().isoformat()
        return result

    def _aggregate_spark(
        self, df: Any, windows: Optional[List[str]],
        value_columns: Optional[List[str]], agg_funcs: List[str],
        reference_date: Optional[Timestamp], include_event_count: bool,
        include_recency: bool, include_tenure: bool,
        exclude_columns: Optional[List[str]],
    ) -> Any:
        # Native-Spark, distributed entity-level aggregation. Mirrors the
        # pandas path's column contract: result columns are
        #   {entity_column}, event_count_{w}*, {col}_{func}_{w}*,
        #   {col}_{val}_count_{w}* (for value_counts), days_since_last_event,
        #   days_since_first_event.
        # Funcs supported here: sum, mean, max, min, count, value_counts.
        # Anything else routes to the pandas fallback in aggregate().
        from pyspark.sql import DataFrame as NativeSparkDF
        from pyspark.sql import functions as F  # noqa: N812
        from pyspark.sql.types import (
            ByteType,
            DecimalType,
            DoubleType,
            FloatType,
            IntegerType,
            LongType,
            ShortType,
        )

        spark_df = as_spark_df(df) if not isinstance(df, NativeSparkDF) else df
        if spark_df.rdd.isEmpty():
            spark = spark_df.sparkSession
            return spark.createDataFrame([], schema=f"{self.entity_column} string")

        # Cast time column once, reuse for every window / aggregation. We do NOT
        # mutate the input frame's user-facing schema.
        time_col_cast = F.col(self.time_column).cast("timestamp")
        spark_df = spark_df.withColumn(self.time_column, time_col_cast)

        ref_dt = self._resolve_reference_date_spark(spark_df, reference_date)
        ref_lit = F.lit(ref_dt).cast("timestamp")

        parsed_windows = [TimeWindow.from_string(w) for w in (windows or ["30d"])]

        exclude_set = set(exclude_columns) if exclude_columns else set()
        if value_columns:
            value_columns = [c for c in value_columns if c not in exclude_set]

        # Numeric column detection from the Spark schema. Avoids any pandas
        # dtype inference and keeps the check distributed-safe.
        numeric_spark_types = (
            IntegerType, LongType, DoubleType, FloatType, ShortType, ByteType, DecimalType,
        )
        schema_field_by_name = {f.name: f for f in spark_df.schema.fields}

        def _is_numeric_spark(col: str) -> bool:
            field = schema_field_by_name.get(col)
            return field is not None and isinstance(field.dataType, numeric_spark_types)

        # Base entity frame — every distinct entity becomes one row in the
        # output. Use distinct() not dropDuplicates() for clarity.
        entity_base = (
            spark_df.select(F.col(self.entity_column))
            .where(F.col(self.entity_column).isNotNull())
            .distinct()
        )

        result = entity_base
        scalar_funcs = [f for f in agg_funcs if f in {"sum", "mean", "max", "min", "count"}]
        wants_value_counts = "value_counts" in agg_funcs

        # Resolve distinct values per value_counts column ONCE (outside the
        # per-window loop). Bounded collect via head_as_list — same cap the
        # pandas path uses (raises if cardinality exceeds the cap).
        value_counts_categories: Dict[str, List[Any]] = {}
        if wants_value_counts and value_columns:
            for col in value_columns:
                distinct_pdf = (
                    spark_df.select(F.col(col)).where(F.col(col).isNotNull())
                    .distinct().limit(_VALUE_COUNTS_MAX_CARDINALITY + 1)
                    .toPandas()
                )
                if len(distinct_pdf) > _VALUE_COUNTS_MAX_CARDINALITY:
                    raise ValueError(
                        f"value_counts on column {col!r} has > "
                        f"{_VALUE_COUNTS_MAX_CARDINALITY} distinct values; "
                        "refusing to fan out into thousands of feature columns. "
                        "Either narrow the column upstream or remove it from "
                        "value_counts_columns."
                    )
                value_counts_categories[col] = [v for v in distinct_pdf[col].tolist()]

        # Per-window: filter, then a single batched .agg(...) for all numeric
        # (column × func) pairs that window needs (Catalyst handles them in
        # one shuffle), plus optional value_counts pivots. One Spark job per
        # window for the numerics; one extra per value_counts column.
        for window in parsed_windows:
            filtered = self._filter_by_window_spark(spark_df, window, ref_lit, F)

            agg_exprs = []
            zero_fill_cols: List[str] = []  # tracked explicitly — no Java introspection

            if include_event_count:
                ec_name = f"event_count_{window.name}"
                agg_exprs.append(F.count(F.lit(1)).alias(ec_name))
                zero_fill_cols.append(ec_name)

            if scalar_funcs and value_columns:
                for col in value_columns:
                    if not _is_numeric_spark(col):
                        # Skip non-numeric for numeric funcs — output column
                        # is intentionally absent (matches pandas path which
                        # would store NaN; downstream callers tolerate both).
                        continue
                    for func in scalar_funcs:
                        out_name = f"{col}_{func}_{window.name}"
                        if func == "sum":
                            agg_exprs.append(F.sum(F.col(col)).alias(out_name))
                            zero_fill_cols.append(out_name)
                        elif func == "mean":
                            agg_exprs.append(F.mean(F.col(col)).alias(out_name))
                        elif func == "max":
                            agg_exprs.append(F.max(F.col(col)).alias(out_name))
                        elif func == "min":
                            agg_exprs.append(F.min(F.col(col)).alias(out_name))
                        elif func == "count":
                            # Per-column count = non-null count, matches pandas.
                            agg_exprs.append(F.count(F.col(col)).alias(out_name))
                            zero_fill_cols.append(out_name)

            if agg_exprs:
                per_entity = filtered.groupBy(self.entity_column).agg(*agg_exprs)
                result = result.join(per_entity, on=self.entity_column, how="left")
                if zero_fill_cols:
                    # Zero-fill count/sum/event_count to match pandas semantics
                    # (no events in window → 0, not null).
                    result = result.fillna(0, subset=zero_fill_cols)

            if wants_value_counts and value_columns:
                for col in value_columns:
                    distinct_vals = value_counts_categories.get(col) or []
                    if not distinct_vals:
                        continue
                    pivot_name = f"__pv_{col}"
                    # Cast the pivot column to string so pivot value matching is
                    # deterministic (Spark pivot key matching is exact-string).
                    pivot_input = filtered.withColumn(pivot_name, F.col(col).cast("string"))
                    distinct_vals_str = [None if v is None else str(v) for v in distinct_vals]
                    distinct_vals_str = [v for v in distinct_vals_str if v is not None]
                    if not distinct_vals_str:
                        continue
                    pivoted = (
                        pivot_input.groupBy(self.entity_column)
                        .pivot(pivot_name, distinct_vals_str)
                        .agg(F.count(F.lit(1)))
                    )
                    rename_map = {
                        v: f"{col}_{v}_count_{window.name}" for v in distinct_vals_str
                    }
                    for old, new in rename_map.items():
                        if old in pivoted.columns:
                            pivoted = pivoted.withColumnRenamed(old, new)
                    result = result.join(pivoted, on=self.entity_column, how="left")
                    result = result.fillna(0, subset=list(rename_map.values()))

        # Recency / tenure on the full (unfiltered) frame, mirroring pandas
        # path which uses df where time <= reference_date.
        if include_recency or include_tenure:
            tenure_filter = spark_df.where(F.col(self.time_column) <= ref_lit)
            tenure_aggs = []
            if include_recency:
                tenure_aggs.append(F.max(F.col(self.time_column)).alias("_last_event"))
            if include_tenure:
                tenure_aggs.append(F.min(F.col(self.time_column)).alias("_first_event"))
            tenure_per_entity = tenure_filter.groupBy(self.entity_column).agg(*tenure_aggs)
            result = result.join(tenure_per_entity, on=self.entity_column, how="left")
            if include_recency:
                result = result.withColumn(
                    "days_since_last_event",
                    F.datediff(ref_lit.cast("date"), F.col("_last_event").cast("date")).cast("double"),
                ).drop("_last_event")
            if include_tenure:
                result = result.withColumn(
                    "days_since_first_event",
                    F.datediff(ref_lit.cast("date"), F.col("_first_event").cast("date")).cast("double"),
                ).drop("_first_event")

        return result

    def _resolve_reference_date_spark(self, spark_df: Any, reference_date: Optional[Timestamp]) -> Timestamp:
        if reference_date is not None:
            ts = native_pd.Timestamp(reference_date)
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        from pyspark.sql import functions as F  # noqa: N812
        row = spark_df.agg(F.max(F.col(self.time_column).cast("timestamp"))).first()
        if row is None or row[0] is None:
            raise ValueError(
                f"Cannot resolve reference_date: time column {self.time_column!r} "
                "is empty or all-null."
            )
        return native_pd.Timestamp(row[0])

    def _filter_by_window_spark(self, spark_df: Any, window: TimeWindow, ref_lit: Any, F: Any) -> Any:
        if window.days is None:
            return spark_df.where(F.col(self.time_column) <= ref_lit)
        # Cutoff arithmetic stays inside Spark — no driver-side date math.
        # `ref_lit - INTERVAL N DAYS` evaluates as timestamp - interval → timestamp.
        cutoff = ref_lit - F.expr(f"INTERVAL {int(window.days)} DAYS")
        return spark_df.where(
            (F.col(self.time_column) >= cutoff) & (F.col(self.time_column) <= ref_lit)
        )

    def _add_value_aggregations(
        self, result_data: Dict, df: DataFrame, entities: np.ndarray,
        windows: List[TimeWindow], value_columns: List[str], agg_funcs: List[str],
        reference_date: Timestamp,
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

        # Resolve the unique-value set for each value_counts column ONCE,
        # outside the per-window loop. `df[col].dropna().unique()` returns
        # a pyspark.pandas Series whose `__iter__` is unimplemented; iterate
        # via `head_as_list(...)` (compat helper that bounds the collect)
        # so we never materialize an unbounded series on the driver.
        unique_vals_by_col: Dict[str, List[Any]] = {}
        if "value_counts" in agg_funcs:
            for col in value_columns:
                if col not in unique_vals_by_col:
                    unique_vals_by_col[col] = head_as_list(
                        df[col].dropna().unique(), _VALUE_COUNTS_MAX_CARDINALITY
                    )

        for window in windows:
            for col in value_columns:
                for func in agg_funcs:
                    if func == "value_counts":
                        unique_vals = unique_vals_by_col[col]
                        value_counts_categories[col] = unique_vals
                        feature_columns.extend(f"{col}_{val}_count_{window.name}" for val in unique_vals)
                    else:
                        feature_columns.append(f"{col}_{func}_{window.name}")

        if include_recency:
            feature_columns.append("days_since_last_event")
        if include_tenure:
            feature_columns.append("days_since_first_event")

        return feature_columns, value_counts_categories

    def _validate_reference_date(self, df: DataFrame, reference_date: Optional[Timestamp]) -> Timestamp:
        data_min, data_max = df[self.time_column].min(), df[self.time_column].max()
        current_date = Timestamp.now()

        if reference_date is not None:
            reference_date = as_tz_naive(reference_date)

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
        self, df: DataFrame, entities: np.ndarray, window: TimeWindow, reference_date: Timestamp,
    ) -> np.ndarray:
        filtered_df = self._filter_by_window(df, window, reference_date)
        counts = filtered_df.groupby(self.entity_column).size()
        return np.array([counts.get(e, 0) for e in entities])

    def _filter_by_window(self, df: DataFrame, window: TimeWindow, reference_date: Timestamp) -> DataFrame:
        if window.days is None:
            return df[df[self.time_column] <= reference_date]
        cutoff = reference_date - Timedelta(days=window.days)
        mask = (df[self.time_column] >= cutoff) & (df[self.time_column] <= reference_date)
        return df[mask]

    def _compute_aggregation(
        self,
        df: DataFrame,
        entities: np.ndarray,
        value_column: str,
        agg_func: str,
        window: TimeWindow,
        reference_date: Timestamp,
    ) -> np.ndarray:
        filtered_df = self._filter_by_window(df, window, reference_date)
        if len(filtered_df) == 0:
            default = 0 if agg_func in ["sum", "count", "nunique"] else np.nan
            return np.full(len(entities), default)

        is_numeric = is_numeric_dtype(df[value_column])
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
            probs = x.value_counts(normalize=True).to_numpy()
            if len(probs) == 1:
                return 0.0
            return -np.sum(probs * np.log2(probs))

        entropy_result = df.groupby(self.entity_column)[col].apply(calc_entropy)
        return np.array([entropy_result.get(e, np.nan) for e in entities])

    def _compute_value_counts(
        self, df: DataFrame, entities: np.ndarray, col: str, window: TimeWindow, reference_date: Timestamp
    ) -> Dict[str, np.ndarray]:
        filtered_df = self._filter_by_window(df, window, reference_date)
        # Bounded collect — pyspark.pandas Series has no iterable __iter__;
        # head_as_list materialises the first N distinct values and raises
        # if the cardinality exceeds the cap (safety net against misuse).
        unique_values = head_as_list(df[col].dropna().unique(), _VALUE_COUNTS_MAX_CARDINALITY)
        result = {}
        for val in unique_values:
            col_name = f"{col}_{val}_count_{window.name}"
            if len(filtered_df) == 0:
                result[col_name] = np.zeros(len(entities))
            else:
                counts = filtered_df[filtered_df[col] == val].groupby(self.entity_column).size()
                result[col_name] = np.array([counts.get(e, 0) for e in entities])
        return result

    def _compute_recency(self, df: DataFrame, entities: np.ndarray, reference_date: Timestamp) -> np.ndarray:
        valid_df = df[df[self.time_column] <= reference_date]
        last_dates = valid_df.groupby(self.entity_column)[self.time_column].max()
        days_since_last = timedelta_to_days(reference_date - last_dates)
        return np.array([days_since_last.get(e, np.nan) for e in entities])

    def _compute_tenure(self, df: DataFrame, entities: np.ndarray, reference_date: Timestamp) -> np.ndarray:
        valid_df = df[df[self.time_column] <= reference_date]
        first_dates = valid_df.groupby(self.entity_column)[self.time_column].min()
        days_since_first = timedelta_to_days(reference_date - first_dates)
        return np.array([days_since_first.get(e, np.nan) for e in entities])


def _extra_datetime_feature_names(datetime_columns: list[str]) -> list[str]:
    names: list[str] = []
    for col in datetime_columns:
        names.extend([
            f"{col}_delta_hours", f"{col}_hour", f"{col}_dow", f"{col}_is_weekend",
        ])
    return names


def derive_extra_datetime_features(
    df: DataFrame, time_column: str, datetime_columns: list[str],
    mask_future_columns: Optional[list[str]] = None,
) -> tuple[DataFrame, list[str]]:
    if not datetime_columns:
        return df, []

    mask_set = set(mask_future_columns) if mask_future_columns else set()
    new_columns = _extra_datetime_feature_names(datetime_columns)

    if _is_spark_pandas(df) or _is_native_spark_df(df):
        return _derive_extra_datetime_features_spark(
            df, time_column, datetime_columns, mask_set,
        ), new_columns

    df = df.copy()
    time_series = safe_to_datetime(df[time_column], errors="coerce")

    for col in datetime_columns:
        parsed = safe_to_datetime(df[col], errors="coerce")

        delta_hours_name = f"{col}_delta_hours"
        hour_name = f"{col}_hour"
        dow_name = f"{col}_dow"
        is_weekend_name = f"{col}_is_weekend"

        df[delta_hours_name] = timestamp_diff_seconds(parsed, time_series) / 3600
        df[hour_name] = parsed.dt.hour.astype("Float64")
        df[dow_name] = parsed.dt.dayofweek.astype("Float64")
        df[is_weekend_name] = (parsed.dt.dayofweek >= 5).astype("Float64")

        if col in mask_set:
            future_mask = parsed > time_series
            for name in [delta_hours_name, hour_name, dow_name, is_weekend_name]:
                df[name] = df[name].where(~future_mask)

    return df, new_columns


def _derive_extra_datetime_features_spark(
    df: Any, time_column: str, datetime_columns: list[str], mask_set: set,
) -> Any:
    import pyspark.sql.functions as F  # noqa: N812

    from customer_retention.core.compat.spark_backend import _as_pandas_api

    was_pandas_api = not _is_native_spark_df(df)
    spark_df = as_spark_df(df) if was_pandas_api else df

    time_ts = F.expr(f"try_to_timestamp(`{time_column}`)")
    time_epoch = F.unix_timestamp(time_ts).cast("double")

    derived_names = set(_extra_datetime_feature_names(datetime_columns))
    select_exprs: list[Any] = [F.col(c) for c in spark_df.columns if c not in derived_names]

    for col in datetime_columns:
        parsed_ts = F.expr(f"try_to_timestamp(`{col}`)")
        parsed_epoch = F.unix_timestamp(parsed_ts).cast("double")

        delta_hours = ((parsed_epoch - time_epoch) / F.lit(3600.0)).cast("double")
        hour = F.hour(parsed_ts).cast("double")
        dow = ((F.dayofweek(parsed_ts) + F.lit(5)) % F.lit(7)).cast("double")
        is_weekend = F.when(dow >= F.lit(5.0), F.lit(1.0)).otherwise(F.lit(0.0))

        if col in mask_set:
            future_mask = parsed_ts > time_ts
            null_d = F.lit(None).cast("double")
            delta_hours = F.when(future_mask, null_d).otherwise(delta_hours)
            hour = F.when(future_mask, null_d).otherwise(hour)
            dow = F.when(future_mask, null_d).otherwise(dow)
            is_weekend = F.when(future_mask, null_d).otherwise(is_weekend)

        select_exprs.extend([
            delta_hours.alias(f"{col}_delta_hours"),
            hour.alias(f"{col}_hour"),
            dow.alias(f"{col}_dow"),
            is_weekend.alias(f"{col}_is_weekend"),
        ])

    result = spark_df.select(*select_exprs)
    return _as_pandas_api(result) if was_pandas_api else result


def detect_milestone_pairs(datetime_columns: list[str]) -> list[tuple[str, str]]:
    suffix_map = {
        "_start": "_end", "_begin": "_expire", "_created": "_closed",
        "_open": "_close", "_from": "_to", "_since": "_until",
    }
    col_set = set(datetime_columns)
    pairs: list[tuple[str, str]] = []
    for col in datetime_columns:
        for start_sfx, end_sfx in suffix_map.items():
            if col.endswith(start_sfx):
                prefix = col[: -len(start_sfx)]
                partner = prefix + end_sfx
                if partner in col_set:
                    pairs.append((col, partner))
    return pairs


def derive_entity_datetime_features(
    df: DataFrame, time_column: str, datetime_columns: list[str],
    milestone_pairs: Optional[list[tuple[str, str]]] = None,
    mask_future_columns: Optional[list[str]] = None,
) -> tuple[DataFrame, list[str]]:
    if not datetime_columns:
        return df, []
    df = df.copy()
    time_series = safe_to_datetime(df[time_column], errors="coerce")
    mask_set = set(mask_future_columns) if mask_future_columns else set()
    new_columns: list[str] = []

    for col in datetime_columns:
        parsed = safe_to_datetime(df[col], errors="coerce")
        cols = _derive_universal_features(df, time_series, col, parsed, mask_set)
        new_columns.extend(cols)

    for start_col, end_col in (milestone_pairs or []):
        start_parsed = safe_to_datetime(df[start_col], errors="coerce")
        end_parsed = safe_to_datetime(df[end_col], errors="coerce")
        cols = _derive_milestone_features(
            df, time_series, start_col, end_col, start_parsed, end_parsed, mask_set,
        )
        new_columns.extend(cols)

    return df, new_columns


def _derive_universal_features(
    df: DataFrame, time_series, col: str, parsed, mask_set: set,
) -> list[str]:
    delta = timestamp_diff_days(time_series, parsed)
    names = {
        "days_since": f"days_since_{col}",
        "days_until": f"days_until_{col}",
        "log1p": f"log1p_days_since_{col}",
        "is_missing": f"is_missing_{col}",
        "is_future": f"is_future_{col}",
    }
    df[names["days_since"]] = delta.astype("Float64")
    df[names["days_until"]] = (-delta).astype("Float64")
    df[names["log1p"]] = np.log1p(delta.abs()).astype("Float64")
    df[names["is_missing"]] = parsed.isna().astype("Float64")
    df[names["is_future"]] = (parsed > time_series).astype("Float64")

    if col in mask_set:
        future_mask = parsed > time_series
        for name in names.values():
            if name != names["is_missing"]:
                df[name] = df[name].where(~future_mask)

    col_names = list(names.values())
    return col_names


def _derive_milestone_features(
    df: DataFrame, time_series, start_col: str, end_col: str,
    start_parsed, end_parsed, mask_set: set,
) -> list[str]:
    prefix = _common_prefix(start_col, end_col)

    tenure_name = f"tenure_days_{start_col}"
    dtm_name = f"days_to_milestone_{end_col}"
    w7_name = f"within_7d_{end_col}"
    w30_name = f"within_30d_{end_col}"
    w90_name = f"within_90d_{end_col}"
    bucket_name = f"milestone_bucket_{end_col}"
    progress_name = f"contract_progress_{prefix}"

    tenure = timestamp_diff_days(time_series, start_parsed).astype("Float64")
    dtm = timestamp_diff_days(end_parsed, time_series).astype("Float64")
    duration = timestamp_diff_days(end_parsed, start_parsed).astype("Float64")
    progress = (tenure / duration).clip(0, 2)

    edges = [-np.inf, -90, -30, -7, 0, 7, 30, 90, np.inf]
    bucket = cut(dtm, bins=edges, labels=False).astype("Float64")

    df[tenure_name] = tenure
    df[dtm_name] = dtm
    df[w7_name] = (dtm.abs() <= 7).astype("Float64")
    df[w30_name] = (dtm.abs() <= 30).astype("Float64")
    df[w90_name] = (dtm.abs() <= 90).astype("Float64")
    df[bucket_name] = bucket
    df[progress_name] = progress

    col_names = [tenure_name, dtm_name, w7_name, w30_name, w90_name, bucket_name, progress_name]

    for col, parsed in [(start_col, start_parsed), (end_col, end_parsed)]:
        if col in mask_set:
            future_mask = parsed > time_series
            for name in col_names:
                df[name] = df[name].where(~future_mask)

    return col_names


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    prefix = a[:i].rstrip("_")
    return prefix


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


def save_aggregated_delta(df: DataFrame, path: Union[str, Path]) -> Dict[str, str]:
    import json as _json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _extract_temporal_metadata(df)

    try:
        from customer_retention.integrations.adapters.factory import get_delta
        storage = get_delta()
        storage.write(df, str(path))
    except ImportError:
        df.to_parquet(str(path) + ".parquet", index=False)

    if metadata:
        sidecar_path = path.parent / f"{path.name}_temporal_metadata.json"
        with open(sidecar_path, "w") as f:
            _json.dump(metadata, f, indent=2)

    return metadata


def save_aggregated(df: DataFrame, path: Union[str, Path]) -> Dict[str, str]:
    try:
        from customer_retention.integrations.adapters.factory import get_delta  # noqa: F401
        return save_aggregated_delta(df, path)
    except ImportError:
        return save_aggregated_parquet(df, path)
