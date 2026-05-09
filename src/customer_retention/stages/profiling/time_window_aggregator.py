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

# Catalyst's analyzer allocates per-expression and trips over an O(N^2) plan
# step at very wide aggregations. The empirically-safe ceiling is ~200 agg
# exprs per `.agg()` call; we chunk in groups of this size and `outer`-join
# the per-entity chunk results together. See `coding_practice_batched_agg`.
_SPARK_AGG_CHUNK_SIZE = 150


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
        value_counts_columns: Optional[List[str]] = None,
    ) -> DataFrame:
        # `value_counts_columns` scopes the `value_counts` aggregation to an
        # EXPLICIT list of categorical columns — independent of the numeric
        # `value_columns × agg_funcs` fan-out. Callers that wire per-value-type
        # counts via `registry.add_bronze_value_counts(...)` pass the registry
        # list here; the aggregator will NOT apply `value_counts` to every
        # entry of `value_columns`, which previously caused continuous numeric
        # columns (e.g. `*_delta_hours`) to be fanned out into thousands of
        # feature columns and fail-fast under the cardinality cap.
        #
        # Back-compat: if `value_counts_columns` is None/empty AND `"value_counts"`
        # appears in `agg_funcs`, the legacy behavior (value_counts applied to
        # every `value_columns` entry) is preserved.
        _vc_cols_effective = list(value_counts_columns or [])
        _vc_in_funcs = "value_counts" in (agg_funcs or [])
        if _vc_cols_effective and not _vc_in_funcs:
            # Normalize: if caller passed the explicit list, we still need
            # "value_counts" in the dispatch key so the Spark path knows to
            # emit the per-value count columns.
            agg_funcs = list(agg_funcs or []) + ["value_counts"]
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
                    value_counts_columns=_vc_cols_effective or None,
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
                result_data, df, entities, parsed_windows, value_columns, agg_funcs, reference_date,
                value_counts_columns=_vc_cols_effective or None,
            )

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
        value_counts_columns: Optional[List[str]] = None,
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

        # Empty-frame short-circuit via pure Spark SQL. The RDD-layer
        # equivalent (`isEmpty()` on the underlying RDD) is forbidden on
        # Databricks shared clusters under Unity Catalog (NOT_IMPLEMENTED).
        # `.limit(1).count()` works on every cluster mode.
        if spark_df.limit(1).count() == 0:
            spark = spark_df.sparkSession
            return spark.createDataFrame([], schema=f"{self.entity_column} string")

        # Cast time column once, reuse for every window / aggregation. We do NOT
        # mutate the input frame's user-facing schema.
        spark_df = spark_df.withColumn(
            self.time_column, F.col(self.time_column).cast("timestamp")
        )

        ref_dt = self._resolve_reference_date_spark(spark_df, reference_date)
        ref_lit = F.lit(ref_dt).cast("timestamp")

        parsed_windows = [TimeWindow.from_string(w) for w in (windows or ["30d"])]

        exclude_set = set(exclude_columns) if exclude_columns else set()
        if value_columns:
            value_columns = [c for c in value_columns if c not in exclude_set]

        # Numeric column detection from the Spark schema. Distributed-safe;
        # no pandas dtype inference, no driver collect.
        numeric_spark_types = (
            IntegerType, LongType, DoubleType, FloatType, ShortType, ByteType, DecimalType,
        )
        schema_field_by_name = {f.name: f for f in spark_df.schema.fields}

        def _is_numeric_spark(col: str) -> bool:
            field = schema_field_by_name.get(col)
            return field is not None and isinstance(field.dataType, numeric_spark_types)

        # Base entity frame — every distinct entity becomes one row in the output.
        entity_base = (
            spark_df.select(F.col(self.entity_column))
            .where(F.col(self.entity_column).isNotNull())
            .distinct()
        )

        scalar_funcs = [f for f in agg_funcs if f in {"sum", "mean", "max", "min", "count"}]
        wants_value_counts = "value_counts" in agg_funcs

        # Resolve the EFFECTIVE set of columns to fan out via value_counts.
        # Precedence:
        #   1. Explicit `value_counts_columns` kwarg (new path) — scoped to
        #      a named list of categorical columns, independent of
        #      `value_columns`. Back-compat-safe.
        #   2. Legacy: when `"value_counts" in agg_funcs` and no kwarg,
        #      every `value_columns` entry gets value_counts. This is the
        #      historical behavior; retained so existing callers keep working.
        if value_counts_columns:
            _vc_target_cols = [c for c in value_counts_columns if c not in exclude_set]
        elif wants_value_counts and value_columns:
            _vc_target_cols = list(value_columns)
        else:
            _vc_target_cols = []

        # Resolve distinct values per target column. Bounded collect
        # (_VALUE_COUNTS_MAX_CARDINALITY cap; raises if exceeded so a
        # continuous numeric column isn't silently fanned out into thousands
        # of feature columns).
        value_counts_categories: Dict[str, List[str]] = {}
        for col in _vc_target_cols:
            distinct_rows = (
                spark_df.select(F.col(col).cast("string").alias(col))
                .where(F.col(col).isNotNull())
                .distinct()
                .limit(_VALUE_COUNTS_MAX_CARDINALITY + 1)
                .collect()
            )
            if len(distinct_rows) > _VALUE_COUNTS_MAX_CARDINALITY:
                raise ValueError(
                    f"value_counts on column {col!r} has > "
                    f"{_VALUE_COUNTS_MAX_CARDINALITY} distinct values; "
                    "refusing to fan out into thousands of feature columns. "
                    "Either narrow the column upstream or pass a scoped "
                    "`value_counts_columns=[...]` that excludes it "
                    "(continuous numeric columns don't belong here)."
                )
            value_counts_categories[col] = [
                r[col] for r in distinct_rows if r[col] is not None
            ]

        # Build ALL aggregation expressions for ALL windows in ONE pass via
        # conditional aggregation. Each expression is `F.sum/mean/max/min(
        # F.when(in_window_for_W, expr).otherwise(...))`. Catalyst evaluates
        # the whole agg with a single shuffle (chunked at
        # _SPARK_AGG_CHUNK_SIZE to avoid the analyzer's O(N²) plan blowup —
        # see `coding_practice_batched_agg`).
        #
        # This collapses what was previously N_windows × N_value_cols
        # separate Spark jobs into ceil(total_exprs / chunk_size) jobs.
        all_exprs: List[Any] = []
        zero_fill_cols: List[str] = []

        for window in parsed_windows:
            in_window = self._in_window_expr(window, ref_lit, F)

            if include_event_count:
                ec_name = f"event_count_{window.name}"
                all_exprs.append(
                    F.sum(F.when(in_window, 1).otherwise(0)).alias(ec_name)
                )
                zero_fill_cols.append(ec_name)

            if scalar_funcs and value_columns:
                for col in value_columns:
                    if not _is_numeric_spark(col):
                        continue
                    col_ref = F.col(col)
                    in_window_col = F.when(in_window, col_ref)
                    for func in scalar_funcs:
                        out_name = f"{col}_{func}_{window.name}"
                        if func == "sum":
                            all_exprs.append(F.sum(in_window_col).alias(out_name))
                            zero_fill_cols.append(out_name)
                        elif func == "mean":
                            all_exprs.append(F.mean(in_window_col).alias(out_name))
                        elif func == "max":
                            all_exprs.append(F.max(in_window_col).alias(out_name))
                        elif func == "min":
                            all_exprs.append(F.min(in_window_col).alias(out_name))
                        elif func == "count":
                            # Non-null count of in-window rows for this column.
                            all_exprs.append(
                                F.sum(F.when(in_window & col_ref.isNotNull(), 1).otherwise(0))
                                .alias(out_name)
                            )
                            zero_fill_cols.append(out_name)

            # value_counts fan-out is scoped to _vc_target_cols (explicit
            # `value_counts_columns` kwarg, or legacy = value_columns when
            # only `"value_counts" in agg_funcs`). Continuous numeric columns
            # in `value_columns` are NO LONGER iterated here.
            for col in _vc_target_cols:
                distinct_vals = value_counts_categories.get(col) or []
                for val in distinct_vals:
                    out_name = f"{col}_{val}_count_{window.name}"
                    all_exprs.append(
                        F.sum(
                            F.when(
                                in_window & (F.col(col).cast("string") == F.lit(val)),
                                1,
                            ).otherwise(0)
                        ).alias(out_name)
                    )
                    zero_fill_cols.append(out_name)

        # Recency / tenure also folded into the same single-pass agg.
        if include_recency:
            all_exprs.append(
                F.max(
                    F.when(F.col(self.time_column) <= ref_lit, F.col(self.time_column))
                ).alias("_last_event")
            )
        if include_tenure:
            all_exprs.append(
                F.min(
                    F.when(F.col(self.time_column) <= ref_lit, F.col(self.time_column))
                ).alias("_first_event")
            )

        # ONE chunked groupBy(entity).agg() — Catalyst parallelises across
        # all executor cores; chunks merge back via outer join on entity.
        per_entity = self._chunked_groupby_agg(
            spark_df, [self.entity_column], all_exprs, _SPARK_AGG_CHUNK_SIZE
        )

        result = entity_base.join(per_entity, on=self.entity_column, how="left")

        if zero_fill_cols:
            # Pandas-path parity: no events in window → 0, not null. Applies
            # to event_count_*, *_count_*, and *_sum_* (matches pandas
            # `fillna(0)` semantics for count/sum on empty groups).
            result = result.fillna(0, subset=zero_fill_cols)

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

    def _in_window_expr(self, window: "TimeWindow", ref_lit: Any, F: Any) -> Any:
        time_col = F.col(self.time_column)
        if window.days is None:
            # all_time: include every row whose time is <= reference_date.
            return time_col <= ref_lit
        cutoff = ref_lit - F.expr(f"INTERVAL {int(window.days)} DAYS")
        return (time_col >= cutoff) & (time_col <= ref_lit)

    @staticmethod
    def _chunked_groupby_agg(spark_df: Any, group_cols: List[str], exprs: List[Any],
                              chunk_size: int) -> Any:
        if not exprs:
            return spark_df.select(*group_cols).distinct()
        if len(exprs) <= chunk_size:
            return spark_df.groupBy(*group_cols).agg(*exprs)
        # Split into chunks; each chunk is its own groupBy().agg() shuffle.
        # Outer-join the chunk results back together on the group keys so we
        # don't lose entities that appear only in some chunks (won't happen
        # because all chunks group on the same key, but `outer` is the safe
        # default to mirror what a single big agg would yield).
        chunks = [exprs[i:i + chunk_size] for i in range(0, len(exprs), chunk_size)]
        chunk_results = [spark_df.groupBy(*group_cols).agg(*chunk) for chunk in chunks]
        merged = chunk_results[0]
        for cr in chunk_results[1:]:
            merged = merged.join(cr, on=group_cols, how="outer")
        return merged

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

    def _add_value_aggregations(
        self, result_data: Dict, df: DataFrame, entities: np.ndarray,
        windows: List[TimeWindow], value_columns: List[str], agg_funcs: List[str],
        reference_date: Timestamp,
        value_counts_columns: Optional[List[str]] = None,
    ) -> None:
        # Scoping rule matches `_aggregate_spark`: an explicit
        # `value_counts_columns` list restricts where `value_counts` fans out.
        # Absent the kwarg we fall back to the legacy behavior (every
        # `value_columns` entry gets value_counts).
        _vc_target_cols = list(value_counts_columns) if value_counts_columns else list(value_columns)
        _scalar_funcs = [f for f in agg_funcs if f != "value_counts"]
        _wants_vc = "value_counts" in agg_funcs or bool(value_counts_columns)
        for window in windows:
            for col in value_columns:
                for func in _scalar_funcs:
                    result_data[f"{col}_{func}_{window.name}"] = self._compute_aggregation(
                        df, entities, col, func, window, reference_date)
            if _wants_vc:
                for col in _vc_target_cols:
                    result_data.update(
                        self._compute_value_counts(df, entities, col, window, reference_date)
                    )

    def generate_plan(
        self, df: DataFrame, windows: List[str], value_columns: List[str], agg_funcs: List[str],
        include_event_count: bool = True, include_recency: bool = False, include_tenure: bool = False,
        exclude_columns: Optional[List[str]] = None,
        value_counts_columns: Optional[List[str]] = None,
    ) -> AggregationPlan:
        parsed_windows = [TimeWindow.from_string(w) for w in windows]
        exclude_set = set(exclude_columns) if exclude_columns else set()
        value_columns = [c for c in value_columns if c not in exclude_set]
        _vc_cols = (
            [c for c in value_counts_columns if c not in exclude_set]
            if value_counts_columns else None
        )

        feature_columns, value_counts_categories = self._build_feature_column_list(
            df, parsed_windows, value_columns, agg_funcs, include_event_count, include_recency,
            include_tenure, value_counts_columns=_vc_cols,
        )

        return AggregationPlan(
            entity_column=self.entity_column, time_column=self.time_column, windows=parsed_windows,
            value_columns=value_columns, agg_funcs=agg_funcs, feature_columns=feature_columns,
            include_event_count=include_event_count, include_recency=include_recency,
            include_tenure=include_tenure, value_counts_categories=value_counts_categories)

    def _build_feature_column_list(
        self, df: DataFrame, windows: List[TimeWindow], value_columns: List[str],
        agg_funcs: List[str], include_event_count: bool, include_recency: bool, include_tenure: bool,
        value_counts_columns: Optional[List[str]] = None,
    ) -> tuple:
        feature_columns = []
        value_counts_categories: Dict[str, List[str]] = {}

        if include_event_count:
            feature_columns.extend(f"event_count_{w.name}" for w in windows)

        # Scoping (same rule as `_aggregate_spark` / `_add_value_aggregations`):
        # `value_counts_columns` kwarg scopes the fan-out to a named list;
        # legacy behavior (empty kwarg + `"value_counts"` in `agg_funcs`)
        # falls back to every `value_columns` entry.
        _vc_target_cols = (
            list(value_counts_columns) if value_counts_columns
            else (list(value_columns) if "value_counts" in agg_funcs else [])
        )
        _wants_vc = bool(_vc_target_cols)

        # Resolve the unique-value set for each value_counts target column
        # ONCE, outside the per-window loop. Bounded via head_as_list (caps
        # at _VALUE_COUNTS_MAX_CARDINALITY) so a continuous numeric column
        # can't materialise thousands of feature names.
        unique_vals_by_col: Dict[str, List[Any]] = {}
        for col in _vc_target_cols:
            unique_vals_by_col[col] = head_as_list(
                df[col].dropna().unique(), _VALUE_COUNTS_MAX_CARDINALITY
            )

        _scalar_funcs = [f for f in agg_funcs if f != "value_counts"]

        for window in windows:
            for col in value_columns:
                for func in _scalar_funcs:
                    feature_columns.append(f"{col}_{func}_{window.name}")
            if _wants_vc:
                for col in _vc_target_cols:
                    unique_vals = unique_vals_by_col[col]
                    value_counts_categories[col] = unique_vals
                    feature_columns.extend(f"{col}_{val}_count_{window.name}" for val in unique_vals)

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

        new_cols = {
            delta_hours_name: timestamp_diff_seconds(parsed, time_series) / 3600,
            hour_name: parsed.dt.hour.astype("Float64"),
            dow_name: parsed.dt.dayofweek.astype("Float64"),
            is_weekend_name: (parsed.dt.dayofweek >= 5).astype("Float64"),
        }

        if col in mask_set:
            future_mask = parsed > time_series
            for name in [delta_hours_name, hour_name, dow_name, is_weekend_name]:
                new_cols[name] = new_cols[name].where(~future_mask)

        # Single batched assign — one Spark plan rewrite for pyspark.pandas
        # callers, one ``__setitem__`` call for native pandas. Replaces 4-8
        # chained ``df[col] = expr`` mutations that ballooned the Spark plan
        # under pyspark.pandas (banned pattern, see core.compat).
        df = df.assign(**new_cols)

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


def _universal_feature_names(col: str) -> dict[str, str]:
    return {
        "days_since": f"days_since_{col}",
        "days_until": f"days_until_{col}",
        "log1p": f"log1p_days_since_{col}",
        "is_missing": f"is_missing_{col}",
        "is_future": f"is_future_{col}",
    }


def _milestone_feature_names(start_col: str, end_col: str) -> tuple[dict[str, str], str]:
    prefix = _common_prefix(start_col, end_col)
    names = {
        "tenure": f"tenure_days_{start_col}",
        "dtm": f"days_to_milestone_{end_col}",
        "w7": f"within_7d_{end_col}",
        "w30": f"within_30d_{end_col}",
        "w90": f"within_90d_{end_col}",
        "bucket": f"milestone_bucket_{end_col}",
        "progress": f"contract_progress_{prefix}",
    }
    return names, prefix


def _entity_datetime_feature_names(
    datetime_columns: list[str],
    milestone_pairs: Optional[list[tuple[str, str]]] = None,
) -> list[str]:
    out: list[str] = []
    for col in datetime_columns:
        out.extend(_universal_feature_names(col).values())
    for start_col, end_col in (milestone_pairs or []):
        names, _ = _milestone_feature_names(start_col, end_col)
        out.extend(names.values())
    return out


def derive_entity_datetime_features(
    df: DataFrame, time_column: str, datetime_columns: list[str],
    milestone_pairs: Optional[list[tuple[str, str]]] = None,
    mask_future_columns: Optional[list[str]] = None,
) -> tuple[DataFrame, list[str]]:
    if not datetime_columns:
        return df, []

    mask_set = set(mask_future_columns) if mask_future_columns else set()
    new_columns = _entity_datetime_feature_names(datetime_columns, milestone_pairs)

    if _is_spark_pandas(df) or _is_native_spark_df(df):
        return _derive_entity_datetime_features_spark(
            df, time_column, datetime_columns, milestone_pairs, mask_set,
        ), new_columns

    df = df.copy()
    time_series = safe_to_datetime(df[time_column], errors="coerce")

    for col in datetime_columns:
        parsed = safe_to_datetime(df[col], errors="coerce")
        df = _derive_universal_features(df, time_series, col, parsed, mask_set)

    for start_col, end_col in (milestone_pairs or []):
        start_parsed = safe_to_datetime(df[start_col], errors="coerce")
        end_parsed = safe_to_datetime(df[end_col], errors="coerce")
        df = _derive_milestone_features(
            df, time_series, start_col, end_col, start_parsed, end_parsed, mask_set,
        )

    return df, new_columns


def _derive_universal_features(
    df: DataFrame, time_series, col: str, parsed, mask_set: set,
) -> DataFrame:
    """Compute the 5 universal datetime features for ``col`` and return a new
    DataFrame with them attached.

    Returns the augmented DataFrame instead of mutating in place so that one
    batched ``df.assign(**dict)`` is emitted per call (single Spark plan
    rewrite under pyspark.pandas) rather than 5-9 chained ``df[col] = expr``
    mutations. Chained assignment on pyspark.pandas DataFrames balloons the
    Spark plan and was the cause of the SparkConnect 404 on the ``account``
    dataset in spschurn-fa23ccd0.
    """
    delta = timestamp_diff_days(time_series, parsed)
    names = _universal_feature_names(col)

    new_cols = {
        names["days_since"]: delta.astype("Float64"),
        names["days_until"]: (-delta).astype("Float64"),
        names["log1p"]: np.log1p(delta.abs()).astype("Float64"),
        names["is_missing"]: parsed.isna().astype("Float64"),
        names["is_future"]: (parsed > time_series).astype("Float64"),
    }

    if col in mask_set:
        future_mask = parsed > time_series
        for name in (
            names["days_since"], names["days_until"],
            names["log1p"], names["is_future"],
        ):
            new_cols[name] = new_cols[name].where(~future_mask)

    return df.assign(**new_cols)


def _derive_milestone_features(
    df: DataFrame, time_series, start_col: str, end_col: str,
    start_parsed, end_parsed, mask_set: set,
) -> DataFrame:
    """Compute the 7 milestone-pair features and return a new DataFrame with
    them attached. See ``_derive_universal_features`` for why this returns a
    new frame instead of mutating in place.
    """
    names, _ = _milestone_feature_names(start_col, end_col)

    tenure = timestamp_diff_days(time_series, start_parsed).astype("Float64")
    dtm = timestamp_diff_days(end_parsed, time_series).astype("Float64")
    duration = timestamp_diff_days(end_parsed, start_parsed).astype("Float64")
    progress = (tenure / duration).clip(0, 2)

    edges = [-np.inf, -90, -30, -7, 0, 7, 30, 90, np.inf]
    bucket = cut(dtm, bins=edges, labels=False).astype("Float64")

    new_cols = {
        names["tenure"]: tenure,
        names["dtm"]: dtm,
        names["w7"]: (dtm.abs() <= 7).astype("Float64"),
        names["w30"]: (dtm.abs() <= 30).astype("Float64"),
        names["w90"]: (dtm.abs() <= 90).astype("Float64"),
        names["bucket"]: bucket,
        names["progress"]: progress,
    }

    for col, parsed in [(start_col, start_parsed), (end_col, end_parsed)]:
        if col in mask_set:
            future_mask = parsed > time_series
            for name in list(new_cols):
                new_cols[name] = new_cols[name].where(~future_mask)

    return df.assign(**new_cols)


def _derive_entity_datetime_features_spark(
    df: Any, time_column: str, datetime_columns: list[str],
    milestone_pairs: Optional[list[tuple[str, str]]], mask_set: set,
) -> Any:
    """Spark-native variant of :func:`derive_entity_datetime_features`.

    Builds every per-column / per-pair feature expression upfront and applies
    them as a single ``select(*)`` so the resulting Spark plan has one node
    per call instead of one per column-add. Mirrors
    :func:`_derive_extra_datetime_features_spark`.
    """
    import pyspark.sql.functions as F  # noqa: N812

    from customer_retention.core.compat.spark_backend import _as_pandas_api

    was_pandas_api = not _is_native_spark_df(df)
    spark_df = as_spark_df(df) if was_pandas_api else df

    time_ts = F.expr(f"try_to_timestamp(`{time_column}`)")
    time_epoch = F.unix_timestamp(time_ts).cast("double")
    null_d = F.lit(None).cast("double")
    seconds_per_day = F.lit(86400.0)

    derived_names = set(_entity_datetime_feature_names(datetime_columns, milestone_pairs))
    select_exprs: list[Any] = [F.col(c) for c in spark_df.columns if c not in derived_names]

    for col in datetime_columns:
        parsed_ts = F.expr(f"try_to_timestamp(`{col}`)")
        parsed_epoch = F.unix_timestamp(parsed_ts).cast("double")
        delta_days = ((time_epoch - parsed_epoch) / seconds_per_day).cast("double")
        days_since = delta_days
        days_until = (-delta_days).cast("double")
        log1p_val = F.log1p(F.abs(delta_days)).cast("double")
        is_missing = F.when(parsed_ts.isNull(), F.lit(1.0)).otherwise(F.lit(0.0))
        is_future = F.when(parsed_ts > time_ts, F.lit(1.0)).otherwise(F.lit(0.0))

        if col in mask_set:
            future_mask = parsed_ts > time_ts
            days_since = F.when(future_mask, null_d).otherwise(days_since)
            days_until = F.when(future_mask, null_d).otherwise(days_until)
            log1p_val = F.when(future_mask, null_d).otherwise(log1p_val)
            is_future = F.when(future_mask, null_d).otherwise(is_future)

        names = _universal_feature_names(col)
        select_exprs.extend([
            days_since.alias(names["days_since"]),
            days_until.alias(names["days_until"]),
            log1p_val.alias(names["log1p"]),
            is_missing.alias(names["is_missing"]),
            is_future.alias(names["is_future"]),
        ])

    for start_col, end_col in (milestone_pairs or []):
        start_ts = F.expr(f"try_to_timestamp(`{start_col}`)")
        end_ts = F.expr(f"try_to_timestamp(`{end_col}`)")
        start_epoch = F.unix_timestamp(start_ts).cast("double")
        end_epoch = F.unix_timestamp(end_ts).cast("double")

        tenure = ((time_epoch - start_epoch) / seconds_per_day).cast("double")
        dtm = ((end_epoch - time_epoch) / seconds_per_day).cast("double")
        duration = ((end_epoch - start_epoch) / seconds_per_day).cast("double")

        ratio = F.when(duration != F.lit(0.0), tenure / duration).otherwise(null_d)
        progress = F.when(ratio < F.lit(0.0), F.lit(0.0)) \
            .when(ratio > F.lit(2.0), F.lit(2.0)) \
            .otherwise(ratio).cast("double")

        bucket = (
            F.when(dtm.isNull(), null_d)
            .when(dtm < F.lit(-90.0), F.lit(0.0))
            .when(dtm < F.lit(-30.0), F.lit(1.0))
            .when(dtm < F.lit(-7.0), F.lit(2.0))
            .when(dtm < F.lit(0.0), F.lit(3.0))
            .when(dtm < F.lit(7.0), F.lit(4.0))
            .when(dtm < F.lit(30.0), F.lit(5.0))
            .when(dtm < F.lit(90.0), F.lit(6.0))
            .otherwise(F.lit(7.0))
        ).cast("double")

        w7 = F.when(F.abs(dtm) <= F.lit(7.0), F.lit(1.0)).otherwise(F.lit(0.0))
        w30 = F.when(F.abs(dtm) <= F.lit(30.0), F.lit(1.0)).otherwise(F.lit(0.0))
        w90 = F.when(F.abs(dtm) <= F.lit(90.0), F.lit(1.0)).otherwise(F.lit(0.0))

        if start_col in mask_set or end_col in mask_set:
            masks = []
            if start_col in mask_set:
                masks.append(start_ts > time_ts)
            if end_col in mask_set:
                masks.append(end_ts > time_ts)
            future_mask = masks[0]
            for m in masks[1:]:
                future_mask = future_mask | m
            tenure = F.when(future_mask, null_d).otherwise(tenure)
            dtm = F.when(future_mask, null_d).otherwise(dtm)
            w7 = F.when(future_mask, null_d).otherwise(w7)
            w30 = F.when(future_mask, null_d).otherwise(w30)
            w90 = F.when(future_mask, null_d).otherwise(w90)
            bucket = F.when(future_mask, null_d).otherwise(bucket)
            progress = F.when(future_mask, null_d).otherwise(progress)

        names, _ = _milestone_feature_names(start_col, end_col)
        select_exprs.extend([
            tenure.alias(names["tenure"]),
            dtm.alias(names["dtm"]),
            w7.alias(names["w7"]),
            w30.alias(names["w30"]),
            w90.alias(names["w90"]),
            bucket.alias(names["bucket"]),
            progress.alias(names["progress"]),
        ])

    result = spark_df.select(*select_exprs)
    return _as_pandas_api(result) if was_pandas_api else result


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
