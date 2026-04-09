from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

import pandas as _pandas

from .detection import (
    configure_spark_pandas,
    connect_remote_spark,
    enable_arrow_optimization,
    get_dbutils,
    get_display_function,
    get_spark_session,
    is_databricks,
    is_notebook,
    is_pandas_api_on_spark,
    is_remote_spark,
    is_spark_available,
    set_spark_config,
    use_distributed_processing,
)
from .ops import DataOps, ops
from .remote_path import RemotePath, make_path

_SPARK_PANDAS_AVAILABLE = is_spark_available()

_DATAFRAME_TYPES: tuple[type, ...] = (_pandas.DataFrame,)

if _SPARK_PANDAS_AVAILABLE:
    try:
        import pyspark.pandas as ps

        pd = ps
        DataFrame = Union[ps.DataFrame, _pandas.DataFrame]
        Series = Union[ps.Series, _pandas.Series]
        _DATAFRAME_TYPES = (_pandas.DataFrame, ps.DataFrame)
    except Exception:
        try:
            import databricks.koalas as ps

            pd = ps
            DataFrame = Union[ps.DataFrame, _pandas.DataFrame]
            Series = Union[ps.Series, _pandas.Series]
            _DATAFRAME_TYPES = (_pandas.DataFrame, ps.DataFrame)
        except Exception:
            _SPARK_PANDAS_AVAILABLE = False
            pd = _pandas
            DataFrame = _pandas.DataFrame
            Series = _pandas.Series
else:
    pd = _pandas
    DataFrame = _pandas.DataFrame
    Series = _pandas.Series

_SparkDF: type | None = None
try:
    from pyspark.sql import DataFrame as _SparkDF

    _DATAFRAME_TYPES = (*_DATAFRAME_TYPES, _SparkDF)
except ImportError:
    pass


def _activate_spark_pandas() -> None:
    global pd, DataFrame, Series, _SPARK_PANDAS_AVAILABLE, _DATAFRAME_TYPES
    import sys

    try:
        import pyspark.pandas as _ps

        pd = _ps
        DataFrame = Union[_ps.DataFrame, _pandas.DataFrame]
        Series = Union[_ps.Series, _pandas.Series]
        _DATAFRAME_TYPES = (_pandas.DataFrame, _ps.DataFrame)
        _SPARK_PANDAS_AVAILABLE = True
    except Exception:
        return
    try:
        from pyspark.sql import DataFrame as _SparkDF

        _DATAFRAME_TYPES = (*_DATAFRAME_TYPES, _SparkDF)
    except ImportError:
        pass
    this = sys.modules[__name__]
    this.pd = pd
    this.DataFrame = DataFrame
    this.Series = Series


def is_dataframe(obj: Any) -> bool:
    return isinstance(obj, _DATAFRAME_TYPES)


def to_pandas(df: Any) -> Any:
    if isinstance(df, (_pandas.DataFrame, _pandas.Series)):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    try:
        from pyspark.sql import DataFrame as NativeSparkDF

        if isinstance(df, NativeSparkDF):
            return df.toPandas()
    except ImportError:
        pass
    return _pandas.DataFrame(df)


def concat(objs: list, axis: int = 0, ignore_index: bool = False, **kwargs: Any) -> Any:
    if not objs:
        return pd.DataFrame()
    if _SPARK_PANDAS_AVAILABLE and isinstance(objs[0], (_pandas.DataFrame, _pandas.Series)):
        return _pandas.concat(objs, axis=axis, ignore_index=ignore_index, **kwargs)
    return pd.concat(objs, axis=axis, ignore_index=ignore_index, **kwargs)


def merge(left: Any, right: Any, how: str = "inner", on: Any = None, **kwargs: Any) -> Any:
    return pd.merge(left, right, how=how, on=on, **kwargs)


native_pd = _pandas

Timestamp = _pandas.Timestamp
Timedelta = _pandas.Timedelta
DatetimeIndex = _pandas.DatetimeIndex
CategoricalDtype = _pandas.CategoricalDtype
NA = _pandas.NA
NaT = _pandas.NaT

# Utility functions — dispatch to pyspark.pandas when input is distributed
to_timedelta = _pandas.to_timedelta
get_dummies = _pandas.get_dummies
crosstab = _pandas.crosstab
notna = _pandas.notna
isna = _pandas.isna


def resolve_column_name(columns: Any, name: str) -> str:
    if name in columns:
        return name
    lower = name.lower()
    for c in columns:
        if c.lower() == lower:
            return c
    raise KeyError(f"Column '{name}' not found (tried case-insensitive). Available: {list(columns)}")


def _is_native_pandas(obj: Any) -> bool:
    return isinstance(obj, (_pandas.DataFrame, _pandas.Series))


def _collect(obj: Any) -> Any:
    return obj.to_pandas() if hasattr(obj, "to_pandas") else obj


def to_datetime(arg: Any, **kwargs: Any) -> Any:
    if _SPARK_PANDAS_AVAILABLE and not _is_native_pandas(arg):
        return pd.to_datetime(arg, **kwargs)
    return _pandas.to_datetime(arg, **kwargs)


def to_numeric(arg: Any, **kwargs: Any) -> Any:
    if _SPARK_PANDAS_AVAILABLE and not _is_native_pandas(arg):
        return pd.to_numeric(arg, **kwargs)
    return _pandas.to_numeric(arg, **kwargs)


def cut(x: Any, bins: Any, **kwargs: Any) -> Any:
    if _SPARK_PANDAS_AVAILABLE and not _is_native_pandas(x):
        return _pandas.cut(_collect(x), bins, **kwargs)
    return _pandas.cut(x, bins, **kwargs)


def qcut(x: Any, q: Any, **kwargs: Any) -> Any:
    if _SPARK_PANDAS_AVAILABLE and not _is_native_pandas(x):
        return _pandas.qcut(_collect(x), q, **kwargs)
    return _pandas.qcut(x, q, **kwargs)


api_types = _pandas.api.types


def _extract_dtype(arr_or_dtype: Any) -> Any:
    return arr_or_dtype.dtype if hasattr(arr_or_dtype, "dtype") else arr_or_dtype


def is_numeric_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_numeric_dtype(_extract_dtype(arr_or_dtype))


def is_string_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_string_dtype(_extract_dtype(arr_or_dtype))


def is_datetime64_any_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_datetime64_any_dtype(_extract_dtype(arr_or_dtype))


def is_bool_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_bool_dtype(_extract_dtype(arr_or_dtype))


def is_categorical_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_categorical_dtype(_extract_dtype(arr_or_dtype))


def is_integer_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_integer_dtype(_extract_dtype(arr_or_dtype))


def is_float_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_float_dtype(_extract_dtype(arr_or_dtype))


def is_extension_array_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_extension_array_dtype(_extract_dtype(arr_or_dtype))


def _infer_epoch_unit(value: int) -> str:
    """Infer the epoch unit from a representative integer timestamp value.

    Spark LongType timestamps become int64 after ``to_pandas()``.  The bare
    ``pd.to_datetime()`` call assumes nanoseconds for large integers, which
    silently produces wrong dates when the source used seconds or milliseconds.
    This helper picks the right ``unit`` based on magnitude.
    """
    abs_val = abs(int(value))
    if abs_val > 1e17:
        return "ns"
    if abs_val > 1e14:
        return "us"
    if abs_val > 1e11:
        return "ms"
    return "s"


def safe_memory_usage_bytes(obj: Any) -> int:
    """Return memory usage in bytes, returning 0 when unsupported (e.g. PySpark)."""
    try:
        usage = obj.memory_usage(deep=True)
        return int(usage.sum()) if hasattr(usage, "sum") else int(usage)
    except Exception:
        return 0


def safe_to_datetime(series: Any, **kwargs: Any) -> Any:
    if _is_spark_pandas(series):
        import pyspark.sql.functions as F  # noqa: N812

        dtype_str = str(series.dtype).lower()
        if "timestamp" in dtype_str or "datetime" in dtype_str:
            return series
        if "int" in dtype_str or "long" in dtype_str:
            return series.spark.transform(lambda c: F.from_unixtime(c).cast("timestamp"))
        return series.spark.transform(lambda c: F.to_timestamp(c))
    if _pandas.api.types.is_datetime64_any_dtype(series):
        return as_tz_naive(series if isinstance(series, _pandas.Series) else _pandas.Series(series))
    arr = series.to_numpy() if hasattr(series, "to_numpy") else _pandas.array(series)
    if _pandas.api.types.is_integer_dtype(arr) or _pandas.api.types.is_integer_dtype(series):
        arr = _pandas.to_numeric(arr, errors="coerce")
        non_null = arr[~_pandas.isna(arr)]
        if len(non_null) > 0:
            unit = _infer_epoch_unit(non_null[0])
            return _pandas.Series(_pandas.to_datetime(arr, unit=unit, **kwargs))
    try:
        result = _pandas.Series(_pandas.to_datetime(arr, **kwargs))
    except ValueError:
        result = _pandas.Series(_pandas.to_datetime(arr, format="ISO8601", **kwargs))
    return as_tz_naive(result)


def ensure_datetime_column(df: _pandas.DataFrame, column: str) -> _pandas.DataFrame:
    if not _pandas.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = safe_to_datetime(df[column])
    return df


def ensure_timestamp(df: Any, column: str) -> Any:
    if _is_spark_pandas(df):
        import pyspark.sql.functions as F  # noqa: N812

        dtype_str = str(df[column].dtype).lower()
        if "timestamp" in dtype_str or "datetime" in dtype_str:
            return df
        df[column] = df[column].spark.transform(lambda c: F.to_timestamp(c))
        return df
    return ensure_datetime_column(df, column)


def as_tz_naive(value: Any) -> Any:
    if isinstance(value, _pandas.Series):
        if hasattr(value.dtype, "tz") and value.dtype.tz is not None:
            return value.dt.tz_localize(None)
        return value
    if isinstance(value, _pandas.DatetimeIndex):
        return value.tz_localize(None) if value.tz else value
    if hasattr(value, "tzinfo") and value.tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def _normalize_timestamp_columns(df: _pandas.DataFrame) -> _pandas.DataFrame:
    import datetime as _dt
    import decimal as _decimal

    df = df.copy()
    for col in df.columns:
        if _pandas.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = as_tz_naive(df[col])
        elif df[col].dtype == "object":
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample = non_null.iloc[:10].to_numpy()
                if all(isinstance(v, _dt.date) for v in sample):
                    df[col] = _pandas.to_datetime(df[col])
                elif all(isinstance(v, _decimal.Decimal) for v in sample):
                    df[col] = _pandas.to_numeric(df[col], errors="coerce")
    return df


def normalize_timestamps(df: Any) -> Any:
    if _is_spark_pandas(df):
        return _normalize_timestamps_distributed(df)
    return _normalize_timestamp_columns(df)


def normalize_decimal_columns(df: Any) -> Any:
    """Cast Decimal-typed columns to float64/double.

    PySpark ``DecimalType`` values are ``decimal.Decimal`` in Python, which
    cannot be multiplied by ``float``.  This causes failures in numpy
    quantile / percentile operations.  Call at profiling entry points.
    """
    if _is_spark_pandas(df):
        return _normalize_decimal_distributed(df)
    return _normalize_decimal_pandas(df)


def _normalize_decimal_pandas(df: _pandas.DataFrame) -> _pandas.DataFrame:
    import decimal as _decimal

    for col in df.columns:
        if df[col].dtype == "object":
            non_null = df[col].dropna()
            if len(non_null) > 0 and isinstance(non_null.iloc[0], _decimal.Decimal):
                df = df.copy()
                df[col] = _pandas.to_numeric(df[col], errors="coerce")
    return df


def _normalize_decimal_distributed(df: Any) -> Any:
    from pyspark.sql.types import DecimalType

    spark_df = as_spark_df(df)
    decimal_cols = {f.name for f in spark_df.schema.fields if isinstance(f.dataType, DecimalType)}
    if not decimal_cols:
        return df
    from pyspark.sql.functions import col as spark_col

    casts = [
        spark_col(f.name).cast("double").alias(f.name) if f.name in decimal_cols else spark_col(f.name)
        for f in spark_df.schema.fields
    ]
    from .spark_backend import _as_pandas_api

    return _as_pandas_api(spark_df.select(casts))


def clamp_distributed_timestamps(df: Any) -> Any:
    """Null out extreme timestamp values in a pyspark.pandas DataFrame.

    Must be called BEFORE ``.to_spark()`` so that the clamp expression is
    part of the internal Spark plan and is evaluated before any Arrow
    serialization that Spark Connect may trigger.
    """
    if not _is_spark_pandas(df):
        return df
    import pyspark.sql.functions as F  # noqa: N812

    for col_name in list(df.columns):
        dtype_str = str(df[col_name].dtype).lower()
        if "datetime" in dtype_str or "timestamp" in dtype_str:
            df[col_name] = df[col_name].spark.transform(lambda c: F.when(F.year(c).between(1678, 2261), c))
    return df


def strip_spark_timestamp_tz(spark_df: Any) -> Any:
    from pyspark.sql.functions import col as spark_col
    from pyspark.sql.types import TimestampType

    ts_fields = [f for f in spark_df.schema.fields if isinstance(f.dataType, TimestampType)]
    if not ts_fields:
        return spark_df
    casts = [
        spark_col(f.name).cast("timestamp_ntz").alias(f.name)
        if isinstance(f.dataType, TimestampType)
        else spark_col(f.name)
        for f in spark_df.schema.fields
    ]
    return spark_df.select(casts)


def sanitize_spark_timestamps(spark_df: Any) -> Any:
    """Strip timezone and clamp extreme timestamps on a native Spark DataFrame.

    Apply at system boundaries (data load) BEFORE wrapping as pyspark.pandas
    to prevent Arrow serialization failures in downstream Python UDFs.
    """
    if not hasattr(spark_df, "schema"):
        return spark_df
    return clamp_spark_timestamps(strip_spark_timestamp_tz(spark_df))


def clamp_spark_timestamps(spark_df: Any) -> Any:
    from pyspark.sql.functions import col as spark_col
    from pyspark.sql.functions import when, year
    from pyspark.sql.types import TimestampNTZType, TimestampType

    ts_names = {f.name for f in spark_df.schema.fields if isinstance(f.dataType, (TimestampType, TimestampNTZType))}
    if not ts_names:
        return spark_df
    cols = [
        when(
            year(spark_col(f.name)).between(1678, 2261),
            spark_col(f.name).cast("timestamp_ntz"),
        ).alias(f.name)
        if f.name in ts_names
        else spark_col(f.name)
        for f in spark_df.schema.fields
    ]
    return spark_df.select(cols)


def _normalize_timestamps_distributed(df: Any) -> Any:
    spark_df = df.to_spark()
    stripped = strip_spark_timestamp_tz(spark_df)
    clamped = clamp_spark_timestamps(stripped)
    if clamped is spark_df:
        return df
    return as_pandas_api(clamped)


def _infer_object_column_spark_type(series: _pandas.Series) -> "Any":
    from pyspark.sql.types import BooleanType, DoubleType, LongType, StringType, TimestampNTZType

    inferred = _pandas.api.types.infer_dtype(series, skipna=True)
    _INFERRED_MAP = {
        "floating": DoubleType(),
        "mixed-integer-float": DoubleType(),
        "decimal": DoubleType(),
        "integer": LongType(),
        "boolean": BooleanType(),
        "datetime64": TimestampNTZType(),
        "datetime": TimestampNTZType(),
    }
    return _INFERRED_MAP.get(inferred, StringType())


def pandas_dtype_to_spark_schema(df: _pandas.DataFrame) -> "Any":
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampNTZType,
    )

    _MAP = {
        "int8": IntegerType(),
        "int16": IntegerType(),
        "int32": IntegerType(),
        "int64": LongType(),
        "float32": FloatType(),
        "float64": DoubleType(),
        "bool": BooleanType(),
    }
    fields = []
    for col in df.columns:
        dtype = df[col].dtype
        if _pandas.api.types.is_datetime64_any_dtype(dtype):
            spark_type = TimestampNTZType()
        elif hasattr(dtype, "numpy_dtype"):
            spark_type = _MAP.get(str(dtype.numpy_dtype), StringType())
        elif str(dtype) == "object":
            spark_type = _infer_object_column_spark_type(df[col])
        else:
            spark_type = _MAP.get(str(dtype), StringType())
        fields.append(StructField(col, spark_type, True))
    return StructType(fields)


_SECONDS_PER_DAY = 86400


def timedelta_to_days(series: Any) -> Any:
    if is_numeric_dtype(series):
        return series // _SECONDS_PER_DAY
    try:
        return series.dt.days
    except AttributeError:
        return series.astype("long") // (_SECONDS_PER_DAY * 1_000_000)


def timedelta_to_seconds(series: Any) -> Any:
    if is_numeric_dtype(series):
        return series.astype(float)
    try:
        return series.dt.total_seconds()
    except AttributeError:
        return series.astype("long").astype(float) / 1_000_000


def _is_spark_pandas(obj: Any) -> bool:
    return hasattr(obj, "spark") or hasattr(obj, "to_spark")


def _is_native_spark_df(obj: Any) -> bool:
    return _SparkDF is not None and isinstance(obj, _SparkDF)


def as_spark_df(df: Any) -> Any:
    if not hasattr(df, "to_spark"):
        return df
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*index_col.*")
        return df.to_spark()


def timestamp_diff_seconds(a: Any, b: Any) -> Any:
    if _is_spark_pandas(a):
        import pyspark.sql.functions as F  # noqa: N812

        a_epoch = a.spark.transform(lambda c: F.unix_timestamp(c.cast("timestamp")).cast("double"))
        b_epoch = b.spark.transform(lambda c: F.unix_timestamp(c.cast("timestamp")).cast("double"))
        return a_epoch - b_epoch
    return timedelta_to_seconds(a - b)


def timestamp_diff_days(a: Any, b: Any) -> Any:
    return timestamp_diff_seconds(a, b) / _SECONDS_PER_DAY


def timestamp_diffs_seconds(series: Any) -> Any:
    if _is_spark_pandas(series):
        import pyspark.sql.functions as F  # noqa: N812

        epoch = series.spark.transform(lambda c: F.unix_timestamp(c.cast("timestamp")).cast("double"))
        return epoch - epoch.shift(1)
    return timedelta_to_seconds(series - series.shift(1))


_FREQ_TO_SPARK: dict[str, str] = {
    "D": "day",
    "W": "week",
    "M": "month",
    "ME": "month",
    "MS": "month",
    "Q": "quarter",
    "QS": "quarter",
    "Y": "year",
    "YS": "year",
}


def period_start_time(series: Any, freq: str) -> Any:
    if _is_spark_pandas(series):
        import pyspark.sql.functions as F  # noqa: N812

        spark_freq = _FREQ_TO_SPARK.get(freq, freq.lower())
        return series.spark.transform(lambda c: F.date_trunc(spark_freq, c))
    return series.dt.to_period(freq).dt.start_time


def groupby_multi_agg(df: Any, group_col: str, agg_col: str, agg_funcs: list) -> Any:
    if hasattr(df, "to_spark"):
        import pyspark.sql.functions as F  # noqa: N812

        spark_df = as_spark_df(df)
        exprs = [getattr(F, fn)(agg_col).alias(fn) for fn in agg_funcs]
        result = spark_df.groupBy(group_col).agg(*exprs)
        from .spark_backend import _as_pandas_api

        return _as_pandas_api(result)
    return df.groupby(group_col)[agg_col].agg(agg_funcs).reset_index()


def _to_native_list(obj: Any) -> list:
    if _is_spark_pandas(obj):
        raise TypeError("_to_native_list cannot be used with distributed pandas-on-Spark objects")

    if hasattr(obj, "tolist"):
        return obj.tolist()

    if hasattr(obj, "to_list"):
        return obj.to_list()

    return list(obj)


def _spark_head_values(series: Any, n: int) -> list:
    if n <= 0:
        return []
    spark_df = as_spark_df(series.to_frame() if hasattr(series, "to_frame") else series)
    col_name = spark_df.columns[0]
    rows = spark_df.select(_spark_col(col_name)).limit(n).collect()
    return [row[col_name] for row in rows]


def head_as_list(obj: Any, n: int) -> list:
    if n <= 0:
        return []

    series_obj = obj.to_series() if hasattr(obj, "to_series") and not _is_spark_pandas(obj) else obj
    if _is_spark_pandas(series_obj):
        return _spark_head_values(series_obj, n)

    if hasattr(obj, "head"):
        bounded = obj.head(n)
    elif hasattr(obj, "to_series"):
        bounded = obj.to_series().head(n)
    else:
        bounded = obj[:n]
    return _to_native_list(bounded)


def unique_overlap_counts(series_a: Any, series_b: Any) -> tuple[int, int, int]:
    if _is_spark_pandas(series_a) and _is_spark_pandas(series_b):
        return _spark_unique_overlap(series_a, series_b)
    left = set(_to_native_list(series_a.dropna().unique()))
    right = set(_to_native_list(series_b.dropna().unique()))
    return len(left), len(right), len(left & right)


def _spark_unique_overlap(series_a: Any, series_b: Any) -> tuple[int, int, int]:
    col_a = series_a.name or "__a__"
    col_b = series_b.name or "__b__"
    spark_a = as_spark_df(series_a.dropna().to_frame()).select(_spark_col(col_a).alias("__key__")).distinct()
    spark_b = as_spark_df(series_b.dropna().to_frame()).select(_spark_col(col_b).alias("__key__")).distinct()
    left_count = spark_a.count()
    right_count = spark_b.count()
    overlap_count = spark_a.join(spark_b, on="__key__", how="inner").count()
    return left_count, right_count, overlap_count


def _spark_col(name: str) -> Any:
    import pyspark.sql.functions as F  # noqa: N812

    return F.col(name)


def distributed_case_variations(str_series: Any, max_results: int = 10) -> list[str]:
    if _is_spark_pandas(str_series):
        return _spark_case_variations(str_series, max_results)
    unique_values = str_series.dropna().astype(str).drop_duplicates()
    if len(unique_values) == 0:
        return []
    grouped = native_pd.DataFrame({"original": unique_values.to_numpy()})
    grouped["lower"] = grouped["original"].str.lower()
    lower_to_originals: dict[str, list[str]] = {}
    for _, group in grouped.groupby("lower", sort=False):
        variants = group["original"].tolist()
        lower_to_originals[group["lower"].iloc[0]] = variants
    variations = []
    for originals in lower_to_originals.values():
        if len(originals) > 1:
            variations.append(f"{originals[0]} vs {originals[1]}")
    return variations[:max_results]


def _spark_case_variations(str_series: Any, max_results: int) -> list[str]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(str_series.astype(str).to_frame())
    col_name = spark_df.columns[0]
    grouped = (
        spark_df.groupBy(F.lower(F.col(col_name)).alias("__lower__"))
        .agg(F.collect_set(F.col(col_name)).alias("__variants__"))
        .filter(F.size(F.col("__variants__")) > 1)
        .limit(max_results)
        .collect()
    )
    return [f"{row['__variants__'][0]} vs {row['__variants__'][1]}" for row in grouped]


def _spark_safe_query_expr(expr: str) -> str:
    import re

    return re.sub(r"\bin\s*\[([^\]]*)\]", r"in (\1)", expr)


def safe_query(df: Any, expr: str) -> Any:
    if _is_spark_pandas(df):
        return df.query(_spark_safe_query_expr(expr))
    return df.query(expr)


_ISIN_JOIN_THRESHOLD = 100_000


def safe_isin(df: Any, column: str, values: Any, negate: bool = False) -> Any:
    values_list = list(values) if not isinstance(values, list) else values
    if not _is_spark_pandas(df) or len(values_list) <= _ISIN_JOIN_THRESHOLD:
        mask = df[column].isin(values_list)
        return df[~mask] if negate else df[mask]
    spark_df = as_spark_df(df)
    values_df = spark_df.sparkSession.createDataFrame(
        [(v,) for v in values_list],
        [column],
    )
    join_type = "left_anti" if negate else "left_semi"
    result = spark_df.join(values_df, on=column, how=join_type)
    return result.pandas_api()


def safe_isinf(series: Any) -> Any:
    return (series == float("inf")) | (series == float("-inf"))


def safe_isfinite(series: Any) -> Any:
    return series.notna() & ~safe_isinf(series)


def safe_drop_duplicates(df: Any, subset: list[str], keep: str = "last") -> Any:
    if _is_spark_pandas(df):
        return df.drop_duplicates(subset=subset)
    if _is_native_spark_df(df):
        return df.dropDuplicates(subset)
    return df.drop_duplicates(subset=subset, keep=keep)


def safe_len(df: Any) -> int:
    """Row count that does NOT trigger a per-call Spark action inside loops.

    On pyspark.pandas ``len(df)`` calls ``.__len__()`` which fires a full
    ``.count()`` Spark job every time.  This helper caches nothing but is
    clearly named so callers store the result in a local variable.
    """
    if _is_spark_pandas(df):
        return int(as_spark_df(df).count())
    return len(df)


def safe_sample(df: Any, n: int, random_state: int = 42) -> Any:
    if n <= 0:
        return df.head(0)
    total = safe_len(df)
    n = min(n, total)
    if n == 0:
        return df.head(0)
    if _is_spark_pandas(df):
        frac = min(1.0, n / max(1, total))
        result = df.sample(frac=frac, random_state=random_state)
        return result.head(n) if safe_len(result) > n else result
    return df.sample(n=n, random_state=random_state)


def safe_select(conditions: list, choices: list, default: Any = "") -> Any:
    import numpy as np

    if not conditions or not any(_is_spark_pandas(c) for c in conditions):
        return np.select(conditions, choices, default=default)
    from pyspark.sql import functions as F  # noqa: N812

    expr = F.lit(default)
    for cond, label in reversed(list(zip(conditions, choices))):
        expr = F.when(cond.spark.column, F.lit(label)).otherwise(expr)
    return conditions[0].spark.transform(lambda _: expr)


def safe_describe(df: Any) -> Any:
    try:
        return df.describe()
    except Exception:
        numeric = df.select_dtypes(include="number")
        if len(numeric.columns) == 0:
            return _pandas.DataFrame()
        return numeric.describe()


def temporal_quantile(series: Any, q: float) -> _pandas.Timestamp:
    if _is_spark_pandas(series):
        return _spark_temporal_quantile(series, q)
    return series.quantile(q)


def _spark_temporal_quantile(series: Any, q: float) -> _pandas.Timestamp:
    from pyspark.sql import functions as F  # noqa: N812

    spark_df = as_spark_df(series.to_frame())
    col_name = spark_df.columns[0]
    epoch = F.unix_timestamp(F.col(col_name).cast("timestamp"))
    row = spark_df.select(F.percentile_approx(epoch, float(q)).alias("v")).head()
    if row["v"] is None:
        return _pandas.NaT
    return _pandas.Timestamp(row["v"], unit="s")


def _spark_numeric_cols(spark_df: Any, columns: list[str] | None = None) -> list[str]:
    from pyspark.sql.types import NumericType

    allowed = set(columns) if columns is not None else None
    return [
        f.name
        for f in spark_df.schema.fields
        if isinstance(f.dataType, NumericType) and (allowed is None or f.name in allowed)
    ]


_STACK_CHUNK = 2000


def _spark_stacked(spark_df: Any, columns: list[str]) -> Any:
    from functools import reduce

    chunks = []
    for start in range(0, len(columns), _STACK_CHUNK):
        batch = columns[start : start + _STACK_CHUNK]
        stack_args = ", ".join(f"'{c}', cast(`{c}` as double)" for c in batch)
        chunks.append(
            spark_df.selectExpr(
                f"stack({len(batch)}, {stack_args}) as (__col_name, __col_value)",
            )
        )
    return reduce(lambda a, b: a.unionAll(b), chunks)


def _spark_stacked_with_extra(spark_df: Any, columns: list[str], extra_col: str) -> Any:
    from functools import reduce

    chunks = []
    for start in range(0, len(columns), _STACK_CHUNK):
        batch = columns[start : start + _STACK_CHUNK]
        stack_args = ", ".join(f"'{c}', cast(`{c}` as double)" for c in batch)
        chunks.append(
            spark_df.selectExpr(
                f"cast(`{extra_col}` as double) as `{extra_col}`",
                f"stack({len(batch)}, {stack_args}) as (__col_name, __col_value)",
            )
        )
    return reduce(lambda a, b: a.unionAll(b), chunks)


def _numeric_column_names(df: Any, columns: list[str]) -> set[str]:
    if _is_spark_pandas(df):
        return set(_spark_numeric_cols(as_spark_df(df), columns))
    return set(df[columns].select_dtypes(include="number").columns)


_ML_CORR_THRESHOLD = 100
_CORR_BLOCK_SIZE = 500
_AGG_BATCH_SIZE = 500
_CORR_SAMPLE_ROWS = 50_000


def bulk_variance(df: Any, columns: list[str]) -> _pandas.Series:
    """Compute variance per column. Batched Spark SQL on Databricks."""
    if not columns:
        return _pandas.Series(dtype=float)
    if not _is_spark_pandas(df):
        return df[columns].var()
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df[columns])
    safe = {c: f"__v{i}__" for i, c in enumerate(columns)}
    spark_df = spark_df.toDF(*[safe[c] for c in columns])
    result: dict[str, float] = {}
    for start in range(0, len(columns), _AGG_BATCH_SIZE):
        batch = columns[start : start + _AGG_BATCH_SIZE]
        safe_batch = [safe[c] for c in batch]
        row = spark_df.agg(*[F.variance(F.col(s).cast("double")).alias(s) for s in safe_batch]).head()
        for c, s in zip(batch, safe_batch):
            val = row[s]
            result[c] = float(val) if val is not None else float("nan")
    return _pandas.Series(result)


def batched_corr_matrix(
    df: Any,
    columns: list[str],
    progress_fn: Any = None,
    precomputed_medians: dict[str, float] | None = None,
    precomputed_non_null: dict[str, int] | None = None,
    cross_columns: list[str] | None = None,
) -> _pandas.DataFrame:
    import numpy as _np

    valid_cols = [c for c in columns if c in df.columns]
    if len(valid_cols) < 2:
        return _pandas.DataFrame(index=valid_cols, columns=valid_cols)
    numeric = _numeric_column_names(df, valid_cols)
    num_cols = [c for c in valid_cols if c in numeric]
    if not _is_spark_pandas(df):
        corr = df[num_cols].corr() if len(num_cols) >= 2 else _pandas.DataFrame(index=num_cols, columns=num_cols)
        full = _pandas.DataFrame(_np.nan, index=valid_cols, columns=valid_cols)
        full.loc[corr.index, corr.columns] = corr
        return full
    if len(num_cols) > _ML_CORR_THRESHOLD:
        return _spark_corr_matrix_ml(df, valid_cols, numeric, progress_fn, precomputed_medians, precomputed_non_null, cross_columns)
    return _spark_pairwise_corr(df, valid_cols, numeric, cross_columns)


def _spark_corr_matrix_ml(
    df: Any,
    cols: list[str],
    numeric: set[str],
    progress_fn: Any = None,
    precomputed_medians: dict[str, float] | None = None,
    precomputed_non_null: dict[str, int] | None = None,
    cross_columns: list[str] | None = None,
) -> _pandas.DataFrame:
    import time as _time

    import numpy as _np
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation

    log = progress_fn or (lambda msg: None)
    num_cols = [c for c in cols if c in numeric]
    if len(num_cols) < 2:
        return _pandas.DataFrame(_np.nan, index=cols, columns=cols)

    spark_df = as_spark_df(df[num_cols])
    safe = {c: f"__f_{i}__" for i, c in enumerate(num_cols)}
    spark_df = spark_df.select([F.col(f"`{c}`").cast("double").alias(safe[c]) for c in num_cols])
    safe_cols = list(safe.values())

    # Non-null counts — use precomputed if available
    rev = {v: k for k, v in safe.items()}
    if precomputed_non_null and all(rev[s] in precomputed_non_null for s in safe_cols):
        log(f"    Using precomputed non-null counts ({len(safe_cols)} columns)")
        non_null = {s: precomputed_non_null[rev[s]] for s in safe_cols}
    else:
        log(f"    Counting non-nulls ({len(safe_cols)} columns)...")
        non_null = {}
        for start in range(0, len(safe_cols), _AGG_BATCH_SIZE):
            batch = safe_cols[start : start + _AGG_BATCH_SIZE]
            row = spark_df.agg(*[F.count(c).alias(c) for c in batch]).head()
            for c in batch:
                non_null[c] = row[c]
    valid_safe = [c for c in safe_cols if non_null.get(c, 0) > 0]
    if len(valid_safe) < 2:
        return _pandas.DataFrame(_np.nan, index=cols, columns=cols)
    if len(valid_safe) < len(safe_cols):
        spark_df = spark_df.select(valid_safe)

    # Sample large datasets to bound O(rows × features²) time
    total_rows = spark_df.count()
    if total_rows > _CORR_SAMPLE_ROWS:
        fraction = _CORR_SAMPLE_ROWS / total_rows
        spark_df = spark_df.sample(fraction=fraction, seed=42)
        log(f"    Sampled ~{_CORR_SAMPLE_ROWS:,} of {total_rows:,} rows")

    # Medians for imputation — use precomputed if available
    fill_map: dict[str, float] = {}
    if precomputed_medians:
        covered = {s for s in valid_safe if rev[s] in precomputed_medians}
        missing = [s for s in valid_safe if s not in covered]
        for s in covered:
            fill_map[s] = precomputed_medians[rev[s]]
        if missing:
            log(f"    Computing medians for {len(missing)} columns not in precomputed...")
            for start in range(0, len(missing), _AGG_BATCH_SIZE):
                batch = missing[start : start + _AGG_BATCH_SIZE]
                row = spark_df.agg(*[F.percentile_approx(c, 0.5).alias(c) for c in batch]).head()
                for c in batch:
                    if row[c] is not None:
                        fill_map[c] = float(row[c])
        else:
            log(f"    Using precomputed medians ({len(valid_safe)} columns)")
    else:
        log(f"    Computing medians ({len(valid_safe)} columns)...")
        for start in range(0, len(valid_safe), _AGG_BATCH_SIZE):
            batch = valid_safe[start : start + _AGG_BATCH_SIZE]
            row = spark_df.agg(*[F.percentile_approx(c, 0.5).alias(c) for c in batch]).head()
            for c in batch:
                if row[c] is not None:
                    fill_map[c] = float(row[c])
    imputed_df = spark_df.fillna(fill_map)

    # Block-wise correlation
    blocks = [valid_safe[i : i + _CORR_BLOCK_SIZE] for i in range(0, len(valid_safe), _CORR_BLOCK_SIZE)]
    n_blocks = len(blocks)
    n_valid = len(valid_safe)

    if n_blocks <= 1:
        assembler = VectorAssembler(inputCols=valid_safe, outputCol="__features__")
        vec_df = assembler.transform(imputed_df).select("__features__")
        corr_values = Correlation.corr(vec_df, "__features__", "pearson").head()[0].toArray()
    else:
        imputed_df.cache()
        cached_rows = imputed_df.count()
        if cross_columns is not None:
            safe_cross = {safe[c] for c in cross_columns if c in safe}
            cross_block_idx = {bi for bi, block in enumerate(blocks) if safe_cross.intersection(block)}
            block_pairs = [(i, j) for i in range(n_blocks) for j in range(i, n_blocks) if i in cross_block_idx or j in cross_block_idx]
        else:
            block_pairs = [(i, j) for i in range(n_blocks) for j in range(i, n_blocks)]
        total_pairs = len(block_pairs)
        log(f"    Correlation: {n_valid} features, {n_blocks} blocks, {total_pairs} pairs ({cached_rows:,} rows)")
        pos = {c: i for i, c in enumerate(valid_safe)}
        corr_values = _np.full((n_valid, n_valid), _np.nan)
        _np.fill_diagonal(corr_values, 1.0)
        for pair_idx, (bi, bj) in enumerate(block_pairs):
            t0 = _time.monotonic()
            block_cols = list(dict.fromkeys(blocks[bi] + blocks[bj]))
            asm = VectorAssembler(inputCols=block_cols, outputCol="__features__")
            vec_df = asm.transform(imputed_df.select(block_cols)).select("__features__")
            sub = Correlation.corr(vec_df, "__features__", "pearson").head()[0].toArray()
            if bi == bj:
                idx = _np.array([pos[c] for c in blocks[bi]])
                corr_values[_np.ix_(idx, idx)] = sub
            else:
                n_i = len(blocks[bi])
                idx_i = _np.array([pos[c] for c in blocks[bi]])
                idx_j = _np.array([pos[c] for c in blocks[bj]])
                corr_values[_np.ix_(idx_i, idx_j)] = sub[:n_i, n_i:]
                corr_values[_np.ix_(idx_j, idx_i)] = sub[:n_i, n_i:].T
            elapsed = _time.monotonic() - t0
            log(f"      Block [{pair_idx + 1}/{total_pairs}] ({len(block_cols)} cols) — {elapsed:.0f}s")
        imputed_df.unpersist()

    # Map back to original column names via vectorized numpy indexing
    full_idx = _np.array([cols.index(rev[s]) for s in valid_safe])
    n = len(cols)
    full = _np.full((n, n), _np.nan)
    full[_np.ix_(full_idx, full_idx)] = corr_values
    return _pandas.DataFrame(full, columns=cols, index=cols)


def _safe_corr_expr(col_a: Any, col_b: Any) -> Any:
    import pyspark.sql.functions as F  # noqa: N812

    denom = F.stddev_samp(col_a) * F.stddev_samp(col_b)
    return F.covar_samp(col_a, col_b) / F.when(denom > 0, denom)


def _spark_pairwise_corr(df: Any, cols: list[str], numeric: set[str], cross_columns: list[str] | None = None) -> _pandas.DataFrame:
    import numpy as _np
    import pyspark.sql.functions as F  # noqa: N812

    num_idx = {i for i, c in enumerate(cols) if c in numeric}
    spark_df = as_spark_df(df[[cols[i] for i in num_idx]] if len(num_idx) < len(cols) else df[cols])
    n = len(cols)
    matrix = _np.full((n, n), _np.nan)

    stddev_exprs = [F.stddev(cols[i]).alias(f"s_{i}") for i in num_idx]
    has_variance: set[int] = set()
    if stddev_exprs:
        stddev_row = spark_df.select(*stddev_exprs).head()
        for i in num_idx:
            val = stddev_row[f"s_{i}"]
            if val is not None and float(val) > 0:
                has_variance.add(i)
                matrix[i, i] = 1.0

    if cross_columns is not None:
        cross_idx = {i for i, c in enumerate(cols) if c in set(cross_columns)}
        pairs = [(i, j) for i in has_variance for j in has_variance if i < j and (i in cross_idx or j in cross_idx)]
    else:
        pairs = [(i, j) for i in has_variance for j in has_variance if i < j]
    _BATCH = 500
    for start in range(0, len(pairs), _BATCH):
        batch = pairs[start : start + _BATCH]
        exprs = [_safe_corr_expr(cols[i], cols[j]).alias(f"c_{i}_{j}") for i, j in batch]
        row = spark_df.select(*exprs).head()
        for i, j in batch:
            val = row[f"c_{i}_{j}"]
            val = float(val) if val is not None else _np.nan
            matrix[i, j] = val
            matrix[j, i] = val

    return _pandas.DataFrame(matrix, columns=cols, index=cols)


def bulk_corr_with_target(df: Any, columns: list[str], target_column: str, progress_fn: Any = None) -> dict[str, float]:
    import math

    if target_column not in df.columns:
        return {}
    valid = [c for c in columns if c in df.columns and c != target_column]
    if not valid:
        return {}
    if not _is_spark_pandas(df):
        return _pandas_corr_with_target(df, valid, target_column, progress_fn)
    return _spark_batched_corr_with_target(df, valid, target_column, progress_fn)


def _pandas_corr_with_target(
    df: Any, columns: list[str], target_column: str, progress_fn: Any = None
) -> dict[str, float]:
    import math

    log = progress_fn or (lambda msg: None)
    numeric = set(df[columns].select_dtypes(include="number").columns)
    num_cols = [c for c in columns if c in numeric]
    result: dict[str, float] = {c: math.nan for c in columns if c not in numeric}
    if not num_cols:
        return result
    target_float = df[target_column].astype(float)
    _CHUNK = 500
    total = (len(num_cols) + _CHUNK - 1) // _CHUNK
    for i, start in enumerate(range(0, len(num_cols), _CHUNK)):
        batch = num_cols[start : start + _CHUNK]
        corrs = df[batch].corrwith(target_float)
        result.update({c: float(v) if _pandas.notna(v) else math.nan for c, v in corrs.items()})
        log(f"    pandas corr batch {i + 1}/{total} ({len(batch)} cols)")
    return result


def _spark_batched_corr_with_target(
    df: Any, columns: list[str], target_column: str, progress_fn: Any = None
) -> dict[str, float]:
    import math
    import time as _time

    log = progress_fn or (lambda msg: None)
    numeric = _numeric_column_names(df, columns + [target_column])
    num_cols = [c for c in columns if c in numeric]
    result: dict[str, float] = {c: math.nan for c in columns if c not in numeric}
    if not num_cols or target_column not in numeric:
        result.update({c: math.nan for c in num_cols})
        return result
    spark_df = as_spark_df(df[num_cols + [target_column]])
    _BATCH = 500
    total_batches = (len(num_cols) + _BATCH - 1) // _BATCH
    t0 = _time.monotonic()
    for batch_idx, start in enumerate(range(0, len(num_cols), _BATCH)):
        batch = num_cols[start : start + _BATCH]
        exprs = [_safe_corr_expr(c, target_column).alias(f"c_{i}") for i, c in enumerate(batch)]
        row = spark_df.agg(*exprs).head()
        for i, c in enumerate(batch):
            val = row[f"c_{i}"]
            result[c] = float(val) if val is not None else math.nan
        log(f"    corr batch {batch_idx + 1}/{total_batches} ({len(batch)} cols, {_time.monotonic() - t0:.0f}s)")
    return result


def bulk_null_corr_with_target(
    df: Any,
    columns: list[str],
    target_column: str,
    progress_fn: Any = None,
    precomputed_null_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    if target_column not in df.columns:
        return {}
    valid = [c for c in columns if c in df.columns and c != target_column]
    if not valid:
        return {}
    if _is_spark_pandas(df):
        return _spark_null_corr_with_target(df, valid, target_column, progress_fn, precomputed_null_counts)
    return _pandas_null_corr_with_target(df, valid, target_column)


def _pandas_null_corr_with_target(df: Any, columns: list[str], target_column: str) -> dict[str, float]:
    import math

    target_float = df[target_column].astype(float)
    null_indicators = df[columns].isnull().astype(float)
    corrs = null_indicators.corrwith(target_float)
    return {c: float(v) if _pandas.notna(v) else math.nan for c, v in corrs.items()}


def _spark_null_corr_with_target(
    df: Any,
    columns: list[str],
    target_column: str,
    progress_fn: Any = None,
    precomputed_null_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    import math
    import time as _time

    import pyspark.sql.functions as F  # noqa: N812

    log = progress_fn or (lambda msg: None)
    spark_df = as_spark_df(df[columns + [target_column]])
    result: dict[str, float] = {}
    has_nulls: list[str] = []
    t0 = _time.monotonic()
    if precomputed_null_counts is not None and all(c in precomputed_null_counts for c in columns):
        for c in columns:
            if precomputed_null_counts[c] > 0:
                has_nulls.append(c)
            else:
                result[c] = math.nan
    else:
        _BATCH = 2000
        for start in range(0, len(columns), _BATCH):
            batch = columns[start : start + _BATCH]
            exprs = [F.sum(F.col(c).isNull().cast("long")).alias(c) for c in batch]
            row = spark_df.agg(*exprs).head()
            for c in batch:
                if (row[c] or 0) > 0:
                    has_nulls.append(c)
                else:
                    result[c] = math.nan
    log(f"    pre-filter: {len(has_nulls)}/{len(columns)} columns have nulls ({_time.monotonic() - t0:.0f}s)")
    if not has_nulls:
        return result
    _CORR_BATCH = 100
    total_batches = (len(has_nulls) + _CORR_BATCH - 1) // _CORR_BATCH
    for batch_idx, start in enumerate(range(0, len(has_nulls), _CORR_BATCH)):
        batch = has_nulls[start : start + _CORR_BATCH]
        t1 = _time.monotonic()
        null_exprs = [F.when(F.col(c).isNull(), 1.0).otherwise(0.0).alias(f"__null_{c}") for c in batch]
        batch_df = spark_df.select(*null_exprs, F.col(target_column))
        corr_exprs = [_safe_corr_expr(f"__null_{c}", target_column).alias(f"c_{i}") for i, c in enumerate(batch)]
        row = batch_df.agg(*corr_exprs).head()
        for i, c in enumerate(batch):
            val = row[f"c_{i}"]
            result[c] = float(val) if val is not None else math.nan
        log(f"    null-corr batch {batch_idx + 1}/{total_batches} ({len(batch)} cols, {_time.monotonic() - t1:.0f}s)")
    return result


def bulk_null_counts(df: Any, columns: list[str] | None = None, progress_fn: Any = None) -> dict[str, int]:
    from .bulk_profiling import bulk_null_counts as _bulk_null_counts

    return _bulk_null_counts(df, columns, progress_fn=progress_fn)


def _spark_batched_leakage_corr_combined(
    df: Any,
    columns: list[str],
    target_column: str,
    progress_fn: Any = None,
) -> tuple[dict[str, float], dict[str, float]]:
    import math
    import time as _time

    import pyspark.sql.functions as F  # noqa: N812

    log = progress_fn or (lambda msg: None)
    spark_df = as_spark_df(df[columns + [target_column]])
    null_corrs: dict[str, float] = {}
    value_corrs: dict[str, float] = {}
    _BATCH = 50
    total_batches = (len(columns) + _BATCH - 1) // _BATCH
    t0 = _time.monotonic()
    for batch_idx, start in enumerate(range(0, len(columns), _BATCH)):
        batch = columns[start : start + _BATCH]
        null_indicator_exprs = [F.when(F.col(c).isNull(), 1.0).otherwise(0.0).alias(f"__null_{c}") for c in batch]
        batch_df = spark_df.select(*null_indicator_exprs, *[F.col(c) for c in batch], F.col(target_column))
        corr_exprs = [
            *[_safe_corr_expr(f"__null_{c}", target_column).alias(f"n_{i}") for i, c in enumerate(batch)],
            *[_safe_corr_expr(c, target_column).alias(f"v_{i}") for i, c in enumerate(batch)],
        ]
        row = batch_df.agg(*corr_exprs).head()
        for i, c in enumerate(batch):
            null_corrs[c] = float(row[f"n_{i}"]) if row[f"n_{i}"] is not None else math.nan
            value_corrs[c] = float(row[f"v_{i}"]) if row[f"v_{i}"] is not None else math.nan
        log(
            f"    combined-corr batch {batch_idx + 1}/{total_batches} ({len(batch)} cols, {_time.monotonic() - t0:.0f}s)"
        )
    return null_corrs, value_corrs


def leakage_corr_combined(
    df: Any,
    columns: list[str],
    target_column: str,
    progress_fn: Any = None,
) -> tuple[dict[str, float], dict[str, float]]:
    if _is_spark_pandas(df):
        return _spark_batched_leakage_corr_combined(df, columns, target_column, progress_fn)
    null_corrs = bulk_null_corr_with_target(df, columns, target_column, progress_fn)
    value_corrs = bulk_corr_with_target(df, columns, target_column, progress_fn)
    return null_corrs, value_corrs


def bulk_skew(df: Any, columns: list[str]) -> dict[str, float]:
    import math

    valid = [c for c in columns if c in df.columns]
    if not valid:
        return {}
    if not _is_spark_pandas(df):
        numeric = set(df[valid].select_dtypes(include="number").columns)
        result: dict[str, float] = {c: math.nan for c in valid if c not in numeric}
        num_cols = [c for c in valid if c in numeric]
        if num_cols:
            skew_series = df[num_cols].skew()
            result.update({c: float(skew_series[c]) if _pandas.notna(skew_series[c]) else math.nan for c in num_cols})
        return result
    return _spark_bulk_skew(df, valid)


def _spark_bulk_skew(df: Any, columns: list[str]) -> dict[str, float]:
    import math

    import pyspark.sql.functions as F  # noqa: N812

    numeric = _numeric_column_names(df, columns)
    num_cols = [c for c in columns if c in numeric]
    result: dict[str, float] = {c: math.nan for c in columns if c not in numeric}
    if not num_cols:
        return result
    spark_df = as_spark_df(df[num_cols])
    _BATCH = 500
    for start in range(0, len(num_cols), _BATCH):
        batch = num_cols[start : start + _BATCH]
        exprs = [F.skewness(F.col(c)).alias(f"s_{i}") for i, c in enumerate(batch)]
        row = spark_df.select(*exprs).head()
        for i, c in enumerate(batch):
            val = row[f"s_{i}"]
            result[c] = float(val) if val is not None else math.nan
    return result


def bulk_max(df: Any, columns: list[str]) -> dict[str, Any]:
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return {}
    if not _is_spark_pandas(df):
        return {c: df[c].max() for c in valid}
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df[valid])
    row = spark_df.agg(*[F.max(F.col(c)).alias(c) for c in valid]).head()
    return {c: row[c] for c in valid if row[c] is not None}


def bulk_class_overlap(df: Any, columns: list[str], target: Any) -> dict[str, float]:
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return {}
    if not _is_spark_pandas(df):
        from customer_retention.core.utils.leakage import calculate_class_overlap

        return {c: calculate_class_overlap(df[c], target) for c in valid}
    return _spark_bulk_class_overlap(df, valid, target)


def _spark_bulk_class_overlap(df: Any, columns: list[str], target: Any) -> dict[str, float]:
    import pyspark.sql.functions as F  # noqa: N812

    _TARGET = "__overlap_target__"
    combined = df[columns].copy()
    combined[_TARGET] = target
    spark_df = as_spark_df(combined)
    _BATCH = 200
    result: dict[str, float] = {}
    for start in range(0, len(columns), _BATCH):
        batch = columns[start : start + _BATCH]
        exprs: list[Any] = []
        for c in batch:
            exprs.extend(
                [
                    F.min(F.when(F.col(_TARGET) == 0, F.col(c))).alias(f"{c}__min0"),
                    F.max(F.when(F.col(_TARGET) == 0, F.col(c))).alias(f"{c}__max0"),
                    F.min(F.when(F.col(_TARGET) == 1, F.col(c))).alias(f"{c}__min1"),
                    F.max(F.when(F.col(_TARGET) == 1, F.col(c))).alias(f"{c}__max1"),
                ]
            )
        row = spark_df.agg(*exprs).collect()[0]
        for c in batch:
            vals = [row[f"{c}__{k}"] for k in ("min0", "max0", "min1", "max1")]
            if any(v is None for v in vals):
                result[c] = 100.0
                continue
            min0, max0, min1, max1 = (float(v) for v in vals)
            total_range = max(max0, max1) - min(min0, min1)
            if total_range == 0:
                result[c] = 100.0
                continue
            overlap = max(0, min(max0, max1) - max(min0, min1))
            result[c] = (overlap / total_range) * 100
    return result


@dataclass
class BulkEffectSizeResult:
    effect_sizes: dict[str, float] = field(default_factory=dict)
    class_stats: dict[str, dict] = field(default_factory=dict)


def bulk_effect_sizes(df: Any, columns: list[str], target_column: str) -> BulkEffectSizeResult:
    if target_column not in df.columns:
        return BulkEffectSizeResult()
    valid = [c for c in columns if c in df.columns and c != target_column]
    if not valid:
        return BulkEffectSizeResult()
    if _is_spark_pandas(df):
        return _spark_bulk_effect_sizes(df, valid, target_column)
    return _pandas_bulk_effect_sizes(df, valid, target_column)


def _pandas_bulk_effect_sizes(
    df: Any,
    columns: list[str],
    target_column: str,
) -> BulkEffectSizeResult:
    import numpy as _np

    mask0 = df[target_column] == 0
    mask1 = df[target_column] == 1
    effect_sizes: dict[str, float] = {}
    class_stats: dict[str, dict] = {}

    for col in columns:
        g0 = df.loc[mask0, col].dropna()
        g1 = df.loc[mask1, col].dropna()
        n0, n1 = len(g0), len(g1)
        stats: dict[str, Any] = {
            "mean_0": float(g0.mean()) if n0 > 0 else 0.0,
            "std_0": float(g0.std()) if n0 > 1 else 0.0,
            "count_0": n0,
            "median_0": float(g0.median()) if n0 > 0 else 0.0,
            "mean_1": float(g1.mean()) if n1 > 0 else 0.0,
            "std_1": float(g1.std()) if n1 > 1 else 0.0,
            "count_1": n1,
            "median_1": float(g1.median()) if n1 > 0 else 0.0,
        }
        class_stats[col] = stats

        if n0 < 2 or n1 < 2:
            effect_sizes[col] = 0.0
            continue
        pooled_std = _np.sqrt(((n0 - 1) * stats["std_0"] ** 2 + (n1 - 1) * stats["std_1"] ** 2) / (n0 + n1 - 2))
        if pooled_std == 0:
            effect_sizes[col] = 0.0
        else:
            effect_sizes[col] = (stats["mean_1"] - stats["mean_0"]) / pooled_std

    return BulkEffectSizeResult(effect_sizes=effect_sizes, class_stats=class_stats)


def _spark_bulk_effect_sizes(
    df: Any,
    columns: list[str],
    target_column: str,
) -> BulkEffectSizeResult:
    import numpy as _np
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df[columns + [target_column]])
    _BATCH = 200
    effect_sizes: dict[str, float] = {}
    class_stats: dict[str, dict] = {}

    for start in range(0, len(columns), _BATCH):
        batch = columns[start : start + _BATCH]
        exprs: list[Any] = []
        for c in batch:
            for cls in (0, 1):
                cond = F.col(target_column) == cls
                filtered = F.when(cond, F.col(c))
                exprs.extend(
                    [
                        F.mean(filtered).alias(f"{c}__mean_{cls}"),
                        F.stddev(filtered).alias(f"{c}__std_{cls}"),
                        F.count(filtered).alias(f"{c}__count_{cls}"),
                        F.percentile_approx(filtered, 0.5).alias(f"{c}__median_{cls}"),
                    ]
                )
        row = spark_df.agg(*exprs).collect()[0]

        for c in batch:
            stats: dict[str, Any] = {}
            for cls in (0, 1):
                mean_val = row[f"{c}__mean_{cls}"]
                std_val = row[f"{c}__std_{cls}"]
                count_val = row[f"{c}__count_{cls}"]
                median_val = row[f"{c}__median_{cls}"]
                stats[f"mean_{cls}"] = float(mean_val) if mean_val is not None else 0.0
                stats[f"std_{cls}"] = float(std_val) if std_val is not None else 0.0
                stats[f"count_{cls}"] = int(count_val) if count_val is not None else 0
                stats[f"median_{cls}"] = float(median_val) if median_val is not None else 0.0
            class_stats[c] = stats

            n0, n1 = stats["count_0"], stats["count_1"]
            if n0 < 2 or n1 < 2:
                effect_sizes[c] = 0.0
                continue
            pooled_std = _np.sqrt(((n0 - 1) * stats["std_0"] ** 2 + (n1 - 1) * stats["std_1"] ** 2) / (n0 + n1 - 2))
            if pooled_std == 0:
                effect_sizes[c] = 0.0
            else:
                effect_sizes[c] = (stats["mean_1"] - stats["mean_0"]) / pooled_std

    return BulkEffectSizeResult(effect_sizes=effect_sizes, class_stats=class_stats)


def spark_checkpoint(df: Any) -> Any:
    if _is_spark_pandas(df):
        spark_df = as_spark_df(df).localCheckpoint(eager=True)
        from .spark_backend import _as_pandas_api

        return _as_pandas_api(spark_df)
    return df


def spark_persist(df: Any) -> Any:
    """Persist distributed DataFrame to memory+disk. No-op on pandas."""
    if not _is_spark_pandas(df):
        return df
    from pyspark import StorageLevel

    spark_df = as_spark_df(df)
    spark_df.persist(StorageLevel.MEMORY_AND_DISK)
    spark_df.count()
    from .spark_backend import _as_pandas_api

    return _as_pandas_api(spark_df)


def spark_cast_float32(df: Any) -> Any:
    """Cast all numeric columns to float32. Works on both pandas and pyspark.pandas."""
    if not _is_spark_pandas(df):
        numeric = df.select_dtypes(include=["number"]).columns
        if len(numeric) == 0:
            return df
        return df.astype({c: "float32" for c in numeric})
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.sql.types import DoubleType, IntegerType, LongType, ShortType

    spark_df = as_spark_df(df)
    _NUMERIC = (DoubleType, LongType, IntegerType, ShortType)
    select_exprs = [
        F.col(c).cast("float").alias(c) if isinstance(spark_df.schema[c].dataType, _NUMERIC) else F.col(c)
        for c in spark_df.columns
    ]
    from .spark_backend import _as_pandas_api

    return _as_pandas_api(spark_df.select(*select_exprs))


def safe_fillna(df: Any, value: Any) -> Any:
    if _is_spark_pandas(df):
        spark_df = as_spark_df(df).fillna(value).localCheckpoint(eager=True)
        from .spark_backend import _as_pandas_api

        return _as_pandas_api(spark_df)
    return df.fillna(value)


def lazy_fillna(df: Any, value: Any) -> Any:
    """fillna without checkpoint — use when a downstream checkpoint will materialize."""
    if _is_spark_pandas(df):
        from .spark_backend import _as_pandas_api

        return _as_pandas_api(as_spark_df(df).fillna(value))
    return df.fillna(value)


def bulk_label_encode(df: Any, columns: list[str]) -> Any:
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return df
    if _is_spark_pandas(df):
        return _spark_bulk_label_encode(df, valid)
    return _pandas_bulk_label_encode(df, valid)


def _pandas_bulk_label_encode(df: Any, columns: list[str]) -> Any:
    df = df.copy()
    for col in columns:
        codes, _ = df[col].astype(str).factorize(sort=True)
        df[col] = codes
    return df


_MAX_LABEL_CARDINALITY = 10_000
_ENCODE_BATCH = 200


def _spark_bulk_label_encode(df: Any, columns: list[str]) -> Any:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)

    # --- Step 1: batched countDistinct ---
    cardinality: dict[str, int] = {}
    for start in range(0, len(columns), _ENCODE_BATCH):
        batch = columns[start : start + _ENCODE_BATCH]
        card_exprs = [F.countDistinct(F.col(c)).alias(c) for c in batch]
        row = spark_df.agg(*card_exprs).collect()[0]
        for c in batch:
            cardinality[c] = row[c] or 0

    low_card = [c for c in columns if cardinality[c] <= _MAX_LABEL_CARDINALITY]
    high_card = [c for c in columns if cardinality[c] > _MAX_LABEL_CARDINALITY]
    if high_card:
        print(
            f"Hash-encoding {len(high_card)} high-cardinality columns "
            f"(>{_MAX_LABEL_CARDINALITY:,} unique): "
            f"{high_card[:5]}{'...' if len(high_card) > 5 else ''}"
        )

    # --- Step 2: batched collect_set for vocab ---
    vocab: dict[str, list[str]] = {}
    for start in range(0, len(low_card), _ENCODE_BATCH):
        batch = low_card[start : start + _ENCODE_BATCH]
        set_exprs = [F.sort_array(F.collect_set(F.col(c).cast("string"))).alias(c) for c in batch]
        row = spark_df.agg(*set_exprs).collect()[0]
        for c in batch:
            vocab[c] = [v for v in (row[c] or []) if v is not None]

    # --- Step 3: batched select (avoids Catalyst plan explosion) ---
    low_set, high_set = set(low_card), set(high_card)
    encode_cols = list(low_set | high_set)
    for start in range(0, len(encode_cols), _ENCODE_BATCH):
        batch_set = set(encode_cols[start : start + _ENCODE_BATCH])
        select_exprs = []
        for c in spark_df.columns:
            if c not in batch_set:
                select_exprs.append(F.col(c))
            elif c in high_set:
                select_exprs.append(F.hash(F.col(c).cast("string")).alias(c))
            else:
                distinct = vocab.get(c, [])
                if distinct:
                    mapping = F.create_map(*[x for i, v in enumerate(distinct) for x in (F.lit(v), F.lit(i))])
                    select_exprs.append(F.coalesce(mapping[F.col(c).cast("string")], F.lit(-1)).cast("int").alias(c))
                else:
                    select_exprs.append(F.lit(0).cast("int").alias(c))
        spark_df = spark_df.select(*select_exprs)

    from .spark_backend import _as_pandas_api

    return _as_pandas_api(spark_df)


def bulk_median_impute(df: Any, columns: list[str] | None = None) -> Any:
    if _is_spark_pandas(df):
        return _spark_bulk_median_impute(df, columns)
    return _pandas_bulk_median_impute(df, columns)


def _pandas_bulk_median_impute(df: Any, columns: list[str] | None = None) -> Any:
    if columns is not None:
        num_cols = [c for c in columns if is_numeric_dtype(df[c])]
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return df
    medians = df[num_cols].median()
    fill_dict = {c: (float(v) if _pandas.notna(v) else 0) for c, v in medians.items()}
    return df.fillna(fill_dict)


def _spark_bulk_median_impute(df: Any, columns: list[str] | None = None) -> Any:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    num_cols = _spark_numeric_cols(spark_df, columns)
    from .spark_backend import _as_pandas_api

    if not num_cols:
        return _as_pandas_api(spark_df)
    safe = {c: f"__m{i}__" for i, c in enumerate(num_cols)}
    work_df = spark_df.toDF(*[safe.get(c, c) for c in spark_df.columns])
    cols_with_nulls: list[str] = []
    for start in range(0, len(num_cols), _AGG_BATCH_SIZE):
        batch = num_cols[start : start + _AGG_BATCH_SIZE]
        safe_batch = [safe[c] for c in batch]
        exprs = [F.sum(F.when(F.col(s).isNull(), F.lit(1)).otherwise(F.lit(0))).alias(s) for s in safe_batch]
        row = work_df.agg(*exprs).head()
        for c, s in zip(batch, safe_batch):
            val = row[s]
            if val is not None and val > 0:
                cols_with_nulls.append(c)
    if not cols_with_nulls:
        return _as_pandas_api(spark_df)
    fill_dict: dict[str, float] = {}
    for start in range(0, len(cols_with_nulls), _AGG_BATCH_SIZE):
        batch = cols_with_nulls[start : start + _AGG_BATCH_SIZE]
        safe_batch = [safe[c] for c in batch]
        exprs = [F.percentile_approx(F.col(s).cast("double"), 0.5).alias(s) for s in safe_batch]
        row = work_df.agg(*exprs).head()
        for c, s in zip(batch, safe_batch):
            val = row[s]
            fill_dict[c] = float(val) if val is not None else 0.0
    return _as_pandas_api(spark_df.fillna(fill_dict))


def bulk_zero_variance_cols(df: Any) -> list[str]:
    if _is_spark_pandas(df):
        return _spark_bulk_zero_variance_cols(df)
    num = df.select_dtypes(include="number")
    if num.empty:
        return []
    stds = num.std()
    return stds[(stds == 0) | stds.isna()].index.tolist()


def _spark_bulk_zero_variance_cols(df: Any) -> list[str]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    num_cols = _spark_numeric_cols(spark_df)
    if not num_cols:
        return []
    # stddev uses O(1) memory per column (running sum, sum-of-squares, count)
    _BATCH = 2000
    zero_var: list[str] = []
    for start in range(0, len(num_cols), _BATCH):
        batch = num_cols[start : start + _BATCH]
        exprs = [F.stddev(F.col(c)).alias(c) for c in batch]
        row = spark_df.agg(*exprs).head()
        zero_var.extend(c for c in batch if row[c] is None or float(row[c]) == 0)
    return zero_var


def collect_for_sklearn(obj: Any) -> Any:
    if isinstance(obj, (_pandas.DataFrame, _pandas.Series)):
        return obj
    if _is_spark_pandas(obj):
        enable_arrow_optimization()
        is_series = getattr(obj, "ndim", 2) == 1
        spark_df = as_spark_df(obj) if hasattr(obj, "to_spark") else as_spark_df(obj.to_frame())
        result = spark_df.toPandas()
        return result.iloc[:, 0] if is_series else result
    return obj


def as_pandas_api(spark_df: Any) -> Any:
    from .spark_backend import _as_pandas_api

    return _as_pandas_api(sanitize_spark_timestamps(spark_df))


def load_spark_table(source: str) -> Any:
    spark = get_spark_session()
    if not spark:
        raise RuntimeError(f"No active Spark session to read table '{source}'")
    return spark.table(source)


def register_temp_view(spark_df: Any, view_name: str, *, purpose: str) -> str:
    """Register ``spark_df`` as a Spark global temp view.

    ``purpose`` is a required, non-empty audit string documenting why the
    caller needs a session-scoped view (e.g. ``"lifecycle_round_trip_test"``,
    ``"interactive_debug"``). Code review can scan call sites and ask "is a
    temp view really the right tool here?" — for cross-notebook persistence
    the right tool is ``save_active_dataset``, since temp views vanish when
    the next notebook starts a new SparkSession.
    """
    _require_purpose(purpose)
    _assert_no_case_insensitive_duplicate_columns(spark_df, view_name)
    spark_df.createOrReplaceGlobalTempView(view_name)
    return f"global_temp.{view_name}"


def _require_purpose(purpose: str) -> None:
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError(
            "register_temp_view: 'purpose' kwarg is required and must be a "
            "non-empty audit string. Spark global temp views are session-"
            "scoped — for cross-notebook persistence call `save_active_dataset`."
        )


def _assert_no_case_insensitive_duplicate_columns(spark_df: Any, view_name: str) -> None:
    columns = list(getattr(spark_df, "columns", ()))
    if not columns:
        return
    seen: dict[str, list[str]] = {}
    for c in columns:
        seen.setdefault(str(c).lower(), []).append(str(c))
    duplicates = sorted(
        name for variants in seen.values() if len(variants) > 1 for name in variants
    )
    if duplicates:
        raise ValueError(
            f"register_temp_view({view_name!r}): DataFrame has case-insensitive "
            f"duplicate columns: {duplicates}. Spark resolves column names "
            f"case-insensitively by default, so any downstream `F.col(name)` "
            f"or `as_pandas_api()` call would raise [AMBIGUOUS_REFERENCE]. "
            f"Drop or rename the offending columns before registering the view."
        )


_stage_objects: list[Any] = []


def track_stage_object(*objects: Any) -> None:
    """Register heavy objects (DataFrames, models) for cleanup by release_stage_memory()."""
    _stage_objects.extend(objects)


def _try_unpersist(obj: Any) -> None:
    if hasattr(obj, "unpersist") and hasattr(obj, "to_spark"):
        obj.unpersist()
    elif hasattr(obj, "spark") and hasattr(obj.spark, "unpersist"):
        obj.spark.unpersist()


def release_stage_memory() -> None:
    """Unpersist tracked objects, run GC, and clear Spark cache."""
    for obj in _stage_objects:
        try:
            _try_unpersist(obj)
        except (AttributeError, RuntimeError, OSError):
            pass
    _stage_objects.clear()
    import gc

    gc.collect()
    import os

    if not (is_databricks() or os.environ.get("CR_SPARK_REMOTE")):
        return
    session = get_spark_session()
    if session is None:
        return
    session.catalog.clearCache()


__all__ = [
    "pd",
    "native_pd",
    "DataFrame",
    "Series",
    "Timestamp",
    "Timedelta",
    "DatetimeIndex",
    "CategoricalDtype",
    "NA",
    "NaT",
    "to_datetime",
    "to_timedelta",
    "to_numeric",
    "cut",
    "qcut",
    "get_dummies",
    "crosstab",
    "notna",
    "isna",
    "is_dataframe",
    "is_spark_available",
    "is_pandas_api_on_spark",
    "is_remote_spark",
    "use_distributed_processing",
    "connect_remote_spark",
    "to_pandas",
    "safe_isinf",
    "safe_isfinite",
    "concat",
    "merge",
    "api_types",
    "is_numeric_dtype",
    "is_string_dtype",
    "is_datetime64_any_dtype",
    "is_bool_dtype",
    "is_categorical_dtype",
    "is_integer_dtype",
    "is_float_dtype",
    "is_extension_array_dtype",
    "_extract_dtype",
    "get_spark_session",
    "set_spark_config",
    "enable_arrow_optimization",
    "configure_spark_pandas",
    "is_databricks",
    "is_notebook",
    "get_display_function",
    "get_dbutils",
    "safe_memory_usage_bytes",
    "as_tz_naive",
    "groupby_multi_agg",
    "timestamp_diff_seconds",
    "timestamp_diff_days",
    "timestamp_diffs_seconds",
    "head_as_list",
    "unique_overlap_counts",
    "distributed_case_variations",
    "timedelta_to_days",
    "timedelta_to_seconds",
    "_is_spark_pandas",
    "period_start_time",
    "safe_to_datetime",
    "ensure_datetime_column",
    "ensure_timestamp",
    "_normalize_timestamp_columns",
    "clamp_distributed_timestamps",
    "normalize_decimal_columns",
    "sanitize_spark_timestamps",
    "strip_spark_timestamp_tz",
    "pandas_dtype_to_spark_schema",
    "ops",
    "DataOps",
    "RemotePath",
    "make_path",
    "safe_drop_duplicates",
    "safe_isin",
    "safe_len",
    "safe_sample",
    "safe_select",
    "safe_describe",
    "batched_corr_matrix",
    "bulk_corr_with_target",
    "bulk_null_counts",
    "leakage_corr_combined",
    "bulk_class_overlap",
    "bulk_effect_sizes",
    "BulkEffectSizeResult",
    "temporal_quantile",
    "spark_cast_float32",
    "spark_checkpoint",
    "spark_persist",
    "safe_fillna",
    "lazy_fillna",
    "bulk_label_encode",
    "bulk_median_impute",
    "bulk_zero_variance_cols",
    "collect_for_sklearn",
    "as_pandas_api",
    "as_spark_df",
    "load_spark_table",
    "register_temp_view",
    "release_stage_memory",
    "track_stage_object",
]
