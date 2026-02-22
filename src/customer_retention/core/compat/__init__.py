from __future__ import annotations

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
        return int(usage.sum()) if hasattr(usage, 'sum') else int(usage)
    except Exception:
        return 0


def safe_to_datetime(series: Any, **kwargs: Any) -> _pandas.Series:
    if _pandas.api.types.is_datetime64_any_dtype(series):
        return as_tz_naive(series if isinstance(series, _pandas.Series) else _pandas.Series(series))
    arr = series.to_numpy() if hasattr(series, 'to_numpy') else _pandas.array(series)
    if _pandas.api.types.is_integer_dtype(arr) or _pandas.api.types.is_integer_dtype(series):
        arr = _pandas.to_numeric(arr, errors='coerce')
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


def normalize_timestamp_columns(df: _pandas.DataFrame) -> _pandas.DataFrame:
    import datetime as _dt

    df = df.copy()
    for col in df.columns:
        if _pandas.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = as_tz_naive(df[col])
        elif df[col].dtype == "object":
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample = non_null.iloc[:10]
                if all(isinstance(v, _dt.date) for v in sample):
                    df[col] = _pandas.to_datetime(df[col])
    return df


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
        "object": StringType(),
    }
    fields = []
    for col in df.columns:
        dtype = df[col].dtype
        if _pandas.api.types.is_datetime64_any_dtype(dtype):
            spark_type = TimestampNTZType()
        elif hasattr(dtype, "numpy_dtype"):
            spark_type = _MAP.get(str(dtype.numpy_dtype), StringType())
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
    return hasattr(obj, 'spark') or hasattr(obj, 'to_spark')


def timestamp_diffs_seconds(series: Any) -> Any:
    if _is_spark_pandas(series):
        import pyspark.sql.functions as F  # noqa: N812
        epoch = series.spark.transform(lambda c: F.unix_timestamp(c.cast("timestamp")).cast("double"))
        return epoch - epoch.shift(1)
    return timedelta_to_seconds(series - series.shift(1))


_FREQ_TO_SPARK: dict[str, str] = {
    "D": "day", "W": "week", "M": "month", "ME": "month",
    "MS": "month", "Q": "quarter", "QS": "quarter", "Y": "year", "YS": "year",
}


def period_start_time(series: Any, freq: str) -> Any:
    if _is_spark_pandas(series):
        import pyspark.sql.functions as F  # noqa: N812
        spark_freq = _FREQ_TO_SPARK.get(freq, freq.lower())
        return series.spark.transform(lambda c: F.date_trunc(spark_freq, c))
    return series.dt.to_period(freq).dt.start_time


def groupby_multi_agg(df: Any, group_col: str, agg_col: str, agg_funcs: list) -> Any:
    if hasattr(df, 'to_spark'):
        import pyspark.sql.functions as F  # noqa: N812
        spark_df = df.to_spark()
        exprs = [getattr(F, fn)(agg_col).alias(fn) for fn in agg_funcs]
        result = spark_df.groupBy(group_col).agg(*exprs)
        from .spark_backend import _as_pandas_api
        return _as_pandas_api(result)
    return df.groupby(group_col)[agg_col].agg(agg_funcs).reset_index()


def safe_to_list(obj: Any) -> list:
    if isinstance(obj, (_pandas.Series, _pandas.Index)):
        return obj.to_list()
    if hasattr(obj, 'to_numpy'):
        return obj.to_numpy().tolist()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return list(obj)


def safe_isinf(series: Any) -> Any:
    return (series == float('inf')) | (series == float('-inf'))


def safe_isfinite(series: Any) -> Any:
    return series.notna() & ~safe_isinf(series)


def safe_sample(df: Any, n: int, random_state: int = 42) -> Any:
    if n <= 0:
        return df.head(0)
    n = min(n, len(df))
    if n == 0:
        return df.head(0)
    if _is_spark_pandas(df):
        frac = min(1.0, n / max(1, len(df)))
        result = df.sample(frac=frac, random_state=random_state)
        return result.head(n) if len(result) > n else result
    return df.sample(n=n, random_state=random_state)


def safe_describe(df: Any) -> Any:
    try:
        return df.describe()
    except Exception:
        numeric = df.select_dtypes(include="number")
        if len(numeric.columns) == 0:
            return _pandas.DataFrame()
        return numeric.describe()


def batched_corr_matrix(df: Any, columns: list[str]) -> _pandas.DataFrame:
    valid_cols = [c for c in columns if c in df.columns]
    if len(valid_cols) < 2:
        return _pandas.DataFrame(index=valid_cols, columns=valid_cols)
    if not _is_spark_pandas(df):
        return df[valid_cols].corr()
    return _spark_pairwise_corr(df, valid_cols)


def _spark_pairwise_corr(df: Any, cols: list[str]) -> _pandas.DataFrame:
    import numpy as _np
    import pyspark.sql.functions as F  # noqa: N812

    def _safe_corr(col_a: str, col_b: str) -> Any:
        denom = F.stddev_samp(col_a) * F.stddev_samp(col_b)
        return F.when(denom > 0, F.covar_samp(col_a, col_b) / denom)

    spark_df = df[cols].to_spark()
    n = len(cols)
    matrix = _np.full((n, n), _np.nan)

    stddev_exprs = [F.stddev(c).alias(f"s_{i}") for i, c in enumerate(cols)]
    stddev_row = spark_df.select(*stddev_exprs).head()
    has_variance = set()
    for i in range(n):
        val = stddev_row[f"s_{i}"]
        if val is not None and float(val) > 0:
            has_variance.add(i)
            matrix[i, i] = 1.0

    pairs = [(i, j) for i in has_variance for j in has_variance if i < j]
    _BATCH = 500
    for start in range(0, len(pairs), _BATCH):
        batch = pairs[start:start + _BATCH]
        exprs = [_safe_corr(cols[i], cols[j]).alias(f"c_{i}_{j}") for i, j in batch]
        row = spark_df.select(*exprs).head()
        for i, j in batch:
            val = row[f"c_{i}_{j}"]
            val = float(val) if val is not None else _np.nan
            matrix[i, j] = val
            matrix[j, i] = val

    return _pandas.DataFrame(matrix, columns=cols, index=cols)


def bulk_corr_with_target(df: Any, columns: list[str], target_column: str) -> dict[str, float]:
    import math
    if target_column not in df.columns:
        return {}
    valid = [c for c in columns if c in df.columns and c != target_column]
    if not valid:
        return {}
    if not _is_spark_pandas(df):
        target_float = df[target_column].astype(float)
        result: dict[str, float] = {}
        for c in valid:
            try:
                result[c] = df[c].astype(float).corr(target_float)
            except (ValueError, TypeError):
                result[c] = math.nan
        return result
    return _spark_bulk_corr_with_target(df, valid, target_column)


def _spark_bulk_corr_with_target(df: Any, columns: list[str], target_column: str) -> dict[str, float]:
    import math

    import pyspark.sql.functions as F  # noqa: N812
    spark_df = df[columns + [target_column]].to_spark()
    _BATCH = 500
    result: dict[str, float] = {}
    for start in range(0, len(columns), _BATCH):
        batch = columns[start:start + _BATCH]
        exprs = [F.corr(F.col(c).cast("double"), F.col(target_column).cast("double")).alias(f"c_{i}")
                 for i, c in enumerate(batch)]
        row = spark_df.select(*exprs).head()
        for i, c in enumerate(batch):
            val = row[f"c_{i}"]
            result[c] = float(val) if val is not None else math.nan
    return result


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
    "timestamp_diffs_seconds",
    "safe_to_list",
    "timedelta_to_days",
    "timedelta_to_seconds",
    "_is_spark_pandas",
    "period_start_time",
    "safe_to_datetime",
    "ensure_datetime_column",
    "ensure_timestamp",
    "normalize_timestamp_columns",
    "pandas_dtype_to_spark_schema",
    "ops",
    "DataOps",
    "RemotePath",
    "make_path",
    "safe_sample",
    "safe_describe",
    "batched_corr_matrix",
    "bulk_corr_with_target",
]
