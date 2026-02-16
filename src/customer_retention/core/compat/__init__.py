from __future__ import annotations

from typing import Any, Union

import pandas as _pandas

from .detection import (
    configure_spark_pandas,
    enable_arrow_optimization,
    get_dbutils,
    get_display_function,
    get_spark_session,
    is_databricks,
    is_notebook,
    is_pandas_api_on_spark,
    is_spark_available,
    set_spark_config,
)
from .ops import DataOps, ops

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


def is_dataframe(obj: Any) -> bool:
    return isinstance(obj, _DATAFRAME_TYPES)


def to_pandas(df: Any) -> _pandas.DataFrame:
    if isinstance(df, _pandas.DataFrame):
        return df
    if _SPARK_PANDAS_AVAILABLE:
        try:
            import pyspark.pandas as ps
            if isinstance(df, ps.DataFrame):
                return df.to_pandas()
        except Exception:
            pass
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

# Utility functions (always use real pandas, never pyspark.pandas)
to_datetime = _pandas.to_datetime
to_timedelta = _pandas.to_timedelta
to_numeric = _pandas.to_numeric
cut = _pandas.cut
qcut = _pandas.qcut
get_dummies = _pandas.get_dummies
crosstab = _pandas.crosstab
notna = _pandas.notna
isna = _pandas.isna

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
    """Ensure *column* in a **pandas** DataFrame is ``datetime64``.

    Call this after ``to_pandas()`` to safely convert columns that may have
    arrived as int64 epoch values from Spark.  Returns the DataFrame
    (modified in-place).
    """
    if not _pandas.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = safe_to_datetime(df[column])
    return df


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
    df = df.copy()
    for col in df.columns:
        if _pandas.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = as_tz_naive(df[col])
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


def timestamp_diffs_seconds(series: Any) -> Any:
    if hasattr(series, 'to_spark'):
        epoch = (series - series.min()).astype(float)
        return epoch - epoch.shift(1)
    return timedelta_to_seconds(series - series.shift(1))


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
    "safe_to_datetime",
    "ensure_datetime_column",
    "normalize_timestamp_columns",
    "pandas_dtype_to_spark_schema",
    "ops",
    "DataOps",
]
