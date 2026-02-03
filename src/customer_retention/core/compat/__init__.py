from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
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

if _SPARK_PANDAS_AVAILABLE:
    try:
        import pyspark.pandas as ps
        pd = ps
        DataFrame = Union[ps.DataFrame, _pandas.DataFrame]
        Series = Union[ps.Series, _pandas.Series]
    except ImportError:
        import databricks.koalas as ps
        pd = ps
        DataFrame = Union[ps.DataFrame, _pandas.DataFrame]
        Series = Union[ps.Series, _pandas.Series]
else:
    pd = _pandas
    DataFrame = _pandas.DataFrame
    Series = _pandas.Series


def get_pandas() -> Any:
    return _pandas


def to_pandas(df: Any) -> _pandas.DataFrame:
    if isinstance(df, _pandas.DataFrame):
        return df
    if _SPARK_PANDAS_AVAILABLE:
        try:
            import pyspark.pandas as ps
            if isinstance(df, ps.DataFrame):
                return df.to_pandas()
        except ImportError:
            pass
    try:
        from pyspark.sql import DataFrame as NativeSparkDF
        if isinstance(df, NativeSparkDF):
            return df.toPandas()
    except ImportError:
        pass
    return _pandas.DataFrame(df)


def to_spark_pandas(df: Any) -> Any:
    if not _SPARK_PANDAS_AVAILABLE:
        return df if isinstance(df, _pandas.DataFrame) else _pandas.DataFrame(df)
    try:
        import pyspark.pandas as ps
        if isinstance(df, ps.DataFrame):
            return df
        if isinstance(df, _pandas.DataFrame):
            return ps.from_pandas(df)
        return ps.DataFrame(df)
    except ImportError:
        return df


def ensure_pandas_series(series: Any) -> _pandas.Series:
    if isinstance(series, _pandas.Series):
        return series
    if _SPARK_PANDAS_AVAILABLE:
        try:
            import pyspark.pandas as ps
            if isinstance(series, ps.Series):
                return series.to_pandas()
        except ImportError:
            pass
    return _pandas.Series(series)


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


def is_numeric_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_numeric_dtype(arr_or_dtype)


def is_string_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_string_dtype(arr_or_dtype)


def is_datetime64_any_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_datetime64_any_dtype(arr_or_dtype)


def is_bool_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_bool_dtype(arr_or_dtype)


def is_categorical_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_categorical_dtype(arr_or_dtype)


def is_integer_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_integer_dtype(arr_or_dtype)


def is_float_dtype(arr_or_dtype: Any) -> bool:
    return _pandas.api.types.is_float_dtype(arr_or_dtype)


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
    """Convert a Series to datetime, handling Spark LongType epoch integers.

    Like ``pd.to_datetime`` but automatically detects integer epoch columns
    and passes the correct ``unit`` parameter.  Any extra *kwargs* are
    forwarded to ``pd.to_datetime``.
    """
    series = ensure_pandas_series(series)
    if _pandas.api.types.is_datetime64_any_dtype(series):
        return series
    if _pandas.api.types.is_integer_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            unit = _infer_epoch_unit(non_null.iloc[0])
            return _pandas.to_datetime(series, unit=unit, **kwargs)
    return _pandas.to_datetime(series, **kwargs)


def ensure_datetime_column(df: _pandas.DataFrame, column: str) -> _pandas.DataFrame:
    """Ensure *column* in a **pandas** DataFrame is ``datetime64``.

    Call this after ``to_pandas()`` to safely convert columns that may have
    arrived as int64 epoch values from Spark.  Returns the DataFrame
    (modified in-place).
    """
    if not _pandas.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = safe_to_datetime(df[column])
    return df


class PandasCompat:
    @staticmethod
    def value_counts_normalize(series: Any, normalize: bool = False) -> Any:
        return series.value_counts(normalize=normalize)

    @staticmethod
    def apply_with_meta(df: Any, func: Any, meta: Any = None, **kwargs: Any) -> Any:
        return df.apply(func, **kwargs)

    @staticmethod
    def groupby_apply(grouped: Any, func: Any, **kwargs: Any) -> Any:
        return grouped.apply(func, **kwargs)


compat = PandasCompat()

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
    "is_spark_available",
    "is_pandas_api_on_spark",
    "get_pandas",
    "to_pandas",
    "to_spark_pandas",
    "ensure_pandas_series",
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
    "get_spark_session",
    "set_spark_config",
    "enable_arrow_optimization",
    "configure_spark_pandas",
    "compat",
    "PandasCompat",
    "is_databricks",
    "is_notebook",
    "get_display_function",
    "get_dbutils",
    "safe_memory_usage_bytes",
    "safe_to_datetime",
    "ensure_datetime_column",
    "ops",
    "DataOps",
]
