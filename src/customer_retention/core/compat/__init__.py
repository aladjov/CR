from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union

import pandas as _pandas
import numpy as np

from .detection import (
    is_spark_available,
    is_pandas_api_on_spark,
    is_databricks,
    is_notebook,
    get_spark_session,
    get_display_function,
    get_dbutils,
    set_spark_config,
    enable_arrow_optimization,
    configure_spark_pandas,
)
from .ops import ops, DataOps

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


Timestamp = _pandas.Timestamp
Timedelta = _pandas.Timedelta
DatetimeIndex = _pandas.DatetimeIndex
CategoricalDtype = _pandas.CategoricalDtype
NA = _pandas.NA
NaT = _pandas.NaT

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
    "DataFrame",
    "Series",
    "Timestamp",
    "Timedelta",
    "DatetimeIndex",
    "CategoricalDtype",
    "NA",
    "NaT",
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
    "ops",
    "DataOps",
]
