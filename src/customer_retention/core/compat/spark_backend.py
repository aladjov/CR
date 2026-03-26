from __future__ import annotations

from typing import Any, Dict, List, Optional

_SparkDF: type | None = None
try:
    import pyspark.pandas as ps
    from pyspark.sql import DataFrame as _SparkDF
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def _is_native_spark(obj: Any) -> bool:
    return _SparkDF is not None and isinstance(obj, _SparkDF)


def _get_spark() -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required for Spark backend")
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
    return spark


def read_csv(path: str, **kwargs: Any) -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    return ps.read_csv(path, **kwargs)


def _as_pandas_api(spark_df: Any) -> Any:
    result = spark_df.pandas_api() if hasattr(spark_df, "pandas_api") else spark_df.to_pandas_on_spark()
    if hasattr(result, '_psseries'):
        result._psseries = None
    return result


def read_delta(path: str, version: Optional[int] = None) -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    spark = _get_spark()
    try:
        reader = spark.read.format("delta")
        if version is not None:
            reader = reader.option("versionAsOf", version)
        return _as_pandas_api(reader.load(path))
    except Exception:
        from . import pandas_backend
        return pandas_backend.read_delta(path, version=version)


def write_delta(df: Any, path: str, mode: str = "overwrite",
                partition_by: Optional[List[str]] = None) -> None:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    if hasattr(df, "to_spark"):
        from . import as_spark_df
        spark_df = as_spark_df(df)
    elif _is_native_spark(df):
        spark_df = df
    else:
        from . import pandas_backend
        pandas_backend.write_delta(df, path, mode=mode, partition_by=partition_by)
        return
    writer = spark_df.write.format("delta").mode(mode)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)


def get_missing_stats(df: Any) -> Dict[str, float]:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    return (df.isnull().sum() / len(df)).to_dict()


def correlation_matrix(df: Any, columns: Optional[List[str]] = None) -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    subset = df[columns] if columns else df.select_dtypes(include=["number"])
    result = subset.corr()
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    return result


def get_dtype_info(df: Any) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def sample(df: Any, n: int, random_state: int = 42) -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    fraction = min(1.0, n / len(df))
    return df.sample(frac=fraction, random_state=random_state).head(n)


def concat(dfs: List[Any], axis: int = 0, ignore_index: bool = True) -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    return ps.concat(dfs, axis=axis, ignore_index=ignore_index)
