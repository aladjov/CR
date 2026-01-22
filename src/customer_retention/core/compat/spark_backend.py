from __future__ import annotations
from typing import Any, Dict, List, Optional

try:
    import pyspark.pandas as ps
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def _get_spark() -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required for Spark backend")
    return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()


def read_csv(path: str, **kwargs: Any) -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    return ps.read_csv(path, **kwargs)


def read_delta(path: str, version: Optional[int] = None) -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    spark = _get_spark()
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", version)
    return reader.load(path).to_pandas_on_spark()


def write_delta(df: Any, path: str, mode: str = "overwrite",
                partition_by: Optional[List[str]] = None) -> None:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    spark_df = df.to_spark() if hasattr(df, "to_spark") else df
    writer = spark_df.write.format("delta").mode(mode)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)


def get_missing_stats(df: Any) -> Dict[str, float]:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    return (pdf.isnull().sum() / len(pdf)).to_dict()


def correlation_matrix(df: Any, columns: Optional[List[str]] = None) -> Any:
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark required")
    if columns:
        return df[columns].to_pandas().corr()
    return df.select_dtypes(include=["number"]).to_pandas().corr()


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
