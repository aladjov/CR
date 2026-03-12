import json
from typing import Any, Dict, List, Optional

from customer_retention.core.compat import as_spark_df, pd
from customer_retention.core.compat.detection import get_spark_session, is_spark_available

from .base import DeltaStorage


class DatabricksDelta(DeltaStorage):
    def __init__(self):
        if not is_spark_available():
            raise ImportError("PySpark required for DatabricksDelta")
        self._spark = None

    @staticmethod
    def _normalize_path(path: str) -> str:
        if path.startswith("/dbfs/"):
            return path[5:]
        return path

    @property
    def spark(self) -> Any:
        if self._spark is None:
            self._spark = get_spark_session()
            if self._spark is None:
                from pyspark.sql import SparkSession
                self._spark = SparkSession.builder.getOrCreate()
        return self._spark

    @staticmethod
    def _as_pandas_api(spark_df: Any) -> Any:
        if hasattr(spark_df, "pandas_api"):
            return spark_df.pandas_api()
        return spark_df.to_pandas_on_spark()

    def read(self, path: str, version: Optional[int] = None) -> Any:
        path = self._normalize_path(path)
        reader = self.spark.read.format("delta")
        if version is not None:
            reader = reader.option("versionAsOf", version)
        try:
            spark_df = reader.load(path)
            from customer_retention.core.compat import sanitize_spark_timestamps
            return self._as_pandas_api(sanitize_spark_timestamps(spark_df))
        except Exception as exc:
            if "DELTA_TABLE_NOT_FOUND" in str(exc):
                raise FileNotFoundError(f"Delta table not found: {path}") from exc
            raise

    def _to_spark_df(self, df: pd.DataFrame) -> Any:
        from customer_retention.core.compat import normalize_timestamps, pandas_dtype_to_spark_schema
        normalized = normalize_timestamps(df)
        schema = pandas_dtype_to_spark_schema(normalized)
        return self.spark.createDataFrame(normalized, schema=schema)

    @staticmethod
    def _strip_spark_timestamp_tz(spark_df: Any) -> Any:
        from customer_retention.core.compat import strip_spark_timestamp_tz
        return strip_spark_timestamp_tz(spark_df)

    @staticmethod
    def _is_native_spark_df(df: Any) -> bool:
        from pyspark.sql import DataFrame as NativeSparkDF
        return isinstance(df, NativeSparkDF)

    def write(self, df: Any, path: str, mode: str = "overwrite",
              partition_by: Optional[List[str]] = None,
              metadata: Optional[Dict[str, str]] = None) -> None:
        path = self._normalize_path(path)
        self.spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        self.spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        if metadata:
            self.spark.conf.set(
                "spark.databricks.delta.commitInfo.userMetadata",
                json.dumps(metadata),
            )
        if self._is_native_spark_df(df):
            spark_df = df
        elif hasattr(df, "to_spark"):
            spark_df = as_spark_df(df)
        else:
            spark_df = self._to_spark_df(df)
        spark_df = self._strip_spark_timestamp_tz(spark_df)
        from customer_retention.core.compat import clamp_spark_timestamps
        spark_df = clamp_spark_timestamps(spark_df)
        writer = spark_df.write.format("delta").mode(mode)
        if mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)

    def merge(self, df: Any, path: str, condition: str,
              update_cols: Optional[List[str]] = None) -> None:
        path = self._normalize_path(path)
        from delta.tables import DeltaTable
        if self._is_native_spark_df(df):
            spark_df = df
        elif hasattr(df, "to_spark"):
            spark_df = as_spark_df(df)
        else:
            spark_df = self._to_spark_df(df)
        spark_df = self._strip_spark_timestamp_tz(spark_df)
        target = DeltaTable.forPath(self.spark, path)
        merge_builder = target.alias("target").merge(spark_df.alias("source"), condition)
        if update_cols:
            update_dict = {col: f"source.{col}" for col in update_cols}
            merge_builder = merge_builder.whenMatchedUpdate(set=update_dict)
        else:
            merge_builder = merge_builder.whenMatchedUpdateAll()
        merge_builder.whenNotMatchedInsertAll().execute()

    def history(self, path: str) -> List[Dict[str, Any]]:
        path = self._normalize_path(path)
        from delta.tables import DeltaTable
        dt = DeltaTable.forPath(self.spark, path)
        history_df = dt.history()
        return history_df.toPandas().to_dict("records")

    def optimize(self, path: str, z_order_columns: Optional[List[str]] = None) -> None:
        path = self._normalize_path(path)
        from delta.tables import DeltaTable
        dt = DeltaTable.forPath(self.spark, path)
        if z_order_columns:
            dt.optimize().executeZOrderBy(z_order_columns)
        else:
            dt.optimize().executeCompaction()

    def vacuum(self, path: str, retention_hours: int = 168) -> None:
        path = self._normalize_path(path)
        from delta.tables import DeltaTable
        dt = DeltaTable.forPath(self.spark, path)
        dt.vacuum(retention_hours)

    def exists(self, path: str) -> bool:
        path = self._normalize_path(path)
        from delta.tables import DeltaTable
        try:
            DeltaTable.forPath(self.spark, path)
            return True
        except Exception:
            return False
