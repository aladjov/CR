import json
from typing import Any, Dict, List, Optional

import pandas as pd

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

    def read(self, path: str, version: Optional[int] = None) -> Any:
        path = self._normalize_path(path)
        reader = self.spark.read.format("delta")
        if version is not None:
            reader = reader.option("versionAsOf", version)
        return reader.load(path).to_pandas_on_spark()

    def _to_spark_df(self, df: pd.DataFrame) -> Any:
        from customer_retention.core.compat import normalize_timestamp_columns, pandas_dtype_to_spark_schema
        normalized = normalize_timestamp_columns(df)
        schema = pandas_dtype_to_spark_schema(normalized)
        return self.spark.createDataFrame(normalized, schema=schema)

    def write(self, df: Any, path: str, mode: str = "overwrite",
              partition_by: Optional[List[str]] = None,
              metadata: Optional[Dict[str, str]] = None) -> None:
        path = self._normalize_path(path)
        if metadata:
            self.spark.conf.set(
                "spark.databricks.delta.commitInfo.userMetadata",
                json.dumps(metadata),
            )
        spark_df = df.to_spark() if hasattr(df, "to_spark") else self._to_spark_df(df)
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
        spark_df = df.to_spark() if hasattr(df, "to_spark") else self._to_spark_df(df)
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
