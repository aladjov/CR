from typing import Any, Dict, List, Optional
import pandas as pd
from .base import DeltaStorage
from customer_retention.core.compat.detection import is_spark_available, get_spark_session


class DatabricksDelta(DeltaStorage):
    def __init__(self):
        if not is_spark_available():
            raise ImportError("PySpark required for DatabricksDelta")
        self._spark = None

    @property
    def spark(self) -> Any:
        if self._spark is None:
            self._spark = get_spark_session()
            if self._spark is None:
                from pyspark.sql import SparkSession
                self._spark = SparkSession.builder.getOrCreate()
        return self._spark

    def read(self, path: str, version: Optional[int] = None) -> pd.DataFrame:
        reader = self.spark.read.format("delta")
        if version is not None:
            reader = reader.option("versionAsOf", version)
        return reader.load(path).toPandas()

    def write(self, df: pd.DataFrame, path: str, mode: str = "overwrite",
              partition_by: Optional[List[str]] = None) -> None:
        spark_df = self.spark.createDataFrame(df)
        writer = spark_df.write.format("delta").mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)

    def merge(self, df: pd.DataFrame, path: str, condition: str,
              update_cols: Optional[List[str]] = None) -> None:
        from delta.tables import DeltaTable
        spark_df = self.spark.createDataFrame(df)
        target = DeltaTable.forPath(self.spark, path)
        merge_builder = target.alias("target").merge(spark_df.alias("source"), condition)
        if update_cols:
            update_dict = {col: f"source.{col}" for col in update_cols}
            merge_builder = merge_builder.whenMatchedUpdate(set=update_dict)
        else:
            merge_builder = merge_builder.whenMatchedUpdateAll()
        merge_builder.whenNotMatchedInsertAll().execute()

    def history(self, path: str) -> List[Dict[str, Any]]:
        from delta.tables import DeltaTable
        dt = DeltaTable.forPath(self.spark, path)
        history_df = dt.history()
        return history_df.toPandas().to_dict("records")

    def vacuum(self, path: str, retention_hours: int = 168) -> None:
        from delta.tables import DeltaTable
        dt = DeltaTable.forPath(self.spark, path)
        dt.vacuum(retention_hours)
