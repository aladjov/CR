from typing import Any, Dict, List, Optional

import pandas as pd

from customer_retention.core.compat.detection import get_spark_session, is_spark_available

from ..base import AdapterResult
from .base import FeatureStoreAdapter, FeatureViewConfig


class DatabricksFeatureStore(FeatureStoreAdapter):
    def __init__(self, catalog: str = "main", schema: str = "default"):
        if not is_spark_available():
            raise ImportError("PySpark required for DatabricksFeatureStore")
        self.catalog = catalog
        self.schema = schema
        self._fe_client = None

    @property
    def fe_client(self) -> Any:
        if self._fe_client is None:
            from databricks.feature_engineering import FeatureEngineeringClient
            self._fe_client = FeatureEngineeringClient()
        return self._fe_client

    def _full_name(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    def create_table(self, name: str, schema: Dict[str, str], primary_keys: List[str]) -> AdapterResult:
        full_name = self._full_name(name)
        spark = get_spark_session()
        df = spark.createDataFrame([], self._schema_to_spark(schema))
        self.fe_client.create_table(name=full_name, primary_keys=primary_keys, df=df)
        return AdapterResult(success=True, metadata={"name": full_name})

    def _schema_to_spark(self, schema: Dict[str, str]) -> Any:
        from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType
        type_map = {"int": IntegerType(), "float": FloatType(), "string": StringType()}
        fields = [StructField(name, type_map.get(dtype, StringType()), True) for name, dtype in schema.items()]
        return StructType(fields)

    def write_table(self, name: str, df: pd.DataFrame, mode: str = "merge") -> AdapterResult:
        full_name = self._full_name(name)
        spark = get_spark_session()
        spark_df = spark.createDataFrame(df)
        self.fe_client.write_table(name=full_name, df=spark_df, mode=mode)
        return AdapterResult(success=True)

    def read_table(self, name: str, version: Optional[int] = None) -> pd.DataFrame:
        full_name = self._full_name(name)
        spark = get_spark_session()
        reader = spark.read.format("delta").table(full_name)
        if version is not None:
            reader = spark.read.format("delta").option("versionAsOf", version).table(full_name)
        return reader.toPandas()

    def get_table_metadata(self, name: str) -> Dict[str, Any]:
        full_name = self._full_name(name)
        table_info = self.fe_client.get_table(full_name)
        return {"name": full_name, "primary_keys": table_info.primary_keys, "features": table_info.features}

    def list_tables(self) -> List[str]:
        tables = self.fe_client.list_tables()
        return [t.name for t in tables if t.name.startswith(f"{self.catalog}.{self.schema}")]

    def delete_table(self, name: str) -> AdapterResult:
        full_name = self._full_name(name)
        self.fe_client.drop_table(full_name)
        return AdapterResult(success=True)

    def register_feature_view(self, config: FeatureViewConfig, df: pd.DataFrame) -> str:
        table_name = self._full_name(config.name)
        spark = get_spark_session()
        spark_df = spark.createDataFrame(df)
        self.fe_client.create_table(name=table_name, primary_keys=[config.entity_key], df=spark_df)
        return table_name

    def get_historical_features(self, entity_df: pd.DataFrame, feature_refs: List[str]) -> pd.DataFrame:
        from databricks.feature_engineering import FeatureLookup
        spark = get_spark_session()
        lookups = [FeatureLookup(table_name=ref.split(":")[0], lookup_key=[entity_df.columns[0]]) for ref in feature_refs]
        training_set = self.fe_client.create_training_set(df=spark.createDataFrame(entity_df), feature_lookups=lookups, label=None)
        return training_set.load_df().toPandas()

    def materialize(self, feature_views: List[str], start_date: str, end_date: str) -> None:
        pass

    def get_online_features(self, entity_keys: Dict[str, List[Any]], feature_refs: List[str]) -> Dict:
        entity_df = pd.DataFrame(entity_keys)
        spark = get_spark_session()
        from databricks.feature_engineering import FeatureLookup
        lookups = [FeatureLookup(table_name=ref.split(":")[0], lookup_key=list(entity_keys.keys())) for ref in feature_refs]
        result = self.fe_client.score_batch(df=spark.createDataFrame(entity_df), feature_lookups=lookups)
        return result.toPandas().to_dict()
