from customer_retention.core.compat.detection import is_spark_available

from .feature_store import DatabricksFeatureStore, FeatureStoreAdapter, LocalFeatureStore
from .mlflow import DatabricksMLflow, LocalMLflow, MLflowAdapter
from .storage import DatabricksDelta, DeltaStorage, LocalDelta


def get_delta(force_local: bool = False) -> DeltaStorage:
    if force_local or not is_spark_available():
        return LocalDelta()
    return DatabricksDelta()


def get_feature_store(base_path: str = "./feature_store", catalog: str = "main",
                      schema: str = "default", force_local: bool = False) -> FeatureStoreAdapter:
    if force_local or not is_spark_available():
        return LocalFeatureStore(base_path=base_path)
    return DatabricksFeatureStore(catalog=catalog, schema=schema)


def get_mlflow(tracking_uri: str = "./mlruns", registry_uri: str = "databricks-uc",
               force_local: bool = False) -> MLflowAdapter:
    if force_local or not is_spark_available():
        return LocalMLflow(tracking_uri=tracking_uri)
    return DatabricksMLflow(registry_uri=registry_uri)
