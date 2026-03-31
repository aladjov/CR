from customer_retention.core.compat.detection import is_databricks, is_remote_spark
from customer_retention.core.config.experiments import get_catalog, get_schema

from .feature_store import DatabricksFeatureStore, FeatureStoreAdapter, LocalFeatureStore
from .mlflow import DatabricksMLflow, LocalMLflow, MLflowAdapter
from .storage import DatabricksDelta, DeltaStorage, LocalDelta


def _use_distributed_backend() -> bool:
    return is_databricks() or is_remote_spark()


def get_delta(force_local: bool = False) -> DeltaStorage:
    from .storage.local import deltalake_available
    if force_local and deltalake_available():
        return LocalDelta()
    if _use_distributed_backend():
        return DatabricksDelta()
    if deltalake_available():
        return LocalDelta()
    raise ImportError(
        "No Delta storage backend available. "
        "Install deltalake (pip install deltalake) or use PySpark."
    )


def get_feature_store(
    base_path: str = "./feature_store", catalog: str | None = None, schema: str | None = None, force_local: bool = False
) -> FeatureStoreAdapter:
    if force_local or not _use_distributed_backend():
        return LocalFeatureStore(base_path=base_path)
    return DatabricksFeatureStore(catalog=catalog or get_catalog(), schema=schema or get_schema())


def get_mlflow(
    tracking_uri: str = "./mlruns", registry_uri: str = "databricks-uc", force_local: bool = False
) -> MLflowAdapter:
    if force_local or not _use_distributed_backend():
        return LocalMLflow(tracking_uri=tracking_uri)
    return DatabricksMLflow(registry_uri=registry_uri)
