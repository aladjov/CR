from __future__ import annotations

from pathlib import Path
from typing import Any

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.compat import to_pandas as _compat_to_pandas
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.integrations.adapters.factory import get_delta


def _local_delta() -> Any:
    return get_delta(force_local=True)


def _to_native_pandas(df: Any) -> Any:
    result = _compat_to_pandas(df)
    result.attrs.clear()
    return result


def save_active_dataset(namespace: RunNamespace, dataset_name: str, df: Any) -> Path:
    dlt_path = namespace.landing_table_dir(dataset_name)
    _local_delta().write(_to_native_pandas(df), str(dlt_path), mode="overwrite")
    return dlt_path


def load_active_dataset(namespace: RunNamespace, dataset_name: str) -> Any:
    dlt_path = namespace.landing_table_dir(dataset_name)
    if not dlt_path.is_dir():
        raise FileNotFoundError(f"Active dataset not found: {dlt_path}")
    return _to_native_pandas(_local_delta().read(str(dlt_path)))


def save_aggregated_dataset(namespace: RunNamespace, dataset_name: str, df: Any) -> Path:
    dlt_path = namespace.bronze_table_dir(dataset_name)
    _local_delta().write(_to_native_pandas(df), str(dlt_path), mode="overwrite")
    return dlt_path


def load_merge_dataset(
    namespace: RunNamespace,
    dataset_name: str,
    granularity: DatasetGranularity,
) -> Any:
    if granularity == DatasetGranularity.EVENT_LEVEL:
        dlt_path = namespace.bronze_table_dir(dataset_name)
        if dlt_path.is_dir():
            return _to_native_pandas(_local_delta().read(str(dlt_path)))
    return load_active_dataset(namespace, dataset_name)


def load_merge_dataset_distributed(
    namespace: RunNamespace,
    dataset_name: str,
    granularity: DatasetGranularity,
) -> Any:
    delta = get_delta()
    if granularity == DatasetGranularity.EVENT_LEVEL:
        dlt_path = namespace.bronze_table_dir(dataset_name)
        if dlt_path.is_dir():
            return delta.read(str(dlt_path))
    dlt_path = namespace.landing_table_dir(dataset_name)
    if not dlt_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dlt_path}")
    return delta.read(str(dlt_path))


def load_silver_merged(
    namespace: RunNamespace,
    dataset_name: str,
    granularity: DatasetGranularity,
) -> Any:
    silver = namespace.silver_merged_path
    if silver.is_dir():
        return _to_native_pandas(_local_delta().read(str(silver)))
    return load_merge_dataset(namespace, dataset_name, granularity)


def save_gold_features(namespace: RunNamespace, composite_name: str, df: Any) -> Path:
    dlt_path = namespace.gold_table_dir(composite_name)
    _local_delta().write(_to_native_pandas(df), str(dlt_path), mode="overwrite")
    return dlt_path


def load_gold_features(namespace: RunNamespace, composite_name: str) -> Any:
    dlt_path = namespace.gold_table_dir(composite_name)
    if not dlt_path.is_dir():
        raise FileNotFoundError(f"Gold features not found: {dlt_path}")
    return _to_native_pandas(_local_delta().read(str(dlt_path)))


def load_active_dataset_distributed(namespace: RunNamespace, dataset_name: str) -> Any:
    dlt_path = namespace.landing_table_dir(dataset_name)
    if not dlt_path.is_dir():
        raise FileNotFoundError(f"Active dataset not found: {dlt_path}")
    return get_delta().read(str(dlt_path))


def load_silver_merged_distributed(
    namespace: RunNamespace,
    dataset_name: str,
    granularity: DatasetGranularity,
) -> Any:
    silver = namespace.silver_merged_path
    if silver.is_dir():
        return get_delta().read(str(silver))
    return load_merge_dataset_distributed(namespace, dataset_name, granularity)


def load_gold_features_distributed(namespace: RunNamespace, composite_name: str) -> Any:
    dlt_path = namespace.gold_table_dir(composite_name)
    if not dlt_path.is_dir():
        raise FileNotFoundError(f"Gold features not found: {dlt_path}")
    return get_delta().read(str(dlt_path))
