from __future__ import annotations

from pathlib import Path
from typing import Any

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.integrations.adapters.factory import get_delta


def save_active_dataset(namespace: RunNamespace, dataset_name: str, df: Any) -> Path:
    dlt_path = namespace.landing_table_dir(dataset_name)
    delta = get_delta()
    delta.write(df, str(dlt_path), mode="overwrite")
    return dlt_path


def load_active_dataset(namespace: RunNamespace, dataset_name: str) -> Any:
    dlt_path = namespace.landing_table_dir(dataset_name)
    if not dlt_path.is_dir():
        raise FileNotFoundError(f"Active dataset not found: {dlt_path}")
    delta = get_delta()
    return delta.read(str(dlt_path))


def save_aggregated_dataset(namespace: RunNamespace, dataset_name: str, df: Any) -> Path:
    dlt_path = namespace.bronze_table_dir(dataset_name)
    delta = get_delta()
    delta.write(df, str(dlt_path), mode="overwrite")
    return dlt_path


def load_merge_dataset(
    namespace: RunNamespace,
    dataset_name: str,
    granularity: DatasetGranularity,
) -> Any:
    if granularity == DatasetGranularity.EVENT_LEVEL:
        dlt_path = namespace.bronze_table_dir(dataset_name)
        if dlt_path.is_dir():
            delta = get_delta()
            return delta.read(str(dlt_path))
    return load_active_dataset(namespace, dataset_name)


def load_silver_merged(
    namespace: RunNamespace,
    dataset_name: str,
    granularity: DatasetGranularity,
) -> Any:
    silver = namespace.silver_merged_path
    if silver.is_dir():
        return get_delta().read(str(silver))
    return load_merge_dataset(namespace, dataset_name, granularity)


def save_gold_features(namespace: RunNamespace, composite_name: str, df: Any) -> Path:
    dlt_path = namespace.gold_table_dir(composite_name)
    get_delta().write(df, str(dlt_path), mode="overwrite")
    return dlt_path


def load_gold_features(namespace: RunNamespace, composite_name: str) -> Any:
    dlt_path = namespace.gold_table_dir(composite_name)
    if not dlt_path.is_dir():
        raise FileNotFoundError(f"Gold features not found: {dlt_path}")
    return get_delta().read(str(dlt_path))
