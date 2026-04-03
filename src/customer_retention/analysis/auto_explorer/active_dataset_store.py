from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.compat import (
    as_spark_df,
    normalize_timestamps,
    release_stage_memory,
)
from customer_retention.core.compat import to_pandas as _compat_to_pandas
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.integrations.adapters.factory import get_delta

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.project_context import ProjectContext

logger = logging.getLogger(__name__)


def _local_delta() -> Any:
    return get_delta(force_local=True)


def _to_native_pandas(df: Any) -> Any:
    result = _compat_to_pandas(df)
    result.attrs.clear()
    return normalize_timestamps(result)


def optimize_delta(path: str, z_order_columns: Optional[List[str]] = None) -> None:
    _local_delta().optimize(path, z_order_columns or None)


def delta_write_summary(path: str) -> dict:
    """Query Delta table metadata for user-facing write summary."""
    from customer_retention.core.compat.detection import get_default_parallelism, get_spark_session
    result: dict = {}
    cores = get_default_parallelism()
    if cores:
        result["cores"] = cores
    spark = get_spark_session()
    if spark:
        try:
            files = spark.read.format("delta").load(path).inputFiles()
            if isinstance(files, (list, tuple)):
                result["files"] = len(files)
        except Exception:
            pass
    if "files" not in result:
        try:
            import deltalake
            result["files"] = len(deltalake.DeltaTable(path).files())
        except Exception:
            pass
    return result


def _write_delta(df: Any, path: str, z_order_columns: Optional[List[str]] = None) -> None:
    if hasattr(df, "to_spark"):
        get_delta().write(as_spark_df(df), path, mode="overwrite", z_order_columns=z_order_columns)
        return
    _local_delta().write(
        normalize_timestamps(_compat_to_pandas(df)), path, mode="overwrite",
        z_order_columns=z_order_columns,
    )


def save_active_dataset(
    namespace: RunNamespace,
    dataset_name: str,
    df: Any,
    z_order_columns: Optional[List[str]] = None,
) -> Path:
    dlt_path = namespace.landing_table_dir(dataset_name)
    _write_delta(df, str(dlt_path))
    if z_order_columns:
        optimize_delta(str(dlt_path), z_order_columns)
    return dlt_path


def load_active_dataset(namespace: RunNamespace, dataset_name: str) -> Any:
    dlt_path = namespace.landing_table_dir(dataset_name)
    if not dlt_path.is_dir():
        raise FileNotFoundError(f"Active dataset not found: {dlt_path}")
    return _to_native_pandas(_local_delta().read(str(dlt_path)))


def save_aggregated_dataset(
    namespace: RunNamespace,
    dataset_name: str,
    df: Any,
    z_order_columns: Optional[List[str]] = None,
) -> Path:
    dlt_path = namespace.bronze_table_dir(dataset_name)
    _write_delta(df, str(dlt_path))
    if z_order_columns:
        optimize_delta(str(dlt_path), z_order_columns)
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


def require_silver_merged(namespace: RunNamespace) -> Any:
    silver = namespace.silver_merged_path
    if not silver.is_dir():
        raise FileNotFoundError(
            f"Silver merged dataset not found: {silver}. Run notebook 03 (dataset_merge) first."
        )
    return _to_native_pandas(_local_delta().read(str(silver)))


def require_silver_merged_distributed(namespace: RunNamespace) -> Any:
    silver = namespace.silver_merged_path
    if not silver.is_dir():
        raise FileNotFoundError(
            f"Silver merged dataset not found: {silver}. Run notebook 03 (dataset_merge) first."
        )
    return get_delta().read(str(silver))


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
    z_cols = [c for c in ("entity_id", "as_of_date") if c in df.columns]
    _write_delta(df, str(dlt_path), z_order_columns=z_cols or None)
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


def load_bridge_distributed(
    namespace: RunNamespace,
    dataset_name: str,
    project_ctx: "Optional[ProjectContext]" = None,
) -> Any:
    dlt_path = namespace.landing_table_dir(dataset_name)
    if dlt_path.is_dir():
        return get_delta().read(str(dlt_path))
    if project_ctx is None or dataset_name not in project_ctx.datasets:
        raise FileNotFoundError(
            f"Bridge dataset '{dataset_name}' not found in landing ({dlt_path}) "
            f"and no raw source available. Ensure the bridge dataset's NB01 "
            f"completes before datasets that depend on it for key resolution."
        )
    entry = project_ctx.datasets[dataset_name]
    logger.info(
        "Bridge '%s' not in landing yet — loading from raw source: %s",
        dataset_name, entry.path,
    )
    return _load_raw_source(entry.path, entry.storage_format)


def _load_raw_source(path: str, storage_format: str) -> Any:
    from customer_retention.analysis.auto_explorer.dataset_fingerprinter import is_table_name
    from customer_retention.core.compat import as_pandas_api, load_spark_table, native_pd
    from customer_retention.core.compat.detection import get_spark_session

    if is_table_name(path):
        return as_pandas_api(load_spark_table(path))
    spark = get_spark_session()
    fmt = storage_format or "csv"
    if spark:
        if fmt == "parquet":
            return as_pandas_api(spark.read.parquet(path))
        if fmt == "delta":
            return as_pandas_api(spark.read.format("delta").load(path))
        return as_pandas_api(
            spark.read.option("header", "true").option("inferSchema", "true").csv(path),
        )
    if fmt == "parquet":
        return native_pd.read_parquet(path)
    return native_pd.read_csv(path)


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


def merge_datasets_incremental(
    namespace: RunNamespace,
    spine_sdf: Any,
    datasets: "list[Any]",
    merger: Any,
) -> Any:
    """Incrementally merge datasets via Delta checkpointing.

    Instead of accumulating a wide Spark plan and relying on
    ``localCheckpoint``, this function writes the intermediate result to
    Delta after every dataset join, reads it back (breaking lineage
    completely), and releases all Spark resources before the next step.

    Parameters
    ----------
    namespace:
        Active ``RunNamespace`` — used to locate ``silver_merged_path``.
    spine_sdf:
        Native Spark DataFrame with the temporal spine.
    datasets:
        ``DatasetMergeInput`` objects to merge.
    merger:
        ``SparkTemporalMerger`` instance.

    Returns
    -------
    ``MergeReport`` with merge statistics.
    """
    from customer_retention.core.compat.detection import get_spark_session
    from customer_retention.stages.temporal.temporal_merger import MergeReport

    delta = get_delta()
    output_path = str(namespace.silver_merged_path)
    entity_key = merger.config.entity_key
    as_of_column = merger.config.as_of_column

    report = MergeReport(
        spine_rows=spine_sdf.count(),
        spine_entities=spine_sdf.select(entity_key).distinct().count(),
        spine_dates=spine_sdf.select(as_of_column).distinct().count(),
    )

    # Step 1: write spine to Delta
    delta.write(spine_sdf, output_path, mode="overwrite")
    del spine_sdf
    gc.collect()
    logger.info("Spine written to Delta (%d rows)", report.spine_rows)

    # Step 2: merge each dataset incrementally
    for i, ds in enumerate(datasets):
        t0 = time.monotonic()

        spark = get_spark_session()
        current_sdf = spark.read.format("delta").load(output_path)

        merged_sdf, new_cols = merger.merge_one(current_sdf, ds)

        delta.write(merged_sdf, output_path, mode="overwrite")

        # Release all Spark resources
        del current_sdf, merged_sdf
        release_stage_memory()

        report.datasets_merged.append(ds.name)
        report.columns_per_dataset[ds.name] = len(new_cols)

        logger.info(
            "Incremental merge %d/%d '%s': +%d cols (%.1fs)",
            i + 1, len(datasets), ds.name, len(new_cols),
            time.monotonic() - t0,
        )

    # Step 3: compact + Z-ORDER by entity/temporal columns
    z_cols = [c for c in [entity_key, as_of_column] if c]
    delta.optimize(output_path, z_cols or None)
    logger.info("OPTIMIZE complete (Z-ORDER: %s)", z_cols or "compaction-only")

    # Step 4: read final schema for report
    spark = get_spark_session()
    final_sdf = spark.read.format("delta").load(output_path)
    report.total_columns = len(final_sdf.columns)
    report.renamed_columns = {
        col: col
        for col in final_sdf.columns
        if merger.config.conflict_separator in col
    }
    del final_sdf
    gc.collect()

    return report
