from typing import Any, Dict, List, Optional

from customer_retention.core.compat import pd

from .base import DeltaStorage


def deltalake_available() -> bool:
    try:
        import deltalake  # noqa: F401
        return True
    except ImportError:
        return False


def _import_deltalake() -> Any:
    try:
        import deltalake
        return deltalake
    except ImportError:
        raise ImportError("deltalake package required: pip install deltalake") from None


def _coerce_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    import pyarrow as pa

    null_cols = [c for c in df.columns if pa.array(df[c]).type == pa.null()]
    if not null_cols:
        return df
    df = df.copy()
    for col in null_cols:
        df[col] = df[col].astype(pd.StringDtype())
    return df


class LocalDelta(DeltaStorage):
    def __init__(self):
        if not deltalake_available():
            raise ImportError("deltalake package required: pip install deltalake")

    def read(self, path: str, version: Optional[int] = None) -> pd.DataFrame:
        dl = _import_deltalake()
        if version is not None:
            dt = dl.DeltaTable(path, version=version)
        else:
            dt = dl.DeltaTable(path)
        return dt.to_pandas()

    def write(
        self,
        df: Any,
        path: str,
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        z_order_columns: Optional[List[str]] = None,
    ) -> None:
        if hasattr(df, "to_spark"):
            from .databricks import DatabricksDelta

            DatabricksDelta().write(df, path, mode, partition_by, metadata, z_order_columns)
            return
        df = _coerce_null_columns(df)
        dl = _import_deltalake()
        kwargs: Dict[str, Any] = {"mode": mode}
        if mode == "overwrite":
            kwargs["schema_mode"] = "overwrite"
        if partition_by:
            kwargs["partition_by"] = partition_by
        if metadata:
            kwargs["commit_properties"] = dl.CommitProperties(custom_metadata=metadata)
        dl.write_deltalake(path, df, **kwargs)
        if z_order_columns:
            self.optimize(path, z_order_columns)

    def merge(self, df: pd.DataFrame, path: str, condition: str, update_cols: Optional[List[str]] = None) -> None:
        dl = _import_deltalake()
        dt = dl.DeltaTable(path)
        merge_builder = dt.merge(df, predicate=condition, source_alias="source", target_alias="target")
        if update_cols:
            update_dict = {col: f"source.{col}" for col in update_cols}
            merge_builder = merge_builder.when_matched_update(updates=update_dict)
        else:
            merge_builder = merge_builder.when_matched_update_all()
        merge_builder.when_not_matched_insert_all().execute()

    def history(self, path: str) -> List[Dict[str, Any]]:
        dl = _import_deltalake()
        dt = dl.DeltaTable(path)
        return dt.history()

    def optimize(self, path: str, z_order_columns: Optional[List[str]] = None) -> None:
        dl = _import_deltalake()
        dt = dl.DeltaTable(path)
        if z_order_columns:
            dt.optimize.z_order(z_order_columns)
        else:
            dt.optimize.compact()

    def vacuum(self, path: str, retention_hours: int = 168) -> None:
        dl = _import_deltalake()
        dt = dl.DeltaTable(path)
        dt.vacuum(retention_hours=retention_hours, enforce_retention_duration=False, dry_run=False)

    def exists(self, path: str) -> bool:
        from pathlib import Path

        return Path(path).joinpath("_delta_log").is_dir()
