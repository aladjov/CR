from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DeltaStorage

try:
    import deltalake
    from deltalake import DeltaTable, write_deltalake
    DELTA_RS_AVAILABLE = True
except ImportError:
    DELTA_RS_AVAILABLE = False


class LocalDelta(DeltaStorage):
    def __init__(self):
        if not DELTA_RS_AVAILABLE:
            raise ImportError("deltalake package required: pip install deltalake")

    def read(self, path: str, version: Optional[int] = None) -> pd.DataFrame:
        if version is not None:
            dt = DeltaTable(path, version=version)
        else:
            dt = DeltaTable(path)
        return dt.to_pandas()

    def write(self, df: pd.DataFrame, path: str, mode: str = "overwrite",
              partition_by: Optional[List[str]] = None) -> None:
        write_deltalake(path, df, mode=mode, partition_by=partition_by)

    def merge(self, df: pd.DataFrame, path: str, condition: str,
              update_cols: Optional[List[str]] = None) -> None:
        dt = DeltaTable(path)
        merge_builder = dt.merge(df, predicate=condition, source_alias="source", target_alias="target")
        if update_cols:
            update_dict = {col: f"source.{col}" for col in update_cols}
            merge_builder = merge_builder.when_matched_update(updates=update_dict)
        else:
            merge_builder = merge_builder.when_matched_update_all()
        merge_builder.when_not_matched_insert_all().execute()

    def history(self, path: str) -> List[Dict[str, Any]]:
        dt = DeltaTable(path)
        return dt.history()

    def vacuum(self, path: str, retention_hours: int = 168) -> None:
        dt = DeltaTable(path)
        dt.vacuum(retention_hours=retention_hours, enforce_retention_duration=False, dry_run=False)
