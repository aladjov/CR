from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import deltalake
    DELTA_RS_AVAILABLE = True
except ImportError:
    DELTA_RS_AVAILABLE = False


def read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def read_delta(path: str, version: Optional[int] = None) -> pd.DataFrame:
    if not DELTA_RS_AVAILABLE:
        raise ImportError("deltalake package required: pip install deltalake")
    if version is not None:
        dt = deltalake.DeltaTable(path, version=version)
    else:
        dt = deltalake.DeltaTable(path)
    return dt.to_pandas()


def write_delta(df: pd.DataFrame, path: str, mode: str = "overwrite",
                partition_by: Optional[List[str]] = None) -> None:
    if not DELTA_RS_AVAILABLE:
        raise ImportError("deltalake package required: pip install deltalake")
    from deltalake import write_deltalake
    write_deltalake(path, df, mode=mode, partition_by=partition_by)


def get_missing_stats(df: pd.DataFrame) -> Dict[str, float]:
    return (df.isnull().sum() / len(df)).to_dict()


def correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns:
        return df[columns].corr()
    return df.select_dtypes(include=["number"]).corr()


def get_dtype_info(df: pd.DataFrame) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def sample(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=random_state)


def concat(dfs: List[pd.DataFrame], axis: int = 0, ignore_index: bool = True) -> pd.DataFrame:
    if axis == 1:
        return pd.concat(dfs, axis=axis, ignore_index=False)
    return pd.concat(dfs, axis=axis, ignore_index=ignore_index)
