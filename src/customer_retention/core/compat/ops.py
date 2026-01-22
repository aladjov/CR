from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from . import pandas_backend
from .detection import is_spark_available


class DataOps:
    def __init__(self):
        self._use_spark = is_spark_available()

    def _get_backend(self) -> Any:
        if self._use_spark:
            from . import spark_backend
            return spark_backend
        return pandas_backend

    def read_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        return self._get_backend().read_csv(path, **kwargs)

    def read_delta(self, path: str, version: Optional[int] = None) -> pd.DataFrame:
        return self._get_backend().read_delta(path, version=version)

    def write_delta(self, df: Union[pd.DataFrame, Any], path: str, mode: str = "overwrite",
                    partition_by: Optional[List[str]] = None) -> None:
        self._get_backend().write_delta(df, path, mode=mode, partition_by=partition_by)

    def get_missing_stats(self, df: Union[pd.DataFrame, Any]) -> Dict[str, float]:
        return self._get_backend().get_missing_stats(df)

    def correlation_matrix(self, df: Union[pd.DataFrame, Any],
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        return self._get_backend().correlation_matrix(df, columns=columns)

    def get_dtype_info(self, df: Union[pd.DataFrame, Any]) -> Dict[str, str]:
        return self._get_backend().get_dtype_info(df)

    def sample(self, df: Union[pd.DataFrame, Any], n: int, random_state: int = 42) -> pd.DataFrame:
        return self._get_backend().sample(df, n=n, random_state=random_state)

    def concat(self, dfs: List[Union[pd.DataFrame, Any]], axis: int = 0) -> pd.DataFrame:
        return self._get_backend().concat(dfs, axis=axis)


ops = DataOps()
