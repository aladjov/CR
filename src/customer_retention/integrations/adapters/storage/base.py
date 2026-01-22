from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class DeltaStorage(ABC):
    @abstractmethod
    def read(self, path: str, version: Optional[int] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def write(self, df: pd.DataFrame, path: str, mode: str = "overwrite",
              partition_by: Optional[List[str]] = None) -> None:
        pass

    @abstractmethod
    def merge(self, df: pd.DataFrame, path: str, condition: str,
              update_cols: Optional[List[str]] = None) -> None:
        pass

    @abstractmethod
    def history(self, path: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def vacuum(self, path: str, retention_hours: int = 168) -> None:
        pass
