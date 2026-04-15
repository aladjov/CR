from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DeltaStorage(ABC):
    @abstractmethod
    def read(self, path: str, version: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def write(self, df: Any, path: str, mode: str = "overwrite",
              partition_by: Optional[List[str]] = None,
              metadata: Optional[Dict[str, str]] = None,
              z_order_columns: Optional[List[str]] = None,
              target_partitions: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def merge(self, df: Any, path: str, condition: str,
              update_cols: Optional[List[str]] = None) -> None:
        pass

    @abstractmethod
    def history(self, path: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def optimize(self, path: str, z_order_columns: Optional[List[str]] = None) -> None:
        pass

    @abstractmethod
    def vacuum(self, path: str, retention_hours: int = 168) -> None:
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass
