from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import pandas as pd
from .base import FeatureStoreAdapter
from ..base import AdapterResult
from ..storage import LocalDelta


class LocalFeatureStore(FeatureStoreAdapter):
    def __init__(self, base_path: str = "./feature_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_path / "registry.json"
        self.storage = LocalDelta()
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self._registry = json.load(f)
        else:
            self._registry = {"tables": {}}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def _table_path(self, name: str) -> str:
        return str(self.base_path / "tables" / name)

    def create_table(self, name: str, schema: Dict[str, str], primary_keys: List[str]) -> AdapterResult:
        self._registry["tables"][name] = {
            "schema": schema,
            "primary_keys": primary_keys,
            "path": self._table_path(name)
        }
        self._save_registry()
        return AdapterResult(success=True, metadata={"name": name})

    def write_table(self, name: str, df: pd.DataFrame, mode: str = "merge") -> AdapterResult:
        if name not in self._registry["tables"]:
            return AdapterResult(success=False, error=f"Table {name} not found")
        table_info = self._registry["tables"][name]
        path = table_info["path"]
        if mode == "merge" and Path(path).exists():
            primary_keys = table_info["primary_keys"]
            condition = " AND ".join([f"source.{k} = target.{k}" for k in primary_keys])
            self.storage.merge(df, path, condition)
        else:
            self.storage.write(df, path, mode="overwrite" if mode == "merge" else mode)
        return AdapterResult(success=True)

    def read_table(self, name: str, version: Optional[int] = None) -> pd.DataFrame:
        if name not in self._registry["tables"]:
            raise KeyError(f"Table {name} not found")
        path = self._registry["tables"][name]["path"]
        return self.storage.read(path, version=version)

    def get_table_metadata(self, name: str) -> Dict[str, Any]:
        if name not in self._registry["tables"]:
            raise KeyError(f"Table {name} not found")
        return self._registry["tables"][name]

    def list_tables(self) -> List[str]:
        return list(self._registry["tables"].keys())

    def delete_table(self, name: str) -> AdapterResult:
        if name not in self._registry["tables"]:
            return AdapterResult(success=False, error=f"Table {name} not found")
        del self._registry["tables"][name]
        self._save_registry()
        return AdapterResult(success=True)
