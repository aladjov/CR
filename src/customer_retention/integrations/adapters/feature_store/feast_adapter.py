from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import AdapterResult
from .base import FeatureStoreAdapter, FeatureViewConfig


class FeastAdapter(FeatureStoreAdapter):
    def __init__(self, repo_path: str = "./feature_store/feature_repo"):
        self._repo_path = repo_path
        self._store = None
        self._feature_views: Dict[str, FeatureViewConfig] = {}
        self._data_sources: Dict[str, pd.DataFrame] = {}

    @property
    def store(self):
        if self._store is None:
            from feast import FeatureStore
            self._store = FeatureStore(repo_path=self._repo_path)
        return self._store

    def register_feature_view(self, config: FeatureViewConfig, df: pd.DataFrame) -> str:
        self._feature_views[config.name] = config
        self._data_sources[config.name] = df
        data_path = Path(self._repo_path) / "data" / f"{config.name}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        return config.name

    def get_historical_features(self, entity_df: pd.DataFrame, feature_refs: List[str]) -> pd.DataFrame:
        return self.store.get_historical_features(entity_df=entity_df, features=feature_refs).to_df()

    def materialize(self, feature_views: List[str], start_date: str, end_date: str) -> None:
        self.store.materialize(
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            feature_views=feature_views
        )

    def get_online_features(self, entity_keys: Dict[str, List[Any]], feature_refs: List[str]) -> Dict:
        entity_rows = [{k: v[i] for k, v in entity_keys.items()} for i in range(len(next(iter(entity_keys.values()))))]
        return self.store.get_online_features(features=feature_refs, entity_rows=entity_rows).to_dict()

    def create_table(self, name: str, schema: Dict[str, str], primary_keys: List[str]) -> AdapterResult:
        config = FeatureViewConfig(name=name, entity_key=primary_keys[0], features=list(schema.keys()))
        self._feature_views[name] = config
        return AdapterResult(success=True, metadata={"name": name})

    def write_table(self, name: str, df: pd.DataFrame, mode: str = "merge") -> AdapterResult:
        if name not in self._feature_views:
            return AdapterResult(success=False, error=f"Feature view {name} not found")
        config = self._feature_views[name]
        self.register_feature_view(config, df)
        return AdapterResult(success=True)

    def read_table(self, name: str, version: Optional[int] = None) -> pd.DataFrame:
        if name not in self._data_sources:
            data_path = Path(self._repo_path) / "data" / f"{name}.parquet"
            if data_path.exists():
                return pd.read_parquet(data_path)
            raise KeyError(f"Feature view {name} not found")
        return self._data_sources[name]

    def get_table_metadata(self, name: str) -> Dict[str, Any]:
        if name not in self._feature_views:
            raise KeyError(f"Feature view {name} not found")
        config = self._feature_views[name]
        return {"name": config.name, "entity_key": config.entity_key, "features": config.features, "ttl_days": config.ttl_days}

    def list_tables(self) -> List[str]:
        return list(self._feature_views.keys())

    def delete_table(self, name: str) -> AdapterResult:
        if name not in self._feature_views:
            return AdapterResult(success=False, error=f"Feature view {name} not found")
        del self._feature_views[name]
        if name in self._data_sources:
            del self._data_sources[name]
        return AdapterResult(success=True)
