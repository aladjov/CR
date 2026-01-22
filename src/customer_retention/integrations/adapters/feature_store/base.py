from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import AdapterResult


@dataclass
class FeatureViewConfig:
    name: str
    entity_key: str
    features: List[str]
    ttl_days: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)
    cutoff_date: Optional[datetime] = None
    data_hash: Optional[str] = None


class FeatureStoreAdapter(ABC):
    @abstractmethod
    def create_table(self, name: str, schema: Dict[str, str], primary_keys: List[str]) -> AdapterResult:
        pass

    @abstractmethod
    def write_table(self, name: str, df: pd.DataFrame, mode: str = "merge") -> AdapterResult:
        pass

    @abstractmethod
    def read_table(self, name: str, version: Optional[int] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_table_metadata(self, name: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_tables(self) -> List[str]:
        pass

    @abstractmethod
    def delete_table(self, name: str) -> AdapterResult:
        pass

    def register_feature_view(self, config: FeatureViewConfig, df: pd.DataFrame) -> str:
        raise NotImplementedError("Subclass must implement register_feature_view")

    def get_historical_features(self, entity_df: pd.DataFrame, feature_refs: List[str]) -> pd.DataFrame:
        raise NotImplementedError("Subclass must implement get_historical_features")

    def materialize(self, feature_views: List[str], start_date: str, end_date: str) -> None:
        raise NotImplementedError("Subclass must implement materialize")

    def get_online_features(self, entity_keys: Dict[str, List[Any]], feature_refs: List[str]) -> Dict:
        raise NotImplementedError("Subclass must implement get_online_features")
