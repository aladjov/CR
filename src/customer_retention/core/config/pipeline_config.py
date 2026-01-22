from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from .source_config import DataSourceConfig


class TimestampStrategy(str, Enum):
    PRODUCTION = "production"
    SYNTHETIC_RANDOM = "synthetic_random"
    SYNTHETIC_INDEX = "synthetic_index"
    SYNTHETIC_FIXED = "synthetic_fixed"
    DERIVED = "derived"


class DedupStrategy(str, Enum):
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    KEEP_MOST_COMPLETE = "keep_most_complete"


class BronzeConfig(BaseModel):
    deduplicate: bool = True
    dedup_strategy: DedupStrategy = DedupStrategy.KEEP_LAST
    dedup_keys: list[str] = ["custid"]
    max_missing_pct: float = 0.95
    min_distinct_values: int = 2


class SilverConfig(BaseModel):
    entity_key: str = "custid"
    reference_date_column: Optional[str] = None
    auto_detect_encoding: bool = True
    auto_detect_scaling: bool = True


class GoldConfig(BaseModel):
    feature_store_catalog: str = "main"
    feature_store_schema: str = "feature_store"
    feature_table_name: str = "customer_features"
    version: str = "v1"

    def get_full_feature_table_name(self) -> str:
        return f"{self.feature_store_catalog}.{self.feature_store_schema}.{self.feature_table_name}"


class ModelingConfig(BaseModel):
    target_column: str = "retained"
    positive_class: int = 1
    test_size: float = 0.2
    stratify: bool = True
    primary_metric: str = "average_precision"
    cost_false_negative: float = 100.0
    cost_false_positive: float = 10.0

    def get_cost_ratio(self) -> float:
        return self.cost_false_negative / self.cost_false_positive


class ValidationConfig(BaseModel):
    fail_on_critical: bool = True
    fail_on_high: bool = False
    leakage_correlation_threshold: float = 0.90
    max_overfit_gap: float = 0.15


class TemporalConfig(BaseModel):
    timestamp_strategy: TimestampStrategy = TimestampStrategy.PRODUCTION
    feature_timestamp_column: Optional[str] = None
    label_timestamp_column: Optional[str] = None
    observation_window_days: int = 90
    synthetic_base_date: str = "2024-01-01"
    synthetic_range_days: int = 365
    snapshot_prefix: str = "ml_training_snapshot"
    enforce_point_in_time: bool = True
    max_feature_target_correlation: float = 0.90
    block_future_features: bool = True
    derive_label_from_feature: bool = False
    derivation_config: Optional[dict[str, Any]] = None


class PathConfig(BaseModel):
    landing_zone: Optional[str] = None
    bronze: Optional[str] = None
    silver: Optional[str] = None
    gold: Optional[str] = None


class PipelineConfig(BaseModel):
    project_name: str
    project_description: Optional[str] = None
    version: str = "1.0.0"

    data_sources: list[DataSourceConfig] = []
    bronze: BronzeConfig = BronzeConfig()
    silver: SilverConfig = SilverConfig()
    gold: GoldConfig = GoldConfig()
    modeling: ModelingConfig = ModelingConfig()
    validation: ValidationConfig = ValidationConfig()
    temporal: TemporalConfig = TemporalConfig()
    paths: PathConfig = PathConfig()

    def get_source_by_name(self, name: str) -> Optional[DataSourceConfig]:
        return next((s for s in self.data_sources if s.name == name), None)

    def get_target_source(self) -> Optional[DataSourceConfig]:
        for source in self.data_sources:
            if any(c.column_type.value == "target" for c in source.columns):
                return source
        return None

    def get_all_feature_columns(self) -> list[str]:
        feature_cols = []
        for source in self.data_sources:
            feature_cols.extend([c.name for c in source.get_feature_columns()])
        return feature_cols
