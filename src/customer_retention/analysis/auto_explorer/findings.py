import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from customer_retention.core.config.column_config import ColumnConfig, ColumnType, DatasetGranularity


def _convert_to_native(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if type(obj).__module__ == 'numpy':
        return obj.item() if hasattr(obj, 'item') else float(obj)
    return obj


@dataclass
class TimeSeriesMetadata:
    granularity: DatasetGranularity = DatasetGranularity.UNKNOWN
    temporal_pattern: Optional[str] = None  # TIME_SERIES, EVENT_LOG, SNAPSHOT
    entity_column: Optional[str] = None
    time_column: Optional[str] = None
    avg_events_per_entity: Optional[float] = None
    time_span_days: Optional[int] = None
    unique_entities: Optional[int] = None
    suggested_aggregations: List[str] = field(default_factory=list)
    window_coverage_threshold: Optional[float] = None
    heterogeneity_level: Optional[str] = None
    eta_squared_intensity: Optional[float] = None
    eta_squared_event_count: Optional[float] = None
    temporal_segmentation_advisory: Optional[str] = None
    temporal_segmentation_recommendation: Optional[str] = None
    drift_risk_level: Optional[str] = None
    volume_drift_risk: Optional[str] = None
    population_stability: Optional[float] = None
    regime_count: Optional[int] = None
    recommended_training_start: Optional[str] = None
    def populate_from_coverage(self, windows: list, coverage_threshold: float) -> None:
        self.suggested_aggregations = windows
        self.window_coverage_threshold = coverage_threshold

    def populate_from_heterogeneity(
        self, heterogeneity_level: str, eta_squared_intensity: float,
        eta_squared_event_count: float, segmentation_advisory: str,
    ) -> None:
        self.heterogeneity_level = heterogeneity_level
        self.eta_squared_intensity = eta_squared_intensity
        self.eta_squared_event_count = eta_squared_event_count
        self.temporal_segmentation_advisory = segmentation_advisory
        self.temporal_segmentation_recommendation = (
            "include_lifecycle_quadrant" if segmentation_advisory != "single_model" else None
        )

    def populate_from_drift(
        self, risk_level: str, volume_drift_risk: str,
        population_stability: float, regime_count: int,
        recommended_training_start: Optional[str],
    ) -> None:
        self.drift_risk_level = risk_level
        self.volume_drift_risk = volume_drift_risk
        self.population_stability = population_stability
        self.regime_count = regime_count
        self.recommended_training_start = recommended_training_start

    aggregation_executed: bool = False
    aggregated_data_path: Optional[str] = None
    aggregated_findings_path: Optional[str] = None
    aggregation_windows_used: List[str] = field(default_factory=list)
    aggregation_timestamp: Optional[str] = None


@dataclass
class TextProcessingMetadata:
    column_name: str
    embedding_model: str
    embedding_dim: int
    n_components: int
    explained_variance: float
    component_columns: List[str]
    variance_threshold_used: float
    processing_approach: str = "pca"


@dataclass
class FeatureAvailabilityInfo:
    first_valid_date: Optional[str]
    last_valid_date: Optional[str]
    coverage_pct: float
    availability_type: str
    days_from_start: Optional[int]
    days_before_end: Optional[int]


@dataclass
class FeatureAvailabilityMetadata:
    data_start: str
    data_end: str
    time_span_days: int
    new_tracking: List[str]
    retired_tracking: List[str]
    partial_window: List[str]
    features: Dict[str, FeatureAvailabilityInfo] = field(default_factory=dict)


@dataclass
class ColumnFinding:
    name: str
    inferred_type: ColumnType
    confidence: float
    evidence: List[str]
    alternatives: List[ColumnType] = field(default_factory=list)
    universal_metrics: Dict[str, Any] = field(default_factory=dict)
    type_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    cleaning_needed: bool = False
    cleaning_recommendations: List[str] = field(default_factory=list)
    transformation_recommendations: List[str] = field(default_factory=list)

    def to_column_config(self) -> ColumnConfig:
        return ColumnConfig(
            name=self.name,
            column_type=self.inferred_type,
            nullable=self.universal_metrics.get("null_count", 0) > 0
        )


@dataclass
class ExplorationFindings:
    source_path: str
    source_format: str
    exploration_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    row_count: int = 0
    column_count: int = 0
    memory_usage_mb: float = 0.0
    columns: Dict[str, ColumnFinding] = field(default_factory=dict)
    target_column: Optional[str] = None
    target_type: Optional[str] = None
    identifier_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    overall_quality_score: float = 100.0
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    modeling_ready: bool = False
    blocking_issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    time_series_metadata: Optional[TimeSeriesMetadata] = None
    text_processing: Dict[str, TextProcessingMetadata] = field(default_factory=dict)
    feature_availability: Optional[FeatureAvailabilityMetadata] = None
    iteration_id: Optional[str] = None
    parent_iteration_id: Optional[str] = None
    # Snapshot-related fields (from temporal framework)
    snapshot_id: Optional[str] = None
    snapshot_path: Optional[str] = None
    timestamp_scenario: Optional[str] = None
    timestamp_strategy: Optional[str] = None

    @property
    def is_time_series(self) -> bool:
        if self.time_series_metadata is None:
            return False
        return self.time_series_metadata.granularity == DatasetGranularity.EVENT_LEVEL

    @property
    def has_aggregated_output(self) -> bool:
        return (self.time_series_metadata is not None and
                self.time_series_metadata.aggregation_executed)

    @property
    def column_types(self) -> Dict[str, ColumnType]:
        return {name: col.inferred_type for name, col in self.columns.items()}

    @property
    def column_configs(self) -> Dict[str, ColumnConfig]:
        return {name: col.to_column_config() for name, col in self.columns.items()}

    @property
    def has_availability_issues(self) -> bool:
        if self.feature_availability is None:
            return False
        return bool(
            self.feature_availability.new_tracking
            or self.feature_availability.retired_tracking
            or self.feature_availability.partial_window
        )

    @property
    def problematic_availability_columns(self) -> List[str]:
        if self.feature_availability is None:
            return []
        return (
            self.feature_availability.new_tracking
            + self.feature_availability.retired_tracking
            + self.feature_availability.partial_window
        )

    def get_feature_availability(self, column: str) -> Optional[FeatureAvailabilityInfo]:
        if self.feature_availability is None:
            return None
        return self.feature_availability.features.get(column)

    @staticmethod
    def _normalize_enum_value(obj: Any) -> Any:
        return obj.value if hasattr(obj, 'value') else obj

    def to_dict(self) -> dict:
        result = _convert_to_native(asdict(self))
        for col_data in result.get("columns", {}).values():
            if "inferred_type" in col_data:
                col_data["inferred_type"] = self._normalize_enum_value(col_data["inferred_type"])
            if "alternatives" in col_data:
                col_data["alternatives"] = [self._normalize_enum_value(t) for t in col_data["alternatives"]]
        ts_meta = result.get("time_series_metadata")
        if ts_meta is not None and "granularity" in ts_meta:
            ts_meta["granularity"] = self._normalize_enum_value(ts_meta["granularity"])
        return result

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str):
        content = self.to_yaml() if path.endswith((".yaml", ".yml")) else self.to_json()
        with open(path, "w") as f:
            f.write(content)

    @classmethod
    def _deserialize_columns(cls, data: dict) -> Dict[str, "ColumnFinding"]:
        columns = {}
        for col_name, col_data in data.get("columns", {}).items():
            if "inferred_type" in col_data:
                col_data["inferred_type"] = ColumnType(col_data["inferred_type"])
            if "alternatives" in col_data:
                col_data["alternatives"] = [ColumnType(t) for t in col_data["alternatives"]]
            columns[col_name] = ColumnFinding(**col_data)
        return columns

    @classmethod
    def _deserialize_time_series_metadata(cls, ts_meta: Optional[dict]) -> Optional["TimeSeriesMetadata"]:
        if ts_meta is None:
            return None
        if "granularity" in ts_meta:
            ts_meta["granularity"] = DatasetGranularity(ts_meta["granularity"])
        return TimeSeriesMetadata(**ts_meta)

    @classmethod
    def _deserialize_feature_availability(cls, fa_data: Optional[dict]) -> Optional["FeatureAvailabilityMetadata"]:
        if fa_data is None:
            return None
        fa_data["features"] = {
            k: FeatureAvailabilityInfo(**v)
            for k, v in fa_data.get("features", {}).items()
        }
        return FeatureAvailabilityMetadata(**fa_data)

    @classmethod
    def from_dict(cls, data: dict) -> "ExplorationFindings":
        data["columns"] = cls._deserialize_columns(data)
        data["time_series_metadata"] = cls._deserialize_time_series_metadata(data.get("time_series_metadata"))
        data["text_processing"] = {k: TextProcessingMetadata(**v) for k, v in data.get("text_processing", {}).items()}
        data["feature_availability"] = cls._deserialize_feature_availability(data.get("feature_availability"))
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ExplorationFindings":
        return cls.from_dict(yaml.safe_load(yaml_str))

    @classmethod
    def from_json(cls, json_str: str) -> "ExplorationFindings":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str) -> "ExplorationFindings":
        with open(path, "r") as f:
            content = f.read()
        return cls.from_yaml(content) if path.endswith((".yaml", ".yml")) else cls.from_json(content)
