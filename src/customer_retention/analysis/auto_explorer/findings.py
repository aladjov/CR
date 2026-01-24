import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from customer_retention.core.config.column_config import ColumnConfig, ColumnType, DatasetGranularity


def _convert_to_native(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python types for serialization."""
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
    iteration_id: Optional[str] = None
    parent_iteration_id: Optional[str] = None
    # Snapshot-related fields (from temporal framework)
    snapshot_id: Optional[str] = None
    snapshot_path: Optional[str] = None
    timestamp_scenario: Optional[str] = None
    timestamp_strategy: Optional[str] = None

    @property
    def is_time_series(self) -> bool:
        """Check if this dataset is time series (event-level)."""
        if self.time_series_metadata is None:
            return False
        return self.time_series_metadata.granularity == DatasetGranularity.EVENT_LEVEL

    @property
    def has_aggregated_output(self) -> bool:
        """Check if this event-level dataset has been aggregated."""
        return (self.time_series_metadata is not None and
                self.time_series_metadata.aggregation_executed)

    @property
    def column_types(self) -> Dict[str, ColumnType]:
        return {name: col.inferred_type for name, col in self.columns.items()}

    @property
    def column_configs(self) -> Dict[str, ColumnConfig]:
        return {name: col.to_column_config() for name, col in self.columns.items()}

    def to_dict(self) -> dict:
        result = asdict(self)
        result = _convert_to_native(result)
        for col_name, col_data in result.get("columns", {}).items():
            if "inferred_type" in col_data:
                col_data["inferred_type"] = col_data["inferred_type"].value if hasattr(col_data["inferred_type"], 'value') else col_data["inferred_type"]
            if "alternatives" in col_data:
                col_data["alternatives"] = [t.value if hasattr(t, 'value') else t for t in col_data["alternatives"]]
        # Handle TimeSeriesMetadata serialization
        if result.get("time_series_metadata") is not None:
            ts_meta = result["time_series_metadata"]
            if "granularity" in ts_meta:
                ts_meta["granularity"] = ts_meta["granularity"].value if hasattr(ts_meta["granularity"], 'value') else ts_meta["granularity"]
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
    def from_dict(cls, data: dict) -> "ExplorationFindings":
        columns = {}
        for col_name, col_data in data.get("columns", {}).items():
            if "inferred_type" in col_data:
                col_data["inferred_type"] = ColumnType(col_data["inferred_type"])
            if "alternatives" in col_data:
                col_data["alternatives"] = [ColumnType(t) for t in col_data["alternatives"]]
            columns[col_name] = ColumnFinding(**col_data)
        data["columns"] = columns
        ts_meta = data.get("time_series_metadata")
        if ts_meta is not None:
            if "granularity" in ts_meta:
                ts_meta["granularity"] = DatasetGranularity(ts_meta["granularity"])
            data["time_series_metadata"] = TimeSeriesMetadata(**ts_meta)
        text_proc = data.get("text_processing", {})
        data["text_processing"] = {k: TextProcessingMetadata(**v) for k, v in text_proc.items()}
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
