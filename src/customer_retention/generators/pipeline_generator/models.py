from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PipelineTransformationType(Enum):
    IMPUTE_NULL = "impute_null"
    CAP_OUTLIER = "cap_outlier"
    TYPE_CAST = "type_cast"
    ENCODE = "encode"
    SCALE = "scale"
    AGGREGATE = "aggregate"
    JOIN = "join"


@dataclass
class SourceConfig:
    name: str
    path: str
    format: str
    entity_key: str
    time_column: Optional[str] = None
    is_event_level: bool = False


@dataclass
class TransformationStep:
    type: PipelineTransformationType
    column: str
    parameters: Dict[str, Any]
    rationale: str


@dataclass
class BronzeLayerConfig:
    source: SourceConfig
    transformations: List[TransformationStep] = field(default_factory=list)


@dataclass
class SilverLayerConfig:
    joins: List[Dict[str, str]] = field(default_factory=list)
    aggregations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GoldLayerConfig:
    encodings: List[TransformationStep] = field(default_factory=list)
    scalings: List[TransformationStep] = field(default_factory=list)
    feature_selections: List[str] = field(default_factory=list)


@dataclass
class FeastConfig:
    repo_path: str = "./feature_repo"
    feature_view_name: str = "customer_features"
    entity_name: str = "customer"
    entity_key: str = "customer_id"
    timestamp_column: str = "event_timestamp"
    ttl_days: int = 365
    exclude_prefixes: List[str] = field(default_factory=lambda: ["original_"])


@dataclass
class ScoringConfig:
    holdout_manifest_path: Optional[str] = None
    original_column: Optional[str] = None
    model_uri: Optional[str] = None
    output_predictions_path: Optional[str] = None


@dataclass
class PipelineConfig:
    name: str
    target_column: str
    sources: List[SourceConfig]
    bronze: Dict[str, BronzeLayerConfig]
    silver: SilverLayerConfig
    gold: GoldLayerConfig
    output_dir: str
    iteration_id: Optional[str] = None
    parent_iteration_id: Optional[str] = None
    recommendations_hash: Optional[str] = None
    feast: Optional[FeastConfig] = None
    scoring: Optional[ScoringConfig] = None
    experiments_dir: Optional[str] = None
    fit_mode: bool = True
    artifacts_path: Optional[str] = None
