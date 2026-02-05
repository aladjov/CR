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
    DROP_COLUMN = "drop_column"
    WINSORIZE = "winsorize"
    SEGMENT_AWARE_CAP = "segment_aware_cap"
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    YEO_JOHNSON = "yeo_johnson"
    ZERO_INFLATION_HANDLING = "zero_inflation_handling"
    CAP_THEN_LOG = "cap_then_log"
    FEATURE_SELECT = "feature_select"
    DERIVED_COLUMN = "derived_column"


@dataclass
class SourceConfig:
    name: str
    path: str
    format: str
    entity_key: str
    raw_source_path: str = ""
    time_column: Optional[str] = None
    is_event_level: bool = False
    excluded: bool = False


@dataclass
class TransformationStep:
    type: PipelineTransformationType
    column: str
    parameters: Dict[str, Any]
    rationale: str
    source_notebook: Optional[str] = None


@dataclass
class BronzeLayerConfig:
    source: SourceConfig
    transformations: List[TransformationStep] = field(default_factory=list)
    lifecycle: Optional["LifecycleConfig"] = None
    entity_column: Optional[str] = None
    time_column: Optional[str] = None
    raw_time_column: Optional[str] = None


@dataclass
class SilverLayerConfig:
    joins: List[Dict[str, str]] = field(default_factory=list)
    aggregations: List[Dict[str, Any]] = field(default_factory=list)
    derived_columns: List[TransformationStep] = field(default_factory=list)


@dataclass
class GoldLayerConfig:
    encodings: List[TransformationStep] = field(default_factory=list)
    scalings: List[TransformationStep] = field(default_factory=list)
    feature_selections: List[str] = field(default_factory=list)
    transformations: List[TransformationStep] = field(default_factory=list)


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
class AggregationWindowConfig:
    windows: List[str] = field(default_factory=list)
    value_columns: List[str] = field(default_factory=list)
    agg_funcs: List[str] = field(default_factory=list)
    reference_date: Optional[str] = None


@dataclass
class LifecycleConfig:
    include_lifecycle_quadrant: bool = False
    include_cyclical_features: bool = False
    include_recency_bucket: bool = False
    momentum_pairs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TimestampCoalesceConfig:
    datetime_columns_ordered: List[str]
    output_column: str = "feature_timestamp"


@dataclass
class LabelTimestampConfig:
    label_column: Optional[str] = None
    fallback_window_days: int = 180
    output_column: str = "label_timestamp"


@dataclass
class BronzeEventConfig:
    source: SourceConfig
    entity_column: str
    time_column: str
    deduplicate: bool = False
    pre_shaping: List[TransformationStep] = field(default_factory=list)
    aggregation: Optional[AggregationWindowConfig] = None
    lifecycle: Optional[LifecycleConfig] = None
    post_shaping: List[TransformationStep] = field(default_factory=list)
    raw_time_column: Optional[str] = None


@dataclass
class LandingLayerConfig:
    source: SourceConfig
    raw_source_path: str
    raw_source_format: str
    entity_column: str
    time_column: str
    target_column: str
    original_target_column: Optional[str] = None
    raw_time_column: Optional[str] = None
    timestamp_coalesce: Optional[TimestampCoalesceConfig] = None
    label_timestamp: Optional[LabelTimestampConfig] = None


@dataclass
class PipelineConfig:
    name: str
    target_column: str
    sources: List[SourceConfig]
    bronze: Dict[str, BronzeLayerConfig]
    silver: SilverLayerConfig
    gold: GoldLayerConfig
    output_dir: str
    composite_name: Optional[str] = None
    iteration_id: Optional[str] = None
    parent_iteration_id: Optional[str] = None
    recommendations_hash: Optional[str] = None
    feast: Optional[FeastConfig] = None
    scoring: Optional[ScoringConfig] = None
    experiments_dir: Optional[str] = None
    production_dir: Optional[str] = None
    fit_mode: bool = True
    artifacts_path: Optional[str] = None
    landing: Dict[str, LandingLayerConfig] = field(default_factory=dict)
    bronze_event: Dict[str, BronzeEventConfig] = field(default_factory=dict)
