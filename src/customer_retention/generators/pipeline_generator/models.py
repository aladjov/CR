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
    FILTER = "filter"
    LANDING_FILTER = "landing_filter"
    LANDING_LIFECYCLE_ENRICHMENT = "landing_lifecycle_enrichment"


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
    text_features: List["TextFeatureConfig"] = field(default_factory=list)


@dataclass
class KeyResolutionStepConfig:
    bridge_dataset: str
    source_key: str
    bridge_key: str
    resolve_column: str


@dataclass
class TemporalMergeSourceConfig:
    name: str
    granularity: str
    feature_timestamp_column: Optional[str] = None
    key_resolution_steps: List[KeyResolutionStepConfig] = field(default_factory=list)


@dataclass
class SilverLayerConfig:
    joins: List[Dict[str, str]] = field(default_factory=list)
    aggregations: List[Dict[str, Any]] = field(default_factory=list)
    derived_columns: List[TransformationStep] = field(default_factory=list)
    grid_dates: List[str] = field(default_factory=list)
    entity_key: Optional[str] = None
    merge_sources: List[TemporalMergeSourceConfig] = field(default_factory=list)
    holdout_entity_ids: Optional[List] = None


@dataclass
class GoldLayerConfig:
    encodings: List[TransformationStep] = field(default_factory=list)
    scalings: List[TransformationStep] = field(default_factory=list)
    feature_selections: List[str] = field(default_factory=list)
    feature_exclusion_prefixes: List[str] = field(default_factory=list)
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
    categorical_columns: List[str] = field(default_factory=list)
    categorical_agg_funcs: List[str] = field(default_factory=list)
    binary_columns: List[str] = field(default_factory=list)
    binary_agg_funcs: List[str] = field(default_factory=lambda: ["rate", "count", "any"])
    reference_date: Optional[str] = None
    column_blocked_funcs: Dict[str, List[str]] = field(default_factory=dict)
    sparse_prune_threshold: float = 2.0
    categorical_value_counts: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class LifecycleConfig:
    include_lifecycle_quadrant: bool = False
    include_cyclical_features: bool = False
    include_recency_bucket: bool = False
    momentum_pairs: List[Dict[str, Any]] = field(default_factory=list)
    include_trend_features: bool = False
    include_cohort_features: bool = False
    include_month_cyclical: bool = False
    include_quarter_cyclical: bool = False
    recency_bucket_edges: List[float] = field(default_factory=lambda: [0, 7, 30, 90, 180])
    recency_bucket_labels: List[str] = field(
        default_factory=lambda: ["0-7d", "8-30d", "31-90d", "91-180d", ">180d"]
    )


@dataclass
class TemporalFeatureConfig:
    lag_window_days: int = 30
    num_lags: int = 4
    lag_columns: List[str] = field(default_factory=list)
    lag_agg_funcs: List[str] = field(default_factory=lambda: ["sum", "mean", "count", "max"])
    feature_groups: List[str] = field(default_factory=lambda: [
        "lagged_windows", "velocity", "acceleration", "lifecycle",
        "recency", "regularity", "cohort_comparison",
    ])
    time_column_known: bool = True

    _LAG_FAMILY = frozenset({"lagged_windows", "velocity", "acceleration", "lifecycle", "cohort_comparison"})
    _TIME_ONLY_FAMILY = frozenset({"recency", "regularity"})

    def has_renderable_content(self) -> bool:
        groups = set(self.feature_groups or [])
        if self.lag_columns and (groups & self._LAG_FAMILY):
            return True
        if self.time_column_known and (groups & self._TIME_ONLY_FAMILY):
            return True
        return False


@dataclass
class TextFeatureConfig:
    column: str
    embedding_model: str = "all-MiniLM-L6-v2"
    n_components: int = 5
    component_columns: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    split_strategy: str = "temporal"
    test_size: float = 0.2
    random_state: int = 42
    temporal_column: Optional[str] = None
    purge_gap_days: Optional[int] = None
    recommended_training_start: Optional[str] = None
    filter_future_dates: bool = False
    imbalance_strategy: str = "class_weight"
    imbalance_ratio: Optional[float] = None
    exploration_feature_profile: Optional[Dict[str, Any]] = None
    best_model_type: Optional[str] = None
    production_cv_folds: Optional[int] = None
    production_full_panel_fit: bool = False
    feature_spec_path: Optional[str] = None
    production_internal_split_test_size: float = 0.1


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
class DatetimeDerivationConfig:
    source_columns: List[str]
    reference_column: str
    mask_future_columns: List[str] = field(default_factory=list)


@dataclass
class FeatureExclusion:
    column: str
    blocked_categories: List[str] = field(default_factory=list)
    blocked_funcs: List[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self):
        if not self.column:
            raise ValueError("FeatureExclusion.column must not be empty")


@dataclass
class DeduplicationConfig:
    strategy: str = "keep_first"
    key_columns: List[str] = field(default_factory=list)
    conflict_columns: List[str] = field(default_factory=list)


@dataclass
class BronzeEventConfig:
    source: SourceConfig
    entity_column: str
    time_column: str
    deduplicate: Any = False
    pre_shaping: List[TransformationStep] = field(default_factory=list)
    aggregation: Optional[AggregationWindowConfig] = None
    lifecycle: Optional[LifecycleConfig] = None
    post_shaping: List[TransformationStep] = field(default_factory=list)
    raw_time_column: Optional[str] = None
    datetime_derivation: Optional[DatetimeDerivationConfig] = None
    temporal_features: Optional[TemporalFeatureConfig] = None
    text_features: List[TextFeatureConfig] = field(default_factory=list)
    per_grid_date_mode: bool = False
    value_counts_columns: tuple = ()


@dataclass
class HistoryWindowConfig:
    time_column: str
    upper_limit: Optional[str] = None
    lookback_periods: Optional[int] = None
    cadence_days: int = 7


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
    datetime_derivation: Optional[DatetimeDerivationConfig] = None
    history_window: Optional[HistoryWindowConfig] = None
    key_resolution_steps: List[KeyResolutionStepConfig] = field(default_factory=list)
    filters: List[TransformationStep] = field(default_factory=list)
    lifecycle_enrichments: List[TransformationStep] = field(default_factory=list)
    drop_columns: List[str] = field(default_factory=list)


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
    training: Optional[TrainingConfig] = None
    landing: Dict[str, LandingLayerConfig] = field(default_factory=dict)
    bronze_event: Dict[str, BronzeEventConfig] = field(default_factory=dict)
    feature_spec_path: Optional[str] = None
