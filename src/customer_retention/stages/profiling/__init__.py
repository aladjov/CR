from customer_retention.core.components.enums import Severity
from customer_retention.core.utils import compute_effect_size

from .column_profiler import CategoricalProfiler, ColumnProfiler, NumericProfiler, ProfilerFactory
from .distribution_analysis import (
    DistributionAnalysis,
    DistributionAnalyzer,
    DistributionTransformationType,
    TransformationRecommendation,
)
from .drift_detector import BaselineDriftChecker
from .drift_detector import DriftResult as BaselineDriftResult
from .profile_result import (
    BinaryMetrics,
    CategoricalMetrics,
    ColumnProfile,
    DatetimeMetrics,
    GranularityResult,
    IdentifierMetrics,
    NumericMetrics,
    ProfileResult,
    TargetMetrics,
    TextMetrics,
    TypeConfidence,
    TypeInference,
    UniversalMetrics,
)
from .quality_checks import QualityCheck, QualityCheckRegistry, QualityCheckResult
from .report_generator import ReportGenerator
from .scd_analyzer import SCDAnalyzer, SCDResult
from .type_detector import TypeDetector

# Backward compatibility alias
TransformationType = DistributionTransformationType
from .categorical_distribution import (
    CategoricalDistributionAnalysis,
    CategoricalDistributionAnalyzer,
    EncodingRecommendation,
    EncodingType,
)
from .categorical_target_analyzer import (
    CategoricalAnalysisResult,
    CategoricalFeatureInsight,
    CategoricalTargetAnalyzer,
    CategoricalTargetResult,
    analyze_categorical_features,
    filter_categorical_columns,
)
from .feature_capacity import (
    EffectiveFeaturesResult,
    FeatureCapacityAnalyzer,
    FeatureCapacityResult,
    ModelComplexityGuidance,
    SegmentCapacityResult,
)
from .pattern_analysis_config import (
    AggregationFeatureConfig,
    FindingsValidationResult,
    PatternAnalysisConfig,
    PatternAnalysisResult,
    SparklineData,
    SparklineDataBuilder,
    create_momentum_ratio_features,
    create_recency_bucket_feature,
    deduplicate_events,
    get_analysis_frequency,
    get_duplicate_event_count,
    get_sparkline_frequency,
    validate_temporal_findings,
)
from .relationship_detector import DatasetRelationship, JoinSuggestion, RelationshipDetector, RelationshipType
from .relationship_recommender import (
    RecommendationCategory,
    RelationshipAnalysisSummary,
    RelationshipRecommendation,
    RelationshipRecommender,
)
from .segment_analyzer import (
    ClusterVisualizationResult,
    DimensionReductionMethod,
    FullSegmentationResult,
    SegmentAnalyzer,
    SegmentationDecisionMetrics,
    SegmentationMethod,
    SegmentationResult,
    SegmentProfile,
)
from .segment_aware_outlier import SegmentAwareOutlierAnalyzer, SegmentAwareOutlierResult
from .target_level_analyzer import (
    AggregationMethod,
    TargetColumnDetector,
    TargetDistribution,
    TargetLevel,
    TargetLevelAnalyzer,
    TargetLevelResult,
)
from .temporal_analyzer import (
    SeasonalityResult,
    TemporalAnalysis,
    TemporalAnalyzer,
    TemporalGranularity,
    TemporalRecommendation,
    TemporalRecommendationType,
)
from .temporal_coverage import (
    DriftImplication,
    EntityWindowCoverage,
    FeatureAvailability,
    FeatureAvailabilityResult,
    TemporalCoverageResult,
    TemporalGap,
    analyze_feature_availability,
    analyze_temporal_coverage,
    derive_drift_implications,
)
from .temporal_feature_analyzer import (
    CohortMomentumResult,
    CohortVelocityResult,
    FeatureRecommendation,
    FeatureType,
    LagCorrelationResult,
    MomentumResult,
    PredictivePowerResult,
    TemporalFeatureAnalyzer,
    VelocityRecommendation,
    VelocityResult,
)
from .temporal_feature_engineer import (
    FeatureGroup,
    FeatureGroupResult,
    ReferenceMode,
    TemporalAggregationConfig,
    TemporalFeatureEngineer,
    TemporalFeatureResult,
)
from .temporal_pattern_analyzer import (
    AnomalyDiagnostics,
    CohortDistribution,
    CohortRecommendation,
    GroupStats,
    RecencyBucketStats,
    RecencyComparisonResult,
    RecencyInsight,
    RecencyResult,
    SeasonalityPeriod,
    TemporalPatternAnalysis,
    TemporalPatternAnalyzer,
    TrendDirection,
    TrendRecommendation,
    TrendResult,
    analyze_cohort_distribution,
    classify_distribution_pattern,
    compare_recency_by_target,
    compute_group_stats,
    compute_recency_buckets,
    detect_inflection_bucket,
    generate_cohort_recommendations,
    generate_recency_insights,
    generate_trend_recommendations,
)
from .temporal_quality_checks import (
    DuplicateEventCheck,
    EventOrderCheck,
    FutureDateCheck,
    TemporalGapCheck,
    TemporalQualityCheck,
    TemporalQualityReporter,
    TemporalQualityResult,
    TemporalQualityScore,
)
from .temporal_target_analyzer import TemporalTargetAnalyzer, TemporalTargetResult
from .text_embedder import EMBEDDING_MODELS, TextEmbedder, get_model_info, list_available_models
from .text_processor import TextColumnProcessor, TextColumnResult, TextProcessingConfig
from .text_reducer import ReductionResult, TextDimensionalityReducer
from .time_series_profiler import (
    ActivitySegmentResult,
    DistributionStats,
    EntityLifecycle,
    LifecycleQuadrantResult,
    TimeSeriesProfile,
    TimeSeriesProfiler,
    classify_activity_segments,
    classify_lifecycle_quadrants,
)
from .time_window_aggregator import AggregationPlan, AggregationType, TimeWindow, TimeWindowAggregator
from .window_recommendation import (
    TemporalHeterogeneityResult,
    WindowRecommendationCollector,
    WindowUnionResult,
)

__all__ = [
    "Severity",
    "ProfileResult", "ColumnProfile", "TypeInference", "TypeConfidence",
    "UniversalMetrics", "NumericMetrics", "CategoricalMetrics",
    "DatetimeMetrics", "BinaryMetrics", "IdentifierMetrics", "TargetMetrics", "TextMetrics",
    "GranularityResult",
    "TypeDetector",
    "ColumnProfiler", "ProfilerFactory", "NumericProfiler", "CategoricalProfiler",
    "QualityCheck", "QualityCheckResult", "QualityCheckRegistry",
    "BaselineDriftChecker", "BaselineDriftResult",
    "ReportGenerator",
    "SCDAnalyzer", "SCDResult",
    "DistributionAnalyzer", "DistributionAnalysis",
    "TransformationRecommendation", "DistributionTransformationType", "TransformationType",
    "CategoricalDistributionAnalyzer", "CategoricalDistributionAnalysis",
    "EncodingRecommendation", "EncodingType",
    "TemporalAnalyzer", "TemporalAnalysis", "TemporalGranularity", "SeasonalityResult",
    "TemporalRecommendation", "TemporalRecommendationType",
    "SegmentAnalyzer", "SegmentationResult", "SegmentProfile", "SegmentationMethod",
    "DimensionReductionMethod", "ClusterVisualizationResult",
    "SegmentationDecisionMetrics", "FullSegmentationResult",
    "SegmentAwareOutlierAnalyzer", "SegmentAwareOutlierResult",
    "CategoricalTargetAnalyzer", "CategoricalTargetResult",
    "CategoricalAnalysisResult", "CategoricalFeatureInsight",
    "analyze_categorical_features", "filter_categorical_columns",
    "TemporalTargetAnalyzer", "TemporalTargetResult",
    "TargetLevelAnalyzer", "TargetLevelResult", "TargetLevel", "AggregationMethod",
    "TargetDistribution", "TargetColumnDetector",
    "PatternAnalysisConfig", "PatternAnalysisResult",
    "SparklineData", "SparklineDataBuilder",
    "get_analysis_frequency", "get_sparkline_frequency",
    "AggregationFeatureConfig", "FindingsValidationResult", "validate_temporal_findings",
    "get_duplicate_event_count", "deduplicate_events",
    "create_recency_bucket_feature", "create_momentum_ratio_features",
    "TimeSeriesProfiler", "TimeSeriesProfile", "DistributionStats", "EntityLifecycle",
    "LifecycleQuadrantResult", "classify_lifecycle_quadrants",
    "ActivitySegmentResult", "classify_activity_segments",
    "TemporalQualityCheck", "TemporalQualityReporter", "TemporalQualityResult", "TemporalQualityScore",
    "DuplicateEventCheck", "TemporalGapCheck", "FutureDateCheck", "EventOrderCheck",
    "TemporalPatternAnalyzer", "TemporalPatternAnalysis",
    "TrendResult", "TrendDirection", "TrendRecommendation", "SeasonalityPeriod", "RecencyResult",
    "RecencyComparisonResult", "RecencyBucketStats", "RecencyInsight", "AnomalyDiagnostics",
    "GroupStats", "CohortDistribution", "CohortRecommendation",
    "generate_trend_recommendations", "generate_cohort_recommendations", "generate_recency_insights",
    "analyze_cohort_distribution", "compare_recency_by_target",
    "compute_effect_size", "compute_group_stats", "compute_recency_buckets",
    "detect_inflection_bucket", "classify_distribution_pattern",
    "RelationshipDetector", "DatasetRelationship", "RelationshipType", "JoinSuggestion",
    "TimeWindowAggregator", "AggregationPlan", "TimeWindow", "AggregationType",
    "TemporalFeatureAnalyzer", "VelocityResult", "MomentumResult", "CohortVelocityResult",
    "CohortMomentumResult", "VelocityRecommendation", "LagCorrelationResult", "PredictivePowerResult", "FeatureRecommendation", "FeatureType",
    "RelationshipRecommender", "RelationshipRecommendation",
    "RecommendationCategory", "RelationshipAnalysisSummary",
    "FeatureCapacityAnalyzer", "FeatureCapacityResult",
    "SegmentCapacityResult", "EffectiveFeaturesResult", "ModelComplexityGuidance",
    "TemporalFeatureEngineer", "TemporalAggregationConfig",
    "ReferenceMode", "FeatureGroup", "FeatureGroupResult", "TemporalFeatureResult",
    "TextEmbedder", "EMBEDDING_MODELS", "get_model_info", "list_available_models",
    "TextDimensionalityReducer", "ReductionResult",
    "TextColumnProcessor", "TextProcessingConfig", "TextColumnResult",
    "WindowRecommendationCollector", "WindowUnionResult", "TemporalHeterogeneityResult",
    "analyze_temporal_coverage", "TemporalCoverageResult", "TemporalGap", "EntityWindowCoverage",
    "derive_drift_implications", "DriftImplication",
    "analyze_feature_availability", "FeatureAvailability", "FeatureAvailabilityResult",
]
