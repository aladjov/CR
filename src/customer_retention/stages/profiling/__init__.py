from customer_retention.core.components.enums import Severity

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
from .categorical_target_analyzer import CategoricalTargetAnalyzer, CategoricalTargetResult
from .feature_capacity import (
    EffectiveFeaturesResult,
    FeatureCapacityAnalyzer,
    FeatureCapacityResult,
    ModelComplexityGuidance,
    SegmentCapacityResult,
)
from .pattern_analysis_config import (
    PatternAnalysisConfig,
    PatternAnalysisResult,
    SparklineData,
    SparklineDataBuilder,
    get_analysis_frequency,
    get_sparkline_frequency,
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
    TemporalCoverageResult,
    TemporalGap,
    analyze_temporal_coverage,
    derive_drift_implications,
)
from .temporal_feature_analyzer import (
    FeatureRecommendation,
    FeatureType,
    LagCorrelationResult,
    MomentumResult,
    PredictivePowerResult,
    TemporalFeatureAnalyzer,
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
    RecencyResult,
    SeasonalityPeriod,
    TemporalPatternAnalysis,
    TemporalPatternAnalyzer,
    TrendDirection,
    TrendResult,
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
    "TemporalTargetAnalyzer", "TemporalTargetResult",
    "TargetLevelAnalyzer", "TargetLevelResult", "TargetLevel", "AggregationMethod",
    "TargetDistribution", "TargetColumnDetector",
    "PatternAnalysisConfig", "PatternAnalysisResult",
    "SparklineData", "SparklineDataBuilder",
    "get_analysis_frequency", "get_sparkline_frequency",
    "TimeSeriesProfiler", "TimeSeriesProfile", "DistributionStats", "EntityLifecycle",
    "LifecycleQuadrantResult", "classify_lifecycle_quadrants",
    "ActivitySegmentResult", "classify_activity_segments",
    "TemporalQualityCheck", "TemporalQualityReporter", "TemporalQualityResult", "TemporalQualityScore",
    "DuplicateEventCheck", "TemporalGapCheck", "FutureDateCheck", "EventOrderCheck",
    "TemporalPatternAnalyzer", "TemporalPatternAnalysis",
    "TrendResult", "TrendDirection", "SeasonalityPeriod", "RecencyResult",
    "RelationshipDetector", "DatasetRelationship", "RelationshipType", "JoinSuggestion",
    "TimeWindowAggregator", "AggregationPlan", "TimeWindow", "AggregationType",
    "TemporalFeatureAnalyzer", "VelocityResult", "MomentumResult",
    "LagCorrelationResult", "PredictivePowerResult", "FeatureRecommendation", "FeatureType",
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
]
