from customer_retention.core.components.enums import Severity
from .profile_result import (
    ProfileResult, ColumnProfile, TypeInference, TypeConfidence,
    UniversalMetrics, NumericMetrics, CategoricalMetrics,
    DatetimeMetrics, BinaryMetrics, IdentifierMetrics, TargetMetrics, TextMetrics,
    GranularityResult
)
from .type_detector import TypeDetector
from .column_profiler import (
    ColumnProfiler, ProfilerFactory, NumericProfiler, CategoricalProfiler
)
from .quality_checks import (
    QualityCheck, QualityCheckResult, QualityCheckRegistry
)
from .drift_detector import BaselineDriftChecker, DriftResult as BaselineDriftResult
from .report_generator import ReportGenerator
from .scd_analyzer import SCDAnalyzer, SCDResult
from .distribution_analysis import (
    DistributionAnalyzer, DistributionAnalysis,
    TransformationRecommendation, DistributionTransformationType
)
# Backward compatibility alias
TransformationType = DistributionTransformationType
from .categorical_distribution import (
    CategoricalDistributionAnalyzer, CategoricalDistributionAnalysis,
    EncodingRecommendation, EncodingType
)
from .temporal_analyzer import (
    TemporalAnalyzer, TemporalAnalysis, TemporalGranularity, SeasonalityResult,
    TemporalRecommendation, TemporalRecommendationType
)
from .segment_analyzer import (
    SegmentAnalyzer, SegmentationResult, SegmentProfile, SegmentationMethod,
    DimensionReductionMethod, ClusterVisualizationResult,
    SegmentationDecisionMetrics, FullSegmentationResult
)
from .segment_aware_outlier import (
    SegmentAwareOutlierAnalyzer, SegmentAwareOutlierResult
)
from .categorical_target_analyzer import (
    CategoricalTargetAnalyzer, CategoricalTargetResult
)
from .temporal_target_analyzer import (
    TemporalTargetAnalyzer, TemporalTargetResult
)
from .time_series_profiler import (
    TimeSeriesProfiler, TimeSeriesProfile, DistributionStats, EntityLifecycle
)
from .temporal_quality_checks import (
    TemporalQualityCheck, TemporalQualityResult,
    DuplicateEventCheck, TemporalGapCheck, FutureDateCheck, EventOrderCheck
)
from .temporal_pattern_analyzer import (
    TemporalPatternAnalyzer, TemporalPatternAnalysis,
    TrendResult, TrendDirection, SeasonalityPeriod, RecencyResult
)
from .relationship_detector import (
    RelationshipDetector, DatasetRelationship, RelationshipType, JoinSuggestion
)
from .time_window_aggregator import (
    TimeWindowAggregator, AggregationPlan, TimeWindow, AggregationType
)
from .temporal_feature_analyzer import (
    TemporalFeatureAnalyzer, VelocityResult, MomentumResult,
    LagCorrelationResult, PredictivePowerResult, FeatureRecommendation, FeatureType
)
from .relationship_recommender import (
    RelationshipRecommender, RelationshipRecommendation,
    RecommendationCategory, RelationshipAnalysisSummary
)
from .feature_capacity import (
    FeatureCapacityAnalyzer, FeatureCapacityResult,
    SegmentCapacityResult, EffectiveFeaturesResult, ModelComplexityGuidance
)
from .temporal_feature_engineer import (
    TemporalFeatureEngineer, TemporalAggregationConfig,
    ReferenceMode, FeatureGroup, FeatureGroupResult, TemporalFeatureResult
)
from .text_embedder import TextEmbedder, EMBEDDING_MODELS, get_model_info, list_available_models
from .text_reducer import TextDimensionalityReducer, ReductionResult
from .text_processor import TextColumnProcessor, TextProcessingConfig, TextColumnResult

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
    "TimeSeriesProfiler", "TimeSeriesProfile", "DistributionStats", "EntityLifecycle",
    "TemporalQualityCheck", "TemporalQualityResult",
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
    "TextColumnProcessor", "TextProcessingConfig", "TextColumnResult"
]
