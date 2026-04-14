"""
Feature engineering module for customer retention analysis.

This module provides classes for deriving features from transformed data.
"""

from customer_retention.stages.features.behavioral_features import (
    BehavioralFeatureGenerator,
    BehavioralFeatureResult,
)
from customer_retention.stages.features.customer_segmentation import (
    CustomerSegmenter,
    SegmentationResult,
    SegmentationType,
    SegmentDefinition,
)
from customer_retention.stages.features.feature_definitions import (
    FeatureCatalog,
    FeatureCategory,
    FeatureDefinition,
    LeakageRisk,
)
from customer_retention.stages.features.feature_engineer import (
    FeatureEngineer,
    FeatureEngineerConfig,
    FeatureEngineerResult,
)
from customer_retention.stages.features.feature_manifest import (
    FeatureManifest,
    FeatureSet,
    FeatureSetRegistry,
)
from customer_retention.stages.features.feature_selector import (
    FeatureSelectionResult,
    FeatureSelector,
    SelectionMethod,
    extract_precomputed_stats,
    resolve_gbdt_selection_mode,
    run_chi_squared_rescue_selection,
    run_chi_squared_selection,
    run_gbdt_importance_selection,
    run_selection_pipeline,
)
from customer_retention.stages.features.interaction_features import (
    InteractionFeatureGenerator,
    InteractionFeatureResult,
)
from customer_retention.stages.features.temporal_features import (
    ReferenceDateSource,
    TemporalFeatureGenerator,
    TemporalFeatureResult,
)

__all__ = [
    "TemporalFeatureGenerator",
    "ReferenceDateSource",
    "TemporalFeatureResult",
    "BehavioralFeatureGenerator",
    "BehavioralFeatureResult",
    "InteractionFeatureGenerator",
    "InteractionFeatureResult",
    "FeatureDefinition",
    "FeatureCategory",
    "LeakageRisk",
    "FeatureCatalog",
    "FeatureEngineer",
    "FeatureEngineerConfig",
    "FeatureEngineerResult",
    "FeatureSelector",
    "SelectionMethod",
    "FeatureSelectionResult",
    "run_selection_pipeline",
    "run_chi_squared_selection",
    "run_chi_squared_rescue_selection",
    "run_gbdt_importance_selection",
    "resolve_gbdt_selection_mode",
    "extract_precomputed_stats",
    "FeatureManifest",
    "FeatureSet",
    "FeatureSetRegistry",
    "CustomerSegmenter",
    "SegmentationType",
    "SegmentDefinition",
    "SegmentationResult",
]
