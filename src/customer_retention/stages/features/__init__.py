"""
Feature engineering module for customer retention analysis.

This module provides classes for deriving features from transformed data.
"""

from customer_retention.stages.features.temporal_features import (
    TemporalFeatureGenerator,
    ReferenceDateSource,
    TemporalFeatureResult,
)
from customer_retention.stages.features.behavioral_features import (
    BehavioralFeatureGenerator,
    BehavioralFeatureResult,
)
from customer_retention.stages.features.interaction_features import (
    InteractionFeatureGenerator,
    InteractionFeatureResult,
)
from customer_retention.stages.features.feature_definitions import (
    FeatureDefinition,
    FeatureCategory,
    LeakageRisk,
    FeatureCatalog,
)
from customer_retention.stages.features.feature_engineer import (
    FeatureEngineer,
    FeatureEngineerConfig,
    FeatureEngineerResult,
)
from customer_retention.stages.features.feature_selector import (
    FeatureSelector,
    SelectionMethod,
    FeatureSelectionResult,
)
from customer_retention.stages.features.feature_manifest import (
    FeatureManifest,
    FeatureSet,
    FeatureSetRegistry,
)
from customer_retention.stages.features.customer_segmentation import (
    CustomerSegmenter,
    SegmentationType,
    SegmentDefinition,
    SegmentationResult,
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
    "FeatureManifest",
    "FeatureSet",
    "FeatureSetRegistry",
    "CustomerSegmenter",
    "SegmentationType",
    "SegmentDefinition",
    "SegmentationResult",
]
