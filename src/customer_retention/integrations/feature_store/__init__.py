"""Feature store module for leakage-safe feature management.

This module provides a unified interface for managing ML features with
point-in-time correctness. It supports both local development (Feast)
and production (Databricks Feature Engineering) backends.

Key Components:
    - TemporalFeatureDefinition: Feature definition with temporal metadata
    - FeatureRegistry: Central registry for all feature definitions
    - FeatureStoreManager: Unified interface for feature store operations

Example:
    >>> from customer_retention.integrations.feature_store import (
    ...     FeatureStoreManager, TemporalFeatureDefinition, FeatureRegistry
    ... )
    >>>
    >>> # Create feature definitions
    >>> registry = FeatureRegistry()
    >>> registry.register(TemporalFeatureDefinition(
    ...     name="tenure_months",
    ...     description="Customer tenure in months",
    ...     entity_key="customer_id",
    ...     timestamp_column="feature_timestamp",
    ...     source_columns=["tenure"],
    ... ))
    >>>
    >>> # Create feature store manager
    >>> manager = FeatureStoreManager.create(backend="feast")
    >>> manager.publish_features(df, registry)
"""

from .definitions import (
    TemporalFeatureDefinition,
    FeatureComputationType,
    TemporalAggregation,
)
from .registry import FeatureRegistry
from .manager import FeatureStoreManager, get_feature_store_manager

__all__ = [
    "TemporalFeatureDefinition",
    "FeatureComputationType",
    "TemporalAggregation",
    "FeatureRegistry",
    "FeatureStoreManager",
    "get_feature_store_manager",
]
