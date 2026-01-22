"""Feature registry for centralized feature management.

This module provides a central registry for all feature definitions,
enabling consistent feature computation across training and inference.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .definitions import (
    TemporalFeatureDefinition,
    FeatureGroup,
    FeatureComputationType,
)


class FeatureRegistry:
    """Central registry for feature definitions.

    The FeatureRegistry provides a single source of truth for all feature
    definitions, ensuring consistency between training and inference.

    Example:
        >>> registry = FeatureRegistry()
        >>> registry.register(TemporalFeatureDefinition(
        ...     name="tenure_months",
        ...     description="Customer tenure in months",
        ...     entity_key="customer_id",
        ...     timestamp_column="feature_timestamp",
        ... ))
        >>> registry.get("tenure_months")
        TemporalFeatureDefinition(name='tenure_months', ...)
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._features: dict[str, TemporalFeatureDefinition] = {}
        self._groups: dict[str, FeatureGroup] = {}
        self._metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
        }

    def register(
        self,
        feature: TemporalFeatureDefinition,
        group_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Register a feature definition.

        Args:
            feature: Feature definition to register
            group_name: Optional group to add the feature to
            overwrite: If True, overwrite existing feature with same name

        Raises:
            ValueError: If feature already exists and overwrite=False
        """
        if feature.name in self._features and not overwrite:
            raise ValueError(
                f"Feature '{feature.name}' already registered. "
                "Use overwrite=True to replace."
            )

        self._features[feature.name] = feature

        if group_name:
            if group_name not in self._groups:
                self._groups[group_name] = FeatureGroup(
                    name=group_name,
                    description=f"Feature group: {group_name}",
                    entity_key=feature.entity_key,
                    timestamp_column=feature.timestamp_column,
                )
            self._groups[group_name].add_feature(feature)

    def register_group(self, group: FeatureGroup, overwrite: bool = False) -> None:
        """Register a feature group with all its features.

        Args:
            group: Feature group to register
            overwrite: If True, overwrite existing features
        """
        self._groups[group.name] = group
        for feature in group.features:
            self.register(feature, overwrite=overwrite)

    def get(self, name: str) -> Optional[TemporalFeatureDefinition]:
        """Get a feature definition by name.

        Args:
            name: Feature name

        Returns:
            Feature definition or None if not found
        """
        return self._features.get(name)

    def get_group(self, name: str) -> Optional[FeatureGroup]:
        """Get a feature group by name.

        Args:
            name: Group name

        Returns:
            Feature group or None if not found
        """
        return self._groups.get(name)

    def remove(self, name: str) -> bool:
        """Remove a feature from the registry.

        Args:
            name: Feature name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._features:
            del self._features[name]
            # Also remove from groups
            for group in self._groups.values():
                group.features = [f for f in group.features if f.name != name]
            return True
        return False

    def list_features(self) -> list[str]:
        """List all registered feature names.

        Returns:
            List of feature names
        """
        return list(self._features.keys())

    def list_groups(self) -> list[str]:
        """List all registered group names.

        Returns:
            List of group names
        """
        return list(self._groups.keys())

    def list_by_computation_type(
        self, computation_type: FeatureComputationType
    ) -> list[TemporalFeatureDefinition]:
        """List features by computation type.

        Args:
            computation_type: Type to filter by

        Returns:
            List of matching feature definitions
        """
        return [
            f for f in self._features.values()
            if f.computation_type == computation_type
        ]

    def list_by_entity(self, entity_key: str) -> list[TemporalFeatureDefinition]:
        """List features by entity key.

        Args:
            entity_key: Entity key to filter by

        Returns:
            List of matching feature definitions
        """
        return [
            f for f in self._features.values()
            if f.entity_key == entity_key
        ]

    def list_high_leakage_risk(self) -> list[TemporalFeatureDefinition]:
        """List features with high leakage risk.

        Returns:
            List of high-risk feature definitions
        """
        return [
            f for f in self._features.values()
            if f.leakage_risk == "high"
        ]

    def validate_features(self, columns: list[str]) -> dict[str, list[str]]:
        """Validate all features against available columns.

        Args:
            columns: Available columns in source data

        Returns:
            Dictionary mapping feature names to missing columns
        """
        issues = {}
        for name, feature in self._features.items():
            missing = feature.validate_against_schema(columns)
            if missing:
                issues[name] = missing
        return issues

    def get_feature_refs(
        self,
        feature_view_name: str,
        feature_names: Optional[list[str]] = None,
    ) -> list[str]:
        """Get Feast-style feature references.

        Args:
            feature_view_name: Name of the feature view
            feature_names: Specific features (all if None)

        Returns:
            List of feature references like "view:feature"
        """
        names = feature_names or self.list_features()
        return [
            self._features[name].get_feature_ref(feature_view_name)
            for name in names
            if name in self._features
        ]

    def save(self, path: Path) -> None:
        """Save registry to JSON file.

        Args:
            path: Path to save to
        """
        data = {
            "metadata": self._metadata,
            "features": {name: f.to_dict() for name, f in self._features.items()},
            "groups": {name: g.to_dict() for name, g in self._groups.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FeatureRegistry":
        """Load registry from JSON file.

        Args:
            path: Path to load from

        Returns:
            Loaded FeatureRegistry
        """
        with open(path) as f:
            data = json.load(f)

        registry = cls()
        registry._metadata = data.get("metadata", {})

        # Load features
        for name, feature_data in data.get("features", {}).items():
            feature = TemporalFeatureDefinition.from_dict(feature_data)
            registry._features[name] = feature

        # Load groups
        for name, group_data in data.get("groups", {}).items():
            group = FeatureGroup(
                name=group_data["name"],
                description=group_data["description"],
                entity_key=group_data.get("entity_key", "customer_id"),
                timestamp_column=group_data.get("timestamp_column", "feature_timestamp"),
                source_table=group_data.get("source_table"),
                tags=group_data.get("tags", {}),
            )
            # Link features to group
            for feature_data in group_data.get("features", []):
                feature_name = feature_data["name"]
                if feature_name in registry._features:
                    group.features.append(registry._features[feature_name])
            registry._groups[name] = group

        return registry

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "metadata": self._metadata,
            "features": {name: f.to_dict() for name, f in self._features.items()},
            "groups": {name: g.to_dict() for name, g in self._groups.items()},
        }

    def __len__(self) -> int:
        """Return number of registered features."""
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        """Check if feature is registered."""
        return name in self._features


def create_standard_churn_features() -> FeatureRegistry:
    """Create a registry with standard churn prediction features.

    This provides a starting point for churn prediction projects
    with commonly used features.

    Returns:
        FeatureRegistry with standard features
    """
    registry = FeatureRegistry()

    # Demographic features
    demographic_group = FeatureGroup(
        name="demographic",
        description="Customer demographic features",
        entity_key="customer_id",
    )

    demographic_group.add_feature(TemporalFeatureDefinition(
        name="tenure_months",
        description="Customer tenure in months",
        entity_key="customer_id",
        source_columns=["tenure"],
        computation_type=FeatureComputationType.PASSTHROUGH,
        data_type="int64",
        leakage_risk="low",
    ))

    demographic_group.add_feature(TemporalFeatureDefinition(
        name="age",
        description="Customer age",
        entity_key="customer_id",
        source_columns=["age"],
        computation_type=FeatureComputationType.PASSTHROUGH,
        data_type="int64",
        leakage_risk="low",
    ))

    registry.register_group(demographic_group)

    # Behavioral features
    behavioral_group = FeatureGroup(
        name="behavioral",
        description="Customer behavioral features",
        entity_key="customer_id",
    )

    behavioral_group.add_feature(TemporalFeatureDefinition(
        name="total_spend_30d",
        description="Total spend in last 30 days",
        entity_key="customer_id",
        source_columns=["amount"],
        computation_type=FeatureComputationType.WINDOW,
        aggregation=TemporalAggregation.SUM,
        window_days=30,
        data_type="float64",
        fill_value=0.0,
        leakage_risk="low",
    ))

    behavioral_group.add_feature(TemporalFeatureDefinition(
        name="transaction_count_30d",
        description="Number of transactions in last 30 days",
        entity_key="customer_id",
        source_columns=["transaction_id"],
        computation_type=FeatureComputationType.WINDOW,
        aggregation=TemporalAggregation.COUNT,
        window_days=30,
        data_type="int64",
        fill_value=0,
        leakage_risk="low",
    ))

    behavioral_group.add_feature(TemporalFeatureDefinition(
        name="avg_transaction_amount",
        description="Average transaction amount",
        entity_key="customer_id",
        source_columns=["amount"],
        computation_type=FeatureComputationType.AGGREGATION,
        aggregation=TemporalAggregation.MEAN,
        data_type="float64",
        leakage_risk="low",
    ))

    registry.register_group(behavioral_group)

    # Engagement features
    engagement_group = FeatureGroup(
        name="engagement",
        description="Customer engagement features",
        entity_key="customer_id",
    )

    engagement_group.add_feature(TemporalFeatureDefinition(
        name="days_since_last_activity",
        description="Days since last activity",
        entity_key="customer_id",
        source_columns=["last_activity_date", "feature_timestamp"],
        computation_type=FeatureComputationType.DERIVED,
        derivation_formula="feature_timestamp - last_activity_date",
        data_type="int64",
        leakage_risk="medium",
        leakage_notes="Ensure last_activity_date is before feature_timestamp",
    ))

    registry.register_group(engagement_group)

    return registry


# Import for convenience
from .definitions import TemporalAggregation
