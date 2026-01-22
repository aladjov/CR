"""Temporal feature definitions for leakage-safe feature engineering.

This module provides feature definition classes that include temporal
metadata required for point-in-time correct feature computation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class FeatureComputationType(Enum):
    """How the feature is computed."""

    PASSTHROUGH = "passthrough"  # Direct column from source
    AGGREGATION = "aggregation"  # Aggregated from events
    DERIVED = "derived"  # Computed from other features
    WINDOW = "window"  # Time-window aggregation
    RATIO = "ratio"  # Ratio of two features
    INTERACTION = "interaction"  # Feature interaction


class TemporalAggregation(Enum):
    """Aggregation functions for temporal features."""

    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STD = "std"
    LAST = "last"
    FIRST = "first"
    COUNT_DISTINCT = "count_distinct"


@dataclass
class TemporalFeatureDefinition:
    """Feature definition with temporal metadata for PIT correctness.

    This class defines a feature with all metadata required to ensure
    point-in-time correct computation, including timestamp columns,
    aggregation windows, and leakage risk assessment.

    Attributes:
        name: Feature name (snake_case)
        description: Human-readable description
        entity_key: Entity column for joins (e.g., "customer_id")
        timestamp_column: Column containing feature observation time
        source_columns: Input columns used to compute this feature
        computation_type: How the feature is computed
        aggregation: Aggregation function (if applicable)
        window_days: Aggregation window in days (if applicable)
        derivation_formula: Formula for derived features
        data_type: Output data type
        fill_value: Value to use for missing data
        leakage_risk: Assessment of leakage risk (low, medium, high)
        leakage_notes: Explanation of leakage risk
        tags: Additional metadata tags
        created_at: When the definition was created
        version: Version number for tracking changes

    Example:
        >>> feature = TemporalFeatureDefinition(
        ...     name="total_spend_30d",
        ...     description="Total spend in last 30 days",
        ...     entity_key="customer_id",
        ...     timestamp_column="feature_timestamp",
        ...     source_columns=["amount"],
        ...     computation_type=FeatureComputationType.WINDOW,
        ...     aggregation=TemporalAggregation.SUM,
        ...     window_days=30,
        ... )
    """

    name: str
    description: str
    entity_key: str
    timestamp_column: str = "feature_timestamp"
    source_columns: list[str] = field(default_factory=list)
    computation_type: FeatureComputationType = FeatureComputationType.PASSTHROUGH
    aggregation: Optional[TemporalAggregation] = None
    window_days: Optional[int] = None
    derivation_formula: Optional[str] = None
    data_type: str = "float64"
    fill_value: Optional[Any] = None
    leakage_risk: str = "low"
    leakage_notes: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def __post_init__(self):
        """Validate the feature definition."""
        if self.computation_type == FeatureComputationType.WINDOW:
            if self.window_days is None:
                raise ValueError("window_days required for WINDOW computation type")
            if self.aggregation is None:
                raise ValueError("aggregation required for WINDOW computation type")

        if self.computation_type == FeatureComputationType.DERIVED:
            if not self.derivation_formula:
                raise ValueError("derivation_formula required for DERIVED computation type")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "entity_key": self.entity_key,
            "timestamp_column": self.timestamp_column,
            "source_columns": self.source_columns,
            "computation_type": self.computation_type.value,
            "aggregation": self.aggregation.value if self.aggregation else None,
            "window_days": self.window_days,
            "derivation_formula": self.derivation_formula,
            "data_type": self.data_type,
            "fill_value": self.fill_value,
            "leakage_risk": self.leakage_risk,
            "leakage_notes": self.leakage_notes,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalFeatureDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            entity_key=data["entity_key"],
            timestamp_column=data.get("timestamp_column", "feature_timestamp"),
            source_columns=data.get("source_columns", []),
            computation_type=FeatureComputationType(data.get("computation_type", "passthrough")),
            aggregation=TemporalAggregation(data["aggregation"]) if data.get("aggregation") else None,
            window_days=data.get("window_days"),
            derivation_formula=data.get("derivation_formula"),
            data_type=data.get("data_type", "float64"),
            fill_value=data.get("fill_value"),
            leakage_risk=data.get("leakage_risk", "low"),
            leakage_notes=data.get("leakage_notes"),
            tags=data.get("tags", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            version=data.get("version", 1),
        )

    def get_feature_ref(self, feature_view_name: str) -> str:
        """Get Feast-style feature reference."""
        return f"{feature_view_name}:{self.name}"

    def validate_against_schema(self, columns: list[str]) -> list[str]:
        """Validate that source columns exist in schema.

        Args:
            columns: Available columns in the source data

        Returns:
            List of missing columns (empty if all present)
        """
        missing = [col for col in self.source_columns if col not in columns]
        return missing


@dataclass
class FeatureGroup:
    """A group of related features.

    Feature groups help organize features by domain or computation pattern.

    Attributes:
        name: Group name
        description: Group description
        features: List of feature definitions in this group
        entity_key: Common entity key for all features
        timestamp_column: Common timestamp column
        source_table: Source table or dataset name
        tags: Additional metadata
    """

    name: str
    description: str
    features: list[TemporalFeatureDefinition] = field(default_factory=list)
    entity_key: str = "customer_id"
    timestamp_column: str = "feature_timestamp"
    source_table: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)

    def add_feature(self, feature: TemporalFeatureDefinition) -> None:
        """Add a feature to this group."""
        self.features.append(feature)

    def get_feature(self, name: str) -> Optional[TemporalFeatureDefinition]:
        """Get a feature by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None

    def list_feature_names(self) -> list[str]:
        """List all feature names in this group."""
        return [f.name for f in self.features]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "features": [f.to_dict() for f in self.features],
            "entity_key": self.entity_key,
            "timestamp_column": self.timestamp_column,
            "source_table": self.source_table,
            "tags": self.tags,
        }
