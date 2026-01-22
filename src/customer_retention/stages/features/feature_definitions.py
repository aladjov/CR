"""
Feature definitions and catalog for customer retention analysis.

This module provides classes for documenting and managing feature metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from customer_retention.core.compat import pd, DataFrame


class FeatureCategory(Enum):
    """Categories for feature classification."""
    TEMPORAL = "TEMPORAL"
    BEHAVIORAL = "BEHAVIORAL"
    MONETARY = "MONETARY"
    ENGAGEMENT = "ENGAGEMENT"
    ADOPTION = "ADOPTION"
    DEMOGRAPHIC = "DEMOGRAPHIC"
    AGGREGATE = "AGGREGATE"
    RATIO = "RATIO"
    TREND = "TREND"
    INTERACTION = "INTERACTION"


class LeakageRisk(Enum):
    """Leakage risk levels for features."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class FeatureDefinition:
    """
    Definition and metadata for a single feature.

    Parameters
    ----------
    name : str
        Feature name (snake_case).
    description : str
        What the feature represents.
    category : FeatureCategory
        Feature category.
    derivation : str
        How the feature is calculated.
    source_columns : List[str]
        Input columns used.
    data_type : str
        Output data type.
    business_meaning : str
        Business interpretation.
    display_name : str, optional
        Human-readable name.
    value_range : Tuple[float, float], optional
        Expected min/max.
    leakage_risk : LeakageRisk, default LOW
        Leakage risk level.
    created_date : datetime, optional
        When feature was added (auto-populated).
    created_by : str, optional
        Who added the feature.
    """
    name: str
    description: str
    category: FeatureCategory
    derivation: str
    source_columns: List[str]
    data_type: str
    business_meaning: str
    display_name: Optional[str] = None
    value_range: Optional[Tuple[float, float]] = None
    leakage_risk: LeakageRisk = LeakageRisk.LOW
    created_date: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "derivation": self.derivation,
            "source_columns": self.source_columns,
            "data_type": self.data_type,
            "value_range": self.value_range,
            "business_meaning": self.business_meaning,
            "leakage_risk": self.leakage_risk.value,
            "created_date": self.created_date.isoformat(),
            "created_by": self.created_by,
        }


class FeatureCatalog:
    """
    Catalog for managing feature definitions.

    Provides methods for adding, retrieving, and filtering features.
    """

    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}

    def add(
        self,
        feature: FeatureDefinition,
        overwrite: bool = False
    ) -> None:
        """
        Add a feature definition to the catalog.

        Parameters
        ----------
        feature : FeatureDefinition
            The feature to add.
        overwrite : bool, default False
            Whether to overwrite if feature already exists.

        Raises
        ------
        ValueError
            If feature name already exists and overwrite is False.
        """
        if feature.name in self._features and not overwrite:
            raise ValueError(
                f"Feature '{feature.name}' already exists. "
                "Use overwrite=True to replace."
            )
        self._features[feature.name] = feature

    def get(self, name: str) -> Optional[FeatureDefinition]:
        """
        Get a feature definition by name.

        Parameters
        ----------
        name : str
            Feature name.

        Returns
        -------
        FeatureDefinition or None
            The feature definition, or None if not found.
        """
        return self._features.get(name)

    def remove(self, name: str) -> None:
        """
        Remove a feature from the catalog.

        Parameters
        ----------
        name : str
            Feature name to remove.
        """
        if name in self._features:
            del self._features[name]

    def list_names(self) -> List[str]:
        """
        List all feature names in the catalog.

        Returns
        -------
        List[str]
            List of feature names.
        """
        return list(self._features.keys())

    def list_by_category(
        self,
        category: FeatureCategory
    ) -> List[FeatureDefinition]:
        """
        List features by category.

        Parameters
        ----------
        category : FeatureCategory
            Category to filter by.

        Returns
        -------
        List[FeatureDefinition]
            Features in the specified category.
        """
        return [
            f for f in self._features.values()
            if f.category == category
        ]

    def list_by_leakage_risk(
        self,
        risk: LeakageRisk
    ) -> List[FeatureDefinition]:
        """
        List features by leakage risk level.

        Parameters
        ----------
        risk : LeakageRisk
            Risk level to filter by.

        Returns
        -------
        List[FeatureDefinition]
            Features with the specified risk level.
        """
        return [
            f for f in self._features.values()
            if f.leakage_risk == risk
        ]

    def to_dataframe(self) -> DataFrame:
        """
        Convert catalog to DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with feature metadata.
        """
        if not self._features:
            return pd.DataFrame()

        records = []
        for feature in self._features.values():
            records.append({
                "name": feature.name,
                "display_name": feature.display_name,
                "description": feature.description,
                "category": feature.category.value,
                "derivation": feature.derivation,
                "source_columns": feature.source_columns,
                "data_type": feature.data_type,
                "value_range": feature.value_range,
                "business_meaning": feature.business_meaning,
                "leakage_risk": feature.leakage_risk.value,
                "created_date": feature.created_date,
                "created_by": feature.created_by,
            })
        return pd.DataFrame(records)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert catalog to dictionary.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary keyed by feature name.
        """
        return {
            name: feature.to_dict()
            for name, feature in self._features.items()
        }

    def __len__(self) -> int:
        """Return number of features in catalog."""
        return len(self._features)
