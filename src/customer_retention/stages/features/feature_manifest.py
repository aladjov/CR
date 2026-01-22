"""
Feature manifest and versioning for customer retention analysis.

This module provides classes for tracking feature sets, manifests,
and registry for version management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import uuid

from customer_retention.core.compat import pd, DataFrame


@dataclass
class FeatureManifest:
    """
    Manifest tracking the composition and provenance of a feature set.

    Attributes
    ----------
    manifest_id : str
        Unique identifier for this manifest.
    created_at : datetime
        When the manifest was created.
    created_by : str, optional
        Who created the manifest.
    feature_table : str, optional
        Source table name.
    feature_table_version : int, optional
        Delta version number if applicable.
    features_included : List[str]
        List of feature names included.
    features_excluded : List[str]
        Excluded features and reasons.
    row_count : int
        Number of rows in the dataset.
    column_count : int
        Number of feature columns.
    checksum : str
        Data integrity hash.
    """
    manifest_id: str
    created_at: datetime
    features_included: List[str]
    row_count: int
    column_count: int
    checksum: str
    created_by: Optional[str] = None
    feature_table: Optional[str] = None
    feature_table_version: Optional[int] = None
    features_excluded: List[str] = field(default_factory=list)
    feature_transformations: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        df: DataFrame,
        feature_columns: List[str],
        entity_column: Optional[str] = None,
        created_by: Optional[str] = None,
        feature_table: Optional[str] = None,
    ) -> "FeatureManifest":
        """
        Create a manifest from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Source DataFrame.
        feature_columns : List[str]
            List of feature column names.
        entity_column : str, optional
            Entity/ID column name.
        created_by : str, optional
            Creator name.
        feature_table : str, optional
            Source table name.

        Returns
        -------
        FeatureManifest
            New manifest instance.
        """
        # Generate unique ID
        manifest_id = str(uuid.uuid4())

        # Compute checksum from feature data
        feature_data = df[feature_columns].values
        checksum = cls._compute_checksum(feature_data)

        return cls(
            manifest_id=manifest_id,
            created_at=datetime.now(),
            created_by=created_by,
            feature_table=feature_table,
            features_included=feature_columns.copy(),
            row_count=len(df),
            column_count=len(feature_columns),
            checksum=checksum,
        )

    @staticmethod
    def _compute_checksum(data) -> str:
        """Compute MD5 checksum of data."""
        # Convert to bytes and hash
        data_bytes = pd.util.hash_pandas_object(
            pd.DataFrame(data), index=False
        ).values.tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "manifest_id": self.manifest_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "feature_table": self.feature_table,
            "feature_table_version": self.feature_table_version,
            "features_included": self.features_included,
            "features_excluded": self.features_excluded,
            "feature_transformations": self.feature_transformations,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }


@dataclass
class FeatureSet:
    """
    Named, versioned collection of features.

    Attributes
    ----------
    name : str
        Feature set name.
    version : str
        Version identifier (semver format).
    description : str
        Purpose of this feature set.
    features_included : List[str]
        Selected features.
    features_excluded : List[str]
        Dropped features.
    exclusion_reasons : Dict[str, str]
        Why each was dropped.
    created_at : datetime
        Creation timestamp.
    created_by : str, optional
        Creator.
    parent_feature_set : str, optional
        If derived from another set.
    metadata : Dict
        Additional info.
    """
    name: str
    version: str
    description: str
    features_included: List[str]
    features_excluded: List[str] = field(default_factory=list)
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    parent_feature_set: Optional[str] = None
    feature_table: Optional[str] = None
    feature_table_version: Optional[int] = None
    transformations: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature set to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "features_included": self.features_included,
            "features_excluded": self.features_excluded,
            "exclusion_reasons": self.exclusion_reasons,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "parent_feature_set": self.parent_feature_set,
            "feature_table": self.feature_table,
            "feature_table_version": self.feature_table_version,
            "transformations": self.transformations,
            "metadata": self.metadata,
        }


class FeatureSetRegistry:
    """
    Registry for managing feature sets.

    Provides methods for registering, retrieving, and comparing
    feature sets.
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, FeatureSet]] = {}

    def register(self, feature_set: FeatureSet) -> None:
        """
        Register a new feature set.

        Parameters
        ----------
        feature_set : FeatureSet
            Feature set to register.

        Raises
        ------
        ValueError
            If feature set with same name and version exists.
        """
        name = feature_set.name
        version = feature_set.version

        if name not in self._registry:
            self._registry[name] = {}

        if version in self._registry[name]:
            raise ValueError(
                f"Feature set '{name}' version '{version}' already registered."
            )

        self._registry[name][version] = feature_set

    def get(
        self,
        name: str,
        version: str
    ) -> Optional[FeatureSet]:
        """
        Get a feature set by name and version.

        Parameters
        ----------
        name : str
            Feature set name.
        version : str
            Version string.

        Returns
        -------
        FeatureSet or None
            The feature set, or None if not found.
        """
        if name not in self._registry:
            return None
        return self._registry[name].get(version)

    def get_latest(self, name: str) -> Optional[FeatureSet]:
        """
        Get the latest version of a feature set.

        Parameters
        ----------
        name : str
            Feature set name.

        Returns
        -------
        FeatureSet or None
            Latest version, or None if not found.
        """
        if name not in self._registry:
            return None

        versions = list(self._registry[name].keys())
        if not versions:
            return None

        # Sort versions (assumes semver-like format)
        versions.sort(key=lambda v: [int(x) for x in v.split(".")])
        return self._registry[name][versions[-1]]

    def list_all(self) -> List[FeatureSet]:
        """
        List all registered feature sets.

        Returns
        -------
        List[FeatureSet]
            All feature sets.
        """
        result = []
        for versions in self._registry.values():
            result.extend(versions.values())
        return result

    def list_versions(self, name: str) -> List[str]:
        """
        List all versions of a feature set.

        Parameters
        ----------
        name : str
            Feature set name.

        Returns
        -------
        List[str]
            Available versions.
        """
        if name not in self._registry:
            return []
        return list(self._registry[name].keys())

    def compare(
        self,
        set1: FeatureSet,
        set2: FeatureSet
    ) -> Dict[str, List[str]]:
        """
        Compare two feature sets.

        Parameters
        ----------
        set1 : FeatureSet
            First feature set.
        set2 : FeatureSet
            Second feature set.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary with 'added', 'removed', and 'unchanged' keys.
        """
        features1 = set(set1.features_included)
        features2 = set(set2.features_included)

        return {
            "added": list(features2 - features1),
            "removed": list(features1 - features2),
            "unchanged": list(features1 & features2),
        }
