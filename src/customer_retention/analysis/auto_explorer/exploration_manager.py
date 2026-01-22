"""
Exploration Manager for managing multiple dataset explorations.

Provides functionality for:
- Discovering and loading exploration findings
- Managing dataset inclusion/exclusion
- Detecting relationships between datasets
- Planning aggregations for multi-dataset analysis
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from customer_retention.core.config.column_config import DatasetGranularity
from .findings import ExplorationFindings

if TYPE_CHECKING:
    from .layered_recommendations import RecommendationRegistry


@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    name: str
    findings_path: str
    source_path: str
    granularity: DatasetGranularity
    row_count: int
    column_count: int
    entity_column: Optional[str] = None
    time_column: Optional[str] = None
    target_column: Optional[str] = None
    excluded: bool = False


@dataclass
class DatasetRelationshipInfo:
    """Information about relationship between two datasets."""
    left_dataset: str
    right_dataset: str
    left_column: str
    right_column: str
    relationship_type: str  # one_to_one, one_to_many, many_to_many
    confidence: float = 1.0
    auto_detected: bool = False


@dataclass
class AggregationPlanItem:
    """Plan for aggregating one event dataset."""
    dataset_name: str
    entity_column: str
    time_column: str
    windows: List[str]
    value_columns: List[str]
    agg_funcs: List[str]


@dataclass
class MultiDatasetFindings:
    """Findings for multiple related datasets."""
    datasets: Dict[str, DatasetInfo] = field(default_factory=dict)
    relationships: List[DatasetRelationshipInfo] = field(default_factory=list)
    primary_entity_dataset: Optional[str] = None
    event_datasets: List[str] = field(default_factory=list)
    excluded_datasets: List[str] = field(default_factory=list)
    aggregation_windows: List[str] = field(default_factory=lambda: ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"])
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def selected_datasets(self) -> Dict[str, DatasetInfo]:
        """Return only datasets that are not excluded."""
        return {name: info for name, info in self.datasets.items()
                if name not in self.excluded_datasets and not info.excluded}

    def exclude_dataset(self, name: str) -> None:
        """Exclude a dataset from the pipeline."""
        if name in self.datasets:
            if name not in self.excluded_datasets:
                self.excluded_datasets.append(name)
            self.datasets[name].excluded = True

    def select_dataset(self, name: str) -> None:
        """Re-include a previously excluded dataset."""
        if name in self.excluded_datasets:
            self.excluded_datasets.remove(name)
        if name in self.datasets:
            self.datasets[name].excluded = False

    def add_relationship(
        self,
        left_dataset: str,
        right_dataset: str,
        left_column: str,
        right_column: str,
        relationship_type: str = "one_to_many",
        confidence: float = 1.0,
    ) -> None:
        """Add a relationship between datasets."""
        rel = DatasetRelationshipInfo(
            left_dataset=left_dataset,
            right_dataset=right_dataset,
            left_column=left_column,
            right_column=right_column,
            relationship_type=relationship_type,
            confidence=confidence,
            auto_detected=False,
        )
        self.relationships.append(rel)

    def get_aggregation_plan(self) -> Dict[str, AggregationPlanItem]:
        """Generate aggregation plan for all event datasets."""
        plan = {}

        for dataset_name in self.event_datasets:
            if dataset_name in self.excluded_datasets:
                continue

            dataset_info = self.datasets.get(dataset_name)
            if dataset_info and dataset_info.entity_column and dataset_info.time_column:
                plan[dataset_name] = AggregationPlanItem(
                    dataset_name=dataset_name,
                    entity_column=dataset_info.entity_column,
                    time_column=dataset_info.time_column,
                    windows=self.aggregation_windows.copy(),
                    value_columns=[],  # To be filled by user
                    agg_funcs=["sum", "mean", "count"],
                )

        return plan

    def to_recommendation_registry(self) -> "RecommendationRegistry":
        """Create a RecommendationRegistry from selected datasets."""
        from .layered_recommendations import RecommendationRegistry

        registry = RecommendationRegistry()

        for name, info in self.selected_datasets.items():
            registry.add_source(name, info.source_path)

        if self.primary_entity_dataset and self.primary_entity_dataset in self.selected_datasets:
            primary_info = self.datasets[self.primary_entity_dataset]
            entity_col = primary_info.entity_column or "id"
            time_col = primary_info.time_column
            registry.init_silver(entity_col, time_col)

            if primary_info.target_column:
                registry.init_gold(primary_info.target_column)

        for rel in self.relationships:
            if (rel.left_dataset in self.selected_datasets and
                rel.right_dataset in self.selected_datasets and
                registry.silver):
                registry.add_silver_join(
                    rel.left_dataset, rel.right_dataset,
                    [rel.left_column], rel.relationship_type,
                    f"Join {rel.left_dataset} with {rel.right_dataset}"
                )

        return registry

    def save(self, path: str) -> None:
        """Save multi-dataset findings to YAML."""
        data = {
            "datasets": {
                name: {
                    "name": info.name,
                    "findings_path": info.findings_path,
                    "source_path": info.source_path,
                    "granularity": info.granularity.value,
                    "row_count": info.row_count,
                    "column_count": info.column_count,
                    "entity_column": info.entity_column,
                    "time_column": info.time_column,
                    "target_column": info.target_column,
                    "excluded": info.excluded,
                }
                for name, info in self.datasets.items()
            },
            "relationships": [
                {
                    "left_dataset": rel.left_dataset,
                    "right_dataset": rel.right_dataset,
                    "left_column": rel.left_column,
                    "right_column": rel.right_column,
                    "relationship_type": rel.relationship_type,
                    "confidence": rel.confidence,
                    "auto_detected": rel.auto_detected,
                }
                for rel in self.relationships
            ],
            "primary_entity_dataset": self.primary_entity_dataset,
            "event_datasets": self.event_datasets,
            "excluded_datasets": self.excluded_datasets,
            "aggregation_windows": self.aggregation_windows,
            "notes": self.notes,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "MultiDatasetFindings":
        """Load multi-dataset findings from YAML."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        datasets = {}
        for name, info in data.get("datasets", {}).items():
            datasets[name] = DatasetInfo(
                name=info["name"],
                findings_path=info["findings_path"],
                source_path=info["source_path"],
                granularity=DatasetGranularity(info["granularity"]),
                row_count=info["row_count"],
                column_count=info["column_count"],
                entity_column=info.get("entity_column"),
                time_column=info.get("time_column"),
                target_column=info.get("target_column"),
                excluded=info.get("excluded", False),
            )

        relationships = [
            DatasetRelationshipInfo(
                left_dataset=rel["left_dataset"],
                right_dataset=rel["right_dataset"],
                left_column=rel["left_column"],
                right_column=rel["right_column"],
                relationship_type=rel["relationship_type"],
                confidence=rel.get("confidence", 1.0),
                auto_detected=rel.get("auto_detected", False),
            )
            for rel in data.get("relationships", [])
        ]

        return cls(
            datasets=datasets,
            relationships=relationships,
            primary_entity_dataset=data.get("primary_entity_dataset"),
            event_datasets=data.get("event_datasets", []),
            excluded_datasets=data.get("excluded_datasets", []),
            aggregation_windows=data.get("aggregation_windows", ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"]),
            notes=data.get("notes", {}),
        )


class ExplorationManager:
    """Manages multiple exploration findings."""

    def __init__(self, explorations_dir: Path):
        self.explorations_dir = Path(explorations_dir)
        self._findings_cache: Dict[str, ExplorationFindings] = {}
        self._excluded_datasets: set = set()

    def discover_findings(self) -> List[Path]:
        """Discover all findings files in the explorations directory.

        Excludes multi_dataset_findings.yaml as it has a different structure.
        """
        if not self.explorations_dir.exists():
            return []

        return [
            f for f in self.explorations_dir.glob("*_findings.yaml")
            if "multi_dataset" not in f.name
        ]

    def load_findings(self, name_pattern: str) -> Optional[ExplorationFindings]:
        """Load findings by name pattern (partial match)."""
        for path in self.discover_findings():
            if name_pattern.lower() in path.stem.lower():
                if str(path) not in self._findings_cache:
                    self._findings_cache[str(path)] = ExplorationFindings.load(str(path))
                return self._findings_cache[str(path)]
        return None

    def list_datasets(self, include_excluded: bool = False) -> List[DatasetInfo]:
        """List all discovered datasets with their info."""
        datasets = []

        for path in self.discover_findings():
            findings = ExplorationFindings.load(str(path))

            # Determine granularity
            if findings.is_time_series and findings.time_series_metadata:
                granularity = findings.time_series_metadata.granularity
                entity_col = findings.time_series_metadata.entity_column
                time_col = findings.time_series_metadata.time_column
            else:
                granularity = DatasetGranularity.ENTITY_LEVEL
                entity_col = None
                time_col = None

            # Extract dataset name from path
            name = self._extract_dataset_name(path)
            is_excluded = name in self._excluded_datasets

            if not include_excluded and is_excluded:
                continue

            datasets.append(DatasetInfo(
                name=name,
                findings_path=str(path),
                source_path=findings.source_path,
                granularity=granularity,
                row_count=findings.row_count,
                column_count=findings.column_count,
                entity_column=entity_col,
                time_column=time_col,
                target_column=findings.target_column,
                excluded=is_excluded,
            ))

        return datasets

    def create_multi_dataset_findings(
        self, dataset_names: Optional[List[str]] = None
    ) -> MultiDatasetFindings:
        """Create a MultiDatasetFindings from discovered datasets.

        Args:
            dataset_names: Optional list of dataset names to include.
                          If None, all discovered datasets are included.
                          If provided, only datasets matching these names are included.
        """
        datasets_info = self.list_datasets(include_excluded=True)

        # Filter to specified datasets if provided
        if dataset_names:
            datasets_info = [d for d in datasets_info if d.name in dataset_names]

        datasets = {d.name: d for d in datasets_info}
        event_datasets = [d.name for d in datasets_info
                        if d.granularity == DatasetGranularity.EVENT_LEVEL]
        excluded = [d.name for d in datasets_info if d.excluded]

        # Determine primary entity dataset (one with target, or largest entity-level)
        primary = None
        for d in datasets_info:
            if d.granularity == DatasetGranularity.ENTITY_LEVEL:
                if d.target_column:
                    primary = d.name
                    break
                elif primary is None:
                    primary = d.name

        return MultiDatasetFindings(
            datasets=datasets,
            relationships=[],
            primary_entity_dataset=primary,
            event_datasets=event_datasets,
            excluded_datasets=excluded,
        )

    def exclude_dataset(self, name_pattern: str) -> None:
        """Exclude a dataset from multi-dataset analysis."""
        for dataset in self.list_datasets(include_excluded=True):
            if name_pattern.lower() in dataset.name.lower():
                self._excluded_datasets.add(dataset.name)
                return

    def include_dataset(self, name_pattern: str) -> None:
        """Re-include a previously excluded dataset."""
        to_remove = None
        for name in self._excluded_datasets:
            if name_pattern.lower() in name.lower():
                to_remove = name
                break
        if to_remove:
            self._excluded_datasets.remove(to_remove)

    def get_aggregated_path(self, original_findings_path: str) -> Optional[str]:
        """Get the aggregated findings path for an event-level dataset.

        Returns the path to the aggregated findings file if:
        - The original findings is event-level
        - Aggregation has been executed (via 01d notebook)

        Returns None if the dataset is entity-level or not yet aggregated.
        """
        findings = ExplorationFindings.load(original_findings_path)

        if not findings.has_aggregated_output:
            return None

        return findings.time_series_metadata.aggregated_findings_path

    def _extract_dataset_name(self, path: Path) -> str:
        """Extract dataset name from findings path."""
        # Pattern: {name}_{hash}_findings.yaml or {name}_{hash}_aggregated_findings.yaml
        stem = path.stem  # e.g., "customers_abc123_findings" or "customers_abc123_aggregated_findings"

        # Remove _findings suffix
        stem = stem.replace("_findings", "")

        # Check for _aggregated suffix
        if "_aggregated" in stem:
            # Keep "aggregated" as part of name to distinguish from original
            parts = stem.rsplit("_aggregated", 1)
            base_name = parts[0]
            # Remove hash from base_name
            name_parts = base_name.rsplit("_", 1)
            if len(name_parts) == 2:
                return f"{name_parts[0]}_aggregated"
            return f"{base_name}_aggregated"

        # Regular findings - remove hash
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]  # Return name without hash
        return stem
