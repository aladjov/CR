"""
Exploration Manager for managing multiple dataset explorations.

Provides functionality for:
- Discovering and loading exploration findings
- Managing dataset inclusion/exclusion
- Detecting relationships between datasets
- Planning aggregations for multi-dataset analysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.modeling.feature_spec import LeakageExclusion

from .findings import ExplorationFindings

if TYPE_CHECKING:
    from customer_retention.generators.pipeline_generator.models import FeatureExclusion

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
    raw_source_path: Optional[str] = None
    excluded: bool = False
    feature_exclusions: List[FeatureExclusion] = field(default_factory=list)
    excluded_leaking_features: List[LeakageExclusion] = field(default_factory=list)
    zero_inflation_opt_in: List[str] = field(default_factory=list)


@dataclass
class DatasetRelationshipInfo:
    """Information about relationship between two datasets."""

    left_dataset: str
    right_dataset: str
    left_columns: list
    right_columns: list
    relationship_type: str  # one_to_one, one_to_many, many_to_many
    confidence: float = 1.0
    auto_detected: bool = False

    @property
    def left_column(self) -> str:
        return self.left_columns[0] if self.left_columns else ""

    @property
    def right_column(self) -> str:
        return self.right_columns[0] if self.right_columns else ""


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
    aggregation_windows: List[str] = field(
        default_factory=lambda: ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"]
    )
    bronze_aggregation_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    composite_dataset_name: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def selected_datasets(self) -> Dict[str, DatasetInfo]:
        """Return only datasets that are not excluded."""
        return {
            name: info
            for name, info in self.datasets.items()
            if name not in self.excluded_datasets and not info.excluded
        }

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
            left_columns=[left_column],
            right_columns=[right_column],
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
            if (
                rel.left_dataset in self.selected_datasets
                and rel.right_dataset in self.selected_datasets
                and registry.silver
            ):
                registry.add_silver_join(
                    rel.left_dataset,
                    rel.right_dataset,
                    [rel.left_column],
                    rel.relationship_type,
                    f"Join {rel.left_dataset} with {rel.right_dataset}",
                )

        return registry

    def save(self, path) -> None:
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
                    "raw_source_path": info.raw_source_path,
                    "excluded": info.excluded,
                    "feature_exclusions": [
                        {
                            "column": e.column,
                            "blocked_categories": e.blocked_categories,
                            "blocked_funcs": e.blocked_funcs,
                            "rationale": e.rationale,
                        }
                        for e in info.feature_exclusions
                    ],
                    "excluded_leaking_features": [e.to_dict() for e in info.excluded_leaking_features],
                    "zero_inflation_opt_in": info.zero_inflation_opt_in,
                }
                for name, info in self.datasets.items()
            },
            "relationships": [
                {
                    "left_dataset": rel.left_dataset,
                    "right_dataset": rel.right_dataset,
                    "left_columns": rel.left_columns,
                    "right_columns": rel.right_columns,
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
            "bronze_aggregation_overrides": self.bronze_aggregation_overrides,
            "composite_dataset_name": self.composite_dataset_name,
            "notes": self.notes,
        }

        p = path if isinstance(path, Path) else Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path) -> MultiDatasetFindings:
        """Load multi-dataset findings from YAML."""
        from customer_retention.generators.pipeline_generator.models import FeatureExclusion

        p = path if isinstance(path, Path) else Path(str(path))
        with p.open("r") as f:
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
                raw_source_path=info.get("raw_source_path"),
                excluded=info.get("excluded", False),
                feature_exclusions=[
                    FeatureExclusion(
                        column=e["column"],
                        blocked_categories=e.get("blocked_categories", []),
                        blocked_funcs=e.get("blocked_funcs", []),
                        rationale=e.get("rationale", ""),
                    )
                    for e in info.get("feature_exclusions", [])
                ],
                excluded_leaking_features=[
                    LeakageExclusion.from_dict(e) for e in info.get("excluded_leaking_features") or []
                ],
                zero_inflation_opt_in=info.get("zero_inflation_opt_in", []),
            )

        relationships = [
            DatasetRelationshipInfo(
                left_dataset=rel["left_dataset"],
                right_dataset=rel["right_dataset"],
                left_columns=rel.get("left_columns") or [rel["left_column"]],
                right_columns=rel.get("right_columns") or [rel["right_column"]],
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
            aggregation_windows=data.get(
                "aggregation_windows", ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"]
            ),
            bronze_aggregation_overrides=data.get("bronze_aggregation_overrides") or {},
            composite_dataset_name=data.get("composite_dataset_name"),
            notes=data.get("notes", {}),
        )


class ExplorationManager:
    """Manages multiple exploration findings."""

    def __init__(self, explorations_dir, findings_paths: Optional[List[Path]] = None):
        self.explorations_dir = explorations_dir if hasattr(explorations_dir, "glob") else Path(explorations_dir)
        self._findings_paths = findings_paths
        self._findings_cache: Dict[str, ExplorationFindings] = {}
        self._excluded_datasets: set = set()

    def discover_findings(self, prefer_aggregated: bool = True) -> List[Path]:
        """Discover all findings files in the explorations directory.

        Excludes multi_dataset_findings.yaml as it has a different structure.

        Args:
            prefer_aggregated: If True, when both event-level and aggregated findings
                exist for the same dataset, only return the aggregated one.
        """
        if self._findings_paths is not None:
            return self._findings_paths

        if not self.explorations_dir.exists():
            return []

        all_files = [f for f in self.explorations_dir.glob("*_findings.yaml") if "multi_dataset" not in f.name]

        if not prefer_aggregated:
            return all_files

        return self._filter_prefer_aggregated(all_files)

    def _filter_prefer_aggregated(self, files: List[Path]) -> List[Path]:
        """Filter findings files to prefer aggregated over event-level.

        Groups files by base dataset name and returns aggregated version when available.
        """
        aggregated = {f for f in files if "_aggregated" in f.name}
        non_aggregated = [f for f in files if "_aggregated" not in f.name]

        if not aggregated:
            return files

        result = list(aggregated)
        aggregated_base_names = {self._get_base_name(f) for f in aggregated}

        for f in non_aggregated:
            base_name = self._get_base_name(f)
            if base_name not in aggregated_base_names:
                result.append(f)

        return result

    def _get_base_name(self, path: Path) -> str:
        """Extract base dataset name without aggregated suffix."""
        stem = path.stem.replace("_findings", "")
        if "_aggregated" in stem:
            stem = stem.rsplit("_aggregated", 1)[0]
        return stem

    def _source_event_level_metadata(self, path: Path):
        """Recover ``(TimeSeriesMetadata, raw_source_path)`` from the non-aggregated sibling.

        When `prefer_aggregated=True` serves a `<name>_aggregated_findings.yaml`
        whose `time_series_metadata` is null (aggregation output), the
        event-level signal and the original raw source both live on the sibling
        `<name>_findings.yaml`. The sibling's ``source_path`` is the raw CSV /
        table — the aggregated findings' ``source_path`` points at the
        aggregation OUTPUT and must not be used as the landing raw source.
        Returns None when there is no qualifying event-level sibling.
        """
        if "_aggregated" not in path.stem:
            return None
        sibling = path.parent / f"{self._get_base_name(path)}_findings.yaml"
        if not sibling.exists():
            return None
        sibling_findings = ExplorationFindings.load(str(sibling))
        ts_meta = sibling_findings.time_series_metadata
        if ts_meta is None or ts_meta.granularity != DatasetGranularity.EVENT_LEVEL:
            return None
        return ts_meta, sibling_findings.source_path

    def get_skipped_event_findings(self) -> List[Path]:
        """Return event-level findings that were skipped in favor of aggregated versions."""
        if not self.explorations_dir.exists():
            return []

        all_files = [f for f in self.explorations_dir.glob("*_findings.yaml") if "multi_dataset" not in f.name]

        aggregated = {f for f in all_files if "_aggregated" in f.name}
        if not aggregated:
            return []

        aggregated_base_names = {self._get_base_name(f) for f in aggregated}
        return [f for f in all_files if "_aggregated" not in f.name and self._get_base_name(f) in aggregated_base_names]

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
            raw_source_override: Optional[str] = None
            if findings.is_time_series and findings.time_series_metadata:
                granularity = findings.time_series_metadata.granularity
                entity_col = findings.time_series_metadata.entity_column
                time_col = findings.time_series_metadata.time_column
            else:
                sibling_info = self._source_event_level_metadata(path)
                if sibling_info is not None:
                    source_ts_meta, raw_source_override = sibling_info
                    granularity = DatasetGranularity.EVENT_LEVEL
                    entity_col = source_ts_meta.entity_column
                    time_col = source_ts_meta.time_column
                else:
                    granularity = DatasetGranularity.ENTITY_LEVEL
                    entity_col = None
                    time_col = None

            # Extract dataset name from path
            name = self._extract_dataset_name(path)
            is_excluded = name in self._excluded_datasets

            if not include_excluded and is_excluded:
                continue

            datasets.append(
                DatasetInfo(
                    name=name,
                    findings_path=str(path),
                    source_path=findings.source_path,
                    granularity=granularity,
                    row_count=findings.row_count,
                    column_count=findings.column_count,
                    entity_column=entity_col,
                    time_column=time_col,
                    target_column=findings.target_column,
                    raw_source_path=raw_source_override,
                    excluded=is_excluded,
                )
            )

        return datasets

    def create_multi_dataset_findings(
        self,
        dataset_names: Optional[List[str]] = None,
        merge_scaffold=None,
        recommendations_registry=None,
        bronze_aggregation_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> MultiDatasetFindings:
        """Create a MultiDatasetFindings from discovered datasets.

        Args:
            dataset_names: Optional list of dataset names to include.
                          If None, all discovered datasets are included.
                          If provided, only datasets matching these names are included.
            merge_scaffold: Optional list of MergeScaffoldEntry from ProjectContext.
                           Converted to relationships for the silver layer.
            recommendations_registry: Optional RecommendationRegistry whose
                          `bronze_aggregations` is propagated into the multi
                          findings so the NB10 parser sees the override.
            bronze_aggregation_overrides: Direct override dict, used when no
                          registry is available (e.g. cycles loading raw YAML).
                          Wins over the registry when both are passed.
        """
        datasets_info = self.list_datasets(include_excluded=True)

        # Filter to specified datasets if provided
        if dataset_names:
            datasets_info = [d for d in datasets_info if d.name in dataset_names]

        datasets = {d.name: d for d in datasets_info}
        event_datasets = [d.name for d in datasets_info if d.granularity == DatasetGranularity.EVENT_LEVEL]
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

        relationships = self._scaffold_to_relationships(merge_scaffold) if merge_scaffold else []

        # Resolve bronze_aggregation_overrides: explicit arg > registry attr > {}.
        resolved_overrides: Dict[str, Dict[str, Any]] = {}
        if recommendations_registry is not None:
            for attr in ("bronze_aggregations", "bronze_aggregation_overrides"):
                _candidate = getattr(recommendations_registry, attr, None)
                if _candidate:
                    resolved_overrides = dict(_candidate)
                    break
        if bronze_aggregation_overrides:
            resolved_overrides = dict(bronze_aggregation_overrides)

        return MultiDatasetFindings(
            datasets=datasets,
            relationships=relationships,
            primary_entity_dataset=primary,
            event_datasets=event_datasets,
            excluded_datasets=excluded,
            bronze_aggregation_overrides=resolved_overrides,
        )

    @staticmethod
    def _scaffold_to_relationships(scaffold) -> List[DatasetRelationshipInfo]:
        result = []
        for entry in scaffold:
            result.append(
                DatasetRelationshipInfo(
                    left_dataset=entry.right_dataset,
                    right_dataset=entry.left_dataset,
                    left_columns=list(entry.join_keys),
                    right_columns=list(entry.join_keys),
                    relationship_type=entry.relationship,
                    confidence=1.0,
                    auto_detected=True,
                )
            )
        return result

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
        """Extract dataset name from findings path.

        Pattern: {name}_findings.yaml or {name}_aggregated_findings.yaml
        """
        stem = path.stem.replace("_findings", "")
        if "_aggregated" in stem:
            stem = stem.rsplit("_aggregated", 1)[0]
        return stem
