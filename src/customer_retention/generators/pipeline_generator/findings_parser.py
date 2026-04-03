import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)

from customer_retention.analysis.auto_explorer.exploration_manager import (
    DatasetInfo,
    DatasetRelationshipInfo,
    MultiDatasetFindings,
)
from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry
from customer_retention.core.config.column_config import ColumnType

from .models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    BronzeLayerConfig,
    DatetimeDerivationConfig,
    DeduplicationConfig,
    GoldLayerConfig,
    HistoryWindowConfig,
    KeyResolutionStepConfig,
    LabelTimestampConfig,
    LandingLayerConfig,
    LifecycleConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TemporalFeatureConfig,
    TemporalMergeSourceConfig,
    TextFeatureConfig,
    TimestampCoalesceConfig,
    TrainingConfig,
    TransformationStep,
)

_EXPLORATION_TO_PRODUCTION_MODEL = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "Gradient Boosting": "xgboost",
    "XGBoost": "xgboost",
    "GBT": "xgboost",
    "GBTClassifier": "xgboost",
}


def _edges_to_labels(edges: List[float]) -> List[str]:
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"{int(edges[i])}-{int(edges[i + 1])}d")
    labels.append(f"{int(edges[-1])}d+")
    return labels


class FindingsParser:
    def __init__(self, findings_dir: str, namespace=None, intent=None):
        self._findings_dir = Path(findings_dir)
        self._namespace = namespace
        self._intent = intent
        self._source_findings_paths: Dict[str, Path] = {}
        self._raw_source_columns: Dict[str, Set[str]] = {}

    def parse(self) -> PipelineConfig:
        multi_dataset = self._load_multi_dataset_findings()
        selected_sources = list(multi_dataset.datasets.keys())
        source_findings = self._load_source_findings(selected_sources, self._findings_dir, multi_dataset)
        discovered_events = self._discover_event_sources(source_findings)
        self._index_raw_source_columns(source_findings)
        self._index_raw_source_columns(discovered_events)
        recommendations_registry = self._load_recommendations()
        recommendations_hash = (
            recommendations_registry.compute_recommendations_hash() if recommendations_registry else None
        )
        config = self._build_pipeline_config(multi_dataset, source_findings, recommendations_hash)
        config.training = self._build_training_config(multi_dataset, source_findings)
        self._build_landing_configs(config, multi_dataset, source_findings)
        self._build_discovered_landing_configs(config, discovered_events, multi_dataset)
        self._build_bronze_event_configs(config, multi_dataset, source_findings, discovered_events)
        if recommendations_registry:
            self._apply_recommendations_to_config(config, recommendations_registry, multi_dataset)
            self._apply_event_recommendations(config, recommendations_registry)
        config.gold.feature_exclusion_prefixes = self._collect_leakage_exclusion_prefixes(source_findings, multi_dataset)
        self._reconcile_discovered_event_transforms(config, discovered_events)
        self._reconcile_event_post_shaping(config)
        self._reconcile_bronze_columns(config)
        self._reconcile_gold_columns(config)
        return config

    def _index_raw_source_columns(self, discovered_events: Dict[str, ExplorationFindings]) -> None:
        for name, findings in discovered_events.items():
            self._raw_source_columns[name] = set(findings.columns.keys())

    def _load_recommendations(self) -> Optional[RecommendationRegistry]:
        if self._namespace is not None:
            recommendations_path = self._namespace.merged_recommendations_path
            if recommendations_path.exists():
                with recommendations_path.open("r") as f:
                    return RecommendationRegistry.from_dict(yaml.safe_load(f))
            return None
        recommendations_path = None
        pattern_matches = sorted(self._findings_dir.glob("*_recommendations.yaml"))
        if pattern_matches:
            recommendations_path = pattern_matches[-1]
        elif (self._findings_dir / "recommendations.yaml").exists():
            recommendations_path = self._findings_dir / "recommendations.yaml"
        if recommendations_path and recommendations_path.exists():
            with recommendations_path.open("r") as f:
                return RecommendationRegistry.from_dict(yaml.safe_load(f))
        return None

    def _load_multi_dataset_findings(self) -> MultiDatasetFindings:
        if self._namespace is not None:
            path = self._namespace.multi_dataset_findings_path
        else:
            path = self._findings_dir / "multi_dataset_findings.yaml"
        if not path.exists():
            return self._synthesize_from_single_source()
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return self._dict_to_multi_dataset_findings(data)

    def _synthesize_from_single_source(self) -> MultiDatasetFindings:
        from customer_retention.core.config.column_config import DatasetGranularity

        candidates = []
        if self._namespace is not None:
            candidates = self._namespace.discover_all_findings(prefer_aggregated=True)
        if not candidates:
            candidates = [
                p for p in self._findings_dir.glob("*_findings.yaml") if p.name != "multi_dataset_findings.yaml"
            ]
        if not candidates:
            raise FileNotFoundError(f"No findings files found in {self._findings_dir}")

        datasets = {}
        first_name = None
        for path in candidates:
            findings = ExplorationFindings.load(str(path))
            name = path.stem.replace("_findings", "")
            if first_name is None:
                first_name = name
            datasets[name] = DatasetInfo(
                name=name,
                findings_path=str(path),
                source_path=findings.source_path,
                granularity=DatasetGranularity.ENTITY_LEVEL,
                row_count=findings.row_count,
                column_count=findings.column_count,
                entity_column=(findings.identifier_columns[0] if findings.identifier_columns else None),
                target_column=findings.target_column,
            )

        return MultiDatasetFindings(
            datasets=datasets,
            primary_entity_dataset=first_name,
        )

    def _strip_aggregated(self, name: str) -> str:
        if name and name.endswith("_aggregated"):
            return name[: -len("_aggregated")]
        return name

    def _dict_to_multi_dataset_findings(self, data: Dict) -> MultiDatasetFindings:
        from customer_retention.core.config.column_config import DatasetGranularity

        datasets = {}
        for name, info in data.get("datasets", {}).items():
            clean_name = self._strip_aggregated(name)
            granularity_str = info.get("granularity", "unknown")
            granularity = DatasetGranularity(granularity_str) if granularity_str else DatasetGranularity.UNKNOWN
            datasets[clean_name] = DatasetInfo(
                name=self._strip_aggregated(info["name"]),
                findings_path=info.get("findings_path", ""),
                source_path=info.get("source_path", ""),
                granularity=granularity,
                row_count=info.get("row_count", 0),
                column_count=info.get("column_count", 0),
                entity_column=info.get("entity_column"),
                time_column=info.get("time_column"),
                target_column=info.get("target_column"),
                excluded=info.get("excluded", False),
            )
        relationships = [
            DatasetRelationshipInfo(
                left_dataset=r["left_dataset"],
                right_dataset=r["right_dataset"],
                left_columns=r.get("left_columns") or [r["left_column"]],
                right_columns=r.get("right_columns") or [r["right_column"]],
                relationship_type=r.get("relationship_type", "one_to_many"),
                confidence=r.get("confidence", 1.0),
                auto_detected=r.get("auto_detected", False),
            )
            for r in data.get("relationships", [])
        ]
        return MultiDatasetFindings(
            datasets=datasets,
            relationships=relationships,
            primary_entity_dataset=self._strip_aggregated(data.get("primary_entity_dataset", "") or ""),
            event_datasets=[self._strip_aggregated(e) for e in data.get("event_datasets", [])],
            excluded_datasets=data.get("excluded_datasets", []),
            aggregation_windows=data.get(
                "aggregation_windows", ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"]
            ),
            notes=data.get("notes", {}),
        )

    def _load_source_findings(
        self, sources: List[str], findings_dir: Path, multi_dataset: MultiDatasetFindings = None
    ) -> Dict[str, ExplorationFindings]:
        result = {}
        for name in sources:
            path = None
            if multi_dataset and name in multi_dataset.datasets:
                dataset_info = multi_dataset.datasets[name]
                if dataset_info.findings_path:
                    raw_path = Path(dataset_info.findings_path)
                    if raw_path.is_absolute():
                        path = raw_path
                    else:
                        path = (findings_dir / raw_path).resolve()
                        if not path.exists():
                            path = findings_dir / raw_path.name
            if (path is None or not path.exists()) and self._namespace is not None:
                ns_dir = self._namespace.dataset_findings_dir(name)
                candidates = sorted(ns_dir.glob(f"{name}_*_findings.yaml")) if ns_dir.is_dir() else []
                if candidates:
                    path = candidates[0]
                elif (ns_dir / f"{name}_findings.yaml").exists():
                    path = ns_dir / f"{name}_findings.yaml"
            if path is None or not path.exists():
                candidates = list(findings_dir.glob(f"{name}_*_findings.yaml"))
                if candidates:
                    path = candidates[0]
                else:
                    path = findings_dir / f"{name}_findings.yaml"
            if path.exists():
                result[name] = ExplorationFindings.load(str(path))
                self._source_findings_paths[name] = path.resolve()
        return result

    def _build_pipeline_config(
        self,
        multi: MultiDatasetFindings,
        sources: Dict[str, ExplorationFindings],
        recommendations_hash: Optional[str] = None,
    ) -> PipelineConfig:
        source_configs = self._build_source_configs(multi, sources)
        bronze_configs = self._build_bronze_configs(sources, source_configs)
        return PipelineConfig(
            name="",
            target_column=self._find_target_column(sources),
            sources=source_configs,
            bronze=bronze_configs,
            silver=self._build_silver_config(multi, sources),
            gold=self._build_gold_config(sources),
            output_dir="",
            recommendations_hash=recommendations_hash,
        )

    def _build_source_configs(
        self, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings]
    ) -> List[SourceConfig]:
        result = []
        for name, findings in sources.items():
            dataset_info = multi.datasets.get(name)
            is_event = name in multi.event_datasets
            is_excluded = name in multi.excluded_datasets or (dataset_info and dataset_info.excluded)
            raw_source = str(Path(
                dataset_info.raw_source_path or dataset_info.source_path
                if dataset_info else findings.source_path
            ).resolve())
            time_col = None
            entity_key = findings.identifier_columns[0] if findings.identifier_columns else "id"
            if is_event and findings.time_series_metadata:
                time_col = findings.time_series_metadata.time_column
                if findings.time_series_metadata.entity_column:
                    entity_key = findings.time_series_metadata.entity_column
            result.append(
                SourceConfig(
                    name=name,
                    path=Path(findings.source_path).name,
                    format=self._infer_format(raw_source),
                    entity_key=entity_key,
                    raw_source_path=raw_source,
                    time_column=time_col,
                    is_event_level=is_event,
                    excluded=is_excluded,
                )
            )
        return result

    def _build_bronze_configs(
        self, sources: Dict[str, ExplorationFindings], source_configs: List[SourceConfig]
    ) -> Dict[str, BronzeLayerConfig]:
        result = {}
        source_map = {s.name: s for s in source_configs}
        for name, findings in sources.items():
            source_cfg = source_map[name]
            if source_cfg.is_event_level:
                continue
            result[name] = BronzeLayerConfig(source=source_cfg, transformations=self._extract_transformations(findings))
        return result

    def _extract_transformations(self, findings: ExplorationFindings) -> List[TransformationStep]:
        transformations = []
        for col_name, col_finding in findings.columns.items():
            if not col_finding.cleaning_needed:
                continue
            for rec in col_finding.cleaning_recommendations:
                step = self._parse_cleaning_recommendation(col_name, rec)
                if step:
                    transformations.append(step)
        return transformations

    def _parse_cleaning_recommendation(self, column: str, recommendation: str) -> TransformationStep:
        if ":" in recommendation:
            action, param = recommendation.split(":", 1)
        else:
            action, param = recommendation, ""
        if action == "impute_null":
            return TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column=column,
                parameters={"value": param if param else 0},
                rationale=f"Impute nulls in {column}",
            )
        if action == "cap_outlier":
            return TransformationStep(
                type=PipelineTransformationType.CAP_OUTLIER,
                column=column,
                parameters={"method": param if param else "iqr"},
                rationale=f"Cap outliers in {column}",
            )
        return None

    def _build_silver_config(
        self, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings]
    ) -> SilverLayerConfig:
        joins = []
        event_set = set(multi.event_datasets)
        for rel in multi.relationships:
            left_keys = list(rel.left_columns)
            right_keys = list(rel.right_columns)
            if rel.right_dataset in event_set:
                left_keys = self._strip_temporal_join_keys(left_keys)
                right_keys = self._strip_temporal_join_keys(right_keys)
            joins.append(
                {
                    "left_keys": left_keys,
                    "right_keys": right_keys,
                    "right_source": rel.right_dataset,
                    "how": "left",
                }
            )
        grid_dates, entity_key, merge_sources = self._build_temporal_merge_metadata(multi, sources)
        holdout_ids = self._load_holdout_entity_ids()
        return SilverLayerConfig(
            joins=joins,
            aggregations=[],
            grid_dates=grid_dates,
            entity_key=entity_key,
            merge_sources=merge_sources,
            holdout_entity_ids=holdout_ids,
        )

    def _load_holdout_entity_ids(self) -> Optional[list]:
        """Load pre-computed holdout entity IDs from namespace or findings dir."""
        import json

        path = None
        if self._namespace is not None:
            ns_path = self._namespace.holdout_entity_ids_path
            if ns_path.exists():
                path = ns_path
        if path is None:
            fallback = self._findings_dir / "holdout_entity_ids.json"
            if fallback.exists():
                path = fallback
        if path is None:
            return None
        return json.loads(path.read_text())

    def _build_temporal_merge_metadata(
        self,
        multi: MultiDatasetFindings,
        sources: Dict[str, ExplorationFindings],
    ) -> Tuple[List[str], Optional[str], List[TemporalMergeSourceConfig]]:
        if self._namespace is None:
            return [], None, []
        grid_path = self._namespace.snapshot_grid_path
        if not grid_path.exists():
            return [], None, []

        from customer_retention.analysis.auto_explorer.snapshot_grid import SnapshotGrid

        grid = SnapshotGrid.load(grid_path)
        grid_dates = list(grid.grid_dates) if grid.grid_dates else []
        if not grid_dates:
            return [], None, []

        entity_key = None
        key_resolutions: Dict[str, list] = {}
        entity_asof_ts: Dict[str, str] = {}
        ctx_path = self._namespace.project_context_path
        if ctx_path.exists():
            from customer_retention.analysis.auto_explorer.project_context import (
                ProjectContext,
                RawTimeColumnRole,
            )

            ctx = ProjectContext.load(ctx_path)
            entity_key = ctx.entity_column
            for ds_name, ds_entry in ctx.datasets.items():
                if ds_entry.key_resolution:
                    key_resolutions[ds_name] = [
                        KeyResolutionStepConfig(
                            bridge_dataset=step.bridge_dataset,
                            source_key=step.source_key,
                            bridge_key=step.bridge_key,
                            resolve_column=step.resolve_column,
                        )
                        for step in ds_entry.key_resolution
                    ]
                if (ds_entry.time_column
                        and ds_entry.raw_time_column_role != RawTimeColumnRole.ENTITY_UPDATE_TIME):
                    entity_asof_ts[ds_name] = ds_entry.time_column

        from customer_retention.core.config.column_config import DatasetGranularity

        event_set = set(multi.event_datasets)
        excluded_set = set(multi.excluded_datasets)
        merge_srcs: List[TemporalMergeSourceConfig] = []
        for name, dataset_info in multi.datasets.items():
            if name in excluded_set or dataset_info.excluded:
                continue
            is_event = name in event_set
            granularity = DatasetGranularity.EVENT_LEVEL.value if is_event else DatasetGranularity.ENTITY_LEVEL.value
            ts_col = None
            if is_event:
                ts_col = dataset_info.time_column
                if not ts_col:
                    findings = sources.get(name)
                    if findings and findings.time_series_metadata:
                        ts_col = findings.time_series_metadata.time_column
            else:
                ts_col = entity_asof_ts.get(name)
            merge_srcs.append(TemporalMergeSourceConfig(
                name=name,
                granularity=granularity,
                feature_timestamp_column=ts_col,
                key_resolution_steps=key_resolutions.get(name, []),
            ))
        return grid_dates, entity_key, merge_srcs

    @staticmethod
    def _strip_temporal_join_keys(keys: List[str]) -> List[str]:
        temporal = {"as_of_date", "event_timestamp", "feature_timestamp"}
        filtered = [k for k in keys if k not in temporal]
        return filtered or keys

    _CATEGORICAL_TYPES = {ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL, ColumnType.CATEGORICAL_CYCLICAL}
    _NUMERIC_TYPES = {ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE}

    def _build_gold_config(self, sources: Dict[str, ExplorationFindings]) -> GoldLayerConfig:
        encodings = []
        scalings = []
        target_column = self._find_target_column(sources)
        for findings in sources.values():
            for col_name, col_finding in findings.columns.items():
                if col_name == target_column:
                    continue
                col_type = col_finding.inferred_type
                if col_type in self._CATEGORICAL_TYPES:
                    encodings.append(
                        TransformationStep(
                            type=PipelineTransformationType.ENCODE,
                            column=col_name,
                            parameters={"method": "one_hot"},
                            rationale=f"One-hot encode {col_name}",
                        )
                    )
                elif col_type in self._NUMERIC_TYPES:
                    scalings.append(
                        TransformationStep(
                            type=PipelineTransformationType.SCALE,
                            column=col_name,
                            parameters={"method": "standard"},
                            rationale=f"Standardize {col_name}",
                        )
                    )
        return GoldLayerConfig(encodings=encodings, scalings=scalings)

    def _find_target_column(self, sources: Dict[str, ExplorationFindings]) -> str:
        for findings in sources.values():
            if findings.target_column:
                return findings.target_column
        return "target"

    def _apply_recommendations_to_config(
        self, config: PipelineConfig, registry: RecommendationRegistry, multi: MultiDatasetFindings
    ) -> None:
        self._apply_bronze_recommendations(config, registry)
        self._apply_imbalance_recommendations(config, registry)
        self._apply_silver_recommendations(config, registry)
        self._apply_gold_recommendations(config, registry)

    def _apply_event_recommendations(
        self, config: PipelineConfig, registry: RecommendationRegistry
    ) -> None:
        self._apply_dedup_recommendations(config, registry)
        self._apply_filter_recommendations(config, registry)

    def _resolve_bronze_source_name(self, config: PipelineConfig, target_bronze) -> Optional[str]:
        for name, cfg in config.bronze.items():
            if cfg is target_bronze:
                return name
        return None

    def _apply_bronze_recommendations(self, config: PipelineConfig, registry: RecommendationRegistry) -> None:
        sources_to_process = dict(registry.sources)
        if not sources_to_process and hasattr(registry, "bronze") and registry.bronze is not None:
            sources_to_process = {"_default": registry.bronze}
        for source_name, bronze_recs in sources_to_process.items():
            target_bronze = self._find_bronze_config_for_source(config, source_name, bronze_recs.source_file)
            if target_bronze is None:
                continue
            resolved = self._resolve_bronze_source_name(config, target_bronze) or source_name
            for rec in bronze_recs.null_handling:
                if not self._column_in_raw_source(resolved, rec.target_column):
                    continue
                step = self._map_bronze_null(rec)
                if step:
                    target_bronze.transformations.append(step)
            for rec in bronze_recs.outlier_handling:
                if not self._column_in_raw_source(resolved, rec.target_column):
                    continue
                step = self._map_bronze_outlier(rec)
                if step:
                    target_bronze.transformations.append(step)
            target_bronze.transformations = self._deduplicate_steps(target_bronze.transformations)

    def _apply_imbalance_recommendations(self, config: PipelineConfig, registry: RecommendationRegistry) -> None:
        all_sources = dict(registry.sources)
        if not all_sources and hasattr(registry, "bronze") and registry.bronze is not None:
            all_sources = {"_default": registry.bronze}
        for _source_name, bronze_recs in all_sources.items():
            for rec in bronze_recs.modeling_strategy:
                if rec.category == "imbalance":
                    if config.training is None:
                        config.training = TrainingConfig()
                    config.training.imbalance_strategy = rec.action
                    ratio = rec.parameters.get("imbalance_ratio")
                    if ratio is not None:
                        config.training.imbalance_ratio = float(ratio)
                    return

    def _apply_dedup_recommendations(self, config: PipelineConfig, registry: RecommendationRegistry) -> None:
        all_sources = dict(registry.sources)
        if not all_sources and hasattr(registry, "bronze") and registry.bronze is not None:
            all_sources = {"_default": registry.bronze}
        for source_name, bronze_recs in all_sources.items():
            if not bronze_recs.deduplication:
                continue
            rec = bronze_recs.deduplication[0]
            dedup_config = DeduplicationConfig(
                strategy=rec.action,
                conflict_columns=rec.parameters.get("conflict_columns", []),
            )
            target_event = self._find_event_config_for_source(config, source_name, bronze_recs.source_file)
            if target_event is not None:
                target_event.deduplicate = dedup_config

    def _find_event_config_for_source(
        self, config: PipelineConfig, source_name: str, source_file: str
    ) -> Optional[BronzeEventConfig]:
        if source_name in config.bronze_event:
            return config.bronze_event[source_name]
        source_path = Path(source_file) if source_file else None
        for _name, event_cfg in config.bronze_event.items():
            if source_path and Path(event_cfg.source.path).name == source_path.name:
                return event_cfg
            if source_path and Path(event_cfg.source.raw_source_path).name == source_path.name:
                return event_cfg
        if len(config.bronze_event) == 1:
            return next(iter(config.bronze_event.values()))
        return None

    def _apply_filter_recommendations(self, config: PipelineConfig, registry: RecommendationRegistry) -> None:
        all_sources = dict(registry.sources)
        if not all_sources and hasattr(registry, "bronze") and registry.bronze is not None:
            all_sources = {"_default": registry.bronze}
        for source_name, bronze_recs in all_sources.items():
            for rec in bronze_recs.filtering:
                step = self._map_bronze_filter(rec)
                if not step:
                    continue
                target_event = self._find_event_config_for_source(config, source_name, bronze_recs.source_file)
                target_bronze = self._find_bronze_config_for_source(config, source_name, bronze_recs.source_file)
                if target_event:
                    resolved = self._resolve_event_source_name(config, target_event)
                elif target_bronze:
                    resolved = self._resolve_bronze_source_name(config, target_bronze) or source_name
                else:
                    resolved = source_name
                if not self._column_in_raw_source(resolved, step.column):
                    continue
                if target_bronze is not None:
                    target_bronze.transformations.append(step)
                if target_event is not None:
                    target_event.pre_shaping.append(step)

    def _resolve_event_source_name(self, config: PipelineConfig, event_cfg: BronzeEventConfig) -> str:
        for name, cfg in config.bronze_event.items():
            if cfg is event_cfg:
                return name
        return event_cfg.source.name

    def _column_in_raw_source(self, source_name: str, column: str) -> bool:
        raw_cols = self._raw_source_columns.get(source_name)
        if raw_cols is None:
            return False
        return column in raw_cols

    def _map_bronze_filter(self, rec) -> Optional[TransformationStep]:
        return TransformationStep(
            type=PipelineTransformationType.FILTER,
            column=rec.target_column,
            parameters=dict(rec.parameters),
            rationale=rec.rationale,
            source_notebook=rec.source_notebook,
        )

    @staticmethod
    def _deduplicate_steps(steps: List[TransformationStep]) -> List[TransformationStep]:
        seen: Set[Tuple[PipelineTransformationType, str]] = set()
        result: List[TransformationStep] = []
        for step in steps:
            key = (step.type, step.column)
            if key not in seen:
                seen.add(key)
                result.append(step)
        return result

    def _find_bronze_config_for_source(
        self, config: PipelineConfig, source_name: str, source_file: str
    ) -> Optional[BronzeLayerConfig]:
        if source_name in config.bronze:
            return config.bronze[source_name]
        source_path = Path(source_file) if source_file else None
        for name, bronze in config.bronze.items():
            if source_path and Path(bronze.source.path).name == source_path.name:
                return bronze
            if source_path and Path(bronze.source.raw_source_path).name == source_path.name:
                return bronze
        if len(config.bronze) == 1:
            return next(iter(config.bronze.values()))
        return None

    def _map_bronze_null(self, rec) -> Optional[TransformationStep]:
        strategy = rec.parameters.get("strategy", "median")
        if strategy == "drop":
            return TransformationStep(
                type=PipelineTransformationType.DROP_COLUMN,
                column=rec.target_column,
                parameters={"strategy": "drop"},
                rationale=rec.rationale,
                source_notebook=rec.source_notebook,
            )
        return TransformationStep(
            type=PipelineTransformationType.IMPUTE_NULL,
            column=rec.target_column,
            parameters={"value": strategy},
            rationale=rec.rationale,
            source_notebook=rec.source_notebook,
        )

    def _map_bronze_outlier(self, rec) -> Optional[TransformationStep]:
        if rec.action == "segment_aware_cap":
            return TransformationStep(
                type=PipelineTransformationType.SEGMENT_AWARE_CAP,
                column=rec.target_column,
                parameters={
                    "method": rec.parameters.get("method", "segment_iqr"),
                    "n_segments": rec.parameters.get("n_segments", 2),
                },
                rationale=rec.rationale,
                source_notebook=rec.source_notebook,
            )
        if rec.action == "winsorize":
            return TransformationStep(
                type=PipelineTransformationType.WINSORIZE,
                column=rec.target_column,
                parameters={
                    "lower_bound": rec.parameters.get("lower_bound", 0),
                    "upper_bound": rec.parameters.get("upper_bound", 1000000),
                },
                rationale=rec.rationale,
                source_notebook=rec.source_notebook,
            )
        return TransformationStep(
            type=PipelineTransformationType.CAP_OUTLIER,
            column=rec.target_column,
            parameters={"method": rec.parameters.get("method", "iqr")},
            rationale=rec.rationale,
            source_notebook=rec.source_notebook,
        )

    def _apply_silver_recommendations(self, config: PipelineConfig, registry: RecommendationRegistry) -> None:
        if not hasattr(registry, "silver") or registry.silver is None:
            return
        pipeline_columns = self._collect_pipeline_columns(config)
        for rec in getattr(registry.silver, "derived_columns", []):
            if not self._silver_derived_sources_available(rec, pipeline_columns):
                continue
            step = self._map_silver_derived(rec)
            if step:
                config.silver.derived_columns.append(step)

    def _map_silver_derived(self, rec) -> Optional[TransformationStep]:
        action = rec.action
        params = dict(rec.parameters)
        if action in ("ratio", "interaction", "composite"):
            return TransformationStep(
                type=PipelineTransformationType.DERIVED_COLUMN,
                column=rec.target_column,
                parameters={"action": action, **params},
                rationale=rec.rationale,
                source_notebook=rec.source_notebook,
            )
        return None

    def _collect_pipeline_columns(self, config: PipelineConfig) -> Set[str]:
        columns: Set[str] = {"entity_id", "as_of_date"}
        for name, bronze in config.bronze.items():
            raw_cols = self._raw_source_columns.get(name, set())
            dropped = {
                step.column for step in bronze.transformations
                if step.type == PipelineTransformationType.DROP_COLUMN
            }
            columns |= (raw_cols - dropped)
        for _name, event_cfg in config.bronze_event.items():
            columns |= self._event_aggregated_columns(event_cfg)
        return columns

    @staticmethod
    def _predict_gold_generated_columns(config: "PipelineConfig") -> Set[str]:
        generated: Set[str] = set()
        for step in config.gold.transformations:
            if step.type == PipelineTransformationType.ZERO_INFLATION_HANDLING:
                generated.add(f"{step.column}_is_zero")
                generated.add(f"{step.column}_log")
        return generated

    @staticmethod
    def _event_aggregated_columns(event_cfg: "BronzeEventConfig") -> Set[str]:
        columns: Set[str] = set()
        agg = event_cfg.aggregation
        if agg:
            blocked = agg.column_blocked_funcs if agg else {}
            for window in agg.windows:
                for col in agg.value_columns:
                    col_blocked = set(blocked.get(col, []))
                    for func in agg.agg_funcs:
                        if func == "count" or func in col_blocked:
                            continue
                        columns.add(f"{col}_{func}_{window}")
                for col in agg.categorical_columns:
                    col_blocked = set(blocked.get(col, []))
                    for func in agg.categorical_agg_funcs:
                        if func in col_blocked:
                            continue
                        columns.add(f"{col}_{func}_{window}")
                for col in agg.binary_columns:
                    col_blocked = set(blocked.get(col, []))
                    for func in agg.binary_agg_funcs:
                        if func in col_blocked:
                            continue
                        columns.add(f"{col}_{func}_{window}")
                columns.add(f"event_count_{window}")
        lc = event_cfg.lifecycle
        if lc:
            columns |= {"days_since_last", "days_since_first"}
            if lc.include_recency_bucket:
                columns.add("recency_bucket")
            if lc.include_lifecycle_quadrant:
                columns.add("lifecycle_quadrant")
            if lc.include_cyclical_features:
                columns |= {"dow_sin", "dow_cos"}
            if lc.include_month_cyclical:
                columns |= {"month_sin", "month_cos"}
            if lc.include_quarter_cyclical:
                columns |= {"quarter_sin", "quarter_cos"}
            if lc.include_trend_features:
                columns |= {"recent_vs_overall_ratio", "entity_trend_slope"}
            if lc.include_cohort_features:
                columns |= {"cohort_year", "cohort_quarter"}
            for pair in lc.momentum_pairs:
                short = pair.get("short_window", "")
                long = pair.get("long_window", "")
                columns.add(f"momentum_{short}_{long}")
        dd = event_cfg.datetime_derivation
        if dd:
            for src in dd.source_columns:
                for suffix in ("_delta_hours", "_hour", "_dow", "_is_weekend"):
                    columns.add(f"{src}{suffix}")
        for text_cfg in event_cfg.text_features:
            if text_cfg.component_columns:
                columns |= set(text_cfg.component_columns)
            else:
                for i in range(text_cfg.n_components):
                    columns.add(f"{text_cfg.column}_emb_{i}")
        tf = event_cfg.temporal_features
        if tf and tf.lag_columns:
            groups = set(tf.feature_groups or [])
            for lag_idx in range(tf.num_lags):
                for col in tf.lag_columns:
                    for fn in tf.lag_agg_funcs:
                        columns.add(f"lag{lag_idx}_{col}_{fn}")
            if "velocity" in groups:
                for col in tf.lag_columns:
                    columns |= {f"{col}_velocity", f"{col}_velocity_pct"}
            if "acceleration" in groups:
                for col in tf.lag_columns:
                    columns |= {f"{col}_acceleration", f"{col}_momentum"}
            if "lifecycle" in groups:
                for col in tf.lag_columns:
                    columns |= {f"{col}_beginning", f"{col}_middle", f"{col}_end", f"{col}_trend_ratio"}
            if "recency" in groups:
                columns |= {"days_since_last_event", "days_since_first_event", "active_span_days", "recency_ratio"}
            if "regularity" in groups:
                columns |= {"event_frequency", "inter_event_gap_mean", "inter_event_gap_std", "inter_event_gap_max", "regularity_score"}
            if "cohort_comparison" in groups:
                for col in tf.lag_columns:
                    columns |= {f"{col}_vs_cohort_mean", f"{col}_vs_cohort_pct", f"{col}_cohort_zscore"}
        return columns

    @staticmethod
    def _reconcile_bronze_columns(config: "PipelineConfig") -> None:
        for bronze in config.bronze.values():
            drop_targets = {
                s.column for s in bronze.transformations
                if s.type == PipelineTransformationType.DROP_COLUMN
            }
            if not drop_targets:
                continue
            before = len(bronze.transformations)
            bronze.transformations = [
                s for s in bronze.transformations
                if s.type == PipelineTransformationType.DROP_COLUMN or s.column not in drop_targets
            ]
            removed = before - len(bronze.transformations)
            if removed:
                logger.warning(
                    "Removed %d bronze step(s) targeting dropped columns in '%s': %s",
                    removed, bronze.source.name, sorted(drop_targets),
                )
        for name, event_cfg in config.bronze_event.items():
            drop_targets = {
                s.column for s in event_cfg.pre_shaping
                if s.type == PipelineTransformationType.DROP_COLUMN
            }
            if not drop_targets:
                continue
            before = len(event_cfg.pre_shaping)
            event_cfg.pre_shaping = [
                s for s in event_cfg.pre_shaping
                if s.type == PipelineTransformationType.DROP_COLUMN or s.column not in drop_targets
            ]
            removed = before - len(event_cfg.pre_shaping)
            if removed:
                logger.warning(
                    "Removed %d bronze event pre-shaping step(s) targeting dropped columns in '%s': %s",
                    removed, name, sorted(drop_targets),
                )

    def _reconcile_event_post_shaping(self, config: "PipelineConfig") -> None:
        for name, event_cfg in config.bronze_event.items():
            if not event_cfg.post_shaping:
                continue
            valid_columns = self._event_aggregated_columns(event_cfg)
            if not valid_columns:
                continue
            dropped = [s.column for s in event_cfg.post_shaping if s.column not in valid_columns]
            if dropped:
                event_cfg.post_shaping = [s for s in event_cfg.post_shaping if s.column in valid_columns]
                logger.warning(
                    "Dropped post-shaping step(s) from '%s' — columns not in aggregated output: %s",
                    name, ", ".join(dropped),
                )

    @staticmethod
    def _silver_derived_sources_available(rec, pipeline_columns: Set[str]) -> bool:
        action = rec.action
        params = rec.parameters
        if action == "ratio":
            needed = {params.get("numerator", ""), params.get("denominator", "")}
        elif action == "interaction":
            needed = set(params.get("features", []))
        elif action == "composite":
            needed = set(params.get("columns", []))
            if not needed:
                logger.warning(
                    "Skipping silver derived-column '%s': composite recommendation has no 'columns' key",
                    rec.target_column,
                )
                return False
        else:
            return True
        missing = needed - pipeline_columns
        if missing:
            logger.warning(
                "Skipping silver derived-column '%s': missing source columns %s",
                rec.target_column,
                sorted(missing),
            )
            return False
        return True

    def _apply_gold_recommendations(self, config: PipelineConfig, registry: RecommendationRegistry) -> None:
        if not hasattr(registry, "gold") or registry.gold is None:
            return
        gold = registry.gold
        pipeline_columns = self._collect_pipeline_columns(config)
        for step in config.silver.derived_columns:
            pipeline_columns.add(step.column)
        pipeline_columns |= self._predict_gold_generated_columns(config)
        seen_encoding_columns: Set[str] = {e.column for e in config.gold.encodings}
        for rec in getattr(gold, "encoding", []):
            if rec.target_column == config.target_column:
                continue
            if rec.target_column in seen_encoding_columns:
                continue
            if rec.target_column not in pipeline_columns:
                logger.warning(
                    "Skipping gold encoding '%s': column not in pipeline",
                    rec.target_column,
                )
                continue
            seen_encoding_columns.add(rec.target_column)
            method = rec.parameters.get("method", rec.action)
            if method in ("onehot", "one_hot"):
                method = "one_hot"
            config.gold.encodings.append(
                TransformationStep(
                    type=PipelineTransformationType.ENCODE,
                    column=rec.target_column,
                    parameters={"method": method},
                    rationale=rec.rationale,
                    source_notebook=rec.source_notebook,
                )
            )
        seen_scaling_columns: Set[str] = {s.column for s in config.gold.scalings}
        for rec in getattr(gold, "scaling", []):
            if rec.target_column == config.target_column:
                continue
            if rec.target_column in seen_scaling_columns:
                continue
            if rec.target_column not in pipeline_columns:
                logger.warning(
                    "Skipping gold scaling '%s': column not in pipeline",
                    rec.target_column,
                )
                continue
            seen_scaling_columns.add(rec.target_column)
            config.gold.scalings.append(
                TransformationStep(
                    type=PipelineTransformationType.SCALE,
                    column=rec.target_column,
                    parameters={"method": rec.parameters.get("method", "standard")},
                    rationale=rec.rationale,
                    source_notebook=rec.source_notebook,
                )
            )
        for rec in getattr(gold, "transformations", []):
            if rec.target_column not in pipeline_columns:
                logger.warning(
                    "Skipping gold transformation '%s': column not in pipeline",
                    rec.target_column,
                )
                continue
            step = self._map_gold_transformation(rec)
            if step:
                config.gold.transformations.append(step)
        pipeline_columns |= self._predict_gold_generated_columns(config)
        drop_columns = self._collect_feature_selection_drops(gold, set(), config.target_column, pipeline_columns)
        drop_columns |= {c for c in pipeline_columns if c.endswith("_middle")}
        config.gold.feature_selections = list(drop_columns)

    def _map_gold_transformation(self, rec) -> Optional[TransformationStep]:
        action = rec.action
        type_map = {
            "log": PipelineTransformationType.LOG_TRANSFORM,
            "log_transform": PipelineTransformationType.LOG_TRANSFORM,
            "sqrt": PipelineTransformationType.SQRT_TRANSFORM,
            "sqrt_transform": PipelineTransformationType.SQRT_TRANSFORM,
            "yeo_johnson": PipelineTransformationType.YEO_JOHNSON,
            "zero_inflation_handling": PipelineTransformationType.ZERO_INFLATION_HANDLING,
            "cap_then_log": PipelineTransformationType.CAP_THEN_LOG,
        }
        trans_type = type_map.get(action)
        if trans_type is None:
            return None
        return TransformationStep(
            type=trans_type,
            column=rec.target_column,
            parameters=dict(rec.parameters) if rec.parameters else {},
            rationale=rec.rationale,
            source_notebook=rec.source_notebook,
        )

    @staticmethod
    def _collect_leakage_exclusion_prefixes(
        source_findings: Dict[str, "ExplorationFindings"],
        multi_dataset: "MultiDatasetFindings | None" = None,
    ) -> List[str]:
        prefixes: Set[str] = set()
        for findings in source_findings.values():
            for col in getattr(findings, "excluded_leaking_features", []):
                prefixes.add(f"{col}_")
        if multi_dataset is not None:
            for ds_info in multi_dataset.datasets.values():
                for col in getattr(ds_info, "excluded_leaking_features", []):
                    prefixes.add(f"{col}_")
        return sorted(prefixes)

    @staticmethod
    def find_leakage_excluded_columns(columns, prefixes: List[str]) -> List[str]:
        """Return column names matching leakage exclusion prefixes (including lag/velocity variants).

        Pure matching logic shared by NB08, local and Databricks gold templates.
        """
        if not prefixes:
            return []
        import re
        lag_re = re.compile(r"^(?:lag\d+|velocity)_")

        def matches(col: str) -> bool:
            if any(col.startswith(p) for p in prefixes):
                return True
            m = lag_re.match(col)
            return bool(m and any(col[m.end():].startswith(p) for p in prefixes))

        return [c for c in columns if matches(c)]

    def _collect_prioritized_columns(self, gold) -> Set[str]:
        prioritized = set()
        for rec in getattr(gold, "feature_selection", []):
            if rec.action == "prioritize":
                prioritized.add(rec.target_column)
        return prioritized

    def _collect_feature_selection_drops(self, gold, prioritized: Set[str], target_column: str, pipeline_columns: Optional[Set[str]] = None) -> Set[str]:
        drops = set()
        for rec in getattr(gold, "feature_selection", []):
            if rec.action in ("drop_multicollinear", "drop_weak", "drop_l1_zero", "drop_chi_squared", "drop_lgbm_importance", "drop_availability", "drop_zero_variance"):
                if rec.target_column not in prioritized and rec.target_column != target_column:
                    if pipeline_columns is not None and rec.target_column not in pipeline_columns:
                        logger.warning("Skipping feature selection drop '%s': column not in pipeline", rec.target_column)
                        continue
                    drops.add(rec.target_column)
        return drops

    @staticmethod
    def _resolve_raw_time_column(findings: ExplorationFindings) -> Optional[str]:
        """Get the raw data's time column, preferring datetime_columns over metadata.

        time_series_metadata.time_column may be a post-processing name
        (e.g. feature_timestamp) that doesn't exist in the raw data.
        datetime_columns contains the original column names.
        """
        ts = findings.time_series_metadata
        metadata_col = ts.time_column if ts else None
        if metadata_col and metadata_col in findings.columns:
            return metadata_col
        if findings.datetime_columns:
            return findings.datetime_columns[0]
        return metadata_col

    def _build_timestamp_coalesce_config(self, findings: ExplorationFindings) -> Optional[TimestampCoalesceConfig]:
        if len(findings.datetime_ordering) <= 1:
            return None
        output_col = findings.time_series_metadata.time_column if findings.time_series_metadata else "feature_timestamp"
        return TimestampCoalesceConfig(datetime_columns_ordered=findings.datetime_ordering, output_column=output_col)

    @staticmethod
    def _build_datetime_derivation_config(
        findings: ExplorationFindings,
        reference_column: str,
        mask_future: bool,
    ) -> Optional[DatetimeDerivationConfig]:
        if not findings.datetime_derivation_sources:
            return None
        allow_future = set(getattr(findings, "datetime_allow_future_columns", []))
        if mask_future:
            mask_cols = [c for c in findings.datetime_derivation_sources if c not in allow_future]
        else:
            mask_cols = []
        return DatetimeDerivationConfig(
            source_columns=findings.datetime_derivation_sources,
            reference_column=reference_column,
            mask_future_columns=mask_cols,
        )

    def _build_label_timestamp_config(self, findings: ExplorationFindings) -> Optional[LabelTimestampConfig]:
        observation_days = findings.observation_window_days
        if self._intent is not None:
            observation_days = self._intent.observation_window_days
        if not findings.label_timestamp_column and observation_days == 180:
            return None
        return LabelTimestampConfig(
            label_column=findings.label_timestamp_column,
            fallback_window_days=observation_days,
        )

    def _build_history_window_config(self, time_column: str) -> Optional[HistoryWindowConfig]:
        if self._intent is None:
            return None
        if self._intent.history_upper_limit is None and self._intent.lookback_periods is None:
            return None
        from customer_retention.analysis.auto_explorer.snapshot_grid import CADENCE_DAYS

        return HistoryWindowConfig(
            time_column=time_column,
            upper_limit=self._intent.history_upper_limit,
            lookback_periods=self._intent.lookback_periods,
            cadence_days=CADENCE_DAYS[self._intent.cadence_interval],
        )

    def _build_landing_configs(
        self, config: PipelineConfig, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings]
    ) -> None:
        ctx_key_resolutions: Dict[str, list] = {}
        if self._namespace:
            ctx_path = self._namespace.project_context_path
            if ctx_path.exists():
                from customer_retention.analysis.auto_explorer.project_context import ProjectContext

                ctx = ProjectContext.load(ctx_path)
                for ds_name, ds_entry in ctx.datasets.items():
                    if ds_entry.key_resolution:
                        ctx_key_resolutions[ds_name] = ds_entry.key_resolution

        for event_name in multi.event_datasets:
            dataset_info = multi.datasets.get(event_name)
            if not dataset_info:
                continue
            findings = sources.get(event_name)
            if not findings:
                continue
            entity_col = (
                dataset_info.entity_column
                or (findings.time_series_metadata.entity_column if findings.time_series_metadata else None)
                or (findings.identifier_columns[0] if findings.identifier_columns else "id")
            )
            time_col = (
                dataset_info.time_column
                or (findings.time_series_metadata.time_column if findings.time_series_metadata else None)
                or "timestamp"
            )
            raw_time_col = self._resolve_raw_time_column(findings)
            raw_source = str(Path(
                dataset_info.raw_source_path or dataset_info.source_path or findings.source_path
            ).resolve())
            source_cfg = next((s for s in config.sources if s.name == event_name), None)
            if not source_cfg:
                continue
            original_target = self._resolve_original_target(findings, config.target_column)
            key_steps = [
                KeyResolutionStepConfig(
                    bridge_dataset=s.bridge_dataset,
                    source_key=s.source_key,
                    bridge_key=s.bridge_key,
                    resolve_column=s.resolve_column,
                )
                for s in ctx_key_resolutions.get(event_name, [])
            ]
            config.landing[event_name] = LandingLayerConfig(
                source=source_cfg,
                raw_source_path=raw_source,
                raw_source_format=self._infer_format(raw_source),
                entity_column=entity_col,
                time_column=time_col,
                target_column=config.target_column,
                original_target_column=original_target,
                raw_time_column=raw_time_col if raw_time_col and raw_time_col != time_col else None,
                timestamp_coalesce=self._build_timestamp_coalesce_config(findings),
                label_timestamp=self._build_label_timestamp_config(findings),
                datetime_derivation=self._build_datetime_derivation_config(
                    findings,
                    "feature_timestamp",
                    mask_future=True,
                ),
                history_window=self._build_history_window_config(time_col),
                key_resolution_steps=key_steps,
            )

    @staticmethod
    def _resolve_original_target(findings: ExplorationFindings, target_column: str) -> Optional[str]:
        original = findings.metadata.get("original_target_column") if findings.metadata else None
        if original and original != target_column:
            return original
        return None

    def _build_aggregation_config(
        self, multi: MultiDatasetFindings, findings: ExplorationFindings, dataset_name: str = ""
    ) -> Optional[AggregationWindowConfig]:
        ts = findings.time_series_metadata
        windows = getattr(ts, "aggregation_windows_used", None) or [] if ts else []
        if not windows:
            if not ts:
                return None
            raise ValueError(
                f"No aggregation_windows_used in findings for '{dataset_name}'. "
                "NB01d must run before pipeline generation to record actual aggregation windows."
            )
        target = findings.target_column or ""
        entity_col = (findings.time_series_metadata.entity_column if findings.time_series_metadata else None) or ""
        time_col = (findings.time_series_metadata.time_column if findings.time_series_metadata else None) or ""
        exclude = {target, entity_col, time_col}
        value_columns = []
        categorical_columns = []
        binary_columns = []
        for col_name, col_finding in findings.columns.items():
            if col_name in exclude:
                continue
            col_type = col_finding.inferred_type
            if col_type in self._NUMERIC_TYPES:
                value_columns.append(col_name)
            elif col_type == ColumnType.BINARY:
                binary_columns.append(col_name)
                categorical_columns.append(col_name)
            elif col_type in self._CATEGORICAL_TYPES:
                categorical_columns.append(col_name)
        for src in getattr(findings, "datetime_derivation_sources", []):
            for suffix in ("_delta_hours", "_hour", "_dow", "_is_weekend"):
                derived = f"{src}{suffix}"
                if derived not in value_columns:
                    value_columns.append(derived)

        dataset_info = multi.datasets.get(dataset_name) if dataset_name else None
        exclusions = dataset_info.feature_exclusions if dataset_info else []
        column_blocked_funcs: Dict[str, List[str]] = {}
        for exc in exclusions:
            col = exc.column
            for cat in exc.blocked_categories:
                if cat == "aggregation" and col in value_columns:
                    value_columns.remove(col)
                if cat == "categorical" and col in categorical_columns:
                    categorical_columns.remove(col)
                if cat == "binary" and col in binary_columns:
                    binary_columns.remove(col)
            if exc.blocked_funcs:
                column_blocked_funcs[col] = exc.blocked_funcs

        return AggregationWindowConfig(
            windows=windows,
            value_columns=value_columns,
            agg_funcs=["sum", "mean", "max", "count"],
            categorical_columns=categorical_columns,
            categorical_agg_funcs=["nunique", "mode"],
            binary_columns=binary_columns,
            binary_agg_funcs=["rate", "count", "any"],
            column_blocked_funcs=column_blocked_funcs,
        )

    def _build_lifecycle_config(
        self,
        multi: MultiDatasetFindings,
        findings: Optional[ExplorationFindings] = None,
    ) -> Optional[LifecycleConfig]:
        notes = getattr(multi, "notes", None)
        temporal_config = notes.get("temporal_config", {}) if isinstance(notes, dict) and notes else {}
        feature_groups = temporal_config.get("feature_groups", [])
        recency_edges = self._extract_recency_edges(temporal_config, findings)
        if feature_groups:
            has_regularity = "regularity" in feature_groups
            return LifecycleConfig(
                include_lifecycle_quadrant="lifecycle" in feature_groups,
                include_cyclical_features=has_regularity,
                include_recency_bucket="recency" in feature_groups,
                momentum_pairs=self._build_momentum_pairs(multi),
                include_trend_features="trend" in feature_groups,
                include_cohort_features="cohort" in feature_groups,
                include_month_cyclical=has_regularity,
                include_quarter_cyclical=has_regularity,
                **recency_edges,
            )
        if findings is not None:
            meta = getattr(findings, "metadata", None) or {}
            agg = meta.get("aggregation", {})
            ff = meta.get("temporal_patterns", {}).get("feature_flags", {})
            has_lifecycle = agg.get("include_lifecycle_quadrant", ff.get("include_lifecycle_quadrant", False))
            has_recency = agg.get("include_recency", ff.get("include_recency", False))
            has_cyclical = ff.get("include_seasonality_features", False)
            has_trend = ff.get("include_trend_features", False)
            has_cohort = ff.get("include_cohort_features", False)
            if has_lifecycle or has_recency or has_cyclical or has_trend or has_cohort:
                return LifecycleConfig(
                    include_lifecycle_quadrant=bool(has_lifecycle),
                    include_cyclical_features=bool(has_cyclical),
                    include_recency_bucket=bool(has_recency),
                    momentum_pairs=self._build_momentum_pairs(multi),
                    include_trend_features=bool(has_trend),
                    include_cohort_features=bool(has_cohort),
                    include_month_cyclical=bool(has_cyclical),
                    include_quarter_cyclical=bool(has_cyclical),
                    **recency_edges,
                )
        return None

    @staticmethod
    def _extract_recency_edges(temporal_config: Dict, findings: Optional[ExplorationFindings] = None) -> Dict:
        edges = temporal_config.get("recency_bucket_edges")
        if not edges and findings is not None:
            meta = getattr(findings, "metadata", None) or {}
            edges = meta.get("temporal_patterns", {}).get("recency_analysis", {}).get("bucket_boundaries")
        if not edges:
            return {}
        labels = _edges_to_labels(edges)
        return {"recency_bucket_edges": edges, "recency_bucket_labels": labels}

    def _build_momentum_pairs(self, multi: MultiDatasetFindings) -> List[Dict[str, str]]:
        windows = getattr(multi, "aggregation_windows", [])
        day_values = []
        for w in windows:
            w_lower = str(w).lower().strip()
            if w_lower == "all_time":
                continue
            if w_lower.endswith("d"):
                try:
                    day_values.append((int(w_lower[:-1]), w))
                except ValueError:
                    continue
            elif w_lower.endswith("h"):
                try:
                    hours = int(w_lower[:-1])
                    if hours >= 24:
                        day_values.append((hours // 24, w))
                except ValueError:
                    continue
        day_values.sort(key=lambda x: x[0])
        pairs: List[Dict[str, str]] = []
        for i, (short_days, short_label) in enumerate(day_values):
            for long_days, long_label in day_values[i + 1 :]:
                if long_days >= 3 * short_days:
                    pairs.append({"short_window": short_label, "long_window": long_label})
                    break
        return pairs

    def _build_temporal_feature_config(
        self,
        multi: MultiDatasetFindings,
        findings: Optional[ExplorationFindings] = None,
    ) -> Optional[TemporalFeatureConfig]:
        notes = getattr(multi, "notes", None)
        temporal_config = notes.get("temporal_config", {}) if isinstance(notes, dict) and notes else {}
        lag_windows = temporal_config.get("lag_windows")
        if lag_windows:
            return TemporalFeatureConfig(
                lag_window_days=temporal_config.get("lag_window_days", 30),
                num_lags=temporal_config.get("num_lags", len(lag_windows)),
                lag_columns=temporal_config.get("columns", []),
                lag_agg_funcs=temporal_config.get("lag_agg_funcs", ["sum", "mean", "count", "max"]),
                feature_groups=temporal_config.get("temporal_feature_groups", TemporalFeatureConfig().feature_groups),
            )
        if findings is not None:
            meta = getattr(findings, "metadata", None) or {}
            tp = meta.get("temporal_patterns", {})
            if tp.get("lag_features_computed"):
                return TemporalFeatureConfig(
                    lag_window_days=tp.get("lag_window_days", 30),
                    num_lags=tp.get("num_lags", 4),
                    lag_columns=tp.get("lag_columns", []),
                    lag_agg_funcs=tp.get("lag_agg_funcs", ["sum", "mean", "count", "max"]),
                )
        return None

    @staticmethod
    def _build_text_feature_configs(
        findings: Optional[ExplorationFindings],
    ) -> List[TextFeatureConfig]:
        if findings is None:
            return []
        text_processing = getattr(findings, "text_processing", None)
        if not text_processing:
            return []
        configs = []
        for col_name, meta in text_processing.items():
            model = (
                getattr(meta, "model", "all-MiniLM-L6-v2")
                if hasattr(meta, "model")
                else meta.get("model", "all-MiniLM-L6-v2")
            )
            n_comp = getattr(meta, "n_components", 5) if hasattr(meta, "n_components") else meta.get("n_components", 5)
            comp_cols = (
                getattr(meta, "component_columns", [])
                if hasattr(meta, "component_columns")
                else meta.get("component_columns", [])
            )
            configs.append(
                TextFeatureConfig(
                    column=col_name,
                    embedding_model=model,
                    n_components=n_comp,
                    component_columns=list(comp_cols),
                )
            )
        return configs

    def _build_training_config(
        self,
        multi: MultiDatasetFindings,
        source_findings: Dict[str, ExplorationFindings],
    ) -> Optional[TrainingConfig]:
        split_strategy = "temporal"
        temporal_column = None
        purge_gap_days = None
        recommended_start = None
        filter_future = False
        imbalance_strategy = "class_weight"

        if self._intent is not None:
            from customer_retention.analysis.auto_explorer.project_context import SplitStrategy

            if self._intent.split_strategy == SplitStrategy.TEMPORAL:
                split_strategy = "temporal"
            elif self._intent.split_strategy == SplitStrategy.COHORT_BASED:
                split_strategy = "cohort_based"
            purge_gap_days = getattr(self._intent, "purge_gap_days", None)

        for event_name in multi.event_datasets:
            findings = source_findings.get(event_name)
            if findings and findings.time_series_metadata:
                temporal_column = findings.time_series_metadata.time_column
                ts_meta = getattr(findings, "metadata", None) or {}
                start = ts_meta.get("time_series", {}).get("recommended_training_start")
                if start:
                    recommended_start = start
                tq = ts_meta.get("temporal_quality", {})
                checks = tq.get("checks", [])
                for check in checks:
                    code = check.get("code", "") if isinstance(check, dict) else getattr(check, "code", "")
                    status = check.get("status", "") if isinstance(check, dict) else getattr(check, "status", "")
                    if code == "TQ003" and status in ("fail", "warning"):
                        filter_future = True
                break

        if not temporal_column and not filter_future and not recommended_start and self._intent is None:
            return None

        exploration_profile = self._load_exploration_feature_profile()
        best_model_type = self._read_best_model_type()

        return TrainingConfig(
            split_strategy=split_strategy,
            temporal_column=temporal_column,
            purge_gap_days=purge_gap_days,
            recommended_training_start=recommended_start,
            filter_future_dates=filter_future,
            imbalance_strategy=imbalance_strategy,
            exploration_feature_profile=exploration_profile,
            best_model_type=best_model_type,
        )

    def _load_exploration_feature_profile(self) -> Optional[Dict[str, Any]]:
        if getattr(self, "_namespace", None) is None:
            return None
        profile_path = self._namespace.exploration_feature_profile_path
        if not profile_path.exists():
            return None
        try:
            from customer_retention.stages.modeling.feature_profile import FeatureProfile
            profile = FeatureProfile.load(profile_path)
            return profile.to_dict() if profile else None
        except Exception:
            logger.warning("Could not load exploration feature profile from %s", profile_path)
            return None

    def _read_best_model_type(self) -> Optional[str]:
        if getattr(self, "_namespace", None) is None:
            return None
        diag_path = self._namespace.exploration_diagnostics_path
        if not diag_path.exists():
            return None
        try:
            import json as _json

            diag_data = _json.loads(diag_path.read_text())
            exploration_best = diag_data.get("best_model_name")
            if exploration_best:
                return _EXPLORATION_TO_PRODUCTION_MODEL.get(
                    exploration_best, exploration_best.lower().replace(" ", "_")
                )
            return None
        except Exception:
            logger.warning("Could not read best model from %s", diag_path)
            return None

    def _build_bronze_event_configs(
        self,
        config: PipelineConfig,
        multi: MultiDatasetFindings,
        source_findings: Dict[str, ExplorationFindings],
        discovered_events: Optional[Dict[str, ExplorationFindings]] = None,
    ) -> None:
        for event_name in multi.event_datasets:
            findings = source_findings.get(event_name)
            if not findings:
                continue
            source_cfg = next((s for s in config.sources if s.name == event_name), None)
            if not source_cfg:
                continue
            dataset_info = multi.datasets.get(event_name)
            entity_col = (dataset_info.entity_column if dataset_info else None) or source_cfg.entity_key
            time_col = (dataset_info.time_column if dataset_info else None) or source_cfg.time_column or "timestamp"
            raw_time_col = self._resolve_raw_time_column(findings)
            config.bronze_event[event_name] = BronzeEventConfig(
                source=source_cfg,
                entity_column=entity_col,
                time_column=time_col,
                deduplicate=True,
                pre_shaping=self._extract_transformations(findings),
                aggregation=self._build_aggregation_config(multi, findings, event_name),
                lifecycle=self._build_lifecycle_config(multi, findings),
                raw_time_column=raw_time_col if raw_time_col and raw_time_col != time_col else None,
                datetime_derivation=self._build_datetime_derivation_config(
                    findings,
                    time_col,
                    mask_future=False,
                ),
                temporal_features=self._build_temporal_feature_config(multi, findings),
                text_features=self._build_text_feature_configs(findings),
            )
        for agg_name, preagg in (discovered_events or {}).items():
            if agg_name in config.bronze_event:
                continue
            source_cfg = next((s for s in config.sources if s.name == agg_name), None)
            if not source_cfg:
                continue
            ts = preagg.time_series_metadata
            entity_col = (ts.entity_column if ts else None) or source_cfg.entity_key
            time_col = (ts.time_column if ts else None) or source_cfg.time_column or "timestamp"
            raw_time_col = self._resolve_raw_time_column(preagg)
            config.bronze_event[agg_name] = BronzeEventConfig(
                source=source_cfg,
                entity_column=entity_col,
                time_column=time_col,
                deduplicate=True,
                pre_shaping=self._extract_transformations(preagg),
                aggregation=self._build_aggregation_config(multi, preagg, agg_name),
                lifecycle=self._build_lifecycle_config(multi, preagg),
                raw_time_column=raw_time_col if raw_time_col and raw_time_col != time_col else None,
                datetime_derivation=self._build_datetime_derivation_config(
                    preagg,
                    time_col,
                    mask_future=False,
                ),
                temporal_features=self._build_temporal_feature_config(multi, preagg),
                text_features=self._build_text_feature_configs(preagg),
            )

    def _discover_event_sources(
        self, source_findings: Dict[str, ExplorationFindings]
    ) -> Dict[str, ExplorationFindings]:
        index = self._build_aggregated_path_index()
        if not index:
            return {}
        return self._scan_for_preagg_findings(index)

    def _build_aggregated_path_index(self) -> Dict[Path, str]:
        return {path: name for name, path in self._source_findings_paths.items()}

    def _scan_for_preagg_findings(self, index: Dict[Path, str]) -> Dict[str, ExplorationFindings]:
        loaded_paths = set(self._source_findings_paths.values())
        result: Dict[str, ExplorationFindings] = {}
        if self._namespace is not None:
            candidates = self._namespace.discover_all_findings(prefer_aggregated=False)
        else:
            candidates = list(self._findings_dir.glob("*_findings.yaml"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in loaded_paths:
                continue
            if candidate.name == "multi_dataset_findings.yaml":
                continue
            try:
                preagg = ExplorationFindings.load(str(candidate))
            except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping unloadable findings %s: %s", candidate, exc)
                continue
            source_name = self._match_preagg_to_source(preagg, index)
            if source_name is not None:
                result[source_name] = preagg
        return result

    def _match_preagg_to_source(self, preagg: ExplorationFindings, index: Dict[Path, str]) -> Optional[str]:
        if not preagg.has_aggregated_output:
            return None
        agg_path_str = preagg.time_series_metadata.aggregated_findings_path
        if not agg_path_str:
            return None
        agg_path = Path(agg_path_str).resolve()
        return index.get(agg_path)

    def _build_discovered_landing_configs(
        self,
        config: PipelineConfig,
        discovered: Dict[str, ExplorationFindings],
        multi: MultiDatasetFindings,
    ) -> None:
        from customer_retention.core.config.column_config import DatasetGranularity

        for agg_name, preagg in discovered.items():
            if agg_name in config.landing:
                continue
            source_cfg = next((s for s in config.sources if s.name == agg_name), None)
            if not source_cfg:
                continue
            ts = preagg.time_series_metadata
            entity_col = (ts.entity_column if ts else None) or source_cfg.entity_key
            time_col = (ts.time_column if ts else None) or "timestamp"
            raw_time_col = self._resolve_raw_time_column(preagg)
            source_cfg.is_event_level = True
            source_cfg.time_column = time_col
            source_cfg.entity_key = entity_col
            for ms in config.silver.merge_sources:
                if ms.name == agg_name:
                    ms.granularity = DatasetGranularity.EVENT_LEVEL.value
                    break
            raw_source = str(Path(preagg.source_path).resolve())
            original_target = self._resolve_original_target(preagg, config.target_column)
            config.landing[agg_name] = LandingLayerConfig(
                source=source_cfg,
                raw_source_path=raw_source,
                raw_source_format=self._infer_format(raw_source),
                entity_column=entity_col,
                time_column=time_col,
                target_column=config.target_column,
                original_target_column=original_target,
                raw_time_column=raw_time_col if raw_time_col and raw_time_col != time_col else None,
                timestamp_coalesce=self._build_timestamp_coalesce_config(preagg),
                label_timestamp=self._build_label_timestamp_config(preagg),
                datetime_derivation=self._build_datetime_derivation_config(
                    preagg,
                    "feature_timestamp",
                    mask_future=True,
                ),
                history_window=self._build_history_window_config(time_col),
            )

    @staticmethod
    def _reconcile_discovered_event_transforms(
        config: "PipelineConfig", discovered_events: Dict[str, ExplorationFindings]
    ) -> None:
        if not discovered_events:
            return
        for name in list(discovered_events.keys()):
            if name in config.bronze and name in config.bronze_event:
                config.bronze_event[name].post_shaping.extend(config.bronze[name].transformations)
                del config.bronze[name]

    def _reconcile_gold_columns(self, config: "PipelineConfig") -> None:
        if config.gold is None:
            return
        pipeline_columns = self._collect_pipeline_columns(config)
        for step in config.silver.derived_columns:
            pipeline_columns.add(step.column)
        pipeline_columns |= self._predict_gold_generated_columns(config)
        config.gold.scalings = self._filter_gold_steps(config.gold.scalings, pipeline_columns, "scaling")
        config.gold.encodings = self._filter_gold_steps(config.gold.encodings, pipeline_columns, "encoding")
        config.gold.transformations = self._filter_gold_steps(
            config.gold.transformations, pipeline_columns, "transformation",
        )
        before = len(config.gold.feature_selections)
        config.gold.feature_selections = [c for c in config.gold.feature_selections if c in pipeline_columns]
        removed = before - len(config.gold.feature_selections)
        if removed:
            logger.warning("Removed %d feature selection drop(s): column(s) not in pipeline", removed)

    @staticmethod
    def _filter_gold_steps(steps, pipeline_columns, label):
        kept = []
        for step in steps:
            if step.column in pipeline_columns:
                kept.append(step)
            else:
                logger.warning("Removing gold %s '%s': column not in pipeline", label, step.column)
        return kept

    @staticmethod
    def _infer_format(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return "csv"
        if ext in (".parquet", ".pq"):
            return "parquet"
        return "delta"
