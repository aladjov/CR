from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from customer_retention.analysis.auto_explorer.exploration_manager import (
    DatasetInfo,
    DatasetRelationshipInfo,
    MultiDatasetFindings,
)
from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry

from .models import (
    AggregationWindowConfig,
    BronzeEventConfig,
    BronzeLayerConfig,
    GoldLayerConfig,
    LabelTimestampConfig,
    LandingLayerConfig,
    LifecycleConfig,
    PipelineConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TimestampCoalesceConfig,
    TransformationStep,
)


def _resolve_col_type(col_finding) -> str:
    col_type = col_finding.inferred_type
    if hasattr(col_type, 'value'):
        col_type = col_type.value
    return col_type


class FindingsParser:
    def __init__(self, findings_dir: str):
        self._findings_dir = Path(findings_dir)
        self._source_findings_paths: Dict[str, Path] = {}

    def parse(self) -> PipelineConfig:
        multi_dataset = self._load_multi_dataset_findings()
        selected_sources = list(multi_dataset.datasets.keys())
        source_findings = self._load_source_findings(selected_sources, self._findings_dir, multi_dataset)
        discovered_events = self._discover_event_sources(source_findings)
        recommendations_registry = self._load_recommendations()
        recommendations_hash = recommendations_registry.compute_recommendations_hash() if recommendations_registry else None
        config = self._build_pipeline_config(multi_dataset, source_findings, recommendations_hash)
        if recommendations_registry:
            self._apply_recommendations_to_config(config, recommendations_registry, multi_dataset)
        self._build_landing_configs(config, multi_dataset, source_findings)
        self._build_discovered_landing_configs(config, discovered_events, multi_dataset)
        self._build_bronze_event_configs(config, multi_dataset, source_findings, discovered_events)
        self._reconcile_discovered_event_transforms(config, discovered_events)
        return config

    def _load_recommendations(self) -> Optional[RecommendationRegistry]:
        recommendations_path = None
        pattern_matches = list(self._findings_dir.glob("*_recommendations.yaml"))
        if pattern_matches:
            recommendations_path = max(pattern_matches, key=lambda p: p.stat().st_mtime)
        elif (self._findings_dir / "recommendations.yaml").exists():
            recommendations_path = self._findings_dir / "recommendations.yaml"
        if recommendations_path and recommendations_path.exists():
            with open(recommendations_path) as f:
                return RecommendationRegistry.from_dict(yaml.safe_load(f))
        return None

    def _load_multi_dataset_findings(self) -> MultiDatasetFindings:
        path = self._findings_dir / "multi_dataset_findings.yaml"
        if not path.exists():
            return self._synthesize_from_single_source()
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._dict_to_multi_dataset_findings(data)

    def _synthesize_from_single_source(self) -> MultiDatasetFindings:
        from customer_retention.core.config.column_config import DatasetGranularity

        candidates = [
            p for p in self._findings_dir.glob("*_findings.yaml")
            if p.name != "multi_dataset_findings.yaml"
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No findings files found in {self._findings_dir}"
            )

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
                entity_column=(
                    findings.identifier_columns[0]
                    if findings.identifier_columns
                    else None
                ),
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
                excluded=info.get("excluded", False)
            )
        relationships = [
            DatasetRelationshipInfo(
                left_dataset=r["left_dataset"],
                right_dataset=r["right_dataset"],
                left_column=r["left_column"],
                right_column=r["right_column"],
                relationship_type=r.get("relationship_type", "one_to_many"),
                confidence=r.get("confidence", 1.0),
                auto_detected=r.get("auto_detected", False)
            )
            for r in data.get("relationships", [])
        ]
        return MultiDatasetFindings(
            datasets=datasets,
            relationships=relationships,
            primary_entity_dataset=self._strip_aggregated(data.get("primary_entity_dataset", "") or ""),
            event_datasets=[self._strip_aggregated(e) for e in data.get("event_datasets", [])],
            excluded_datasets=data.get("excluded_datasets", []),
            aggregation_windows=data.get("aggregation_windows", ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"]),
            notes=data.get("notes", {}),
        )

    def _load_source_findings(self, sources: List[str], findings_dir: Path, multi_dataset: MultiDatasetFindings = None) -> Dict[str, ExplorationFindings]:
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

    def _build_pipeline_config(self, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings], recommendations_hash: Optional[str] = None) -> PipelineConfig:
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

    def _build_source_configs(self, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings]) -> List[SourceConfig]:
        result = []
        for name, findings in sources.items():
            dataset_info = multi.datasets.get(name)
            is_event = name in multi.event_datasets
            is_excluded = name in multi.excluded_datasets or (dataset_info and dataset_info.excluded)
            raw_source = str(Path(dataset_info.source_path if dataset_info else findings.source_path).resolve())
            time_col = None
            entity_key = findings.identifier_columns[0] if findings.identifier_columns else "id"
            if is_event and findings.time_series_metadata:
                time_col = findings.time_series_metadata.time_column
                if findings.time_series_metadata.entity_column:
                    entity_key = findings.time_series_metadata.entity_column
            result.append(SourceConfig(
                name=name,
                path=Path(findings.source_path).name,
                format=findings.source_format,
                entity_key=entity_key,
                raw_source_path=raw_source,
                time_column=time_col,
                is_event_level=is_event,
                excluded=is_excluded
            ))
        return result

    def _build_bronze_configs(self, sources: Dict[str, ExplorationFindings], source_configs: List[SourceConfig]) -> Dict[str, BronzeLayerConfig]:
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
                rationale=f"Impute nulls in {column}"
            )
        if action == "cap_outlier":
            return TransformationStep(
                type=PipelineTransformationType.CAP_OUTLIER,
                column=column,
                parameters={"method": param if param else "iqr"},
                rationale=f"Cap outliers in {column}"
            )
        return None

    def _build_silver_config(self, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings]) -> SilverLayerConfig:
        joins = []
        for rel in multi.relationships:
            joins.append({
                "left_key": rel.left_column,
                "right_key": rel.right_column,
                "right_source": rel.right_dataset,
                "how": "left"
            })
        return SilverLayerConfig(joins=joins, aggregations=[])

    def _build_gold_config(self, sources: Dict[str, ExplorationFindings]) -> GoldLayerConfig:
        encodings = []
        scalings = []
        for findings in sources.values():
            for col_name, col_finding in findings.columns.items():
                col_type = _resolve_col_type(col_finding)
                if col_type == "categorical":
                    encodings.append(TransformationStep(
                        type=PipelineTransformationType.ENCODE,
                        column=col_name,
                        parameters={"method": "one_hot"},
                        rationale=f"One-hot encode {col_name}"
                    ))
                elif col_type == "numeric":
                    scalings.append(TransformationStep(
                        type=PipelineTransformationType.SCALE,
                        column=col_name,
                        parameters={"method": "standard"},
                        rationale=f"Standardize {col_name}"
                    ))
        return GoldLayerConfig(encodings=encodings, scalings=scalings)

    def _find_target_column(self, sources: Dict[str, ExplorationFindings]) -> str:
        for findings in sources.values():
            if findings.target_column:
                return findings.target_column
        return "target"

    def _apply_recommendations_to_config(self, config: PipelineConfig, registry: RecommendationRegistry, multi: MultiDatasetFindings) -> None:
        self._apply_bronze_recommendations(config, registry)
        self._apply_silver_recommendations(config, registry)
        self._apply_gold_recommendations(config, registry)

    def _apply_bronze_recommendations(self, config: PipelineConfig, registry: RecommendationRegistry) -> None:
        sources_to_process = dict(registry.sources)
        if not sources_to_process and hasattr(registry, 'bronze') and registry.bronze is not None:
            sources_to_process = {"_default": registry.bronze}
        for source_name, bronze_recs in sources_to_process.items():
            target_bronze = self._find_bronze_config_for_source(config, source_name, bronze_recs.source_file)
            if target_bronze is None:
                continue
            for rec in bronze_recs.null_handling:
                step = self._map_bronze_null(rec)
                if step:
                    target_bronze.transformations.append(step)
            for rec in bronze_recs.outlier_handling:
                step = self._map_bronze_outlier(rec)
                if step:
                    target_bronze.transformations.append(step)
            target_bronze.transformations = self._deduplicate_steps(target_bronze.transformations)

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

    def _find_bronze_config_for_source(self, config: PipelineConfig, source_name: str, source_file: str) -> Optional[BronzeLayerConfig]:
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
        if not hasattr(registry, 'silver') or registry.silver is None:
            return
        for rec in getattr(registry.silver, 'derived_columns', []):
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

    def _apply_gold_recommendations(self, config: PipelineConfig, registry: RecommendationRegistry) -> None:
        if not hasattr(registry, 'gold') or registry.gold is None:
            return
        gold = registry.gold
        seen_encoding_columns: Set[str] = {e.column for e in config.gold.encodings}
        for rec in getattr(gold, 'encoding', []):
            if rec.target_column in seen_encoding_columns:
                continue
            seen_encoding_columns.add(rec.target_column)
            method = rec.parameters.get("method", rec.action)
            if method in ("onehot", "one_hot"):
                method = "one_hot"
            config.gold.encodings.append(TransformationStep(
                type=PipelineTransformationType.ENCODE,
                column=rec.target_column,
                parameters={"method": method},
                rationale=rec.rationale,
                source_notebook=rec.source_notebook,
            ))
        seen_scaling_columns: Set[str] = {s.column for s in config.gold.scalings}
        for rec in getattr(gold, 'scaling', []):
            if rec.target_column in seen_scaling_columns:
                continue
            seen_scaling_columns.add(rec.target_column)
            config.gold.scalings.append(TransformationStep(
                type=PipelineTransformationType.SCALE,
                column=rec.target_column,
                parameters={"method": rec.parameters.get("method", "standard")},
                rationale=rec.rationale,
                source_notebook=rec.source_notebook,
            ))
        for rec in getattr(gold, 'transformations', []):
            step = self._map_gold_transformation(rec)
            if step:
                config.gold.transformations.append(step)
        prioritized_columns = self._collect_prioritized_columns(gold)
        drop_columns = self._collect_feature_selection_drops(gold, prioritized_columns)
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

    def _collect_prioritized_columns(self, gold) -> Set[str]:
        prioritized = set()
        for rec in getattr(gold, 'feature_selection', []):
            if rec.action == "prioritize":
                prioritized.add(rec.target_column)
        return prioritized

    def _collect_feature_selection_drops(self, gold, prioritized: Set[str]) -> Set[str]:
        drops = set()
        for rec in getattr(gold, 'feature_selection', []):
            if rec.action in ("drop_multicollinear", "drop_weak"):
                if rec.target_column not in prioritized:
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

    def _build_label_timestamp_config(self, findings: ExplorationFindings) -> Optional[LabelTimestampConfig]:
        if not findings.label_timestamp_column and findings.observation_window_days == 180:
            return None
        return LabelTimestampConfig(
            label_column=findings.label_timestamp_column,
            fallback_window_days=findings.observation_window_days,
        )

    def _build_landing_configs(self, config: PipelineConfig, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings]) -> None:
        for event_name in multi.event_datasets:
            dataset_info = multi.datasets.get(event_name)
            if not dataset_info:
                continue
            findings = sources.get(event_name)
            if not findings:
                continue
            entity_col = (dataset_info.entity_column
                         or (findings.time_series_metadata.entity_column if findings.time_series_metadata else None)
                         or (findings.identifier_columns[0] if findings.identifier_columns else "id"))
            time_col = (dataset_info.time_column
                       or (findings.time_series_metadata.time_column if findings.time_series_metadata else None)
                       or "timestamp")
            raw_time_col = self._resolve_raw_time_column(findings)
            raw_source = str(Path(dataset_info.source_path or findings.source_path).resolve())
            source_cfg = next((s for s in config.sources if s.name == event_name), None)
            if not source_cfg:
                continue
            original_target = self._resolve_original_target(findings, config.target_column)
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
            )

    @staticmethod
    def _resolve_original_target(findings: ExplorationFindings, target_column: str) -> Optional[str]:
        original = findings.metadata.get("original_target_column") if findings.metadata else None
        if original and original != target_column:
            return original
        return None

    def _build_aggregation_config(self, multi: MultiDatasetFindings, findings: ExplorationFindings) -> Optional[AggregationWindowConfig]:
        windows = getattr(multi, 'aggregation_windows', None) or []
        if not windows and findings.time_series_metadata:
            windows = getattr(findings.time_series_metadata, 'suggested_aggregations', []) or []
        if not windows:
            return None
        value_columns = []
        for col_name, col_finding in findings.columns.items():
            col_type = _resolve_col_type(col_finding)
            if col_type in ("numeric_continuous", "numeric_discrete", "numeric", "binary"):
                if col_name not in (findings.target_column or ""):
                    value_columns.append(col_name)
        return AggregationWindowConfig(
            windows=windows,
            value_columns=value_columns,
            agg_funcs=["sum", "mean", "max", "count"],
        )

    def _build_lifecycle_config(self, multi: MultiDatasetFindings) -> Optional[LifecycleConfig]:
        notes = getattr(multi, 'notes', None)
        if not notes:
            return None
        temporal_config = notes.get("temporal_config", {}) if isinstance(notes, dict) else {}
        feature_groups = temporal_config.get("feature_groups", [])
        return LifecycleConfig(
            include_lifecycle_quadrant="lifecycle" in feature_groups,
            include_cyclical_features="regularity" in feature_groups,
            include_recency_bucket="recency" in feature_groups,
            momentum_pairs=[],
        )

    def _build_bronze_event_configs(
        self,
        config: PipelineConfig,
        multi: MultiDatasetFindings,
        source_findings: Dict[str, ExplorationFindings],
        discovered_events: Optional[Dict[str, ExplorationFindings]] = None,
    ) -> None:
        lifecycle_config = self._build_lifecycle_config(multi)
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
                source=source_cfg, entity_column=entity_col, time_column=time_col,
                deduplicate=True,
                pre_shaping=self._extract_transformations(findings),
                aggregation=self._build_aggregation_config(multi, findings),
                lifecycle=lifecycle_config,
                raw_time_column=raw_time_col if raw_time_col and raw_time_col != time_col else None,
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
                source=source_cfg, entity_column=entity_col, time_column=time_col,
                deduplicate=True,
                pre_shaping=self._extract_transformations(preagg),
                aggregation=self._build_aggregation_config(multi, preagg),
                lifecycle=lifecycle_config,
                raw_time_column=raw_time_col if raw_time_col and raw_time_col != time_col else None,
            )

    def _discover_event_sources(self, source_findings: Dict[str, ExplorationFindings]) -> Dict[str, ExplorationFindings]:
        index = self._build_aggregated_path_index()
        if not index:
            return {}
        return self._scan_for_preagg_findings(index)

    def _build_aggregated_path_index(self) -> Dict[Path, str]:
        return {path: name for name, path in self._source_findings_paths.items()}

    def _scan_for_preagg_findings(self, index: Dict[Path, str]) -> Dict[str, ExplorationFindings]:
        loaded_paths = set(self._source_findings_paths.values())
        result: Dict[str, ExplorationFindings] = {}
        for candidate in self._findings_dir.glob("*_findings.yaml"):
            resolved = candidate.resolve()
            if resolved in loaded_paths:
                continue
            if candidate.name == "multi_dataset_findings.yaml":
                continue
            try:
                preagg = ExplorationFindings.load(str(candidate))
            except Exception:
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
            )

    @staticmethod
    def _reconcile_discovered_event_transforms(config: "PipelineConfig", discovered_events: Dict[str, ExplorationFindings]) -> None:
        if not discovered_events:
            return
        for name in list(discovered_events.keys()):
            if name in config.bronze and name in config.bronze_event:
                config.bronze_event[name].post_shaping.extend(config.bronze[name].transformations)
                del config.bronze[name]

    @staticmethod
    def _infer_format(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return "csv"
        return "delta"
