from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.analysis.auto_explorer.exploration_manager import MultiDatasetFindings, DatasetInfo, DatasetRelationshipInfo
from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry
from .models import (
    PipelineConfig, SourceConfig, BronzeLayerConfig, SilverLayerConfig,
    GoldLayerConfig, TransformationStep, PipelineTransformationType
)


class FindingsParser:
    def __init__(self, findings_dir: str):
        self._findings_dir = Path(findings_dir)

    def parse(self) -> PipelineConfig:
        multi_dataset = self._load_multi_dataset_findings()
        selected_sources = list(multi_dataset.datasets.keys())
        source_findings = self._load_source_findings(selected_sources, self._findings_dir, multi_dataset)
        recommendations_registry = self._load_recommendations()
        recommendations_hash = recommendations_registry.compute_recommendations_hash() if recommendations_registry else None
        return self._build_pipeline_config(multi_dataset, source_findings, recommendations_hash)

    def _load_recommendations(self) -> Optional[RecommendationRegistry]:
        # Prefer pattern-matched files (more specific) over generic recommendations.yaml
        recommendations_path = None
        pattern_matches = list(self._findings_dir.glob("*_recommendations.yaml"))
        if pattern_matches:
            # Use the most recently modified pattern-matched file
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
            raise FileNotFoundError(f"Multi-dataset findings not found at {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._dict_to_multi_dataset_findings(data)

    def _dict_to_multi_dataset_findings(self, data: Dict) -> MultiDatasetFindings:
        from customer_retention.core.config.column_config import DatasetGranularity
        datasets = {}
        for name, info in data.get("datasets", {}).items():
            granularity_str = info.get("granularity", "unknown")
            granularity = DatasetGranularity(granularity_str) if granularity_str else DatasetGranularity.UNKNOWN
            datasets[name] = DatasetInfo(
                name=info["name"],
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
            primary_entity_dataset=data.get("primary_entity_dataset"),
            event_datasets=data.get("event_datasets", []),
            excluded_datasets=data.get("excluded_datasets", [])
        )

    def _load_source_findings(self, sources: List[str], findings_dir: Path, multi_dataset: MultiDatasetFindings = None) -> Dict[str, ExplorationFindings]:
        result = {}
        for name in sources:
            path = None
            if multi_dataset and name in multi_dataset.datasets:
                dataset_info = multi_dataset.datasets[name]
                if dataset_info.findings_path:
                    path = findings_dir / dataset_info.findings_path
                    if not path.exists():
                        path = Path(dataset_info.findings_path)
            if path is None or not path.exists():
                candidates = list(findings_dir.glob(f"{name}_*_findings.yaml"))
                if candidates:
                    path = candidates[0]
                else:
                    path = findings_dir / f"{name}_findings.yaml"
            if path.exists():
                result[name] = ExplorationFindings.load(str(path))
        return result

    def _build_pipeline_config(self, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings],
                                recommendations_hash: Optional[str] = None) -> PipelineConfig:
        source_configs = self._build_source_configs(multi, sources)
        bronze_configs = self._build_bronze_configs(sources, source_configs)
        silver_config = self._build_silver_config(multi, sources)
        gold_config = self._build_gold_config(sources)
        target_column = self._find_target_column(sources)
        return PipelineConfig(
            name="",
            target_column=target_column,
            sources=source_configs,
            bronze=bronze_configs,
            silver=silver_config,
            gold=gold_config,
            output_dir="",
            recommendations_hash=recommendations_hash
        )

    def _build_source_configs(self, multi: MultiDatasetFindings, sources: Dict[str, ExplorationFindings]) -> List[SourceConfig]:
        result = []
        for name, findings in sources.items():
            dataset_info = multi.datasets.get(name)
            is_event = name in multi.event_datasets
            time_col = None
            entity_key = findings.identifier_columns[0] if findings.identifier_columns else "id"
            if is_event and findings.time_series_metadata:
                time_col = findings.time_series_metadata.time_column
                if findings.time_series_metadata.entity_column:
                    entity_key = findings.time_series_metadata.entity_column
            result.append(SourceConfig(
                name=name,
                path=findings.source_path,
                format=findings.source_format,
                entity_key=entity_key,
                time_column=time_col,
                is_event_level=is_event
            ))
        return result

    def _build_bronze_configs(self, sources: Dict[str, ExplorationFindings], source_configs: List[SourceConfig]) -> Dict[str, BronzeLayerConfig]:
        result = {}
        source_map = {s.name: s for s in source_configs}
        for name, findings in sources.items():
            transformations = self._extract_transformations(findings)
            result[name] = BronzeLayerConfig(source=source_map[name], transformations=transformations)
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
        aggregations = []
        return SilverLayerConfig(joins=joins, aggregations=aggregations)

    def _build_gold_config(self, sources: Dict[str, ExplorationFindings]) -> GoldLayerConfig:
        encodings = []
        scalings = []
        for findings in sources.values():
            for col_name, col_finding in findings.columns.items():
                col_type = col_finding.inferred_type
                if hasattr(col_type, 'value'):
                    col_type = col_type.value
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
