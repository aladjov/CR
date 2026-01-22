from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import yaml

from customer_retention.core.config.column_config import ColumnType


@dataclass
class SourceSpec:
    name: str
    path: str
    format: str
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ColumnSpec:
    name: str
    data_type: str
    semantic_type: str
    nullable: bool = True
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_column_finding(cls, finding) -> "ColumnSpec":
        type_to_dtype = {
            ColumnType.IDENTIFIER: "string",
            ColumnType.TARGET: "integer",
            ColumnType.BINARY: "integer",
            ColumnType.NUMERIC_CONTINUOUS: "float",
            ColumnType.NUMERIC_DISCRETE: "integer",
            ColumnType.CATEGORICAL_NOMINAL: "string",
            ColumnType.CATEGORICAL_ORDINAL: "string",
            ColumnType.CATEGORICAL_CYCLICAL: "string",
            ColumnType.DATETIME: "timestamp",
            ColumnType.TEXT: "string"
        }
        return cls(
            name=finding.name,
            data_type=type_to_dtype.get(finding.inferred_type, "string"),
            semantic_type=finding.inferred_type.value,
            nullable=finding.universal_metrics.get("null_count", 0) > 0
        )


@dataclass
class SchemaSpec:
    columns: List[ColumnSpec]
    primary_key: Optional[str] = None
    partition_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "columns": [c.to_dict() for c in self.columns],
            "primary_key": self.primary_key,
            "partition_columns": self.partition_columns
        }

    @classmethod
    def from_findings(cls, findings) -> "SchemaSpec":
        columns = [ColumnSpec.from_column_finding(col) for col in findings.columns.values()]
        primary_key = findings.identifier_columns[0] if findings.identifier_columns else None
        return cls(columns=columns, primary_key=primary_key)


@dataclass
class TransformSpec:
    name: str
    transform_type: str
    input_columns: List[str]
    output_columns: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureSpec:
    name: str
    source_columns: List[str]
    computation: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelSpec:
    name: str
    model_type: str
    target_column: str
    feature_columns: List[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=lambda: ["auc", "precision", "recall", "f1"])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityGateSpec:
    name: str
    gate_type: str
    column: str
    threshold: float
    action: str = "fail"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineSpec:
    name: str = "pipeline"
    version: str = "1.0.0"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    sources: List[SourceSpec] = field(default_factory=list)
    schema: Optional[SchemaSpec] = None
    bronze_transforms: List[TransformSpec] = field(default_factory=list)
    silver_transforms: List[TransformSpec] = field(default_factory=list)
    gold_transforms: List[TransformSpec] = field(default_factory=list)
    feature_definitions: List[FeatureSpec] = field(default_factory=list)
    model_config: Optional[ModelSpec] = None
    quality_gates: List[QualityGateSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_findings(cls, findings, name: str = None) -> "PipelineSpec":
        spec = cls(
            name=name or Path(findings.source_path).stem + "_pipeline",
            description=f"Pipeline generated from {findings.source_path}"
        )
        spec.sources.append(SourceSpec(
            name="primary_source",
            path=findings.source_path,
            format=findings.source_format
        ))
        spec.schema = SchemaSpec.from_findings(findings)
        spec._add_default_transforms(findings)
        spec._add_default_features(findings)
        spec._add_default_model(findings)
        spec._add_default_quality_gates(findings)
        return spec

    def _add_default_transforms(self, findings):
        for name, col in findings.columns.items():
            if col.inferred_type == ColumnType.IDENTIFIER:
                continue
            if col.inferred_type == ColumnType.TARGET:
                continue
            if col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
                self.silver_transforms.append(TransformSpec(
                    name=f"scale_{name}",
                    transform_type="standard_scaling",
                    input_columns=[name],
                    output_columns=[f"{name}_scaled"]
                ))
            elif col.inferred_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL]:
                self.silver_transforms.append(TransformSpec(
                    name=f"encode_{name}",
                    transform_type="one_hot_encoding",
                    input_columns=[name],
                    output_columns=[f"{name}_encoded"]
                ))

    def _add_default_features(self, findings):
        for name in findings.datetime_columns:
            self.feature_definitions.append(FeatureSpec(
                name=f"days_since_{name}",
                source_columns=[name],
                computation="days_since_today",
                description=f"Days since {name}"
            ))

    def _add_default_model(self, findings):
        if findings.target_column:
            feature_cols = [
                name for name, col in findings.columns.items()
                if col.inferred_type not in [ColumnType.IDENTIFIER, ColumnType.TARGET]
            ]
            self.model_config = ModelSpec(
                name="default_model",
                model_type="gradient_boosting",
                target_column=findings.target_column,
                feature_columns=feature_cols
            )

    def _add_default_quality_gates(self, findings):
        self.quality_gates.append(QualityGateSpec(
            name="schema_check",
            gate_type="schema_validation",
            column="*",
            threshold=0
        ))
        self.quality_gates.append(QualityGateSpec(
            name="null_check",
            gate_type="null_percentage",
            column="*",
            threshold=50.0,
            action="warn"
        ))

    def add_transform(self, transform: TransformSpec, stage: str = "silver"):
        if stage == "bronze":
            self.bronze_transforms.append(transform)
        elif stage == "silver":
            self.silver_transforms.append(transform)
        elif stage == "gold":
            self.gold_transforms.append(transform)

    def add_feature(self, feature: FeatureSpec):
        self.feature_definitions.append(feature)

    def add_quality_gate(self, gate: QualityGateSpec):
        self.quality_gates.append(gate)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "sources": [s.to_dict() for s in self.sources],
            "schema": self.schema.to_dict() if self.schema else None,
            "bronze_transforms": [t.to_dict() for t in self.bronze_transforms],
            "silver_transforms": [t.to_dict() for t in self.silver_transforms],
            "gold_transforms": [t.to_dict() for t in self.gold_transforms],
            "feature_definitions": [f.to_dict() for f in self.feature_definitions],
            "model_config": self.model_config.to_dict() if self.model_config else None,
            "quality_gates": [g.to_dict() for g in self.quality_gates],
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save(self, path: str):
        content = self.to_yaml() if path.endswith((".yaml", ".yml")) else self.to_json()
        with open(path, "w") as f:
            f.write(content)

    @classmethod
    def load(cls, path: str) -> "PipelineSpec":
        with open(path, "r") as f:
            content = f.read()
        data = yaml.safe_load(content) if path.endswith((".yaml", ".yml")) else json.loads(content)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "PipelineSpec":
        spec = cls(
            name=data.get("name", "pipeline"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat())
        )
        for src_data in data.get("sources", []):
            spec.sources.append(SourceSpec(**src_data))
        if data.get("schema"):
            schema_data = data["schema"]
            columns = [ColumnSpec(**c) for c in schema_data.get("columns", [])]
            spec.schema = SchemaSpec(
                columns=columns,
                primary_key=schema_data.get("primary_key"),
                partition_columns=schema_data.get("partition_columns", [])
            )
        for t_data in data.get("bronze_transforms", []):
            spec.bronze_transforms.append(TransformSpec(**t_data))
        for t_data in data.get("silver_transforms", []):
            spec.silver_transforms.append(TransformSpec(**t_data))
        for t_data in data.get("gold_transforms", []):
            spec.gold_transforms.append(TransformSpec(**t_data))
        for f_data in data.get("feature_definitions", []):
            spec.feature_definitions.append(FeatureSpec(**f_data))
        if data.get("model_config"):
            spec.model_config = ModelSpec(**data["model_config"])
        for g_data in data.get("quality_gates", []):
            spec.quality_gates.append(QualityGateSpec(**g_data))
        spec.metadata = data.get("metadata", {})
        return spec
