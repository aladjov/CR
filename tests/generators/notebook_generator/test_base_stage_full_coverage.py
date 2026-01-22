from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest


@dataclass
class MockColumnFinding:
    name: str
    inferred_type: Any
    universal_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockExplorationFindings:
    source_path: str = "/data/test.csv"
    source_format: str = "csv"
    target_column: Optional[str] = None
    identifier_columns: List[str] = field(default_factory=list)
    columns: Dict[str, MockColumnFinding] = field(default_factory=dict)


class TestStageGeneratorDescription:
    def test_default_description_is_empty(self):

        from customer_retention.generators.notebook_generator.base import NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.base_stage import StageGenerator

        class MinimalStage(StageGenerator):
            @property
            def stage(self) -> NotebookStage:
                return NotebookStage.INGESTION

            @property
            def title(self) -> str:
                return "Test Stage"

            def generate_local_cells(self):
                return []

            def generate_databricks_cells(self):
                return []

        config = NotebookConfig()
        stage = MinimalStage(config, None)
        assert stage.description == ""
        header_cells = stage.header_cells()
        assert len(header_cells) == 1


class TestStageGeneratorGetTargetColumn:
    def test_returns_target_from_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        findings = MockExplorationFindings(target_column="churn")
        config = NotebookConfig()
        stage = IngestionStage(config, findings)
        assert stage.get_target_column() == "churn"

    def test_returns_default_when_no_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        config = NotebookConfig()
        stage = IngestionStage(config, None)
        assert stage.get_target_column() == "target"


class TestStageGeneratorGetIdentifierColumns:
    def test_returns_identifiers_from_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        findings = MockExplorationFindings(identifier_columns=["user_id", "account_id"])
        config = NotebookConfig()
        stage = IngestionStage(config, findings)
        assert stage.get_identifier_columns() == ["user_id", "account_id"]

    def test_returns_default_when_no_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        config = NotebookConfig()
        stage = IngestionStage(config, None)
        assert stage.get_identifier_columns() == ["customer_id"]


class TestStageGeneratorGetFeatureColumns:
    def test_returns_empty_when_no_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        config = NotebookConfig()
        stage = IngestionStage(config, None)
        assert stage.get_feature_columns() == []

    def test_returns_feature_columns_from_findings(self):
        from customer_retention.core.config import ColumnType as CT
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        columns = {
            "age": MockColumnFinding("age", CT.NUMERIC_CONTINUOUS),
            "gender": MockColumnFinding("gender", CT.CATEGORICAL_NOMINAL),
            "customer_id": MockColumnFinding("customer_id", CT.IDENTIFIER),
        }
        findings = MockExplorationFindings(columns=columns)
        config = NotebookConfig()
        stage = IngestionStage(config, findings)
        features = stage.get_feature_columns()
        assert "age" in features
        assert "gender" in features
        assert "customer_id" not in features


class TestStageGeneratorGetNumericColumns:
    def test_returns_empty_when_no_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        config = NotebookConfig()
        stage = IngestionStage(config, None)
        assert stage.get_numeric_columns() == []

    def test_returns_numeric_columns_from_findings(self):
        from customer_retention.core.config import ColumnType as CT
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        columns = {
            "age": MockColumnFinding("age", CT.NUMERIC_CONTINUOUS),
            "count": MockColumnFinding("count", CT.NUMERIC_DISCRETE),
            "gender": MockColumnFinding("gender", CT.CATEGORICAL_NOMINAL),
        }
        findings = MockExplorationFindings(columns=columns)
        config = NotebookConfig()
        stage = IngestionStage(config, findings)
        numeric = stage.get_numeric_columns()
        assert "age" in numeric
        assert "count" in numeric
        assert "gender" not in numeric


class TestStageGeneratorGetCategoricalColumns:
    def test_returns_empty_when_no_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        config = NotebookConfig()
        stage = IngestionStage(config, None)
        assert stage.get_categorical_columns() == []

    def test_returns_categorical_columns_from_findings(self):
        from customer_retention.core.config import ColumnType as CT
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        columns = {
            "age": MockColumnFinding("age", CT.NUMERIC_CONTINUOUS),
            "gender": MockColumnFinding("gender", CT.CATEGORICAL_NOMINAL),
            "tier": MockColumnFinding("tier", CT.CATEGORICAL_ORDINAL),
        }
        findings = MockExplorationFindings(columns=columns)
        config = NotebookConfig()
        stage = IngestionStage(config, findings)
        categorical = stage.get_categorical_columns()
        assert "gender" in categorical
        assert "tier" in categorical
        assert "age" not in categorical


class TestAllStageProperties:
    @pytest.mark.parametrize("stage_class,expected_stage", [
        ("IngestionStage", "INGESTION"),
        ("ProfilingStage", "PROFILING"),
        ("CleaningStage", "CLEANING"),
        ("TransformationStage", "TRANSFORMATION"),
        ("FeatureEngineeringStage", "FEATURE_ENGINEERING"),
        ("FeatureSelectionStage", "FEATURE_SELECTION"),
        ("ModelTrainingStage", "MODEL_TRAINING"),
        ("DeploymentStage", "DEPLOYMENT"),
        ("MonitoringStage", "MONITORING"),
        ("BatchInferenceStage", "BATCH_INFERENCE"),
    ])
    def test_stage_property_returns_correct_enum(self, stage_class, expected_stage):
        from customer_retention.generators.notebook_generator import stages
        from customer_retention.generators.notebook_generator.base import NotebookStage
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        cls = getattr(stages, stage_class)
        config = NotebookConfig()
        instance = cls(config, None)
        assert instance.stage == NotebookStage[expected_stage]
        assert instance.stage.name == expected_stage
