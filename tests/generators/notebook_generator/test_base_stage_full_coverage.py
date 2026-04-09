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


class TestStageGeneratorGetDatasetName:
    def test_returns_stem_from_findings_source_path(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        findings = MockExplorationFindings(source_path="/data/customer_emails.csv")
        stage = IngestionStage(NotebookConfig(), findings)
        assert stage.get_dataset_name() == "customer_emails"

    def test_returns_fallback_without_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        stage = IngestionStage(NotebookConfig(), None)
        assert stage.get_dataset_name() == "dataset"

    def test_returns_fallback_with_empty_source_path(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        findings = MockExplorationFindings(source_path="")
        stage = IngestionStage(NotebookConfig(), findings)
        assert stage.get_dataset_name() == "dataset"


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


class TestStagePathsUseDatasetName:
    def _code_for_stage(self, stage_cls, source_path="/data/customer_emails.csv"):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        findings = MockExplorationFindings(source_path=source_path)
        stage = stage_cls(NotebookConfig(), findings)
        cells = stage.generate_local_cells()
        return "\n".join(c.source for c in cells if c.cell_type == "code")

    def test_ingestion_passes_dataset_name_to_preparer(self):
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        code = self._code_for_stage(IngestionStage)
        assert 'dataset_name="customer_emails"' in code

    def test_cleaning_uses_dataset_name_in_paths(self):
        from customer_retention.generators.notebook_generator.stages.s03_cleaning import CleaningStage
        code = self._code_for_stage(CleaningStage)
        assert "bronze/customer_emails" in code
        assert "silver/customer_emails_cleaned" in code

    def test_transformation_uses_dataset_name_in_paths(self):
        from customer_retention.generators.notebook_generator.stages.s04_transformation import TransformationStage
        code = self._code_for_stage(TransformationStage)
        assert "silver/customer_emails_cleaned" in code
        assert "silver/customer_emails_transformed" in code

    def test_feature_engineering_uses_dataset_name_in_paths(self):
        from customer_retention.generators.notebook_generator.stages.s05_feature_engineering import (
            FeatureEngineeringStage,
        )
        code = self._code_for_stage(FeatureEngineeringStage)
        assert "silver/customer_emails_transformed" in code
        assert "gold/customer_emails_features" in code

    def test_feature_selection_uses_dataset_name_in_paths(self):
        from customer_retention.generators.notebook_generator.stages.s06_feature_selection import FeatureSelectionStage
        code = self._code_for_stage(FeatureSelectionStage)
        assert "gold/customer_emails_features.parquet" in code
        assert "gold/customer_emails_selected.parquet" in code

    def test_batch_inference_uses_dataset_name_in_paths(self):
        # Post-refactor (Phase 1.6): the inline gold-path business logic moved
        # into customer_retention.stages.scoring.batch_inference. The cell now
        # passes the dataset_name through to BatchInferenceConfig and the
        # framework's _load_local_customers() owns the gold/{name}_to_score
        # and gold/{name}_features fallback chain. This test verifies the
        # dataset_name still flows through to the framework call.
        from customer_retention.generators.notebook_generator.stages.s10_batch_inference import BatchInferenceStage
        code = self._code_for_stage(BatchInferenceStage)
        assert "BatchInferenceConfig" in code
        assert "run_batch_inference" in code
        assert "dataset_name='customer_emails'" in code or 'dataset_name="customer_emails"' in code

    def test_batch_inference_local_customer_fallback_chain_in_framework(self):
        # The gold/{name}_to_score and gold/{name}_features fallback paths now
        # live in the framework module, not the cell text. This guards the
        # contract between the cell's dataset_name argument and the framework
        # paths so a future refactor can't silently drop one.
        from customer_retention.stages.scoring.batch_inference import _load_local_customers  # noqa
        import inspect
        source = inspect.getsource(_load_local_customers)
        assert "_to_score" in source
        assert "_features" in source
        assert "_delta_log" in source

    def test_fallback_dataset_name_without_findings(self):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s03_cleaning import CleaningStage
        stage = CleaningStage(NotebookConfig(), None)
        cells = stage.generate_local_cells()
        code = "\n".join(c.source for c in cells if c.cell_type == "code")
        assert "bronze/dataset" in code
        assert "silver/dataset_cleaned" in code


class TestIngestionDatabricksReadDispatch:
    def _databricks_code(self, source_path, source_format="csv"):
        from customer_retention.generators.notebook_generator.config import NotebookConfig
        from customer_retention.generators.notebook_generator.stages.s01_ingestion import IngestionStage
        findings = MockExplorationFindings(source_path=source_path, source_format=source_format)
        stage = IngestionStage(NotebookConfig(), findings)
        cells = stage.generate_databricks_cells()
        return "\n".join(c.source for c in cells if c.cell_type == "code")

    def test_databricks_uc_table_path_uses_spark_read_table(self):
        code = self._databricks_code("sps.production.case", "delta")
        assert "spark.read.table(" in code
        assert '.format("delta").option' not in code
        assert '.load(DATA_PATH)' not in code

    def test_databricks_file_path_uses_format_load(self):
        code = self._databricks_code("/mnt/landing/customers.csv", "csv")
        assert '.format("csv")' in code
        assert ".load(DATA_PATH)" in code

    def test_databricks_dbfs_path_uses_format_load(self):
        code = self._databricks_code("dbfs:/mnt/raw/orders.parquet", "parquet")
        assert '.format("parquet")' in code
        assert ".load(DATA_PATH)" in code


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
