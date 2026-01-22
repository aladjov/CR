import pandas as pd

from customer_retention.core.config import DataSourceConfig, FileFormat, PipelineConfig, SourceType
from customer_retention.stages.ingestion import LoaderFactory
from customer_retention.stages.validation import DataQualityGate


class TestRetailDatasetSpecification:
    """Tests against actual retail dataset specification (Section 3 of spec)"""

    def test_retail_dataset_exists_and_has_correct_dimensions(self, retail_dataframe):
        """AC5.2: Retail dataset has 30,801 rows and 15 columns"""
        assert len(retail_dataframe) == 30801, "Dataset should have exactly 30,801 rows"
        assert len(retail_dataframe.columns) == 15, "Dataset should have exactly 15 columns"

    def test_retail_dataset_has_correct_columns(self, retail_dataframe):
        """Verify all 15 columns from spec Section 3.2 are present"""
        expected_columns = {
            "custid", "retained", "created", "firstorder", "lastorder",
            "esent", "eopenrate", "eclickrate", "avgorder", "ordfreq",
            "paperless", "refill", "doorstep", "favday", "city"
        }
        actual_columns = set(retail_dataframe.columns)
        assert actual_columns == expected_columns, f"Missing columns: {expected_columns - actual_columns}"

    def test_retail_dataset_column_types(self, retail_dataframe):
        """Verify data types match specification"""
        df = retail_dataframe

        # custid is alphanumeric string identifier in actual data
        assert df["custid"].dtype == "object", "custid should be string/object type"
        assert pd.api.types.is_integer_dtype(df["retained"]), "retained should be binary (0/1)"
        assert pd.api.types.is_integer_dtype(df["esent"]), "esent should be integer"
        assert pd.api.types.is_numeric_dtype(df["eopenrate"]), "eopenrate should be numeric"
        assert pd.api.types.is_numeric_dtype(df["avgorder"]), "avgorder should be numeric"
        assert pd.api.types.is_integer_dtype(df["paperless"]), "paperless should be binary"
        assert df["city"].dtype == "object", "city should be string/object type"

    def test_retail_dataset_target_column_binary(self, retail_dataframe):
        """Verify target column contains only 0 and 1"""
        unique_values = set(retail_dataframe["retained"].unique())
        assert unique_values == {0, 1}, f"retained should only have values 0 and 1, found: {unique_values}"


class TestRetailDatasetLoading:
    """Tests for loading retail dataset (AC2.1, Section 5.2)"""

    def test_csv_loader_loads_retail_dataset_successfully(self, retail_dataset_path, sample_columns):
        """AC2.1: CSV files load successfully"""
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=retail_dataset_path,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        loader = LoaderFactory.get_loader(source_config)
        df, result = loader.load(source_config)

        assert result.success is True
        assert len(df) == 30801
        assert result.row_count == 30801, "AC2.4: LoadResult.row_count should match actual count"
        assert result.column_count == 15, "AC2.4: LoadResult.column_count should match actual count"

    def test_load_result_schema_info_contains_all_columns(self, retail_dataset_path, sample_columns):
        """AC2.5: Schema info is captured correctly"""
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=retail_dataset_path,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        loader = LoaderFactory.get_loader(source_config)
        df, result = loader.load(source_config)

        assert len(result.schema_info) == 15, "Schema info should contain all 15 columns"
        assert "custid" in result.schema_info
        assert "retained" in result.schema_info
        assert "city" in result.schema_info

    def test_sample_loading_returns_exactly_100_rows(self, retail_dataset_path, sample_columns):
        """AC2.7: Load with sample_size=100 returns exactly 100 rows"""
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=retail_dataset_path,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        loader = LoaderFactory.get_loader(source_config)
        df, result = loader.load(source_config, sample_size=100)

        assert len(df) == 100, "Sample loading should return exactly 100 rows"
        assert result.row_count == 100


class TestRetailDataQualityGate:
    """Tests for Data Quality Gate on retail dataset (AC4.1, AC5.2, Section 5.4)"""

    def test_gate_runs_on_retail_dataset_without_error(self, retail_dataframe, sample_pipeline_config):
        """AC4.1: All checks execute without error on retail dataset"""
        gate = DataQualityGate()
        result = gate.run(retail_dataframe, sample_pipeline_config)

        assert result is not None
        assert result.gate_name == "DataQualityGate"
        assert result.timestamp is not None
        assert result.duration_seconds > 0

    def test_retail_dataset_passes_data_quality_gate(self, retail_dataframe, sample_pipeline_config):
        """AC5.2: Retail dataset passes Data Quality Gate"""
        gate = DataQualityGate()
        result = gate.run(retail_dataframe, sample_pipeline_config)

        assert result.passed is True, f"Gate should pass. Issues: {[i.get_display_string() for i in result.issues]}"

    def test_gate_result_serializes_to_dict(self, retail_dataframe, sample_pipeline_config):
        """AC4.4: GateResult.to_dict() produces valid output"""
        gate = DataQualityGate()
        result = gate.run(retail_dataframe, sample_pipeline_config)

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "gate_name" in result_dict
        assert "passed" in result_dict
        assert "issues" in result_dict
        assert "timestamp" in result_dict
        assert "duration_seconds" in result_dict
        assert result_dict["gate_name"] == "DataQualityGate"

    def test_gate_detects_no_missing_values_in_clean_retail_data(self, retail_dataframe):
        """Section 5.4: Gate detects missing value columns correctly"""
        gate = DataQualityGate()
        issues = gate.check_missing_values(retail_dataframe)

        assert len(issues) == 0, "Clean retail dataset should have no missing value issues"

    def test_gate_detects_duplicates_in_retail_data(self, retail_dataframe, sample_pipeline_config):
        """Section 5.4: Gate detects duplicate rows"""
        gate = DataQualityGate()
        issues = gate.check_duplicates(retail_dataframe, sample_pipeline_config.bronze)

        # Note: Retail data has 31 duplicate custid values which the gate should detect
        dup_issues = [i for i in issues if i.code == "DQ012"]
        assert len(dup_issues) > 0, "Gate should detect duplicate custid values in retail data"

    def test_gate_validates_target_column_exists(self, retail_dataframe, sample_pipeline_config):
        """Section 5.4: Gate validates target column exists"""
        gate = DataQualityGate()
        issues = gate.check_target_column(retail_dataframe, sample_pipeline_config)

        assert len(issues) == 0, "Target column 'retained' should exist in retail dataset"

    def test_gate_checks_temporal_logic(self, retail_dataframe, sample_pipeline_config):
        """Section 5.4: Gate checks date logic (created <= firstorder)"""
        gate = DataQualityGate()
        issues = gate.check_temporal_logic(retail_dataframe, sample_pipeline_config)

        # Note: Retail data has 659 rows with temporal logic violations (created > firstorder)
        # The gate should detect these data quality issues
        temporal_violations = [i for i in issues if i.code == "DQ031"]
        assert len(temporal_violations) > 0, "Gate should detect temporal logic violations"


class TestRetailEndToEndWorkflow:
    """End-to-end tests with retail dataset (AC5.1, AC5.3, Section 5.5)"""

    def test_full_pipeline_config_load_gate_with_retail_data(self, retail_dataset_path, sample_columns):
        """AC5.1: Full pipeline: Config → Load → Gate"""
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=retail_dataset_path,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        pipeline_config = PipelineConfig(
            project_name="retail_churn_2024",
            data_sources=[source_config]
        )

        loader = LoaderFactory.get_loader(source_config)
        df, load_result = loader.load(source_config)

        assert load_result.success is True

        gate = DataQualityGate()
        gate_result = gate.run(df, pipeline_config)

        assert gate_result.passed is True

    def test_workflow_completes_in_reasonable_time(self, retail_dataset_path, sample_columns):
        """Section 5.5: Full workflow completes in reasonable time (<30 seconds)"""
        import time

        start_time = time.time()

        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=retail_dataset_path,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        pipeline_config = PipelineConfig(
            project_name="retail_churn_2024",
            data_sources=[source_config]
        )

        loader = LoaderFactory.get_loader(source_config)
        df, load_result = loader.load(source_config)

        gate = DataQualityGate()
        gate_result = gate.run(df, pipeline_config)

        duration = time.time() - start_time

        assert duration < 30.0, f"Workflow should complete in <30s, took {duration:.2f}s"

    def test_clear_reporting_of_gate_results(self, retail_dataframe, sample_pipeline_config):
        """AC5.3: Clear reporting of any issues"""
        gate = DataQualityGate()
        result = gate.run(retail_dataframe, sample_pipeline_config)

        summary = result.get_summary()
        assert "DataQualityGate" in summary
        assert "PASSED" in summary or "FAILED" in summary

        for issue in result.issues:
            display = issue.get_display_string()
            assert len(display) > 0
            assert issue.code in display
            assert issue.severity.value.upper() in display
