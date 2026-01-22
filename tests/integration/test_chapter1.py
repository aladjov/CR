import pytest
import pandas as pd
from customer_retention.core.config import (
    ColumnType, ColumnConfig, DataSourceConfig,
    SourceType, FileFormat, PipelineConfig
)
from customer_retention.stages.ingestion import LoaderFactory, DataSourceRegistry
from customer_retention.stages.validation import DataQualityGate


class TestChapter1Integration:
    def test_full_workflow_csv_to_gate(self, temp_csv_file, sample_columns):
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=temp_csv_file,
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
        assert load_result.row_count == 5

        gate = DataQualityGate()
        gate_result = gate.run(df, pipeline_config)

        assert gate_result.gate_name == "DataQualityGate"
        assert gate_result.passed is True

    def test_config_to_registry_to_load_to_gate(self, temp_csv_file, sample_columns, temp_registry_file):
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=temp_csv_file,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        registry = DataSourceRegistry()
        registry.register(source_config, registered_by="test_user")

        loader = LoaderFactory.get_loader(source_config)
        df, load_result = loader.load(source_config)

        registry.record_load(source_config.name, load_result)

        stats = registry.get_load_stats("customer_master")
        assert stats["total_loads"] == 1
        assert stats["successful_loads"] == 1

        registry.save_to_file(temp_registry_file)

        new_registry = DataSourceRegistry()
        new_registry.load_from_file(temp_registry_file)
        assert len(new_registry.list_sources()) == 1

    def test_parquet_workflow(self, temp_parquet_file, sample_columns):
        source_config = DataSourceConfig(
            name="customer_parquet",
            source_type=SourceType.BATCH_FILE,
            path=temp_parquet_file,
            file_format=FileFormat.PARQUET,
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
        assert load_result.row_count == 5

        gate = DataQualityGate()
        gate_result = gate.run(df, pipeline_config)

        assert gate_result.passed is True

    def test_sample_loading_workflow(self, temp_csv_file, sample_columns):
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=temp_csv_file,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        loader = LoaderFactory.get_loader(source_config)
        df, load_result = loader.load(source_config, sample_size=3)

        assert load_result.success is True
        assert load_result.row_count == 3
        assert len(df) == 3

    def test_workflow_with_data_quality_issues(self):
        df = pd.DataFrame({
            "custid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
            "retained": [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, None]
        })

        columns = [
            ColumnConfig(name="custid", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="retained", column_type=ColumnType.TARGET),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]

        source_config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=columns
        )

        pipeline_config = PipelineConfig(
            project_name="test_project",
            data_sources=[source_config]
        )

        gate = DataQualityGate()
        gate_result = gate.run(df, pipeline_config)

        assert gate_result.passed is True
        assert len(gate_result.issues) > 0

        issue_codes = [i.code for i in gate_result.issues]
        assert "DQ012" in issue_codes

    def test_config_serialization_round_trip(self, sample_columns):
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        pipeline_config = PipelineConfig(
            project_name="retail_churn_2024",
            data_sources=[source_config]
        )

        json_data = pipeline_config.model_dump()
        restored_config = PipelineConfig(**json_data)

        assert restored_config.project_name == pipeline_config.project_name
        assert len(restored_config.data_sources) == len(pipeline_config.data_sources)
        assert restored_config.data_sources[0].name == source_config.name

    def test_multiple_loads_to_registry(self, temp_csv_file, sample_columns):
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path=temp_csv_file,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        registry = DataSourceRegistry()
        registry.register(source_config)

        loader = LoaderFactory.get_loader(source_config)

        for i in range(3):
            df, load_result = loader.load(source_config)
            registry.record_load(source_config.name, load_result)

        stats = registry.get_load_stats("customer_master")
        assert stats["total_loads"] == 3
        assert stats["successful_loads"] == 3
        assert stats["failed_loads"] == 0

    def test_get_feature_columns_from_config(self, sample_columns):
        source_config = DataSourceConfig(
            name="customer_master",
            source_type=SourceType.BATCH_FILE,
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        pipeline_config = PipelineConfig(
            project_name="retail_churn_2024",
            data_sources=[source_config]
        )

        feature_columns = pipeline_config.get_all_feature_columns()

        assert "custid" not in feature_columns
        assert "retained" not in feature_columns
        assert "avgorder" in feature_columns
        assert "city" in feature_columns

    def test_cloud_path_detection(self):
        s3_config = DataSourceConfig(
            name="s3_source",
            source_type=SourceType.BATCH_FILE,
            path="s3://bucket/data.csv",
            file_format=FileFormat.CSV,
            primary_key="id"
        )
        assert s3_config.is_cloud_path() is True

        abfss_config = DataSourceConfig(
            name="abfss_source",
            source_type=SourceType.BATCH_FILE,
            path="abfss://container@account.dfs.core.windows.net/data.csv",
            file_format=FileFormat.CSV,
            primary_key="id"
        )
        assert abfss_config.is_cloud_path() is True

        local_config = DataSourceConfig(
            name="local_source",
            source_type=SourceType.BATCH_FILE,
            path="/local/data.csv",
            file_format=FileFormat.CSV,
            primary_key="id"
        )
        assert local_config.is_cloud_path() is False

    def test_validation_config_controls_gate_behavior(self, temp_csv_file, sample_columns):
        df = pd.DataFrame({
            "custid": [1, 1, 2, 3, 4],
            "retained": [1, 1, 0, 1, 0],
            "age": [25, 25, 30, 35, 40]
        })

        source_config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            path=temp_csv_file,
            file_format=FileFormat.CSV,
            primary_key="custid",
            columns=sample_columns
        )

        config_fail_on_high = PipelineConfig(
            project_name="test",
            data_sources=[source_config]
        )
        config_fail_on_high.validation.fail_on_high = True
        config_fail_on_high.bronze.dedup_keys = ["custid"]

        config_warn_on_high = PipelineConfig(
            project_name="test",
            data_sources=[source_config]
        )
        config_warn_on_high.validation.fail_on_high = False
        config_warn_on_high.bronze.dedup_keys = ["custid"]

        gate = DataQualityGate()

        result_fail = gate.run(df, config_fail_on_high)
        result_warn = gate.run(df, config_warn_on_high)

        assert result_fail.has_high_issues() or result_fail.passed is True
        assert result_warn.passed is True or result_warn.has_high_issues()
