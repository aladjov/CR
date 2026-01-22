import pandas as pd

from customer_retention.core.config import (
    BronzeConfig,
    ColumnConfig,
    ColumnType,
    DataSourceConfig,
    FileFormat,
    PipelineConfig,
    SourceType,
)
from customer_retention.stages.validation import DataQualityGate, Severity


class TestDataQualityGate:
    def test_run_gate_on_clean_data(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        result = gate.run(sample_dataframe, sample_pipeline_config)

        assert result.gate_name == "DataQualityGate"
        assert result.metadata["row_count"] == 5
        assert result.metadata["column_count"] == 15

    def test_check_missing_values_none(self, sample_dataframe):
        gate = DataQualityGate()
        issues = gate.check_missing_values(sample_dataframe)
        assert len(issues) == 0

    def test_check_missing_values_critical(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "age": [25, None, None, None]
        })
        gate = DataQualityGate()
        issues = gate.check_missing_values(df)

        assert len(issues) == 1
        assert issues[0].severity == Severity.CRITICAL
        assert issues[0].code == "DQ001"
        assert issues[0].column == "age"
        assert issues[0].affected_pct == 0.75

    def test_check_missing_values_high(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "age": [25, None, None, 30, 35]
        })
        gate = DataQualityGate()
        issues = gate.check_missing_values(df)

        assert len(issues) == 1
        assert issues[0].severity == Severity.HIGH
        assert issues[0].code == "DQ002"
        assert issues[0].affected_pct == 0.4

    def test_check_duplicates_none(self, sample_dataframe):
        config = BronzeConfig(dedup_keys=["custid"])
        gate = DataQualityGate()
        issues = gate.check_duplicates(sample_dataframe, config)
        assert len(issues) == 0

    def test_check_duplicates_present(self):
        df = pd.DataFrame({
            "custid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 25]
        })
        config = BronzeConfig(dedup_keys=["custid"])
        gate = DataQualityGate()
        issues = gate.check_duplicates(df, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ012"
        assert issues[0].severity == Severity.MEDIUM
        assert issues[0].auto_fixable is True

    def test_check_duplicates_high_rate(self):
        df = pd.DataFrame({
            "custid": [1, 1, 1, 1, 1, 2, 3, 4, 5, 6],
            "age": [25] * 10
        })
        config = BronzeConfig(dedup_keys=["custid"])
        gate = DataQualityGate()
        issues = gate.check_duplicates(df, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ011"
        assert issues[0].severity == Severity.HIGH

    def test_check_duplicates_missing_keys(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "age": [25, 30, 35]
        })
        config = BronzeConfig(dedup_keys=["custid"])
        gate = DataQualityGate()
        issues = gate.check_duplicates(df, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ010"
        assert issues[0].severity == Severity.HIGH

    def test_check_target_column_exists(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        issues = gate.check_target_column(sample_dataframe, sample_pipeline_config)
        assert len(issues) == 0

    def test_check_target_column_missing(self, sample_dataframe):
        config = PipelineConfig(
            project_name="test",
            data_sources=[]
        )
        config.modeling.target_column = "nonexistent"

        gate = DataQualityGate()
        issues = gate.check_target_column(sample_dataframe, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ020"
        assert issues[0].severity == Severity.CRITICAL

    def test_check_target_column_has_nulls(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "retained": [1, None, 0]
        })
        config = PipelineConfig(
            project_name="test",
            data_sources=[]
        )

        gate = DataQualityGate()
        issues = gate.check_target_column(df, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ021"
        assert issues[0].severity == Severity.CRITICAL

    def test_check_class_imbalance_none(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        issues = gate.check_class_imbalance(sample_dataframe, sample_pipeline_config)
        assert len(issues) == 0

    def test_check_class_imbalance_severe(self):
        df = pd.DataFrame({
            "retained": [1] * 199 + [0]
        })
        config = PipelineConfig(project_name="test", data_sources=[])

        gate = DataQualityGate()
        issues = gate.check_class_imbalance(df, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ022"
        assert issues[0].severity == Severity.HIGH

    def test_check_class_imbalance_moderate(self):
        df = pd.DataFrame({
            "retained": [1] * 95 + [0] * 5
        })
        config = PipelineConfig(project_name="test", data_sources=[])

        gate = DataQualityGate()
        issues = gate.check_class_imbalance(df, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ023"
        assert issues[0].severity == Severity.MEDIUM

    def test_check_temporal_validity_no_future_dates(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        issues = gate.check_temporal_validity(sample_dataframe, sample_pipeline_config)
        assert all(i.code != "DQ030" for i in issues)

    def test_check_temporal_validity_future_dates(self):
        df = pd.DataFrame({
            "created": pd.to_datetime(["2030-01-01", "2023-01-01"])
        })
        columns = [ColumnConfig(name="created", column_type=ColumnType.DATETIME)]
        source = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        config = PipelineConfig(project_name="test", data_sources=[source])

        gate = DataQualityGate()
        issues = gate.check_temporal_validity(df, config)

        future_issues = [i for i in issues if i.code == "DQ030"]
        assert len(future_issues) == 1
        assert future_issues[0].severity == Severity.HIGH

    def test_check_temporal_logic_valid(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        issues = gate.check_temporal_logic(sample_dataframe, sample_pipeline_config)
        assert len(issues) == 0

    def test_check_temporal_logic_violation(self):
        df = pd.DataFrame({
            "created": pd.to_datetime(["2023-06-01", "2023-01-01"]),
            "firstorder": pd.to_datetime(["2023-01-01", "2023-01-02"])
        })
        config = PipelineConfig(project_name="test", data_sources=[])

        gate = DataQualityGate()
        issues = gate.check_temporal_logic(df, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ031"
        assert issues[0].severity == Severity.HIGH

    def test_check_type_mismatches_none(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        issues = gate.check_type_mismatches(sample_dataframe, sample_pipeline_config)
        assert len(issues) == 0

    def test_check_type_mismatches_numeric_as_string(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "age": ["25", "30", "35"]
        })
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]
        source = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        config = PipelineConfig(project_name="test", data_sources=[source])

        gate = DataQualityGate()
        issues = gate.check_type_mismatches(df, config)

        assert len(issues) == 1
        assert issues[0].code == "DQ040"
        assert issues[0].severity == Severity.MEDIUM
        assert issues[0].auto_fixable is True

    def test_get_date_columns(self):
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="created", column_type=ColumnType.DATETIME),
            ColumnConfig(name="updated", column_type=ColumnType.DATETIME),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]
        source = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        config = PipelineConfig(project_name="test", data_sources=[source])

        gate = DataQualityGate()
        df = pd.DataFrame()
        date_columns = gate.get_date_columns(df, config)

        assert len(date_columns) == 2
        assert "created" in date_columns
        assert "updated" in date_columns

    def test_gate_passes_clean_data(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        result = gate.run(sample_dataframe, sample_pipeline_config)

        assert result.passed is True

    def test_gate_fails_on_critical_issue(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "age": [25, 30, 35]
        })
        config = PipelineConfig(project_name="test", data_sources=[])
        config.modeling.target_column = "nonexistent"

        gate = DataQualityGate()
        result = gate.run(df, config)

        assert result.passed is False
        assert result.has_critical_issues() is True

    def test_gate_respects_fail_on_high_setting(self):
        df = pd.DataFrame({
            "custid": [1, 1, 2],
            "retained": [1, 1, 0]
        })
        config = PipelineConfig(project_name="test", data_sources=[])
        config.validation.fail_on_high = True
        config.bronze.dedup_keys = ["custid"]

        gate = DataQualityGate()
        result = gate.run(df, config)

        assert result.passed is False

    def test_gate_warning_only_on_high_when_disabled(self):
        df = pd.DataFrame({
            "custid": [1, 1, 2, 2, 2, 3, 4, 5, 6, 7],
            "retained": [1] * 10
        })
        config = PipelineConfig(project_name="test", data_sources=[])
        config.validation.fail_on_high = False
        config.bronze.dedup_keys = ["custid"]

        gate = DataQualityGate()
        result = gate.run(df, config)

        assert result.has_high_issues() is True
        assert result.passed is True

    def test_run_returns_duration(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        result = gate.run(sample_dataframe, sample_pipeline_config)

        assert result.duration_seconds > 0

    def test_run_includes_timestamp(self, sample_dataframe, sample_pipeline_config):
        gate = DataQualityGate()
        result = gate.run(sample_dataframe, sample_pipeline_config)

        assert result.timestamp is not None
        assert len(result.timestamp) > 0
