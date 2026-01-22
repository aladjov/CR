import pytest
import pandas as pd
from customer_retention.core.config import ColumnConfig, DataSourceConfig, ColumnType, SourceType, FileFormat
from customer_retention.stages.validation import FeatureQualityGate


@pytest.fixture
def sample_config():
    return DataSourceConfig(
        name="test_source",
        source_type=SourceType.BATCH_FILE,
        path="/tmp/test.csv",
        file_format=FileFormat.CSV,
        primary_key="id",
        columns=[
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS),
            ColumnConfig(name="category", column_type=ColumnType.CATEGORICAL_NOMINAL),
            ColumnConfig(name="is_active", column_type=ColumnType.BINARY),
            ColumnConfig(name="created_date", column_type=ColumnType.DATETIME),
            ColumnConfig(name="target", column_type=ColumnType.TARGET),
        ]
    )


class TestFeatureQualityGateBasic:
    def test_gate_initialization(self):
        gate = FeatureQualityGate()

        assert gate.name == "Feature Quality Gate (Checkpoint 2)"
        assert gate.fail_on_critical is True
        assert gate.fail_on_high is False

    def test_gate_with_clean_data(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0] * 60 + [1] * 40
        })

        result = gate.run(df, sample_config)

        assert result.passed is True or (not result.passed and not result.has_critical_issues())

    def test_gate_with_missing_column(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
        })

        result = gate.run(df, sample_config)

        assert result.passed is False
        assert result.has_critical_issues()
        critical_issues = result.get_critical_issues()
        assert any("not found in dataframe" in issue.message for issue in critical_issues)


class TestFeatureQualityGateMissingValues:
    def test_excessive_missing_values(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [None] * 71 + list(range(29)),  # 71% missing (>70% threshold for HIGH)
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0, 1] * 50
        })

        result = gate.run(df, sample_config)

        issues_for_age = [i for i in result.issues if i.column == "age"]
        assert len(issues_for_age) > 0
        assert any("missing" in i.message.lower() for i in issues_for_age)


class TestFeatureQualityGateConstantFeatures:
    def test_constant_feature_detection(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25] * 100,  # All same value (constant)
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0, 1] * 50
        })

        result = gate.run(df, sample_config)

        issues_for_age = [i for i in result.issues if i.column == "age" and i.code == "FQ003"]
        assert len(issues_for_age) > 0


class TestFeatureQualityGateImbalancedTarget:
    def test_severe_target_imbalance(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0] * 95 + [1] * 5
        })

        result = gate.run(df, sample_config)

        target_issues = [i for i in result.issues if i.column == "target" and i.code in ["CAT002", "TG002"]]
        assert len(target_issues) > 0
        assert any("imbalance" in i.message.lower() for i in target_issues)


class TestFeatureQualityGateIdentifierLeakage:
    def test_identifier_marked_as_feature(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            path="/tmp/test.csv",
            file_format=FileFormat.CSV,
            primary_key="id",
            columns=[
                ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER, is_feature=True),
                ColumnConfig(name="target", column_type=ColumnType.TARGET),
            ]
        )

        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "target": [0, 1] * 50
        })

        result = gate.run(df, config)

        assert result.has_critical_issues()
        id_issues = [i for i in result.issues if i.column == "id" and i.code == "LEAK001"]
        assert len(id_issues) > 0


class TestFeatureQualityGateHighCardinality:
    def test_high_cardinality_categorical(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
            "category": [f"cat_{i}" for i in range(100)],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0, 1] * 50
        })

        result = gate.run(df, sample_config)

        category_issues = [i for i in result.issues if i.column == "category" and i.code in ["CAT001", "CN001", "FQ009"]]
        assert len(category_issues) > 0


class TestFeatureQualityGateDatetimeChecks:
    def test_future_date_detection(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2030-01-01", periods=100),
            "target": [0, 1] * 50
        })

        result = gate.run(df, sample_config)

        date_issues = [i for i in result.issues if i.column == "created_date" and i.code == "DT001"]
        assert len(date_issues) > 0


class TestFeatureQualityGateNumericChecks:
    def test_outlier_detection(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": list(range(1, 81)) + [1000] * 20,
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0, 1] * 50
        })

        result = gate.run(df, sample_config)

        age_issues = [i for i in result.issues if i.column == "age" and i.code in ["NUM003", "NC001", "NC002"]]
        assert len(age_issues) > 0

    def test_zero_inflation_detection(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            path="/tmp/test.csv",
            file_format=FileFormat.CSV,
            primary_key="id",
            columns=[
                ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
                ColumnConfig(name="value", column_type=ColumnType.NUMERIC_CONTINUOUS),
            ]
        )

        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "value": [0] * 60 + list(range(1, 41))
        })

        result = gate.run(df, config)

        value_issues = [i for i in result.issues if i.column == "value" and i.code in ["NUM004", "NC004"]]
        assert len(value_issues) > 0


class TestFeatureQualityGateProfileAndValidate:
    def test_profile_and_validate_combined(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0] * 60 + [1] * 40
        })

        profile_result, gate_result = gate.profile_and_validate(df, sample_config)

        assert profile_result is not None
        assert profile_result.total_rows == 100
        assert profile_result.total_columns == 6
        assert len(profile_result.column_profiles) > 0

        assert gate_result is not None

    def test_profile_includes_type_inference(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0] * 60 + [1] * 40
        })

        profile_result, _ = gate.profile_and_validate(df, sample_config)

        for col_name, profile in profile_result.column_profiles.items():
            assert profile.inferred_type is not None
            assert profile.configured_type is not None


class TestFeatureQualityGateFailureModes:
    def test_fail_on_critical_enabled(self):
        gate = FeatureQualityGate(fail_on_critical=True)

        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            path="/tmp/test.csv",
            file_format=FileFormat.CSV,
            primary_key="id",
            columns=[
                ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER, is_feature=True),
            ]
        )

        df = pd.DataFrame({"id": range(1, 101)})

        result = gate.run(df, config)

        assert result.passed is False

    def test_fail_on_high_enabled(self):
        gate = FeatureQualityGate(fail_on_critical=False, fail_on_high=True)

        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            path="/tmp/test.csv",
            file_format=FileFormat.CSV,
            primary_key="id",
            columns=[
                ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS),
            ]
        )

        df = pd.DataFrame({"age": [None] * 71 + list(range(29))})  # 71% missing (>70% threshold for HIGH)

        result = gate.run(df, config)

        assert result.has_high_issues()


class TestFeatureQualityGateMetadata:
    def test_metadata_includes_column_count(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0, 1] * 50
        })

        result = gate.run(df, sample_config)

        assert "total_columns" in result.metadata
        assert result.metadata["total_columns"] == 6

    def test_metadata_includes_duration(self, sample_config):
        gate = FeatureQualityGate()
        df = pd.DataFrame({
            "id": range(1, 101),
            "age": [25 + i % 50 for i in range(100)],
            "category": ["A", "B", "C"] * 33 + ["A"],
            "is_active": [0, 1] * 50,
            "created_date": pd.date_range("2023-01-01", periods=100),
            "target": [0, 1] * 50
        })

        result = gate.run(df, sample_config)

        assert "duration_seconds" in result.metadata
        assert result.metadata["duration_seconds"] > 0
