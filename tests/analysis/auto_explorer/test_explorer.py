import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.explorer import DataExplorer
from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.core.config.column_config import ColumnType


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(100)],
        "age": np.random.randint(18, 80, 100),
        "monthly_charges": np.random.uniform(20, 100, 100),
        "tenure": np.random.randint(0, 72, 100),
        "gender": np.random.choice(["Male", "Female"], 100),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], 100),
        "churned": np.random.choice([0, 1], 100),
        "signup_date": pd.date_range("2020-01-01", periods=100, freq="D")
    })


@pytest.fixture
def sample_csv_file(sample_dataframe):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_data.csv"
        sample_dataframe.to_csv(path, index=False)
        yield str(path)


class TestDataExplorerInit:
    def test_default_init(self):
        explorer = DataExplorer()
        assert explorer.visualize
        assert explorer.save_findings
        assert explorer.output_dir == Path("../explorations")

    def test_custom_init(self):
        explorer = DataExplorer(
            visualize=False,
            save_findings=False,
            output_dir="/custom/path"
        )
        assert not explorer.visualize
        assert not explorer.save_findings
        assert explorer.output_dir == Path("/custom/path")


class TestDataExplorerExplore:
    def test_explore_dataframe(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        assert isinstance(findings, ExplorationFindings)
        assert findings.row_count == 100
        assert findings.column_count == 8
        assert findings.source_path == "<DataFrame>"
        assert findings.source_format == "dataframe"

    def test_explore_csv_file(self, sample_csv_file):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_csv_file)

        assert findings.source_format == "csv"
        assert findings.row_count == 100

    def test_column_types_detected(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        assert "customer_id" in findings.column_types
        assert findings.column_types["customer_id"] == ColumnType.IDENTIFIER

    def test_target_detection_with_hint(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe, target_hint="churned")

        assert findings.target_column == "churned"

    def test_datetime_column_tracked(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        assert "signup_date" in findings.datetime_columns

    def test_quality_score_calculated(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        assert 0 <= findings.overall_quality_score <= 100

    def test_memory_usage_calculated(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        assert findings.memory_usage_mb > 0

    def test_findings_saved_to_disk(self, sample_dataframe):
        with tempfile.TemporaryDirectory() as tmpdir:
            explorer = DataExplorer(
                visualize=False,
                save_findings=True,
                output_dir=tmpdir
            )
            findings = explorer.explore(sample_dataframe, name="test_exploration")

            # Implementation adds hash to filename: {name}_{hash}_findings.yaml
            saved_files = list(Path(tmpdir).glob("test_exploration_*_findings.yaml"))
            assert len(saved_files) == 1


class TestDataExplorerColumnExploration:
    def test_identifier_detection(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        assert "customer_id" in findings.identifier_columns

    def test_numeric_continuous_detection(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        assert findings.column_types["monthly_charges"] == ColumnType.NUMERIC_CONTINUOUS

    def test_categorical_detection(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        gender_type = findings.column_types["gender"]
        assert gender_type in [ColumnType.BINARY, ColumnType.CATEGORICAL_NOMINAL]

    def test_column_finding_has_metrics(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        age_finding = findings.columns["age"]
        assert "total_count" in age_finding.universal_metrics
        assert age_finding.universal_metrics["total_count"] == 100


class TestDataExplorerQualityAnalysis:
    def test_quality_issues_detected_for_nulls(self):
        df = pd.DataFrame({
            "col_with_nulls": [1, 2, None, None, None, None, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["col_with_nulls"]
        assert any("missing" in issue.lower() for issue in col_finding.quality_issues)

    def test_cleaning_recommendations_generated(self):
        df = pd.DataFrame({
            "col_with_nulls": [1, 2, None, None, None, None, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["col_with_nulls"]
        assert col_finding.cleaning_needed
        assert len(col_finding.cleaning_recommendations) > 0


class TestDataExplorerModelingReadiness:
    def test_modeling_ready_with_target(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe, target_hint="churned")

        assert findings.target_column is not None

    def test_not_ready_without_target(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df)

        assert not findings.modeling_ready
        assert "No target column detected" in findings.blocking_issues

    def test_target_type_binary(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe, target_hint="churned")

        assert findings.target_type == "binary"

    def test_target_type_multiclass(self):
        df = pd.DataFrame({
            "features": [1, 2, 3, 4, 5],
            "target_class": ["A", "B", "C", "A", "B"]
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target_class")

        assert findings.target_type == "multiclass"


class TestDataExplorerParquetLoading:
    def test_explore_parquet_file(self, sample_dataframe):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.parquet"
            sample_dataframe.to_parquet(path, index=False)

            explorer = DataExplorer(visualize=False, save_findings=False)
            findings = explorer.explore(str(path))

            assert findings.source_format == "parquet"
            assert findings.row_count == 100


class TestDataExplorerQualityIssues:
    def test_high_cardinality_detection(self):
        df = pd.DataFrame({
            "high_card_col": [f"cat_{i % 150}" for i in range(300)],
            "target": [0, 1] * 150
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["high_card_col"]
        col_type = col_finding.inferred_type
        assert col_type in [ColumnType.TEXT, ColumnType.CATEGORICAL_NOMINAL]

    def test_critical_nulls_detection(self):
        df = pd.DataFrame({
            "mostly_null": [None] * 60 + [1] * 40,
            "target": [0, 1] * 50
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["mostly_null"]
        assert any("CRITICAL" in issue for issue in col_finding.quality_issues)
        assert any("dropping" in rec.lower() for rec in col_finding.cleaning_recommendations)

    def test_warning_nulls_detection(self):
        df = pd.DataFrame({
            "some_nulls": [None] * 25 + [1] * 75,
            "target": [0, 1] * 50
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["some_nulls"]
        assert any("WARNING" in issue for issue in col_finding.quality_issues)

    def test_modeling_ready_with_critical_issues(self):
        df = pd.DataFrame({
            "mostly_null": [None] * 60 + [1] * 40,
            "target": [0, 1] * 50
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        assert not findings.modeling_ready
        assert any("Critical issues" in issue for issue in findings.blocking_issues)


class TestDataExplorerTransformationRecommendations:
    def test_datetime_recommendations(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        date_finding = findings.columns["signup_date"]
        assert any("temporal" in rec.lower() for rec in date_finding.transformation_recommendations)

    def test_categorical_recommendations(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        contract_finding = findings.columns["contract"]
        assert any("encoding" in rec.lower() for rec in contract_finding.transformation_recommendations)


class TestDataExplorerSaveFindings:
    def test_auto_generated_name(self, sample_dataframe):
        with tempfile.TemporaryDirectory() as tmpdir:
            explorer = DataExplorer(
                visualize=False,
                save_findings=True,
                output_dir=tmpdir
            )
            explorer.explore(sample_dataframe)

            # Implementation adds hash: exploration_{hash}_findings.yaml
            saved_files = list(Path(tmpdir).glob("exploration_*_findings.yaml"))
            assert len(saved_files) == 1

    def test_name_from_source_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "my_data.csv"
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

            explorer = DataExplorer(
                visualize=False,
                save_findings=True,
                output_dir=tmpdir
            )
            explorer.explore(str(csv_path))

            # Implementation adds hash: my_data_{hash}_findings.yaml
            saved_files = list(Path(tmpdir).glob("my_data_*_findings.yaml"))
            assert len(saved_files) == 1


class TestDataExplorerVisualization:
    def test_print_text_summary_fallback(self, sample_dataframe, capsys):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        explorer._print_text_summary(findings)
        captured = capsys.readouterr()
        assert "EXPLORATION SUMMARY" in captured.out
        assert "Column Types Detected" in captured.out

    def test_print_text_summary_with_blocking_issues(self, capsys):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df)

        explorer._print_text_summary(findings)
        captured = capsys.readouterr()
        assert "BLOCKING ISSUES" in captured.out
        assert "Modeling Ready: NO" in captured.out
