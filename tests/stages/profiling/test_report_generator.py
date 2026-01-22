import pytest
import json
import pandas as pd
from pathlib import Path
from customer_retention.stages.profiling import ReportGenerator, ProfileResult, ColumnProfile, TypeInference
from customer_retention.stages.profiling.profile_result import UniversalMetrics, NumericMetrics, CategoricalMetrics
from customer_retention.core.config import ColumnType


@pytest.fixture
def sample_profile_result():
    """Create a sample ProfileResult for testing."""
    column_profiles = {
        "numeric_col": ColumnProfile(
            column_name="numeric_col",
            configured_type=ColumnType.NUMERIC_CONTINUOUS,
            inferred_type=TypeInference(
                inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                confidence="high",
                evidence=["numeric dtype", "continuous values"]
            ),
            universal_metrics=UniversalMetrics(
                total_count=100,
                null_count=5,
                null_percentage=5.0,
                distinct_count=95,
                distinct_percentage=95.0
            ),
            numeric_metrics=NumericMetrics(
                mean=100.0,
                std=15.0,
                min_value=50.0,
                max_value=150.0,
                range_value=100.0,
                median=100.0,
                q1=90.0,
                q3=110.0,
                iqr=20.0,
                skewness=0.5,
                kurtosis=0.2,
                zero_count=0,
                zero_percentage=0.0,
                negative_count=0,
                negative_percentage=0.0,
                inf_count=0,
                inf_percentage=0.0,
                outlier_count_iqr=5,
                outlier_count_zscore=3,
                outlier_percentage=5.0
            )
        ),
        "categorical_col": ColumnProfile(
            column_name="categorical_col",
            configured_type=ColumnType.CATEGORICAL_NOMINAL,
            inferred_type=TypeInference(
                inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                confidence="high",
                evidence=["string dtype", "low cardinality"]
            ),
            universal_metrics=UniversalMetrics(
                total_count=100,
                null_count=0,
                null_percentage=0.0,
                distinct_count=4,
                distinct_percentage=4.0
            ),
            categorical_metrics=CategoricalMetrics(
                cardinality=4,
                cardinality_ratio=0.04,
                value_counts={"A": 40, "B": 30, "C": 20, "D": 10},
                top_categories=[("A", 40), ("B", 30), ("C", 20), ("D", 10)],
                rare_categories=["D"],
                rare_category_count=1,
                rare_category_percentage=10.0,
                contains_unknown=False
            )
        )
    }

    return ProfileResult(
        dataset_name="test_dataset",
        total_rows=100,
        total_columns=2,
        column_profiles=column_profiles,
        profiling_timestamp="2024-01-01T00:00:00",
        profiling_duration_seconds=1.5
    )


class TestReportGeneratorBasic:
    def test_generator_initialization(self):
        generator = ReportGenerator()
        assert generator is not None

    def test_generator_with_profile(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        assert generator.profile == sample_profile_result


class TestJSONReportGeneration:
    def test_to_json(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        json_report = generator.to_json()

        assert json_report is not None
        assert isinstance(json_report, str)

        # Verify it's valid JSON
        parsed = json.loads(json_report)
        assert parsed is not None

    def test_json_contains_dataset_info(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        json_report = generator.to_json()
        parsed = json.loads(json_report)

        assert "dataset_name" in parsed
        assert parsed["dataset_name"] == "test_dataset"
        assert "total_rows" in parsed
        assert parsed["total_rows"] == 100

    def test_json_contains_column_profiles(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        json_report = generator.to_json()
        parsed = json.loads(json_report)

        assert "column_profiles" in parsed
        assert "numeric_col" in parsed["column_profiles"]
        assert "categorical_col" in parsed["column_profiles"]

    def test_save_json_to_file(self, sample_profile_result, tmp_path):
        generator = ReportGenerator(sample_profile_result)
        output_file = tmp_path / "report.json"

        generator.save_json(str(output_file))

        assert output_file.exists()

        with open(output_file, 'r') as f:
            content = json.load(f)
            assert content["dataset_name"] == "test_dataset"


class TestHTMLReportGeneration:
    def test_to_html(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        html_report = generator.to_html()

        assert html_report is not None
        assert isinstance(html_report, str)
        assert "<html" in html_report.lower()

    def test_html_contains_dataset_name(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        html_report = generator.to_html()

        assert "test_dataset" in html_report

    def test_html_contains_column_sections(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        html_report = generator.to_html()

        assert "numeric_col" in html_report
        assert "categorical_col" in html_report

    def test_html_contains_executive_summary(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        html_report = generator.to_html()

        assert "executive" in html_report.lower() or "summary" in html_report.lower()
        assert "100" in html_report  # total rows

    def test_save_html_to_file(self, sample_profile_result, tmp_path):
        generator = ReportGenerator(sample_profile_result)
        output_file = tmp_path / "report.html"

        generator.save_html(str(output_file))

        assert output_file.exists()

        with open(output_file, 'r') as f:
            content = f.read()
            assert "<html" in content.lower()


class TestMarkdownReportGeneration:
    def test_to_markdown(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        md_report = generator.to_markdown()

        assert md_report is not None
        assert isinstance(md_report, str)
        assert "#" in md_report  # Markdown headers

    def test_markdown_contains_dataset_info(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        md_report = generator.to_markdown()

        assert "test_dataset" in md_report
        assert "100" in md_report  # total rows

    def test_markdown_contains_column_sections(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        md_report = generator.to_markdown()

        assert "numeric_col" in md_report
        assert "categorical_col" in md_report

    def test_markdown_table_format(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        md_report = generator.to_markdown()

        assert "|" in md_report  # Markdown table separator
        assert "---" in md_report  # Table header separator

    def test_save_markdown_to_file(self, sample_profile_result, tmp_path):
        generator = ReportGenerator(sample_profile_result)
        output_file = tmp_path / "report.md"

        generator.save_markdown(str(output_file))

        assert output_file.exists()

        with open(output_file, 'r') as f:
            content = f.read()
            assert "#" in content


class TestExecutiveSummary:
    def test_executive_summary_generation(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        summary = generator.generate_executive_summary()

        assert summary is not None
        assert isinstance(summary, dict)

    def test_summary_contains_dataset_overview(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        summary = generator.generate_executive_summary()

        assert "total_rows" in summary
        assert "total_columns" in summary
        assert summary["total_rows"] == 100
        assert summary["total_columns"] == 2

    def test_summary_contains_type_breakdown(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        summary = generator.generate_executive_summary()

        assert "column_types" in summary
        assert "numeric_continuous" in summary["column_types"] or ColumnType.NUMERIC_CONTINUOUS.value in str(summary)

    def test_summary_contains_quality_metrics(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        summary = generator.generate_executive_summary()

        assert "missing_percentage" in summary or "null" in str(summary).lower()

    def test_summary_calculates_quality_score(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)
        summary = generator.generate_executive_summary()

        assert "quality_score" in summary
        assert 0 <= summary["quality_score"] <= 100


class TestCorrelationMatrix:
    def test_calculate_correlations_with_numeric_data(self):
        column_profiles = {
            "col1": ColumnProfile(
                column_name="col1",
                configured_type=ColumnType.NUMERIC_CONTINUOUS,
                inferred_type=TypeInference(
                    inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence="high",
                    evidence=[]
                ),
                universal_metrics=UniversalMetrics(
                    total_count=100, null_count=0, null_percentage=0.0,
                    distinct_count=100, distinct_percentage=100.0
                )
            ),
            "col2": ColumnProfile(
                column_name="col2",
                configured_type=ColumnType.NUMERIC_CONTINUOUS,
                inferred_type=TypeInference(
                    inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence="high",
                    evidence=[]
                ),
                universal_metrics=UniversalMetrics(
                    total_count=100, null_count=0, null_percentage=0.0,
                    distinct_count=100, distinct_percentage=100.0
                )
            )
        }

        profile = ProfileResult(
            dataset_name="test",
            total_rows=100,
            total_columns=2,
            column_profiles=column_profiles,
            profiling_timestamp="2024-01-01T00:00:00",
            profiling_duration_seconds=1.0
        )

        # Create actual data for correlation
        df = pd.DataFrame({
            "col1": range(100),
            "col2": range(100)
        })

        generator = ReportGenerator(profile)
        correlations = generator.calculate_correlations(df)

        assert correlations is not None
        assert isinstance(correlations, dict)

    def test_correlations_only_numeric_columns(self):
        column_profiles = {
            "numeric": ColumnProfile(
                column_name="numeric",
                configured_type=ColumnType.NUMERIC_CONTINUOUS,
                inferred_type=TypeInference(
                    inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence="high",
                    evidence=[]
                ),
                universal_metrics=UniversalMetrics(
                    total_count=100, null_count=0, null_percentage=0.0,
                    distinct_count=100, distinct_percentage=100.0
                )
            ),
            "categorical": ColumnProfile(
                column_name="categorical",
                configured_type=ColumnType.CATEGORICAL_NOMINAL,
                inferred_type=TypeInference(
                    inferred_type=ColumnType.CATEGORICAL_NOMINAL,
                    confidence="high",
                    evidence=[]
                ),
                universal_metrics=UniversalMetrics(
                    total_count=100, null_count=0, null_percentage=0.0,
                    distinct_count=4, distinct_percentage=4.0
                )
            )
        }

        profile = ProfileResult(
            dataset_name="test",
            total_rows=100,
            total_columns=2,
            column_profiles=column_profiles,
            profiling_timestamp="2024-01-01T00:00:00",
            profiling_duration_seconds=1.0
        )

        df = pd.DataFrame({
            "numeric": range(100),
            "categorical": ["A", "B", "C", "D"] * 25
        })

        generator = ReportGenerator(profile)
        correlations = generator.calculate_correlations(df)

        # Should only include numeric column
        if correlations:
            assert "categorical" not in str(correlations) or len(correlations) == 0


class TestReportFormats:
    def test_all_formats_generate_successfully(self, sample_profile_result):
        generator = ReportGenerator(sample_profile_result)

        json_report = generator.to_json()
        html_report = generator.to_html()
        md_report = generator.to_markdown()

        assert json_report is not None
        assert html_report is not None
        assert md_report is not None

    def test_generate_all_reports_to_directory(self, sample_profile_result, tmp_path):
        generator = ReportGenerator(sample_profile_result)

        generator.save_all_formats(str(tmp_path), "test_report")

        assert (tmp_path / "test_report.json").exists()
        assert (tmp_path / "test_report.html").exists()
        assert (tmp_path / "test_report.md").exists()
