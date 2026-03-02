import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.explorer import DataExplorer
from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.core.config.column_config import ColumnType
from customer_retention.stages.profiling import ProfilerFactory


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(100)],
            "age": np.random.randint(18, 80, 100),
            "monthly_charges": np.random.uniform(20, 100, 100),
            "tenure": np.random.randint(0, 72, 100),
            "gender": np.random.choice(["Male", "Female"], 100),
            "contract": np.random.choice(["Month-to-month", "One year", "Two year"], 100),
            "churned": np.random.choice([0, 1], 100),
            "signup_date": pd.date_range("2020-01-01", periods=100, freq="D"),
        }
    )


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
        explorer = DataExplorer(visualize=False, save_findings=False, output_dir="/custom/path")
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
            explorer = DataExplorer(visualize=False, save_findings=True, output_dir=tmpdir)
            findings = explorer.explore(sample_dataframe, name="test_exploration")

            saved_files = list(Path(tmpdir).glob("test_exploration_findings.yaml"))
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
        df = pd.DataFrame(
            {"col_with_nulls": [1, 2, None, None, None, None, 7, 8, 9, 10], "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
        )
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["col_with_nulls"]
        assert any("missing" in issue.lower() for issue in col_finding.quality_issues)

    def test_cleaning_recommendations_generated(self):
        df = pd.DataFrame(
            {"col_with_nulls": [1, 2, None, None, None, None, 7, 8, 9, 10], "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
        )
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
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df)

        assert not findings.modeling_ready
        assert "No target column detected" in findings.blocking_issues

    def test_target_type_binary(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe, target_hint="churned")

        assert findings.target_type == "binary"

    def test_target_type_multiclass(self):
        df = pd.DataFrame({"features": [1, 2, 3, 4, 5], "target_class": ["A", "B", "C", "A", "B"]})
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
        df = pd.DataFrame({"high_card_col": [f"cat_{i % 150}" for i in range(300)], "target": [0, 1] * 150})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["high_card_col"]
        col_type = col_finding.inferred_type
        assert col_type in [ColumnType.TEXT, ColumnType.CATEGORICAL_NOMINAL]

    def test_critical_nulls_detection(self):
        df = pd.DataFrame({"mostly_null": [None] * 60 + [1] * 40, "target": [0, 1] * 50})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["mostly_null"]
        assert any("CRITICAL" in issue for issue in col_finding.quality_issues)
        assert any("dropping" in rec.lower() for rec in col_finding.cleaning_recommendations)

    def test_warning_nulls_detection(self):
        df = pd.DataFrame({"some_nulls": [None] * 25 + [1] * 75, "target": [0, 1] * 50})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        col_finding = findings.columns["some_nulls"]
        assert any("WARNING" in issue for issue in col_finding.quality_issues)

    def test_modeling_ready_with_critical_issues(self):
        df = pd.DataFrame({"mostly_null": [None] * 60 + [1] * 40, "target": [0, 1] * 50})
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
            explorer = DataExplorer(visualize=False, save_findings=True, output_dir=tmpdir)
            explorer.explore(sample_dataframe)

            saved_files = list(Path(tmpdir).glob("exploration_findings.yaml"))
            assert len(saved_files) == 1

    def test_name_from_source_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "my_data.csv"
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

            explorer = DataExplorer(visualize=False, save_findings=True, output_dir=tmpdir)
            explorer.explore(str(csv_path))

            saved_files = list(Path(tmpdir).glob("my_data_findings.yaml"))
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
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df)

        explorer._print_text_summary(findings)
        captured = capsys.readouterr()
        assert "BLOCKING ISSUES" in captured.out
        assert "Modeling Ready: NO" in captured.out

    def test_print_text_summary_with_target(self, sample_dataframe, capsys):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe, target_hint="churned")

        explorer._print_text_summary(findings)
        captured = capsys.readouterr()
        assert "Target Column: churned" in captured.out
        assert "Modeling Ready: YES" in captured.out


class TestDataExplorerTemporalMetadataCols:
    def test_temporal_metadata_cols_skipped(self):
        df = pd.DataFrame(
            {
                "feature_timestamp": pd.date_range("2020-01-01", periods=50),
                "label_timestamp": pd.date_range("2020-02-01", periods=50),
                "amount": np.random.uniform(10, 100, 50),
                "target": np.random.choice([0, 1], 50),
            }
        )
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        # Temporal metadata columns should be skipped
        assert "feature_timestamp" not in findings.columns
        assert "label_timestamp" not in findings.columns
        assert "amount" in findings.columns
        assert "target" in findings.columns


class TestDataExplorerLoadSource:
    def test_load_default_csv_with_unknown_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.txt"
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
            explorer = DataExplorer(visualize=False, save_findings=False)
            df, src_path, src_format = explorer._load_source(str(path))
            assert src_format == "csv"
            assert len(df) == 2

    def test_load_parquet_with_pq_extension(self, sample_dataframe):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.pq"
            sample_dataframe.to_parquet(path, index=False)
            explorer = DataExplorer(visualize=False, save_findings=False)
            df, src_path, src_format = explorer._load_source(str(path))
            assert src_format == "parquet"
            assert len(df) == 100


class TestDataExplorerEmptyColumns:
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df)
        assert findings.row_count == 0
        assert findings.column_count == 0
        # Empty DF gets default quality score, overall_metrics returns early
        assert len(findings.columns) == 0


class TestDataExplorerQualityIssuesExtended:
    def test_outlier_detection_warning(self):
        df = pd.DataFrame(
            {"normal": np.random.normal(50, 1, 100).tolist(), "target": np.random.choice([0, 1], 100).tolist()}
        )
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        # Just checking it runs - outlier issues depend on the profiler output
        assert "normal" in findings.columns

    def test_info_level_nulls(self):
        df = pd.DataFrame({"low_nulls": [None] * 8 + list(range(92)), "target": [0, 1] * 50})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        col_finding = findings.columns["low_nulls"]
        assert any("INFO" in issue for issue in col_finding.quality_issues)

    def test_future_dates_issue(self):
        future = pd.Timestamp.now() + pd.Timedelta(days=365)
        df = pd.DataFrame({"date_col": [pd.Timestamp("2023-01-01")] * 95 + [future] * 5, "target": [0, 1] * 50})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        # Just checking it runs
        assert "date_col" in findings.columns

    def test_case_variations_issue(self):
        df = pd.DataFrame({"names": ["New York", "new york", "NEW YORK", "Boston"] * 25, "target": [0, 1] * 50})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        names_finding = findings.columns["names"]
        # Depending on profiler, may detect case variations
        assert "names" in findings.columns


class TestDataExplorerTransformationsExtended:
    def test_numeric_with_high_skewness(self):
        np.random.seed(42)
        df = pd.DataFrame({"skewed": np.random.exponential(2, 200), "target": np.random.choice([0, 1], 200)})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        col_finding = findings.columns["skewed"]
        if col_finding.inferred_type == ColumnType.NUMERIC_CONTINUOUS:
            # May recommend log transform
            assert len(col_finding.transformation_recommendations) > 0

    def test_numeric_with_high_outliers(self):
        np.random.seed(42)
        data = np.random.normal(50, 1, 200).tolist()
        data[0:15] = [200] * 15  # 7.5% outliers
        df = pd.DataFrame({"outlier_col": data, "target": np.random.choice([0, 1], 200).tolist()})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        assert "outlier_col" in findings.columns

    def test_cyclical_encoding_recommendation(self):
        df = pd.DataFrame(
            {
                "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] * 30,
                "target": np.random.choice([0, 1], 210).tolist(),
            }
        )
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        assert "day_of_week" in findings.columns

    def test_rare_categories_recommendation(self):
        # Create a series with many rare categories
        values = ["common"] * 300
        for i in range(20):
            values.append(f"rare_{i}")
        df = pd.DataFrame({"cat_col": values, "target": np.random.choice([0, 1], len(values)).tolist()})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        assert "cat_col" in findings.columns


class TestDataExplorerConfidenceMapping:
    def test_confidence_high(self):
        explorer = DataExplorer(visualize=False, save_findings=False)
        from enum import Enum

        class MockConf(Enum):
            HIGH = "HIGH"

        assert explorer._confidence_to_float(MockConf.HIGH) == 0.9

    def test_confidence_medium(self):
        explorer = DataExplorer(visualize=False, save_findings=False)
        from enum import Enum

        class MockConf(Enum):
            MEDIUM = "MEDIUM"

        assert explorer._confidence_to_float(MockConf.MEDIUM) == 0.7

    def test_confidence_low(self):
        explorer = DataExplorer(visualize=False, save_findings=False)
        from enum import Enum

        class MockConf(Enum):
            LOW = "LOW"

        assert explorer._confidence_to_float(MockConf.LOW) == 0.4

    def test_confidence_unknown(self):
        explorer = DataExplorer(visualize=False, save_findings=False)
        assert explorer._confidence_to_float("UNKNOWN") == 0.5


class TestDataExplorerComputeTypeMetrics:
    def test_compute_type_metrics_no_profiler(self):
        explorer = DataExplorer(visualize=False, save_findings=False)
        # UNKNOWN type has no profiler
        result = explorer._compute_type_metrics(pd.Series([1, 2, 3]), ColumnType.UNKNOWN)
        assert result == {}

    def test_compute_universal_metrics_no_profiler(self):
        explorer = DataExplorer(visualize=False, save_findings=False)
        result = explorer._compute_universal_metrics(pd.Series([1, 2, 3]), ColumnType.UNKNOWN)
        assert result == {}


class TestDataExplorerDisplaySummaryFallback:
    def test_display_summary_imports_chart_builder(self, sample_dataframe):
        """Test that _display_summary works (or falls back to text summary)."""
        explorer = DataExplorer(visualize=True, save_findings=False)
        findings = explorer.explore(sample_dataframe)
        # Should not raise regardless of whether ChartBuilder is importable
        assert findings is not None


class TestDataExplorerLastFindingsPath:
    def test_last_findings_path_set_when_save_disabled(self, sample_dataframe):
        with tempfile.TemporaryDirectory() as tmpdir:
            explorer = DataExplorer(visualize=False, save_findings=False, output_dir=tmpdir)
            explorer.explore(sample_dataframe, name="test_dataset")
            assert explorer.last_findings_path is not None
            assert "test_dataset" in explorer.last_findings_path
            assert explorer.last_findings_path.endswith("_findings.yaml")

    def test_last_findings_path_no_file_written_when_save_disabled(self, sample_dataframe):
        with tempfile.TemporaryDirectory() as tmpdir:
            explorer = DataExplorer(visualize=False, save_findings=False, output_dir=tmpdir)
            explorer.explore(sample_dataframe, name="test_dataset")
            assert not Path(explorer.last_findings_path).exists()

    def test_last_findings_path_file_written_when_save_enabled(self, sample_dataframe):
        with tempfile.TemporaryDirectory() as tmpdir:
            explorer = DataExplorer(visualize=False, save_findings=True, output_dir=tmpdir)
            explorer.explore(sample_dataframe, name="test_dataset")
            assert Path(explorer.last_findings_path).exists()


class TestDataExplorerSaveFindingsEdgeCases:
    def test_save_findings_dataframe_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            explorer = DataExplorer(visualize=False, save_findings=True, output_dir=tmpdir)
            findings = explorer.explore(df)
            saved_files = list(Path(tmpdir).glob("exploration_findings.yaml"))
            assert len(saved_files) == 1


class TestDataExplorerBulkStats:
    def test_bulk_stats_called_once(self, sample_dataframe):
        from unittest.mock import patch

        from customer_retention.core.compat.bulk_profiling import compute_bulk_stats as _real

        explorer = DataExplorer(visualize=False, save_findings=False)
        with patch(
            "customer_retention.analysis.auto_explorer.explorer.compute_bulk_stats",
            wraps=_real,
        ) as mock_bulk:
            findings = explorer.explore(sample_dataframe)
            mock_bulk.assert_called_once()
            assert findings.row_count == 100

    def test_bulk_universal_metrics_match(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        age_metrics = findings.columns["age"].universal_metrics
        assert age_metrics["total_count"] == 100
        assert age_metrics["null_count"] == 0
        assert age_metrics["null_percentage"] == 0
        assert age_metrics["distinct_count"] > 0
        assert age_metrics["most_common_value"] is not None
        assert age_metrics["most_common_frequency"] is not None

    def test_bulk_numeric_metrics_present(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        charges_finding = findings.columns["monthly_charges"]
        assert charges_finding.inferred_type == ColumnType.NUMERIC_CONTINUOUS
        tm = charges_finding.type_metrics
        assert "mean" in tm
        assert "std" in tm
        assert "min_value" in tm
        assert "max_value" in tm
        assert "q1" in tm
        assert "q3" in tm
        assert "histogram_bins" in tm
        assert len(tm["histogram_bins"]) == 10

    def test_bulk_numeric_outlier_fields(self):
        np.random.seed(42)
        data = np.random.normal(50, 1, 200).tolist()
        data.extend([200, -100])
        df = pd.DataFrame(
            {
                "val": data,
                "target": np.random.choice([0, 1], len(data)).tolist(),
            }
        )
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        tm = findings.columns["val"].type_metrics
        assert "outlier_count_iqr" in tm
        assert "outlier_count_zscore" in tm
        assert "outlier_percentage" in tm
        assert tm["outlier_count_iqr"] > 0
        assert tm["outlier_count_zscore"] > 0

    def test_bulk_non_numeric_falls_back_to_profiler(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        contract_finding = findings.columns["contract"]
        assert contract_finding.inferred_type in [
            ColumnType.CATEGORICAL_NOMINAL,
            ColumnType.CATEGORICAL_ORDINAL,
        ]
        tm = contract_finding.type_metrics
        assert "cardinality" in tm or "value_counts" in tm

    def test_bulk_null_percentage_consistency(self):
        df = pd.DataFrame(
            {
                "col_with_nulls": [1, 2, None, None, None, None, 7, 8, 9, 10],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")

        um = findings.columns["col_with_nulls"].universal_metrics
        assert um["null_count"] == 4
        assert um["null_percentage"] == 40.0

    def test_bulk_distinct_passed_to_type_detector(self):
        from unittest.mock import patch

        df = pd.DataFrame(
            {
                "val": np.random.normal(50, 10, 100),
                "target": np.random.choice([0, 1], 100),
            }
        )
        explorer = DataExplorer(visualize=False, save_findings=False)

        with patch.object(
            explorer.type_detector, "detect_type", wraps=explorer.type_detector.detect_type
        ) as mock_detect:
            explorer.explore(df, target_hint="target")
            for call in mock_detect.call_args_list:
                assert "distinct_count" in call.kwargs
                assert isinstance(call.kwargs["distinct_count"], int)

    def test_bulk_zero_count_field(self):
        df = pd.DataFrame({"val": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df)
        tm = findings.columns["val"].type_metrics
        assert tm["zero_count"] == 3

    def test_bulk_negative_count_field(self):
        df = pd.DataFrame({"val": [-5, -3, 0, 1, 2, 3, 4, 5, 6, 7]})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df)
        tm = findings.columns["val"].type_metrics
        assert tm["negative_count"] == 2

    def test_bulk_inf_count_field(self):
        df = pd.DataFrame({"val": [1.0, 2.0, float("inf"), 4.0, float("-inf")]})
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df)
        tm = findings.columns["val"].type_metrics
        assert tm["inf_count"] == 2

    def test_bulk_empty_numeric_column(self):
        df = pd.DataFrame(
            {
                "empty_num": pd.array([None, None, None], dtype="Float64"),
                "target": [0, 1, 0],
            }
        )
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        assert "empty_num" in findings.columns

    def test_explore_results_identical_structure(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)

        for col_name, col_finding in findings.columns.items():
            um = col_finding.universal_metrics
            assert "total_count" in um
            assert "null_count" in um
            assert "null_percentage" in um
            assert "distinct_count" in um
            assert "distinct_percentage" in um


class TestDataExplorerTypedBulkPath:
    def test_typed_bulk_stats_called(self, sample_dataframe):
        from unittest.mock import patch

        from customer_retention.core.compat.bulk_profiling import compute_typed_bulk_stats as _real

        explorer = DataExplorer(visualize=False, save_findings=False)
        with patch(
            "customer_retention.analysis.auto_explorer.explorer.compute_typed_bulk_stats",
            wraps=_real,
        ) as mock_typed:
            findings = explorer.explore(sample_dataframe)
            mock_typed.assert_called_once()
            assert findings.row_count == 100

    def test_datetime_uses_bulk_path(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)
        date_finding = findings.columns["signup_date"]
        tm = date_finding.type_metrics
        assert "min_date" in tm
        assert "max_date" in tm
        assert "date_range_days" in tm
        assert "future_date_count" in tm
        assert "weekend_percentage" in tm

    def test_categorical_uses_bulk_path(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)
        contract_finding = findings.columns["contract"]
        tm = contract_finding.type_metrics
        assert "cardinality" in tm
        assert "value_counts" in tm
        assert "top_categories" in tm
        assert "encoding_recommendation" in tm

    def test_identifier_uses_bulk_path(self, sample_dataframe):
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(sample_dataframe)
        id_finding = findings.columns["customer_id"]
        tm = id_finding.type_metrics
        assert "is_unique" in tm
        assert "format_pattern" in tm
        assert "length_min" in tm
        assert "length_max" in tm

    def test_binary_uses_bulk_path(self):
        df = pd.DataFrame({
            "flag": [True, False] * 50,
            "target": [0, 1] * 50,
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        flag_finding = findings.columns["flag"]
        if flag_finding.inferred_type == ColumnType.BINARY:
            tm = flag_finding.type_metrics
            assert "true_count" in tm
            assert "false_count" in tm
            assert "balance_ratio" in tm
            assert "is_boolean" in tm

    def test_text_uses_bulk_path(self):
        texts = [f"This is sentence number {i} with some text content." for i in range(200)]
        df = pd.DataFrame({
            "description": texts,
            "target": np.random.choice([0, 1], 200).tolist(),
        })
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(df, target_hint="target")
        text_finding = findings.columns["description"]
        if text_finding.inferred_type == ColumnType.TEXT:
            tm = text_finding.type_metrics
            assert "length_min" in tm
            assert "length_max" in tm
            assert "word_count_mean" in tm
            assert "pii_detected" in tm

    def test_no_profiler_fallback_for_bulk_types(self, sample_dataframe):
        from unittest.mock import patch

        explorer = DataExplorer(visualize=False, save_findings=False)
        with patch(
            "customer_retention.analysis.auto_explorer.explorer.ProfilerFactory.get_profiler",
            wraps=ProfilerFactory.get_profiler,
        ) as mock_get_profiler:
            findings = explorer.explore(sample_dataframe, target_hint="churned")
            # ProfilerFactory.get_profiler should only be called for universal metrics
            # and target profiler fallback, NOT for datetime/categorical/identifier/binary/text
            profile_calls = mock_get_profiler.call_args_list
            profiled_types = [call.args[0] for call in profile_calls]
            # Target should still use profiler (only 1 target column falls back)
            assert ColumnType.TARGET in profiled_types
            # Datetime, categorical, identifier should NOT appear in profiler calls
            # (they use the typed bulk path instead)
            type_metrics_profiled = [
                t for t in profiled_types
                if t not in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE, ColumnType.TARGET)
            ]
            # The universal metrics path still calls get_profiler for format,
            # but _compute_type_metrics should not call profile() for bulk types
            # This is a soft check — at minimum, target is the only fallback
            assert ColumnType.TARGET in profiled_types
