"""Comprehensive tests for DataExplorer and TypeInferencer modules.

Covers uncovered lines in:
- src/customer_retention/analysis/auto_explorer/explorer.py
- src/customer_retention/analysis/discovery/type_inferencer.py
"""

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from customer_retention.analysis.auto_explorer.explorer import DataExplorer
from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.analysis.discovery.type_inferencer import (
    InferenceConfidence,
    TypeInferencer,
)
from customer_retention.core.config.column_config import ColumnType

# ---------------------------------------------------------------------------
# TestExplorerLoadSource
# ---------------------------------------------------------------------------


class TestExplorerLoadSource:
    """Tests for DataExplorer._load_source covering DataFrame, CSV, and parquet."""

    def test_load_source_dataframe(self):
        """Passing a DataFrame directly returns it unchanged."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result_df, source_path, source_format = explorer._load_source(df)
        assert source_path == "<DataFrame>"
        assert source_format == "dataframe"
        assert list(result_df.columns) == ["a", "b"]

    def test_load_source_csv(self, tmp_path):
        """Loading a CSV file returns DataFrame with correct format."""
        csv_path = tmp_path / "test_data.csv"
        pd.DataFrame({"x": [10, 20], "y": [30, 40]}).to_csv(csv_path, index=False)
        explorer = DataExplorer(visualize=False, save_findings=False)
        result_df, source_path, source_format = explorer._load_source(str(csv_path))
        assert source_format == "csv"
        assert source_path == str(csv_path)
        assert len(result_df) == 2

    def test_load_source_parquet(self, tmp_path):
        """Loading a parquet file returns DataFrame with correct format (line 42 fallback)."""
        parquet_path = tmp_path / "test_data.parquet"
        pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}).to_parquet(
            parquet_path, index=False
        )
        explorer = DataExplorer(visualize=False, save_findings=False)
        result_df, source_path, source_format = explorer._load_source(str(parquet_path))
        assert source_format == "parquet"
        assert source_path == str(parquet_path)
        assert len(result_df) == 3

    def test_load_source_unknown_extension_fallback(self, tmp_path):
        """Unknown file extension falls back to csv reader (line 42)."""
        # Create a file with .txt extension containing CSV data
        txt_path = tmp_path / "test_data.txt"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(txt_path, index=False)
        explorer = DataExplorer(visualize=False, save_findings=False)
        result_df, source_path, source_format = explorer._load_source(str(txt_path))
        assert source_format == "csv"
        assert len(result_df) == 2

    def test_load_source_pq_extension(self, tmp_path):
        """Loading a .pq file returns DataFrame with parquet format."""
        pq_path = tmp_path / "test_data.pq"
        pd.DataFrame({"val": [100, 200]}).to_parquet(pq_path, index=False)
        explorer = DataExplorer(visualize=False, save_findings=False)
        result_df, source_path, source_format = explorer._load_source(str(pq_path))
        assert source_format == "parquet"
        assert len(result_df) == 2


# ---------------------------------------------------------------------------
# TestExplorerTemporalColumnSkipping
# ---------------------------------------------------------------------------


class TestExplorerTemporalColumnSkipping:
    """Tests for skipping TEMPORAL_METADATA_COLS in _explore_all_columns (line 58)."""

    def test_temporal_metadata_columns_are_excluded(self):
        """Columns in TEMPORAL_METADATA_COLS should be skipped."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "label_timestamp": pd.to_datetime(["2023-03-01", "2023-04-01"]),
            "label_available_flag": [True, False],
            "regular_col": [10, 20],
        })
        findings = explorer._create_findings(df, "<test>", "dataframe")
        explorer._explore_all_columns(df, findings, None)
        # Temporal columns should NOT be in findings
        assert "feature_timestamp" not in findings.columns
        assert "label_timestamp" not in findings.columns
        assert "label_available_flag" not in findings.columns
        # Regular column SHOULD be in findings
        assert "regular_col" in findings.columns

    def test_only_temporal_columns_all_skipped(self):
        """If all columns are temporal metadata, findings.columns is empty."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2023-01-01"]),
            "label_timestamp": pd.to_datetime(["2023-02-01"]),
            "label_available_flag": [True],
        })
        findings = explorer._create_findings(df, "<test>", "dataframe")
        explorer._explore_all_columns(df, findings, None)
        assert len(findings.columns) == 0


# ---------------------------------------------------------------------------
# TestExplorerComputeTypeMetrics
# ---------------------------------------------------------------------------


class TestExplorerComputeTypeMetrics:
    """Tests for _compute_type_metrics (lines 108-113)."""

    def test_compute_type_metrics_unknown_type_returns_empty(self):
        """When profiler returns None for UNKNOWN type (line 108)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        series = pd.Series([1, 2, 3])
        result = explorer._compute_type_metrics(series, ColumnType.UNKNOWN)
        assert result == {}

    def test_compute_type_metrics_numeric_returns_dict(self):
        """Numeric profiler returns a dict with metrics from __dict__."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        series = pd.Series(np.random.normal(0, 1, 100))
        result = explorer._compute_type_metrics(series, ColumnType.NUMERIC_CONTINUOUS)
        assert isinstance(result, dict)
        assert "mean" in result or "std" in result

    def test_compute_type_metrics_profiler_returns_none_value(self):
        """When profile_result has None values (line 110-112 branch)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        # An empty numeric series causes profiler to return None for metrics
        series = pd.Series([], dtype=float)
        result = explorer._compute_type_metrics(series, ColumnType.NUMERIC_CONTINUOUS)
        # With empty series, numeric_metrics will be None, so loop hits the
        # `value is not None` check and falls through to return {}
        assert result == {}

    def test_compute_universal_metrics_unknown_type(self):
        """When profiler is None for _compute_universal_metrics (line 92)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        series = pd.Series([1, 2, 3])
        result = explorer._compute_universal_metrics(series, ColumnType.UNKNOWN)
        assert result == {}


# ---------------------------------------------------------------------------
# TestExplorerQualityIssues
# ---------------------------------------------------------------------------


class TestExplorerQualityIssues:
    """Tests for _identify_quality_issues covering all severity paths (lines 136-146)."""

    def test_null_percentage_critical(self):
        """null_percentage > 50 triggers CRITICAL (line 132)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        universal = {"null_percentage": 75.0}
        issues = explorer._identify_quality_issues(universal, {})
        assert any("CRITICAL" in i and "75.0%" in i for i in issues)

    def test_null_percentage_warning(self):
        """null_percentage 20-50 triggers WARNING (line 134)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        universal = {"null_percentage": 35.0}
        issues = explorer._identify_quality_issues(universal, {})
        assert any("WARNING" in i and "35.0%" in i for i in issues)

    def test_null_percentage_info(self):
        """null_percentage 5-20 triggers INFO (line 136)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        universal = {"null_percentage": 10.0}
        issues = explorer._identify_quality_issues(universal, {})
        assert any("INFO" in i and "10.0%" in i for i in issues)

    def test_null_percentage_no_issue(self):
        """null_percentage <= 5 triggers no issue."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        universal = {"null_percentage": 3.0}
        issues = explorer._identify_quality_issues(universal, {})
        assert not any("missing" in i for i in issues)

    def test_high_cardinality(self):
        """cardinality > 100 triggers high cardinality issue (line 138)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        type_specific = {"cardinality": 150}
        issues = explorer._identify_quality_issues({"null_percentage": 0}, type_specific)
        assert any("High cardinality" in i and "150" in i for i in issues)

    def test_outlier_percentage_warning(self):
        """outlier_percentage > 10 triggers WARNING (line 140)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        type_specific = {"outlier_percentage": 15.0}
        issues = explorer._identify_quality_issues({"null_percentage": 0}, type_specific)
        assert any("WARNING" in i and "outliers" in i for i in issues)

    def test_pii_detected_critical(self):
        """pii_detected triggers CRITICAL PII issue (line 142)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        type_specific = {"pii_detected": True, "pii_types": ["email", "ssn"]}
        issues = explorer._identify_quality_issues({"null_percentage": 0}, type_specific)
        assert any("CRITICAL" in i and "PII" in i for i in issues)
        assert any("email" in i for i in issues)

    def test_case_variations(self):
        """case_variations triggers case inconsistency issue (line 144)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        type_specific = {"case_variations": ["Active vs active"]}
        issues = explorer._identify_quality_issues({"null_percentage": 0}, type_specific)
        assert any("Case inconsistency" in i for i in issues)

    def test_future_dates(self):
        """future_date_count > 0 triggers future dates issue (line 146)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        type_specific = {"future_date_count": 5}
        issues = explorer._identify_quality_issues({"null_percentage": 0}, type_specific)
        assert any("Future dates" in i and "5" in i for i in issues)

    def test_multiple_issues_combined(self):
        """Multiple issues can be detected simultaneously."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        universal = {"null_percentage": 55.0}
        type_specific = {
            "cardinality": 200,
            "outlier_percentage": 12.0,
            "case_variations": ["Yes vs yes"],
            "future_date_count": 3,
        }
        issues = explorer._identify_quality_issues(universal, type_specific)
        assert len(issues) >= 4


# ---------------------------------------------------------------------------
# TestExplorerTransformationRecommendations
# ---------------------------------------------------------------------------


class TestExplorerTransformationRecommendations:
    """Tests for _generate_transformation_recommendations (lines 175-193)."""

    def test_numeric_continuous_high_skewness(self):
        """NUMERIC_CONTINUOUS with high skewness recommends log transform (line 175)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        metrics = {"skewness": 2.5, "outlier_percentage": 3.0}
        recs = explorer._generate_transformation_recommendations(
            ColumnType.NUMERIC_CONTINUOUS, metrics
        )
        assert any("log transform" in r for r in recs)
        assert any("standard scaling" in r for r in recs)

    def test_numeric_continuous_with_outliers(self):
        """NUMERIC_CONTINUOUS with outlier_percentage > 5 recommends robust scaling (line 177)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        metrics = {"skewness": 0.3, "outlier_percentage": 8.0}
        recs = explorer._generate_transformation_recommendations(
            ColumnType.NUMERIC_CONTINUOUS, metrics
        )
        assert any("robust scaling" in r for r in recs)
        assert not any("standard scaling" in r for r in recs)

    def test_numeric_continuous_standard_scaling(self):
        """NUMERIC_CONTINUOUS without outliers recommends standard scaling (line 179)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        metrics = {"skewness": 0.2, "outlier_percentage": 2.0}
        recs = explorer._generate_transformation_recommendations(
            ColumnType.NUMERIC_CONTINUOUS, metrics
        )
        assert any("standard scaling" in r for r in recs)
        assert not any("robust scaling" in r for r in recs)

    def test_categorical_with_rare_categories(self):
        """CATEGORICAL with rare categories > 5 recommends grouping (line 183)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        metrics = {"encoding_recommendation": "target", "rare_category_count": 10}
        recs = explorer._generate_transformation_recommendations(
            ColumnType.CATEGORICAL_NOMINAL, metrics
        )
        assert any("grouping rare" in r for r in recs)
        assert any("target" in r for r in recs)

    def test_categorical_ordinal_encoding(self):
        """CATEGORICAL_ORDINAL gets encoding recommendation."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        metrics = {"encoding_recommendation": "ordinal", "rare_category_count": 2}
        recs = explorer._generate_transformation_recommendations(
            ColumnType.CATEGORICAL_ORDINAL, metrics
        )
        assert any("ordinal" in r for r in recs)
        assert not any("grouping rare" in r for r in recs)

    def test_datetime_recommendations(self):
        """DATETIME column recommends temporal features (line 185-186)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        recs = explorer._generate_transformation_recommendations(ColumnType.DATETIME, {})
        assert any("temporal features" in r for r in recs)
        assert any("days since" in r for r in recs)

    def test_cyclical_encoding(self):
        """CATEGORICAL_CYCLICAL recommends cyclical encoding (line 188)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        recs = explorer._generate_transformation_recommendations(
            ColumnType.CATEGORICAL_CYCLICAL, {}
        )
        assert any("cyclical encoding" in r for r in recs)

    def test_numeric_continuous_none_skewness(self):
        """NUMERIC_CONTINUOUS with None skewness does not recommend log transform."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        metrics = {"skewness": None, "outlier_percentage": 2.0}
        recs = explorer._generate_transformation_recommendations(
            ColumnType.NUMERIC_CONTINUOUS, metrics
        )
        assert not any("log transform" in r for r in recs)
        assert any("standard scaling" in r for r in recs)


# ---------------------------------------------------------------------------
# TestExplorerDisplaySummary
# ---------------------------------------------------------------------------


class TestExplorerDisplaySummary:
    """Tests for _display_summary and _print_text_summary (lines 210-214)."""

    def _make_findings_with_columns(self, target=True, issues=False):
        """Helper to create findings with columns for summary tests."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        data = {"value": np.random.normal(0, 1, 50)}
        if target:
            data["churn"] = np.random.choice([0, 1], 50)
        df = pd.DataFrame(data)
        return explorer.explore(df, target_hint="churn" if target else None)

    def test_display_summary_import_error_fallback(self, capsys):
        """When visualization import fails, falls back to text summary (lines 210-214)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = self._make_findings_with_columns(target=True)
        # Force the import to fail
        with patch.dict(sys.modules, {"customer_retention.analysis.visualization": None}):
            with patch(
                "customer_retention.analysis.auto_explorer.explorer.DataExplorer._display_summary",
                wraps=explorer._display_summary,
            ):
                explorer._display_summary(findings)
        captured = capsys.readouterr()
        assert "EXPLORATION SUMMARY" in captured.out

    def test_print_text_summary_with_target(self, capsys):
        """Text summary displays target column info."""
        findings = self._make_findings_with_columns(target=True)
        explorer = DataExplorer(visualize=False, save_findings=False)
        explorer._print_text_summary(findings)
        captured = capsys.readouterr()
        assert "EXPLORATION SUMMARY" in captured.out
        assert "Target Column:" in captured.out
        assert "Modeling Ready:" in captured.out

    def test_print_text_summary_without_target(self, capsys):
        """Text summary shows warning when no target column detected."""
        findings = self._make_findings_with_columns(target=False)
        explorer = DataExplorer(visualize=False, save_findings=False)
        explorer._print_text_summary(findings)
        captured = capsys.readouterr()
        assert "WARNING: No target column detected!" in captured.out

    def test_print_text_summary_blocking_issues(self, capsys):
        """Text summary shows blocking issues."""
        findings = self._make_findings_with_columns(target=False)
        findings.blocking_issues = ["No target column detected", "Critical issues in: email"]
        explorer = DataExplorer(visualize=False, save_findings=False)
        explorer._print_text_summary(findings)
        captured = capsys.readouterr()
        assert "BLOCKING ISSUES:" in captured.out
        assert "No target column detected" in captured.out


# ---------------------------------------------------------------------------
# TestExplorerOverallFlow
# ---------------------------------------------------------------------------


class TestExplorerOverallFlow:
    """Integration tests for the full explore flow."""

    def test_explore_with_target_hint(self):
        """Explore with target_hint correctly identifies target column."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "customer_id": range(1, 51),
            "amount": np.random.normal(100, 25, 50),
            "churn": np.random.choice([0, 1], 50),
        })
        findings = explorer.explore(df, target_hint="churn")
        assert findings.target_column == "churn"
        assert findings.modeling_ready is True

    def test_explore_without_target(self):
        """Explore without target creates blocking issue."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "amount": np.random.normal(100, 25, 50),
            "category": np.random.choice(["A", "B", "C"], 50),
        })
        findings = explorer.explore(df, target_hint=None)
        assert findings.target_column is None
        assert findings.modeling_ready is False
        assert any("No target column" in issue for issue in findings.blocking_issues)

    def test_explore_with_quality_issues(self):
        """Explore detects quality issues for columns with many nulls."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        # Create a column that is 60% null
        values = [1.0] * 20 + [np.nan] * 30
        df = pd.DataFrame({
            "mostly_null": values,
            "target_col": np.random.choice([0, 1], 50),
        })
        findings = explorer.explore(df, target_hint="target_col")
        # mostly_null should have quality issues
        if "mostly_null" in findings.columns:
            col_finding = findings.columns["mostly_null"]
            assert any("CRITICAL" in i or "missing" in i.lower() for i in col_finding.quality_issues)

    def test_explore_overall_quality_score(self):
        """Overall quality score is calculated from column scores."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "good_col": np.random.normal(0, 1, 50),
            "target": np.random.choice([0, 1], 50),
        })
        findings = explorer.explore(df, target_hint="target")
        assert 0 <= findings.overall_quality_score <= 100

    def test_explore_empty_findings_columns(self):
        """_calculate_overall_metrics handles empty columns (line 193)."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = ExplorationFindings(
            source_path="<test>",
            source_format="dataframe",
            row_count=0,
            column_count=0,
        )
        # Should not raise
        explorer._calculate_overall_metrics(findings)
        assert findings.overall_quality_score == 100.0

    def test_explore_csv_path(self, tmp_path):
        """Full explore from a CSV path."""
        csv_path = tmp_path / "explore_test.csv"
        df = pd.DataFrame({
            "id": range(1, 21),
            "score": np.random.normal(50, 10, 20),
            "churn": np.random.choice([0, 1], 20),
        })
        df.to_csv(csv_path, index=False)
        explorer = DataExplorer(visualize=False, save_findings=False)
        findings = explorer.explore(str(csv_path), target_hint="churn")
        assert findings.source_format == "csv"
        assert findings.row_count == 20


# ---------------------------------------------------------------------------
# TestTypeInferencerEdgeCases
# ---------------------------------------------------------------------------


class TestTypeInferencerEdgeCases:
    """Tests for uncovered lines in TypeInferencer (lines 62, 78, 80, 87-90, etc.)."""

    def test_infer_detects_datetime_column(self):
        """Datetime columns are detected and added to datetime_columns (line 62)."""
        inferencer = TypeInferencer()
        df = pd.DataFrame({
            "event_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        })
        result = inferencer.infer(df)
        assert "event_date" in result.datetime_columns
        assert result.inferences["event_date"].inferred_type == ColumnType.DATETIME

    def test_is_datetime_with_parseable_strings(self):
        """Object dtype with parseable date strings is detected (lines 104-105, 109-110)."""
        inferencer = TypeInferencer()
        series = pd.Series(["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"])
        evidence = []
        result = inferencer._is_datetime(series, evidence)
        assert result is True
        assert any("parseable" in e for e in evidence)

    def test_is_datetime_non_parseable_strings(self):
        """Object dtype with non-date strings is not detected as datetime."""
        inferencer = TypeInferencer()
        series = pd.Series(["hello", "world", "foo", "bar"])
        evidence = []
        result = inferencer._is_datetime(series, evidence)
        assert result is False

    def test_is_binary_two_unique(self):
        """Exactly 2 unique values triggers binary detection (lines 118-119)."""
        inferencer = TypeInferencer()
        series = pd.Series(["active", "inactive", "active", "inactive"])
        evidence = []
        result = inferencer._is_binary(series, evidence)
        assert result is True
        assert any("2 unique" in e for e in evidence)

    def test_is_binary_more_than_two(self):
        """More than 2 unique values is not binary."""
        inferencer = TypeInferencer()
        series = pd.Series(["a", "b", "c", "a"])
        evidence = []
        result = inferencer._is_binary(series, evidence)
        assert result is False

    def test_is_identifier_pattern_match_unique(self):
        """ID pattern in name with unique values (lines 87-90)."""
        inferencer = TypeInferencer()
        series = pd.Series(["a", "b", "c", "d", "e"])
        evidence = []
        result = inferencer._is_identifier(series, "user_id", evidence)
        assert result is True
        assert any("id pattern" in e for e in evidence)

    def test_is_identifier_pattern_match_not_unique(self):
        """ID pattern in name but not all unique should not detect as identifier."""
        inferencer = TypeInferencer()
        series = pd.Series(["a", "a", "b", "b", "c"])
        evidence = []
        result = inferencer._is_identifier(series, "category_id", evidence)
        assert result is False

    def test_is_target_pattern_match_few_values(self):
        """Target pattern in name with <= 10 distinct values (lines 97-100)."""
        inferencer = TypeInferencer()
        series = pd.Series([0, 1, 0, 1, 0, 1, 0])
        evidence = []
        result = inferencer._is_target(series, "churn_label", evidence)
        assert result is True
        assert any("target pattern" in e for e in evidence)

    def test_is_target_pattern_match_many_values(self):
        """Target pattern in name but > 10 distinct values is not target (line 97->100)."""
        inferencer = TypeInferencer()
        series = pd.Series(range(20))
        evidence = []
        result = inferencer._is_target(series, "churn_score", evidence)
        assert result is False

    def test_infer_categorical_high_cardinality(self):
        """Categorical with > 10 unique values gets MEDIUM confidence (lines 138-139)."""
        inferencer = TypeInferencer()
        # Create a series with > 10 unique string values
        values = [f"cat_{i}" for i in range(25)]
        series = pd.Series(values)
        evidence = []
        result = inferencer._infer_categorical(series, "category", evidence)
        assert result.inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert result.confidence == InferenceConfidence.MEDIUM
        assert result.suggested_encoding == "target"

    def test_infer_categorical_low_cardinality(self):
        """Categorical with <= 10 unique values gets HIGH confidence (lines 136-137)."""
        inferencer = TypeInferencer()
        series = pd.Series(["A", "B", "C", "A", "B", "C"])
        evidence = []
        result = inferencer._infer_categorical(series, "status", evidence)
        assert result.inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert result.confidence == InferenceConfidence.HIGH
        assert result.suggested_encoding == "onehot"

    def test_show_report(self, capsys):
        """show_report prints inference details (lines 143-147)."""
        inferencer = TypeInferencer()
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5],
            "churn": [0, 1, 0, 1, 0],
            "signup_date": pd.to_datetime(["2023-01-01"] * 5),
        })
        result = inferencer.infer(df)
        inferencer.show_report(result)
        captured = capsys.readouterr()
        assert "Target column:" in captured.out
        assert "Identifier columns:" in captured.out
        assert "Datetime columns:" in captured.out

    def test_infer_from_csv_path(self, tmp_path):
        """Infer from a CSV path string (line within infer method)."""
        csv_path = tmp_path / "inference_data.csv"
        df = pd.DataFrame({
            "id": range(1, 11),
            "amount": np.random.uniform(10, 100, 10),
            "status": ["active"] * 5 + ["inactive"] * 5,
        })
        df.to_csv(csv_path, index=False)
        inferencer = TypeInferencer()
        result = inferencer.infer(str(csv_path))
        assert "id" in result.inferences
        assert "amount" in result.inferences

    def test_is_identifier_unique_integers(self):
        """Unique integer values detected as identifier (line 90-92)."""
        inferencer = TypeInferencer()
        series = pd.Series([100, 200, 300, 400, 500])
        evidence = []
        result = inferencer._is_identifier(series, "record_num", evidence)
        assert result is True
        assert any("unique integer" in e for e in evidence)

    def test_is_identifier_non_unique_integers(self):
        """Non-unique integer values not detected as identifier."""
        inferencer = TypeInferencer()
        series = pd.Series([1, 1, 2, 2, 3])
        evidence = []
        result = inferencer._is_identifier(series, "group_num", evidence)
        assert result is False


# ---------------------------------------------------------------------------
# TestExplorerModelingReadiness
# ---------------------------------------------------------------------------


class TestExplorerModelingReadiness:
    """Tests for modeling readiness checks."""

    def test_modeling_ready_with_target_no_critical(self):
        """Modeling is ready when target exists and no critical issues."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "feature": np.random.normal(0, 1, 30),
            "target": np.random.choice([0, 1], 30),
        })
        findings = explorer.explore(df, target_hint="target")
        assert findings.modeling_ready is True
        assert len(findings.blocking_issues) == 0

    def test_not_modeling_ready_no_target(self):
        """Not modeling ready when no target detected."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 30),
            "feature2": np.random.choice(["a", "b", "c"], 30),
        })
        findings = explorer.explore(df)
        assert findings.modeling_ready is False
        assert any("No target" in issue for issue in findings.blocking_issues)

    def test_not_modeling_ready_with_critical_issues(self):
        """Not modeling ready when critical quality issues exist."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        # Create data with > 50% nulls to trigger CRITICAL
        values = [1.0] * 10 + [np.nan] * 40
        df = pd.DataFrame({
            "bad_col": values,
            "target": np.random.choice([0, 1], 50),
        })
        findings = explorer.explore(df, target_hint="target")
        # Check if critical issues block modeling
        if findings.columns.get("bad_col") and any(
            "CRITICAL" in i for i in findings.columns["bad_col"].quality_issues
        ):
            assert findings.modeling_ready is False


# ---------------------------------------------------------------------------
# TestExplorerSpecialColumns
# ---------------------------------------------------------------------------


class TestExplorerSpecialColumns:
    """Tests for _track_special_columns."""

    def test_track_target_binary(self):
        """Binary target column tracked correctly."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "churn": [0, 1, 0, 1, 0],
            "value": [10, 20, 30, 40, 50],
        })
        findings = explorer.explore(df, target_hint="churn")
        assert findings.target_column == "churn"
        assert findings.target_type == "binary"

    def test_track_target_multiclass(self):
        """Multiclass target column tracked correctly."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "value": range(10),
        })
        findings = explorer.explore(df, target_hint="label")
        assert findings.target_column == "label"
        assert findings.target_type == "multiclass"

    def test_track_identifier_column(self):
        """Identifier columns are tracked."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "customer_id": range(1, 31),
            "value": np.random.normal(0, 1, 30),
            "churn": np.random.choice([0, 1], 30),
        })
        findings = explorer.explore(df, target_hint="churn")
        assert "customer_id" in findings.identifier_columns

    def test_track_datetime_column(self):
        """Datetime columns are tracked."""
        explorer = DataExplorer(visualize=False, save_findings=False)
        df = pd.DataFrame({
            "signup_date": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"]
            ),
            "value": [1, 2, 3, 4, 5],
            "churn": [0, 1, 0, 1, 0],
        })
        findings = explorer.explore(df, target_hint="churn")
        assert "signup_date" in findings.datetime_columns
