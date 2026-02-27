from __future__ import annotations

import pandas as pd

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.core.config.column_config import ColumnType


def _findings_with_columns(col_specs: dict[str, ColumnType]) -> ExplorationFindings:
    columns = {
        name: ColumnFinding(
            name=name, inferred_type=ctype, confidence=0.9, evidence=["test"],
        )
        for name, ctype in col_specs.items()
    }
    return ExplorationFindings(
        source_path="test", source_format="csv", columns=columns,
    )


class TestReconcileColumnTypes:

    def test_string_column_marked_numeric_continuous_becomes_categorical(self):
        findings = _findings_with_columns({"lifecycle": ColumnType.NUMERIC_CONTINUOUS})
        df = pd.DataFrame({"lifecycle": ["steady_loyal", "at_risk", "new"]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["lifecycle"].inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert "lifecycle" in corrected

    def test_string_column_marked_numeric_discrete_becomes_categorical(self):
        findings = _findings_with_columns({"status_code": ColumnType.NUMERIC_DISCRETE})
        df = pd.DataFrame({"status_code": ["active", "inactive", "pending"]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["status_code"].inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert "status_code" in corrected

    def test_numeric_column_marked_numeric_is_unchanged(self):
        findings = _findings_with_columns({"score": ColumnType.NUMERIC_CONTINUOUS})
        df = pd.DataFrame({"score": [1.0, 2.5, 3.7]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["score"].inferred_type == ColumnType.NUMERIC_CONTINUOUS
        assert corrected == []

    def test_string_column_marked_categorical_is_unchanged(self):
        findings = _findings_with_columns({"region": ColumnType.CATEGORICAL_NOMINAL})
        df = pd.DataFrame({"region": ["east", "west", "north"]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["region"].inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert corrected == []

    def test_binary_column_with_string_values_is_unchanged(self):
        findings = _findings_with_columns({"flag": ColumnType.BINARY})
        df = pd.DataFrame({"flag": ["yes", "no", "yes"]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["flag"].inferred_type == ColumnType.BINARY
        assert corrected == []

    def test_column_missing_from_df_is_ignored(self):
        findings = _findings_with_columns({"gone": ColumnType.NUMERIC_CONTINUOUS})
        df = pd.DataFrame({"other": [1, 2, 3]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["gone"].inferred_type == ColumnType.NUMERIC_CONTINUOUS
        assert corrected == []

    def test_mixed_columns_only_mismatches_corrected(self):
        findings = _findings_with_columns({
            "lifecycle": ColumnType.NUMERIC_DISCRETE,
            "amount": ColumnType.NUMERIC_CONTINUOUS,
            "category": ColumnType.CATEGORICAL_NOMINAL,
            "label": ColumnType.NUMERIC_CONTINUOUS,
        })
        df = pd.DataFrame({
            "lifecycle": ["steady", "churn", "new"],
            "amount": [100.0, 200.0, 300.0],
            "category": ["a", "b", "c"],
            "label": ["x", "y", "z"],
        })
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["lifecycle"].inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert findings.columns["amount"].inferred_type == ColumnType.NUMERIC_CONTINUOUS
        assert findings.columns["category"].inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert findings.columns["label"].inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert sorted(corrected) == ["label", "lifecycle"]

    def test_confidence_lowered_on_correction(self):
        findings = _findings_with_columns({"col": ColumnType.NUMERIC_CONTINUOUS})
        findings.columns["col"].confidence = 0.9
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        findings.reconcile_column_types(df)
        assert findings.columns["col"].confidence < 0.9

    def test_evidence_updated_on_correction(self):
        findings = _findings_with_columns({"col": ColumnType.NUMERIC_CONTINUOUS})
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        findings.reconcile_column_types(df)
        assert any("dtype mismatch" in e.lower() or "reconcil" in e.lower()
                    for e in findings.columns["col"].evidence)

    def test_identifier_column_is_not_touched(self):
        findings = _findings_with_columns({"id_col": ColumnType.IDENTIFIER})
        df = pd.DataFrame({"id_col": ["abc", "def", "ghi"]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["id_col"].inferred_type == ColumnType.IDENTIFIER
        assert corrected == []

    def test_datetime_column_is_not_touched(self):
        findings = _findings_with_columns({"ts": ColumnType.DATETIME})
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["ts"].inferred_type == ColumnType.DATETIME
        assert corrected == []

    def test_integer_column_marked_numeric_discrete_stays_numeric(self):
        findings = _findings_with_columns({"count": ColumnType.NUMERIC_DISCRETE})
        df = pd.DataFrame({"count": [1, 2, 3, 4, 5]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["count"].inferred_type == ColumnType.NUMERIC_DISCRETE
        assert corrected == []

    def test_empty_dataframe_returns_empty_corrections(self):
        findings = _findings_with_columns({"x": ColumnType.NUMERIC_CONTINUOUS})
        df = pd.DataFrame()
        corrected = findings.reconcile_column_types(df)
        assert corrected == []

    def test_boolean_column_marked_numeric_is_unchanged(self):
        findings = _findings_with_columns({"active": ColumnType.NUMERIC_DISCRETE})
        df = pd.DataFrame({"active": [True, False, True]})
        corrected = findings.reconcile_column_types(df)
        assert findings.columns["active"].inferred_type == ColumnType.NUMERIC_DISCRETE
        assert corrected == []
