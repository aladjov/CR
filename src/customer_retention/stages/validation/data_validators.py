"""
Data validators for exploratory data analysis.

This module provides reusable validation functions for data quality assessment,
including duplicate detection, date logic validation, and value range validation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from customer_retention.core.compat import DataFrame, pd
from customer_retention.core.components.enums import Severity


@dataclass
class DuplicateResult:
    """Result of duplicate analysis."""
    key_column: str
    total_rows: int
    unique_keys: int
    duplicate_keys: int
    duplicate_rows: int
    duplicate_percentage: float
    has_value_conflicts: bool
    conflict_columns: List[str] = field(default_factory=list)
    conflict_examples: List[Dict[str, Any]] = field(default_factory=list)
    exact_duplicate_rows: int = 0
    severity: Severity = Severity.INFO

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "key_column": self.key_column,
            "total_rows": self.total_rows,
            "unique_keys": self.unique_keys,
            "duplicate_keys": self.duplicate_keys,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_percentage": round(self.duplicate_percentage, 2),
            "has_value_conflicts": self.has_value_conflicts,
            "conflict_columns": self.conflict_columns,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "severity": self.severity.value
        }


@dataclass
class DateLogicResult:
    """Result of date logic validation."""
    date_columns: List[str]
    total_rows: int
    valid_rows: int
    invalid_rows: int
    invalid_percentage: float
    violations: List[Dict[str, Any]] = field(default_factory=list)
    violation_types: Dict[str, int] = field(default_factory=dict)
    severity: Severity = Severity.INFO

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "date_columns": self.date_columns,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "invalid_percentage": round(self.invalid_percentage, 2),
            "violation_types": self.violation_types,
            "severity": self.severity.value
        }


@dataclass
class RangeValidationResult:
    """Result of value range validation."""
    column_name: str
    total_values: int
    valid_values: int
    invalid_values: int
    invalid_percentage: float
    rule_type: str
    expected_range: str
    actual_range: str
    invalid_examples: List[Any] = field(default_factory=list)
    severity: Severity = Severity.INFO

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "column": self.column_name,
            "rule_type": self.rule_type,
            "expected_range": self.expected_range,
            "actual_range": self.actual_range,
            "invalid_count": self.invalid_values,
            "invalid_percentage": round(self.invalid_percentage, 2),
            "severity": self.severity.value
        }


class DataValidator:
    """
    Validator for data quality checks in exploratory analysis.

    Provides methods for duplicate detection, date logic validation,
    and value range validation.
    """

    def check_duplicates(
        self,
        df: DataFrame,
        key_column: str,
        check_value_conflicts: bool = True,
        exclude_columns: Optional[List[str]] = None
    ) -> DuplicateResult:
        """
        Comprehensive duplicate analysis with conflict detection.

        Parameters
        ----------
        df : DataFrame
            Data to analyze
        key_column : str
            Column to check for duplicates (e.g., customer ID)
        check_value_conflicts : bool
            Whether to check if duplicate keys have different values
        exclude_columns : List[str], optional
            Columns to exclude from conflict checking

        Returns
        -------
        DuplicateResult
            Detailed analysis of duplicates and conflicts
        """
        if key_column not in df.columns:
            return DuplicateResult(
                key_column=key_column,
                total_rows=len(df),
                unique_keys=0,
                duplicate_keys=0,
                duplicate_rows=0,
                duplicate_percentage=0.0,
                has_value_conflicts=False,
                severity=Severity.CRITICAL
            )

        total_rows = len(df)
        unique_keys = df[key_column].nunique()
        duplicate_mask = df[key_column].duplicated(keep=False)
        duplicate_rows = duplicate_mask.sum()
        duplicate_keys = df[duplicate_mask][key_column].nunique()
        duplicate_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0

        # Check for exact duplicate rows
        exact_duplicate_rows = df.duplicated(keep=False).sum()

        # Determine severity based on duplicate percentage
        if duplicate_percentage > 10:
            severity = Severity.CRITICAL
        elif duplicate_percentage > 5:
            severity = Severity.WARNING
        elif duplicate_percentage > 0:
            severity = Severity.INFO
        else:
            severity = Severity.INFO

        # Check for value conflicts
        has_value_conflicts = False
        conflict_columns = []
        conflict_examples = []

        if check_value_conflicts and duplicate_keys > 0:
            exclude = set(exclude_columns or [])
            exclude.add(key_column)
            value_columns = [c for c in df.columns if c not in exclude]

            duplicated_keys = df[duplicate_mask][key_column].unique()
            sample_keys = duplicated_keys[:5]  # Check up to 5 duplicate keys

            for key_value in sample_keys:
                key_rows = df[df[key_column] == key_value]
                for col in value_columns:
                    unique_vals = key_rows[col].dropna().unique()
                    if len(unique_vals) > 1:
                        has_value_conflicts = True
                        if col not in conflict_columns:
                            conflict_columns.append(col)
                        if len(conflict_examples) < 3:
                            conflict_examples.append({
                                "key": key_value,
                                "column": col,
                                "values": unique_vals[:5].tolist()
                            })

            # Value conflicts are additional concern - only increase severity, never decrease
            if has_value_conflicts and severity == Severity.INFO:
                severity = Severity.WARNING

        return DuplicateResult(
            key_column=key_column,
            total_rows=total_rows,
            unique_keys=unique_keys,
            duplicate_keys=duplicate_keys,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
            has_value_conflicts=has_value_conflicts,
            conflict_columns=conflict_columns,
            conflict_examples=conflict_examples,
            exact_duplicate_rows=exact_duplicate_rows,
            severity=severity
        )

    def validate_date_logic(
        self,
        df: DataFrame,
        date_columns: List[str],
        expected_order: Optional[List[str]] = None
    ) -> DateLogicResult:
        """
        Validate temporal consistency of date fields.

        Parameters
        ----------
        df : DataFrame
            Data to validate
        date_columns : List[str]
            List of date column names in expected chronological order
        expected_order : List[str], optional
            Explicit order of dates (if different from date_columns order)

        Returns
        -------
        DateLogicResult
            Detailed analysis of date logic violations
        """
        # Filter to columns that exist
        existing_cols = [c for c in date_columns if c in df.columns]

        if len(existing_cols) < 2:
            return DateLogicResult(
                date_columns=existing_cols,
                total_rows=len(df),
                valid_rows=len(df),
                invalid_rows=0,
                invalid_percentage=0.0,
                severity=Severity.INFO
            )

        order = expected_order if expected_order else existing_cols
        order = [c for c in order if c in existing_cols]

        # Convert to datetime if needed
        df_dates = df[order].copy()
        for col in order:
            if not pd.api.types.is_datetime64_any_dtype(df_dates[col]):
                df_dates[col] = pd.to_datetime(df_dates[col], errors='coerce', format='mixed')

        # Check sequential ordering
        violations = []
        violation_types = {}
        invalid_mask = pd.Series(False, index=df.index)

        for i in range(len(order) - 1):
            col1, col2 = order[i], order[i + 1]
            # col1 should be <= col2
            invalid = df_dates[col1] > df_dates[col2]
            # Exclude rows where either is NaT
            valid_comparison = df_dates[col1].notna() & df_dates[col2].notna()
            invalid = invalid & valid_comparison

            if invalid.any():
                violation_key = f"{col1} > {col2}"
                violation_count = invalid.sum()
                violation_types[violation_key] = int(violation_count)
                invalid_mask = invalid_mask | invalid

                # Sample violations
                if len(violations) < 5:
                    sample_idx = df[invalid].head(3).index
                    for idx in sample_idx:
                        violations.append({
                            "row": int(idx),
                            "violation": violation_key,
                            col1: str(df_dates.loc[idx, col1]),
                            col2: str(df_dates.loc[idx, col2])
                        })

        total_rows = len(df)
        invalid_rows = int(invalid_mask.sum())
        valid_rows = total_rows - invalid_rows
        invalid_percentage = (invalid_rows / total_rows * 100) if total_rows > 0 else 0.0

        # Determine severity
        if invalid_percentage > 10:
            severity = Severity.CRITICAL
        elif invalid_percentage > 5:
            severity = Severity.WARNING
        elif invalid_percentage > 0:
            severity = Severity.INFO
        else:
            severity = Severity.INFO

        return DateLogicResult(
            date_columns=order,
            total_rows=total_rows,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            invalid_percentage=invalid_percentage,
            violations=violations,
            violation_types=violation_types,
            severity=severity
        )

    def validate_value_ranges(
        self,
        df: DataFrame,
        rules: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[RangeValidationResult]:
        """
        Validate logical ranges for numeric fields.

        Parameters
        ----------
        df : DataFrame
            Data to validate
        rules : Dict[str, Dict[str, Any]], optional
            Custom validation rules. If None, uses default rules.
            Format: {"column": {"type": "percentage|binary|non_negative", "min": 0, "max": 100}}

        Returns
        -------
        List[RangeValidationResult]
            Validation results for each rule
        """
        results = []

        if rules is None:
            rules = self._infer_default_rules(df)

        for col_name, rule in rules.items():
            if col_name not in df.columns:
                continue

            series = df[col_name].dropna()
            total_values = len(series)

            if total_values == 0:
                continue

            rule_type = rule.get("type", "range")
            min_val = rule.get("min")
            max_val = rule.get("max")

            if rule_type == "percentage":
                min_val = min_val if min_val is not None else 0
                max_val = max_val if max_val is not None else 100
                invalid_mask = (series < min_val) | (series > max_val)
                expected_range = f"[{min_val}, {max_val}]"
            elif rule_type == "binary":
                valid_values = rule.get("valid_values", [0, 1])
                invalid_mask = ~series.isin(valid_values)
                expected_range = str(valid_values)
            elif rule_type == "non_negative":
                invalid_mask = series < 0
                expected_range = "[0, +∞)"
            else:  # general range
                invalid_mask = pd.Series(False, index=series.index)
                if min_val is not None:
                    invalid_mask = invalid_mask | (series < min_val)
                if max_val is not None:
                    invalid_mask = invalid_mask | (series > max_val)
                expected_range = f"[{min_val or '-∞'}, {max_val or '+∞'}]"

            invalid_values = int(invalid_mask.sum())
            valid_values = total_values - invalid_values
            invalid_percentage = (invalid_values / total_values * 100) if total_values > 0 else 0.0

            # Get actual range
            actual_min = float(series.min())
            actual_max = float(series.max())
            actual_range = f"[{actual_min:.2f}, {actual_max:.2f}]"

            # Get invalid examples
            invalid_examples = series[invalid_mask].head(5).tolist() if invalid_values > 0 else []

            # Determine severity
            if invalid_percentage > 10:
                severity = Severity.CRITICAL
            elif invalid_percentage > 5:
                severity = Severity.WARNING
            elif invalid_percentage > 0:
                severity = Severity.INFO
            else:
                severity = Severity.INFO

            results.append(RangeValidationResult(
                column_name=col_name,
                total_values=total_values,
                valid_values=valid_values,
                invalid_values=invalid_values,
                invalid_percentage=invalid_percentage,
                rule_type=rule_type,
                expected_range=expected_range,
                actual_range=actual_range,
                invalid_examples=invalid_examples,
                severity=severity
            ))

        return results

    def _infer_default_rules(self, df: DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Infer default validation rules based on column names.

        Parameters
        ----------
        df : DataFrame
            Data to analyze

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Inferred validation rules
        """
        rules = {}

        for col in df.columns:
            col_lower = col.lower()

            # Percentage columns (rates, percentages)
            if any(pattern in col_lower for pattern in ['rate', 'pct', 'percent', 'ratio']):
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Check if it's 0-1 scale or 0-100 scale
                    max_val = df[col].max()
                    if max_val <= 1.0:
                        rules[col] = {"type": "percentage", "min": 0, "max": 1}
                    else:
                        rules[col] = {"type": "percentage", "min": 0, "max": 100}

            # Binary columns
            elif df[col].nunique() == 2:
                unique_vals = df[col].dropna().unique()
                if set(unique_vals).issubset({0, 1, True, False, 0.0, 1.0}):
                    rules[col] = {"type": "binary", "valid_values": [0, 1]}

            # Count/amount columns (non-negative)
            elif any(pattern in col_lower for pattern in ['count', 'amount', 'quantity', 'num_', 'n_']):
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    rules[col] = {"type": "non_negative"}

        return rules

    def validate_all(
        self,
        df: DataFrame,
        key_column: Optional[str] = None,
        date_columns: Optional[List[str]] = None,
        range_rules: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run all validations and return comprehensive results.

        Parameters
        ----------
        df : DataFrame
            Data to validate
        key_column : str, optional
            Column to check for duplicates
        date_columns : List[str], optional
            Date columns to validate
        range_rules : Dict[str, Dict[str, Any]], optional
            Custom range validation rules

        Returns
        -------
        Dict[str, Any]
            Comprehensive validation results
        """
        results = {
            "duplicates": None,
            "date_logic": None,
            "range_validations": [],
            "overall_severity": Severity.INFO
        }

        severities = []

        if key_column:
            dup_result = self.check_duplicates(df, key_column)
            results["duplicates"] = dup_result.to_dict()
            severities.append(dup_result.severity)

        if date_columns:
            date_result = self.validate_date_logic(df, date_columns)
            results["date_logic"] = date_result.to_dict()
            severities.append(date_result.severity)

        range_results = self.validate_value_ranges(df, range_rules)
        results["range_validations"] = [r.to_dict() for r in range_results]
        severities.extend([r.severity for r in range_results])

        # Determine overall severity (highest)
        severity_order = [
            Severity.INFO,
            Severity.WARNING,
            Severity.CRITICAL,
            Severity.CRITICAL
        ]
        if severities:
            results["overall_severity"] = max(severities, key=lambda s: severity_order.index(s)).value
        else:
            results["overall_severity"] = Severity.INFO.value

        return results
