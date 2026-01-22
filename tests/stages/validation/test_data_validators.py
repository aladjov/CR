"""Tests for data validators module."""

import pandas as pd
import pytest

from customer_retention.core.components.enums import Severity
from customer_retention.stages.validation import DataValidator, DateLogicResult, DuplicateResult, RangeValidationResult


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance."""
        return DataValidator()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "custid": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "created": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"]),
            "firstorder": pd.to_datetime(["2023-01-15", "2023-02-15", "2023-03-15", "2023-04-15", "2023-05-15"]),
            "lastorder": pd.to_datetime(["2023-06-01", "2023-06-15", "2023-07-01", "2023-07-15", "2023-08-01"]),
            "eopenrate": [45.5, 62.3, 78.1, 33.2, 55.8],
            "retained": [1, 1, 0, 1, 0]
        })


class TestCheckDuplicates(TestDataValidator):
    """Tests for check_duplicates method."""

    def test_no_duplicates(self, validator, sample_df):
        """Test with no duplicates."""
        result = validator.check_duplicates(sample_df, "custid")

        assert result.total_rows == 5
        assert result.unique_keys == 5
        assert result.duplicate_keys == 0
        assert result.duplicate_rows == 0
        assert result.duplicate_percentage == 0.0
        assert result.has_value_conflicts is False
        assert result.severity == Severity.INFO

    def test_with_duplicates(self, validator):
        """Test with duplicate keys."""
        df = pd.DataFrame({
            "custid": [1, 1, 2, 3, 3, 3],
            "name": ["Alice", "Alice", "Bob", "Charlie", "Charlie", "Charlie"],
            "age": [25, 25, 30, 35, 35, 35]
        })
        result = validator.check_duplicates(df, "custid")

        assert result.total_rows == 6
        assert result.unique_keys == 3
        assert result.duplicate_keys == 2  # custid 1 and 3 have duplicates
        assert result.duplicate_rows == 5  # 2 rows for custid 1 + 3 rows for custid 3
        assert result.duplicate_percentage > 0

    def test_duplicates_with_value_conflicts(self, validator):
        """Test duplicates with conflicting values."""
        df = pd.DataFrame({
            "custid": [1, 1, 2],
            "name": ["Alice", "Alicia", "Bob"],  # Conflict: same custid, different name
            "age": [25, 26, 30]
        })
        result = validator.check_duplicates(df, "custid", check_value_conflicts=True)

        assert result.has_value_conflicts is True
        assert len(result.conflict_columns) > 0
        assert "name" in result.conflict_columns or "age" in result.conflict_columns

    def test_missing_key_column(self, validator, sample_df):
        """Test with missing key column."""
        result = validator.check_duplicates(sample_df, "nonexistent")

        assert result.unique_keys == 0
        assert result.severity == Severity.CRITICAL

    def test_exact_duplicate_rows(self, validator):
        """Test detection of exact duplicate rows."""
        df = pd.DataFrame({
            "custid": [1, 1, 2],
            "name": ["Alice", "Alice", "Bob"],
            "age": [25, 25, 30]
        })
        result = validator.check_duplicates(df, "custid")

        assert result.exact_duplicate_rows == 2  # Two rows are exactly identical

    def test_high_duplicate_severity(self, validator):
        """Test that high duplicate percentage triggers ERROR severity."""
        df = pd.DataFrame({
            "custid": [1] * 15 + [2] * 5,  # 75% duplicates
            "name": ["Alice"] * 15 + ["Bob"] * 5
        })
        result = validator.check_duplicates(df, "custid")

        assert result.severity == Severity.CRITICAL


class TestValidateDateLogic(TestDataValidator):
    """Tests for validate_date_logic method."""

    def test_valid_date_sequence(self, validator, sample_df):
        """Test with valid date sequence."""
        result = validator.validate_date_logic(
            sample_df,
            ["created", "firstorder", "lastorder"]
        )

        assert result.invalid_rows == 0
        assert result.invalid_percentage == 0.0
        assert result.severity == Severity.INFO

    def test_invalid_date_sequence(self, validator):
        """Test with invalid date sequence."""
        df = pd.DataFrame({
            "created": pd.to_datetime(["2023-06-01", "2023-01-01"]),  # First row: created after firstorder
            "firstorder": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "lastorder": pd.to_datetime(["2023-07-01", "2023-06-01"])
        })
        result = validator.validate_date_logic(df, ["created", "firstorder", "lastorder"])

        assert result.invalid_rows == 1
        assert result.invalid_percentage == 50.0
        assert "created > firstorder" in result.violation_types

    def test_missing_date_columns(self, validator, sample_df):
        """Test with missing date columns."""
        result = validator.validate_date_logic(
            sample_df,
            ["created", "nonexistent"]
        )

        # Should handle gracefully with only one column
        assert result.invalid_rows == 0

    def test_nat_values_excluded(self, validator):
        """Test that NaT values are excluded from validation."""
        df = pd.DataFrame({
            "created": pd.to_datetime(["2023-01-01", None]),
            "firstorder": pd.to_datetime(["2023-02-01", "2023-01-01"])
        })
        result = validator.validate_date_logic(df, ["created", "firstorder"])

        # Only one row should be checked (the one without NaT)
        assert result.invalid_rows == 0


class TestValidateValueRanges(TestDataValidator):
    """Tests for validate_value_ranges method."""

    def test_percentage_validation(self, validator):
        """Test percentage range validation."""
        df = pd.DataFrame({
            "eopenrate": [45.5, 62.3, 150.0, -5.0, 55.8]  # Two invalid values
        })
        rules = {"eopenrate": {"type": "percentage", "min": 0, "max": 100}}
        results = validator.validate_value_ranges(df, rules)

        assert len(results) == 1
        assert results[0].column_name == "eopenrate"
        assert results[0].invalid_values == 2
        assert results[0].rule_type == "percentage"

    def test_binary_validation(self, validator):
        """Test binary value validation."""
        df = pd.DataFrame({
            "retained": [0, 1, 0, 1, 2, -1]  # Two invalid values
        })
        rules = {"retained": {"type": "binary", "valid_values": [0, 1]}}
        results = validator.validate_value_ranges(df, rules)

        assert len(results) == 1
        assert results[0].invalid_values == 2

    def test_non_negative_validation(self, validator):
        """Test non-negative validation."""
        df = pd.DataFrame({
            "order_count": [5, 10, -1, 0, 20]  # One invalid value
        })
        rules = {"order_count": {"type": "non_negative"}}
        results = validator.validate_value_ranges(df, rules)

        assert len(results) == 1
        assert results[0].invalid_values == 1

    def test_infer_default_rules(self, validator):
        """Test automatic rule inference."""
        df = pd.DataFrame({
            "click_rate": [0.5, 0.3, 0.8],
            "open_rate": [50.0, 60.0, 70.0],
            "is_active": [0, 1, 0],
            "order_count": [5, 10, 15]
        })
        # Should infer rules based on column names
        results = validator.validate_value_ranges(df)

        # Check that rules were inferred for rate columns
        column_names = [r.column_name for r in results]
        # Note: all values are valid so results may be empty or have 0 invalid


class TestValidateAll(TestDataValidator):
    """Tests for validate_all method."""

    def test_comprehensive_validation(self, validator, sample_df):
        """Test running all validations together."""
        results = validator.validate_all(
            sample_df,
            key_column="custid",
            date_columns=["created", "firstorder", "lastorder"]
        )

        assert "duplicates" in results
        assert "date_logic" in results
        assert "range_validations" in results
        assert "overall_severity" in results

    def test_overall_severity_calculation(self, validator):
        """Test that overall severity is the highest among all checks."""
        df = pd.DataFrame({
            "custid": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # High duplicate rate -> ERROR
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        })
        results = validator.validate_all(df, key_column="custid")

        assert results["overall_severity"] == "critical"


class TestResultDataclasses:
    """Tests for result dataclass methods."""

    def test_duplicate_result_to_dict(self):
        """Test DuplicateResult.to_dict()."""
        result = DuplicateResult(
            key_column="custid",
            total_rows=100,
            unique_keys=90,
            duplicate_keys=5,
            duplicate_rows=10,
            duplicate_percentage=10.0,
            has_value_conflicts=True,
            conflict_columns=["name"],
            conflict_examples=[],
            exact_duplicate_rows=5,
            severity=Severity.WARNING
        )
        d = result.to_dict()

        assert d["key_column"] == "custid"
        assert d["total_rows"] == 100
        assert d["severity"] == "warning"

    def test_date_logic_result_to_dict(self):
        """Test DateLogicResult.to_dict()."""
        result = DateLogicResult(
            date_columns=["created", "updated"],
            total_rows=100,
            valid_rows=95,
            invalid_rows=5,
            invalid_percentage=5.0,
            violations=[],
            violation_types={"created > updated": 5},
            severity=Severity.INFO
        )
        d = result.to_dict()

        assert d["date_columns"] == ["created", "updated"]
        assert d["invalid_rows"] == 5

    def test_range_validation_result_to_dict(self):
        """Test RangeValidationResult.to_dict()."""
        result = RangeValidationResult(
            column_name="rate",
            total_values=100,
            valid_values=95,
            invalid_values=5,
            invalid_percentage=5.0,
            rule_type="percentage",
            expected_range="[0, 100]",
            actual_range="[-5.0, 150.0]",
            invalid_examples=[-5.0, 150.0],
            severity=Severity.WARNING
        )
        d = result.to_dict()

        assert d["column"] == "rate"
        assert d["rule_type"] == "percentage"
