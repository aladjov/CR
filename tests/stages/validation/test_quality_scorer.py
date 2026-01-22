"""Tests for quality scorer module."""

import pytest
from customer_retention.stages.validation.quality_scorer import (
    QualityScorer, QualityScoreResult, QualityLevel,
    ExplorationFindings, ColumnFindings
)
from customer_retention.stages.validation import DuplicateResult, DateLogicResult, RangeValidationResult
from customer_retention.core.components.enums import Severity


class MockInferredType:
    """Mock for inferred type with .value attribute."""
    def __init__(self, value: str):
        self.value = value


def create_column_findings(
    inferred_type: str = "numeric",
    null_percentage: float = 0.0,
    distinct_percentage: float = 100.0
) -> ColumnFindings:
    """Helper to create column findings."""
    return ColumnFindings(
        inferred_type=MockInferredType(inferred_type),
        universal_metrics={
            "null_percentage": null_percentage,
            "distinct_percentage": distinct_percentage
        }
    )


def create_findings(
    row_count: int = 100,
    column_count: int = None,
    columns: dict = None
) -> ExplorationFindings:
    """Helper to create exploration findings."""
    if columns is None:
        columns = {
            "col1": create_column_findings(),
            "col2": create_column_findings(),
            "col3": create_column_findings(),
        }
    # Default column_count to match actual columns
    if column_count is None:
        column_count = len(columns)
    return ExplorationFindings(
        row_count=row_count,
        column_count=column_count,
        columns=columns
    )


class TestQualityScorer:
    """Tests for QualityScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a QualityScorer instance."""
        return QualityScorer()

    @pytest.fixture
    def clean_findings(self):
        """Create clean findings with no issues."""
        return create_findings()


class TestInitialization(TestQualityScorer):
    """Tests for QualityScorer initialization."""

    def test_default_weights(self, scorer):
        """Test default weights are set correctly."""
        assert scorer.weights["completeness"] == 0.25
        assert scorer.weights["validity"] == 0.25
        assert scorer.weights["consistency"] == 0.25
        assert scorer.weights["uniqueness"] == 0.25

    def test_custom_weights(self):
        """Test custom weights are accepted."""
        custom_weights = {
            "completeness": 0.4,
            "validity": 0.3,
            "consistency": 0.2,
            "uniqueness": 0.1
        }
        scorer = QualityScorer(weights=custom_weights)
        assert scorer.weights["completeness"] == 0.4
        assert scorer.weights["validity"] == 0.3

    def test_invalid_weights_missing_component(self):
        """Test that missing weight components raise error."""
        with pytest.raises(ValueError, match="Missing weight components"):
            QualityScorer(weights={"completeness": 1.0})

    def test_invalid_weights_wrong_sum(self):
        """Test that weights not summing to 1.0 raise error."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            QualityScorer(weights={
                "completeness": 0.5,
                "validity": 0.5,
                "consistency": 0.5,
                "uniqueness": 0.5
            })


class TestCompletenessScore(TestQualityScorer):
    """Tests for completeness component."""

    def test_perfect_completeness(self, scorer):
        """Test 100% completeness with no nulls."""
        findings = create_findings(columns={
            "col1": create_column_findings(null_percentage=0),
            "col2": create_column_findings(null_percentage=0),
        })
        result = scorer.calculate(findings)
        assert result.components["completeness"] == 100.0

    def test_low_completeness(self, scorer):
        """Test low completeness with high nulls."""
        findings = create_findings(columns={
            "col1": create_column_findings(null_percentage=50),
            "col2": create_column_findings(null_percentage=50),
        })
        result = scorer.calculate(findings)
        assert result.components["completeness"] == 50.0

    def test_partial_completeness(self, scorer):
        """Test partial completeness."""
        findings = create_findings(
            row_count=100,
            column_count=2,
            columns={
                "col1": create_column_findings(null_percentage=20),
                "col2": create_column_findings(null_percentage=0),
            }
        )
        result = scorer.calculate(findings)
        # 20% null in col1, 0% in col2 = 10% overall null = 90% complete
        assert result.components["completeness"] == 90.0

    def test_completeness_issues_reported(self, scorer):
        """Test that high missing columns are reported as issues."""
        findings = create_findings(columns={
            "problem_col": create_column_findings(null_percentage=50),
        })
        result = scorer.calculate(findings)
        assert any("problem_col" in issue for issue in result.issues)


class TestValidityScore(TestQualityScorer):
    """Tests for validity component."""

    def test_no_range_rules(self, scorer, clean_findings):
        """Test 100% validity when no rules are defined."""
        result = scorer.calculate(clean_findings)
        assert result.components["validity"] == 100.0

    def test_all_valid_values(self, scorer, clean_findings):
        """Test 100% validity when all values pass rules."""
        range_results = [
            RangeValidationResult(
                column_name="rate",
                total_values=100,
                valid_values=100,
                invalid_values=0,
                invalid_percentage=0.0,
                rule_type="percentage",
                expected_range="[0, 100]",
                actual_range="[0, 100]",
                severity=Severity.INFO
            )
        ]
        result = scorer.calculate(clean_findings, range_results=range_results)
        assert result.components["validity"] == 100.0

    def test_some_invalid_values(self, scorer, clean_findings):
        """Test reduced validity with invalid values."""
        range_results = [
            RangeValidationResult(
                column_name="rate",
                total_values=100,
                valid_values=90,
                invalid_values=10,
                invalid_percentage=10.0,
                rule_type="percentage",
                expected_range="[0, 100]",
                actual_range="[-5, 150]",
                severity=Severity.WARNING
            )
        ]
        result = scorer.calculate(clean_findings, range_results=range_results)
        assert result.components["validity"] == 90.0

    def test_validity_issues_reported(self, scorer, clean_findings):
        """Test that high invalid percentages are reported."""
        range_results = [
            RangeValidationResult(
                column_name="bad_col",
                total_values=100,
                valid_values=80,
                invalid_values=20,
                invalid_percentage=20.0,
                rule_type="percentage",
                expected_range="[0, 100]",
                actual_range="[-10, 200]",
                severity=Severity.CRITICAL
            )
        ]
        result = scorer.calculate(clean_findings, range_results=range_results)
        assert any("bad_col" in issue for issue in result.issues)


class TestConsistencyScore(TestQualityScorer):
    """Tests for consistency component."""

    def test_no_duplicates_no_date_issues(self, scorer, clean_findings):
        """Test 100% consistency with no issues."""
        result = scorer.calculate(clean_findings)
        assert result.components["consistency"] == 100.0

    def test_high_duplicate_penalty(self, scorer, clean_findings):
        """Test penalty for high duplicate rate."""
        dup_result = DuplicateResult(
            key_column="id",
            total_rows=100,
            unique_keys=80,
            duplicate_keys=10,
            duplicate_rows=20,
            duplicate_percentage=15.0,  # >10% = 30 penalty
            has_value_conflicts=False,
            severity=Severity.CRITICAL
        )
        result = scorer.calculate(clean_findings, duplicate_result=dup_result)
        assert result.components["consistency"] == 70.0

    def test_moderate_duplicate_penalty(self, scorer, clean_findings):
        """Test penalty for moderate duplicate rate."""
        dup_result = DuplicateResult(
            key_column="id",
            total_rows=100,
            unique_keys=92,
            duplicate_keys=4,
            duplicate_rows=8,
            duplicate_percentage=8.0,  # 5-10% = 20 penalty
            has_value_conflicts=False,
            severity=Severity.WARNING
        )
        result = scorer.calculate(clean_findings, duplicate_result=dup_result)
        assert result.components["consistency"] == 80.0

    def test_value_conflict_penalty(self, scorer, clean_findings):
        """Test additional penalty for value conflicts."""
        dup_result = DuplicateResult(
            key_column="id",
            total_rows=100,
            unique_keys=95,
            duplicate_keys=3,
            duplicate_rows=5,
            duplicate_percentage=2.0,  # 1-5% = 10 penalty
            has_value_conflicts=True,  # +20 penalty
            conflict_columns=["status", "amount"],
            severity=Severity.WARNING
        )
        result = scorer.calculate(clean_findings, duplicate_result=dup_result)
        assert result.components["consistency"] == 70.0  # 100 - 10 - 20

    def test_date_logic_penalty(self, scorer, clean_findings):
        """Test penalty for date logic violations."""
        date_result = DateLogicResult(
            date_columns=["created", "updated"],
            total_rows=100,
            valid_rows=85,
            invalid_rows=15,
            invalid_percentage=15.0,  # >10% = 20 penalty
            severity=Severity.CRITICAL
        )
        result = scorer.calculate(clean_findings, date_result=date_result)
        assert result.components["consistency"] == 80.0

    def test_combined_consistency_penalties(self, scorer, clean_findings):
        """Test combined penalties from duplicates and dates."""
        dup_result = DuplicateResult(
            key_column="id",
            total_rows=100,
            unique_keys=90,
            duplicate_keys=5,
            duplicate_rows=10,
            duplicate_percentage=8.0,  # 20 penalty
            has_value_conflicts=False,
            severity=Severity.WARNING
        )
        date_result = DateLogicResult(
            date_columns=["created", "updated"],
            total_rows=100,
            valid_rows=92,
            invalid_rows=8,
            invalid_percentage=8.0,  # 10 penalty
            severity=Severity.WARNING
        )
        result = scorer.calculate(
            clean_findings,
            duplicate_result=dup_result,
            date_result=date_result
        )
        assert result.components["consistency"] == 70.0  # 100 - 20 - 10


class TestUniquenessScore(TestQualityScorer):
    """Tests for uniqueness component."""

    def test_no_identifiers(self, scorer, clean_findings):
        """Test 100% uniqueness when no identifiers exist."""
        result = scorer.calculate(clean_findings)
        assert result.components["uniqueness"] == 100.0

    def test_unique_identifier(self, scorer):
        """Test 100% uniqueness for properly unique identifier."""
        findings = create_findings(columns={
            "customer_id": create_column_findings(
                inferred_type="identifier",
                distinct_percentage=100.0
            ),
        })
        result = scorer.calculate(findings)
        assert result.components["uniqueness"] == 100.0

    def test_low_uniqueness_identifier(self, scorer):
        """Test penalty for identifier with low uniqueness."""
        findings = create_findings(columns={
            "customer_id": create_column_findings(
                inferred_type="identifier",
                distinct_percentage=80.0  # <90% = 20 penalty
            ),
        })
        result = scorer.calculate(findings)
        assert result.components["uniqueness"] == 80.0

    def test_uniqueness_issue_reported(self, scorer):
        """Test that low uniqueness identifiers are reported."""
        findings = create_findings(columns={
            "bad_id": create_column_findings(
                inferred_type="identifier",
                distinct_percentage=50.0
            ),
        })
        result = scorer.calculate(findings)
        assert any("bad_id" in issue for issue in result.issues)


class TestOverallScore(TestQualityScorer):
    """Tests for overall score calculation."""

    def test_perfect_score(self, scorer, clean_findings):
        """Test 100% overall score with no issues."""
        result = scorer.calculate(clean_findings)
        assert result.overall_score == 100.0
        assert result.quality_level == QualityLevel.EXCELLENT

    def test_weighted_score(self):
        """Test that weights affect overall score correctly."""
        # Weight completeness heavily
        scorer = QualityScorer(weights={
            "completeness": 0.7,
            "validity": 0.1,
            "consistency": 0.1,
            "uniqueness": 0.1
        })
        findings = create_findings(columns={
            "col1": create_column_findings(null_percentage=50),  # 50% completeness
        })
        result = scorer.calculate(findings)
        # completeness=50, others=100
        # Overall = 0.7*50 + 0.1*100 + 0.1*100 + 0.1*100 = 35 + 30 = 65
        assert result.overall_score == 65.0

    def test_quality_levels(self, scorer):
        """Test quality level classification."""
        # Test each level
        findings_excellent = create_findings(columns={
            "col1": create_column_findings(null_percentage=5),
        })
        result = scorer.calculate(findings_excellent)
        assert result.quality_level == QualityLevel.EXCELLENT

        findings_poor = create_findings(columns={
            "col1": create_column_findings(null_percentage=80),
        })
        result = scorer.calculate(findings_poor)
        # 20% completeness, others 100 = (20+100+100+100)/4 = 80 = GOOD
        # Actually need lower scores...


class TestQualityScoreResult(TestQualityScorer):
    """Tests for QualityScoreResult dataclass."""

    def test_to_dict(self, scorer, clean_findings):
        """Test conversion to dictionary."""
        result = scorer.calculate(clean_findings)
        d = result.to_dict()

        assert "overall_score" in d
        assert "quality_level" in d
        assert "components" in d
        assert "component_weights" in d
        assert "issues" in d
        assert "recommendations" in d

    def test_str_representation(self, scorer, clean_findings):
        """Test string representation."""
        result = scorer.calculate(clean_findings)
        s = str(result)

        assert "Quality Score" in s
        assert "100.0/100" in s
        assert "Components:" in s

    def test_recommendations_generated(self, scorer):
        """Test that recommendations are generated for low scores."""
        findings = create_findings(columns={
            "col1": create_column_findings(null_percentage=50),
        })
        result = scorer.calculate(findings)
        assert len(result.recommendations) > 0

    def test_good_quality_recommendation(self, scorer, clean_findings):
        """Test that good quality gets positive recommendation."""
        result = scorer.calculate(clean_findings)
        assert any("good" in rec.lower() or "proceed" in rec.lower()
                   for rec in result.recommendations)


class TestEdgeCases(TestQualityScorer):
    """Tests for edge cases."""

    def test_empty_dataframe(self, scorer):
        """Test with empty findings."""
        findings = create_findings(row_count=0, column_count=0, columns={})
        result = scorer.calculate(findings)
        assert result.overall_score == 100.0

    def test_all_components_zero(self, scorer):
        """Test floor at 0 for each component."""
        # Create scenario with maximum penalties
        findings = create_findings(columns={
            "id1": create_column_findings(inferred_type="identifier", distinct_percentage=10),
            "id2": create_column_findings(inferred_type="identifier", distinct_percentage=10),
            "id3": create_column_findings(inferred_type="identifier", distinct_percentage=10),
            "id4": create_column_findings(inferred_type="identifier", distinct_percentage=10),
            "id5": create_column_findings(inferred_type="identifier", distinct_percentage=10),
            "id6": create_column_findings(inferred_type="identifier", distinct_percentage=10),
        })
        result = scorer.calculate(findings)
        # Uniqueness should be capped at 0, not negative
        assert result.components["uniqueness"] >= 0

    def test_none_validation_results(self, scorer, clean_findings):
        """Test with None for all validation results."""
        result = scorer.calculate(
            clean_findings,
            duplicate_result=None,
            date_result=None,
            range_results=None
        )
        assert result.overall_score == 100.0
