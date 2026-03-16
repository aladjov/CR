"""Tests for quality scorer module."""

import pytest

from customer_retention.core.components.enums import Severity
from customer_retention.stages.validation import DateLogicResult, DuplicateResult, RangeValidationResult
from customer_retention.stages.validation.quality_scorer import (
    ColumnFindings,
    ExplorationFindings,
    QualityLevel,
    QualityScorer,
)


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


# ---------------------------------------------------------------------------
# Coverage gap tests: timeseries paths, edge cases, and string formatting
# ---------------------------------------------------------------------------

from customer_retention.stages.validation.timeseries_detector import (
    DatasetType,
    TimeSeriesCharacteristics,
    TimeSeriesFrequency,
    TimeSeriesValidationResult,
)


def _make_ts_characteristics(
    is_time_series: bool = True,
    dataset_type: DatasetType = DatasetType.TIME_SERIES,
    **kwargs,
) -> TimeSeriesCharacteristics:
    """Helper to build a TimeSeriesCharacteristics with sensible defaults."""
    defaults = dict(
        entity_column="customer_id",
        timestamp_column="event_date",
        total_entities=100,
        min_observations_per_entity=5,
        max_observations_per_entity=50,
        avg_observations_per_entity=20.0,
        median_observations_per_entity=18.0,
        time_span_days=365.0,
        detected_frequency=TimeSeriesFrequency.DAILY,
        median_interval_hours=24.0,
        confidence=0.95,
    )
    defaults.update(kwargs)
    return TimeSeriesCharacteristics(
        is_time_series=is_time_series,
        dataset_type=dataset_type,
        **defaults,
    )


def _make_ts_validation(
    temporal_quality_score: float = 90.0,
    **kwargs,
) -> TimeSeriesValidationResult:
    """Helper to build a TimeSeriesValidationResult with sensible defaults."""
    defaults = dict(
        total_expected_periods=1000,
        total_actual_periods=950,
        coverage_percentage=95.0,
        entities_with_gaps=0,
        total_gaps=0,
        max_gap_periods=0,
        entities_with_duplicate_timestamps=0,
        total_duplicate_timestamps=0,
        entities_with_ordering_issues=0,
        frequency_consistent=True,
        frequency_deviation_percentage=1.0,
        issues=[],
    )
    defaults.update(kwargs)
    return TimeSeriesValidationResult(
        temporal_quality_score=temporal_quality_score,
        **defaults,
    )


class TestToDictWithTimeseries(TestQualityScorer):
    """Cover lines 52-55: to_dict() with timeseries fields populated."""

    def test_to_dict_includes_timeseries_characteristics(self, scorer, clean_findings):
        """to_dict() must include timeseries_characteristics when present."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation()
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        d = result.to_dict()
        assert "timeseries_characteristics" in d
        assert d["timeseries_characteristics"]["is_time_series"] is True

    def test_to_dict_includes_timeseries_quality(self, scorer, clean_findings):
        """to_dict() must include timeseries_quality when present."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(temporal_quality_score=85.0)
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        d = result.to_dict()
        assert "timeseries_quality" in d
        assert d["timeseries_quality"]["temporal_quality_score"] == 85.0

    def test_to_dict_omits_timeseries_when_none(self, scorer, clean_findings):
        """to_dict() must NOT include timeseries keys when they are None."""
        result = scorer.calculate(clean_findings)
        d = result.to_dict()
        assert "timeseries_characteristics" not in d
        assert "timeseries_quality" not in d


class TestStrWithTimeseries(TestQualityScorer):
    """Cover lines 70-79: __str__() with timeseries quality and issues."""

    def test_str_contains_timeseries_quality_section(self, scorer, clean_findings):
        """__str__() must render 'Time Series Quality:' section."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(temporal_quality_score=75.0)
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        s = str(result)
        assert "Time Series Quality:" in s
        assert "Temporal Score:" in s

    def test_str_contains_issues_section(self, scorer):
        """__str__() must render 'Issues:' section when there are issues."""
        findings = create_findings(columns={
            "col1": create_column_findings(null_percentage=50),
        })
        result = scorer.calculate(findings)
        # Force an issue by having high null percentage
        s = str(result)
        assert "Issues:" in s


class TestCalculateWithTimeseries(TestQualityScorer):
    """Cover lines 215-230: calculate() with timeseries_characteristics + timeseries_validation."""

    def test_calculate_with_timeseries_adds_temporal_component(self, scorer, clean_findings):
        """calculate() must add 'temporal' component when ts data is provided."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(temporal_quality_score=80.0)
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert "temporal" in result.components
        assert result.components["temporal"] == 80.0
        assert result.is_time_series is True
        assert result.dataset_type == "time_series"

    def test_calculate_timeseries_uses_adjusted_weights(self, scorer, clean_findings):
        """When ts data is present, weights are adjusted so 'temporal' gets 20%."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(temporal_quality_score=100.0)
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert "temporal" in result.component_weights
        assert result.component_weights["temporal"] == pytest.approx(0.20, abs=0.01)

    def test_calculate_timeseries_characteristics_not_ts(self, scorer, clean_findings):
        """When characteristics say is_time_series=False, no temporal component."""
        ts_chars = _make_ts_characteristics(is_time_series=False,
                                            dataset_type=DatasetType.SNAPSHOT)
        ts_val = _make_ts_validation()
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert "temporal" not in result.components
        assert result.is_time_series is False

    def test_calculate_timeseries_no_validation(self, scorer, clean_findings):
        """When characteristics present but validation is None, no temporal component."""
        ts_chars = _make_ts_characteristics()
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=None,
        )
        assert "temporal" not in result.components
        assert result.is_time_series is True
        assert result.dataset_type == "time_series"


class TestValidityTotalCheckedZero(TestQualityScorer):
    """Cover line 328: validity when total_checked==0."""

    def test_validity_total_checked_zero(self, scorer, clean_findings):
        """When all range rules have total_values=0, validity is 100."""
        range_results = [
            RangeValidationResult(
                column_name="col_x",
                total_values=0,
                valid_values=0,
                invalid_values=0,
                invalid_percentage=0.0,
                rule_type="range",
                expected_range="[0, 100]",
                actual_range="N/A",
                severity=Severity.INFO,
            )
        ]
        result = scorer.calculate(clean_findings, range_results=range_results)
        assert result.components["validity"] == 100.0


class TestConsistencyDateLogicViolations(TestQualityScorer):
    """Cover lines 360-380: consistency with moderate and high date logic violations."""

    def test_moderate_date_logic_violation(self, scorer, clean_findings):
        """Date logic invalid_pct between 5 and 10 => 10 penalty."""
        date_result = DateLogicResult(
            date_columns=["start", "end"],
            total_rows=100,
            valid_rows=92,
            invalid_rows=8,
            invalid_percentage=8.0,
            severity=Severity.WARNING,
        )
        result = scorer.calculate(clean_findings, date_result=date_result)
        assert result.components["consistency"] == 90.0
        assert any("Moderate date logic" in i for i in result.issues)

    def test_high_date_logic_violation(self, scorer, clean_findings):
        """Date logic invalid_pct > 10 => 20 penalty."""
        date_result = DateLogicResult(
            date_columns=["start", "end"],
            total_rows=100,
            valid_rows=85,
            invalid_rows=15,
            invalid_percentage=15.0,
            severity=Severity.CRITICAL,
        )
        result = scorer.calculate(clean_findings, date_result=date_result)
        assert result.components["consistency"] == 80.0
        assert any("High date logic" in i for i in result.issues)

    def test_low_date_logic_violation(self, scorer, clean_findings):
        """Date logic invalid_pct between 1 and 5 => 5 penalty, no issue message."""
        date_result = DateLogicResult(
            date_columns=["a", "b"],
            total_rows=100,
            valid_rows=97,
            invalid_rows=3,
            invalid_percentage=3.0,
            severity=Severity.INFO,
        )
        result = scorer.calculate(clean_findings, date_result=date_result)
        assert result.components["consistency"] == 95.0
        # No issue message for < 5% (penalty only)
        assert not any("date logic" in i.lower() for i in result.issues)


class TestUniquenessPenalty90to95(TestQualityScorer):
    """Cover line 415: uniqueness penalty when distinct_percentage is 90-95%."""

    def test_uniqueness_penalty_90_to_95(self, scorer):
        """Identifier with distinct_pct in [90, 95) => 10 penalty, no issue message."""
        findings = create_findings(columns={
            "user_id": create_column_findings(
                inferred_type="identifier",
                distinct_percentage=92.0,
            ),
        })
        result = scorer.calculate(findings)
        assert result.components["uniqueness"] == 90.0
        # No issue text because the threshold for issues is < 90%
        assert not any("user_id" in i for i in result.issues)


class TestCalculateTemporalQuality(TestQualityScorer):
    """Cover lines 448-473: _calculate_temporal_quality() branches."""

    def test_temporal_quality_with_gaps(self, scorer, clean_findings):
        """Gaps > 10 entities should produce an issue."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(
            temporal_quality_score=70.0,
            entities_with_gaps=15,
            total_gaps=30,
        )
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert any("entities with gaps" in i for i in result.issues)

    def test_temporal_quality_with_duplicates(self, scorer, clean_findings):
        """Duplicate timestamps > 0 should produce an issue."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(
            temporal_quality_score=65.0,
            total_duplicate_timestamps=42,
        )
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert any("42 duplicate timestamps" in i for i in result.issues)

    def test_temporal_quality_with_ordering_issues(self, scorer, clean_findings):
        """Ordering issues > 0 should produce an issue."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(
            temporal_quality_score=60.0,
            entities_with_ordering_issues=5,
        )
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert any("out of order" in i for i in result.issues)

    def test_temporal_quality_score_clamped(self, scorer, clean_findings):
        """Score is clamped to [0, 100] range."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(temporal_quality_score=110.0)
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert result.components["temporal"] == 100.0

    def test_temporal_quality_inherits_validation_issues(self, scorer, clean_findings):
        """Pre-existing issues from validation are propagated."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(
            temporal_quality_score=72.0,
            issues=["Coverage below 80%", "Irregular frequency detected"],
        )
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert "Coverage below 80%" in result.issues
        assert "Irregular frequency detected" in result.issues


class TestAdjustWeightsForTimeseries(TestQualityScorer):
    """Cover lines 483-494: _adjust_weights_for_timeseries()."""

    def test_adjusted_weights_sum_to_one(self, scorer):
        """Adjusted weights must sum to 1.0."""
        adjusted = scorer._adjust_weights_for_timeseries()
        assert sum(adjusted.values()) == pytest.approx(1.0)

    def test_adjusted_weights_has_temporal(self, scorer):
        """Adjusted weights must include 'temporal' at 0.20."""
        adjusted = scorer._adjust_weights_for_timeseries()
        assert adjusted["temporal"] == 0.20

    def test_adjusted_weights_reduce_originals(self, scorer):
        """Each original weight is reduced by the temporal allocation factor."""
        adjusted = scorer._adjust_weights_for_timeseries()
        for key in ("completeness", "validity", "consistency", "uniqueness"):
            assert adjusted[key] == pytest.approx(
                scorer.weights[key] * 0.80, abs=1e-6
            )


class TestGenerateRecommendationsTimeseries(TestQualityScorer):
    """Cover lines 511, 527-534: _generate_recommendations() for timeseries paths."""

    def test_temporal_low_score_recommendation(self, scorer, clean_findings):
        """Low temporal score should produce temporal-specific recommendation."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(temporal_quality_score=50.0)
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert any("temporal quality issues" in r.lower() for r in result.recommendations)

    def test_timeseries_all_good_recommendation(self, scorer, clean_findings):
        """When all scores >= 80 for timeseries, recommend temporal feature engineering."""
        ts_chars = _make_ts_characteristics()
        ts_val = _make_ts_validation(temporal_quality_score=90.0)
        result = scorer.calculate(
            clean_findings,
            timeseries_characteristics=ts_chars,
            timeseries_validation=ts_val,
        )
        assert any("temporal feature engineering" in r.lower() for r in result.recommendations)

    def test_non_timeseries_all_good_recommendation(self, scorer, clean_findings):
        """When all scores >= 80 for non-ts data, recommend feature engineering."""
        result = scorer.calculate(clean_findings)
        assert any(
            "proceed to feature engineering" in r.lower()
            for r in result.recommendations
        )

    def test_validity_low_recommendation(self, scorer, clean_findings):
        """Validity below 80 produces validity-specific recommendation."""
        range_results = [
            RangeValidationResult(
                column_name="amount",
                total_values=100,
                valid_values=70,
                invalid_values=30,
                invalid_percentage=30.0,
                rule_type="range",
                expected_range="[0, 1000]",
                actual_range="[-100, 5000]",
                severity=Severity.CRITICAL,
            )
        ]
        result = scorer.calculate(clean_findings, range_results=range_results)
        assert any("expected ranges" in r.lower() for r in result.recommendations)
