"""
Quality scoring for data exploration.

This module provides a comprehensive quality scorer that calculates
data quality scores based on validation results from the exploration phase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .data_validators import DateLogicResult, DuplicateResult, RangeValidationResult
from .timeseries_detector import TimeSeriesCharacteristics, TimeSeriesValidationResult


class QualityLevel(Enum):
    """Quality level classifications."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"            # 70-89
    FAIR = "fair"            # 50-69
    POOR = "poor"            # 0-49


@dataclass
class QualityScoreResult:
    """Result of quality score calculation."""
    overall_score: float
    quality_level: QualityLevel
    components: Dict[str, float]
    component_weights: Dict[str, float]
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Time series specific
    is_time_series: bool = False
    dataset_type: str = "snapshot"
    timeseries_characteristics: Optional[Dict[str, Any]] = None
    timeseries_quality: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "overall_score": round(self.overall_score, 1),
            "quality_level": self.quality_level.value,
            "components": {k: round(v, 1) for k, v in self.components.items()},
            "component_weights": self.component_weights,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "is_time_series": self.is_time_series,
            "dataset_type": self.dataset_type,
        }
        if self.timeseries_characteristics:
            result["timeseries_characteristics"] = self.timeseries_characteristics
        if self.timeseries_quality:
            result["timeseries_quality"] = self.timeseries_quality
        return result

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Quality Score: {self.overall_score:.1f}/100 ({self.quality_level.value})",
            f"Dataset Type: {self.dataset_type}",
            "",
            "Components:"
        ]
        for component, score in self.components.items():
            weight = self.component_weights.get(component, 0) * 100
            lines.append(f"  - {component}: {score:.1f} (weight: {weight:.0f}%)")

        if self.is_time_series and self.timeseries_quality:
            lines.append("")
            lines.append("Time Series Quality:")
            lines.append(f"  - Temporal Score: {self.timeseries_quality.get('temporal_quality_score', 'N/A')}")

        if self.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")

        return "\n".join(lines)


@dataclass
class ColumnFindings:
    """Minimal column findings interface for quality scoring."""
    inferred_type: Any  # Should have .value attribute
    universal_metrics: Dict[str, Any]


@dataclass
class ExplorationFindings:
    """Minimal exploration findings interface for quality scoring."""
    row_count: int
    column_count: int
    columns: Dict[str, ColumnFindings]
    target_column: Optional[str] = None


class QualityScorer:
    """
    Calculate comprehensive data quality scores based on validation results.

    The quality score is composed of four components:
    - Completeness: Percentage of non-null values
    - Validity: Values within expected ranges and formats
    - Consistency: No conflicting duplicates or logical violations
    - Uniqueness: Identifier columns have appropriate cardinality

    Each component can be weighted differently based on use case.

    Example
    -------
    >>> scorer = QualityScorer()
    >>> result = scorer.calculate(
    ...     findings=exploration_findings,
    ...     duplicate_result=dup_result,
    ...     date_result=date_result,
    ...     range_results=range_results
    ... )
    >>> print(f"Quality Score: {result.overall_score:.1f}/100")
    """

    DEFAULT_WEIGHTS = {
        "completeness": 0.25,
        "validity": 0.25,
        "consistency": 0.25,
        "uniqueness": 0.25
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the quality scorer.

        Parameters
        ----------
        weights : Dict[str, float], optional
            Custom weights for each component. Must sum to 1.0.
            Keys: 'completeness', 'validity', 'consistency', 'uniqueness'
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0 and all components are present."""
        required = set(self.DEFAULT_WEIGHTS.keys())
        provided = set(self.weights.keys())

        missing = required - provided
        if missing:
            raise ValueError(f"Missing weight components: {missing}")

        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point variance
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def calculate(
        self,
        findings: ExplorationFindings,
        duplicate_result: Optional[DuplicateResult] = None,
        date_result: Optional[DateLogicResult] = None,
        range_results: Optional[List[RangeValidationResult]] = None,
        timeseries_characteristics: Optional[TimeSeriesCharacteristics] = None,
        timeseries_validation: Optional[TimeSeriesValidationResult] = None
    ) -> QualityScoreResult:
        """
        Calculate comprehensive quality score.

        Parameters
        ----------
        findings : ExplorationFindings
            Results from data exploration (column info, row counts, etc.)
        duplicate_result : DuplicateResult, optional
            Results from duplicate validation
        date_result : DateLogicResult, optional
            Results from date logic validation
        range_results : List[RangeValidationResult], optional
            Results from value range validations
        timeseries_characteristics : TimeSeriesCharacteristics, optional
            Results from time series detection
        timeseries_validation : TimeSeriesValidationResult, optional
            Results from time series validation

        Returns
        -------
        QualityScoreResult
            Comprehensive quality score with component breakdown
        """
        components = {}
        issues = []
        recommendations = []

        # Calculate each component
        components["completeness"], comp_issues = self._calculate_completeness(findings)
        issues.extend(comp_issues)

        components["validity"], val_issues = self._calculate_validity(range_results)
        issues.extend(val_issues)

        components["consistency"], cons_issues = self._calculate_consistency(
            duplicate_result, date_result
        )
        issues.extend(cons_issues)

        components["uniqueness"], uniq_issues = self._calculate_uniqueness(findings)
        issues.extend(uniq_issues)

        # Handle time series specific scoring
        is_time_series = False
        dataset_type = "snapshot"
        ts_characteristics_dict = None
        ts_quality_dict = None

        if timeseries_characteristics is not None:
            is_time_series = timeseries_characteristics.is_time_series
            dataset_type = timeseries_characteristics.dataset_type.value
            ts_characteristics_dict = timeseries_characteristics.to_dict()

            if is_time_series and timeseries_validation is not None:
                # Add temporal quality as a component for time series data
                components["temporal"], ts_issues = self._calculate_temporal_quality(
                    timeseries_validation
                )
                issues.extend(ts_issues)
                ts_quality_dict = timeseries_validation.to_dict()

                # Adjust weights for time series data
                adjusted_weights = self._adjust_weights_for_timeseries()
            else:
                adjusted_weights = self.weights
        else:
            adjusted_weights = self.weights

        # Calculate weighted overall score
        overall_score = 0.0
        for comp, score in components.items():
            weight = adjusted_weights.get(comp, 0)
            overall_score += score * weight

        # Determine quality level
        quality_level = self._get_quality_level(overall_score)

        # Generate recommendations based on issues
        recommendations = self._generate_recommendations(
            components, issues, is_time_series
        )

        return QualityScoreResult(
            overall_score=overall_score,
            quality_level=quality_level,
            components=components,
            component_weights=adjusted_weights,
            issues=issues,
            recommendations=recommendations,
            is_time_series=is_time_series,
            dataset_type=dataset_type,
            timeseries_characteristics=ts_characteristics_dict,
            timeseries_quality=ts_quality_dict
        )

    def _calculate_completeness(
        self,
        findings: ExplorationFindings
    ) -> tuple[float, List[str]]:
        """
        Calculate completeness score based on missing values.

        Returns
        -------
        tuple[float, List[str]]
            Score (0-100) and list of issues found
        """
        issues = []

        if findings.row_count == 0 or findings.column_count == 0:
            return 100.0, issues

        total_cells = findings.row_count * findings.column_count
        missing_cells = 0
        columns_with_high_missing = []

        for col_name, col in findings.columns.items():
            null_pct = col.universal_metrics.get("null_percentage", 0)
            missing_cells += (null_pct / 100) * findings.row_count

            if null_pct > 20:
                columns_with_high_missing.append((col_name, null_pct))

        completeness = 100 * (1 - missing_cells / total_cells)

        # Add issues for high missing columns
        for col_name, pct in columns_with_high_missing[:3]:  # Top 3
            issues.append(f"Column '{col_name}' has {pct:.1f}% missing values")

        return max(0, min(100, completeness)), issues

    def _calculate_validity(
        self,
        range_results: Optional[List[RangeValidationResult]]
    ) -> tuple[float, List[str]]:
        """
        Calculate validity score based on range validation results.

        Returns
        -------
        tuple[float, List[str]]
            Score (0-100) and list of issues found
        """
        issues = []

        if not range_results:
            return 100.0, issues  # No rules defined, assume valid

        total_checked = 0
        total_invalid = 0

        for result in range_results:
            total_checked += result.total_values
            total_invalid += result.invalid_values

            if result.invalid_percentage > 5:
                issues.append(
                    f"Column '{result.column_name}' has {result.invalid_percentage:.1f}% "
                    f"values outside {result.rule_type} range"
                )

        if total_checked == 0:
            return 100.0, issues

        validity = 100 * (1 - total_invalid / total_checked)

        return max(0, min(100, validity)), issues

    def _calculate_consistency(
        self,
        duplicate_result: Optional[DuplicateResult],
        date_result: Optional[DateLogicResult]
    ) -> tuple[float, List[str]]:
        """
        Calculate consistency score based on duplicates and date logic.

        Returns
        -------
        tuple[float, List[str]]
            Score (0-100) and list of issues found
        """
        issues = []
        penalties = 0

        # Duplicate penalties
        if duplicate_result is not None:
            dup_pct = duplicate_result.duplicate_percentage

            if dup_pct > 10:
                penalties += 30
                issues.append(f"High duplicate rate: {dup_pct:.1f}%")
            elif dup_pct > 5:
                penalties += 20
                issues.append(f"Moderate duplicate rate: {dup_pct:.1f}%")
            elif dup_pct > 1:
                penalties += 10

            # Value conflicts are more severe
            if duplicate_result.has_value_conflicts:
                penalties += 20
                conflict_cols = ", ".join(duplicate_result.conflict_columns[:3])
                issues.append(f"Value conflicts in duplicate records: {conflict_cols}")

        # Date logic penalties
        if date_result is not None:
            invalid_pct = date_result.invalid_percentage

            if invalid_pct > 10:
                penalties += 20
                issues.append(f"High date logic violation rate: {invalid_pct:.1f}%")
            elif invalid_pct > 5:
                penalties += 10
                issues.append(f"Moderate date logic violations: {invalid_pct:.1f}%")
            elif invalid_pct > 1:
                penalties += 5

        consistency = max(0, 100 - penalties)

        return consistency, issues

    def _calculate_uniqueness(
        self,
        findings: ExplorationFindings
    ) -> tuple[float, List[str]]:
        """
        Calculate uniqueness score for identifier columns.

        Returns
        -------
        tuple[float, List[str]]
            Score (0-100) and list of issues found
        """
        issues = []
        penalties = 0
        identifier_count = 0

        for col_name, col in findings.columns.items():
            col_type = getattr(col.inferred_type, 'value', str(col.inferred_type))

            if col_type in ('identifier', 'id'):
                identifier_count += 1
                distinct_pct = col.universal_metrics.get("distinct_percentage", 100)

                if distinct_pct < 90:
                    penalties += 20
                    issues.append(
                        f"Identifier '{col_name}' has low uniqueness: {distinct_pct:.1f}%"
                    )
                elif distinct_pct < 95:
                    penalties += 10

        # If no identifiers found, full score
        if identifier_count == 0:
            return 100.0, issues

        uniqueness = max(0, 100 - penalties)

        return uniqueness, issues

    def _get_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 70:
            return QualityLevel.GOOD
        elif score >= 50:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR

    def _calculate_temporal_quality(
        self,
        timeseries_validation: TimeSeriesValidationResult
    ) -> tuple[float, List[str]]:
        """
        Calculate temporal quality score for time series data.

        Returns
        -------
        tuple[float, List[str]]
            Score (0-100) and list of issues found
        """
        issues = []

        # Start with the temporal quality score from validation
        score = timeseries_validation.temporal_quality_score

        # Add issues from validation
        issues.extend(timeseries_validation.issues)

        # Add specific issues based on metrics
        if timeseries_validation.entities_with_gaps > 0:
            gap_rate = timeseries_validation.entities_with_gaps
            if gap_rate > 10:
                issues.append(f"High number of entities with gaps: {gap_rate}")

        if timeseries_validation.total_duplicate_timestamps > 0:
            issues.append(
                f"Found {timeseries_validation.total_duplicate_timestamps} duplicate timestamps"
            )

        if timeseries_validation.entities_with_ordering_issues > 0:
            issues.append(
                f"{timeseries_validation.entities_with_ordering_issues} entities have "
                "timestamps out of order"
            )

        return max(0, min(100, score)), issues

    def _adjust_weights_for_timeseries(self) -> Dict[str, float]:
        """
        Adjust component weights for time series data.

        When temporal quality is included, redistribute weights
        to give appropriate importance to temporal aspects.
        """
        # For time series, include temporal as 20% and reduce others proportionally
        temporal_weight = 0.20
        reduction_factor = 1 - temporal_weight

        adjusted = {
            "completeness": self.weights["completeness"] * reduction_factor,
            "validity": self.weights["validity"] * reduction_factor,
            "consistency": self.weights["consistency"] * reduction_factor,
            "uniqueness": self.weights["uniqueness"] * reduction_factor,
            "temporal": temporal_weight
        }

        return adjusted

    def _generate_recommendations(
        self,
        components: Dict[str, float],
        issues: List[str],
        is_time_series: bool = False
    ) -> List[str]:
        """Generate recommendations based on component scores and issues."""
        recommendations = []

        if components["completeness"] < 80:
            recommendations.append(
                "Review missing value imputation strategies before modeling"
            )

        if components["validity"] < 80:
            recommendations.append(
                "Investigate values outside expected ranges - may need cleaning or rule adjustment"
            )

        if components["consistency"] < 80:
            recommendations.append(
                "Resolve duplicate records and date logic violations before analysis"
            )

        if components["uniqueness"] < 80:
            recommendations.append(
                "Verify identifier columns - low uniqueness may indicate data issues"
            )

        # Time series specific recommendations
        if is_time_series:
            temporal_score = components.get("temporal", 100)
            if temporal_score < 80:
                recommendations.append(
                    "Address temporal quality issues: gaps, duplicates, or ordering problems"
                )

            if all(score >= 80 for score in components.values()):
                recommendations.append(
                    "Time series data quality is good - proceed to temporal feature engineering"
                )
        else:
            # Add general recommendation if score is good
            if all(score >= 80 for score in components.values()):
                recommendations.append(
                    "Data quality is good - proceed to feature engineering"
                )

        return recommendations
