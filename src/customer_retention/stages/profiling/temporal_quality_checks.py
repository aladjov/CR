"""
Temporal quality checks for time series datasets.

Provides quality checks specific to event-level data including:
- Duplicate event detection
- Temporal gap detection
- Future date detection
- Event ordering validation
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, List

from customer_retention.core.compat import pd, DataFrame
from customer_retention.core.components.enums import Severity


@dataclass
class TemporalQualityResult:
    """Result of a temporal quality check."""
    check_id: str
    check_name: str
    passed: bool
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)
    recommendation: Optional[str] = None

    # Check-specific fields (set by individual checks)
    duplicate_count: int = 0
    gap_count: int = 0
    max_gap_days: float = 0
    future_count: int = 0
    ambiguous_count: int = 0


class TemporalQualityCheck:
    """Base class for temporal quality checks."""

    def __init__(self, check_id: str, check_name: str, severity: Severity):
        self.check_id = check_id
        self.check_name = check_name
        self.severity = severity

    def run(self, df: DataFrame) -> TemporalQualityResult:
        """Run the check on the dataframe."""
        raise NotImplementedError


class DuplicateEventCheck(TemporalQualityCheck):
    """Detects duplicate events (same entity + timestamp)."""

    def __init__(self, entity_column: str, time_column: str):
        super().__init__(
            check_id="TQ001",
            check_name="Duplicate Events",
            severity=Severity.MEDIUM
        )
        self.entity_column = entity_column
        self.time_column = time_column

    def run(self, df: DataFrame) -> TemporalQualityResult:
        if len(df) == 0:
            return self._pass_result("No data to check")

        duplicates = df.duplicated(subset=[self.entity_column, self.time_column], keep=False)
        duplicate_count = duplicates.sum() - df[duplicates].groupby(
            [self.entity_column, self.time_column]
        ).ngroups

        if duplicate_count > 0:
            duplicate_rows = df[duplicates].head(10)
            examples = duplicate_rows[[self.entity_column, self.time_column]].to_dict('records')

            return TemporalQualityResult(
                check_id=self.check_id,
                check_name=self.check_name,
                passed=False,
                severity=self.severity,
                message=f"Found {duplicate_count} duplicate events (same entity + timestamp)",
                details={
                    "duplicate_examples": examples,
                    "affected_entities": df[duplicates][self.entity_column].nunique(),
                },
                recommendation="Review duplicates - may need deduplication logic",
                duplicate_count=duplicate_count,
            )

        return self._pass_result("No duplicate events found")

    def _pass_result(self, message: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            check_id=self.check_id,
            check_name=self.check_name,
            passed=True,
            severity=Severity.INFO,
            message=message,
            duplicate_count=0,
        )


class TemporalGapCheck(TemporalQualityCheck):
    """Detects unexpected gaps in time series data."""

    FREQ_TO_DAYS = {
        "D": 1, "W": 7, "M": 30, "Q": 90, "Y": 365,
        "H": 1/24, "T": 1/1440, "min": 1/1440,
    }

    def __init__(self, time_column: str, expected_frequency: str = "D",
                 max_gap_multiple: float = 3.0):
        super().__init__(
            check_id="TQ002",
            check_name="Temporal Gaps",
            severity=Severity.MEDIUM
        )
        self.time_column = time_column
        self.expected_frequency = expected_frequency
        self.max_gap_multiple = max_gap_multiple

    def run(self, df: DataFrame) -> TemporalQualityResult:
        if len(df) < 2:
            return self._pass_result("Insufficient data to check gaps")

        df_sorted = df.sort_values(self.time_column)
        time_col = pd.to_datetime(df_sorted[self.time_column])

        diffs = time_col.diff().dropna()
        diffs_days = diffs.dt.total_seconds() / 86400

        expected_days = self.FREQ_TO_DAYS.get(self.expected_frequency, 1)
        threshold_days = expected_days * self.max_gap_multiple

        large_gaps = diffs_days[diffs_days > threshold_days]
        gap_count = len(large_gaps)
        max_gap = float(diffs_days.max()) if len(diffs_days) > 0 else 0

        if gap_count > 0:
            return TemporalQualityResult(
                check_id=self.check_id,
                check_name=self.check_name,
                passed=False,
                severity=self.severity,
                message=f"Found {gap_count} gaps exceeding {threshold_days:.1f} days",
                details={
                    "threshold_days": threshold_days,
                    "expected_frequency": self.expected_frequency,
                    "gap_locations": large_gaps.index.tolist()[:10],
                },
                recommendation="Investigate data collection gaps or missing data",
                gap_count=gap_count,
                max_gap_days=max_gap,
            )

        return TemporalQualityResult(
            check_id=self.check_id,
            check_name=self.check_name,
            passed=True,
            severity=Severity.INFO,
            message="No significant temporal gaps detected",
            gap_count=0,
            max_gap_days=max_gap,
        )

    def _pass_result(self, message: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            check_id=self.check_id,
            check_name=self.check_name,
            passed=True,
            severity=Severity.INFO,
            message=message,
            gap_count=0,
            max_gap_days=0,
        )


class FutureDateCheck(TemporalQualityCheck):
    """Detects dates in the future."""

    def __init__(self, time_column: str,
                 reference_date: Optional[pd.Timestamp] = None):
        super().__init__(
            check_id="TQ003",
            check_name="Future Dates",
            severity=Severity.HIGH
        )
        self.time_column = time_column
        self.reference_date = reference_date or pd.Timestamp.now()

    def run(self, df: DataFrame) -> TemporalQualityResult:
        if len(df) == 0:
            return self._pass_result("No data to check")

        time_col = pd.to_datetime(df[self.time_column])
        future_mask = time_col > self.reference_date
        future_count = future_mask.sum()

        if future_count > 0:
            future_dates = time_col[future_mask].head(10).tolist()

            return TemporalQualityResult(
                check_id=self.check_id,
                check_name=self.check_name,
                passed=False,
                severity=self.severity,
                message=f"Found {future_count} events with future dates",
                details={
                    "reference_date": str(self.reference_date),
                    "future_date_examples": [str(d) for d in future_dates],
                },
                recommendation="Review data entry or timestamp handling",
                future_count=future_count,
            )

        return self._pass_result("No future dates detected")

    def _pass_result(self, message: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            check_id=self.check_id,
            check_name=self.check_name,
            passed=True,
            severity=Severity.INFO,
            message=message,
            future_count=0,
        )


class EventOrderCheck(TemporalQualityCheck):
    """Checks for potential event ordering issues."""

    def __init__(self, entity_column: str, time_column: str):
        super().__init__(
            check_id="TQ004",
            check_name="Event Ordering",
            severity=Severity.LOW
        )
        self.entity_column = entity_column
        self.time_column = time_column

    def run(self, df: DataFrame) -> TemporalQualityResult:
        if len(df) < 2:
            return self._pass_result("Insufficient data to check ordering")

        # Check for timestamp collisions within entities
        time_col = pd.to_datetime(df[self.time_column])
        df_check = df.copy()
        df_check["_parsed_time"] = time_col

        # Count events with same entity + timestamp (ambiguous ordering)
        collision_counts = df_check.groupby(
            [self.entity_column, "_parsed_time"]
        ).size()
        ambiguous = collision_counts[collision_counts > 1]
        ambiguous_count = ambiguous.sum() - len(ambiguous)

        if ambiguous_count > 0:
            return TemporalQualityResult(
                check_id=self.check_id,
                check_name=self.check_name,
                passed=True,  # Pass but with warning
                severity=Severity.LOW,
                message=f"{ambiguous_count} events have ambiguous ordering (same timestamp)",
                details={
                    "collision_groups": len(ambiguous),
                    "total_ambiguous_events": int(ambiguous.sum()),
                },
                recommendation="Consider adding sequence numbers for same-timestamp events",
                ambiguous_count=ambiguous_count,
            )

        return self._pass_result("Event ordering is unambiguous")

    def _pass_result(self, message: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            check_id=self.check_id,
            check_name=self.check_name,
            passed=True,
            severity=Severity.INFO,
            message=message,
            ambiguous_count=0,
        )
