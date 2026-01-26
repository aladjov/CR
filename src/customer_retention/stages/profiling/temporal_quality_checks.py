from dataclasses import dataclass, field
from typing import Optional

from customer_retention.core.compat import DataFrame, pd
from customer_retention.core.components.enums import Severity


@dataclass
class TemporalQualityResult:
    check_id: str
    check_name: str
    passed: bool
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)
    recommendation: Optional[str] = None
    duplicate_count: int = 0
    gap_count: int = 0
    max_gap_days: float = 0
    future_count: int = 0
    ambiguous_count: int = 0


class TemporalQualityCheck:
    def __init__(self, check_id: str, check_name: str, severity: Severity):
        self.check_id = check_id
        self.check_name = check_name
        self.severity = severity

    def run(self, df: DataFrame) -> TemporalQualityResult:
        raise NotImplementedError


class DuplicateEventCheck(TemporalQualityCheck):
    def __init__(self, entity_column: str, time_column: str):
        super().__init__("TQ001", "Duplicate Events", Severity.MEDIUM)
        self.entity_column = entity_column
        self.time_column = time_column

    def run(self, df: DataFrame) -> TemporalQualityResult:
        if len(df) == 0:
            return self._pass_result("No data to check")

        duplicates = df.duplicated(subset=[self.entity_column, self.time_column], keep=False)
        duplicate_count = duplicates.sum() - df[duplicates].groupby([self.entity_column, self.time_column]).ngroups

        if duplicate_count > 0:
            examples = df[duplicates].head(10)[[self.entity_column, self.time_column]].to_dict('records')
            return TemporalQualityResult(
                check_id=self.check_id, check_name=self.check_name, passed=False, severity=self.severity,
                message=f"Found {duplicate_count} duplicate events (same entity + timestamp)",
                details={"duplicate_examples": examples, "affected_entities": df[duplicates][self.entity_column].nunique()},
                recommendation="Review duplicates - may need deduplication logic", duplicate_count=duplicate_count)

        return self._pass_result("No duplicate events found")

    def _pass_result(self, message: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            check_id=self.check_id, check_name=self.check_name, passed=True,
            severity=Severity.INFO, message=message, duplicate_count=0)


class TemporalGapCheck(TemporalQualityCheck):
    FREQ_TO_DAYS = {"D": 1, "W": 7, "M": 30, "Q": 90, "Y": 365, "H": 1/24, "T": 1/1440, "min": 1/1440}

    def __init__(self, time_column: str, expected_frequency: str = "D", max_gap_multiple: float = 3.0):
        super().__init__("TQ002", "Temporal Gaps", Severity.MEDIUM)
        self.time_column = time_column
        self.expected_frequency = expected_frequency
        self.max_gap_multiple = max_gap_multiple

    def run(self, df: DataFrame) -> TemporalQualityResult:
        if len(df) < 2:
            return self._pass_result("Insufficient data to check gaps")

        time_col = pd.to_datetime(df.sort_values(self.time_column)[self.time_column])
        diffs_days = time_col.diff().dropna().dt.total_seconds() / 86400
        expected_days = self.FREQ_TO_DAYS.get(self.expected_frequency, 1)
        threshold_days = expected_days * self.max_gap_multiple

        large_gaps = diffs_days[diffs_days > threshold_days]
        max_gap = float(diffs_days.max()) if len(diffs_days) > 0 else 0

        if len(large_gaps) > 0:
            return TemporalQualityResult(
                check_id=self.check_id, check_name=self.check_name, passed=False, severity=self.severity,
                message=f"Found {len(large_gaps)} gaps exceeding {threshold_days:.1f} days",
                details={"threshold_days": threshold_days, "expected_frequency": self.expected_frequency,
                         "gap_locations": large_gaps.index.tolist()[:10]},
                recommendation="Investigate data collection gaps or missing data",
                gap_count=len(large_gaps), max_gap_days=max_gap)

        return TemporalQualityResult(
            check_id=self.check_id, check_name=self.check_name, passed=True, severity=Severity.INFO,
            message="No significant temporal gaps detected", gap_count=0, max_gap_days=max_gap)

    def _pass_result(self, message: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            check_id=self.check_id, check_name=self.check_name, passed=True,
            severity=Severity.INFO, message=message, gap_count=0, max_gap_days=0)


class FutureDateCheck(TemporalQualityCheck):
    def __init__(self, time_column: str, reference_date: Optional[pd.Timestamp] = None):
        super().__init__("TQ003", "Future Dates", Severity.HIGH)
        self.time_column = time_column
        self.reference_date = reference_date or pd.Timestamp.now()

    def run(self, df: DataFrame) -> TemporalQualityResult:
        if len(df) == 0:
            return self._pass_result("No data to check")

        time_col = pd.to_datetime(df[self.time_column])
        future_mask = time_col > self.reference_date
        future_count = future_mask.sum()

        if future_count > 0:
            return TemporalQualityResult(
                check_id=self.check_id, check_name=self.check_name, passed=False, severity=self.severity,
                message=f"Found {future_count} events with future dates",
                details={"reference_date": str(self.reference_date),
                         "future_date_examples": [str(d) for d in time_col[future_mask].head(10).tolist()]},
                recommendation="Review data entry or timestamp handling", future_count=future_count)

        return self._pass_result("No future dates detected")

    def _pass_result(self, message: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            check_id=self.check_id, check_name=self.check_name, passed=True,
            severity=Severity.INFO, message=message, future_count=0)


class EventOrderCheck(TemporalQualityCheck):
    def __init__(self, entity_column: str, time_column: str):
        super().__init__("TQ004", "Event Ordering", Severity.LOW)
        self.entity_column = entity_column
        self.time_column = time_column

    def run(self, df: DataFrame) -> TemporalQualityResult:
        if len(df) < 2:
            return self._pass_result("Insufficient data to check ordering")

        df_check = df.assign(_parsed_time=pd.to_datetime(df[self.time_column]))
        collision_counts = df_check.groupby([self.entity_column, "_parsed_time"]).size()
        ambiguous = collision_counts[collision_counts > 1]
        ambiguous_count = ambiguous.sum() - len(ambiguous)

        if ambiguous_count > 0:
            return TemporalQualityResult(
                check_id=self.check_id, check_name=self.check_name, passed=True, severity=Severity.LOW,
                message=f"{ambiguous_count} events have ambiguous ordering (same timestamp)",
                details={"collision_groups": len(ambiguous), "total_ambiguous_events": int(ambiguous.sum())},
                recommendation="Consider adding sequence numbers for same-timestamp events",
                ambiguous_count=ambiguous_count)

        return self._pass_result("Event ordering is unambiguous")

    def _pass_result(self, message: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            check_id=self.check_id, check_name=self.check_name, passed=True,
            severity=Severity.INFO, message=message, ambiguous_count=0)


@dataclass
class TemporalQualityScore:
    score: float
    grade: str
    check_scores: list
    passed: int
    total: int

    @property
    def grade_emoji(self) -> str:
        return {"A": "🏆", "B": "✅", "C": "⚠️", "D": "❌"}.get(self.grade, "")

    @property
    def grade_message(self) -> str:
        return {"A": "Excellent - ready for feature engineering", "B": "Good - minor issues, proceed with caution",
                "C": "Fair - address issues before proceeding", "D": "Poor - significant investigation needed"}.get(self.grade, "")


class TemporalQualityReporter:
    ML_IMPACTS = {
        "TQ001": {"impacts": [("Event counts", "Inflated metrics"), ("Aggregations", "Skewed"), ("Sequences", "Artificial patterns")],
                  "fix": "df.drop_duplicates(subset=[entity, time], keep='first')"},
        "TQ002": {"impacts": [("Rolling features", "Low during gaps"), ("Recency", "Inflated"), ("Seasonality", "Distorted")],
                  "fix": "Document gaps; add df['has_gap'] indicator"},
        "TQ003": {"impacts": [("Data leakage", "Future in training"), ("Time splits", "Broken"), ("Recency", "Negative values")],
                  "fix": "df = df[df[time_col] <= reference_date]"},
        "TQ004": {"impacts": [("Sequences", "Undefined order"), ("State tracking", "Ambiguous"), ("Lags", "Unclear")],
                  "fix": "Add sequence: df['seq'] = df.groupby(entity).cumcount()"}
    }

    def __init__(self, results: list, total_rows: int):
        self.results = results
        self.total_rows = total_rows
        self._calculate_scores()

    def _calculate_scores(self):
        self.check_scores = []
        for r in self.results:
            issue_count = r.duplicate_count or r.gap_count or r.future_count or r.ambiguous_count or 0
            score = self._score_from_issues(issue_count, self.total_rows)
            pct = (issue_count / self.total_rows * 100) if self.total_rows > 0 else 0
            self.check_scores.append({
                "check_id": r.check_id, "name": r.check_name, "result": r,
                "issues": issue_count, "pct": pct, "score": score, "contribution": score * 0.25})
        self.quality_score = sum(c["contribution"] for c in self.check_scores)
        self.grade = "A" if self.quality_score >= 90 else "B" if self.quality_score >= 75 else "C" if self.quality_score >= 60 else "D"
        self.passed = sum(1 for r in self.results if r.passed)

    def _score_from_issues(self, issues: int, total: int) -> float:
        if total == 0 or issues == 0:
            return 100.0
        pct = (issues / total) * 100
        if pct < 0.1:
            return 99.0
        if pct < 1.0:
            return 95.0 - (pct * 5)
        if pct < 5.0:
            return 90.0 - (pct * 4)
        if pct < 20.0:
            return 70.0 - (pct * 2)
        return max(0, 30.0 - pct)

    def get_score(self) -> TemporalQualityScore:
        return TemporalQualityScore(
            score=self.quality_score, grade=self.grade,
            check_scores=self.check_scores, passed=self.passed, total=len(self.results))

    def print_results(self):
        severity_icons = {Severity.HIGH: "🔴", Severity.MEDIUM: "🟠", Severity.LOW: "🟡", Severity.INFO: "🔵"}
        print("=" * 70 + "\nTEMPORAL QUALITY CHECK RESULTS\n" + "=" * 70)
        print(f"\n📋 Summary: {self.passed}/{len(self.results)} checks passed\n")

        for c in self.check_scores:
            r = c["result"]
            print(f"{'✅' if r.passed else '❌'} [{r.check_id}] {r.check_name}")
            print(f"   {severity_icons.get(r.severity, '⚪')} Severity: {r.severity.value} | {r.message}")

            if c["issues"] > 0 and r.check_id in self.ML_IMPACTS:
                impact = self.ML_IMPACTS[r.check_id]
                print(f"\n   📊 Impact ({c['issues']:,} issues = {c['pct']:.2f}%):")
                for area, problem in impact["impacts"]:
                    print(f"      • {area}: {problem}")
                print(f"   🛠️ Fix: {impact['fix']}")
            elif r.recommendation:
                print(f"   💡 {r.recommendation}")
            print()

    def print_score(self, bar_width: int = 40):
        grade_emoji = {"A": "🏆", "B": "✅", "C": "⚠️", "D": "❌"}[self.grade]
        print("\n" + "=" * 70)
        print(f"QUALITY SCORE: {self.quality_score:.0f}/100  {grade_emoji} Grade {self.grade}\n" + "=" * 70)

        filled = int((self.quality_score / 100) * bar_width)
        print(f"\n  Total: [{'█' * filled}{'░' * (bar_width - filled)}] {self.quality_score:.0f}%\n")

        for c in self.check_scores:
            filled = int((c["contribution"] / 25) * 20)
            bar = f"[{'█' * filled}{'░' * (20 - filled)}] {c['contribution']:.1f}/25"
            status = "✓" if c["issues"] == 0 else "△" if c["pct"] < 1 else "✗"
            issues_str = f"{c['issues']:,} issues" if c["issues"] > 0 else "no issues"
            print(f"  {status} {c['name']:<18} {bar}  ({issues_str})")

        grade_messages = {"A": "Excellent - ready for feature engineering", "B": "Good - minor issues, proceed with caution",
                         "C": "Fair - address issues before proceeding", "D": "Poor - significant investigation needed"}
        print(f"\n  Grade {self.grade}: {grade_messages[self.grade]}")

    def to_dict(self) -> dict:
        return {
            "temporal_quality_score": self.quality_score, "temporal_quality_grade": self.grade,
            "checks_passed": self.passed, "checks_total": len(self.results),
            "issues": {
                "duplicate_events": self.results[0].duplicate_count if len(self.results) > 0 else 0,
                "temporal_gaps": self.results[1].gap_count if len(self.results) > 1 else 0,
                "future_dates": self.results[2].future_count if len(self.results) > 2 else 0,
                "ambiguous_ordering": self.results[3].ambiguous_count if len(self.results) > 3 else 0}}
