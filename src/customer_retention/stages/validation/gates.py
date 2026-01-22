from abc import ABC, abstractmethod
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel
from customer_retention.core.compat import DataFrame
from customer_retention.core.components.enums import Severity


class ValidationIssue(BaseModel):
    code: str
    message: str
    severity: Severity
    column: Optional[str] = None
    affected_rows: Optional[int] = None
    affected_pct: Optional[float] = None
    recommendation: Optional[str] = None
    auto_fixable: bool = False

    def get_display_string(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.code}: {self.message}"]
        if self.column:
            parts.append(f"Column: {self.column}")
        if self.affected_pct is not None:
            parts.append(f"Affected: {self.affected_pct:.1%}")
        if self.recommendation:
            parts.append(f"Recommendation: {self.recommendation}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return self.model_dump()


class GateResult(BaseModel):
    gate_name: str
    passed: bool
    issues: list[ValidationIssue] = []
    timestamp: str
    duration_seconds: float
    metadata: dict[str, Any] = {}

    def get_critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    def get_high_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.HIGH]

    def has_critical_issues(self) -> bool:
        return len(self.get_critical_issues()) > 0

    def has_high_issues(self) -> bool:
        return len(self.get_high_issues()) > 0

    def get_summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        issue_counts = self.count_issues_by_severity()
        counts_str = ", ".join([f"{sev}: {count}" for sev, count in issue_counts.items()])
        return f"{self.gate_name} - {status} ({counts_str}) in {self.duration_seconds:.2f}s"

    def count_issues_by_severity(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return {k: v for k, v in counts.items() if v > 0}

    def to_dict(self) -> dict:
        return self.model_dump()


class ValidationGate(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, df: DataFrame, config: Any) -> GateResult:
        pass

    def create_issue(self, code: str, message: str, severity: Severity,
                    column: Optional[str] = None, affected_rows: Optional[int] = None,
                    total_rows: Optional[int] = None, recommendation: Optional[str] = None,
                    auto_fixable: bool = False) -> ValidationIssue:
        affected_pct = None
        if affected_rows is not None and total_rows is not None and total_rows > 0:
            affected_pct = affected_rows / total_rows

        return ValidationIssue(
            code=code,
            message=message,
            severity=severity,
            column=column,
            affected_rows=affected_rows,
            affected_pct=affected_pct,
            recommendation=recommendation,
            auto_fixable=auto_fixable
        )

    def create_result(self, issues: list[ValidationIssue], duration: float,
                     fail_on_critical: bool = True, fail_on_high: bool = False,
                     metadata: Optional[dict] = None) -> GateResult:
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        has_high = any(i.severity == Severity.HIGH for i in issues)

        passed = True
        if fail_on_critical and has_critical:
            passed = False
        if fail_on_high and has_high:
            passed = False

        return GateResult(
            gate_name=self.name,
            passed=passed,
            issues=issues,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            metadata=metadata or {}
        )
