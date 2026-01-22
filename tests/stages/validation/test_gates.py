from customer_retention.stages.validation import GateResult, Severity, ValidationGate, ValidationIssue


class TestSeverity:
    def test_all_severity_levels(self):
        assert Severity.CRITICAL == "critical"
        assert Severity.HIGH == "high"
        assert Severity.MEDIUM == "medium"
        assert Severity.LOW == "low"
        assert Severity.INFO == "info"


class TestValidationIssue:
    def test_create_minimal_issue(self):
        issue = ValidationIssue(
            code="DQ001",
            message="Test issue",
            severity=Severity.HIGH
        )
        assert issue.code == "DQ001"
        assert issue.message == "Test issue"
        assert issue.severity == Severity.HIGH

    def test_create_full_issue(self):
        issue = ValidationIssue(
            code="DQ001",
            message="Missing values detected",
            severity=Severity.HIGH,
            column="age",
            affected_rows=50,
            affected_pct=0.25,
            recommendation="Impute missing values",
            auto_fixable=True
        )
        assert issue.column == "age"
        assert issue.affected_rows == 50
        assert issue.affected_pct == 0.25
        assert issue.recommendation == "Impute missing values"
        assert issue.auto_fixable is True

    def test_get_display_string_minimal(self):
        issue = ValidationIssue(
            code="DQ001",
            message="Test issue",
            severity=Severity.HIGH
        )
        display = issue.get_display_string()
        assert "HIGH" in display
        assert "DQ001" in display
        assert "Test issue" in display

    def test_get_display_string_with_column(self):
        issue = ValidationIssue(
            code="DQ001",
            message="Missing values",
            severity=Severity.HIGH,
            column="age"
        )
        display = issue.get_display_string()
        assert "Column: age" in display

    def test_get_display_string_with_affected_pct(self):
        issue = ValidationIssue(
            code="DQ001",
            message="Missing values",
            severity=Severity.HIGH,
            affected_pct=0.25
        )
        display = issue.get_display_string()
        assert "Affected: 25.0%" in display

    def test_get_display_string_with_recommendation(self):
        issue = ValidationIssue(
            code="DQ001",
            message="Missing values",
            severity=Severity.HIGH,
            recommendation="Impute values"
        )
        display = issue.get_display_string()
        assert "Recommendation: Impute values" in display

    def test_to_dict(self):
        issue = ValidationIssue(
            code="DQ001",
            message="Test issue",
            severity=Severity.HIGH,
            column="age",
            affected_rows=50,
            affected_pct=0.25
        )
        issue_dict = issue.to_dict()
        assert isinstance(issue_dict, dict)
        assert issue_dict["code"] == "DQ001"
        assert issue_dict["message"] == "Test issue"
        assert issue_dict["severity"] == "high"
        assert issue_dict["column"] == "age"
        assert issue_dict["affected_rows"] == 50


class TestGateResult:
    def test_create_gate_result(self):
        result = GateResult(
            gate_name="TestGate",
            passed=True,
            issues=[],
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.5
        )
        assert result.gate_name == "TestGate"
        assert result.passed is True
        assert len(result.issues) == 0

    def test_get_critical_issues(self):
        issues = [
            ValidationIssue(code="C1", message="Critical", severity=Severity.CRITICAL),
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH),
            ValidationIssue(code="C2", message="Critical2", severity=Severity.CRITICAL)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=False,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.0
        )
        critical = result.get_critical_issues()
        assert len(critical) == 2
        assert all(i.severity == Severity.CRITICAL for i in critical)

    def test_get_high_issues(self):
        issues = [
            ValidationIssue(code="C1", message="Critical", severity=Severity.CRITICAL),
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH),
            ValidationIssue(code="H2", message="High2", severity=Severity.HIGH)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=False,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.0
        )
        high = result.get_high_issues()
        assert len(high) == 2
        assert all(i.severity == Severity.HIGH for i in high)

    def test_has_critical_issues_true(self):
        issues = [
            ValidationIssue(code="C1", message="Critical", severity=Severity.CRITICAL)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=False,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.0
        )
        assert result.has_critical_issues() is True

    def test_has_critical_issues_false(self):
        issues = [
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=True,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.0
        )
        assert result.has_critical_issues() is False

    def test_has_high_issues_true(self):
        issues = [
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=False,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.0
        )
        assert result.has_high_issues() is True

    def test_has_high_issues_false(self):
        issues = [
            ValidationIssue(code="M1", message="Medium", severity=Severity.MEDIUM)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=True,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.0
        )
        assert result.has_high_issues() is False

    def test_get_summary(self):
        issues = [
            ValidationIssue(code="C1", message="Critical", severity=Severity.CRITICAL),
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=False,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.5
        )
        summary = result.get_summary()
        assert "TestGate" in summary
        assert "FAILED" in summary
        assert "critical: 1" in summary
        assert "high: 1" in summary
        assert "1.50s" in summary

    def test_count_issues_by_severity(self):
        issues = [
            ValidationIssue(code="C1", message="Critical", severity=Severity.CRITICAL),
            ValidationIssue(code="C2", message="Critical2", severity=Severity.CRITICAL),
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH),
            ValidationIssue(code="M1", message="Medium", severity=Severity.MEDIUM)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=False,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.0
        )
        counts = result.count_issues_by_severity()
        assert counts["critical"] == 2
        assert counts["high"] == 1
        assert counts["medium"] == 1
        assert "low" not in counts
        assert "info" not in counts

    def test_metadata(self):
        result = GateResult(
            gate_name="TestGate",
            passed=True,
            issues=[],
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.0,
            metadata={"row_count": 100, "column_count": 10}
        )
        assert result.metadata["row_count"] == 100
        assert result.metadata["column_count"] == 10

    def test_to_dict(self):
        issues = [
            ValidationIssue(code="C1", message="Critical", severity=Severity.CRITICAL),
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH)
        ]
        result = GateResult(
            gate_name="TestGate",
            passed=False,
            issues=issues,
            timestamp="2024-01-01T00:00:00",
            duration_seconds=1.5,
            metadata={"test": "value"}
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["gate_name"] == "TestGate"
        assert result_dict["passed"] is False
        assert len(result_dict["issues"]) == 2
        assert result_dict["duration_seconds"] == 1.5
        assert result_dict["metadata"]["test"] == "value"


class MockValidationGate(ValidationGate):
    def run(self, df, config):
        return self.create_result([], 1.0)


class TestValidationGate:
    def test_create_issue_minimal(self):
        gate = MockValidationGate("TestGate")
        issue = gate.create_issue(
            code="DQ001",
            message="Test issue",
            severity=Severity.HIGH
        )
        assert issue.code == "DQ001"
        assert issue.message == "Test issue"
        assert issue.severity == Severity.HIGH

    def test_create_issue_with_affected_rows(self):
        gate = MockValidationGate("TestGate")
        issue = gate.create_issue(
            code="DQ001",
            message="Test issue",
            severity=Severity.HIGH,
            column="age",
            affected_rows=50,
            total_rows=200
        )
        assert issue.affected_rows == 50
        assert issue.affected_pct == 0.25

    def test_create_issue_zero_total_rows(self):
        gate = MockValidationGate("TestGate")
        issue = gate.create_issue(
            code="DQ001",
            message="Test issue",
            severity=Severity.HIGH,
            affected_rows=50,
            total_rows=0
        )
        assert issue.affected_pct is None

    def test_create_result_passed(self):
        gate = MockValidationGate("TestGate")
        issues = [
            ValidationIssue(code="I1", message="Info", severity=Severity.INFO)
        ]
        result = gate.create_result(issues, 1.5)
        assert result.passed is True
        assert result.gate_name == "TestGate"
        assert result.duration_seconds == 1.5

    def test_create_result_failed_on_critical(self):
        gate = MockValidationGate("TestGate")
        issues = [
            ValidationIssue(code="C1", message="Critical", severity=Severity.CRITICAL)
        ]
        result = gate.create_result(issues, 1.0, fail_on_critical=True)
        assert result.passed is False

    def test_create_result_not_failed_on_critical_when_disabled(self):
        gate = MockValidationGate("TestGate")
        issues = [
            ValidationIssue(code="C1", message="Critical", severity=Severity.CRITICAL)
        ]
        result = gate.create_result(issues, 1.0, fail_on_critical=False)
        assert result.passed is True

    def test_create_result_failed_on_high(self):
        gate = MockValidationGate("TestGate")
        issues = [
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH)
        ]
        result = gate.create_result(issues, 1.0, fail_on_high=True)
        assert result.passed is False

    def test_create_result_not_failed_on_high_when_disabled(self):
        gate = MockValidationGate("TestGate")
        issues = [
            ValidationIssue(code="H1", message="High", severity=Severity.HIGH)
        ]
        result = gate.create_result(issues, 1.0, fail_on_high=False)
        assert result.passed is True

    def test_create_result_with_metadata(self):
        gate = MockValidationGate("TestGate")
        result = gate.create_result(
            [],
            1.0,
            metadata={"test": "value"}
        )
        assert result.metadata["test"] == "value"

    def test_gate_has_name(self):
        gate = MockValidationGate("CustomGate")
        assert gate.name == "CustomGate"
