from customer_retention.analysis.diagnostics.leakage_detector import (
    LeakageCheck,
    LeakageResult,
)
from customer_retention.core.components.enums import Severity
from customer_retention.stages.modeling.feature_spec import LeakageExclusion


class TestLeakageCheckToExclusion:
    def test_maps_all_fields(self):
        chk = LeakageCheck(
            check_id="LD063", feature="event_count_365d", severity=Severity.HIGH,
            recommendation="180d horizon inside 365d window",
        )
        excl = chk.to_exclusion()
        assert isinstance(excl, LeakageExclusion)
        assert excl.column == "event_count_365d"
        assert excl.code == "LD063"
        assert excl.severity == "HIGH"
        assert excl.rationale == "180d horizon inside 365d window"

    def test_critical_severity_uppercased(self):
        chk = LeakageCheck(
            check_id="LD062", feature="event_count_365d_is_zero",
            severity=Severity.CRITICAL, recommendation="zero-flag on horizon window",
        )
        assert chk.to_exclusion().severity == "CRITICAL"


class TestLeakageResultToExclusions:
    def test_filters_below_min_severity(self):
        result = LeakageResult(
            passed=True,
            checks=[
                LeakageCheck(check_id="LD000", feature="a", severity=Severity.INFO, recommendation="ok"),
                LeakageCheck(check_id="LD003", feature="b", severity=Severity.MEDIUM, recommendation="mid"),
                LeakageCheck(check_id="LD002", feature="c", severity=Severity.HIGH, recommendation="high"),
                LeakageCheck(check_id="LD001", feature="d", severity=Severity.CRITICAL, recommendation="crit"),
            ],
        )
        excls = result.to_exclusions(min_severity=Severity.HIGH)
        assert sorted(e.column for e in excls) == ["c", "d"]

    def test_dedupes_feature_keeping_highest_severity(self):
        result = LeakageResult(
            passed=True,
            checks=[
                LeakageCheck(check_id="LD063", feature="x", severity=Severity.HIGH, recommendation="hi"),
                LeakageCheck(check_id="LD062", feature="x", severity=Severity.CRITICAL, recommendation="crit"),
            ],
        )
        excls = result.to_exclusions(min_severity=Severity.HIGH)
        assert len(excls) == 1
        assert excls[0].severity == "CRITICAL"
        assert excls[0].code == "LD062"

    def test_empty_when_no_matches(self):
        result = LeakageResult(
            passed=True,
            checks=[LeakageCheck(check_id="LD000", feature="a", severity=Severity.INFO, recommendation="ok")],
        )
        assert result.to_exclusions(min_severity=Severity.HIGH) == []
