"""Business Sense Gate (Checkpoint 5) for model validation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from customer_retention.core.components.enums import Severity


@dataclass
class BusinessCheck:
    check_id: str
    name: str
    severity: Severity
    recommendation: str
    passed: bool = True


@dataclass
class BusinessSenseResult:
    passed: bool
    checks: List[BusinessCheck] = field(default_factory=list)
    critical_issues: List[BusinessCheck] = field(default_factory=list)
    review_notes: List[str] = field(default_factory=list)
    recommendation: str = ""
    roi: float = 0.0


EXPECTED_TOP_FEATURES = {
    "days_since_last_order", "recency", "tenure", "tenure_days", "email_engagement",
    "engagement", "order_frequency", "avgorder", "service_adoption", "paperless",
    "refill", "doorstep", "eclickrate", "eopenrate", "ordfreq",
}


class BusinessSenseGate:
    SUSPICIOUS_PR_AUC = 0.90

    def __init__(self, required_sign_offs: Optional[List[str]] = None):
        self.required_sign_offs = required_sign_offs or []
        self._sign_offs: Dict[str, str] = {}

    def check_feature_explainability(self, feature_importance: Dict[str, float]) -> BusinessSenseResult:
        checks = []
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        unexplainable = []
        for feature, importance in top_features:
            feature_lower = feature.lower()
            is_expected = any(expected in feature_lower for expected in EXPECTED_TOP_FEATURES)
            if not is_expected and importance > 0.15:
                unexplainable.append(feature)
        if unexplainable:
            checks.append(BusinessCheck(
                check_id="BS001",
                name="Top features explainable",
                severity=Severity.HIGH,
                recommendation=f"HIGH: Features {unexplainable} are unexpectedly important. Verify business logic.",
                passed=False,
            ))
        else:
            checks.append(BusinessCheck(
                check_id="BS001",
                name="Top features explainable",
                severity=Severity.INFO,
                recommendation="OK: Top features align with business expectations.",
                passed=True,
            ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL and not c.passed]
        return BusinessSenseResult(passed=len(critical) == 0, checks=checks, critical_issues=critical)

    def check_roi(self, cost_benefit: Dict[str, float]) -> BusinessSenseResult:
        checks = []
        intervention_cost = cost_benefit.get("intervention_cost", 0)
        customer_value = cost_benefit.get("customer_value", 0)
        expected_lift = cost_benefit.get("expected_lift", 0)
        target_population = cost_benefit.get("target_population", 0)
        expected_saves = target_population * expected_lift
        revenue_saved = expected_saves * customer_value
        total_cost = target_population * intervention_cost
        roi = (revenue_saved - total_cost) / total_cost if total_cost > 0 else 0
        if roi < 0:
            checks.append(BusinessCheck(
                check_id="BS005",
                name="ROI is positive",
                severity=Severity.CRITICAL,
                recommendation=f"CRITICAL: ROI is negative ({roi:.1%}). Intervention costs exceed expected benefit.",
                passed=False,
            ))
        elif roi < 0.5:
            checks.append(BusinessCheck(
                check_id="BS005",
                name="ROI is positive",
                severity=Severity.MEDIUM,
                recommendation=f"MEDIUM: ROI is low ({roi:.1%}). Consider optimizing intervention targeting.",
                passed=True,
            ))
        else:
            checks.append(BusinessCheck(
                check_id="BS005",
                name="ROI is positive",
                severity=Severity.INFO,
                recommendation=f"OK: ROI is positive ({roi:.1%}).",
                passed=True,
            ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL and not c.passed]
        return BusinessSenseResult(passed=len(critical) == 0, checks=checks, critical_issues=critical, roi=roi)

    def run(self, metrics: Dict[str, float], feature_importance: Dict[str, float], cost_benefit: Optional[Dict[str, float]] = None) -> BusinessSenseResult:
        all_checks = []
        feature_result = self.check_feature_explainability(feature_importance)
        all_checks.extend(feature_result.checks)
        pr_auc = metrics.get("pr_auc_test", metrics.get("pr_auc", 0))
        if pr_auc > self.SUSPICIOUS_PR_AUC:
            all_checks.append(BusinessCheck(
                check_id="BS000",
                name="Performance sanity check",
                severity=Severity.CRITICAL,
                recommendation=f"CRITICAL: PR-AUC {pr_auc:.2f} is suspiciously high. Check for data leakage.",
                passed=False,
            ))
        roi = 0.0
        if cost_benefit:
            roi_result = self.check_roi(cost_benefit)
            all_checks.extend(roi_result.checks)
            roi = roi_result.roi
        sign_off_result = self.check_sign_offs()
        all_checks.extend(sign_off_result.checks)
        critical = [c for c in all_checks if c.severity == Severity.CRITICAL and not c.passed]
        recommendation = self._generate_recommendation(all_checks)
        return BusinessSenseResult(
            passed=len(critical) == 0,
            checks=all_checks,
            critical_issues=critical,
            review_notes=[c.recommendation for c in all_checks],
            recommendation=recommendation,
            roi=roi,
        )

    def add_sign_off(self, role: str, comment: str):
        self._sign_offs[role] = comment

    def get_sign_offs(self) -> Dict[str, str]:
        return self._sign_offs.copy()

    def check_sign_offs(self) -> BusinessSenseResult:
        checks = []
        missing = [role for role in self.required_sign_offs if role not in self._sign_offs]
        if missing:
            checks.append(BusinessCheck(
                check_id="BS007",
                name="Stakeholder approval",
                severity=Severity.HIGH,
                recommendation=f"HIGH: Missing sign-offs from: {', '.join(missing)}",
                passed=False,
            ))
        else:
            checks.append(BusinessCheck(
                check_id="BS007",
                name="Stakeholder approval",
                severity=Severity.INFO,
                recommendation="OK: All required sign-offs obtained.",
                passed=True,
            ))
        failed = [c for c in checks if not c.passed]
        return BusinessSenseResult(passed=len(failed) == 0, checks=checks)

    def _generate_recommendation(self, checks: List[BusinessCheck]) -> str:
        critical = [c for c in checks if c.severity == Severity.CRITICAL and not c.passed]
        high = [c for c in checks if c.severity == Severity.HIGH and not c.passed]
        if critical:
            return "BLOCKED: Critical business issues must be resolved before deployment."
        if high:
            return "CAUTION: High-priority issues should be addressed before production."
        return "READY: Model passes business sense validation. Proceed to deployment."
