"""Model Validity Gate (Checkpoint 4) for customer retention model validation."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from customer_retention.core.components.enums import Severity


@dataclass
class ValidityIssue:
    check_id: str
    severity: Severity
    description: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ModelValidityResult:
    passed: bool
    critical_issues: List[ValidityIssue]
    high_issues: List[ValidityIssue]
    warnings: List[ValidityIssue]
    recommendation: str
    diagnostic_hints: List[str]


class ModelValidityGate:
    def __init__(
        self,
        pr_auc_threshold_suspicious: float = 0.90,
        roc_auc_threshold_suspicious: float = 0.95,
        train_test_gap_severe: float = 0.15,
        train_test_gap_moderate: float = 0.10,
        cv_std_threshold: float = 0.10,
        recall_threshold_poor: float = 0.30,
    ):
        self.pr_auc_threshold_suspicious = pr_auc_threshold_suspicious
        self.roc_auc_threshold_suspicious = roc_auc_threshold_suspicious
        self.train_test_gap_severe = train_test_gap_severe
        self.train_test_gap_moderate = train_test_gap_moderate
        self.cv_std_threshold = cv_std_threshold
        self.recall_threshold_poor = recall_threshold_poor

    def run(self, metrics: Dict[str, float]) -> ModelValidityResult:
        critical_issues: List[ValidityIssue] = []
        high_issues: List[ValidityIssue] = []
        warnings: List[ValidityIssue] = []
        diagnostic_hints: List[str] = []

        self._check_suspicious_pr_auc(metrics, critical_issues, diagnostic_hints)
        self._check_suspicious_roc_auc(metrics, high_issues, diagnostic_hints)
        self._check_overfitting(metrics, critical_issues, high_issues, diagnostic_hints)
        self._check_cv_variance(metrics, high_issues, diagnostic_hints)
        self._check_poor_recall(metrics, high_issues, diagnostic_hints)
        self._check_worse_than_baseline(metrics, critical_issues, diagnostic_hints)
        self._check_perfect_predictions(metrics, critical_issues, diagnostic_hints)

        passed = len(critical_issues) == 0
        recommendation = self._generate_recommendation(passed, critical_issues, high_issues)

        return ModelValidityResult(
            passed=passed,
            critical_issues=critical_issues,
            high_issues=high_issues,
            warnings=warnings,
            recommendation=recommendation,
            diagnostic_hints=diagnostic_hints,
        )

    def _check_suspicious_pr_auc(
        self,
        metrics: Dict[str, float],
        critical_issues: List[ValidityIssue],
        diagnostic_hints: List[str],
    ):
        pr_auc = metrics.get("pr_auc_test", 0)
        if pr_auc > self.pr_auc_threshold_suspicious:
            critical_issues.append(ValidityIssue(
                check_id="MV001",
                severity=Severity.CRITICAL,
                description=f"Suspicious PR-AUC: {pr_auc:.4f} > {self.pr_auc_threshold_suspicious}",
                value=pr_auc,
                threshold=self.pr_auc_threshold_suspicious,
            ))
            diagnostic_hints.append("Check for data leakage - features derived from target")

    def _check_suspicious_roc_auc(
        self,
        metrics: Dict[str, float],
        high_issues: List[ValidityIssue],
        diagnostic_hints: List[str],
    ):
        roc_auc = metrics.get("roc_auc_test", 0)
        if roc_auc > self.roc_auc_threshold_suspicious:
            high_issues.append(ValidityIssue(
                check_id="MV002",
                severity=Severity.HIGH,
                description=f"Suspicious ROC-AUC: {roc_auc:.4f} > {self.roc_auc_threshold_suspicious}",
                value=roc_auc,
                threshold=self.roc_auc_threshold_suspicious,
            ))
            diagnostic_hints.append("Verify feature engineering pipeline for leakage")

    def _check_overfitting(
        self,
        metrics: Dict[str, float],
        critical_issues: List[ValidityIssue],
        high_issues: List[ValidityIssue],
        diagnostic_hints: List[str],
    ):
        gap = metrics.get("train_test_gap", 0)
        if gap > self.train_test_gap_severe:
            critical_issues.append(ValidityIssue(
                check_id="MV003",
                severity=Severity.CRITICAL,
                description=f"Severe overfitting: train-test gap {gap:.4f} > {self.train_test_gap_severe}",
                value=gap,
                threshold=self.train_test_gap_severe,
            ))
            diagnostic_hints.append("Consider more regularization or simpler model")
        elif gap > self.train_test_gap_moderate:
            high_issues.append(ValidityIssue(
                check_id="MV004",
                severity=Severity.HIGH,
                description=f"Moderate overfitting: train-test gap {gap:.4f} > {self.train_test_gap_moderate}",
                value=gap,
                threshold=self.train_test_gap_moderate,
            ))
            diagnostic_hints.append("Monitor overfitting with cross-validation")

    def _check_cv_variance(
        self,
        metrics: Dict[str, float],
        high_issues: List[ValidityIssue],
        diagnostic_hints: List[str],
    ):
        cv_std = metrics.get("cv_std", 0)
        if cv_std > self.cv_std_threshold:
            high_issues.append(ValidityIssue(
                check_id="MV005",
                severity=Severity.HIGH,
                description=f"High CV variance: std {cv_std:.4f} > {self.cv_std_threshold}",
                value=cv_std,
                threshold=self.cv_std_threshold,
            ))
            diagnostic_hints.append("Consider more training data or feature selection")

    def _check_poor_recall(
        self,
        metrics: Dict[str, float],
        high_issues: List[ValidityIssue],
        diagnostic_hints: List[str],
    ):
        recall = metrics.get("recall", 1)
        if recall < self.recall_threshold_poor:
            high_issues.append(ValidityIssue(
                check_id="MV006",
                severity=Severity.HIGH,
                description=f"Poor minority recall: {recall:.4f} < {self.recall_threshold_poor}",
                value=recall,
                threshold=self.recall_threshold_poor,
            ))
            diagnostic_hints.append("Adjust threshold or use class weights")

    def _check_worse_than_baseline(
        self,
        metrics: Dict[str, float],
        critical_issues: List[ValidityIssue],
        diagnostic_hints: List[str],
    ):
        pr_auc = metrics.get("pr_auc_test", 0)
        class_proportion = metrics.get("class_proportion", 0.25)
        if pr_auc < class_proportion:
            critical_issues.append(ValidityIssue(
                check_id="MV007",
                severity=Severity.CRITICAL,
                description=f"Worse than baseline: PR-AUC {pr_auc:.4f} < class proportion {class_proportion:.4f}",
                value=pr_auc,
                threshold=class_proportion,
            ))
            diagnostic_hints.append("Model performs worse than random - check data quality")

    def _check_perfect_predictions(
        self,
        metrics: Dict[str, float],
        critical_issues: List[ValidityIssue],
        diagnostic_hints: List[str],
    ):
        pr_auc = metrics.get("pr_auc_test", 0)
        roc_auc = metrics.get("roc_auc_test", 0)
        recall = metrics.get("recall", 0)

        if pr_auc == 1.0 or roc_auc == 1.0 or recall == 1.0:
            critical_issues.append(ValidityIssue(
                check_id="MV008",
                severity=Severity.CRITICAL,
                description="Perfect predictions detected - almost certainly leakage",
                value=1.0,
            ))
            diagnostic_hints.append("Review all features for direct target leakage")

    def _generate_recommendation(
        self,
        passed: bool,
        critical_issues: List[ValidityIssue],
        high_issues: List[ValidityIssue],
    ) -> str:
        if passed and len(high_issues) == 0:
            return "Proceed with model deployment"
        if passed and len(high_issues) > 0:
            return "Proceed with caution - review high-severity issues"
        return f"Investigate {len(critical_issues)} critical issue(s) before proceeding"
