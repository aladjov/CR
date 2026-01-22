"""Cross-validation stability analysis probes."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from customer_retention.core.components.enums import Severity


@dataclass
class CVCheck:
    check_id: str
    metric: str
    severity: Severity
    recommendation: str
    value: float = 0.0


@dataclass
class CVAnalysisResult:
    passed: bool
    checks: List[CVCheck] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    fold_analysis: List[Dict[str, float]] = field(default_factory=list)
    best_worst_gap: float = 0.0
    outlier_folds: List[int] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class CVAnalyzer:
    STD_CRITICAL = 0.15
    STD_HIGH = 0.10
    STD_MEDIUM = 0.05
    CV_TEST_HIGH = 0.10
    CV_TEST_MEDIUM = -0.10

    def analyze_variance(self, cv_scores: List[float]) -> CVAnalysisResult:
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        checks = []
        severity, check_id = self._classify_variance(cv_std)
        checks.append(CVCheck(
            check_id=check_id,
            metric="cv_std",
            severity=severity,
            recommendation=self._variance_recommendation(cv_std),
            value=cv_std,
        ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return CVAnalysisResult(passed=len(critical) == 0, checks=checks, cv_mean=cv_mean, cv_std=cv_std)

    def _classify_variance(self, cv_std: float) -> tuple:
        if cv_std > self.STD_CRITICAL:
            return Severity.CRITICAL, "CV001"
        if cv_std > self.STD_HIGH:
            return Severity.HIGH, "CV002"
        if cv_std > self.STD_MEDIUM:
            return Severity.MEDIUM, "CV003"
        return Severity.INFO, "CV004"

    def _variance_recommendation(self, cv_std: float) -> str:
        if cv_std > self.STD_CRITICAL:
            return f"CRITICAL: CV std {cv_std:.3f} is very high. Model is unstable. Use more data or robust methods."
        if cv_std > self.STD_HIGH:
            return f"HIGH: CV std {cv_std:.3f} is high. Consider ensemble methods or robust scaling."
        if cv_std > self.STD_MEDIUM:
            return f"MEDIUM: CV std {cv_std:.3f} is moderate. Monitor closely."
        return f"OK: CV std {cv_std:.3f} indicates stable model."

    def analyze_folds(self, cv_scores: List[float]) -> CVAnalysisResult:
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        fold_analysis = [{"fold": i, "score": score, "deviation": score - cv_mean} for i, score in enumerate(cv_scores)]
        best_worst_gap = max(cv_scores) - min(cv_scores)
        outlier_folds = [i for i, score in enumerate(cv_scores) if abs(score - cv_mean) > 2 * cv_std]
        checks = []
        if outlier_folds:
            checks.append(CVCheck(
                check_id="CV005",
                metric="outlier_folds",
                severity=Severity.HIGH,
                recommendation=f"HIGH: Folds {outlier_folds} are outliers (>2 std from mean). Investigate data heterogeneity.",
                value=len(outlier_folds),
            ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return CVAnalysisResult(
            passed=len(critical) == 0,
            checks=checks,
            cv_mean=cv_mean,
            cv_std=cv_std,
            fold_analysis=fold_analysis,
            best_worst_gap=best_worst_gap,
            outlier_folds=outlier_folds,
        )

    def compare_cv_test(self, cv_mean: float, test_score: float) -> CVAnalysisResult:
        gap = cv_mean - test_score
        checks = []
        severity, check_id = self._classify_cv_test_gap(gap)
        checks.append(CVCheck(
            check_id=check_id,
            metric="cv_test_gap",
            severity=severity,
            recommendation=self._cv_test_recommendation(gap, cv_mean, test_score),
            value=gap,
        ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return CVAnalysisResult(passed=len(critical) == 0, checks=checks, cv_mean=cv_mean)

    def _classify_cv_test_gap(self, gap: float) -> tuple:
        if gap > self.CV_TEST_HIGH:
            return Severity.HIGH, "CV010"
        if gap < self.CV_TEST_MEDIUM:
            return Severity.MEDIUM, "CV011"
        return Severity.INFO, "CV012"

    def _cv_test_recommendation(self, gap: float, cv_mean: float, test_score: float) -> str:
        if gap > self.CV_TEST_HIGH:
            return f"HIGH: CV mean {cv_mean:.3f} >> test {test_score:.3f}. CV may be overly optimistic."
        if gap < self.CV_TEST_MEDIUM:
            return f"MEDIUM: CV mean {cv_mean:.3f} << test {test_score:.3f}. CV may be pessimistic."
        return f"OK: CV mean {cv_mean:.3f} ≈ test {test_score:.3f}. Good estimate."

    def run_all(self, cv_scores: List[float], test_score: Optional[float] = None) -> CVAnalysisResult:
        variance_result = self.analyze_variance(cv_scores)
        fold_result = self.analyze_folds(cv_scores)
        all_checks = variance_result.checks + fold_result.checks
        if test_score is not None:
            cv_test_result = self.compare_cv_test(variance_result.cv_mean, test_score)
            all_checks += cv_test_result.checks
        critical = [c for c in all_checks if c.severity == Severity.CRITICAL]
        recommendations = [c.recommendation for c in all_checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        return CVAnalysisResult(
            passed=len(critical) == 0,
            checks=all_checks,
            cv_mean=variance_result.cv_mean,
            cv_std=variance_result.cv_std,
            fold_analysis=fold_result.fold_analysis,
            best_worst_gap=fold_result.best_worst_gap,
            outlier_folds=fold_result.outlier_folds,
            recommendations=recommendations,
        )
