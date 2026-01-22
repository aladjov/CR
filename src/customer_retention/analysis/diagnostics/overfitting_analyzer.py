"""Overfitting analysis probes for model validation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import learning_curve

from customer_retention.core.compat import DataFrame, Series
from customer_retention.core.components.enums import Severity


@dataclass
class OverfittingCheck:
    check_id: str
    metric: str
    severity: Severity
    recommendation: str
    train_value: float = 0.0
    test_value: float = 0.0
    gap: float = 0.0


@dataclass
class OverfittingResult:
    passed: bool
    checks: List[OverfittingCheck] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    learning_curve: List[Dict[str, float]] = field(default_factory=list)
    diagnosis: Optional[str] = None
    sample_to_feature_ratio: float = 0.0


class OverfittingAnalyzer:
    GAP_CRITICAL = 0.15
    GAP_HIGH = 0.10
    GAP_MEDIUM = 0.05
    RATIO_CRITICAL = 10
    RATIO_HIGH = 50
    DEPTH_HIGH = 15
    ESTIMATORS_HIGH = 500

    def analyze_train_test_gap(self, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> OverfittingResult:
        checks = []
        for metric in train_metrics:
            if metric in test_metrics:
                train_val = train_metrics[metric]
                test_val = test_metrics[metric]
                gap = train_val - test_val
                severity, check_id = self._classify_gap(gap)
                checks.append(OverfittingCheck(
                    check_id=check_id,
                    metric=metric,
                    severity=severity,
                    recommendation=self._gap_recommendation(metric, gap),
                    train_value=train_val,
                    test_value=test_val,
                    gap=gap,
                ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        recommendations = [c.recommendation for c in checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        return OverfittingResult(passed=len(critical) == 0, checks=checks, recommendations=recommendations)

    def _classify_gap(self, gap: float) -> tuple:
        if gap > self.GAP_CRITICAL:
            return Severity.CRITICAL, "OF001"
        if gap > self.GAP_HIGH:
            return Severity.HIGH, "OF002"
        if gap > self.GAP_MEDIUM:
            return Severity.MEDIUM, "OF003"
        return Severity.INFO, "OF004"

    def _gap_recommendation(self, metric: str, gap: float) -> str:
        if gap > self.GAP_CRITICAL:
            return f"CRITICAL: {metric} gap {gap:.1%} indicates severe overfitting. Reduce model complexity, add regularization."
        if gap > self.GAP_HIGH:
            return f"HIGH: {metric} gap {gap:.1%} indicates moderate overfitting. Consider feature selection or regularization."
        if gap > self.GAP_MEDIUM:
            return f"MEDIUM: {metric} gap {gap:.1%} shows mild overfitting. Monitor closely."
        return f"OK: {metric} gap {gap:.1%} shows good generalization."

    def analyze_learning_curve(self, model, X: DataFrame, y: Series, cv: int = 5) -> OverfittingResult:
        try:
            train_sizes = np.linspace(0.2, 1.0, 5)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=cv, scoring="roc_auc", random_state=42
            )
            curve_data = []
            for i, size in enumerate(train_sizes_abs):
                curve_data.append({
                    "train_size": int(size),
                    "train_score": float(np.mean(train_scores[i])),
                    "val_score": float(np.mean(val_scores[i])),
                })
            diagnosis = self._diagnose_learning_curve(curve_data)
            return OverfittingResult(passed=True, learning_curve=curve_data, diagnosis=diagnosis)
        except Exception:
            return OverfittingResult(passed=True, learning_curve=[], diagnosis="Unable to generate learning curve")

    def _diagnose_learning_curve(self, curve_data: List[Dict[str, float]]) -> str:
        if not curve_data:
            return "Insufficient data for diagnosis"
        last = curve_data[-1]
        first = curve_data[0]
        train_score = last["train_score"]
        val_score = last["val_score"]
        gap = train_score - val_score
        val_improvement = last["val_score"] - first["val_score"]
        if gap < 0.05 and val_score > 0.7:
            return "Good fit: Both curves converged at high performance"
        if gap > 0.15:
            return "Overfitting: High train score but low validation. Reduce complexity."
        if val_score < 0.6 and train_score < 0.7:
            return "Underfitting: Both scores low. Increase model complexity or add features."
        if val_improvement > 0.05:
            return "More data may help: Validation still improving with more samples."
        return "Validation plateau: More data unlikely to help significantly."

    def analyze_complexity(self, X: DataFrame, y: Series) -> OverfittingResult:
        n_samples, n_features = X.shape
        ratio = n_samples / max(n_features, 1)
        checks = []
        severity, check_id = self._classify_ratio(ratio)
        if severity != Severity.INFO:
            checks.append(OverfittingCheck(
                check_id=check_id,
                metric="sample_to_feature_ratio",
                severity=severity,
                recommendation=self._ratio_recommendation(ratio, n_samples, n_features),
                train_value=ratio,
            ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        recommendations = [c.recommendation for c in checks if c.severity in [Severity.CRITICAL, Severity.HIGH]]
        return OverfittingResult(passed=len(critical) == 0, checks=checks, recommendations=recommendations, sample_to_feature_ratio=ratio)

    def _classify_ratio(self, ratio: float) -> tuple:
        if ratio < self.RATIO_CRITICAL:
            return Severity.CRITICAL, "OF010"
        if ratio < self.RATIO_HIGH:
            return Severity.HIGH, "OF011"
        return Severity.INFO, "OF000"

    def _ratio_recommendation(self, ratio: float, n_samples: int, n_features: int) -> str:
        if ratio < self.RATIO_CRITICAL:
            suggested_features = n_samples // 10
            return f"CRITICAL: Ratio {ratio:.1f}:1 is too low. Reduce to {suggested_features} features or get more data."
        if ratio < self.RATIO_HIGH:
            return f"HIGH: Ratio {ratio:.1f}:1 is concerning. Use L1 regularization and monitor closely."
        return f"OK: Ratio {ratio:.1f}:1 is adequate."

    def analyze_model_complexity(self, model_params: Dict[str, Any]) -> OverfittingResult:
        checks = []
        if "max_depth" in model_params and model_params["max_depth"]:
            depth = model_params["max_depth"]
            if depth > self.DEPTH_HIGH:
                checks.append(OverfittingCheck(
                    check_id="OF012",
                    metric="max_depth",
                    severity=Severity.HIGH,
                    recommendation=f"HIGH: max_depth={depth} may cause overfitting. Consider depth <= 10.",
                    train_value=depth,
                ))
        if "n_estimators" in model_params:
            n_est = model_params["n_estimators"]
            if n_est > self.ESTIMATORS_HIGH and "regularization" not in model_params:
                checks.append(OverfittingCheck(
                    check_id="OF013",
                    metric="n_estimators",
                    severity=Severity.MEDIUM,
                    recommendation=f"MEDIUM: n_estimators={n_est} without regularization may cause overfitting.",
                    train_value=n_est,
                ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return OverfittingResult(passed=len(critical) == 0, checks=checks)

    def run_all(self, model, X: DataFrame, y: Series, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> OverfittingResult:
        gap_result = self.analyze_train_test_gap(train_metrics, test_metrics)
        complexity_result = self.analyze_complexity(X, y)
        learning_result = self.analyze_learning_curve(model, X, y)
        all_checks = gap_result.checks + complexity_result.checks
        all_recommendations = gap_result.recommendations + complexity_result.recommendations
        critical = [c for c in all_checks if c.severity == Severity.CRITICAL]
        return OverfittingResult(
            passed=len(critical) == 0,
            checks=all_checks,
            recommendations=list(set(all_recommendations)),
            learning_curve=learning_result.learning_curve,
            diagnosis=learning_result.diagnosis,
            sample_to_feature_ratio=complexity_result.sample_to_feature_ratio,
        )
