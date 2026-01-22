"""Noise robustness testing probes."""

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series
from customer_retention.core.components.enums import Severity
from sklearn.metrics import roc_auc_score


@dataclass
class NoiseCheck:
    check_id: str
    metric: str
    severity: Severity
    recommendation: str
    value: float = 0.0


@dataclass
class NoiseResult:
    passed: bool
    checks: List[NoiseCheck] = field(default_factory=list)
    degradation_curve: List[Dict[str, float]] = field(default_factory=list)
    robustness_score: float = 1.0
    feature_importance: Dict[str, float] = field(default_factory=dict)


class NoiseTester:
    NOISE_LEVELS = {"low": 0.01, "medium": 0.05, "high": 0.10, "extreme": 0.20}
    DROPOUT_LEVELS = {"low": 0.05, "medium": 0.10, "high": 0.20, "extreme": 0.30}
    DEGRADATION_LOW_THRESHOLD = 0.10
    DEGRADATION_MEDIUM_THRESHOLD = 0.20

    def test_gaussian_noise(self, model, X: DataFrame, y: Series) -> NoiseResult:
        baseline_score = self._get_score(model, X, y)
        degradation_curve = []
        checks = []
        for level_name, noise_factor in self.NOISE_LEVELS.items():
            X_noisy = X.copy()
            for col in X.columns:
                if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    std = X[col].std()
                    X_noisy[col] = X[col] + np.random.randn(len(X)) * std * noise_factor
            noisy_score = self._get_score(model, X_noisy, y)
            degradation = (baseline_score - noisy_score) / baseline_score if baseline_score > 0 else 0
            degradation_curve.append({
                "noise_level": level_name,
                "noise_factor": noise_factor,
                "score": noisy_score,
                "degradation": degradation,
            })
            if level_name == "low" and degradation > self.DEGRADATION_LOW_THRESHOLD:
                checks.append(NoiseCheck(
                    check_id="NR001",
                    metric="degradation_low_noise",
                    severity=Severity.HIGH,
                    recommendation=f"HIGH: Model fragile to low noise ({degradation:.1%} degradation). Consider regularization.",
                    value=degradation,
                ))
            if level_name == "medium" and degradation > self.DEGRADATION_MEDIUM_THRESHOLD:
                checks.append(NoiseCheck(
                    check_id="NR002",
                    metric="degradation_medium_noise",
                    severity=Severity.MEDIUM,
                    recommendation=f"MEDIUM: Model moderately fragile ({degradation:.1%} at medium noise).",
                    value=degradation,
                ))
        robustness_score = self._compute_robustness(degradation_curve)
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return NoiseResult(passed=len(critical) == 0, checks=checks, degradation_curve=degradation_curve, robustness_score=robustness_score)

    def test_feature_dropout(self, model, X: DataFrame, y: Series) -> NoiseResult:
        baseline_score = self._get_score(model, X, y)
        degradation_curve = []
        feature_importance = {}
        for col in X.columns:
            X_dropped = X.copy()
            X_dropped[col] = 0
            dropped_score = self._get_score(model, X_dropped, y)
            importance = (baseline_score - dropped_score) / baseline_score if baseline_score > 0 else 0
            feature_importance[col] = importance
        checks = []
        max_importance = max(feature_importance.values()) if feature_importance else 0
        if max_importance > 0.5:
            dominant_feature = max(feature_importance, key=feature_importance.get)
            checks.append(NoiseCheck(
                check_id="NR003",
                metric="single_feature_dependency",
                severity=Severity.HIGH,
                recommendation=f"HIGH: Feature '{dominant_feature}' causes {max_importance:.1%} degradation when dropped. Model too dependent.",
                value=max_importance,
            ))
        for level_name, dropout_rate in self.DROPOUT_LEVELS.items():
            X_dropout = X.copy()
            n_drop = int(len(X.columns) * dropout_rate)
            cols_to_drop = np.random.choice(X.columns, min(n_drop, len(X.columns)), replace=False)
            for col in cols_to_drop:
                X_dropout[col] = 0
            dropout_score = self._get_score(model, X_dropout, y)
            degradation = (baseline_score - dropout_score) / baseline_score if baseline_score > 0 else 0
            degradation_curve.append({
                "dropout_level": level_name,
                "dropout_rate": dropout_rate,
                "score": dropout_score,
                "degradation": degradation,
            })
        robustness_score = self._compute_robustness(degradation_curve)
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return NoiseResult(passed=len(critical) == 0, checks=checks, degradation_curve=degradation_curve, robustness_score=robustness_score, feature_importance=feature_importance)

    def _get_score(self, model, X: DataFrame, y: Series) -> float:
        try:
            y_proba = model.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_proba)
        except Exception:
            return 0.5

    def _compute_robustness(self, degradation_curve: List[Dict]) -> float:
        if not degradation_curve:
            return 1.0
        degradations = [d.get("degradation", 0) for d in degradation_curve]
        return max(0, 1 - np.mean(degradations))

    def run_all(self, model, X: DataFrame, y: Series) -> NoiseResult:
        gaussian_result = self.test_gaussian_noise(model, X, y)
        dropout_result = self.test_feature_dropout(model, X, y)
        all_checks = gaussian_result.checks + dropout_result.checks
        all_degradation = gaussian_result.degradation_curve + dropout_result.degradation_curve
        avg_robustness = (gaussian_result.robustness_score + dropout_result.robustness_score) / 2
        critical = [c for c in all_checks if c.severity == Severity.CRITICAL]
        return NoiseResult(
            passed=len(critical) == 0,
            checks=all_checks,
            degradation_curve=all_degradation,
            robustness_score=avg_robustness,
            feature_importance=dropout_result.feature_importance,
        )
