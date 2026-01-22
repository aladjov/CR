"""Calibration analysis probes for model validation."""

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from customer_retention.core.components.enums import Severity


@dataclass
class CalibrationCheck:
    check_id: str
    metric: str
    severity: Severity
    recommendation: str
    value: float = 0.0


@dataclass
class CalibrationResult:
    passed: bool
    checks: List[CalibrationCheck] = field(default_factory=list)
    brier_score: float = 0.0
    ece: float = 0.0
    mce: float = 0.0
    reliability_data: List[Dict[str, float]] = field(default_factory=list)
    recommendation: str = ""


class CalibrationAnalyzer:
    BRIER_HIGH = 0.20
    BRIER_MEDIUM = 0.15
    ECE_HIGH = 0.10
    MCE_HIGH = 0.30
    N_BINS = 10

    def analyze_brier(self, y_true: np.ndarray, y_proba: np.ndarray) -> CalibrationResult:
        brier = np.mean((y_proba - y_true) ** 2)
        checks = []
        severity, check_id = self._classify_brier(brier)
        if severity != Severity.INFO:
            checks.append(CalibrationCheck(
                check_id=check_id,
                metric="brier_score",
                severity=severity,
                recommendation=self._brier_recommendation(brier),
                value=brier,
            ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return CalibrationResult(passed=len(critical) == 0, checks=checks, brier_score=brier)

    def _classify_brier(self, brier: float) -> tuple:
        if brier > self.BRIER_HIGH:
            return Severity.HIGH, "CA001"
        if brier > self.BRIER_MEDIUM:
            return Severity.MEDIUM, "CA002"
        return Severity.INFO, "CA000"

    def _brier_recommendation(self, brier: float) -> str:
        if brier > self.BRIER_HIGH:
            return f"HIGH: Brier score {brier:.3f} is poor. Apply calibration (Platt scaling or isotonic)."
        if brier > self.BRIER_MEDIUM:
            return f"MEDIUM: Brier score {brier:.3f} is moderate. Consider calibration."
        return f"OK: Brier score {brier:.3f} is acceptable."

    def analyze_calibration(self, y_true: np.ndarray, y_proba: np.ndarray) -> CalibrationResult:
        brier = np.mean((y_proba - y_true) ** 2)
        reliability_data, ece, mce = self._compute_reliability(y_true, y_proba)
        checks = []
        brier_severity, brier_id = self._classify_brier(brier)
        if brier_severity != Severity.INFO:
            checks.append(CalibrationCheck(
                check_id=brier_id, metric="brier_score", severity=brier_severity,
                recommendation=self._brier_recommendation(brier), value=brier,
            ))
        if ece > self.ECE_HIGH:
            checks.append(CalibrationCheck(
                check_id="CA003", metric="ece", severity=Severity.MEDIUM,
                recommendation=f"MEDIUM: ECE {ece:.3f} is high. Calibration recommended.", value=ece,
            ))
        if mce > self.MCE_HIGH:
            checks.append(CalibrationCheck(
                check_id="CA004", metric="mce", severity=Severity.HIGH,
                recommendation=f"HIGH: MCE {mce:.3f} is extreme. Some probability bins are very miscalibrated.", value=mce,
            ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        recommendation = self._global_recommendation(reliability_data, brier, ece)
        return CalibrationResult(
            passed=len(critical) == 0,
            checks=checks,
            brier_score=brier,
            ece=ece,
            mce=mce,
            reliability_data=reliability_data,
            recommendation=recommendation,
        )

    def _compute_reliability(self, y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
        bin_edges = np.linspace(0, 1, self.N_BINS + 1)
        reliability_data = []
        ece_sum = 0.0
        mce = 0.0
        for i in range(self.N_BINS):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if i == self.N_BINS - 1:
                mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
            if mask.sum() > 0:
                predicted = y_proba[mask].mean()
                actual = y_true[mask].mean()
                bin_size = mask.sum()
                reliability_data.append({
                    "bin": i,
                    "predicted_prob": float(predicted),
                    "actual_prob": float(actual),
                    "count": int(bin_size),
                })
                error = abs(predicted - actual)
                ece_sum += error * bin_size
                mce = max(mce, error)
        ece = ece_sum / len(y_true) if len(y_true) > 0 else 0.0
        return reliability_data, ece, mce

    def _global_recommendation(self, reliability_data: List[Dict], brier: float, ece: float) -> str:
        if brier < 0.10 and ece < 0.05:
            return "Well calibrated. No action needed."
        above_diagonal = sum(1 for b in reliability_data if b["predicted_prob"] > b["actual_prob"] + 0.05)
        below_diagonal = sum(1 for b in reliability_data if b["predicted_prob"] < b["actual_prob"] - 0.05)
        if above_diagonal > below_diagonal:
            return "Overconfident predictions. Apply Platt scaling."
        if below_diagonal > above_diagonal:
            return "Underconfident predictions. Consider isotonic regression."
        return "Apply CalibratedClassifierCV for general calibration improvement."
