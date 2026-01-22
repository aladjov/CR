from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series
from customer_retention.core.components.enums import Severity
from customer_retention.core.utils.statistics import compute_psi_from_series, compute_psi_categorical, compute_ks_statistic


class DriftType(Enum):
    FEATURE = "feature"
    TARGET = "target"
    CONCEPT = "concept"
    DATA_QUALITY = "data_quality"

@dataclass
class DriftConfig:
    ks_warning_threshold: float = 0.05
    ks_critical_threshold: float = 0.10
    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.20
    mean_shift_warning_threshold: float = 0.5
    mean_shift_critical_threshold: float = 1.0
    missing_rate_warning_threshold: float = 0.05
    missing_rate_critical_threshold: float = 0.10


@dataclass
class FeatureDriftResult:
    feature_name: str
    drift_type: DriftType
    metric_name: str
    metric_value: float
    drift_detected: bool
    severity: Severity
    recommendation: Optional[str] = None


@dataclass
class TargetDriftResult:
    drift_detected: bool
    reference_rate: float
    current_rate: float
    change_pct: float
    severity: Severity


@dataclass
class DriftResult:
    feature_results: List[FeatureDriftResult]
    overall_drift_detected: bool
    monitoring_timestamp: datetime = field(default_factory=datetime.now)
    drift_summary: Optional[Dict] = None

    def get_top_drifted_features(self, n: int = 5) -> List[FeatureDriftResult]:
        drifted = [r for r in self.feature_results if r.drift_detected]
        drifted.sort(key=lambda x: x.metric_value, reverse=True)
        return drifted[:n]


class DriftDetector:
    def __init__(self, reference_data: Optional[DataFrame] = None,
                 config: Optional[DriftConfig] = None,
                 reference_type: str = "training"):
        self.reference_data = reference_data
        self.config = config or DriftConfig()
        self.reference_type = reference_type

    def update_reference(self, new_reference: DataFrame):
        self.reference_data = new_reference.copy()

    def detect_drift(self, current_data: DataFrame, method: str = "psi",
                     features: Optional[List[str]] = None) -> DriftResult:
        if features is None:
            features = [col for col in current_data.columns
                       if col in self.reference_data.columns]
        results = []
        for feature in features:
            ref_col = self.reference_data[feature].dropna()
            curr_col = current_data[feature].dropna()
            if method == "ks":
                metric_value = self._compute_ks(ref_col, curr_col)
                metric_name = "ks_statistic"
            elif method == "psi":
                metric_value = self._compute_psi(ref_col, curr_col)
                metric_name = "psi"
            elif method == "mean_shift":
                metric_value = self._compute_mean_shift(ref_col, curr_col)
                metric_name = "mean_shift"
            else:
                raise ValueError(f"Unknown method: {method}")
            severity = self._assign_severity(metric_value, method)
            drift_detected = severity in [Severity.WARNING, Severity.CRITICAL]
            recommendation = self._get_recommendation(feature, severity, method) if drift_detected else None
            results.append(FeatureDriftResult(
                feature_name=feature,
                drift_type=DriftType.FEATURE,
                metric_name=metric_name,
                metric_value=metric_value,
                drift_detected=drift_detected,
                severity=severity,
                recommendation=recommendation
            ))
        overall_drift = any(r.drift_detected for r in results)
        return DriftResult(
            feature_results=results,
            overall_drift_detected=overall_drift
        )

    def detect_missing_rate_drift(self, current_data: DataFrame) -> DriftResult:
        results = []
        for col in current_data.columns:
            if col not in self.reference_data.columns:
                continue
            ref_missing = self.reference_data[col].isnull().mean()
            curr_missing = current_data[col].isnull().mean()
            change = abs(curr_missing - ref_missing)
            if change >= self.config.missing_rate_critical_threshold:
                severity = Severity.CRITICAL
                drift_detected = True
            elif change >= self.config.missing_rate_warning_threshold:
                severity = Severity.WARNING
                drift_detected = True
            else:
                severity = Severity.INFO
                drift_detected = False
            results.append(FeatureDriftResult(
                feature_name=col,
                drift_type=DriftType.DATA_QUALITY,
                metric_name="missing_rate_change",
                metric_value=change,
                drift_detected=drift_detected,
                severity=severity,
                recommendation=f"Investigate missing data increase in {col}" if drift_detected else None
            ))
        return DriftResult(
            feature_results=results,
            overall_drift_detected=any(r.drift_detected for r in results)
        )

    def detect_target_drift(self, reference_target: Series, current_target: Series,
                            threshold: float = 0.20) -> TargetDriftResult:
        ref_rate = reference_target.mean()
        curr_rate = current_target.mean()
        change_pct = abs(curr_rate - ref_rate) / ref_rate if ref_rate > 0 else 0
        drift_detected = change_pct >= threshold
        if change_pct >= threshold * 1.5:
            severity = Severity.CRITICAL
        elif drift_detected:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        return TargetDriftResult(
            drift_detected=drift_detected,
            reference_rate=ref_rate,
            current_rate=curr_rate,
            change_pct=change_pct,
            severity=severity
        )

    def _compute_ks(self, reference: Series, current: Series) -> float:
        statistic, _ = compute_ks_statistic(reference, current)
        return statistic

    def _compute_psi(self, reference: Series, current: Series, n_bins: int = 10) -> float:
        return compute_psi_from_series(reference, current, n_bins)

    def _compute_mean_shift(self, reference: Series, current: Series) -> float:
        ref_std = reference.std()
        if ref_std == 0:
            return 0
        return abs(current.mean() - reference.mean()) / ref_std

    def _assign_severity(self, metric_value: float, method: str) -> Severity:
        if method == "ks":
            if metric_value >= self.config.ks_critical_threshold:
                return Severity.CRITICAL
            elif metric_value >= self.config.ks_warning_threshold:
                return Severity.WARNING
        elif method == "psi":
            if metric_value >= self.config.psi_critical_threshold:
                return Severity.CRITICAL
            elif metric_value >= self.config.psi_warning_threshold:
                return Severity.WARNING
        elif method == "mean_shift":
            if metric_value >= self.config.mean_shift_critical_threshold:
                return Severity.CRITICAL
            elif metric_value >= self.config.mean_shift_warning_threshold:
                return Severity.WARNING
        return Severity.INFO

    def _get_recommendation(self, feature: str, severity: Severity, method: str) -> str:
        if severity == Severity.CRITICAL:
            return f"CRITICAL: Investigate {feature} immediately. Consider model retraining."
        elif severity == Severity.WARNING:
            return f"WARNING: Monitor {feature} closely. Prepare for potential retraining."
        return f"INFO: {feature} showing minor drift."
