"""Leakage detection probes for model validation."""

from dataclasses import dataclass, field
from typing import List, Optional
import re

import numpy as np

from customer_retention.core.compat import pd, DataFrame, Series
from customer_retention.core.components.enums import Severity
from customer_retention.core.utils.leakage import calculate_class_overlap, LeakageThresholds, DEFAULT_THRESHOLDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class LeakageCheck:
    check_id: str
    feature: str
    severity: Severity
    recommendation: str
    correlation: float = 0.0
    overlap_pct: float = 100.0
    auc: float = 0.5


@dataclass
class LeakageResult:
    passed: bool
    checks: List[LeakageCheck] = field(default_factory=list)
    critical_issues: List[LeakageCheck] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class LeakageDetector:
    TEMPORAL_PATTERNS = re.compile(r"(days|since|tenure|recency|last|ago|date|time)", re.IGNORECASE)
    CORRELATION_CRITICAL = 0.90
    CORRELATION_HIGH = 0.70
    CORRELATION_MEDIUM = 0.50
    SEPARATION_CRITICAL = 0.0
    SEPARATION_HIGH = 1.0
    SEPARATION_MEDIUM = 5.0
    AUC_CRITICAL = 0.90
    AUC_HIGH = 0.80

    def __init__(
        self,
        feature_timestamp_column: str = "feature_timestamp",
        label_timestamp_column: str = "label_timestamp",
    ):
        self.feature_timestamp_column = feature_timestamp_column
        self.label_timestamp_column = label_timestamp_column

    def check_correlations(self, X: DataFrame, y: Series) -> LeakageResult:
        checks = []
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                corr = abs(X[col].corr(y))
                if np.isnan(corr):
                    corr = 0.0
                severity, check_id = self._classify_correlation(corr)
                if severity != Severity.INFO:
                    checks.append(LeakageCheck(
                        check_id=check_id,
                        feature=col,
                        severity=severity,
                        recommendation=self._correlation_recommendation(col, corr),
                        correlation=corr,
                    ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return LeakageResult(passed=len(critical) == 0, checks=checks, critical_issues=critical)

    def _classify_correlation(self, corr: float) -> tuple:
        if corr > self.CORRELATION_CRITICAL:
            return Severity.CRITICAL, "LD001"
        if corr > self.CORRELATION_HIGH:
            return Severity.HIGH, "LD002"
        if corr > self.CORRELATION_MEDIUM:
            return Severity.MEDIUM, "LD003"
        return Severity.INFO, "LD000"

    def _correlation_recommendation(self, feature: str, corr: float) -> str:
        if corr > self.CORRELATION_CRITICAL:
            return f"REMOVE {feature}: correlation {corr:.2f} indicates likely data leakage"
        if corr > self.CORRELATION_HIGH:
            return f"INVESTIGATE {feature}: correlation {corr:.2f} is suspiciously high"
        return f"MONITOR {feature}: elevated correlation {corr:.2f}"

    def check_separation(self, X: DataFrame, y: Series) -> LeakageResult:
        checks = []
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                overlap_pct = self._calculate_overlap(X[col], y)
                severity, check_id = self._classify_separation(overlap_pct)
                checks.append(LeakageCheck(
                    check_id=check_id,
                    feature=col,
                    severity=severity,
                    recommendation=self._separation_recommendation(col, overlap_pct),
                    overlap_pct=overlap_pct,
                ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return LeakageResult(passed=len(critical) == 0, checks=checks, critical_issues=critical)

    def _calculate_overlap(self, feature: Series, y: Series) -> float:
        return calculate_class_overlap(feature, y)

    def _classify_separation(self, overlap_pct: float) -> tuple:
        if overlap_pct <= self.SEPARATION_CRITICAL:
            return Severity.CRITICAL, "LD010"
        if overlap_pct < self.SEPARATION_HIGH:
            return Severity.HIGH, "LD011"
        if overlap_pct < self.SEPARATION_MEDIUM:
            return Severity.MEDIUM, "LD012"
        return Severity.INFO, "LD000"

    def _separation_recommendation(self, feature: str, overlap_pct: float) -> str:
        if overlap_pct <= self.SEPARATION_CRITICAL:
            return f"REMOVE {feature}: perfect class separation indicates leakage"
        if overlap_pct < self.SEPARATION_HIGH:
            return f"REMOVE {feature}: near-perfect separation ({overlap_pct:.1f}% overlap)"
        if overlap_pct < self.SEPARATION_MEDIUM:
            return f"INVESTIGATE {feature}: high separation ({overlap_pct:.1f}% overlap)"
        return f"OK: {feature} has normal class overlap"

    def check_temporal_logic(self, X: DataFrame, y: Series) -> LeakageResult:
        checks = []
        for col in X.columns:
            if self.TEMPORAL_PATTERNS.search(col):
                if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    corr = abs(X[col].corr(y))
                    if np.isnan(corr):
                        corr = 0.0
                    if corr > self.CORRELATION_HIGH:
                        checks.append(LeakageCheck(
                            check_id="LD022",
                            feature=col,
                            severity=Severity.HIGH,
                            recommendation=f"REVIEW temporal feature {col}: high correlation ({corr:.2f}) may indicate future data",
                            correlation=corr,
                        ))
                    elif corr > self.CORRELATION_MEDIUM:
                        checks.append(LeakageCheck(
                            check_id="LD022",
                            feature=col,
                            severity=Severity.MEDIUM,
                            recommendation=f"CHECK temporal feature {col}: verify reference date logic",
                            correlation=corr,
                        ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return LeakageResult(passed=len(critical) == 0, checks=checks, critical_issues=critical)

    def check_single_feature_auc(self, X: DataFrame, y: Series) -> LeakageResult:
        checks = []
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                auc = self._compute_single_feature_auc(X[col], y)
                severity, check_id = self._classify_auc(auc)
                if severity != Severity.INFO:
                    checks.append(LeakageCheck(
                        check_id=check_id,
                        feature=col,
                        severity=severity,
                        recommendation=self._auc_recommendation(col, auc),
                        auc=auc,
                    ))
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return LeakageResult(passed=len(critical) == 0, checks=checks, critical_issues=critical)

    def _compute_single_feature_auc(self, feature: Series, y: Series) -> float:
        try:
            X_single = feature.values.reshape(-1, 1)
            mask = ~np.isnan(X_single.flatten())
            X_clean = X_single[mask]
            y_clean = y.values[mask]
            if len(np.unique(y_clean)) < 2:
                return 0.5
            model = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42)
            model.fit(X_clean, y_clean)
            proba = model.predict_proba(X_clean)[:, 1]
            return roc_auc_score(y_clean, proba)
        except Exception:
            return 0.5

    def _classify_auc(self, auc: float) -> tuple:
        if auc > self.AUC_CRITICAL:
            return Severity.CRITICAL, "LD030"
        if auc > self.AUC_HIGH:
            return Severity.HIGH, "LD031"
        return Severity.INFO, "LD000"

    def _auc_recommendation(self, feature: str, auc: float) -> str:
        if auc > self.AUC_CRITICAL:
            return f"REMOVE {feature}: single-feature AUC {auc:.2f} indicates leakage"
        if auc > self.AUC_HIGH:
            return f"INVESTIGATE {feature}: single-feature AUC {auc:.2f} is very high"
        return f"OK: {feature} has normal predictive power"

    def check_point_in_time(self, df: DataFrame) -> LeakageResult:
        """Check for point-in-time constraint violations (LD040, LD041)."""
        checks = []

        if self.feature_timestamp_column not in df.columns:
            return LeakageResult(passed=True, checks=[], critical_issues=[])

        try:
            feature_ts = pd.to_datetime(df[self.feature_timestamp_column], errors='coerce', format='mixed')
        except Exception:
            return LeakageResult(passed=True, checks=[], critical_issues=[])

        if self.label_timestamp_column in df.columns:
            try:
                label_ts = pd.to_datetime(df[self.label_timestamp_column], errors='coerce', format='mixed')
                violations = df[feature_ts > label_ts]
                if len(violations) > 0:
                    checks.append(LeakageCheck(
                        check_id="LD040",
                        feature=self.feature_timestamp_column,
                        severity=Severity.CRITICAL,
                        recommendation=f"FIX: {len(violations)} rows have feature_timestamp > label_timestamp",
                        correlation=0.0,
                    ))
            except Exception:
                pass

        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        for col in datetime_cols:
            if col in [self.feature_timestamp_column, self.label_timestamp_column]:
                continue
            try:
                col_ts = pd.to_datetime(df[col], errors='coerce', format='mixed')
                violations = df[col_ts > feature_ts]
                if len(violations) > 0:
                    pct = len(violations) / len(df) * 100
                    severity = Severity.CRITICAL if pct > 5 else Severity.HIGH
                    checks.append(LeakageCheck(
                        check_id="LD041",
                        feature=col,
                        severity=severity,
                        recommendation=f"INVESTIGATE {col}: {len(violations)} rows ({pct:.1f}%) have values after feature_timestamp",
                        correlation=0.0,
                    ))
            except Exception:
                continue

        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return LeakageResult(passed=len(critical) == 0, checks=checks, critical_issues=critical)

    def run_all_checks(self, X: DataFrame, y: Series, include_pit: bool = True) -> LeakageResult:
        correlation_result = self.check_correlations(X, y)
        separation_result = self.check_separation(X, y)
        temporal_result = self.check_temporal_logic(X, y)
        auc_result = self.check_single_feature_auc(X, y)

        all_checks = (correlation_result.checks + separation_result.checks +
                      temporal_result.checks + auc_result.checks)

        if include_pit:
            df_with_y = X.copy()
            df_with_y["_target"] = y
            pit_result = self.check_point_in_time(df_with_y)
            all_checks.extend(pit_result.checks)

        critical = [c for c in all_checks if c.severity == Severity.CRITICAL]
        recommendations = list(set(c.recommendation for c in all_checks if c.severity in [Severity.CRITICAL, Severity.HIGH]))

        return LeakageResult(
            passed=len(critical) == 0,
            checks=all_checks,
            critical_issues=critical,
            recommendations=recommendations,
        )
