"""Leakage detection probes for model validation."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from customer_retention.core.compat import DataFrame, Series, pd
from customer_retention.core.components.enums import Severity
from customer_retention.core.utils.leakage import TEMPORAL_METADATA_COLUMNS, calculate_class_overlap


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
    DOMAIN_TARGET_PATTERNS = re.compile(
        r"(churn|reten|cancel|unsubscribe|attrit|lapse|defect|convert|active|inactive|"
        r"leave|stay|renew|expir|terminat|close|deactivat)",
        re.IGNORECASE,
    )
    CROSS_ENTITY_PATTERNS = re.compile(
        r"(global|population|all_user|cross_entity|market_avg|cohort_avg|"
        r"overall_mean|overall_std|benchmark|percentile_rank)",
        re.IGNORECASE,
    )
    CORRELATION_CRITICAL, CORRELATION_HIGH, CORRELATION_MEDIUM = 0.90, 0.70, 0.50
    SEPARATION_CRITICAL, SEPARATION_HIGH, SEPARATION_MEDIUM = 0.0, 1.0, 5.0
    AUC_CRITICAL, AUC_HIGH = 0.90, 0.80
    CV_FOLDS = 5
    NUMERIC_DTYPES = (np.float64, np.int64, np.float32, np.int32)

    def __init__(self, feature_timestamp_column: str = "feature_timestamp", label_timestamp_column: str = "label_timestamp"):
        self.feature_timestamp_column = feature_timestamp_column
        self.label_timestamp_column = label_timestamp_column
        self._excluded_columns: Set[str] = set(TEMPORAL_METADATA_COLUMNS)

    def _get_analyzable_columns(self, X: DataFrame) -> List[str]:
        return [c for c in X.columns if c not in self._excluded_columns]

    def _get_numeric_columns(self, X: DataFrame) -> List[str]:
        return [c for c in self._get_analyzable_columns(X) if X[c].dtype in self.NUMERIC_DTYPES]

    def _safe_correlation(self, X: DataFrame, col: str, y: Series) -> float:
        corr = abs(X[col].corr(y))
        return 0.0 if np.isnan(corr) else corr

    def _build_result(self, checks: List[LeakageCheck]) -> LeakageResult:
        critical = [c for c in checks if c.severity == Severity.CRITICAL]
        return LeakageResult(passed=len(critical) == 0, checks=checks, critical_issues=critical)

    def check_correlations(self, X: DataFrame, y: Series) -> LeakageResult:
        checks = []
        for col in self._get_numeric_columns(X):
            corr = self._safe_correlation(X, col, y)
            severity, check_id = self._classify_correlation(corr)
            if severity != Severity.INFO:
                checks.append(LeakageCheck(
                    check_id=check_id, feature=col, severity=severity,
                    recommendation=self._correlation_recommendation(col, corr), correlation=corr,
                ))
        return self._build_result(checks)

    def _classify_correlation(self, corr: float) -> Tuple[Severity, str]:
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
        for col in self._get_numeric_columns(X):
            overlap_pct = calculate_class_overlap(X[col], y)
            severity, check_id = self._classify_separation(overlap_pct)
            checks.append(LeakageCheck(
                check_id=check_id, feature=col, severity=severity,
                recommendation=self._separation_recommendation(col, overlap_pct), overlap_pct=overlap_pct,
            ))
        return self._build_result(checks)

    def _classify_separation(self, overlap_pct: float) -> Tuple[Severity, str]:
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
        for col in self._get_numeric_columns(X):
            if not self.TEMPORAL_PATTERNS.search(col):
                continue
            corr = self._safe_correlation(X, col, y)
            if corr > self.CORRELATION_HIGH:
                checks.append(LeakageCheck(
                    check_id="LD022", feature=col, severity=Severity.HIGH,
                    recommendation=f"REVIEW temporal feature {col}: high correlation ({corr:.2f}) may indicate future data",
                    correlation=corr,
                ))
            elif corr > self.CORRELATION_MEDIUM:
                checks.append(LeakageCheck(
                    check_id="LD022", feature=col, severity=Severity.MEDIUM,
                    recommendation=f"CHECK temporal feature {col}: verify reference date logic", correlation=corr,
                ))
        return self._build_result(checks)

    def check_single_feature_auc(self, X: DataFrame, y: Series) -> LeakageResult:
        checks = []
        for col in self._get_numeric_columns(X):
            auc = self._compute_single_feature_auc(X[col], y)
            is_temporal = bool(self.TEMPORAL_PATTERNS.search(col))
            severity, check_id = self._classify_auc(auc, is_temporal=is_temporal)
            if severity != Severity.INFO:
                checks.append(LeakageCheck(
                    check_id=check_id, feature=col, severity=severity,
                    recommendation=self._auc_recommendation(col, auc, is_temporal=is_temporal), auc=auc,
                ))
        return self._build_result(checks)

    def _compute_single_feature_auc(self, feature: Series, y: Series) -> float:
        try:
            X_single = feature.values.reshape(-1, 1)
            mask = ~np.isnan(X_single.flatten())
            X_clean, y_clean = X_single[mask], y.values[mask]
            if len(np.unique(y_clean)) < 2 or min(np.bincount(y_clean.astype(int))) < self.CV_FOLDS:
                return 0.5
            model = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42)
            cv = StratifiedKFold(n_splits=self.CV_FOLDS, shuffle=True, random_state=42)
            proba = cross_val_predict(model, X_clean, y_clean, cv=cv, method="predict_proba")
            return roc_auc_score(y_clean, proba[:, 1])
        except Exception:
            return 0.5

    def _classify_auc(self, auc: float, *, is_temporal: bool = False) -> Tuple[Severity, str]:
        if is_temporal:
            if auc > self.AUC_CRITICAL:
                return Severity.HIGH, "LD031"
            return Severity.INFO, "LD000"
        if auc > self.AUC_CRITICAL:
            return Severity.CRITICAL, "LD030"
        if auc > self.AUC_HIGH:
            return Severity.HIGH, "LD031"
        return Severity.INFO, "LD000"

    def _auc_recommendation(self, feature: str, auc: float, *, is_temporal: bool = False) -> str:
        if auc > self.AUC_CRITICAL:
            if is_temporal:
                return f"REVIEW {feature}: temporal feature AUC {auc:.2f} is high but expected for recency/tenure features"
            return f"REMOVE {feature}: single-feature AUC {auc:.2f} indicates leakage"
        if auc > self.AUC_HIGH:
            return f"INVESTIGATE {feature}: single-feature AUC {auc:.2f} is very high"
        return f"OK: {feature} has normal predictive power"

    def check_point_in_time(self, df: DataFrame) -> LeakageResult:
        checks = []
        feature_ts = self._parse_timestamp(df, self.feature_timestamp_column)
        if feature_ts is None:
            return self._build_result([])

        self._check_label_timestamp_violation(df, feature_ts, checks)
        self._check_datetime_column_violations(df, feature_ts, checks)
        return self._build_result(checks)

    def _parse_timestamp(self, df: DataFrame, col: str) -> Optional[Series]:
        if col not in df.columns:
            return None
        try:
            return pd.to_datetime(df[col], errors="coerce", format="mixed")
        except Exception:
            return None

    def _check_label_timestamp_violation(self, df: DataFrame, feature_ts: Series, checks: List[LeakageCheck]) -> None:
        label_ts = self._parse_timestamp(df, self.label_timestamp_column)
        if label_ts is None:
            return
        violations = (feature_ts > label_ts).sum()
        if violations > 0:
            checks.append(LeakageCheck(
                check_id="LD040", feature=self.feature_timestamp_column, severity=Severity.CRITICAL,
                recommendation=f"FIX: {violations} rows have feature_timestamp > label_timestamp",
            ))

    def _check_datetime_column_violations(self, df: DataFrame, feature_ts: Series, checks: List[LeakageCheck]) -> None:
        skip_cols = {self.feature_timestamp_column, self.label_timestamp_column}
        for col in df.select_dtypes(include=["datetime64"]).columns:
            if col in skip_cols:
                continue
            try:
                col_ts = pd.to_datetime(df[col], errors="coerce", format="mixed")
                violations = (col_ts > feature_ts).sum()
                if violations > 0:
                    pct = violations / len(df) * 100
                    checks.append(LeakageCheck(
                        check_id="LD041", feature=col,
                        severity=Severity.CRITICAL if pct > 5 else Severity.HIGH,
                        recommendation=f"INVESTIGATE {col}: {violations} rows ({pct:.1f}%) have values after feature_timestamp",
                    ))
            except Exception:
                continue

    def check_uniform_timestamps(self, df: DataFrame, timestamp_column: str = "event_timestamp") -> LeakageResult:
        checks = []
        if timestamp_column not in df.columns:
            return self._build_result([])

        try:
            timestamps = pd.to_datetime(df[timestamp_column], errors="coerce").dropna()
            if len(timestamps) < 2:
                return self._build_result([])

            if timestamps.nunique() == 1:
                checks.append(LeakageCheck(
                    check_id="LD050", feature=timestamp_column, severity=Severity.HIGH,
                    recommendation=(
                        f"INVESTIGATE {timestamp_column}: All {len(timestamps)} timestamps are identical. "
                        "This suggests datetime.now() was used instead of actual aggregation reference dates."
                    ),
                ))
            elif (timestamps.max() - timestamps.min()).total_seconds() < 60:
                time_span = (timestamps.max() - timestamps.min()).total_seconds()
                checks.append(LeakageCheck(
                    check_id="LD050", feature=timestamp_column, severity=Severity.MEDIUM,
                    recommendation=(
                        f"REVIEW {timestamp_column}: Timestamps span only {time_span:.1f} seconds across "
                        f"{len(timestamps)} records. Verify timestamps reflect actual observation dates."
                    ),
                ))
        except Exception:
            pass
        return self._build_result(checks)

    def check_target_in_features(self, X: DataFrame, y: Series, target_name: str = "target") -> LeakageResult:
        checks = []
        self._check_target_column_direct(X, target_name, checks)
        self._check_target_derived_names(X, target_name, checks)
        self._check_perfect_correlation(X, y, target_name, checks)
        return self._build_result(checks)

    def _check_target_column_direct(self, X: DataFrame, target_name: str, checks: List[LeakageCheck]) -> None:
        if target_name in X.columns:
            checks.append(LeakageCheck(
                check_id="LD052", feature=target_name, severity=Severity.CRITICAL,
                recommendation=f"REMOVE {target_name}: Target column found in feature matrix. Direct data leakage.",
                correlation=1.0,
            ))

    def _check_target_derived_names(self, X: DataFrame, target_name: str, checks: List[LeakageCheck]) -> None:
        patterns = [f"{target_name}_", f"_{target_name}"]
        for col in X.columns:
            col_lower = col.lower()
            if any(p.lower() in col_lower for p in patterns):
                checks.append(LeakageCheck(
                    check_id="LD052", feature=col, severity=Severity.CRITICAL,
                    recommendation=f"REMOVE {col}: Column name suggests derivation from target '{target_name}'.",
                ))

    def _check_perfect_correlation(self, X: DataFrame, y: Series, target_name: str, checks: List[LeakageCheck]) -> None:
        already_flagged = {target_name} | {c.feature for c in checks}
        for col in self._get_numeric_columns(X):
            if col in already_flagged:
                continue
            try:
                corr = abs(X[col].corr(y))
                if not np.isnan(corr) and corr > 0.99:
                    checks.append(LeakageCheck(
                        check_id="LD052", feature=col, severity=Severity.CRITICAL,
                        recommendation=f"REMOVE {col}: Perfect correlation ({corr:.4f}) indicates leakage.",
                        correlation=corr,
                    ))
            except Exception:
                pass

    def check_cross_entity_leakage(self, X: DataFrame, y: Series, entity_column: str, timestamp_column: str) -> LeakageResult:
        checks = []
        for col in self._get_numeric_columns(X):
            if not self.CROSS_ENTITY_PATTERNS.search(col):
                continue
            corr = self._safe_correlation(X, col, y)
            severity = Severity.HIGH if corr > self.CORRELATION_MEDIUM else Severity.MEDIUM
            checks.append(LeakageCheck(
                check_id="LD060", feature=col, severity=severity,
                recommendation=(
                    f"REVIEW {col}: Cross-entity aggregation pattern detected. Correlation: {corr:.2f}. "
                    "Verify this feature doesn't use future data from other entities."
                ),
                correlation=corr,
            ))
        return self._build_result(checks)

    def check_temporal_split(self, train_timestamps: Series, test_timestamps: Series, timestamp_column: str = "timestamp") -> LeakageResult:
        checks = []
        try:
            train_ts = pd.to_datetime(train_timestamps, errors="coerce").dropna()
            test_ts = pd.to_datetime(test_timestamps, errors="coerce").dropna()
            if len(train_ts) == 0 or len(test_ts) == 0:
                return self._build_result([])

            train_max, test_min = train_ts.max(), test_ts.min()
            if train_max >= test_min:
                overlap_count = (train_ts >= test_min).sum()
                overlap_pct = overlap_count / len(train_ts) * 100
                checks.append(LeakageCheck(
                    check_id="LD061", feature=timestamp_column, severity=Severity.CRITICAL,
                    recommendation=(
                        f"FIX temporal split: Train max ({train_max}) >= Test min ({test_min}). "
                        f"{overlap_count} train rows ({overlap_pct:.1f}%) overlap with test period."
                    ),
                ))
        except Exception:
            pass
        return self._build_result(checks)

    def check_domain_target_patterns(self, X: DataFrame, y: Series) -> LeakageResult:
        checks = []
        for col in self._get_numeric_columns(X):
            if not self.DOMAIN_TARGET_PATTERNS.search(col):
                continue
            corr = self._safe_correlation(X, col, y)
            severity, recommendation = self._classify_domain_pattern(col, corr)
            checks.append(LeakageCheck(
                check_id="LD053", feature=col, severity=severity,
                recommendation=recommendation, correlation=corr,
            ))
        return self._build_result(checks)

    def _classify_domain_pattern(self, col: str, corr: float) -> Tuple[Severity, str]:
        if corr > self.CORRELATION_HIGH:
            return Severity.CRITICAL, f"REMOVE {col}: Domain pattern with high correlation ({corr:.2f}) confirms likely leakage."
        if corr > self.CORRELATION_MEDIUM:
            return Severity.HIGH, f"INVESTIGATE {col}: Domain pattern with correlation ({corr:.2f}) warrants review."
        return Severity.MEDIUM, f"REVIEW {col}: Contains churn/retention terminology. Low correlation ({corr:.2f}) suggests safe."

    def run_all_checks(self, X: DataFrame, y: Series, include_pit: bool = True) -> LeakageResult:
        all_checks = (
            self.check_correlations(X, y).checks
            + self.check_separation(X, y).checks
            + self.check_temporal_logic(X, y).checks
            + self.check_single_feature_auc(X, y).checks
        )

        if include_pit:
            df_with_y = X.copy()
            df_with_y["_target"] = y
            all_checks.extend(self.check_point_in_time(df_with_y).checks)
            all_checks.extend(self.check_uniform_timestamps(df_with_y, timestamp_column=self.feature_timestamp_column).checks)

        all_checks.extend(self.check_target_in_features(X, y).checks)
        all_checks.extend(self.check_domain_target_patterns(X, y).checks)

        critical = [c for c in all_checks if c.severity == Severity.CRITICAL]
        recommendations = list({c.recommendation for c in all_checks if c.severity in [Severity.CRITICAL, Severity.HIGH]})
        return LeakageResult(passed=len(critical) == 0, checks=all_checks, critical_issues=critical, recommendations=recommendations)
