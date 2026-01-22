"""
Leakage Detection Gate (Checkpoint 3) for customer retention analysis.

This module provides leakage detection to prevent data leakage
in features before model training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
from customer_retention.core.compat import pd, DataFrame, Timestamp, is_numeric_dtype
from customer_retention.core.components.enums import Severity


@dataclass
class LeakageIssue:
    check_id: str
    severity: Severity
    feature: str
    description: str
    value: Optional[float] = None


@dataclass
class LeakageCheckResult:
    """Result of leakage detection gate."""
    passed: bool
    critical_issues: List[LeakageIssue]
    high_issues: List[LeakageIssue]
    suspicious_features: List[str]
    recommended_drops: List[str]
    leakage_report: Dict[str, Any]


class LeakageGate:
    """
    Leakage Detection Gate (Checkpoint 3).

    Detects and prevents data leakage in features before model training.

    Parameters
    ----------
    target_column : str
        Name of the target column.
    correlation_threshold_critical : float, default 0.90
        Correlation threshold for critical leakage (LK001).
    correlation_threshold_high : float, default 0.70
        Correlation threshold for high-risk leakage (LK002).
    reference_date : Timestamp, optional
        Reference date for temporal checks.
    date_columns : List[str], optional
        Columns containing dates for temporal validation.
    exclude_features : List[str], optional
        Features to exclude from leakage checks.

    Attributes
    ----------
    correlation_threshold_critical : float
        Critical correlation threshold.
    correlation_threshold_high : float
        High-risk correlation threshold.
    """

    def __init__(
        self,
        target_column: str,
        correlation_threshold_critical: float = 0.90,
        correlation_threshold_high: float = 0.70,
        reference_date: Optional[Timestamp] = None,
        date_columns: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        feature_timestamp_column: Optional[str] = None,
        label_timestamp_column: Optional[str] = None,
        enforce_point_in_time: bool = True,
    ):
        self.target_column = target_column
        self.correlation_threshold_critical = correlation_threshold_critical
        self.correlation_threshold_high = correlation_threshold_high
        self.reference_date = reference_date
        self.date_columns = date_columns or []
        self.exclude_features = exclude_features or []
        self.feature_timestamp_column = feature_timestamp_column or "feature_timestamp"
        self.label_timestamp_column = label_timestamp_column or "label_timestamp"
        self.enforce_point_in_time = enforce_point_in_time

    def run(self, df: DataFrame) -> LeakageCheckResult:
        """
        Run leakage detection on the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features and target.

        Returns
        -------
        LeakageCheckResult
            Result of leakage detection.
        """
        critical_issues: List[LeakageIssue] = []
        high_issues: List[LeakageIssue] = []
        suspicious_features: List[str] = []
        recommended_drops: List[str] = []
        leakage_report: Dict[str, Any] = {}

        # Get feature columns (exclude target and excluded features)
        feature_cols = [
            c for c in df.columns
            if c != self.target_column and c not in self.exclude_features
        ]

        # Run correlation checks (LK001, LK002)
        corr_issues, corr_report = self._check_correlations(df, feature_cols)
        critical_issues.extend([i for i in corr_issues if i.severity == Severity.CRITICAL])
        high_issues.extend([i for i in corr_issues if i.severity == Severity.HIGH])
        leakage_report["correlations"] = corr_report

        # Run perfect separation check (LK003)
        sep_issues = self._check_perfect_separation(df, feature_cols)
        critical_issues.extend(sep_issues)

        # Run temporal checks (LK004)
        if self.reference_date and self.date_columns:
            temp_issues = self._check_temporal_leakage(df)
            critical_issues.extend(temp_issues)

        # Run near-constant by class check (LK008)
        const_issues = self._check_near_constant_by_class(df, feature_cols)
        high_issues.extend(const_issues)

        # Run point-in-time checks (LK009, LK010)
        if self.enforce_point_in_time:
            pit_issues = self._check_point_in_time_violations(df)
            critical_issues.extend([i for i in pit_issues if i.severity == Severity.CRITICAL])
            high_issues.extend([i for i in pit_issues if i.severity == Severity.HIGH])

            future_issues = self._check_future_dates_in_features(df, feature_cols)
            critical_issues.extend(future_issues)

        # Compile suspicious features and recommendations
        for issue in critical_issues + high_issues:
            if issue.feature not in suspicious_features:
                suspicious_features.append(issue.feature)
            if issue.severity == Severity.CRITICAL and issue.feature not in recommended_drops:
                recommended_drops.append(issue.feature)

        # Determine if gate passes
        passed = len(critical_issues) == 0

        return LeakageCheckResult(
            passed=passed,
            critical_issues=critical_issues,
            high_issues=high_issues,
            suspicious_features=suspicious_features,
            recommended_drops=recommended_drops,
            leakage_report=leakage_report,
        )

    def _check_correlations(
        self,
        df: DataFrame,
        feature_cols: List[str]
    ) -> tuple:
        """Check for high correlations with target (LK001, LK002)."""
        issues = []
        report = {}

        # Get numeric features only
        numeric_features = [
            c for c in feature_cols
            if is_numeric_dtype(df[c])
        ]

        if not numeric_features or self.target_column not in df.columns:
            return issues, report

        # Compute correlations with target
        correlations = {}
        for feature in numeric_features:
            try:
                corr = df[feature].corr(df[self.target_column])
                if pd.notna(corr):
                    correlations[feature] = corr
            except Exception:
                continue

        report["feature_correlations"] = correlations

        # Check against thresholds
        for feature, corr in correlations.items():
            abs_corr = abs(corr)
            if abs_corr >= self.correlation_threshold_critical:
                issues.append(LeakageIssue(
                    check_id="LK001",
                    severity=Severity.CRITICAL,
                    feature=feature,
                    description=f"High target correlation: {corr:.4f}",
                    value=corr,
                ))
            elif abs_corr >= self.correlation_threshold_high:
                issues.append(LeakageIssue(
                    check_id="LK002",
                    severity=Severity.HIGH,
                    feature=feature,
                    description=f"Suspicious target correlation: {corr:.4f}",
                    value=corr,
                ))

        return issues, report

    def _check_perfect_separation(
        self,
        df: DataFrame,
        feature_cols: List[str]
    ) -> List[LeakageIssue]:
        """Check for perfect class separation (LK003)."""
        issues = []

        if self.target_column not in df.columns:
            return issues

        target_values = df[self.target_column].unique()
        if len(target_values) != 2:
            return issues  # Only check for binary classification

        for feature in feature_cols:
            if not is_numeric_dtype(df[feature]):
                continue

            try:
                # Get feature values for each class
                class_0 = df[df[self.target_column] == target_values[0]][feature].dropna()
                class_1 = df[df[self.target_column] == target_values[1]][feature].dropna()

                if len(class_0) == 0 or len(class_1) == 0:
                    continue

                # Check for no overlap
                if class_0.max() < class_1.min() or class_1.max() < class_0.min():
                    issues.append(LeakageIssue(
                        check_id="LK003",
                        severity=Severity.CRITICAL,
                        feature=feature,
                        description="Perfect class separation detected",
                    ))
            except Exception:
                continue

        return issues

    def _check_temporal_leakage(self, df: DataFrame) -> List[LeakageIssue]:
        """Check for temporal leakage (LK004)."""
        issues = []

        for date_col in self.date_columns:
            if date_col not in df.columns:
                continue

            try:
                dates = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
                # Check if any dates are after reference date
                if (dates > self.reference_date).any():
                    issues.append(LeakageIssue(
                        check_id="LK004",
                        severity=Severity.CRITICAL,
                        feature=date_col,
                        description=f"Temporal violation: dates after reference {self.reference_date}",
                    ))
            except Exception:
                continue

        return issues

    def _check_near_constant_by_class(
        self,
        df: DataFrame,
        feature_cols: List[str]
    ) -> List[LeakageIssue]:
        """Check for features nearly constant within each class (LK008)."""
        issues = []

        if self.target_column not in df.columns:
            return issues

        target_values = df[self.target_column].unique()
        if len(target_values) != 2:
            return issues

        for feature in feature_cols:
            if not is_numeric_dtype(df[feature]):
                continue

            try:
                # Check variance within each class
                var_0 = df[df[self.target_column] == target_values[0]][feature].var()
                var_1 = df[df[self.target_column] == target_values[1]][feature].var()

                # Check mean difference
                mean_0 = df[df[self.target_column] == target_values[0]][feature].mean()
                mean_1 = df[df[self.target_column] == target_values[1]][feature].mean()

                # If both variances are very low but means are different
                if (pd.notna(var_0) and pd.notna(var_1) and
                    var_0 < 0.01 and var_1 < 0.01 and
                    abs(mean_0 - mean_1) > 0.1):
                    issues.append(LeakageIssue(
                        check_id="LK008",
                        severity=Severity.HIGH,
                        feature=feature,
                        description="Near-constant within each class",
                    ))
            except Exception:
                continue

        return issues

    def _check_point_in_time_violations(self, df: DataFrame) -> List[LeakageIssue]:
        """Check for point-in-time violations (LK009)."""
        issues = []

        if self.feature_timestamp_column not in df.columns:
            return issues
        if self.label_timestamp_column not in df.columns:
            return issues

        try:
            feature_ts = pd.to_datetime(df[self.feature_timestamp_column], errors='coerce', format='mixed')
            label_ts = pd.to_datetime(df[self.label_timestamp_column], errors='coerce', format='mixed')

            violations = df[feature_ts > label_ts]
            if len(violations) > 0:
                issues.append(LeakageIssue(
                    check_id="LK009",
                    severity=Severity.CRITICAL,
                    feature=self.feature_timestamp_column,
                    description=f"Point-in-time violation: {len(violations)} rows have feature_timestamp > label_timestamp",
                    value=float(len(violations)),
                ))
        except Exception:
            pass

        return issues

    def _check_future_dates_in_features(
        self, df: DataFrame, feature_cols: List[str]
    ) -> List[LeakageIssue]:
        """Check for future dates in feature columns (LK010)."""
        issues = []

        if self.feature_timestamp_column not in df.columns:
            return issues

        try:
            feature_ts = pd.to_datetime(df[self.feature_timestamp_column], errors='coerce', format='mixed')
        except Exception:
            return issues

        datetime_feature_cols = [
            c for c in feature_cols
            if c != self.feature_timestamp_column and c != self.label_timestamp_column
        ]

        for col in datetime_feature_cols:
            try:
                col_dates = pd.to_datetime(df[col], errors='coerce', format='mixed')
                if col_dates.isna().all():
                    continue

                violations = df[col_dates > feature_ts]
                if len(violations) > 0:
                    issues.append(LeakageIssue(
                        check_id="LK010",
                        severity=Severity.CRITICAL,
                        feature=col,
                        description=f"Future data leakage: {len(violations)} rows have {col} > feature_timestamp",
                        value=float(len(violations)),
                    ))
            except Exception:
                continue

        return issues
