from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from customer_retention.core.compat import DataFrame, Timestamp, is_numeric_dtype, pd
from customer_retention.core.components.enums import Severity

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityMetadata


@dataclass
class LeakageIssue:
    check_id: str
    severity: Severity
    feature: str
    description: str
    value: Optional[float] = None


@dataclass
class LeakageCheckResult:
    passed: bool
    critical_issues: List[LeakageIssue]
    high_issues: List[LeakageIssue]
    suspicious_features: List[str]
    recommended_drops: List[str]
    leakage_report: Dict[str, Any]


class LeakageGate:
    _LK011_DESCRIPTIONS = {
        "new_tracking": "Feature tracking started late ({first_date}). Training data before this date will have missing values.",
        "retired": "Feature tracking retired ({last_date}). Test/scoring data after this date will have missing values.",
        "partial_window": "Feature only available {first_date} to {last_date}. Both train and test may have gaps.",
    }

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
        availability_coverage_threshold: float = 50.0,
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
        self.availability_coverage_threshold = availability_coverage_threshold

    def run(
        self, df: DataFrame, feature_availability: Optional["FeatureAvailabilityMetadata"] = None
    ) -> LeakageCheckResult:
        critical_issues: List[LeakageIssue] = []
        high_issues: List[LeakageIssue] = []
        suspicious_features: List[str] = []
        recommended_drops: List[str] = []
        leakage_report: Dict[str, Any] = {}

        feature_cols = [
            c for c in df.columns
            if c != self.target_column and c not in self.exclude_features
        ]

        corr_issues, corr_report = self._check_correlations(df, feature_cols)
        critical_issues.extend([i for i in corr_issues if i.severity == Severity.CRITICAL])
        high_issues.extend([i for i in corr_issues if i.severity == Severity.HIGH])
        leakage_report["correlations"] = corr_report

        sep_issues = self._check_perfect_separation(df, feature_cols)
        critical_issues.extend(sep_issues)

        if self.reference_date and self.date_columns:
            temp_issues = self._check_temporal_leakage(df)
            critical_issues.extend(temp_issues)

        const_issues = self._check_near_constant_by_class(df, feature_cols)
        high_issues.extend(const_issues)

        if self.enforce_point_in_time:
            pit_issues = self._check_point_in_time_violations(df)
            critical_issues.extend([i for i in pit_issues if i.severity == Severity.CRITICAL])
            high_issues.extend([i for i in pit_issues if i.severity == Severity.HIGH])

            future_issues = self._check_future_dates_in_features(df, feature_cols)
            critical_issues.extend(future_issues)

        if feature_availability is not None:
            avail_issues = self._check_feature_availability(df, feature_cols, feature_availability)
            high_issues.extend(avail_issues)

        for issue in critical_issues + high_issues:
            if issue.feature not in suspicious_features:
                suspicious_features.append(issue.feature)
            if issue.severity == Severity.CRITICAL and issue.feature not in recommended_drops:
                recommended_drops.append(issue.feature)

        passed = len(critical_issues) == 0

        return LeakageCheckResult(
            passed=passed,
            critical_issues=critical_issues,
            high_issues=high_issues,
            suspicious_features=suspicious_features,
            recommended_drops=recommended_drops,
            leakage_report=leakage_report,
        )

    def _compute_target_correlations(self, df: DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        numeric_features = [c for c in feature_cols if is_numeric_dtype(df[c])]
        if not numeric_features or self.target_column not in df.columns:
            return {}
        correlations = {}
        for feature in numeric_features:
            try:
                corr = df[feature].corr(df[self.target_column])
                if pd.notna(corr):
                    correlations[feature] = corr
            except Exception:
                continue
        return correlations

    def _create_correlation_issues(self, correlations: Dict[str, float]) -> List[LeakageIssue]:
        issues = []
        for feature, corr in correlations.items():
            abs_corr = abs(corr)
            if abs_corr >= self.correlation_threshold_critical:
                issues.append(LeakageIssue(
                    check_id="LK001", severity=Severity.CRITICAL, feature=feature,
                    description=f"High target correlation: {corr:.4f}", value=corr,
                ))
            elif abs_corr >= self.correlation_threshold_high:
                issues.append(LeakageIssue(
                    check_id="LK002", severity=Severity.HIGH, feature=feature,
                    description=f"Suspicious target correlation: {corr:.4f}", value=corr,
                ))
        return issues

    def _check_correlations(self, df: DataFrame, feature_cols: List[str]) -> tuple:
        correlations = self._compute_target_correlations(df, feature_cols)
        issues = self._create_correlation_issues(correlations)
        return issues, {"feature_correlations": correlations}

    @staticmethod
    def _parse_datetime(series, errors="coerce"):
        return pd.to_datetime(series, errors=errors, format='mixed')

    def _check_perfect_separation(
        self,
        df: DataFrame,
        feature_cols: List[str]
    ) -> List[LeakageIssue]:
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
                class_0 = df[df[self.target_column] == target_values[0]][feature].dropna()
                class_1 = df[df[self.target_column] == target_values[1]][feature].dropna()

                if len(class_0) == 0 or len(class_1) == 0:
                    continue

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
        issues = []

        for date_col in self.date_columns:
            if date_col not in df.columns:
                continue
            try:
                dates = self._parse_datetime(df[date_col])
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
                var_0 = df[df[self.target_column] == target_values[0]][feature].var()
                var_1 = df[df[self.target_column] == target_values[1]][feature].var()
                mean_0 = df[df[self.target_column] == target_values[0]][feature].mean()
                mean_1 = df[df[self.target_column] == target_values[1]][feature].mean()

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
        issues = []

        if self.feature_timestamp_column not in df.columns:
            return issues
        if self.label_timestamp_column not in df.columns:
            return issues

        try:
            feature_ts = self._parse_datetime(df[self.feature_timestamp_column])
            label_ts = self._parse_datetime(df[self.label_timestamp_column])
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
        issues = []

        if self.feature_timestamp_column not in df.columns:
            return issues

        try:
            feature_ts = self._parse_datetime(df[self.feature_timestamp_column])
        except Exception:
            return issues

        datetime_feature_cols = [
            c for c in feature_cols
            if c != self.feature_timestamp_column and c != self.label_timestamp_column
        ]

        for col in datetime_feature_cols:
            try:
                col_dates = self._parse_datetime(df[col])
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

    def _check_feature_availability(self, df: DataFrame, feature_cols: List[str], availability: "FeatureAvailabilityMetadata") -> List[LeakageIssue]:
        issues: List[LeakageIssue] = []

        for col in feature_cols:
            if col not in df.columns:
                continue
            feat_info = availability.features.get(col)
            if feat_info is None:
                continue

            first_date = feat_info.first_valid_date or "unknown"
            last_date = feat_info.last_valid_date or "unknown"
            availability_lists = [
                (availability.new_tracking, "new_tracking"),
                (availability.retired_tracking, "retired"),
                (availability.partial_window, "partial_window"),
            ]
            for tracking_list, tracking_type in availability_lists:
                if col in tracking_list:
                    description = self._LK011_DESCRIPTIONS[tracking_type].format(
                        first_date=first_date, last_date=last_date
                    )
                    issues.append(LeakageIssue(
                        check_id="LK011", severity=Severity.HIGH, feature=col,
                        description=description, value=feat_info.coverage_pct,
                    ))
                    break

            has_availability_issue = any(col in lst for lst, _ in availability_lists)
            if feat_info.coverage_pct < self.availability_coverage_threshold and not has_availability_issue:
                issues.append(LeakageIssue(
                    check_id="LK012", severity=Severity.HIGH, feature=col,
                    description=f"Low feature coverage ({feat_info.coverage_pct:.1f}% < {self.availability_coverage_threshold}% threshold)",
                    value=feat_info.coverage_pct,
                ))

        return issues
