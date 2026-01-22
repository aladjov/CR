from typing import Optional, Any
from pydantic import BaseModel

from customer_retention.core.compat import pd, Series, is_datetime64_any_dtype
from customer_retention.core.config import ColumnType
from customer_retention.core.components.enums import Severity

class QualityCheckResult(BaseModel):
    check_id: str
    check_name: str
    column_name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = {}
    recommendation: Optional[str] = None


class QualityCheck:
    def __init__(self, check_id: str, check_name: str, severity: Severity):
        self.check_id = check_id
        self.check_name = check_name
        self.severity = severity

    def create_result(self, column_name: str, passed: bool, message: str,
                      details: dict = None, recommendation: str = None,
                      severity: Optional[Severity] = None) -> QualityCheckResult:
        return QualityCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            column_name=column_name,
            passed=passed,
            severity=severity or self.severity,
            message=message,
            details=details or {},
            recommendation=recommendation
        )


class MissingValueCheck(QualityCheck):
    def __init__(self):
        super().__init__("FQ001", "Column has >95% missing", Severity.CRITICAL)
        self.threshold_critical = 95.0
        self.threshold_high = 70.0
        self.threshold_medium = 20.0

    def run(self, column_name: str, universal_metrics: Any) -> QualityCheckResult:
        null_pct = universal_metrics.null_percentage

        if null_pct > self.threshold_critical:
            return self.create_result(
                column_name, False,
                f"Critical: {null_pct}% missing values (>{self.threshold_critical}%)",
                {"null_percentage": null_pct, "null_count": universal_metrics.null_count},
                "Consider imputation strategy or feature removal if not informative",
                Severity.CRITICAL
            )
        elif null_pct > self.threshold_high:
            return self.create_result(
                column_name, False,
                f"High: {null_pct}% missing values (>{self.threshold_high}%)",
                {"null_percentage": null_pct, "null_count": universal_metrics.null_count},
                "Review imputation strategy or investigate data collection issues",
                Severity.HIGH
            )
        elif null_pct > self.threshold_medium:
            return self.create_result(
                column_name, True,
                f"Medium: {null_pct}% missing values (>{self.threshold_medium}%)",
                {"null_percentage": null_pct, "null_count": universal_metrics.null_count},
                "Monitor missingness pattern and consider simple imputation",
                Severity.MEDIUM
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable missing values: {null_pct}%",
                {"null_percentage": null_pct, "null_count": universal_metrics.null_count}
            )


class HighCardinalityCheck(QualityCheck):
    def __init__(self):
        super().__init__("CAT001", "High Cardinality Categorical", Severity.MEDIUM)
        self.threshold_ratio = 0.95

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        cardinality_ratio = categorical_metrics.cardinality_ratio

        if cardinality_ratio > self.threshold_ratio:
            return self.create_result(
                column_name, False,
                f"Very high cardinality ratio: {cardinality_ratio:.2%}",
                {"cardinality": categorical_metrics.cardinality, "cardinality_ratio": cardinality_ratio},
                f"Consider using {categorical_metrics.encoding_recommendation} encoding or treating as text"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable cardinality ratio: {cardinality_ratio:.2%}",
                {"cardinality": categorical_metrics.cardinality, "cardinality_ratio": cardinality_ratio}
            )


class LowCardinalityCheck(QualityCheck):
    def __init__(self):
        super().__init__("NUM001", "Low Cardinality Numeric", Severity.LOW)
        self.threshold = 10

    def run(self, column_name: str, universal_metrics: Any, column_type: ColumnType) -> Optional[QualityCheckResult]:
        if column_type not in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            return None

        distinct_count = universal_metrics.distinct_count

        if distinct_count < self.threshold:
            return self.create_result(
                column_name, False,
                f"Low cardinality for numeric: {distinct_count} unique values",
                {"distinct_count": distinct_count},
                "Consider treating as categorical or ordinal feature"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable cardinality for numeric: {distinct_count} unique values",
                {"distinct_count": distinct_count}
            )


class ConstantFeatureCheck(QualityCheck):
    def __init__(self):
        super().__init__("FQ003", "Column is constant", Severity.CRITICAL)
        self.threshold_ratio = 1.0

    def run(self, column_name: str, universal_metrics: Any, column_type: Optional[ColumnType] = None) -> QualityCheckResult:
        if universal_metrics.total_count == 0:
            return self.create_result(column_name, True, "Empty column", {})

        distinct_count = universal_metrics.distinct_count

        if distinct_count == 1:
            return self.create_result(
                column_name, False,
                f"Column is constant: only 1 distinct value ({universal_metrics.most_common_value})",
                {"distinct_count": 1, "constant_value": universal_metrics.most_common_value},
                "CRITICAL: Remove constant column - provides no information for modeling"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Column has {distinct_count} distinct values",
                {"distinct_count": distinct_count}
            )


class ImbalancedTargetCheck(QualityCheck):
    def __init__(self):
        super().__init__("CAT002", "Imbalanced Target Variable", Severity.HIGH)
        self.threshold_severe = 20.0
        self.threshold_moderate = 5.0

    def run(self, column_name: str, target_metrics: Any) -> Optional[QualityCheckResult]:
        if target_metrics is None:
            return None

        imbalance_ratio = target_metrics.imbalance_ratio
        minority_pct = target_metrics.minority_percentage

        if imbalance_ratio > self.threshold_severe:
            return self.create_result(
                column_name, False,
                f"Severe imbalance: {imbalance_ratio:.1f}:1 ratio, minority class {minority_pct}%",
                {"imbalance_ratio": imbalance_ratio, "minority_percentage": minority_pct,
                 "minority_class": target_metrics.minority_class},
                "Apply SMOTE, class weights, or stratified sampling"
            )
        elif imbalance_ratio > self.threshold_moderate:
            return self.create_result(
                column_name, False,
                f"Moderate imbalance: {imbalance_ratio:.1f}:1 ratio, minority class {minority_pct}%",
                {"imbalance_ratio": imbalance_ratio, "minority_percentage": minority_pct,
                 "minority_class": target_metrics.minority_class},
                "Consider class weights or balanced sampling"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable balance: {imbalance_ratio:.1f}:1 ratio, minority class {minority_pct}%",
                {"imbalance_ratio": imbalance_ratio, "minority_percentage": minority_pct}
            )


class TargetNullCheck(QualityCheck):
    def __init__(self):
        super().__init__("TG001", "Target Contains Nulls", Severity.CRITICAL)

    def run(self, column_name: str, universal_metrics: Any) -> Optional[QualityCheckResult]:
        if universal_metrics is None:
            return None

        if universal_metrics.null_count > 0:
            return self.create_result(
                column_name, False,
                f"Target variable contains {universal_metrics.null_count} null values ({universal_metrics.null_percentage}%)",
                {"null_count": universal_metrics.null_count, "null_percentage": universal_metrics.null_percentage},
                "CRITICAL: Target variable must not contain nulls. Remove or impute before modeling."
            )
        else:
            return self.create_result(
                column_name, True,
                "Target variable has no null values",
                {"null_count": 0}
            )


class SingleClassTargetCheck(QualityCheck):
    def __init__(self):
        super().__init__("TG005", "Single Class Target", Severity.CRITICAL)

    def run(self, column_name: str, target_metrics: Any) -> Optional[QualityCheckResult]:
        if target_metrics is None:
            return None

        if target_metrics.n_classes == 1:
            return self.create_result(
                column_name, False,
                f"Target variable has only 1 class: {list(target_metrics.class_distribution.keys())[0]}",
                {"n_classes": 1, "classes": list(target_metrics.class_distribution.keys())},
                "CRITICAL: Cannot train a classifier with only one class. Check data filtering or sampling."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Target variable has {target_metrics.n_classes} classes",
                {"n_classes": target_metrics.n_classes}
            )


class TargetSevereImbalanceCheck(QualityCheck):
    def __init__(self):
        super().__init__("TG002", "Target Severe Imbalance", Severity.HIGH)
        self.threshold = 1.0  # < 1%

    def run(self, column_name: str, target_metrics: Any) -> Optional[QualityCheckResult]:
        if target_metrics is None:
            return None

        minority_pct = target_metrics.minority_percentage

        if minority_pct < self.threshold:
            return self.create_result(
                column_name, False,
                f"Target has severe class imbalance: minority class {minority_pct}% (< {self.threshold}%)",
                {"minority_percentage": minority_pct, "minority_class": target_metrics.minority_class,
                 "imbalance_ratio": target_metrics.imbalance_ratio},
                "Apply SMOTE, class weights, or consider alternative algorithms (e.g., anomaly detection)."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Minority class at {minority_pct}% (>= {self.threshold}%)",
                {"minority_percentage": minority_pct}
            )


class TargetModerateImbalanceCheck(QualityCheck):
    def __init__(self):
        super().__init__("TG003", "Target Moderate Imbalance", Severity.MEDIUM)
        self.threshold = 10.0  # < 10%

    def run(self, column_name: str, target_metrics: Any) -> Optional[QualityCheckResult]:
        if target_metrics is None:
            return None

        minority_pct = target_metrics.minority_percentage

        if minority_pct < self.threshold:
            return self.create_result(
                column_name, False,
                f"Target has moderate class imbalance: minority class {minority_pct}% (< {self.threshold}%)",
                {"minority_percentage": minority_pct, "minority_class": target_metrics.minority_class,
                 "imbalance_ratio": target_metrics.imbalance_ratio},
                "Consider class weights, stratified sampling, or balanced algorithms."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Minority class at {minority_pct}% (>= {self.threshold}%)",
                {"minority_percentage": minority_pct}
            )


class TargetUnexpectedClassesCheck(QualityCheck):
    def __init__(self, expected_classes: Optional[int] = None):
        super().__init__("TG004", "Target Unexpected Classes", Severity.HIGH)
        self.expected_classes = expected_classes

    def run(self, column_name: str, target_metrics: Any) -> Optional[QualityCheckResult]:
        if target_metrics is None or self.expected_classes is None:
            return None

        n_classes = target_metrics.n_classes

        if n_classes != self.expected_classes:
            return self.create_result(
                column_name, False,
                f"Target has {n_classes} classes, expected {self.expected_classes}",
                {"n_classes": n_classes, "expected_classes": self.expected_classes,
                 "classes": list(target_metrics.class_distribution.keys())},
                f"Investigate class mismatch. Check for data leakage, incorrect filtering, or configuration error."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Target has expected {n_classes} classes",
                {"n_classes": n_classes}
            )


class SkewnessCheck(QualityCheck):
    def __init__(self):
        super().__init__("NUM002", "Extreme Skewness", Severity.MEDIUM)
        self.threshold_severe = 3.0
        self.threshold_moderate = 1.0

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None or numeric_metrics.skewness is None:
            return None

        skewness = abs(numeric_metrics.skewness)

        if skewness > self.threshold_severe:
            return self.create_result(
                column_name, False,
                f"Extreme skewness: {numeric_metrics.skewness:.2f}",
                {"skewness": numeric_metrics.skewness},
                "Apply log, sqrt, or Box-Cox transformation"
            )
        elif skewness > self.threshold_moderate:
            return self.create_result(
                column_name, False,
                f"Moderate skewness: {numeric_metrics.skewness:.2f}",
                {"skewness": numeric_metrics.skewness},
                "Consider transformation for linear models"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable skewness: {numeric_metrics.skewness:.2f}",
                {"skewness": numeric_metrics.skewness}
            )


class OutlierCheck(QualityCheck):
    def __init__(self):
        super().__init__("NUM003", "Excessive Outliers", Severity.MEDIUM)
        self.threshold_high = 10.0
        self.threshold_medium = 5.0

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        outlier_pct = numeric_metrics.outlier_percentage

        if outlier_pct > self.threshold_high:
            return self.create_result(
                column_name, False,
                f"High outlier percentage: {outlier_pct}%",
                {"outlier_count_iqr": numeric_metrics.outlier_count_iqr,
                 "outlier_percentage": outlier_pct},
                "Review outliers for data quality issues or apply winsorization/clipping"
            )
        elif outlier_pct > self.threshold_medium:
            return self.create_result(
                column_name, False,
                f"Moderate outlier percentage: {outlier_pct}%",
                {"outlier_count_iqr": numeric_metrics.outlier_count_iqr,
                 "outlier_percentage": outlier_pct},
                "Consider robust scaling or outlier treatment"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable outlier percentage: {outlier_pct}%",
                {"outlier_percentage": outlier_pct}
            )


class ZeroInflationCheck(QualityCheck):
    def __init__(self):
        super().__init__("NUM004", "Zero-Inflated Feature", Severity.LOW)
        self.threshold = 50.0

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        zero_pct = numeric_metrics.zero_percentage

        if zero_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"Zero-inflated: {zero_pct}% zeros",
                {"zero_count": numeric_metrics.zero_count, "zero_percentage": zero_pct},
                "Consider zero-inflated models or separate binary indicator"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable zero percentage: {zero_pct}%",
                {"zero_percentage": zero_pct}
            )


class IdentifierLeakageCheck(QualityCheck):
    def __init__(self):
        super().__init__("LEAK001", "Identifier Column in Features", Severity.CRITICAL)

    def run(self, column_name: str, column_type: ColumnType, should_use_as_feature: bool) -> Optional[QualityCheckResult]:
        if column_type != ColumnType.IDENTIFIER:
            return None

        if should_use_as_feature:
            return self.create_result(
                column_name, False,
                "Identifier column marked as feature",
                {"column_type": column_type.value},
                "Remove identifier from feature set to prevent data leakage"
            )
        else:
            return self.create_result(
                column_name, True,
                "Identifier correctly excluded from features",
                {"column_type": column_type.value}
            )


class DatetimeFutureLeakageCheck(QualityCheck):
    def __init__(self):
        super().__init__("DT001", "Future Dates Detected", Severity.HIGH)

    def run(self, column_name: str, datetime_metrics: Any) -> Optional[QualityCheckResult]:
        if datetime_metrics is None:
            return None

        future_count = datetime_metrics.future_date_count

        if future_count > 0:
            return self.create_result(
                column_name, False,
                f"Found {future_count} future dates",
                {"future_date_count": future_count},
                "Investigate data quality issues or potential temporal leakage"
            )
        else:
            return self.create_result(
                column_name, True,
                "No future dates detected",
                {"future_date_count": 0}
            )


class PlaceholderDateCheck(QualityCheck):
    def __init__(self):
        super().__init__("DT002", "Placeholder Dates", Severity.MEDIUM)
        self.threshold = 0.05

    def run(self, column_name: str, datetime_metrics: Any, total_count: int) -> Optional[QualityCheckResult]:
        if datetime_metrics is None or total_count == 0:
            return None

        placeholder_count = datetime_metrics.placeholder_count
        placeholder_pct = (placeholder_count / total_count) * 100

        if placeholder_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"Placeholder dates found: {placeholder_count} ({placeholder_pct:.2f}%)",
                {"placeholder_count": placeholder_count, "placeholder_percentage": placeholder_pct},
                "Replace placeholder dates with null or investigate data quality"
            )
        else:
            return self.create_result(
                column_name, True,
                f"No significant placeholder dates: {placeholder_count}",
                {"placeholder_count": placeholder_count}
            )


class RareCategoryCheck(QualityCheck):
    def __init__(self):
        super().__init__("CAT003", "High Rare Category Count", Severity.MEDIUM)
        self.threshold_pct = 20.0

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        rare_pct = categorical_metrics.rare_category_percentage
        rare_count = categorical_metrics.rare_category_count

        if rare_pct > self.threshold_pct:
            return self.create_result(
                column_name, False,
                f"High rare category percentage: {rare_pct}% ({rare_count} categories)",
                {"rare_category_count": rare_count, "rare_category_percentage": rare_pct},
                "Consider grouping rare categories or using target encoding"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable rare categories: {rare_pct}% ({rare_count} categories)",
                {"rare_category_count": rare_count, "rare_category_percentage": rare_pct}
            )


class UnknownCategoryCheck(QualityCheck):
    def __init__(self):
        super().__init__("CAT004", "Unknown Categories Present", Severity.LOW)

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        has_unknown = categorical_metrics.contains_unknown

        if has_unknown:
            return self.create_result(
                column_name, False,
                "Contains unknown/missing value indicators",
                {"contains_unknown": has_unknown},
                "Replace with proper nulls or create explicit category"
            )
        else:
            return self.create_result(
                column_name, True,
                "No unknown value indicators found",
                {"contains_unknown": has_unknown}
            )


class PIIDetectedCheck(QualityCheck):
    def __init__(self):
        super().__init__("TX001", "PII Detected", Severity.CRITICAL)

    def run(self, column_name: str, text_metrics: Any) -> Optional[QualityCheckResult]:
        if text_metrics is None:
            return None

        if text_metrics.pii_detected:
            pii_types_str = ", ".join(text_metrics.pii_types)
            return self.create_result(
                column_name, False,
                f"PII detected: {pii_types_str}",
                {"pii_types": text_metrics.pii_types},
                "CRITICAL: Remove PII or mask sensitive data before processing. Consider data anonymization techniques."
            )
        else:
            return self.create_result(
                column_name, True,
                "No PII detected",
                {"pii_detected": False}
            )


class EmptyTextCheck(QualityCheck):
    def __init__(self):
        super().__init__("TX002", "Mostly Empty Text", Severity.HIGH)
        self.threshold = 50.0

    def run(self, column_name: str, text_metrics: Any) -> Optional[QualityCheckResult]:
        if text_metrics is None:
            return None

        empty_pct = text_metrics.empty_percentage

        if empty_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"High percentage of empty text: {empty_pct}%",
                {"empty_percentage": empty_pct, "empty_count": text_metrics.empty_count},
                "Review data quality and consider imputation or feature removal"
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable empty text percentage: {empty_pct}%",
                {"empty_percentage": empty_pct}
            )


class ShortTextCheck(QualityCheck):
    def __init__(self):
        super().__init__("TX003", "Very Short Texts", Severity.MEDIUM)
        self.threshold = 10.0

    def run(self, column_name: str, text_metrics: Any) -> Optional[QualityCheckResult]:
        if text_metrics is None:
            return None

        avg_length = text_metrics.length_mean

        if avg_length < self.threshold:
            return self.create_result(
                column_name, False,
                f"Very short average text length: {avg_length:.1f} characters",
                {"length_mean": avg_length},
                "May be better treated as categorical. Consider reclassifying column type."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable text length: {avg_length:.1f} characters",
                {"length_mean": avg_length}
            )


class InfiniteValuesCheck(QualityCheck):
    def __init__(self):
        super().__init__("NC006", "Infinite Values", Severity.CRITICAL)

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        if numeric_metrics.inf_count > 0:
            return self.create_result(
                column_name, False,
                f"Column contains {numeric_metrics.inf_count} infinite values ({numeric_metrics.inf_percentage}%)",
                {"inf_count": numeric_metrics.inf_count, "inf_percentage": numeric_metrics.inf_percentage},
                "CRITICAL: Remove or replace infinite values before processing. Use imputation or capping strategies."
            )
        else:
            return self.create_result(
                column_name, True,
                "No infinite values detected",
                {"inf_count": 0, "inf_percentage": 0.0}
            )


class ExtremeOutliersCheck(QualityCheck):
    def __init__(self):
        super().__init__("NC001", "Extreme Outliers", Severity.HIGH)
        self.threshold = 5.0  # > 5%

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        outlier_pct = numeric_metrics.outlier_percentage

        if outlier_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"Extreme outlier percentage: {outlier_pct}% (> {self.threshold}%)",
                {"outlier_percentage": outlier_pct, "outlier_count_iqr": numeric_metrics.outlier_count_iqr,
                 "outlier_count_zscore": numeric_metrics.outlier_count_zscore},
                "Apply robust scaling, winsorization, or consider removing outliers."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable outlier percentage: {outlier_pct}%",
                {"outlier_percentage": outlier_pct}
            )


class ModerateOutliersCheck(QualityCheck):
    def __init__(self):
        super().__init__("NC002", "Moderate Outliers", Severity.MEDIUM)
        self.threshold = 1.0  # > 1%

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        outlier_pct = numeric_metrics.outlier_percentage

        if outlier_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"Moderate outlier percentage: {outlier_pct}% (> {self.threshold}%)",
                {"outlier_percentage": outlier_pct, "outlier_count_iqr": numeric_metrics.outlier_count_iqr},
                "Consider investigating outliers and applying transformations."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Low outlier percentage: {outlier_pct}%",
                {"outlier_percentage": outlier_pct}
            )


class HighSkewnessCheck(QualityCheck):
    def __init__(self):
        super().__init__("NC003", "High Skewness", Severity.MEDIUM)
        self.threshold = 2.0  # |skewness| > 2

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None or numeric_metrics.skewness is None:
            return None

        skewness = abs(numeric_metrics.skewness)

        if skewness > self.threshold:
            return self.create_result(
                column_name, False,
                f"High skewness detected: {numeric_metrics.skewness:.2f} (|skew| > {self.threshold})",
                {"skewness": numeric_metrics.skewness, "abs_skewness": skewness},
                "Apply log, sqrt, or Box-Cox transformation to reduce skewness."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable skewness: {numeric_metrics.skewness:.2f}",
                {"skewness": numeric_metrics.skewness}
            )


class NumericZeroInflationCheck(QualityCheck):
    def __init__(self):
        super().__init__("NC004", "Zero Inflation", Severity.MEDIUM)
        self.threshold = 50.0  # > 50%

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        zero_pct = numeric_metrics.zero_percentage

        if zero_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"Zero-inflated: {zero_pct}% zeros (> {self.threshold}%)",
                {"zero_percentage": zero_pct, "zero_count": numeric_metrics.zero_count},
                "Consider zero-inflated models, indicator variable, or separate handling of zeros."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable zero percentage: {zero_pct}%",
                {"zero_percentage": zero_pct}
            )


class UnexpectedNegativesCheck(QualityCheck):
    def __init__(self, allow_negatives: bool = True):
        super().__init__("NC005", "Unexpected Negative Values", Severity.HIGH)
        self.allow_negatives = allow_negatives

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None or self.allow_negatives:
            return None

        negative_count = numeric_metrics.negative_count

        if negative_count > 0:
            return self.create_result(
                column_name, False,
                f"Column contains {negative_count} negative values ({numeric_metrics.negative_percentage}%), but negatives not expected",
                {"negative_count": negative_count, "negative_percentage": numeric_metrics.negative_percentage},
                "Investigate negative values. May indicate data errors or need for transformation."
            )
        else:
            return self.create_result(
                column_name, True,
                "No negative values found",
                {"negative_count": 0}
            )


class ConstantValueCheck(QualityCheck):
    def __init__(self):
        super().__init__("NC007", "Constant Value", Severity.HIGH)

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        if numeric_metrics.std == 0:
            return self.create_result(
                column_name, False,
                f"Column has constant value (std = 0)",
                {"std": 0, "mean": numeric_metrics.mean},
                "Remove constant column - provides no information for modeling."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Column has variance (std = {numeric_metrics.std:.4f})",
                {"std": numeric_metrics.std}
            )


class SuspiciousPrecisionCheck(QualityCheck):
    def __init__(self):
        super().__init__("NC008", "Suspicious Precision", Severity.LOW)

    def run(self, column_name: str, series: pd.Series) -> Optional[QualityCheckResult]:
        # This check needs access to raw series to check decimal places
        # Note: This is a simplified implementation
        if series is None or len(series) == 0:
            return None

        clean_series = series.dropna()
        if len(clean_series) == 0:
            return None

        # Check if all values end in .00 (are whole numbers)
        all_whole = all((isinstance(v, (int, float)) and v == int(v)) for v in clean_series[:min(100, len(clean_series))])

        if all_whole and len(clean_series) > 10:
            return self.create_result(
                column_name, False,
                "All sampled values are whole numbers - may indicate precision loss or rounding",
                {"precision_issue": "all_whole_numbers"},
                "Verify source data precision and check for unintended rounding."
            )
        else:
            return self.create_result(
                column_name, True,
                "Precision appears normal",
                {}
            )


class HighOutliersCheck(QualityCheck):
    def __init__(self):
        super().__init__("FQ005", "Column has >50% outliers", Severity.HIGH)
        self.threshold = 50.0

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        outlier_pct = numeric_metrics.outlier_percentage

        if outlier_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"Extreme outlier percentage: {outlier_pct}% (> {self.threshold}%)",
                {"outlier_percentage": outlier_pct, "outlier_count_iqr": numeric_metrics.outlier_count_iqr},
                "HIGH: Column may be unreliable for modeling. Consider robust transformations or removal."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable outlier percentage: {outlier_pct}%",
                {"outlier_percentage": outlier_pct}
            )


class AllValuesOutliersCheck(QualityCheck):
    def __init__(self):
        super().__init__("FQ011", "All values are outliers", Severity.CRITICAL)

    def run(self, column_name: str, numeric_metrics: Any) -> Optional[QualityCheckResult]:
        if numeric_metrics is None:
            return None

        outlier_pct = numeric_metrics.outlier_percentage

        if outlier_pct == 100.0:
            return self.create_result(
                column_name, False,
                "CRITICAL: All values are outliers (100%)",
                {"outlier_percentage": 100.0},
                "CRITICAL: Column may have data quality issues. Investigate and consider removal."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Not all values are outliers: {outlier_pct}%",
                {"outlier_percentage": outlier_pct}
            )


class UnknownColumnTypeCheck(QualityCheck):
    def __init__(self):
        super().__init__("FQ008", "Unknown column type", Severity.MEDIUM)

    def run(self, column_name: str, column_type: ColumnType) -> Optional[QualityCheckResult]:
        if column_type == ColumnType.UNKNOWN:
            return self.create_result(
                column_name, False,
                "Column type could not be determined",
                {"column_type": "UNKNOWN"},
                "Manually specify column type or investigate data format."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Column type determined: {column_type.value}",
                {"column_type": column_type.value}
            )


class VeryHighCardinalityNominalCheck(QualityCheck):
    def __init__(self):
        super().__init__("FQ009", "Very high cardinality nominal", Severity.MEDIUM)
        self.threshold = 1000

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        cardinality = categorical_metrics.cardinality

        if cardinality > self.threshold:
            return self.create_result(
                column_name, False,
                f"Very high cardinality: {cardinality} unique categories (> {self.threshold})",
                {"cardinality": cardinality},
                "Consider treating as high cardinality or using hashing/embedding encoding."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable cardinality: {cardinality} unique categories",
                {"cardinality": cardinality}
            )


class UnrealisticDateRangeCheck(QualityCheck):
    def __init__(self):
        super().__init__("FQ012", "Date range unrealistic", Severity.HIGH)
        self.threshold_years = 100

    def run(self, column_name: str, datetime_metrics: Any) -> Optional[QualityCheckResult]:
        if datetime_metrics is None:
            return None

        date_range_days = datetime_metrics.date_range_days
        date_range_years = date_range_days / 365.25

        if date_range_years > self.threshold_years:
            return self.create_result(
                column_name, False,
                f"Unrealistic date range: {date_range_years:.1f} years (> {self.threshold_years} years)",
                {"date_range_days": date_range_days, "date_range_years": round(date_range_years, 1),
                 "min_date": datetime_metrics.min_date, "max_date": datetime_metrics.max_date},
                "HIGH: Date range spans > 100 years. Review for data quality issues."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable date range: {date_range_years:.1f} years",
                {"date_range_days": date_range_days, "date_range_years": round(date_range_years, 1)}
            )


class HighUniquenessTextCheck(QualityCheck):
    def __init__(self):
        super().__init__("TX004", "High Uniqueness Text", Severity.MEDIUM)
        self.threshold = 0.95

    def run(self, column_name: str, universal_metrics: Any) -> Optional[QualityCheckResult]:
        if universal_metrics is None:
            return None

        distinct_pct = universal_metrics.distinct_percentage / 100.0

        if distinct_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"Text column has very high uniqueness: {universal_metrics.distinct_percentage}% unique (> {self.threshold * 100}%)",
                {"distinct_percentage": universal_metrics.distinct_percentage, "distinct_count": universal_metrics.distinct_count},
                "Text column may actually be an identifier. Consider reclassifying as IDENTIFIER type."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable text uniqueness: {universal_metrics.distinct_percentage}%",
                {"distinct_percentage": universal_metrics.distinct_percentage}
            )


class BinaryNotBinaryCheck(QualityCheck):
    def __init__(self):
        super().__init__("BN001", "Not Binary", Severity.CRITICAL)

    def run(self, column_name: str, universal_metrics: Any) -> Optional[QualityCheckResult]:
        if universal_metrics is None:
            return None

        distinct_count = universal_metrics.distinct_count

        if distinct_count != 2:
            return self.create_result(
                column_name, False,
                f"Column marked as binary but has {distinct_count} distinct values (expected 2)",
                {"distinct_count": distinct_count},
                "CRITICAL: Binary columns must have exactly 2 distinct values. Review column type or data."
            )
        else:
            return self.create_result(
                column_name, True,
                "Column has exactly 2 distinct values",
                {"distinct_count": 2}
            )


class BinarySevereImbalanceCheck(QualityCheck):
    def __init__(self):
        super().__init__("BN002", "Binary Severe Imbalance", Severity.MEDIUM)
        self.threshold_low = 1.0
        self.threshold_high = 99.0

    def run(self, column_name: str, binary_metrics: Any) -> Optional[QualityCheckResult]:
        if binary_metrics is None:
            return None

        true_pct = binary_metrics.true_percentage

        if true_pct < self.threshold_low or true_pct > self.threshold_high:
            return self.create_result(
                column_name, False,
                f"Severe binary imbalance: {true_pct}% true values (< {self.threshold_low}% or > {self.threshold_high}%)",
                {"true_percentage": true_pct, "balance_ratio": binary_metrics.balance_ratio},
                "Consider class balancing techniques or check if column should be binary."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable binary balance: {true_pct}% true values",
                {"true_percentage": true_pct}
            )


class BinaryAllSameValueCheck(QualityCheck):
    def __init__(self):
        super().__init__("BN003", "Binary All Same Value", Severity.HIGH)

    def run(self, column_name: str, universal_metrics: Any) -> Optional[QualityCheckResult]:
        if universal_metrics is None:
            return None

        distinct_count = universal_metrics.distinct_count

        if distinct_count == 1:
            return self.create_result(
                column_name, False,
                f"Binary column has only 1 distinct value: {universal_metrics.most_common_value}",
                {"distinct_count": 1, "value": universal_metrics.most_common_value},
                "Binary column provides no information. Consider removing."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Binary column has {distinct_count} distinct values",
                {"distinct_count": distinct_count}
            )


class BinaryUnexpectedValuesCheck(QualityCheck):
    def __init__(self):
        super().__init__("BN004", "Binary Unexpected Values", Severity.HIGH)

    def run(self, column_name: str, binary_metrics: Any) -> Optional[QualityCheckResult]:
        if binary_metrics is None:
            return None

        values_found = binary_metrics.values_found
        expected_values = {0, 1, 0.0, 1.0, True, False, "0", "1", "yes", "Yes", "YES", "no", "No", "NO",
                          "true", "True", "TRUE", "false", "False", "FALSE", "y", "Y", "n", "N"}

        unexpected = [v for v in values_found if v not in expected_values]

        if len(unexpected) > 0:
            return self.create_result(
                column_name, False,
                f"Binary column contains unexpected values: {unexpected[:5]}",
                {"unexpected_values": unexpected[:5], "values_found": values_found},
                "Standardize binary values to 0/1 or True/False format."
            )
        else:
            return self.create_result(
                column_name, True,
                "Binary column contains only expected values",
                {"values_found": values_found}
            )


class DatetimeFormatInconsistentCheck(QualityCheck):
    def __init__(self):
        super().__init__("DT003", "Datetime Format Inconsistent", Severity.MEDIUM)
        self.threshold = 95.0

    def run(self, column_name: str, datetime_metrics: Any) -> Optional[QualityCheckResult]:
        if datetime_metrics is None or datetime_metrics.format_consistency is None:
            return None

        format_consistency = datetime_metrics.format_consistency

        if format_consistency < self.threshold:
            return self.create_result(
                column_name, False,
                f"Datetime format inconsistent: {format_consistency}% match format '{datetime_metrics.format_detected}' (< {self.threshold}%)",
                {"format_consistency": format_consistency, "format_detected": datetime_metrics.format_detected},
                "Standardize datetime format during data loading or preprocessing."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Datetime format consistent: {format_consistency}% match format '{datetime_metrics.format_detected}'",
                {"format_consistency": format_consistency, "format_detected": datetime_metrics.format_detected}
            )


class DatetimeMixedTimezonesCheck(QualityCheck):
    def __init__(self):
        super().__init__("DT004", "Mixed Timezones", Severity.MEDIUM)

    def run(self, column_name: str, datetime_metrics: Any) -> Optional[QualityCheckResult]:
        if datetime_metrics is None:
            return None

        timezone_consistent = datetime_metrics.timezone_consistent

        if not timezone_consistent:
            return self.create_result(
                column_name, False,
                "Mixed timezones detected in datetime column",
                {"timezone_consistent": False},
                "Convert all datetimes to a single timezone (e.g., UTC) for consistency."
            )
        else:
            return self.create_result(
                column_name, True,
                "Timezones are consistent",
                {"timezone_consistent": True}
            )


class DatetimeInvalidDatesCheck(QualityCheck):
    def __init__(self):
        super().__init__("DT005", "Invalid Dates", Severity.CRITICAL)

    def run(self, column_name: str, series: pd.Series, universal_metrics: Any) -> Optional[QualityCheckResult]:
        if series is None:
            return None

        clean_series = series.dropna()
        invalid_count = 0

        if not is_datetime64_any_dtype(clean_series):
            for val in clean_series:
                try:
                    pd.to_datetime(val, format='mixed')
                except:
                    invalid_count += 1

        if invalid_count > 0:
            invalid_pct = (invalid_count / len(series)) * 100 if len(series) > 0 else 0.0
            return self.create_result(
                column_name, False,
                f"Column contains {invalid_count} invalid dates ({invalid_pct:.2f}%)",
                {"invalid_count": invalid_count, "invalid_percentage": invalid_pct},
                "CRITICAL: Fix or remove invalid dates before processing."
            )
        else:
            return self.create_result(
                column_name, True,
                "No invalid dates detected",
                {"invalid_count": 0}
            )


class DatetimeUnrealisticRangeCheck(QualityCheck):
    def __init__(self):
        super().__init__("DT006", "Unrealistic Date Range", Severity.MEDIUM)
        self.threshold_years = 50

    def run(self, column_name: str, datetime_metrics: Any) -> Optional[QualityCheckResult]:
        if datetime_metrics is None:
            return None

        date_range_days = datetime_metrics.date_range_days
        date_range_years = date_range_days / 365.25

        if date_range_years > self.threshold_years:
            return self.create_result(
                column_name, False,
                f"Unrealistic date range: {date_range_years:.1f} years (> {self.threshold_years} years)",
                {"date_range_days": date_range_days, "date_range_years": round(date_range_years, 1),
                 "min_date": datetime_metrics.min_date, "max_date": datetime_metrics.max_date},
                "Review min/max dates for data quality issues or placeholder values."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable date range: {date_range_years:.1f} years",
                {"date_range_days": date_range_days, "date_range_years": round(date_range_years, 1)}
            )


class VeryHighCardinalityCheck(QualityCheck):
    def __init__(self):
        super().__init__("CN001", "Very High Cardinality", Severity.HIGH)
        self.threshold = 100

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        cardinality = categorical_metrics.cardinality

        if cardinality > self.threshold:
            return self.create_result(
                column_name, False,
                f"Very high cardinality: {cardinality} unique categories (> {self.threshold})",
                {"cardinality": cardinality},
                f"Consider using {categorical_metrics.encoding_recommendation} encoding or treating as high cardinality feature."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable cardinality: {cardinality} unique categories",
                {"cardinality": cardinality}
            )


class HighCardinalityCategoricalCheck(QualityCheck):
    def __init__(self):
        super().__init__("CN002", "High Cardinality Categorical", Severity.MEDIUM)
        self.threshold = 50

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        cardinality = categorical_metrics.cardinality

        if cardinality > self.threshold:
            return self.create_result(
                column_name, False,
                f"High cardinality: {cardinality} unique categories (> {self.threshold})",
                {"cardinality": cardinality},
                f"Consider using {categorical_metrics.encoding_recommendation} encoding."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable cardinality: {cardinality} unique categories",
                {"cardinality": cardinality}
            )


class ManyRareCategoriesCheck(QualityCheck):
    def __init__(self):
        super().__init__("CN003", "Many Rare Categories", Severity.MEDIUM)
        self.threshold = 10

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        rare_count = categorical_metrics.rare_category_count

        if rare_count > self.threshold:
            return self.create_result(
                column_name, False,
                f"Many rare categories: {rare_count} categories with < 1% frequency (> {self.threshold})",
                {"rare_category_count": rare_count, "rare_categories": categorical_metrics.rare_categories[:5]},
                "Consider grouping rare categories into 'Other' or using frequency encoding."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable rare categories: {rare_count} categories",
                {"rare_category_count": rare_count}
            )


class SignificantRareVolumeCheck(QualityCheck):
    def __init__(self):
        super().__init__("CN004", "Significant Rare Category Volume", Severity.HIGH)
        self.threshold = 20.0

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        rare_pct = categorical_metrics.rare_category_percentage

        if rare_pct > self.threshold:
            return self.create_result(
                column_name, False,
                f"High rare category volume: {rare_pct}% of rows in rare categories (> {self.threshold}%)",
                {"rare_category_percentage": rare_pct, "rare_category_count": categorical_metrics.rare_category_count},
                "Group rare categories or use encoding that handles high cardinality (target encoding, embedding)."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Acceptable rare category volume: {rare_pct}%",
                {"rare_category_percentage": rare_pct}
            )


class CaseInconsistencyCheck(QualityCheck):
    def __init__(self):
        super().__init__("CN005", "Case Inconsistency", Severity.LOW)

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        case_variations = categorical_metrics.case_variations

        if len(case_variations) > 0:
            return self.create_result(
                column_name, False,
                f"Case inconsistency detected: {len(case_variations)} variations found",
                {"case_variations": case_variations},
                "Standardize case (e.g., lowercase all values) during preprocessing."
            )
        else:
            return self.create_result(
                column_name, True,
                "No case inconsistency detected",
                {"case_variations": []}
            )


class WhitespaceIssuesCheck(QualityCheck):
    def __init__(self):
        super().__init__("CN006", "Whitespace Issues", Severity.LOW)

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        whitespace_issues = categorical_metrics.whitespace_issues

        if len(whitespace_issues) > 0:
            return self.create_result(
                column_name, False,
                f"Whitespace issues detected: {len(whitespace_issues)} values with leading/trailing spaces",
                {"whitespace_issues": whitespace_issues},
                "Strip leading/trailing whitespace during preprocessing."
            )
        else:
            return self.create_result(
                column_name, True,
                "No whitespace issues detected",
                {"whitespace_issues": []}
            )


class SingleCategoryCheck(QualityCheck):
    def __init__(self):
        super().__init__("CN007", "Single Category Only", Severity.HIGH)

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        cardinality = categorical_metrics.cardinality

        if cardinality == 1:
            return self.create_result(
                column_name, False,
                f"Column has only 1 category: {categorical_metrics.top_categories[0][0]}",
                {"cardinality": 1, "category": categorical_metrics.top_categories[0][0]},
                "Remove constant categorical column - provides no information for modeling."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Column has {cardinality} categories",
                {"cardinality": cardinality}
            )


class PossibleTyposCheck(QualityCheck):
    def __init__(self):
        super().__init__("CN008", "Possible Typos Detected", Severity.MEDIUM)

    def run(self, column_name: str, categorical_metrics: Any) -> Optional[QualityCheckResult]:
        if categorical_metrics is None:
            return None

        try:
            from difflib import SequenceMatcher
        except ImportError:
            return None

        unique_values = list(categorical_metrics.value_counts.keys())[:100]
        similar_pairs = []

        for i, val1 in enumerate(unique_values):
            for val2 in unique_values[i+1:]:
                if len(val1) > 3 and len(val2) > 3:
                    ratio = SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
                    if 0.8 < ratio < 1.0:
                        similar_pairs.append(f"{val1} ~ {val2}")
                        if len(similar_pairs) >= 5:
                            break
            if len(similar_pairs) >= 5:
                break

        if len(similar_pairs) > 0:
            return self.create_result(
                column_name, False,
                f"Possible typos detected: {len(similar_pairs)} similar value pairs found",
                {"similar_pairs": similar_pairs},
                "Review similar values for potential typos and standardize."
            )
        else:
            return self.create_result(
                column_name, True,
                "No obvious typos detected",
                {"similar_pairs": []}
            )


class IdentifierDuplicatesCheck(QualityCheck):
    def __init__(self):
        super().__init__("ID001", "Identifier Has Duplicates", Severity.CRITICAL)

    def run(self, column_name: str, identifier_metrics: Any) -> Optional[QualityCheckResult]:
        if identifier_metrics is None:
            return None

        if identifier_metrics.duplicate_count > 0:
            return self.create_result(
                column_name, False,
                f"Identifier column has {identifier_metrics.duplicate_count} duplicate values",
                {"duplicate_count": identifier_metrics.duplicate_count,
                 "duplicate_values": identifier_metrics.duplicate_values[:5]},
                "CRITICAL: Identifiers must be unique. Investigate and resolve duplicates or reconsider column type."
            )
        else:
            return self.create_result(
                column_name, True,
                "Identifier column is unique",
                {"duplicate_count": 0}
            )


class IdentifierFormatCheck(QualityCheck):
    def __init__(self):
        super().__init__("ID002", "Identifier Format Inconsistent", Severity.MEDIUM)
        self.threshold = 95.0

    def run(self, column_name: str, identifier_metrics: Any) -> Optional[QualityCheckResult]:
        if identifier_metrics is None or identifier_metrics.format_consistency is None:
            return None

        format_consistency = identifier_metrics.format_consistency

        if format_consistency < self.threshold:
            return self.create_result(
                column_name, False,
                f"Identifier format inconsistent: {format_consistency}% match pattern '{identifier_metrics.format_pattern}' (< {self.threshold}%)",
                {"format_consistency": format_consistency, "format_pattern": identifier_metrics.format_pattern},
                "Standardize identifier format or investigate data quality issues."
            )
        else:
            return self.create_result(
                column_name, True,
                f"Identifier format consistent: {format_consistency}% match pattern '{identifier_metrics.format_pattern}'",
                {"format_consistency": format_consistency, "format_pattern": identifier_metrics.format_pattern}
            )


class IdentifierNullCheck(QualityCheck):
    def __init__(self):
        super().__init__("ID003", "Identifier Contains Nulls", Severity.HIGH)

    def run(self, column_name: str, universal_metrics: Any) -> Optional[QualityCheckResult]:
        if universal_metrics is None:
            return None

        if universal_metrics.null_count > 0:
            return self.create_result(
                column_name, False,
                f"Identifier column contains {universal_metrics.null_count} null values ({universal_metrics.null_percentage}%)",
                {"null_count": universal_metrics.null_count, "null_percentage": universal_metrics.null_percentage},
                "Identifiers should not contain nulls. Investigate missing identifiers or data quality issues."
            )
        else:
            return self.create_result(
                column_name, True,
                "Identifier column has no null values",
                {"null_count": 0}
            )


class QualityCheckRegistry:
    _checks = {
        "FQ001": MissingValueCheck,
        "FQ003": ConstantFeatureCheck,
        "FQ005": HighOutliersCheck,
        "FQ008": UnknownColumnTypeCheck,
        "FQ009": VeryHighCardinalityNominalCheck,
        "FQ011": AllValuesOutliersCheck,
        "FQ012": UnrealisticDateRangeCheck,
        "CAT001": HighCardinalityCheck,
        "NUM001": LowCardinalityCheck,
        "CAT002": ImbalancedTargetCheck,
        "NUM002": SkewnessCheck,
        "NUM003": OutlierCheck,
        "NUM004": ZeroInflationCheck,
        "LEAK001": IdentifierLeakageCheck,
        "DT001": DatetimeFutureLeakageCheck,
        "DT002": PlaceholderDateCheck,
        "CAT003": RareCategoryCheck,
        "CAT004": UnknownCategoryCheck,
        "TX001": PIIDetectedCheck,
        "TX002": EmptyTextCheck,
        "TX003": ShortTextCheck,
        "TX004": HighUniquenessTextCheck,
        "NC001": ExtremeOutliersCheck,
        "NC002": ModerateOutliersCheck,
        "NC003": HighSkewnessCheck,
        "NC004": NumericZeroInflationCheck,
        "NC005": UnexpectedNegativesCheck,
        "NC006": InfiniteValuesCheck,
        "NC007": ConstantValueCheck,
        "NC008": SuspiciousPrecisionCheck,
        "TG001": TargetNullCheck,
        "TG002": TargetSevereImbalanceCheck,
        "TG003": TargetModerateImbalanceCheck,
        "TG004": TargetUnexpectedClassesCheck,
        "TG005": SingleClassTargetCheck,
        "ID001": IdentifierDuplicatesCheck,
        "ID002": IdentifierFormatCheck,
        "ID003": IdentifierNullCheck,
        "CN001": VeryHighCardinalityCheck,
        "CN002": HighCardinalityCategoricalCheck,
        "CN003": ManyRareCategoriesCheck,
        "CN004": SignificantRareVolumeCheck,
        "CN005": CaseInconsistencyCheck,
        "CN006": WhitespaceIssuesCheck,
        "CN007": SingleCategoryCheck,
        "CN008": PossibleTyposCheck,
        "DT003": DatetimeFormatInconsistentCheck,
        "DT004": DatetimeMixedTimezonesCheck,
        "DT005": DatetimeInvalidDatesCheck,
        "DT006": DatetimeUnrealisticRangeCheck,
        "BN001": BinaryNotBinaryCheck,
        "BN002": BinarySevereImbalanceCheck,
        "BN003": BinaryAllSameValueCheck,
        "BN004": BinaryUnexpectedValuesCheck,
    }

    @classmethod
    def get_check(cls, check_id: str):
        check_class = cls._checks.get(check_id)
        return check_class() if check_class else None

    @classmethod
    def get_all_checks(cls):
        return [check_class() for check_class in cls._checks.values()]

    @classmethod
    def get_checks_for_column_type(cls, column_type: ColumnType):
        checks = []

        checks.append(MissingValueCheck())
        checks.append(ConstantFeatureCheck())

        if column_type == ColumnType.IDENTIFIER:
            checks.append(IdentifierLeakageCheck())
            checks.append(IdentifierDuplicatesCheck())
            checks.append(IdentifierFormatCheck())
            checks.append(IdentifierNullCheck())

        elif column_type == ColumnType.TARGET:
            checks.append(TargetNullCheck())
            checks.append(SingleClassTargetCheck())
            checks.append(TargetSevereImbalanceCheck())
            checks.append(TargetModerateImbalanceCheck())
            checks.append(ImbalancedTargetCheck())
            # Note: TG004 (TargetUnexpectedClassesCheck) requires expected_classes configuration

        elif column_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            checks.append(LowCardinalityCheck())
            checks.append(ExtremeOutliersCheck())  # NC001
            checks.append(ModerateOutliersCheck())  # NC002
            checks.append(HighSkewnessCheck())  # NC003
            checks.append(NumericZeroInflationCheck())  # NC004
            # NC005 (UnexpectedNegativesCheck) requires configuration
            checks.append(InfiniteValuesCheck())  # NC006
            checks.append(ConstantValueCheck())  # NC007
            # NC008 (SuspiciousPrecisionCheck) requires series access
            checks.append(SkewnessCheck())  # FQ006
            checks.append(OutlierCheck())  # FQ007
            checks.append(ZeroInflationCheck())  # FQ008

        elif column_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL, ColumnType.CATEGORICAL_CYCLICAL]:
            checks.append(HighCardinalityCheck())
            checks.append(RareCategoryCheck())
            checks.append(UnknownCategoryCheck())
            checks.append(VeryHighCardinalityCheck())
            checks.append(HighCardinalityCategoricalCheck())
            checks.append(ManyRareCategoriesCheck())
            checks.append(SignificantRareVolumeCheck())
            checks.append(CaseInconsistencyCheck())
            checks.append(WhitespaceIssuesCheck())
            checks.append(SingleCategoryCheck())
            checks.append(PossibleTyposCheck())

        elif column_type == ColumnType.DATETIME:
            checks.append(DatetimeFutureLeakageCheck())
            checks.append(PlaceholderDateCheck())
            checks.append(DatetimeFormatInconsistentCheck())
            checks.append(DatetimeMixedTimezonesCheck())
            checks.append(DatetimeUnrealisticRangeCheck())

        elif column_type == ColumnType.BINARY:
            checks.append(BinaryNotBinaryCheck())
            checks.append(BinarySevereImbalanceCheck())
            checks.append(BinaryAllSameValueCheck())
            checks.append(BinaryUnexpectedValuesCheck())

        elif column_type == ColumnType.TEXT:
            checks.append(PIIDetectedCheck())
            checks.append(EmptyTextCheck())
            checks.append(ShortTextCheck())
            checks.append(HighUniquenessTextCheck())

        return checks
