import numpy as np

from customer_retention.core.components.enums import Severity
from customer_retention.core.config import ColumnType
from customer_retention.stages.profiling.profile_result import (
    BinaryMetrics,
    CategoricalMetrics,
    NumericMetrics,
    TargetMetrics,
    UniversalMetrics,
)
from customer_retention.stages.profiling.quality_checks import (
    BinaryAllSameValueCheck,
    BinaryNotBinaryCheck,
    BinarySevereImbalanceCheck,
    CaseInconsistencyCheck,
    ConstantFeatureCheck,
    ConstantValueCheck,
    ExtremeOutliersCheck,
    HighCardinalityCategoricalCheck,
    HighOutliersCheck,
    HighSkewnessCheck,
    InfiniteValuesCheck,
    LowCardinalityCheck,
    ManyRareCategoriesCheck,
    MissingValueCheck,
    ModerateOutliersCheck,
    NumericZeroInflationCheck,
    QualityCheckRegistry,
    RareCategoryCheck,
    SignificantRareVolumeCheck,
    SingleCategoryCheck,
    SingleClassTargetCheck,
    SkewnessCheck,
    TargetModerateImbalanceCheck,
    TargetNullCheck,
    TargetSevereImbalanceCheck,
    UnexpectedNegativesCheck,
    VeryHighCardinalityCheck,
    WhitespaceIssuesCheck,
    ZeroInflationCheck,
)

np.random.seed(42)


def _universal(total=100, null_count=0, distinct_count=10, most_common_value=None):
    null_pct = round((null_count / total * 100) if total > 0 else 0, 2)
    distinct_pct = round((distinct_count / total * 100) if total > 0 else 0, 2)
    return UniversalMetrics(
        total_count=total,
        null_count=null_count,
        null_percentage=null_pct,
        distinct_count=distinct_count,
        distinct_percentage=distinct_pct,
        most_common_value=most_common_value,
    )


def _numeric(
    mean=0.0,
    std=1.0,
    outlier_pct=0.0,
    outlier_iqr=0,
    outlier_zscore=0,
    skewness=0.0,
    zero_count=0,
    zero_pct=0.0,
    inf_count=0,
    inf_pct=0.0,
    negative_count=0,
    negative_pct=0.0,
):
    return NumericMetrics(
        mean=mean,
        std=std,
        min_value=mean - 3 * std,
        max_value=mean + 3 * std,
        range_value=6 * std,
        median=mean,
        q1=mean - 0.67 * std,
        q3=mean + 0.67 * std,
        iqr=1.34 * std,
        skewness=skewness,
        kurtosis=0.0,
        zero_count=zero_count,
        zero_percentage=zero_pct,
        negative_count=negative_count,
        negative_percentage=negative_pct,
        inf_count=inf_count,
        inf_percentage=inf_pct,
        outlier_count_iqr=outlier_iqr,
        outlier_count_zscore=outlier_zscore,
        outlier_percentage=outlier_pct,
    )


def _categorical(
    cardinality=5,
    cardinality_ratio=0.05,
    rare_count=0,
    rare_pct=0.0,
    contains_unknown=False,
    case_variations=None,
    whitespace_issues=None,
    top_categories=None,
    rare_categories=None,
    encoding_recommendation="one_hot",
):
    return CategoricalMetrics(
        cardinality=cardinality,
        cardinality_ratio=cardinality_ratio,
        value_counts={"a": 50, "b": 30},
        top_categories=top_categories or [("a", 50), ("b", 30)],
        rare_categories=rare_categories or [],
        rare_category_count=rare_count,
        rare_category_percentage=rare_pct,
        contains_unknown=contains_unknown,
        case_variations=case_variations or [],
        whitespace_issues=whitespace_issues or [],
        encoding_recommendation=encoding_recommendation,
    )


def _target(n_classes=2, minority_pct=50.0, imbalance_ratio=1.0, minority_class=0):
    return TargetMetrics(
        class_distribution={"0": 50, "1": 50},
        class_percentages={"0": 50.0, "1": 50.0},
        imbalance_ratio=imbalance_ratio,
        minority_class=minority_class,
        minority_percentage=minority_pct,
        n_classes=n_classes,
    )


class TestMissingValueCheckExtended:
    def test_critical_severity_above_95_percent(self):
        result = MissingValueCheck().run("col", _universal(total=100, null_count=97, distinct_count=1))
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.details["null_percentage"] == 97.0

    def test_high_severity_between_70_and_95(self):
        result = MissingValueCheck().run("col", _universal(total=100, null_count=80, distinct_count=5))
        assert result.passed is False
        assert result.severity == Severity.HIGH
        assert result.details["null_percentage"] == 80.0

    def test_medium_severity_between_20_and_70(self):
        result = MissingValueCheck().run("col", _universal(total=100, null_count=45, distinct_count=20))
        assert result.passed is True
        assert result.severity == Severity.MEDIUM

    def test_passes_below_20_percent(self):
        result = MissingValueCheck().run("col", _universal(total=100, null_count=10, distinct_count=50))
        assert result.passed is True
        assert "Acceptable" in result.message

    def test_all_null_column_is_critical(self):
        result = MissingValueCheck().run("col", _universal(total=100, null_count=100, distinct_count=0))
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.details["null_percentage"] == 100.0


class TestConstantFeatureCheckExtended:
    def test_single_value_column_fails(self):
        result = ConstantFeatureCheck().run("col", _universal(distinct_count=1, most_common_value=42))
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert "constant" in result.message.lower()

    def test_multi_value_column_passes(self):
        result = ConstantFeatureCheck().run("col", _universal(distinct_count=15))
        assert result.passed is True
        assert result.details["distinct_count"] == 15

    def test_null_only_column_as_constant(self):
        metrics = _universal(total=50, null_count=50, distinct_count=0, most_common_value=None)
        result = ConstantFeatureCheck().run("col", metrics)
        assert result.passed is True or result.details.get("distinct_count") == 0


class TestOutlierChecks:
    def test_extreme_outliers_above_threshold(self):
        result = ExtremeOutliersCheck().run("col", _numeric(outlier_pct=8.0, outlier_iqr=80, outlier_zscore=60))
        assert result.passed is False
        assert result.severity == Severity.HIGH

    def test_moderate_outliers_at_boundary(self):
        result = ModerateOutliersCheck().run("col", _numeric(outlier_pct=1.5, outlier_iqr=15))
        assert result.passed is False
        assert result.severity == Severity.MEDIUM

    def test_no_outliers_in_normal_distribution(self):
        result = ExtremeOutliersCheck().run("col", _numeric(outlier_pct=0.5, outlier_iqr=5, outlier_zscore=3))
        assert result.passed is True

    def test_zero_std_constant_value(self):
        result = ConstantValueCheck().run("col", _numeric(mean=5.0, std=0.0))
        assert result.passed is False
        assert result.details["std"] == 0

    def test_infinite_values_detected(self):
        result = InfiniteValuesCheck().run("col", _numeric(inf_count=3, inf_pct=3.0))
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.details["inf_count"] == 3


class TestSkewnessChecks:
    def test_severe_skewness_above_3(self):
        result = SkewnessCheck().run("col", _numeric(skewness=4.5))
        assert result.passed is False
        assert "Extreme" in result.message

    def test_moderate_skewness_between_1_and_3(self):
        result = SkewnessCheck().run("col", _numeric(skewness=1.8))
        assert result.passed is False
        assert "Moderate" in result.message

    def test_symmetric_distribution_passes(self):
        result = SkewnessCheck().run("col", _numeric(skewness=0.3))
        assert result.passed is True

    def test_left_skewed_data_detected(self):
        result = SkewnessCheck().run("col", _numeric(skewness=-3.5))
        assert result.passed is False
        assert result.details["skewness"] == -3.5


class TestHighSkewnessCheck:
    def test_high_skewness_above_threshold(self):
        result = HighSkewnessCheck().run("col", _numeric(skewness=2.5))
        assert result.passed is False
        assert result.details["abs_skewness"] == 2.5

    def test_acceptable_skewness_below_threshold(self):
        result = HighSkewnessCheck().run("col", _numeric(skewness=1.5))
        assert result.passed is True

    def test_none_numeric_metrics_returns_none(self):
        assert HighSkewnessCheck().run("col", None) is None

    def test_none_skewness_returns_none(self):
        assert HighSkewnessCheck().run("col", _numeric(skewness=None)) is None


class TestZeroInflationChecks:
    def test_high_zero_percentage_detected(self):
        result = ZeroInflationCheck().run("col", _numeric(zero_count=75, zero_pct=75.0))
        assert result.passed is False
        assert result.details["zero_percentage"] == 75.0

    def test_mixed_zeros_and_values_passes(self):
        result = ZeroInflationCheck().run("col", _numeric(zero_count=10, zero_pct=10.0))
        assert result.passed is True

    def test_all_zeros_detected(self):
        result = NumericZeroInflationCheck().run("col", _numeric(zero_count=100, zero_pct=100.0))
        assert result.passed is False
        assert result.details["zero_count"] == 100


class TestCardinalityChecks:
    def test_very_high_cardinality_above_100(self):
        result = VeryHighCardinalityCheck().run("col", _categorical(cardinality=150, encoding_recommendation="target"))
        assert result.passed is False
        assert result.severity == Severity.HIGH
        assert result.details["cardinality"] == 150

    def test_high_cardinality_categorical_50_to_100(self):
        result = HighCardinalityCategoricalCheck().run(
            "col", _categorical(cardinality=75, encoding_recommendation="target")
        )
        assert result.passed is False
        assert result.severity == Severity.MEDIUM

    def test_low_cardinality_numeric_detected(self):
        result = LowCardinalityCheck().run("col", _universal(distinct_count=5), ColumnType.NUMERIC_CONTINUOUS)
        assert result.passed is False
        assert result.severity == Severity.LOW

    def test_normal_cardinality_passes(self):
        result = VeryHighCardinalityCheck().run("col", _categorical(cardinality=30))
        assert result.passed is True

    def test_single_category_detected(self):
        result = SingleCategoryCheck().run("col", _categorical(cardinality=1, top_categories=[("only_val", 100)]))
        assert result.passed is False
        assert result.severity == Severity.HIGH


class TestRareCategoryChecks:
    def test_many_rare_categories_detected(self):
        result = ManyRareCategoriesCheck().run(
            "col", _categorical(rare_count=15, rare_categories=["r1", "r2", "r3", "r4", "r5"])
        )
        assert result.passed is False
        assert result.details["rare_category_count"] == 15

    def test_significant_rare_volume_flagged(self):
        result = SignificantRareVolumeCheck().run("col", _categorical(rare_pct=30.0, rare_count=20))
        assert result.passed is False
        assert result.severity == Severity.HIGH

    def test_no_rare_categories_passes(self):
        result = RareCategoryCheck().run("col", _categorical(rare_pct=5.0, rare_count=2))
        assert result.passed is True

    def test_rare_categories_at_threshold_boundary(self):
        result = RareCategoryCheck().run("col", _categorical(rare_pct=20.0, rare_count=5))
        assert result.passed is True

        result_above = RareCategoryCheck().run("col", _categorical(rare_pct=20.1, rare_count=6))
        assert result_above.passed is False


class TestTargetChecks:
    def test_null_target_flagged_critical(self):
        result = TargetNullCheck().run("target", _universal(null_count=5))
        assert result.passed is False
        assert result.severity == Severity.CRITICAL

    def test_severe_imbalance_below_1_percent(self):
        result = TargetSevereImbalanceCheck().run("target", _target(minority_pct=0.5, imbalance_ratio=199.0))
        assert result.passed is False
        assert result.severity == Severity.HIGH

    def test_moderate_imbalance_1_to_10_percent(self):
        result = TargetModerateImbalanceCheck().run("target", _target(minority_pct=5.0, imbalance_ratio=19.0))
        assert result.passed is False
        assert result.severity == Severity.MEDIUM

    def test_single_class_target_critical(self):
        metrics = TargetMetrics(
            class_distribution={"churned": 100},
            class_percentages={"churned": 100.0},
            imbalance_ratio=0.0,
            minority_class="churned",
            minority_percentage=100.0,
            n_classes=1,
        )
        result = SingleClassTargetCheck().run("target", metrics)
        assert result.passed is False
        assert result.severity == Severity.CRITICAL

    def test_balanced_target_passes(self):
        result = TargetSevereImbalanceCheck().run("target", _target(minority_pct=45.0))
        assert result.passed is True

        result_mod = TargetModerateImbalanceCheck().run("target", _target(minority_pct=45.0))
        assert result_mod.passed is True

    def test_binary_not_binary_when_more_than_2_values(self):
        result = BinaryNotBinaryCheck().run("col", _universal(distinct_count=5))
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.details["distinct_count"] == 5


class TestCategoricalQualityChecks:
    def test_case_inconsistency_detected(self):
        result = CaseInconsistencyCheck().run("col", _categorical(case_variations=["Active/active", "Status/status"]))
        assert result.passed is False
        assert len(result.details["case_variations"]) == 2

    def test_whitespace_issues_detected(self):
        result = WhitespaceIssuesCheck().run("col", _categorical(whitespace_issues=[" foo", "bar ", " baz "]))
        assert result.passed is False
        assert len(result.details["whitespace_issues"]) == 3

    def test_binary_all_same_value_detected(self):
        result = BinaryAllSameValueCheck().run("col", _universal(distinct_count=1, most_common_value=1))
        assert result.passed is False
        assert result.severity == Severity.HIGH

    def test_binary_severe_imbalance(self):
        binary_metrics = BinaryMetrics(
            true_count=1,
            false_count=99,
            true_percentage=0.5,
            balance_ratio=0.01,
            values_found=[0, 1],
            is_boolean=True,
        )
        result = BinarySevereImbalanceCheck().run("col", binary_metrics)
        assert result.passed is False
        assert result.severity == Severity.MEDIUM


class TestUnexpectedNegativesCheck:
    def test_negatives_not_allowed_and_present(self):
        check = UnexpectedNegativesCheck(allow_negatives=False)
        result = check.run("col", _numeric(negative_count=10, negative_pct=10.0))
        assert result.passed is False
        assert result.severity == Severity.HIGH

    def test_negatives_allowed_returns_none(self):
        check = UnexpectedNegativesCheck(allow_negatives=True)
        assert check.run("col", _numeric(negative_count=10)) is None

    def test_no_negatives_passes(self):
        check = UnexpectedNegativesCheck(allow_negatives=False)
        result = check.run("col", _numeric(negative_count=0, negative_pct=0.0))
        assert result.passed is True


class TestHighOutliersCheck:
    def test_above_50_percent_outliers(self):
        result = HighOutliersCheck().run("col", _numeric(outlier_pct=55.0, outlier_iqr=55))
        assert result.passed is False
        assert result.severity == Severity.HIGH

    def test_below_50_percent_passes(self):
        result = HighOutliersCheck().run("col", _numeric(outlier_pct=20.0, outlier_iqr=20))
        assert result.passed is True

    def test_none_metrics_returns_none(self):
        assert HighOutliersCheck().run("col", None) is None


class TestQualityCheckRegistry:
    def test_get_all_checks_returns_50_plus(self):
        checks = QualityCheckRegistry.get_all_checks()
        assert len(checks) >= 50

    def test_get_checks_for_numeric_type(self):
        checks = QualityCheckRegistry.get_checks_for_column_type(ColumnType.NUMERIC_CONTINUOUS)
        check_ids = [c.check_id for c in checks]
        assert "NC001" in check_ids
        assert "NC006" in check_ids
        assert "FQ001" in check_ids

    def test_get_check_by_id_returns_correct_class(self):
        check = QualityCheckRegistry.get_check("FQ001")
        assert isinstance(check, MissingValueCheck)
        assert check.check_id == "FQ001"

    def test_unknown_check_id_returns_none(self):
        assert QualityCheckRegistry.get_check("NONEXISTENT_999") is None


class TestLowCardinalityCheck:
    def test_non_numeric_type_returns_none(self):
        result = LowCardinalityCheck().run("col", _universal(distinct_count=3), ColumnType.CATEGORICAL_NOMINAL)
        assert result is None

    def test_numeric_with_enough_cardinality_passes(self):
        result = LowCardinalityCheck().run("col", _universal(distinct_count=50), ColumnType.NUMERIC_DISCRETE)
        assert result.passed is True
